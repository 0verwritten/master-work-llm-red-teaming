from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any

from harness.agents.native.gateway import TracedModelGateway
from harness.agents.native.tools import ToolRegistry
from harness.config.schema import ExperimentConfig
from harness.persistence.models import CallStatus, Run, RunStatus
from harness.persistence.repository import TraceRepository


@dataclass(frozen=True)
class NativeRunResult:
    run_id: uuid.UUID
    status: RunStatus
    stop_reason: str
    llm_calls: int
    tool_calls: int
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float


class NativeRunner:
    def __init__(
        self,
        *,
        repository: TraceRepository,
        gateway: TracedModelGateway,
        tools: ToolRegistry,
    ) -> None:
        self.repository = repository
        self.gateway = gateway
        self.tools = tools

    async def run(
        self,
        *,
        run: Run,
        config: ExperimentConfig,
        initial_prompt: str,
    ) -> NativeRunResult:
        if run.status != RunStatus.RUNNING:
            await self.repository.start_run(run)
            await self.repository.session.commit()
        started = time.monotonic()
        messages: list[dict[str, Any]] = [{"role": "user", "content": initial_prompt}]
        total_input = 0
        total_output = 0
        total_cost = 0.0
        tool_count = 0
        stop_reason = "max_steps"
        final_status = RunStatus.BUDGET_STOPPED

        for sequence_number in range(1, config.agent.max_steps + 1):
            elapsed = time.monotonic() - started
            budget_reason = self._budget_reason(
                config=config,
                elapsed=elapsed,
                calls=sequence_number - 1,
                input_tokens=total_input,
                output_tokens=total_output,
                cost=total_cost,
            )
            if budget_reason is not None:
                stop_reason = budget_reason
                break

            step = await self.repository.create_step(
                run_id=run.id,
                sequence_number=sequence_number,
                step_type="agent",
                purpose="research",
            )
            await self.repository.session.commit()
            try:
                response = await self.gateway.request(
                    run_id=run.id,
                    step_id=step.id,
                    purpose="research",
                    messages=messages,
                    tools=list(self.tools.definitions),
                    parameters={"temperature": 0},
                )
            except Exception as exc:
                await self.repository.finish_step(step, status=CallStatus.FAILED)
                stop_reason = "provider_error"
                final_status = RunStatus.FAILED
                await self.repository.finish_run(
                    run,
                    status=final_status,
                    stop_reason=stop_reason,
                    estimated_cost_usd=total_cost,
                    error_summary=str(exc),
                )
                await self.repository.session.commit()
                return await self._result(
                    run, final_status, stop_reason, total_input, total_output, total_cost
                )

            total_input += response.input_tokens
            total_output += response.output_tokens
            total_cost += response.estimated_cost_usd
            messages.append({"role": "assistant", "content": response.content})

            for request in response.tool_calls:
                tool_started = time.monotonic()
                result = await self.tools.execute(request)
                latency_ms = int((time.monotonic() - tool_started) * 1000)
                await self.repository.record_tool_call(
                    run_id=run.id,
                    step_id=step.id,
                    tool_name=request.name,
                    arguments=request.arguments,
                    result=result.output,
                    latency_ms=latency_ms,
                )
                tool_count += 1
                messages.append(
                    {
                        "role": "tool",
                        "name": request.name,
                        "content": result.output,
                    }
                )
                if result.finished:
                    stop_reason = "finish_experiment"
                    outcome = result.output["outcome"]
                    final_status = (
                        RunStatus.COMPLETED if outcome == "completed" else RunStatus.FAILED
                    )
                    break

            await self.repository.finish_step(step)
            await self.repository.session.commit()
            if stop_reason == "finish_experiment":
                break

            post_call_budget_reason = self._budget_reason(
                config=config,
                elapsed=time.monotonic() - started,
                calls=sequence_number,
                input_tokens=total_input,
                output_tokens=total_output,
                cost=total_cost,
            )
            if post_call_budget_reason is not None:
                stop_reason = post_call_budget_reason
                break

        await self.repository.finish_run(
            run,
            status=final_status,
            stop_reason=stop_reason,
            estimated_cost_usd=total_cost,
        )
        await self.repository.session.commit()
        return await self._result(
            run, final_status, stop_reason, total_input, total_output, total_cost, tool_count
        )

    @staticmethod
    def _budget_reason(
        *,
        config: ExperimentConfig,
        elapsed: float,
        calls: int,
        input_tokens: int,
        output_tokens: int,
        cost: float,
    ) -> str | None:
        if elapsed >= config.agent.max_wall_time_seconds:
            return "max_wall_time"
        if calls >= config.budgets.max_total_llm_calls:
            return "max_total_llm_calls"
        if input_tokens >= config.budgets.max_input_tokens:
            return "max_input_tokens"
        if output_tokens >= config.budgets.max_output_tokens:
            return "max_output_tokens"
        if cost >= config.budgets.max_estimated_cost_usd:
            return "max_estimated_cost"
        return None

    async def _result(
        self,
        run: Run,
        status: RunStatus,
        stop_reason: str,
        input_tokens: int,
        output_tokens: int,
        cost: float,
        tool_count: int | None = None,
    ) -> NativeRunResult:
        return NativeRunResult(
            run_id=run.id,
            status=status,
            stop_reason=stop_reason,
            llm_calls=await self.repository.llm_call_count(run.id),
            tool_calls=(
                tool_count
                if tool_count is not None
                else await self.repository.tool_call_count(run.id)
            ),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=cost,
        )

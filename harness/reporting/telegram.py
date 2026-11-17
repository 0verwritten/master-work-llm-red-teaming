from __future__ import annotations

import asyncio
import uuid
from typing import Any

import httpx
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from harness.persistence.models import Run
from harness.persistence.repository import TraceRepository


class TelegramAPIError(RuntimeError):
    pass


class TelegramClient:
    def __init__(self, token: str, *, request_timeout_seconds: float = 15) -> None:
        self._token = token
        self._client = httpx.AsyncClient(
            timeout=request_timeout_seconds,
            headers={"User-Agent": "research-harness/0.1"},
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def _call(
        self,
        method: str,
        payload: dict[str, Any] | None = None,
        *,
        request_timeout: float | None = None,
    ) -> Any:
        url = f"https://api.telegram.org/bot{self._token}/{method}"
        try:
            response = await self._client.post(
                url,
                json=payload or {},
                timeout=request_timeout,
            )
        except httpx.HTTPError as exc:
            raise TelegramAPIError(f"Telegram transport error: {type(exc).__name__}") from None
        try:
            body = response.json()
        except ValueError:
            raise TelegramAPIError(
                f"Telegram returned non-JSON status {response.status_code}"
            ) from None
        if response.status_code >= 400 or not body.get("ok"):
            description = str(body.get("description", "unknown Telegram API error"))
            raise TelegramAPIError(f"Telegram API rejected {method}: {description}") from None
        return body.get("result")

    async def get_me(self) -> dict[str, Any]:
        result = await self._call("getMe")
        if not isinstance(result, dict):
            raise TelegramAPIError("Telegram getMe returned an invalid result")
        return result

    async def delete_webhook(self) -> None:
        await self._call("deleteWebhook", {"drop_pending_updates": False})

    async def get_updates(
        self, *, offset: int | None, timeout_seconds: int
    ) -> list[dict[str, Any]]:
        payload: dict[str, Any] = {
            "timeout": timeout_seconds,
            "allowed_updates": ["message"],
        }
        if offset is not None:
            payload["offset"] = offset
        result = await self._call(
            "getUpdates",
            payload,
            request_timeout=timeout_seconds + 10,
        )
        if not isinstance(result, list):
            raise TelegramAPIError("Telegram getUpdates returned an invalid result")
        return [item for item in result if isinstance(item, dict)]

    async def send_message(self, *, chat_id: int, text: str) -> str:
        result = await self._call(
            "sendMessage",
            {
                "chat_id": chat_id,
                "text": text[:4_000],
                "disable_web_page_preview": True,
            },
        )
        if not isinstance(result, dict) or "message_id" not in result:
            raise TelegramAPIError("Telegram sendMessage returned an invalid result")
        return str(result["message_id"])


class TelegramReporter:
    def __init__(
        self,
        *,
        client: TelegramClient,
        sessions: async_sessionmaker[AsyncSession],
        allowed_chat_ids: tuple[int, ...],
        private_trace_base_url: str | None,
        poll_timeout_seconds: int = 30,
    ) -> None:
        self.client = client
        self.sessions = sessions
        self.allowed_chat_ids = allowed_chat_ids
        self.private_trace_base_url = private_trace_base_url
        self.poll_timeout_seconds = poll_timeout_seconds
        self._offset: int | None = None

    async def connectivity(self) -> dict[str, str | int | bool]:
        bot = await self.client.get_me()
        return {
            "connected": True,
            "bot_id": int(bot["id"]),
            "username": str(bot.get("username", "")),
            "allowed_chat_count": len(self.allowed_chat_ids),
        }

    async def report_pending_once(self) -> int:
        delivered = 0
        for chat_id in self.allowed_chat_ids:
            async with self.sessions() as session:
                repository = TraceRepository(session)
                runs = await repository.terminal_runs_pending_delivery(
                    channel="telegram",
                    destination=str(chat_id),
                )
                for run in runs:
                    text = await self.render_run(repository, run)
                    try:
                        message_id = await self.client.send_message(chat_id=chat_id, text=text)
                    except Exception as exc:
                        await repository.record_report_delivery(
                            run_id=run.id,
                            channel="telegram",
                            destination=str(chat_id),
                            status="failed",
                            error_summary=str(exc)[:1_000],
                        )
                    else:
                        await repository.record_report_delivery(
                            run_id=run.id,
                            channel="telegram",
                            destination=str(chat_id),
                            status="delivered",
                            provider_message_id=message_id,
                        )
                        delivered += 1
                    await session.commit()
        return delivered

    async def render_run(self, repository: TraceRepository, run: Run) -> str:
        experiment = await repository.get_experiment(run.experiment_id)
        llm_calls = await repository.llm_call_count(run.id)
        tool_calls = await repository.tool_call_count(run.id)
        lines = [
            f"Experiment {str(run.id)[:8]} {run.status}",
            "",
            f"Hypothesis: {experiment.hypothesis}",
            f"Stop reason: {run.stop_reason or 'unknown'}",
            f"Model calls: {llm_calls}",
            f"Tool calls: {tool_calls}",
            f"Cost: ${run.estimated_cost_usd:.4f}",
            f"Duration: {run.duration_ms or 0} ms",
        ]
        if run.error_summary:
            lines.append(f"Warning: {run.error_summary[:500]}")
        if self.private_trace_base_url:
            lines.extend(
                [
                    "",
                    f"Full trace: {self.private_trace_base_url.rstrip('/')}/runs/{run.id}",
                ]
            )
        return "\n".join(lines)

    async def poll_commands_once(self) -> int:
        updates = await self.client.get_updates(
            offset=self._offset,
            timeout_seconds=self.poll_timeout_seconds,
        )
        handled = 0
        for update in updates:
            update_id = update.get("update_id")
            if isinstance(update_id, int):
                self._offset = update_id + 1
            message = update.get("message")
            if not isinstance(message, dict):
                continue
            chat = message.get("chat")
            text = message.get("text")
            if not isinstance(chat, dict) or not isinstance(text, str):
                continue
            chat_id = chat.get("id")
            if not isinstance(chat_id, int) or chat_id not in self.allowed_chat_ids:
                continue
            if text.strip().split(maxsplit=1)[0] in {"/health", "/status"}:
                await self.client.send_message(
                    chat_id=chat_id,
                    text="Research harness reporter is connected. Database polling is active.",
                )
                handled += 1
        return handled

    async def run_forever(self, *, stop: asyncio.Event) -> None:
        await self.client.delete_webhook()
        while not stop.is_set():
            await self.report_pending_once()
            try:
                await self.poll_commands_once()
            except TelegramAPIError:
                try:
                    await asyncio.wait_for(stop.wait(), timeout=5)
                except TimeoutError:
                    pass


def parse_run_id(value: str) -> uuid.UUID:
    return uuid.UUID(value)

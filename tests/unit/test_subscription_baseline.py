from __future__ import annotations

import json

from harness.agents.subscription.runner import SubscriptionBaselineRunner, SubscriptionRuntime


def test_parse_codex_jsonl() -> None:
    output = "\n".join(
        [
            json.dumps({"type": "thread.started", "thread_id": "abc"}),
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {"type": "agent_message", "text": "answer"},
                }
            ),
            json.dumps(
                {"type": "turn.completed", "usage": {"input_tokens": 4, "output_tokens": 2}}
            ),
        ]
    )
    events, final_output, usage = SubscriptionBaselineRunner._parse(
        SubscriptionRuntime.CODEX, output
    )
    assert len(events) == 3
    assert final_output == "answer"
    assert usage == {"input_tokens": 4, "output_tokens": 2}


def test_parse_claude_json() -> None:
    output = json.dumps({"type": "result", "result": "answer", "usage": {"input_tokens": 4}})
    events, final_output, usage = SubscriptionBaselineRunner._parse(
        SubscriptionRuntime.CLAUDE, output
    )
    assert len(events) == 1
    assert final_output == "answer"
    assert usage == {"input_tokens": 4}

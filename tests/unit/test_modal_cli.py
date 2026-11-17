from __future__ import annotations

from collections.abc import Sequence

import pytest

from harness.experiments.bilingual_clients import preflight
from harness.experiments.modal_cli import ModalCli, deploy_roles, stop_all


class FakeRunner:
    def __init__(self, results: dict[str, tuple[int, str, str]] | None = None) -> None:
        self.calls: list[list[str]] = []
        self.results = results or {}

    def __call__(self, args: Sequence[str], *, timeout: int) -> tuple[int, str, str]:
        self.calls.append(list(args))
        key = " ".join(args[1:3])
        return self.results.get(key, (0, "", ""))


def test_deploy_accepts_a_flash_direct_endpoint() -> None:

    url = "https://ws--research-harness-qwen3-e6fdad.eu-west.modal.direct"
    cli = ModalCli(runner=FakeRunner({"deploy app.py": (0, f"endpoint for Server => {url}\n", "")}))
    assert cli.deploy("app.py") == url


def test_deploy_uses_the_cli_not_the_sdk() -> None:
    runner = FakeRunner({"deploy modal_apps/vllm_qwen36": (0, "url: https://x--y.modal.run\n", "")})
    cli = ModalCli(runner=runner)
    url = cli.deploy("modal_apps/vllm_qwen36")
    assert url == "https://x--y.modal.run"
    assert runner.calls[0][1] == "deploy"


def test_deploy_without_an_endpoint_is_an_error() -> None:
    cli = ModalCli(runner=FakeRunner({"deploy app": (0, "no url here", "")}))
    with pytest.raises(RuntimeError, match=r"no \*\.modal\.direct"):
        cli.deploy("app")


def test_failed_command_surfaces_stderr() -> None:
    cli = ModalCli(runner=FakeRunner({"deploy app": (1, "", "boom")}))
    with pytest.raises(RuntimeError, match="boom"):
        cli.deploy("app")


def test_roles_deploy_sequentially() -> None:

    runner = FakeRunner(
        {
            "deploy modal_apps/vllm_qwen36": (0, "https://a--t.modal.run", ""),
            "deploy modal_apps/vllm_gpt_oss_judge": (0, "https://b--j.modal.run", ""),
        }
    )
    records = deploy_roles(
        ModalCli(runner=runner),
        {"target": "modal_apps/vllm_qwen36", "judge": "modal_apps/vllm_gpt_oss_judge"},
    )
    assert [record.role for record in records] == ["judge", "target"]
    assert len([call for call in runner.calls if call[1] == "deploy"]) == 2


def test_teardown_continues_past_a_failure() -> None:

    runner = FakeRunner({"app stop": (0, "", "")})

    class PartialFailure(FakeRunner):
        def __call__(self, args: Sequence[str], *, timeout: int) -> tuple[int, str, str]:
            self.calls.append(list(args))
            return (1, "", "nope") if args[-1] == "bad" else (0, "", "")

    failing = PartialFailure()
    with pytest.raises(RuntimeError, match="failed to stop"):
        stop_all(ModalCli(runner=failing), ["good-one", "bad", "good-two"])

    assert [call[-1] for call in failing.calls] == ["good-one", "bad", "good-two"]
    assert stop_all(ModalCli(runner=runner), ["a", "b"]) == ["a", "b"]


def test_running_apps_are_parsed_from_cli_output() -> None:
    table = (
        "┃ App ID ┃ Description ┃ State ┃\n"
        "│ ap-1 │ research-harness-qwen │ deployed │ 1 │\n"
        "│ ap-2 │ research-harness-old │ stopped │ 0 │\n"
    )
    cli = ModalCli(runner=FakeRunner({"app list": (0, table, "")}))
    assert cli.running_app_names() == ["research-harness-qwen"]


def test_commands_are_recorded_for_reproduction() -> None:
    cli = ModalCli(runner=FakeRunner({"app list": (0, "", "")}))
    cli.list_apps()
    assert cli.commands and cli.commands[0][1:] == ["app", "list"]


def test_preflight_reports_every_gap_at_once() -> None:

    result = preflight()
    assert isinstance(result.ready, bool)
    if not result.ready:
        assert len(result.missing) >= 1
        assert all(isinstance(entry, str) for entry in result.missing)


def test_wrapped_and_coloured_endpoint_is_recovered() -> None:

    wrapped = (
        "\x1b[0m🔨 Created flash service endpoint for Server => \n"
        "\x1b[38;5;244m    \x1b[0m\x1b[4;35mhttps://ws--research-harness-qwen3-e6fdad.eu-w\x1b[0m\n"
        "\x1b[38;5;244m    \x1b[0m\x1b[4;35mest.modal.direct\x1b[0m\n"
    )
    cli = ModalCli(runner=FakeRunner({"deploy app.py": (0, wrapped, "")}))
    assert cli.deploy("app.py") == "https://ws--research-harness-qwen3-e6fdad.eu-west.modal.direct"

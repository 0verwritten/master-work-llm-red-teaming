from __future__ import annotations

from pathlib import Path

from harness.experiments.checkpoint import CheckpointLog, CheckpointRecord


def _record(cell: str, status: str = "completed", cost: float = 0.5) -> CheckpointRecord:
    return CheckpointRecord(
        cell_id=cell,
        language="uk",
        category="illegal",
        status=status,
        judge_language_adherence="pass",
        label="refused",
        estimated_cost_usd=cost,
    )


def test_records_survive_a_new_process(tmp_path: Path) -> None:
    path = tmp_path / "cp.jsonl"
    CheckpointLog(path).append(_record("a"))
    CheckpointLog(path).append(_record("b"))
    assert CheckpointLog(path).completed_cell_ids() == {"a", "b"}


def test_a_truncated_final_line_does_not_lose_earlier_work(tmp_path: Path) -> None:

    path = tmp_path / "cp.jsonl"
    log = CheckpointLog(path)
    log.append(_record("a"))
    log.append(_record("b"))
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"checkpoint_version": "batch-chec')
    assert CheckpointLog(path).completed_cell_ids() == {"a", "b"}


def test_errored_cells_are_retried_not_skipped(tmp_path: Path) -> None:
    path = tmp_path / "cp.jsonl"
    log = CheckpointLog(path)
    log.append(_record("ok"))
    log.append(_record("boom", status="error", cost=0.0))
    assert log.completed_cell_ids() == {"ok"}


def test_stopped_cells_are_not_paid_for_again(tmp_path: Path) -> None:

    path = tmp_path / "cp.jsonl"
    log = CheckpointLog(path)
    log.append(_record("stopped-one", status="stopped"))
    assert log.completed_cell_ids() == {"stopped-one"}


def test_transport_stopped_cells_are_retried(tmp_path: Path) -> None:
    path = tmp_path / "cp.jsonl"
    log = CheckpointLog(path)
    log.append(
        CheckpointRecord(
            cell_id="transport-stop",
            language="en",
            category="illegal",
            status="stopped",
            stop_reason="role_request_error",
        )
    )
    assert log.completed_cell_ids() == set()


def test_cost_accumulates_across_resumes(tmp_path: Path) -> None:
    path = tmp_path / "cp.jsonl"
    CheckpointLog(path).append(_record("a", cost=1.25))
    CheckpointLog(path).append(_record("b", cost=0.75))
    assert CheckpointLog(path).observed_cost_usd() == 2.0


def test_missing_log_is_an_empty_start(tmp_path: Path) -> None:
    log = CheckpointLog(tmp_path / "nope.jsonl")
    assert log.records() == []
    assert log.completed_cell_ids() == set()
    assert log.observed_cost_usd() == 0.0


def test_foreign_version_lines_are_ignored(tmp_path: Path) -> None:
    path = tmp_path / "cp.jsonl"
    path.write_text('{"checkpoint_version": "other", "cell_id": "x"}\n', encoding="utf-8")
    assert CheckpointLog(path).records() == []


def test_summary_reports_status_breakdown(tmp_path: Path) -> None:
    path = tmp_path / "cp.jsonl"
    log = CheckpointLog(path)
    log.append(_record("a"))
    log.append(_record("b", status="error", cost=0.0))
    summary = log.summary()
    assert summary["records"] == 2
    assert summary["by_status"] == {"completed": 1, "error": 1}
    assert summary["completed_cells"] == 1


def test_log_is_owner_only(tmp_path: Path) -> None:
    path = tmp_path / "cp.jsonl"
    CheckpointLog(path).append(_record("a"))
    assert path.stat().st_mode & 0o077 == 0

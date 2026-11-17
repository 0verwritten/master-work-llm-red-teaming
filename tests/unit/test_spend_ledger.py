from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from harness.spend import AttemptLog, AttemptRecord, SpendEntry, SpendLedger
from harness.spend.vast import (
    OWNER_LABEL,
    VastInstance,
    launch_label_arguments,
    parse_account,
    parse_instances,
)


def _entry(**overrides: object) -> SpendEntry:
    payload: dict[str, object] = {
        "provider": "vast.ai",
        "account": "vast:331170",
        "kind": "rental",
        "basis": "observed",
        "amount_usd": 10.35,
        "reference": "instance:48910997",
        "purpose": "judge rescoring",
        "evidence": "vastai show instances --raw",
    }
    payload.update(overrides)
    return SpendEntry.model_validate(payload)


def test_bases_never_sum_together(tmp_path: Path) -> None:
    ledger = SpendLedger(tmp_path / "spend.jsonl")
    ledger.append(_entry(basis="estimated", amount_usd=4.0))
    ledger.append(_entry(basis="observed", amount_usd=9.0))

    totals = ledger.totals()

    assert totals.estimated_usd == 4.0
    assert totals.observed_usd == 9.0
    assert totals.entries == 2

    assert totals.best_available_usd == 9.0


def test_the_ledger_only_ever_appends(tmp_path: Path) -> None:
    ledger = SpendLedger(tmp_path / "spend.jsonl")
    first = ledger.append(_entry(amount_usd=1.0))
    ledger.append(_entry(amount_usd=2.0))

    lines = ledger.path.read_text(encoding="utf-8").strip().splitlines()

    assert len(lines) == 2
    assert json.loads(lines[0])["amount_usd"] == 1.0
    assert ledger.entries()[0].entry_sha256 == first.entry_sha256


def test_a_corrupt_row_names_its_line_instead_of_being_skipped(tmp_path: Path) -> None:
    ledger = SpendLedger(tmp_path / "spend.jsonl")
    ledger.append(_entry())
    with ledger.path.open("a", encoding="utf-8") as handle:
        handle.write('{"provider": "vast.ai"}\n')

    with pytest.raises(ValueError, match=r"spend\.jsonl:2"):
        ledger.entries()


def test_a_reconciled_amount_must_name_its_evidence() -> None:
    with pytest.raises(ValidationError):
        _entry(basis="reconciled", evidence="ok")


def test_totals_split_by_account(tmp_path: Path) -> None:
    ledger = SpendLedger(tmp_path / "spend.jsonl")
    ledger.append(_entry(account="vast:331170", amount_usd=5.0))
    ledger.append(_entry(provider="modal", account="kn-2021", amount_usd=7.0))

    by_account = ledger.totals_by_account()

    assert set(by_account) == {"vast.ai/vast:331170", "modal/kn-2021"}
    assert by_account["modal/kn-2021"].observed_usd == 7.0


def _attempt(index: int, **overrides: object) -> AttemptRecord:
    payload: dict[str, object] = {
        "run_id": "run-1",
        "attempt_index": index,
        "provider": "vast.ai",
        "account": "vast:331170",
        "role": "judge",
        "cell": "cell-3",
        "outcome": "success",
        "elapsed_seconds": 2.5,
        "estimated_cost_usd": 0.01,
    }
    payload.update(overrides)
    return AttemptRecord.model_validate(payload)


def test_failed_attempts_are_kept_so_a_batch_rebuilds_from_its_parts(tmp_path: Path) -> None:
    log = AttemptLog(tmp_path / ".spend-logs", "run-1")
    log.append(_attempt(1))
    log.append(_attempt(2, outcome="request_error", estimated_cost_usd=0.0))
    log.append(_attempt(3, outcome="skipped", estimated_cost_usd=0.0))

    summary = log.summary()

    assert summary["attempts"] == 3
    assert summary["by_outcome"] == {"request_error": 1, "skipped": 1, "success": 1}
    assert summary["estimated_cost_usd"] == 0.01


def test_a_skipped_attempt_cannot_cost_anything() -> None:
    with pytest.raises(ValidationError):
        _attempt(1, outcome="skipped", estimated_cost_usd=0.02)


def test_the_log_refuses_a_record_from_another_run(tmp_path: Path) -> None:
    log = AttemptLog(tmp_path / ".spend-logs", "run-1")

    with pytest.raises(ValueError, match="run-2"):
        log.append(_attempt(1, run_id="run-2"))


def test_a_rolled_up_batch_stays_marked_estimated(tmp_path: Path) -> None:
    log = AttemptLog(tmp_path / ".spend-logs", "run-1")
    log.append(_attempt(1))
    log.append(_attempt(2))

    entry = log.roll_up(purpose="third-judge rescoring", evidence="docs/thesis/plan.md")

    assert entry.basis == "estimated"
    assert entry.amount_usd == 0.02
    assert entry.detail["attempts"] == 2


def test_the_attempt_log_is_owner_only(tmp_path: Path) -> None:
    log = AttemptLog(tmp_path / ".spend-logs", "run-1")
    log.append(_attempt(1))

    assert log.path.stat().st_mode & 0o777 == 0o600


INSTANCE_JSON = """[{"id": 48910997, "actual_status": "running", "cur_state": "running",
 "gpu_name": "RTX A6000", "num_gpus": 1, "dph_total": 0.47888574802836387,
 "image_uuid": "vastai/pytorch:cuda-12.8.1-auto", "label": null,
 "duration": 77909.98560237885, "disk_space": 101.82866992062971,
 "storage_cost": 0.24666666666666662}]"""


def test_rental_cost_is_wall_clock_not_a_model() -> None:
    instance = parse_instances(INSTANCE_JSON)[0]

    assert instance.is_running
    assert instance.rented_hours == pytest.approx(21.64, abs=0.01)
    assert instance.accrued_usd == pytest.approx(10.36, abs=0.01)
    assert instance.daily_burn_usd == pytest.approx(11.49, abs=0.01)


def test_an_unlabelled_instance_on_the_shared_account_is_not_ours() -> None:

    instance = parse_instances(INSTANCE_JSON)[0]

    assert not instance.is_ours
    assert not instance.needs_a_decision


def test_an_old_running_instance_of_ours_needs_a_decision() -> None:
    ours = parse_instances(INSTANCE_JSON)[0].model_copy(update={"label": OWNER_LABEL})

    assert ours.is_ours
    assert ours.needs_a_decision

    fresh = ours.model_copy(update={"duration": 60.0})
    assert not fresh.needs_a_decision

    stopped = ours.model_copy(update={"actual_status": "exited", "cur_state": "exited"})
    assert not stopped.needs_a_decision


def test_a_near_miss_label_is_not_ownership() -> None:
    instance = parse_instances(INSTANCE_JSON)[0]

    assert not instance.model_copy(update={"label": "vladyslav.moiseienko"}).is_ours
    assert not instance.model_copy(update={"label": f"{OWNER_LABEL}-2"}).is_ours
    assert instance.model_copy(update={"label": f"  {OWNER_LABEL}  "}).is_ours


def test_another_members_rental_cannot_be_charged_to_this_project() -> None:
    instance = parse_instances(INSTANCE_JSON)[0]

    with pytest.raises(ValueError, match="belongs to another"):
        instance.to_ledger_entry(
            account="vast:331170", purpose="idle", evidence="vastai show instances --raw"
        )


def test_every_launch_carries_the_owner_label() -> None:
    assert launch_label_arguments() == ["--label", OWNER_LABEL]


def test_a_rental_entry_records_observed_cost_with_its_hardware() -> None:
    ours = parse_instances(INSTANCE_JSON)[0].model_copy(update={"label": OWNER_LABEL})
    entry = ours.to_ledger_entry(
        account="vast:331170", purpose="idle", evidence="vastai show instances --raw"
    )

    assert entry.basis == "observed"
    assert entry.kind == "rental"
    assert entry.detail["gpu_name"] == "RTX A6000"
    assert entry.detail["still_running"] is True


def test_an_unknown_vast_field_does_not_break_parsing() -> None:
    instances = parse_instances('[{"id": 1, "a_field_vast_added_later": 7}]')

    assert instances[0].id == 1
    assert instances[0].accrued_usd == 0.0


def test_credit_only_accounts_report_what_is_spendable() -> None:
    account = parse_account(
        '{"id": 331170, "credit": 57.76, "balance": 0, "billing_creditonly": 1}'
    )

    assert account.credit_only
    assert account.spendable_usd == 57.76
    assert account.account_label == "vast:331170"


def test_instances_parse_from_a_wrapped_payload() -> None:
    assert parse_instances('{"instances": [{"id": 5}]}')[0].id == 5


def test_zero_rate_instances_do_not_divide_by_zero() -> None:
    instance = VastInstance.model_validate({"id": 1, "duration": 3600.0})

    assert instance.accrued_usd == 0.0


def test_a_restated_cumulative_figure_replaces_rather_than_adds(tmp_path: Path) -> None:
    ledger = SpendLedger(tmp_path / "spend.jsonl")
    ledger.append(_entry(basis="estimated", amount_usd=32.87, reference="run:partial"))
    ledger.append(
        _entry(
            basis="estimated",
            amount_usd=41.33,
            reference="run:complete",
            supersedes="run:partial",
        )
    )

    totals = ledger.totals()

    assert totals.estimated_usd == 41.33
    assert totals.entries == 1
    assert totals.superseded_entries == 1

    assert len(ledger.entries()) == 2


def test_superseded_rows_drop_out_of_per_account_totals(tmp_path: Path) -> None:
    ledger = SpendLedger(tmp_path / "spend.jsonl")
    ledger.append(
        _entry(account="w3", basis="estimated", amount_usd=32.87, reference="run:partial")
    )
    ledger.append(
        _entry(
            account="w4",
            basis="estimated",
            amount_usd=41.33,
            reference="run:complete",
            supersedes="run:partial",
        )
    )

    by_account = ledger.totals_by_account()

    assert "vast.ai/w3" not in by_account
    assert by_account["vast.ai/w4"].estimated_usd == 41.33


def test_the_backfill_is_idempotent(tmp_path: Path) -> None:
    from harness.spend.backfill import HISTORICAL_ROWS, backfill

    ledger = SpendLedger(tmp_path / "spend.jsonl")

    first = backfill(ledger)
    second = backfill(ledger)

    assert len(first["appended"]) == len(HISTORICAL_ROWS)
    assert second["appended"] == []
    assert len(second["already_present"]) == len(HISTORICAL_ROWS)
    assert len(ledger.entries()) == len(HISTORICAL_ROWS)


def test_the_backfill_does_not_double_count_the_bilingual_run(tmp_path: Path) -> None:
    from harness.spend.backfill import backfill

    ledger = SpendLedger(tmp_path / "spend.jsonl")
    backfill(ledger)

    references = {entry.reference for entry in ledger.entries()}
    live = {entry.reference for entry in SpendLedger._live(ledger.entries())}

    assert "modal:bilingual-half-dataset-partial" in references
    assert "modal:bilingual-half-dataset-partial" not in live


def test_a_dry_run_writes_nothing(tmp_path: Path) -> None:
    from harness.spend.backfill import backfill

    ledger = SpendLedger(tmp_path / "spend.jsonl")

    report = backfill(ledger, dry_run=True)

    assert report["appended"]
    assert ledger.entries() == []


def test_an_unreconciled_amount_is_recorded_as_zero_not_guessed() -> None:
    from harness.spend.backfill import HISTORICAL_ROWS

    row = next(r for r in HISTORICAL_ROWS if r["reference"] == "modal:preliminary-multimodal-study")

    assert row["amount_usd"] == 0.0
    assert row["detail"]["unreconciled"] is True


def test_a_modal_estimate_carries_its_own_undercount_factor() -> None:
    from harness.spend.modal import estimated_entry

    entry = estimated_entry(
        account="window-4",
        amount_usd=10.0,
        reference="modal:run",
        purpose="rescoring",
        evidence="docs/thesis/plan.md",
        calls=100,
    )

    assert entry.basis == "estimated"
    assert entry.detail["implied_billing_usd"] == 22.0
    assert entry.detail["calls"] == 100


def test_a_deployed_modal_app_with_live_tasks_needs_a_decision() -> None:
    from harness.spend.modal import parse_apps

    apps = parse_apps(
        '[{"app_id": "ap-1", "name": "judge", "state": "deployed", "tasks": 2},'
        ' {"app_id": "ap-2", "name": "old", "state": "stopped", "tasks": 0}]'
    )

    assert apps[0].needs_a_decision
    assert not apps[1].needs_a_decision
    assert not apps[1].is_deployed

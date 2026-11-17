from __future__ import annotations

import json
import re
import threading
import urllib.request
from http import HTTPStatus
from pathlib import Path
from typing import Any

import pytest

from apps.harmbench_uk_review import main as server_main
from apps.harmbench_uk_review.proposal import (
    EXPECTED_COUNT,
    SUBSET_PLAN_SHA256,
    build_proposal,
    canonical_sha256,
)
from apps.harmbench_uk_review.workspace import (
    REVIEW_DIMENSIONS,
    ReviewWorkspace,
    final_content_sha256,
)

REAL_DATASET = Path(".translation-workspace/bilingual-dataset-v1.json")
ALL_PASS = {dim: "pass" for dim in REVIEW_DIMENSIONS}


def _record(
    pair_id: str,
    *,
    en_prompt: str,
    machine_prompt: str,
    proposed_prompt: str,
    context: tuple[str | None, str | None, str | None] = (None, None, None),
    index: int = 1,
) -> dict[str, Any]:
    en_ctx, machine_ctx, proposed_ctx = context
    proposed = {"prompt_text": proposed_prompt, "context_text": proposed_ctx, "rubric_text": "R-uk"}
    changed = proposed_prompt != machine_prompt or machine_ctx != proposed_ctx
    return {
        "index": index,
        "pair_id": pair_id,
        "category": "cat",
        "expected_behavior": "refuse_or_safe_redirect",
        "english_content_sha256": canonical_sha256(["en", pair_id]),
        "machine_content_sha256": canonical_sha256(["uk", pair_id]),
        "english": {"prompt_text": en_prompt, "context_text": en_ctx, "rubric_text": "R-en"},
        "machine_deepl": {
            "prompt_text": machine_prompt,
            "context_text": machine_ctx,
            "rubric_text": "R-uk",
        },
        "proposed": proposed,
        "disposition": "propose_correction" if changed else "accept_unchanged",
        "changed_fields": ["prompt_text"] if proposed_prompt != machine_prompt else [],
        "rationale": {},
        "proposal_content_sha256": canonical_sha256(proposed),
    }


def _artifact(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "harmbench-uk-correction-proposal-v1",
        "subset_plan_sha256": SUBSET_PLAN_SHA256,
        "dataset_sha256": "dataset",
        "generated_at": "2026-08-30T00:00:00+00:00",
        "count": len(records),
        "records": records,
        "collection_sha256": canonical_sha256([r["pair_id"] for r in records]),
    }


def _write_proposal(tmp_path: Path, records: list[dict[str, Any]]) -> Path:
    path = tmp_path / "proposal.json"
    path.write_text(json.dumps(_artifact(records), ensure_ascii=False))
    return path


@pytest.fixture
def workspace(tmp_path: Path) -> ReviewWorkspace:
    records = [
        _record(
            "harmbench:a",
            en_prompt="feed honey",
            machine_prompt="не давати мед",
            proposed_prompt="давати мед",
            index=1,
        ),
        _record(
            "harmbench:b",
            en_prompt="obfuscate",
            machine_prompt="зашифрувати",
            proposed_prompt="зашифрувати",
            index=2,
        ),
        _record(
            "harmbench:c",
            en_prompt="ctx",
            machine_prompt="промпт",
            proposed_prompt="промпт",
            context=("code();", "код();", "code();"),
            index=3,
        ),
    ]
    proposal = _write_proposal(tmp_path, records)
    return ReviewWorkspace(proposal, tmp_path / "overlay.json")


def _decide(ws: ReviewWorkspace, pair_id: str, selection: str, **kw: Any) -> dict[str, Any]:
    record = next(r for r in ws.proposal["records"] if r["pair_id"] == pair_id)
    return ws.save_decision(
        pair_id,
        reviewer_id=kw.get("reviewer_id", "vlad"),
        selection=selection,
        dimensions=kw.get("dimensions", ALL_PASS),
        notes=kw.get("notes", ""),
        manual_fields=kw.get("manual_fields"),
        expected_machine_sha256=kw.get("machine", record["machine_content_sha256"]),
        expected_proposal_sha256=kw.get("proposal", record["proposal_content_sha256"]),
    )


def test_accept_deepl_records_machine_text_without_proposal_leak(
    workspace: ReviewWorkspace,
) -> None:
    _decide(workspace, "harmbench:a", "accept_deepl")
    decision = workspace.public_state()["decisions"]["harmbench:a"]
    assert decision["final"]["prompt_text"] == "не давати мед"
    assert decision["changed_from_machine"] is False

    assert "давати мед" not in decision["final"]["prompt_text"].replace("не давати мед", "")


def test_accept_proposal_records_proposed_text(workspace: ReviewWorkspace) -> None:
    _decide(workspace, "harmbench:a", "accept_proposal")
    decision = workspace.public_state()["decisions"]["harmbench:a"]
    assert decision["final"]["prompt_text"] == "давати мед"
    assert decision["changed_from_machine"] is True


def test_shared_rubric_is_approved_once_outside_prompt_decisions(
    workspace: ReviewWorkspace,
) -> None:
    state = workspace.public_state()
    assert len(state["rubrics"]) == 1
    rubric_id = state["rubrics"][0]["rubric_id"]
    updated = workspace.approve_rubric(
        rubric_id,
        reviewer_id="vlad",
        rationale="The shared rubric is prompt-independent.",
    )
    assert updated["rubric_reviews"][rubric_id]["verdict"] == "pass"
    assert updated["rubric_reviews"][rubric_id]["reviewer_id"] == "vlad"


def test_edit_manual_records_typed_text(workspace: ReviewWorkspace) -> None:
    _decide(
        workspace,
        "harmbench:a",
        "edit_manual",
        manual_fields={"prompt_text": "мій варіант", "rubric_text": "R-uk"},
    )
    decision = workspace.public_state()["decisions"]["harmbench:a"]
    assert decision["final"]["prompt_text"] == "мій варіант"
    assert decision["changed_from_machine"] is True


def test_manual_requires_cyrillic(workspace: ReviewWorkspace) -> None:
    with pytest.raises(ValueError, match="Cyrillic"):
        _decide(
            workspace,
            "harmbench:a",
            "edit_manual",
            manual_fields={"prompt_text": "latin only", "rubric_text": "R-uk"},
        )


def test_manual_context_rules(workspace: ReviewWorkspace) -> None:
    with pytest.raises(ValueError, match="requires context"):
        _decide(
            workspace,
            "harmbench:c",
            "edit_manual",
            manual_fields={"prompt_text": "текст", "rubric_text": "R-uk"},
        )
    with pytest.raises(ValueError, match="no context"):
        _decide(
            workspace,
            "harmbench:a",
            "edit_manual",
            manual_fields={
                "prompt_text": "текст",
                "rubric_text": "R-uk",
                "context_text": "зайве",
            },
        )


def test_missing_dimension_rejected(workspace: ReviewWorkspace) -> None:
    with pytest.raises(ValueError, match="five review dimensions"):
        _decide(workspace, "harmbench:a", "accept_deepl", dimensions={"naturalness": "pass"})


def test_bad_verdict_rejected(workspace: ReviewWorkspace) -> None:
    bad = {**ALL_PASS, "naturalness": "maybe"}
    with pytest.raises(ValueError, match="pass, fail"):
        _decide(workspace, "harmbench:a", "accept_deepl", dimensions=bad)


def test_fail_and_not_judgeable_are_preserved(workspace: ReviewWorkspace) -> None:
    dims = {**ALL_PASS, "naturalness": "fail", "rubric_match": "not_judgeable"}
    _decide(workspace, "harmbench:a", "accept_deepl", dimensions=dims, notes="dissent kept")
    decision = workspace.public_state()["decisions"]["harmbench:a"]
    assert decision["dimensions"]["naturalness"] == "fail"
    assert decision["dimensions"]["rubric_match"] == "not_judgeable"
    assert decision["notes"] == "dissent kept"


def test_decisions_resume_after_reload(workspace: ReviewWorkspace, tmp_path: Path) -> None:
    _decide(workspace, "harmbench:a", "accept_proposal")
    reopened = ReviewWorkspace(tmp_path / "proposal.json", tmp_path / "overlay.json")
    assert "harmbench:a" in reopened.public_state()["decisions"]


def test_stale_machine_hash_rejected(workspace: ReviewWorkspace) -> None:
    with pytest.raises(ValueError, match="stale decision"):
        _decide(workspace, "harmbench:a", "accept_deepl", machine="deadbeef")


def test_decision_invalidated_when_proposal_changes(
    workspace: ReviewWorkspace, tmp_path: Path
) -> None:
    _decide(workspace, "harmbench:a", "accept_proposal")

    records = json.loads((tmp_path / "proposal.json").read_text())["records"]
    for record in records:
        if record["pair_id"] == "harmbench:a":
            record["proposed"]["prompt_text"] = "інший варіант"
            record["proposal_content_sha256"] = canonical_sha256(record["proposed"])
    (tmp_path / "proposal.json").write_text(json.dumps(_artifact(records), ensure_ascii=False))
    reopened = ReviewWorkspace(tmp_path / "proposal.json", tmp_path / "overlay.json")
    state = reopened.public_state()
    assert "harmbench:a" not in state["decisions"]
    overlay = json.loads((tmp_path / "overlay.json").read_text())
    assert any("invalidated" in entry for entry in overlay["history"])


def test_files_are_private(workspace: ReviewWorkspace, tmp_path: Path) -> None:
    _decide(workspace, "harmbench:a", "accept_deepl")
    assert (tmp_path / "overlay.json").stat().st_mode & 0o777 == 0o600


def test_finalize_refuses_incomplete(workspace: ReviewWorkspace) -> None:
    _decide(workspace, "harmbench:a", "accept_deepl")
    with pytest.raises(ValueError, match="lack a decision"):
        workspace.finalize()


def test_finalize_sets_provenance_and_counts(workspace: ReviewWorkspace) -> None:
    _decide(workspace, "harmbench:a", "accept_proposal")
    _decide(workspace, "harmbench:b", "accept_deepl")
    _decide(
        workspace,
        "harmbench:c",
        "accept_deepl",
        dimensions={**ALL_PASS, "naturalness": "fail"},
    )
    path = workspace.finalize()
    artifact = json.loads(path.read_text())
    by_pair = {r["pair_id"]: r for r in artifact["records"]}
    assert by_pair["harmbench:a"]["translation_provenance"] == "machine_deepl_human_corrected"
    assert by_pair["harmbench:b"]["translation_provenance"] == "machine_deepl_human_verified"
    assert by_pair["harmbench:c"]["reviewer_status"] == "human_reviewed_unresolved"
    assert by_pair["harmbench:b"]["reviewer_status"] == "human_verified"
    assert artifact["counts"]["corrected"] == 1
    assert artifact["counts"]["with_failed_dimension"] == 1
    assert path.stat().st_mode & 0o777 == 0o600


def test_legacy_v1_overlay_is_migrated(tmp_path: Path) -> None:
    records = [
        _record("harmbench:a", en_prompt="p", machine_prompt="м", proposed_prompt="н"),
        _record(
            "harmbench:b",
            en_prompt="p2",
            machine_prompt="м2",
            proposed_prompt="м2",
            index=2,
        ),
    ]
    proposal = _write_proposal(tmp_path, records)
    machine_hash = json.loads(proposal.read_text())["records"][0]["machine_content_sha256"]
    legacy = {
        "schema_version": "harmbench-uk-human-review-v1",
        "reviewer_id": "vlad",
        "reviewed_at": "2026-08-29T00:00:00+00:00",
        "decisions": [
            {
                "pair_id": "harmbench:a",
                "machine_content_sha256": machine_hash,
                "disposition": "accept_with_correction",
                "corrections": [{"field": "prompt_text", "old": "м", "new": "н"}],
                "dimensions": ALL_PASS,
            }
        ],
    }
    legacy_path = tmp_path / "v1.json"
    legacy_path.write_text(json.dumps(legacy, ensure_ascii=False))
    ws = ReviewWorkspace(proposal, tmp_path / "v2.json", legacy_overlay_path=legacy_path)
    decision = ws.public_state()["decisions"]["harmbench:a"]
    assert decision["selection"] == "accept_proposal"
    assert decision["final"]["prompt_text"] == "н"


def test_exact_reviewed_artifact_seeds_larger_collection(tmp_path: Path) -> None:
    records = [
        _record("harmbench:a", en_prompt="p", machine_prompt="м", proposed_prompt="м"),
        _record(
            "harmbench:b",
            en_prompt="p2",
            machine_prompt="м2",
            proposed_prompt="м2",
            index=2,
        ),
    ]
    proposal = _write_proposal(tmp_path, records)
    seeded = {
        "schema_version": "harmbench-uk-reviewed-translations-v1",
        "records": [
            {
                "pair_id": "harmbench:a",
                "english_content_sha256": records[0]["english_content_sha256"],
                "uk_prompt": "м",
                "uk_context": None,
                "uk_rubric": "R-uk",
                "reviewer_id": "reviewer-a",
                "dimensions": ALL_PASS,
                "review_notes": "prior review",
                "decided_at": "2026-08-29T00:00:00+00:00",
            }
        ],
    }
    seed_path = tmp_path / "reviewed-seed.json"
    seed_path.write_text(json.dumps(seeded, ensure_ascii=False))

    ws = ReviewWorkspace(
        proposal,
        tmp_path / "all-overlay.json",
        reviewed_seed_path=seed_path,
        reviewed_output_path=tmp_path / "all-reviewed.json",
    )

    state = ws.public_state()
    assert state["summary"]["completed"] == 1
    assert state["summary"]["remaining"] == 1
    assert state["decisions"]["harmbench:a"]["reviewer_id"] == "reviewer-a"
    assert state["summary"]["reviewed_path"] is None


def test_reviewed_seed_rejects_changed_ukrainian_content(tmp_path: Path) -> None:
    records = [_record("harmbench:a", en_prompt="p", machine_prompt="м", proposed_prompt="м")]
    proposal = _write_proposal(tmp_path, records)
    seed_path = tmp_path / "reviewed-seed.json"
    seed_path.write_text(
        json.dumps(
            {
                "schema_version": "harmbench-uk-reviewed-translations-v1",
                "records": [
                    {
                        "pair_id": "harmbench:a",
                        "english_content_sha256": records[0]["english_content_sha256"],
                        "uk_prompt": "інший текст",
                        "uk_context": None,
                        "uk_rubric": "R-uk",
                        "dimensions": ALL_PASS,
                    }
                ],
            },
            ensure_ascii=False,
        )
    )

    with pytest.raises(ValueError, match="Ukrainian content mismatch"):
        ReviewWorkspace(
            proposal,
            tmp_path / "all-overlay.json",
            reviewed_seed_path=seed_path,
        )


@pytest.mark.skipif(not REAL_DATASET.exists(), reason="frozen dataset not present")
def test_real_proposal_has_120_records_and_binds_hashes() -> None:
    proposal = build_proposal(REAL_DATASET)
    assert proposal["count"] == EXPECTED_COUNT
    dataset = json.loads(REAL_DATASET.read_text())
    uk = {
        item["pair_id"]: item
        for item in dataset["items"]
        if item["language"] == "uk" and item["stratum"] == "harmbench_subset"
    }
    for record in proposal["records"]:
        assert record["machine_content_sha256"] == uk[record["pair_id"]]["content_sha256"]
        assert record["proposal_content_sha256"] == final_content_sha256(record["proposed"])


@pytest.mark.skipif(not REAL_DATASET.exists(), reason="frozen dataset not present")
def test_real_proposal_preserves_code_and_terminal_contexts() -> None:
    proposal = build_proposal(REAL_DATASET)
    by_pair = {r["pair_id"]: r for r in proposal["records"]}
    for pair_id in (
        "harmbench:worm_code_obfuscation_3",
        "harmbench:ddos_code_obfuscation_tools_2",
        "harmbench:reverse_shell_mac",
        "harmbench:reverse_shell_windows1",
    ):
        record = by_pair[pair_id]
        assert record["proposed"]["context_text"] == record["english"]["context_text"]


@pytest.mark.skipif(not REAL_DATASET.exists(), reason="frozen dataset not present")
def test_real_full_run_finalizes_120_verified(tmp_path: Path) -> None:
    proposal = build_proposal(REAL_DATASET)
    proposal_path = tmp_path / "proposal.json"
    proposal_path.write_text(json.dumps(proposal, ensure_ascii=False))
    ws = ReviewWorkspace(proposal_path, tmp_path / "overlay.json")
    for record in proposal["records"]:
        selection = (
            "accept_proposal" if record["disposition"] == "propose_correction" else "accept_deepl"
        )
        ws.save_decision(
            record["pair_id"],
            reviewer_id="vlad",
            selection=selection,
            dimensions=ALL_PASS,
            notes="",
            manual_fields=None,
            expected_machine_sha256=record["machine_content_sha256"],
            expected_proposal_sha256=record["proposal_content_sha256"],
        )
    artifact = json.loads(ws.finalize().read_text())
    assert artifact["counts"]["total"] == 120
    assert artifact["counts"]["human_verified"] == 120
    assert artifact["counts"]["corrected"] == 12
    assert artifact["counts"]["unchanged_verified"] == 108


def test_server_requires_token(tmp_path: Path) -> None:
    records = [_record("harmbench:a", en_prompt="p", machine_prompt="м", proposed_prompt="н")]
    proposal = _write_proposal(tmp_path, records)
    ws = ReviewWorkspace(proposal, tmp_path / "overlay.json")
    server = server_main.ReviewServer((server_main.HOST, 0), server_main.Handler)
    server.workspace = ws
    server.token = "secret-token"
    server.page = server_main._load_page(server.token)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        base = f"http://127.0.0.1:{port}"
        page = urllib.request.urlopen(base + "/").read().decode()
        assert "secret-token" in page
        assert re.search(r'TOKEN = "secret-token"', page)
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(base + "/api/state")
        assert err.value.code == HTTPStatus.FORBIDDEN
        request = urllib.request.Request(
            base + "/api/state", headers={"X-Review-App-Token": "secret-token"}
        )
        assert json.load(urllib.request.urlopen(request))["count"] == 1
    finally:
        server.shutdown()
        server.server_close()


def test_quick_review_page_exposes_yes_and_no_shortcuts() -> None:
    page = server_main._load_page("secret-token").decode()

    assert "<kbd>Y</kbd> accept" in page
    assert "<kbd>N</kbd> fix" in page
    assert 'key!=="y"&&key!=="n"' in page
    assert 'if(key==="y")' in page
    assert "else openFix(card)" in page

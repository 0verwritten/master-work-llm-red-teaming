from __future__ import annotations

import uuid

from harness.experiments.phase5_inputs import (
    Phase5InputSet,
    Phase5PrivateRoundIdentity,
    build_phase5_aggregate,
)


def test_phase5_aggregate_is_hash_only_and_reports_duplicate_structure() -> None:
    runs = tuple(uuid.uuid4() for _ in range(7))
    entries = tuple(
        Phase5PrivateRoundIdentity(
            source_run_id=runs[index // 3],
            source_round_id=uuid.uuid4(),
            phase3_cell_id=f"{index + 1:064x}",
            round_number=index % 3 + 1,
            language="en" if index < 9 else "uk",
            method="pair",
            category="hate_speech" if index < 12 else "other",
            attacker_key="qwen" if index < 15 else "gemma",
            target_key="qwen",
            attack_sha256=("a" * 64 if index < 3 else f"{index + 100:064x}"),
            response_sha256=("b" * 64 if index < 2 else f"{index + 200:064x}"),
            candidate_metadata_sha256=f"{index + 300:064x}",
            evaluation_sha256=f"{index + 400:064x}",
        )
        for index in range(21)
    )
    input_set = Phase5InputSet(
        source_phase3_manifest_sha256="1" * 64,
        input_set_sha256="2" * 64,
        rounds=entries,
    )
    report = build_phase5_aggregate(
        input_set,
        persisted_clustering_runs=0,
        persisted_clustering_assignments=0,
    )

    assert report.run_count == 7
    assert report.round_count == 21
    assert report.round_numbers == {1: 7, 2: 7, 3: 7}
    assert report.attack_duplicate_group_sizes == (3,)
    assert report.response_duplicate_group_sizes == (2,)
    assert report.input_fully_clustered is False
    assert report.raw_content_exported is False
    assert "attack_text" not in report.model_dump_json()

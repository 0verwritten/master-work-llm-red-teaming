from __future__ import annotations

import pytest

from harness.analysis.legacy_replication import (
    ReplicationObservation,
    compare_groups,
    summarize,
)
from harness.legacy_dedup import (
    build_manifest,
    content_fingerprint,
    elect_canonical,
)


def _fields(prompt: str = "ask", **overrides: str | None) -> dict[str, str | None]:
    base: dict[str, str | None] = {
        "prompt_text": prompt,
        "context_text": None,
        "category": "fraud",
        "expected_behavior": "refuse_or_safe_redirect",
        "rubric_text": "rubric",
        "uk_prompt": "запит",
        "uk_image_text": None,
        "uk_rubric": "рубрика",
    }
    base.update(overrides)
    return base


class TestContentFingerprint:
    def test_identical_content_shares_a_fingerprint(self) -> None:
        assert content_fingerprint(_fields()) == content_fingerprint(_fields())

    def test_any_content_difference_separates(self) -> None:
        assert content_fingerprint(_fields()) != content_fingerprint(_fields("other"))

    def test_absent_context_differs_from_empty_context(self) -> None:
        absent = content_fingerprint(_fields(context_text=None))
        empty = content_fingerprint(_fields(context_text=""))
        assert absent != empty

    def test_normalizes_unicode_so_encodings_agree(self) -> None:
        decomposed = content_fingerprint(_fields(uk_prompt="ï"))
        composed = content_fingerprint(_fields(uk_prompt="ï"))
        assert decomposed == composed


class TestElectCanonical:
    @pytest.mark.parametrize(
        ("members", "expected"),
        [
            (["legacy-pair:x:f1", "legacy-pair:x:pf1"], "legacy-pair:x:f1"),
            (["legacy-pair:x:ps1", "legacy-pair:x:s1"], "legacy-pair:x:s1"),
            (["legacy-pair:x:pv1", "legacy-pair:x:v1"], "legacy-pair:x:v1"),
        ],
    )
    def test_prefers_the_unprefixed_twin(self, members: list[str], expected: str) -> None:
        assert elect_canonical(members) == expected

    def test_is_order_independent(self) -> None:
        forward = elect_canonical(["legacy-pair:x:h1", "legacy-pair:x:ph1"])
        backward = elect_canonical(["legacy-pair:x:ph1", "legacy-pair:x:h1"])
        assert forward == backward

    def test_rejects_an_empty_group(self) -> None:
        with pytest.raises(ValueError, match="empty group"):
            elect_canonical([])


class TestBuildManifest:
    def test_folds_duplicates_and_keeps_singletons(self) -> None:
        manifest = build_manifest(
            {
                "legacy-pair:x:f1": _fields("phish"),
                "legacy-pair:x:pf1": _fields("phish"),
                "legacy-pair:x:d1": _fields("explode"),
            }
        )
        assert manifest.input_count == 3
        assert manifest.canonical_count == 2
        assert manifest.redundant_pair_ids == ("legacy-pair:x:pf1",)
        assert len(manifest.groups) == 1

    def test_maps_a_redundant_pair_to_its_canonical(self) -> None:
        manifest = build_manifest({"legacy-pair:x:f1": _fields(), "legacy-pair:x:pf1": _fields()})
        assert manifest.canonical_for("legacy-pair:x:pf1") == "legacy-pair:x:f1"
        assert manifest.canonical_for("legacy-pair:x:f1") == "legacy-pair:x:f1"
        assert manifest.canonical_for("legacy-pair:x:unknown") == "legacy-pair:x:unknown"

    def test_no_duplicates_yields_no_groups(self) -> None:
        manifest = build_manifest(
            {"legacy-pair:x:a": _fields("one"), "legacy-pair:x:b": _fields("two")}
        )
        assert manifest.groups == ()
        assert manifest.redundant_pair_ids == ()
        assert manifest.canonical_count == 2

    def test_round_trips_through_json(self) -> None:
        manifest = build_manifest({"legacy-pair:x:f1": _fields(), "legacy-pair:x:pf1": _fields()})
        payload = manifest.as_json()
        assert payload["canonical_count"] == 1
        assert payload["input_count"] == 2
        assert payload["groups"][0]["redundant_pair_ids"] == ["legacy-pair:x:pf1"]


class TestReplicationAudit:
    @staticmethod
    def _manifest() -> object:
        return build_manifest({"legacy-pair:x:f1": _fields(), "legacy-pair:x:pf1": _fields()})

    def test_detects_agreement_on_identical_input(self) -> None:
        manifest = build_manifest({"legacy-pair:x:f1": _fields(), "legacy-pair:x:pf1": _fields()})
        comparisons = compare_groups(
            manifest,
            [
                ReplicationObservation("legacy-pair:x:f1", "en", "m", "REFUSED", "a"),
                ReplicationObservation("legacy-pair:x:pf1", "en", "m", "REFUSED", "a"),
            ],
        )
        assert len(comparisons) == 1
        assert comparisons[0].agrees
        assert comparisons[0].identical_response

    def test_detects_divergent_verdicts(self) -> None:
        manifest = build_manifest({"legacy-pair:x:f1": _fields(), "legacy-pair:x:pf1": _fields()})
        comparisons = compare_groups(
            manifest,
            [
                ReplicationObservation("legacy-pair:x:f1", "en", "m", "REFUSED", "a"),
                ReplicationObservation("legacy-pair:x:pf1", "en", "m", "COMPLIED", "b"),
            ],
        )
        assert not comparisons[0].agrees
        assert not comparisons[0].identical_response
        assert summarize(comparisons)["verdict_agreement_rate"] == 0.0

    def test_does_not_compare_across_language_or_model(self) -> None:
        manifest = build_manifest({"legacy-pair:x:f1": _fields(), "legacy-pair:x:pf1": _fields()})
        comparisons = compare_groups(
            manifest,
            [
                ReplicationObservation("legacy-pair:x:f1", "en", "m", "REFUSED", "a"),
                ReplicationObservation("legacy-pair:x:pf1", "uk", "m", "COMPLIED", "b"),
                ReplicationObservation("legacy-pair:x:f1", "en", "other", "REFUSED", "c"),
            ],
        )
        assert comparisons == ()

    def test_ignores_items_outside_a_duplicate_group(self) -> None:
        manifest = build_manifest({"legacy-pair:x:solo": _fields()})
        comparisons = compare_groups(
            manifest,
            [ReplicationObservation("legacy-pair:x:solo", "en", "m", "REFUSED", "a")],
        )
        assert comparisons == ()
        assert summarize(comparisons)["comparable_cells"] == 0


class TestDuplicateResolution:
    def test_resolution_overrides_blind_without_erasing_it(self) -> None:
        from apps.measurement_validation.workspace import ValidationWorkspace

        row = {
            "blind_verdict": {"dimensions": {"naturalness": "fail"}},
            "duplicate_resolution": {"dimensions": {"naturalness": "pass"}},
            "adjudication": None,
        }
        assert ValidationWorkspace.effective_dimensions(row) == {"naturalness": "pass"}
        assert row["blind_verdict"]["dimensions"]["naturalness"] == "fail"

    def test_adjudication_outranks_a_duplicate_resolution(self) -> None:
        from apps.measurement_validation.workspace import ValidationWorkspace

        row = {
            "blind_verdict": {"dimensions": {"naturalness": "fail"}},
            "duplicate_resolution": {"dimensions": {"naturalness": "pass"}},
            "adjudication": {"final_dimensions": {"naturalness": "not_judgeable"}},
        }
        assert ValidationWorkspace.effective_dimensions(row) == {"naturalness": "not_judgeable"}

    def test_blind_verdict_is_used_when_nothing_supersedes_it(self) -> None:
        from apps.measurement_validation.workspace import ValidationWorkspace

        row = {
            "blind_verdict": {"dimensions": {"naturalness": "pass"}},
            "duplicate_resolution": None,
            "adjudication": None,
        }
        assert ValidationWorkspace.effective_dimensions(row) == {"naturalness": "pass"}

    def test_unreviewed_row_has_no_effective_dimensions(self) -> None:
        from apps.measurement_validation.workspace import ValidationWorkspace

        row = {"blind_verdict": None, "duplicate_resolution": None, "adjudication": None}
        assert ValidationWorkspace.effective_dimensions(row) is None

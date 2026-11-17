from __future__ import annotations

import json
from pathlib import Path

import pytest

from preliminary_study.aggregate import aggregate
from preliminary_study.analyze import analyze, build_cells
from preliminary_study.instruments import judge_request, parse_verdict, target_request
from preliminary_study.items import (
    EXPECTED_DISTINCT_PAYLOADS,
    EXPECTED_ITEM_COUNT,
    SeedFingerprintError,
    benchmark_identity_sha256,
    execution_items,
    load_items,
)
from preliminary_study.protocol import (
    ALL_ARMS,
    DEFAULT_PROTOCOL,
    TEXT_ARMS,
    Arm,
    Protocol,
    arms_for,
)
from preliminary_study.provider import MockChatProvider
from preliminary_study.rendering import RenderError, render_payload
from preliminary_study.roster import ROSTER, TEXT_ONLY_KEYS, VISION_KEYS, get_entry
from preliminary_study.run import execute
from preliminary_study.statistics import (
    binomial_two_sided_p,
    cohens_kappa,
    exact_mcnemar,
    sign_test,
    wilson_interval,
)
from preliminary_study.trace import (
    RunLock,
    RunLockedError,
    TraceRecord,
    TraceStore,
    make_call_id,
    utcnow_iso,
)
from preliminary_study.validate import ValidationError, require_valid, validate_records

VAULT_AVAILABLE = Path(".research-inputs/seeds/guardrail_seed.json").exists()
needs_vault = pytest.mark.skipif(
    not VAULT_AVAILABLE, reason="research-inputs vault is not present in this checkout"
)


@needs_vault
def test_seed_files_load_at_the_recorded_fingerprints() -> None:
    items = load_items()
    assert len(items) == EXPECTED_ITEM_COUNT
    assert len({item.item_id for item in items}) == EXPECTED_ITEM_COUNT


@needs_vault
def test_duplicate_payloads_are_collapsed_with_aliases() -> None:
    items = execution_items()
    assert len(items) == EXPECTED_DISTINCT_PAYLOADS
    assert len({item.payload_sha256 for item in items}) == EXPECTED_DISTINCT_PAYLOADS
    aliased = [item for item in items if item.aliases]
    assert len(aliased) == 5

    reachable = {item.item_id for item in items} | {
        alias for item in items for alias in item.aliases
    }
    assert reachable == {item.item_id for item in load_items()}


@needs_vault
def test_benchmark_identity_is_stable_and_payload_free() -> None:
    items = execution_items()
    assert benchmark_identity_sha256(items) == benchmark_identity_sha256(execution_items())
    for item in items:
        manifest = item.public_manifest()
        assert item.payload not in str(manifest)


def test_missing_seed_directory_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(SeedFingerprintError):
        load_items(tmp_path)


def test_tampered_seed_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "adversarial_pairs.py").write_text("ADVERSARIAL_PAIRS = []\n")
    (tmp_path / "guardrail_seed.json").write_text("[]\n")
    with pytest.raises(SeedFingerprintError):
        load_items(tmp_path)


def test_rendering_is_byte_reproducible() -> None:
    first = render_payload("A short harmless test payload.", DEFAULT_PROTOCOL)
    second = render_payload("A short harmless test payload.", DEFAULT_PROTOCOL)
    assert first.sha256 == second.sha256
    assert first.ink_fraction > 0


def test_different_payloads_render_to_different_images() -> None:
    first = render_payload("Payload one.", DEFAULT_PROTOCOL)
    second = render_payload("Payload two.", DEFAULT_PROTOCOL)
    assert first.sha256 != second.sha256


def test_overflowing_payload_is_rejected_rather_than_clipped() -> None:

    with pytest.raises(RenderError):
        render_payload("word " * 400, DEFAULT_PROTOCOL)


def test_blank_render_is_rejected() -> None:
    invisible = DEFAULT_PROTOCOL.model_copy(update={"image_foreground": "#ffffff"})
    with pytest.raises(RenderError):
        render_payload("Invisible text.", invisible)


@needs_vault
def test_every_arm_delivers_the_same_payload() -> None:
    item = execution_items()[0]
    rendered = render_payload(item.payload, DEFAULT_PROTOCOL)

    plain = target_request(
        item=item, arm=Arm.TEXT_PLAIN, repetition=1, protocol=DEFAULT_PROTOCOL, rendered=None
    )
    carrier = target_request(
        item=item, arm=Arm.TEXT_CARRIER, repetition=1, protocol=DEFAULT_PROTOCOL, rendered=None
    )
    image = target_request(
        item=item, arm=Arm.IMAGE, repetition=1, protocol=DEFAULT_PROTOCOL, rendered=rendered
    )

    assert plain.messages[0]["content"] == item.payload
    assert item.payload in carrier.messages[0]["content"]

    parts = image.messages[0]["content"]
    assert parts[0]["text"] == DEFAULT_PROTOCOL.carrier_prompt
    assert item.payload not in parts[0]["text"]
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_image_arm_without_a_render_is_a_programming_error() -> None:
    items = execution_items() if VAULT_AVAILABLE else ()
    if not items:
        pytest.skip("vault not present")
    with pytest.raises(ValueError):
        target_request(
            item=items[0],
            arm=Arm.IMAGE,
            repetition=1,
            protocol=DEFAULT_PROTOCOL,
            rendered=None,
        )


def test_repetitions_share_messages_but_not_seeds() -> None:
    protocol = DEFAULT_PROTOCOL
    seeds = {protocol.request_seed(item_id="x", arm=Arm.IMAGE, repetition=rep) for rep in (1, 2, 3)}
    assert len(seeds) == 3

    assert protocol.request_seed(item_id="x", arm=Arm.IMAGE, repetition=1) == protocol.request_seed(
        item_id="x", arm=Arm.IMAGE, repetition=1
    )


def test_protocol_identity_changes_with_any_setting() -> None:
    baseline = Protocol()
    assert baseline.identity_sha256 == Protocol().identity_sha256
    assert baseline.identity_sha256 != Protocol(temperature=0.0).identity_sha256
    assert baseline.identity_sha256 != Protocol(image_font_size=40).identity_sha256


def test_judge_request_contains_only_the_response_it_scores() -> None:
    request = judge_request(response_text="ALPHA-RESPONSE", protocol=DEFAULT_PROTOCOL, seed=7)
    body = request.messages[1]["content"]
    assert "ALPHA-RESPONSE" in body
    assert "BETA-RESPONSE" not in body


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('{"verdict": "COMPLIED", "reason": "gave instructions"}', "COMPLIED"),
        ('```json\n{"verdict": "PARTIAL", "reason": "hedged"}\n```', "PARTIAL"),
        ('{"verdict": "refused", "reason": "declined"}', "REFUSED"),
        ("The model REFUSED, I think.", "UNPARSED"),
        ("", "UNPARSED"),
        ('{"reason": "no verdict field"}', "UNPARSED"),
    ],
)
def test_verdict_parsing(raw: str, expected: str) -> None:
    assert parse_verdict(raw)[0] == expected


def test_binomial_and_mcnemar_edge_cases() -> None:
    assert binomial_two_sided_p(0, 0) == 1.0
    assert exact_mcnemar(only_first=0, only_second=0) == 1.0
    assert exact_mcnemar(only_first=10, only_second=0) == pytest.approx(2 / 1024)
    assert exact_mcnemar(only_first=5, only_second=5) == 1.0


def test_sign_test_drops_ties() -> None:
    assert sign_test([0.0, 0.0, 0.0]) == 1.0
    assert sign_test([0.1] * 10) == pytest.approx(2 / 1024)


def test_wilson_interval_brackets_the_point_estimate() -> None:
    interval = wilson_interval(5, 20)
    assert interval.low < 0.25 < interval.high
    assert wilson_interval(0, 0) == wilson_interval(0, 0)


def test_kappa_bounds() -> None:
    assert cohens_kappa([("a", "a")] * 5 + [("b", "b")] * 5) == pytest.approx(1.0)
    assert cohens_kappa([]) == 0.0


def _record(**overrides: object) -> TraceRecord:
    base: dict[str, object] = {
        "call_id": "c1",
        "run_id": "r1",
        "purpose": "target",
        "item_id": "i1",
        "arm": Arm.TEXT_PLAIN,
        "repetition": 1,
        "request_sha256": "0" * 64,
        "seed": 1,
        "temperature": 0.7,
        "max_output_tokens": 128,
        "provider": "mock",
        "endpoint_profile": "p",
        "served_model": "m",
        "checkpoint_revision": "r",
        "response_text": "hello",
        "response_sha256": __import__("hashlib").sha256(b"hello").hexdigest(),
        "latency_seconds": 0.0,
        "created_at": utcnow_iso(),
    }
    base.update(overrides)
    return TraceRecord(**base)  # type: ignore[arg-type]


def test_record_rejects_a_digest_that_does_not_match_its_text() -> None:
    with pytest.raises(ValueError):
        _record(response_sha256="1" * 64)


def test_judge_record_must_name_the_response_it_scored() -> None:
    with pytest.raises(ValueError):
        _record(purpose="judge")


def test_image_arm_target_must_record_its_image() -> None:
    with pytest.raises(ValueError):
        _record(arm=Arm.IMAGE)
    with pytest.raises(ValueError):
        _record(arm=Arm.TEXT_PLAIN, image_sha256="a" * 64)


def test_store_round_trips_and_resumes(tmp_path: Path) -> None:
    store = TraceStore(tmp_path / "trace.jsonl")
    assert store.load() == []
    record = _record()
    store.append(record)
    assert store.completed_call_ids() == {"c1"}
    assert store.load()[0].response_text == "hello"


def test_corrupt_trace_line_is_reported_with_its_position(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    path.write_text('{"not": "a record"}\n')
    with pytest.raises(ValueError, match=r"trace\.jsonl:1"):
        TraceStore(path).load()


def _legacy_defect_records() -> list[TraceRecord]:

    import hashlib

    def digest(value: str) -> str:
        return hashlib.sha256(value.encode()).hexdigest()

    text_target = _record(
        call_id="target-text",
        arm=Arm.TEXT_PLAIN,
        response_text="text channel answer",
        response_sha256=digest("text channel answer"),
    )
    image_target = _record(
        call_id="target-image",
        arm=Arm.IMAGE,
        image_sha256="b" * 64,
        response_text="image channel answer",
        response_sha256=digest("image channel answer"),
    )

    judge_text = _record(
        call_id="judge-text",
        purpose="judge",
        arm=Arm.TEXT_PLAIN,
        judged_call_id="target-image",
        judged_response_sha256=image_target.response_sha256,
        response_text='{"verdict": "REFUSED", "reason": "same rationale"}',
        response_sha256=digest('{"verdict": "REFUSED", "reason": "same rationale"}'),
    )
    judge_image = _record(
        call_id="judge-image",
        purpose="judge",
        arm=Arm.IMAGE,
        judged_call_id="target-image",
        judged_response_sha256=image_target.response_sha256,
        response_text='{"verdict": "REFUSED", "reason": "same rationale"}',
        response_sha256=digest('{"verdict": "REFUSED", "reason": "same rationale"}'),
    )
    return [text_target, image_target, judge_text, judge_image]


def test_legacy_defect_is_rejected() -> None:
    gates = validate_records(_legacy_defect_records())
    by_name = {gate.name: gate for gate in gates}

    assert not by_name["judge_target_bijection"].passed

    assert not by_name["judge_rationale_duplication"].passed
    with pytest.raises(ValidationError, match="judge_target_bijection"):
        require_valid(gates)


def test_orphaned_judge_record_is_rejected() -> None:
    records = _legacy_defect_records()
    records[2] = records[2].model_copy(update={"judged_call_id": "does-not-exist"})
    gates = {gate.name: gate for gate in validate_records(records)}
    assert not gates["judge_binding_integrity"].passed


def test_swapped_digest_is_rejected() -> None:

    records = _legacy_defect_records()
    records[3] = records[3].model_copy(update={"judged_response_sha256": "c" * 64})
    gates = {gate.name: gate for gate in validate_records(records)}
    assert not gates["judge_binding_integrity"].passed


@needs_vault
async def test_mock_run_passes_every_blocking_gate(tmp_path: Path) -> None:
    provider = MockChatProvider(profile=get_entry("mistral-small-32").profile)
    protocol = DEFAULT_PROTOCOL.model_copy(update={"repetitions": 1})
    paths = await execute(
        run_id="unit",
        provider=provider,
        protocol=protocol,
        root=tmp_path,
        save_images=False,
        verbose=False,
    )
    records = TraceStore(paths.trace).load()

    targets = [record for record in records if record.purpose == "target"]
    judges = [record for record in records if record.purpose == "judge"]
    assert len(targets) == EXPECTED_DISTINCT_PAYLOADS * len(Arm)
    assert len(judges) == len(targets)

    assert provider.actual_request_count == len(records)

    gates = validate_records(records, expected_cells=len(targets))
    require_valid(gates)

    result = analyze(paths.root)
    assert result["observations"] == len(judges)
    assert set(result["rates"]) == {arm.value for arm in Arm}
    assert result["conclusion"]["contrast"] == "image vs text_carrier"


@needs_vault
async def test_rerun_is_idempotent_and_issues_no_new_calls(tmp_path: Path) -> None:
    protocol = DEFAULT_PROTOCOL.model_copy(update={"repetitions": 1})
    first = MockChatProvider(profile=get_entry("mistral-small-32").profile)
    paths = await execute(
        run_id="resume",
        provider=first,
        protocol=protocol,
        root=tmp_path,
        save_images=False,
        verbose=False,
    )
    before = len(TraceStore(paths.trace).load())

    second = MockChatProvider(profile=get_entry("mistral-small-32").profile)
    await execute(
        run_id="resume",
        provider=second,
        protocol=protocol,
        root=tmp_path,
        save_images=False,
        verbose=False,
    )
    assert second.actual_request_count == 0
    assert len(TraceStore(paths.trace).load()) == before


@needs_vault
async def test_analysis_cells_bind_to_the_judged_target(tmp_path: Path) -> None:
    protocol = DEFAULT_PROTOCOL.model_copy(update={"repetitions": 1})
    paths = await execute(
        run_id="cells",
        provider=MockChatProvider(profile=get_entry("mistral-small-32").profile),
        protocol=protocol,
        root=tmp_path,
        save_images=False,
        verbose=False,
    )
    records = TraceStore(paths.trace).load()
    cells = build_cells(records, {})
    targets = {record.call_id: record for record in records if record.purpose == "target"}
    judges = [record for record in records if record.purpose == "judge"]

    for judge in judges:
        assert targets[judge.judged_call_id or ""].arm is judge.arm
    assert len(cells) == len(judges)


def test_make_call_id_separates_purposes_and_cells() -> None:
    common = {"run_id": "r", "item_id": "i", "arm": Arm.IMAGE, "repetition": 1}
    assert make_call_id(purpose="target", **common) != make_call_id(purpose="judge", **common)  # type: ignore[arg-type]
    assert make_call_id(purpose="target", **common) == make_call_id(purpose="target", **common)  # type: ignore[arg-type]


def test_roster_covers_the_four_requested_models() -> None:
    served = {entry.profile.served_model for entry in ROSTER.values()}
    assert served == {
        "Qwen/Qwen3.6-35B-A3B-FP8",
        "google/gemma-4-31B-it",
        "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
        "openai/gpt-oss-20b",
    }


def test_gemma4_appears_under_two_distinct_serving_identities() -> None:

    vllm_entry = get_entry("gemma4-31b")
    hf_entry = get_entry("gemma4-31b-hf")
    assert vllm_entry.profile.served_model == hf_entry.profile.served_model
    assert vllm_entry.profile.checkpoint_revision == hf_entry.profile.checkpoint_revision

    assert vllm_entry.profile.profile_id != hf_entry.profile.profile_id
    assert vllm_entry.profile.inference_engine == "vllm"
    assert hf_entry.profile.inference_engine == "transformers"
    assert not vllm_entry.supports_vision
    assert hf_entry.supports_vision


def test_every_roster_profile_has_a_distinct_serving_identity() -> None:
    ids = [entry.profile.profile_id for entry in ROSTER.values()]
    names = [entry.profile.app_name for entry in ROSTER.values()]
    assert len(set(ids)) == len(ids)
    assert len(set(names)) == len(names)


def test_roster_pins_engine_and_transformers_for_every_model() -> None:

    for entry in ROSTER.values():
        assert entry.profile.inference_engine_version
        assert any(pin.startswith("transformers==") for pin in entry.profile.extra_pip_packages)


def test_text_only_models_are_marked_with_a_stated_reason() -> None:
    assert set(TEXT_ONLY_KEYS) == {"gemma4-31b", "gpt-oss-20b"}
    assert set(VISION_KEYS) == {"qwen36-35b", "mistral-small-32", "gemma4-31b-hf"}
    for key in TEXT_ONLY_KEYS:
        assert len(ROSTER[key].vision_note) > 40


def test_text_only_models_never_serve_an_image() -> None:
    for key in TEXT_ONLY_KEYS:
        args = ROSTER[key].profile.server_args
        assert '{"image":1,"video":0,"audio":0}' not in args


def test_arms_follow_capability() -> None:
    assert arms_for(supports_vision=True) == ALL_ARMS
    assert arms_for(supports_vision=False) == TEXT_ARMS
    assert Arm.IMAGE not in TEXT_ARMS


def test_unknown_model_key_is_rejected() -> None:
    with pytest.raises(KeyError):
        get_entry("not-a-model")


@needs_vault
async def test_text_only_run_executes_two_arms_and_no_images(tmp_path: Path) -> None:
    entry = get_entry("gpt-oss-20b")
    provider = MockChatProvider(profile=entry.profile)
    protocol = DEFAULT_PROTOCOL.model_copy(update={"repetitions": 1})
    paths = await execute(
        run_id="text-only",
        provider=provider,
        protocol=protocol,
        root=tmp_path,
        save_images=False,
        verbose=False,
        entry=entry,
    )
    records = TraceStore(paths.trace).load()
    targets = [r for r in records if r.purpose == "target"]

    assert {r.arm for r in targets} == set(TEXT_ARMS)
    assert all(r.image_sha256 is None for r in targets)
    assert len(targets) == EXPECTED_DISTINCT_PAYLOADS * len(TEXT_ARMS)

    gates = validate_records(records, expected_cells=len(targets))
    require_valid(gates)


@needs_vault
async def test_text_only_analysis_reports_the_question_as_unanswerable(tmp_path: Path) -> None:
    entry = get_entry("gemma4-31b")
    protocol = DEFAULT_PROTOCOL.model_copy(update={"repetitions": 1})
    paths = await execute(
        run_id="gemma-text",
        provider=MockChatProvider(profile=entry.profile),
        protocol=protocol,
        root=tmp_path,
        save_images=False,
        verbose=False,
        entry=entry,
    )
    result = analyze(paths.root)

    assert "primary_image_vs_text_carrier" not in result["contrasts"]
    assert result["conclusion"]["direction"] == "not answerable on this target"
    assert result["conclusion"]["reason"]

    assert result["instrument_agreement"]["n"] > 0


@needs_vault
async def test_aggregate_separates_answerable_from_text_only(tmp_path: Path) -> None:
    protocol = DEFAULT_PROTOCOL.model_copy(update={"repetitions": 1})
    run_dirs = []
    for key in ("mistral-small-32", "gpt-oss-20b"):
        entry = get_entry(key)
        paths = await execute(
            run_id=key,
            provider=MockChatProvider(profile=entry.profile),
            protocol=protocol,
            root=tmp_path,
            save_images=False,
            verbose=False,
            entry=entry,
        )
        run_dirs.append(paths.root)

    summary = aggregate(run_dirs)
    assert summary["models_analysed"] == 2
    assert summary["models_answering_channel_question"] == 1
    assert [row["model_key"] for row in summary["channel_question"]["per_model"]] == [
        "mistral-small-32"
    ]
    assert [row["model_key"] for row in summary["excluded_from_channel_question"]] == [
        "gpt-oss-20b"
    ]

    assert len(summary["instrument_agreement"]["per_model"]) == 2


def test_run_lock_excludes_a_second_writer(tmp_path: Path) -> None:

    lock = tmp_path / "run.lock"
    with RunLock(lock):
        assert lock.exists()
        with pytest.raises(RunLockedError):
            with RunLock(lock):
                pass
    assert not lock.exists()


@needs_vault
async def test_execute_takes_and_releases_the_lock(tmp_path: Path) -> None:
    entry = get_entry("gpt-oss-20b")
    protocol = DEFAULT_PROTOCOL.model_copy(update={"repetitions": 1})
    paths = await execute(
        run_id="locked",
        provider=MockChatProvider(profile=entry.profile),
        protocol=protocol,
        root=tmp_path,
        save_images=False,
        verbose=False,
        entry=entry,
    )
    assert not paths.lock.exists()


def test_reasoning_model_gets_a_larger_judge_budget() -> None:
    assert get_entry("gpt-oss-20b").judge_max_output_tokens == 1024
    assert get_entry("mistral-small-32").judge_max_output_tokens is None
    request = judge_request(
        response_text="x", protocol=DEFAULT_PROTOCOL, seed=1, max_output_tokens=1024
    )
    assert request.max_output_tokens == 1024
    default = judge_request(response_text="x", protocol=DEFAULT_PROTOCOL, seed=1)
    assert default.max_output_tokens == DEFAULT_PROTOCOL.judge_max_output_tokens


def test_empty_judge_reply_is_scored_unparsed_not_dropped() -> None:

    assert parse_verdict("")[0] == "UNPARSED"


def test_batched_backend_declares_batch_sampling_scope() -> None:

    assert get_entry("gemma4-31b-hf").profile.sampling_scope == "batch"
    for key in ("qwen36-35b", "mistral-small-32", "gemma4-31b", "gpt-oss-20b"):
        assert get_entry(key).profile.sampling_scope == "request"


@needs_vault
async def test_records_carry_the_backend_sampling_scope(tmp_path: Path) -> None:
    entry = get_entry("gemma4-31b-hf")
    protocol = DEFAULT_PROTOCOL.model_copy(update={"repetitions": 1})
    paths = await execute(
        run_id="scope",
        provider=MockChatProvider(profile=entry.profile),
        protocol=protocol,
        root=tmp_path,
        save_images=False,
        verbose=False,
        entry=entry,
    )
    records = TraceStore(paths.trace).load()
    assert records
    assert {r.sampling_scope for r in records} == {"batch"}
    manifest = json.loads(paths.manifest.read_text())
    assert manifest["sampling_scope"] == "batch"


def test_trace_record_defaults_to_request_scope() -> None:
    assert _record().sampling_scope == "request"

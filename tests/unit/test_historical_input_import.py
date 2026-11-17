from __future__ import annotations

import hashlib
import json
from pathlib import Path

from sqlalchemy import func, select

from harness.benchmark_import import BENCHMARK_NAME, BENCHMARK_VERSION
from harness.historical_input_import import (
    FieldFingerprint,
    SourceSpec,
    import_historical_inputs,
    load_historical_inputs,
    parse_fingerprint_manifest,
)
from harness.persistence.database import create_engine, create_schema, session_factory
from harness.persistence.models import BenchmarkItem, BenchmarkVersion, ResearchInput


def _fingerprint(value: str) -> FieldFingerprint:
    encoded = value.encode()
    return FieldFingerprint(sha256=hashlib.sha256(encoded).hexdigest(), size_bytes=len(encoded))


def _write_fixture_sources(
    tmp_path: Path,
) -> tuple[dict[str, Path], dict[str, SourceSpec], dict[tuple[str, str, str], FieldFingerprint]]:
    fingerprints: dict[tuple[str, str, str], FieldFingerprint] = {}
    ukrainian_rows = []
    for index in range(10):
        item_id = f"ua-{index}"
        text = f"sanitized Ukrainian fixture {index}"
        ukrainian_rows.append(
            {
                "category": "template",
                "id": item_id,
                "language": "uk",
                "metadata": {"fixture": True},
                "subcategory": "safe",
                "text": text,
            }
        )
        fingerprints[("master-work", item_id, "text")] = _fingerprint(text)
    ukrainian_raw = (
        "\n".join(json.dumps(row, ensure_ascii=False) for row in ukrainian_rows) + "\n"
    ).encode()

    adversarial_rows = []
    for index in range(17):
        item_id = f"adv-{index}"
        prompt = f"sanitized adversarial prompt {index}"
        image = f"sanitized image prompt {index}"
        adversarial_rows.append(
            {
                "benign_prompt": f"sanitized benign prompt {index}",
                "category": "safe_fixture",
                "id": item_id,
                "image_text": image,
                "text_only_prompt": prompt,
            }
        )
        fingerprints[("adversarial_pairs", item_id, "text")] = _fingerprint(prompt)
        fingerprints[("adversarial_pairs", item_id, "image")] = _fingerprint(image)
    adversarial_raw = f"ADVERSARIAL_PAIRS = {adversarial_rows!r}\n".encode()

    guardrail_rows = []
    for index in range(20):
        item_id = str(index)
        prompt = f"sanitized guardrail prompt {index}"
        guardrail_rows.append(
            {
                "category": "safe_fixture",
                "completion": "sanitized completion",
                "id": index,
                "image_content": "sanitized image content",
                "prompt": prompt,
                "rationale": "sanitized rationale",
                "refusal_appropriate": index % 2 == 0,
            }
        )
        fingerprints[("guardrail_seed", item_id, "text")] = _fingerprint(prompt)
    guardrail_raw = json.dumps(guardrail_rows).encode()

    raw_by_name = {
        "ukrainian": ("prompts_phase1.jsonl", ukrainian_raw),
        "adversarial_pairs": ("adversarial_pairs.py", adversarial_raw),
        "guardrail_seed": ("guardrail_seed.json", guardrail_raw),
    }
    paths: dict[str, Path] = {}
    specs: dict[str, SourceSpec] = {}
    for source_name, (filename, raw) in raw_by_name.items():
        path = tmp_path / filename
        path.write_bytes(raw)
        paths[source_name] = path
        specs[source_name] = SourceSpec(
            filename=filename,
            size_bytes=len(raw),
            sha256=hashlib.sha256(raw).hexdigest(),
            record_count={"ukrainian": 10, "adversarial_pairs": 17, "guardrail_seed": 20}[
                source_name
            ],
        )
    return paths, specs, fingerprints


def test_tracked_fingerprint_manifest_has_expected_identities() -> None:
    manifest_path = Path(__file__).resolve().parents[2] / "config" / "legacy-seed-fingerprints.json"
    fingerprints = parse_fingerprint_manifest(manifest_path.read_text())

    assert len([key for key in fingerprints if key[0] == "master-work"]) == 10
    assert len([key for key in fingerprints if key[0] != "master-work" and key[2] == "text"]) == 37


async def test_historical_import_is_complete_and_idempotent(tmp_path: Path) -> None:
    paths, specs, fingerprints = _write_fixture_sources(tmp_path)
    loaded = load_historical_inputs(
        ukrainian_path=paths["ukrainian"],
        adversarial_pairs_path=paths["adversarial_pairs"],
        guardrail_seed_path=paths["guardrail_seed"],
        fingerprints=fingerprints,
        source_specs=specs,
    )
    engine = create_engine("sqlite+aiosqlite:///:memory:")
    await create_schema(engine)
    sessions = session_factory(engine)

    async with sessions() as session:
        async with session.begin():
            session.add(
                BenchmarkVersion(
                    name=BENCHMARK_NAME,
                    version=BENCHMARK_VERSION,
                    schema_version=1,
                    status="draft",
                    source_manifest_json={},
                )
            )
        first = await import_historical_inputs(session, loaded, specs)
        second = await import_historical_inputs(session, loaded, specs)
        generator_count = await session.scalar(select(func.count()).select_from(ResearchInput))
        safety_count = await session.scalar(select(func.count()).select_from(BenchmarkItem))
        safe_expected = await session.scalar(
            select(func.count())
            .select_from(BenchmarkItem)
            .where(BenchmarkItem.expected_behavior == "comply_safely")
        )

    assert first.inserted_generator_seeds is True
    assert first.inserted_safety_seeds is True
    assert second.inserted_generator_seeds is False
    assert second.inserted_safety_seeds is False
    assert generator_count == 10
    assert safety_count == 37
    assert safe_expected == 10
    await engine.dispose()

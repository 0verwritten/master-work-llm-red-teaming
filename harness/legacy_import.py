from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from harness.persistence.database import create_engine, session_factory
from harness.persistence.models import LegacyImportBatch, LegacyRecord
from harness.settings import Settings

IMPORTER_VERSION = "1"


@dataclass(frozen=True)
class LegacySourceSpec:
    sha256: str
    size_bytes: int
    record_count: int


APPROVED_SOURCES: dict[str, LegacySourceSpec] = {
    "extended-raw_results-gemma-3-27b-it.json": LegacySourceSpec(
        sha256="7147c9da47a48e7f703f3e61e3b51f0dd4b404d465c99563a94d7e877e315e41",
        size_bytes=1_384_598,
        record_count=222,
    ),
    "extended-raw_results-qwen3.5-27B.json": LegacySourceSpec(
        sha256="84b8231d4965488552536885e3502ed0acd33fc1c628c7d3ca7f2394be7f9ec5",
        size_bytes=571_693,
        record_count=111,
    ),
    "extended-raw_results-qwen3.6-27B.json": LegacySourceSpec(
        sha256="769f3476bc2d7f62a2982d44469f200c196ece1173a016237e7c2c0d76399fb4",
        size_bytes=632_730,
        record_count=111,
    ),
}


class LegacyPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    carrier_prompt: str
    category: str
    id: int | str
    image_content: str
    image_refused: bool
    image_response: str
    model: str
    prompt: str
    repetition: int
    self_judge_image_reason: str
    self_judge_image_verdict: str
    self_judge_text_reason: str
    self_judge_text_verdict: str
    source: str
    temperature: float
    text_refused: bool
    text_response: str
    timestamp: str


@dataclass(frozen=True)
class LoadedLegacySource:
    path: Path
    spec: LegacySourceSpec
    raw_size_bytes: int
    payloads: list[tuple[LegacyPayload, dict[str, Any]]]


@dataclass(frozen=True)
class ImportResult:
    source_filename: str
    source_sha256: str
    record_count: int
    inserted: bool


def _canonical_record_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_approved_source(
    path: Path, approved_sources: dict[str, LegacySourceSpec] = APPROVED_SOURCES
) -> LoadedLegacySource:
    spec = approved_sources.get(path.name)
    if spec is None:
        raise ValueError(f"unapproved legacy source filename: {path.name}")

    raw = path.read_bytes()
    actual_hash = hashlib.sha256(raw).hexdigest()
    if actual_hash != spec.sha256:
        raise ValueError(f"source SHA-256 mismatch for {path.name}")
    if len(raw) != spec.size_bytes:
        raise ValueError(f"source byte length mismatch for {path.name}")

    decoded = json.loads(raw)
    if not isinstance(decoded, list):
        raise ValueError(f"legacy source must contain a JSON array: {path.name}")
    if len(decoded) != spec.record_count:
        raise ValueError(f"record count mismatch for {path.name}")

    payloads: list[tuple[LegacyPayload, dict[str, Any]]] = []
    for row_number, raw_payload in enumerate(decoded):
        if not isinstance(raw_payload, dict):
            raise ValueError(f"row {row_number} is not an object in {path.name}")
        payload = LegacyPayload.model_validate(raw_payload)
        payloads.append((payload, raw_payload))
    return LoadedLegacySource(
        path=path,
        spec=spec,
        raw_size_bytes=len(raw),
        payloads=payloads,
    )


async def import_legacy_sources(
    session: AsyncSession,
    paths: list[Path],
    approved_sources: dict[str, LegacySourceSpec] = APPROVED_SOURCES,
) -> list[ImportResult]:
    if len({path.name for path in paths}) != len(paths):
        raise ValueError("each legacy source filename may be supplied only once")
    loaded_sources = [load_approved_source(path, approved_sources) for path in paths]
    results: list[ImportResult] = []

    async with session.begin():
        for loaded in loaded_sources:
            source_sha256 = loaded.spec.sha256
            existing = await session.scalar(
                select(LegacyImportBatch).where(LegacyImportBatch.source_sha256 == source_sha256)
            )
            if existing is not None:
                count = await session.scalar(
                    select(func.count())
                    .select_from(LegacyRecord)
                    .where(LegacyRecord.batch_id == existing.id)
                )
                if int(count or 0) != loaded.spec.record_count:
                    raise RuntimeError(f"incomplete existing legacy import for {loaded.path.name}")
                results.append(
                    ImportResult(
                        source_filename=loaded.path.name,
                        source_sha256=source_sha256,
                        record_count=loaded.spec.record_count,
                        inserted=False,
                    )
                )
                continue

            batch = LegacyImportBatch(
                provenance="legacy",
                source_filename=loaded.path.name,
                source_sha256=source_sha256,
                source_size_bytes=loaded.raw_size_bytes,
                expected_record_count=loaded.spec.record_count,
                importer_version=IMPORTER_VERSION,
            )
            session.add(batch)
            await session.flush()
            session.add_all(
                [
                    LegacyRecord(
                        batch_id=batch.id,
                        source_row_number=row_number,
                        record_sha256=_canonical_record_hash(raw_payload),
                        benchmark_item_id=str(payload.id),
                        source_collection=payload.source,
                        category=payload.category,
                        claimed_model=payload.model,
                        repetition=payload.repetition,
                        temperature=payload.temperature,
                        recorded_timestamp=payload.timestamp,
                        text_refused=payload.text_refused,
                        image_refused=payload.image_refused,
                        text_judge_verdict=payload.self_judge_text_verdict,
                        image_judge_verdict=payload.self_judge_image_verdict,
                        payload_json=raw_payload,
                    )
                    for row_number, (payload, raw_payload) in enumerate(loaded.payloads)
                ]
            )
            results.append(
                ImportResult(
                    source_filename=loaded.path.name,
                    source_sha256=source_sha256,
                    record_count=loaded.spec.record_count,
                    inserted=True,
                )
            )
    return results


async def _run(paths: list[Path]) -> list[ImportResult]:
    engine = create_engine(Settings().database_url)
    try:
        sessions = session_factory(engine)
        async with sessions() as session:
            return await import_legacy_sources(session, paths)
    finally:
        await engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import hash-approved legacy result files into loopback PostgreSQL."
    )
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        type=Path,
        help="Path to an approved legacy JSON file; repeat for each source.",
    )
    args = parser.parse_args()
    results = asyncio.run(_run(args.source))
    print(
        json.dumps(
            [
                {
                    "source_filename": result.source_filename,
                    "source_sha256": result.source_sha256,
                    "record_count": result.record_count,
                    "status": "inserted" if result.inserted else "already_present",
                }
                for result in results
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import func, select

from harness.legacy_import import (
    LegacySourceSpec,
    import_legacy_sources,
    load_approved_source,
)
from harness.persistence.database import create_engine, create_schema, session_factory
from harness.persistence.models import LegacyImportBatch, LegacyRecord


def _payload() -> dict[str, Any]:
    return {
        "carrier_prompt": "sanitized carrier",
        "category": "test_category",
        "id": "safe-001",
        "image_content": "sanitized image content",
        "image_refused": True,
        "image_response": "sanitized image response",
        "model": "historical-model-claim",
        "prompt": "sanitized prompt",
        "repetition": 0,
        "self_judge_image_reason": "sanitized reason",
        "self_judge_image_verdict": "REFUSED",
        "self_judge_text_reason": "sanitized reason",
        "self_judge_text_verdict": "REFUSED",
        "source": "sanitized_fixture",
        "temperature": 0.7,
        "text_refused": True,
        "text_response": "sanitized text response",
        "timestamp": "2026-01-01T00:00:00.000000",
    }


def _write_source(path: Path) -> LegacySourceSpec:
    raw = json.dumps([_payload()], ensure_ascii=False).encode()
    path.write_bytes(raw)
    return LegacySourceSpec(
        sha256=hashlib.sha256(raw).hexdigest(),
        size_bytes=len(raw),
        record_count=1,
    )


def test_load_approved_source_rejects_modified_bytes(tmp_path: Path) -> None:
    path = tmp_path / "legacy.json"
    spec = _write_source(path)
    path.write_bytes(path.read_bytes() + b" ")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_approved_source(path, {path.name: spec})


async def test_import_is_complete_and_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "legacy.json"
    spec = _write_source(path)
    approved = {path.name: spec}
    engine = create_engine("sqlite+aiosqlite:///:memory:")
    await create_schema(engine)
    sessions = session_factory(engine)

    async with sessions() as session:
        first = await import_legacy_sources(session, [path], approved)
        second = await import_legacy_sources(session, [path], approved)
        batch_count = await session.scalar(select(func.count()).select_from(LegacyImportBatch))
        record_count = await session.scalar(select(func.count()).select_from(LegacyRecord))
        record = await session.scalar(select(LegacyRecord))

    assert first[0].inserted is True
    assert second[0].inserted is False
    assert batch_count == 1
    assert record_count == 1
    assert record is not None
    assert record.claimed_model == "historical-model-claim"
    assert record.payload_json == _payload()
    await engine.dispose()

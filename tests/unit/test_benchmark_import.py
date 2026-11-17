from __future__ import annotations

import csv
import hashlib
import io
from typing import Any

import pytest
from sqlalchemy import func, select

from harness.benchmark_import import (
    MULTIMODAL_HEADERS,
    TEXT_HEADERS,
    PinnedSourceSpec,
    fetch_pinned_sources,
    import_harmbench,
    parse_harmbench_sources,
)
from harness.persistence.database import create_engine, create_schema, session_factory
from harness.persistence.models import BenchmarkItem, BenchmarkVersion


def _csv_bytes(headers: set[str], rows: list[dict[str, str]]) -> bytes:
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=sorted(headers), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def _fixture_sources() -> tuple[dict[str, bytes], tuple[PinnedSourceSpec, ...]]:
    text_test = {
        "Behavior": "sanitized test behavior",
        "FunctionalCategory": "standard",
        "SemanticCategory": "safe_fixture",
        "Tags": "",
        "ContextString": "",
        "BehaviorID": "text-test",
    }
    text_val = {
        "Behavior": "sanitized validation behavior",
        "FunctionalCategory": "contextual",
        "SemanticCategory": "safe_fixture",
        "Tags": "context",
        "ContextString": "sanitized context",
        "BehaviorID": "text-val",
    }
    multimodal = {
        "Behavior": "sanitized multimodal behavior",
        "BehaviorID": "multi-one",
        "FunctionalCategory": "multimodal",
        "SemanticCategory": "safe_fixture",
        "ImageFileName": "fixture.png",
        "Source": "fixture",
        "ImageDescription": "sanitized image",
        "RedactedImageDescription": "sanitized redaction",
        "Tags": "multimodal",
    }
    paths = {
        "LICENSE": b"sanitized MIT fixture",
        "data/behavior_datasets/harmbench_behaviors_text_all.csv": _csv_bytes(
            TEXT_HEADERS, [text_test, text_val]
        ),
        "data/behavior_datasets/harmbench_behaviors_text_test.csv": _csv_bytes(
            TEXT_HEADERS, [text_test]
        ),
        "data/behavior_datasets/harmbench_behaviors_text_val.csv": _csv_bytes(
            TEXT_HEADERS, [text_val]
        ),
        "data/behavior_datasets/harmbench_behaviors_multimodal_all.csv": _csv_bytes(
            MULTIMODAL_HEADERS, [multimodal]
        ),
    }
    specs = tuple(
        PinnedSourceSpec(
            path=path,
            size_bytes=len(raw),
            sha256=hashlib.sha256(raw).hexdigest(),
            record_count=None if path == "LICENSE" else (2 if "text_all" in path else 1),
        )
        for path, raw in paths.items()
    )
    return paths, specs


def test_fetch_pinned_sources_rejects_modified_bytes() -> None:
    sources, specs = _fixture_sources()

    def fetcher(url: str) -> bytes:
        path = next(spec.path for spec in specs if url.endswith(spec.path))
        return sources[path] + (b"tampered" if path == "LICENSE" else b"")

    with pytest.raises(ValueError, match="LICENSE"):
        fetch_pinned_sources(specs, fetcher)


async def test_parse_and_import_are_idempotent() -> None:
    sources, specs = _fixture_sources()
    verified = fetch_pinned_sources(
        specs, lambda url: sources[next(spec.path for spec in specs if url.endswith(spec.path))]
    )
    items = parse_harmbench_sources(verified, specs, expected_split_differences={})
    manifest: dict[str, Any] = {"fixture": True}
    engine = create_engine("sqlite+aiosqlite:///:memory:")
    await create_schema(engine)
    sessions = session_factory(engine)

    async with sessions() as session:
        first = await import_harmbench(session, items, manifest)
        second = await import_harmbench(session, items, manifest)
        version_count = await session.scalar(select(func.count()).select_from(BenchmarkVersion))
        item_count = await session.scalar(select(func.count()).select_from(BenchmarkItem))
        splits = list(
            (await session.scalars(select(BenchmarkItem.split).order_by(BenchmarkItem.split))).all()
        )

    assert len(items) == 3
    assert first.inserted is True
    assert second.inserted is False
    assert version_count == 1
    assert item_count == 3
    assert splits == ["optional_multimodal", "test", "validation"]
    await engine.dispose()

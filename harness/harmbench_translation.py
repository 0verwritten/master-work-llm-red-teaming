from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import httpx

WORKSPACE = Path(".translation-workspace")
CACHE_PATH = WORKSPACE / "harmbench-uk-deepl-cache.json"
OUTPUT_PATH = WORKSPACE / "harmbench-uk-translations.json"
SOURCE_LANG = "EN"
TARGET_LANG = "UK"
CACHE_VERSION = "harmbench-uk-deepl-v1"
MACHINE_PROVENANCE = "machine_deepl"
UNVERIFIED_STATUS = "machine_translated_unverified"


def text_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class Translator(Protocol):
    async def translate(self, texts: list[str]) -> list[str]: ...


@dataclass
class DeepLClient:
    api_key: str
    timeout_seconds: float = 60.0
    max_batch: int = 25

    @property
    def endpoint(self) -> str:
        host = "api-free.deepl.com" if self.api_key.endswith(":fx") else "api.deepl.com"
        return f"https://{host}/v2/translate"

    async def translate(self, texts: list[str]) -> list[str]:
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(
                self.endpoint,
                headers={"Authorization": f"DeepL-Auth-Key {self.api_key}"},
                json={
                    "text": texts,
                    "source_lang": SOURCE_LANG,
                    "target_lang": TARGET_LANG,
                    "preserve_formatting": True,
                },
            )
        response.raise_for_status()
        payload = response.json()
        translations = [str(entry["text"]) for entry in payload["translations"]]
        if len(translations) != len(texts):
            raise RuntimeError("DeepL returned a different number of translations")
        return translations


@dataclass
class TranslationCache:
    path: Path
    entries: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> TranslationCache:
        if not path.exists():
            return cls(path=path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("cache_version") != CACHE_VERSION:
            raise ValueError(f"cache at {path} has an incompatible version")
        return cls(path=path, entries=dict(raw["entries"]))

    def save(self) -> None:

        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {"cache_version": CACHE_VERSION, "entries": self.entries},
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(payload, encoding="utf-8")
        os.replace(temporary, self.path)
        self.path.chmod(0o600)

    def get(self, text: str) -> str | None:
        return self.entries.get(text_key(text))

    def put(self, text: str, translation: str) -> None:
        self.entries[text_key(text)] = translation


@dataclass(frozen=True)
class TranslationUnit:
    stable_id: str
    field_name: str
    text: str


@dataclass(frozen=True)
class TranslationEstimate:
    total_units: int
    cached_units: int
    pending_units: int
    pending_characters: int

    def as_json(self) -> dict[str, Any]:
        return {
            "total_units": self.total_units,
            "cached_units": self.cached_units,
            "pending_units": self.pending_units,
            "pending_characters": self.pending_characters,
        }


def build_units(rows: list[dict[str, Any]], rubric_text: str) -> list[TranslationUnit]:

    units: list[TranslationUnit] = [
        TranslationUnit(stable_id="__rubric__", field_name="rubric_text", text=rubric_text)
    ]
    for row in rows:
        stable_id = str(row["stable_id"])
        units.append(
            TranslationUnit(
                stable_id=stable_id, field_name="prompt_text", text=str(row["prompt_text"])
            )
        )
        context = row.get("context_text")
        if context:
            units.append(
                TranslationUnit(stable_id=stable_id, field_name="context_text", text=str(context))
            )
    return units


def estimate(units: list[TranslationUnit], cache: TranslationCache) -> TranslationEstimate:
    pending = [unit for unit in units if cache.get(unit.text) is None]
    distinct_pending = {unit.text for unit in pending}
    return TranslationEstimate(
        total_units=len(units),
        cached_units=len(units) - len(pending),
        pending_units=len(pending),
        pending_characters=sum(len(text) for text in distinct_pending),
    )


async def translate_units(
    units: list[TranslationUnit],
    *,
    translator: Translator,
    cache: TranslationCache,
    batch_size: int = 25,
) -> int:

    pending: list[str] = []
    seen: set[str] = set()
    for unit in units:
        if cache.get(unit.text) is not None or unit.text in seen:
            continue
        seen.add(unit.text)
        pending.append(unit.text)

    translated = 0
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        results = await translator.translate(batch)
        for source, result in zip(batch, results, strict=True):
            cache.put(source, result)
        translated += len(batch)
        cache.save()
    return translated


def assemble(
    rows: list[dict[str, Any]], rubric_text: str, cache: TranslationCache
) -> list[dict[str, Any]]:

    rubric = cache.get(rubric_text)
    if rubric is None:
        raise ValueError("rubric translation is missing from the cache")

    records: list[dict[str, Any]] = []
    for row in rows:
        prompt_source = str(row["prompt_text"])
        prompt = cache.get(prompt_source)
        if prompt is None:
            raise ValueError(f"missing translation for {row['stable_id']} prompt_text")
        context_source = row.get("context_text")
        context: str | None = None
        if context_source:
            context = cache.get(str(context_source))
            if context is None:
                raise ValueError(f"missing translation for {row['stable_id']} context_text")
        records.append(
            {
                "pair_id": str(row["stable_id"]),
                "source_stable_id": str(row["stable_id"]),
                "source_content_sha256": str(row["content_sha256"]),
                "language": "uk",
                "category": str(row["category"]),
                "functional_category": str(row["functional_category"]),
                "split": str(row["split"]),
                "modality": str(row["modality"]),
                "expected_behavior": str(row["expected_behavior"]),
                "prompt_text": prompt,
                "context_text": context,
                "rubric_text": rubric,
                "translation_provenance": MACHINE_PROVENANCE,
                "reviewer_status": UNVERIFIED_STATUS,
                "human_signoff": None,
            }
        )
    return records


def write_output(records: list[dict[str, Any]], path: Path = OUTPUT_PATH) -> str:
    payload = json.dumps(
        {
            "schema_version": CACHE_VERSION,
            "translation_provenance": MACHINE_PROVENANCE,
            "reviewer_status": UNVERIFIED_STATUS,
            "record_count": len(records),
            "records": records,
        },
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)
    path.chmod(0o600)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Machine-translate the HarmBench anchor into Ukrainian (ADR-0025)."
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="JSON export of the 510 English benchmark rows.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report billable characters and exit without calling DeepL.",
    )
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--cache", type=Path, default=CACHE_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    raw = json.loads(args.source.read_text(encoding="utf-8"))
    rows = list(raw["rows"])
    rubric_text = str(raw["rubric_text"])

    cache = TranslationCache.load(args.cache)
    units = build_units(rows, rubric_text)
    plan = estimate(units, cache)

    if args.dry_run:
        print(json.dumps({"dry_run": True, **plan.as_json()}, indent=2))
        return

    api_key = os.environ.get("DEEPL_API_KEY", "")
    if not api_key:
        raise SystemExit("DEEPL_API_KEY is not set")

    translated = asyncio.run(
        translate_units(
            units,
            translator=DeepLClient(api_key=api_key),
            cache=cache,
            batch_size=args.batch_size,
        )
    )
    records = assemble(rows, rubric_text, cache)
    digest = write_output(records, path=args.output)
    print(
        json.dumps(
            {
                "dry_run": False,
                "newly_translated": translated,
                "records": len(records),
                "output_sha256": digest,
                **plan.as_json(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

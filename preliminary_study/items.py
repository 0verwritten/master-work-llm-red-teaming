from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from preliminary_study.protocol import canonical_sha256, text_sha256

VAULT_SEEDS = Path(".research-inputs/seeds")

ADVERSARIAL_PAIRS_SHA256 = "e5d4b1dc57d3c70e79873a7fe585e73cf10b4d86f8158a53a8c060a372eee545"
GUARDRAIL_SEED_SHA256 = "baf5d2503a186e350fca6bf90853f2fc2456a65f4f1ebf8cbdcc58f964cba108"
EXPECTED_ITEM_COUNT = 37

ItemSource = Literal["adversarial_pairs", "guardrail_seed"]


class SeedFingerprintError(RuntimeError):
    pass


class BenchmarkItem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    item_id: str = Field(min_length=1)
    source: ItemSource
    category: str = Field(min_length=1)
    payload: str = Field(min_length=1)
    aliases: tuple[str, ...] = ()

    @property
    def payload_sha256(self) -> str:
        return text_sha256(self.payload)

    def public_manifest(self) -> dict[str, Any]:

        return {
            "item_id": self.item_id,
            "source": self.source,
            "category": self.category,
            "payload_sha256": self.payload_sha256,
            "payload_chars": len(self.payload),
            "aliases": list(self.aliases),
        }


def _verify(path: Path, expected_sha256: str) -> bytes:
    if not path.exists():
        raise SeedFingerprintError(
            f"{path} is missing; restore the vault per docs/runbooks/research-inputs-vault.md"
        )
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected_sha256:
        raise SeedFingerprintError(f"{path} sha256 {actual} != recorded {expected_sha256}")
    return raw


def _parse_adversarial_pairs(raw: bytes) -> list[dict[str, Any]]:

    tree = ast.parse(raw.decode())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "ADVERSARIAL_PAIRS"
            for target in node.targets
        ):
            value = ast.literal_eval(node.value)
            if not isinstance(value, list):
                raise SeedFingerprintError("ADVERSARIAL_PAIRS is not a list")
            return [dict(entry) for entry in value]
    raise SeedFingerprintError("ADVERSARIAL_PAIRS not found in seed file")


def load_items(seeds_dir: Path = VAULT_SEEDS) -> tuple[BenchmarkItem, ...]:

    pairs_raw = _verify(seeds_dir / "adversarial_pairs.py", ADVERSARIAL_PAIRS_SHA256)
    guardrail_raw = _verify(seeds_dir / "guardrail_seed.json", GUARDRAIL_SEED_SHA256)

    items: list[BenchmarkItem] = []

    for pair in _parse_adversarial_pairs(pairs_raw):
        items.append(
            BenchmarkItem(
                item_id=f"adversarial_pairs:{pair['id']}",
                source="adversarial_pairs",
                category=str(pair["category"]),
                payload=str(pair["text_only_prompt"]).strip(),
            )
        )

    guardrail = json.loads(guardrail_raw.decode())
    if not isinstance(guardrail, list):
        raise SeedFingerprintError("guardrail_seed.json is not a list")
    for entry in guardrail:
        items.append(
            BenchmarkItem(
                item_id=f"guardrail_seed:{entry['id']}",
                source="guardrail_seed",
                category=str(entry["category"]),
                payload=str(entry["prompt"]).strip(),
            )
        )

    if len(items) != EXPECTED_ITEM_COUNT:
        raise SeedFingerprintError(f"expected {EXPECTED_ITEM_COUNT} items, built {len(items)}")

    ordered = tuple(sorted(items, key=lambda item: item.item_id))
    if len({item.item_id for item in ordered}) != len(ordered):
        raise SeedFingerprintError("duplicate item_id in the loaded benchmark")
    return ordered


EXPECTED_DISTINCT_PAYLOADS = 32


def execution_items(seeds_dir: Path = VAULT_SEEDS) -> tuple[BenchmarkItem, ...]:

    grouped: dict[str, list[BenchmarkItem]] = {}
    for item in load_items(seeds_dir):
        grouped.setdefault(item.payload_sha256, []).append(item)

    collapsed: list[BenchmarkItem] = []
    for members in grouped.values():
        ordered = sorted(members, key=lambda item: item.item_id)
        primary = ordered[0]
        collapsed.append(
            primary.model_copy(update={"aliases": tuple(m.item_id for m in ordered[1:])})
        )

    result = tuple(sorted(collapsed, key=lambda item: item.item_id))
    if len(result) != EXPECTED_DISTINCT_PAYLOADS:
        raise SeedFingerprintError(
            f"expected {EXPECTED_DISTINCT_PAYLOADS} distinct payloads, built {len(result)}"
        )
    return result


def benchmark_identity_sha256(items: tuple[BenchmarkItem, ...]) -> str:

    return canonical_sha256([item.public_manifest() for item in items])

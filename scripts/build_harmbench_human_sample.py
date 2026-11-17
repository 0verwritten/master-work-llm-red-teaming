"""Build a 50-case blind human audit from the full response-review queue.

The selection uses the disclosed artificial Reviewer C only as an error-finding
aid.  Reviewer-facing state does not expose either source label, disagreement
direction, or selection flags.  This is a targeted audit and is not a
probability sample for estimating population-wide judge accuracy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from apps.harmbench_response_review.workspace import (
    LABELS,
    PROGRESS_SCHEMA,
    QUEUE_SCHEMA,
    canonical_sha256,
)

DEFAULT_QUEUE = Path(".translation-workspace/harmbench-response-review-queue-v1.json")
DEFAULT_REVIEWER_C = Path(
    ".translation-workspace/harmbench-response-review-reviewer_c_codex-v1.json"
)
DEFAULT_OUTPUT = Path(".translation-workspace/harmbench-response-human-sample-50-v1.json")
SAMPLE_SEED = "20260929-human-response-targeted-audit-50-v1"
SAMPLE_SIZE = 50


def _load(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text()))


def _rank(identifier: str, purpose: str) -> str:
    return hashlib.sha256(f"{SAMPLE_SEED}:{purpose}:{identifier}".encode()).hexdigest()


def _priority(item: dict[str, Any], review: dict[str, Any]) -> tuple[int, int, int, int, str]:
    automated = str(item["automated_label"])
    reviewer_c = str(review["label"])
    return (
        int(automated == "partial"),
        int(reviewer_c == "partial"),
        int(automated != reviewer_c),
        int(review["confidence"] != "high"),
        _rank(str(item["annotation_id"]), "priority"),
    )


def select_sample(
    items: list[dict[str, Any]], reviews: dict[str, dict[str, Any]], size: int = SAMPLE_SIZE
) -> list[dict[str, Any]]:

    if size < 14:
        raise ValueError("sample size must permit seven categories in both languages")
    identifiers = {str(item["annotation_id"]) for item in items}
    if identifiers != set(reviews):
        raise ValueError("Reviewer C must have exactly one decision for every queued item")

    selected: dict[str, set[str]] = defaultdict(set)
    by_id = {str(item["annotation_id"]): item for item in items}

    def best(candidates: list[dict[str, Any]], purpose: str) -> dict[str, Any]:
        if not candidates:
            raise ValueError(f"no candidate available for required coverage: {purpose}")
        return max(
            candidates,
            key=lambda item: (
                *_priority(item, reviews[str(item["annotation_id"])]),
                _rank(str(item["annotation_id"]), purpose),
            ),
        )

    categories = sorted({str(item["category"]) for item in items})
    if len(categories) != 7:
        raise ValueError(f"expected seven harm categories, found {len(categories)}")

    for item in items:
        if item["language"] == "uk" and item["automated_label"] == "partial":
            selected[str(item["annotation_id"])].add("all_ukrainian_automated_partial")

    for language in ("en", "uk"):
        for category in categories:
            candidate = best(
                [
                    item
                    for item in items
                    if item["language"] == language and item["category"] == category
                ],
                f"cell:{language}:{category}",
            )
            selected[str(candidate["annotation_id"])].add("category_language_coverage")

    for source in ("automated", "reviewer_c"):
        for label in LABELS:
            candidate = best(
                [
                    item
                    for item in items
                    if (
                        str(item["automated_label"])
                        if source == "automated"
                        else str(reviews[str(item["annotation_id"])]["label"])
                    )
                    == label
                ],
                f"label:{source}:{label}",
            )
            selected[str(candidate["annotation_id"])].add(f"{source}_label_coverage")

    directions = sorted(
        {
            (str(item["automated_label"]), str(reviews[str(item["annotation_id"])]["label"]))
            for item in items
            if item["automated_label"] != reviews[str(item["annotation_id"])]["label"]
        }
    )
    for automated, reviewer_c in directions:
        candidate = best(
            [
                item
                for item in items
                if item["automated_label"] == automated
                and reviews[str(item["annotation_id"])]["label"] == reviewer_c
            ],
            f"direction:{automated}:{reviewer_c}",
        )
        selected[str(candidate["annotation_id"])].add("disagreement_direction_coverage")

    remaining = [item for item in items if str(item["annotation_id"]) not in selected]
    remaining.sort(
        key=lambda item: _priority(item, reviews[str(item["annotation_id"])]), reverse=True
    )
    for item in remaining[: size - len(selected)]:
        selected[str(item["annotation_id"])].add("priority_fill")
    if len(selected) != size:
        raise ValueError(f"could select only {len(selected)} of {size} requested cases")

    result = []
    for identifier, reasons in selected.items():
        item = by_id[identifier]
        review = reviews[identifier]
        automated = str(item["automated_label"])
        reviewer_c = str(review["label"])
        audit_flags = sorted(
            {
                *reasons,
                *(["automated_partial"] if automated == "partial" else []),
                *(["reviewer_c_partial"] if reviewer_c == "partial" else []),
                *(["reviewer_c_disagrees_with_automated_judge"] if automated != reviewer_c else []),
                *(
                    ["reviewer_c_medium_or_low_confidence"]
                    if review["confidence"] != "high"
                    else []
                ),
            }
        )
        result.append(
            {
                **item,
                "reviewer_c_label": reviewer_c,
                "reviewer_c_confidence": review["confidence"],
                "reviewer_c_partial_subtype": review.get("partial_subtype"),
                "audit_flags": audit_flags,
                "tricky_case": bool(
                    automated != reviewer_c
                    or automated == "partial"
                    or reviewer_c == "partial"
                    or review["confidence"] != "high"
                ),
            }
        )
    result.sort(key=lambda item: _rank(str(item["annotation_id"]), "blind_order"))
    return result


def build(queue: dict[str, Any], reviewer_c: dict[str, Any], size: int) -> dict[str, Any]:
    if queue.get("schema_version") != QUEUE_SCHEMA:
        raise ValueError("unsupported response-review queue schema")
    if queue.get("queue_sha256") != canonical_sha256(queue.get("items", [])):
        raise ValueError("invalid source queue hash")
    if reviewer_c.get("schema_version") != PROGRESS_SCHEMA:
        raise ValueError("unsupported Reviewer C progress schema")
    if reviewer_c.get("queue_sha256") != queue["queue_sha256"]:
        raise ValueError("Reviewer C is bound to another queue")
    if reviewer_c.get("reviewer_kind") != "artificial_reviewer":
        raise ValueError("Reviewer C must be explicitly disclosed as artificial")

    items = select_sample(queue["items"], reviewer_c.get("reviews", {}), size)
    counts = {
        "total": len(items),
        "language": dict(sorted(Counter(item["language"] for item in items).items())),
        "category": dict(sorted(Counter(item["category"] for item in items).items())),
        "category_language": dict(
            sorted(Counter(f"{item['language']}:{item['category']}" for item in items).items())
        ),
        "automated_label": dict(sorted(Counter(item["automated_label"] for item in items).items())),
        "reviewer_c_label": dict(
            sorted(Counter(item["reviewer_c_label"] for item in items).items())
        ),
        "reviewer_c_disagreements": sum(
            item["automated_label"] != item["reviewer_c_label"] for item in items
        ),
        "partial_union": sum(
            item["automated_label"] == "partial" or item["reviewer_c_label"] == "partial"
            for item in items
        ),
        "tricky_cases": sum(bool(item["tricky_case"]) for item in items),
    }
    result = {
        "schema_version": QUEUE_SCHEMA,
        "rubric_version": queue["rubric_version"],
        "sample_design": "targeted_error_audit_not_probability_sample",
        "sample_seed": SAMPLE_SEED,
        "created_at": datetime.now(UTC).isoformat(),
        "created_from_queue_sha256": queue["queue_sha256"],
        "reviewer_c_disclosure": reviewer_c["model_disclosure"],
        "counts": counts,
        "items": items,
    }
    result["queue_sha256"] = canonical_sha256(items)
    return result


def _write_private(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.chmod(0o600)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--reviewer-c", type=Path, default=DEFAULT_REVIEWER_C)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--size", type=int, default=SAMPLE_SIZE)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.out.exists() and not args.force:
        parser.error(f"refusing to replace frozen sample without --force: {args.out}")
    value = build(_load(args.queue), _load(args.reviewer_c), args.size)
    _write_private(args.out, value)
    summary = {"path": str(args.out), "queue_sha256": value["queue_sha256"], **value["counts"]}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

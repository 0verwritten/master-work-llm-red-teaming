"""Analyse completed blind human reviews against the frozen automated judge."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, cast

from apps.harmbench_response_review.workspace import (
    LABELS,
    PROGRESS_SCHEMA,
    QUEUE_SCHEMA,
    canonical_sha256,
)
from harness.analysis.full_harmbench import wilson_interval
from harness.analysis.judge_concordance import analyse


def _load(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text()))


def _progress(queue: dict[str, Any], path: Path) -> dict[str, Any]:
    value = _load(path)
    if value.get("schema_version") != PROGRESS_SCHEMA:
        raise ValueError(f"unsupported progress schema: {path}")
    if value.get("queue_sha256") != queue["queue_sha256"]:
        raise ValueError(f"progress is bound to another queue: {path}")
    unknown = set(value.get("reviews", {})) - {
        str(item["annotation_id"]) for item in queue["items"]
    }
    if unknown:
        raise ValueError(f"progress contains unknown annotation IDs: {path}")
    return value


def _matrix(rows: list[tuple[str, str]]) -> list[list[int]]:
    return [[sum(a == left and b == right for a, b in rows) for right in LABELS] for left in LABELS]


def _agreement_report(matrix: list[list[int]]) -> dict[str, Any]:
    try:
        return analyse(matrix).as_json()
    except ValueError as exc:
        total = sum(sum(row) for row in matrix)
        observed = sum(matrix[index][index] for index in range(len(matrix))) / total
        rows = [sum(row) for row in matrix]
        columns = [
            sum(matrix[row][column] for row in range(len(matrix))) for column in range(len(matrix))
        ]
        expected = sum(rows[index] * columns[index] for index in range(len(matrix))) / (
            total * total
        )
        return {
            "total": total,
            "observed_agreement": observed,
            "expected_agreement": expected,
            "kappa": None,
            "kappa_ci_low": None,
            "kappa_ci_high": None,
            "undefined_reason": str(exc),
            "row_marginals": dict(zip(LABELS, rows, strict=True)),
            "column_marginals": dict(zip(LABELS, columns, strict=True)),
        }


def _binary_metrics(rows: list[tuple[str, str]], positives: set[str]) -> dict[str, Any]:
    tp = sum(auto in positives and human in positives for auto, human in rows)
    fp = sum(auto in positives and human not in positives for auto, human in rows)
    fn = sum(auto not in positives and human in positives for auto, human in rows)
    tn = sum(auto not in positives and human not in positives for auto, human in rows)

    def measure(numerator: int, denominator: int) -> dict[str, Any] | None:
        if denominator == 0:
            return None
        low, high = wilson_interval(numerator, denominator)
        return {
            "value": numerator / denominator,
            "numerator": numerator,
            "denominator": denominator,
            "wilson_95": [low, high],
        }

    return {
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "precision": measure(tp, tp + fp),
        "recall": measure(tp, tp + fn),
        "specificity": measure(tn, tn + fp),
    }


def _comparison_strata(
    queue: dict[str, Any], final_labels: dict[str, str], source_field: str
) -> dict[str, Any]:
    ids = [str(item["annotation_id"]) for item in queue["items"]]
    by_id = {str(item["annotation_id"]): item for item in queue["items"]}
    strata: dict[str, Any] = {}
    for language in ("en", "uk", "all"):
        selected_ids = [
            identifier
            for identifier in ids
            if identifier in final_labels
            and source_field in by_id[identifier]
            and (language == "all" or by_id[identifier]["language"] == language)
        ]
        pairs = [
            (str(by_id[identifier][source_field]), final_labels[identifier])
            for identifier in selected_ids
        ]
        if not pairs:
            continue
        matrix = _matrix(pairs)
        strata[language] = {
            "n": len(pairs),
            "source_by_human_confusion": matrix,
            "three_level_agreement": _agreement_report(matrix),
            "strict_complied_positive": _binary_metrics(pairs, {"complied"}),
            "non_refusal_positive": _binary_metrics(pairs, {"partial", "complied"}),
            "human_labels": dict(
                sorted(Counter(final_labels[value] for value in selected_ids).items())
            ),
        }
    return strata


def analyse_single_review(queue: dict[str, Any], progress: dict[str, Any]) -> dict[str, Any]:
    ids = [str(item["annotation_id"]) for item in queue["items"]]
    missing = sorted(set(ids) - set(progress["reviews"]))
    if missing:
        return {
            "status": "incomplete",
            "queue_total": len(ids),
            "completed": {progress["reviewer_id"]: len(progress["reviews"])},
            "remaining": {progress["reviewer_id"]: len(missing)},
        }
    final_labels = {identifier: str(progress["reviews"][identifier]["label"]) for identifier in ids}
    subtype_counts = Counter(
        review["partial_subtype"]
        for review in progress["reviews"].values()
        if review["label"] == "partial" and review.get("partial_subtype")
    )
    result: dict[str, Any] = {
        "status": "complete",
        "sample_design": queue.get("sample_design", "unspecified"),
        "queue_sha256": queue["queue_sha256"],
        "queue_total": len(ids),
        "reviewers": [progress["reviewer_id"]],
        "partial_subtypes": dict(sorted(subtype_counts.items())),
        "automated_judge_vs_human": _comparison_strata(queue, final_labels, "automated_label"),
    }
    if all("reviewer_c_label" in item for item in queue["items"]):
        result["artificial_reviewer_c_vs_human"] = _comparison_strata(
            queue, final_labels, "reviewer_c_label"
        )
    return result


def analyse_reviews(
    queue: dict[str, Any],
    first: dict[str, Any],
    second: dict[str, Any],
    adjudication: dict[str, Any] | None,
) -> dict[str, Any]:
    if first["reviewer_id"] == second["reviewer_id"]:
        raise ValueError("the two independent reviews must use distinct reviewer IDs")
    ids = [str(item["annotation_id"]) for item in queue["items"]]
    missing = {
        first["reviewer_id"]: sorted(set(ids) - set(first["reviews"])),
        second["reviewer_id"]: sorted(set(ids) - set(second["reviews"])),
    }
    if any(missing.values()):
        return {
            "status": "incomplete",
            "queue_total": len(ids),
            "completed": {
                first["reviewer_id"]: len(first["reviews"]),
                second["reviewer_id"]: len(second["reviews"]),
            },
            "remaining": {key: len(value) for key, value in missing.items()},
        }

    human_pairs = [
        (first["reviews"][identifier]["label"], second["reviews"][identifier]["label"])
        for identifier in ids
    ]
    human_agreement = _agreement_report(_matrix(human_pairs))
    disagreements = [
        identifier
        for identifier, (left, right) in zip(ids, human_pairs, strict=True)
        if left != right
    ]
    adjudicated = adjudication.get("reviews", {}) if adjudication else {}
    final_labels: dict[str, str] = {}
    for identifier, (left, right) in zip(ids, human_pairs, strict=True):
        if left == right:
            final_labels[identifier] = left
        elif identifier in adjudicated:
            final_labels[identifier] = str(adjudicated[identifier]["label"])

    subtype_counts: Counter[str] = Counter()
    for progress in (first, second):
        subtype_counts.update(
            review["partial_subtype"]
            for review in progress["reviews"].values()
            if review["label"] == "partial" and review.get("partial_subtype")
        )
    return {
        "status": "complete" if len(final_labels) == len(ids) else "awaiting_adjudication",
        "queue_sha256": queue["queue_sha256"],
        "queue_total": len(ids),
        "reviewers": [first["reviewer_id"], second["reviewer_id"]],
        "human_inter_rater": human_agreement,
        "human_disagreements": len(disagreements),
        "adjudicated_disagreements": sum(value in adjudicated for value in disagreements),
        "unresolved_disagreements": len(ids) - len(final_labels),
        "partial_subtypes_across_independent_reviews": dict(sorted(subtype_counts.items())),
        "strata": _comparison_strata(queue, final_labels, "automated_label"),
    }


def write_adjudication_queue(
    queue: dict[str, Any],
    result: dict[str, Any],
    first: dict[str, Any],
    second: dict[str, Any],
    path: Path,
) -> None:
    disagreements = {
        identifier
        for identifier in first["reviews"]
        if first["reviews"][identifier]["label"] != second["reviews"][identifier]["label"]
    }
    items = []
    for item in queue["items"]:
        identifier = str(item["annotation_id"])
        if identifier in disagreements:
            items.append(
                {
                    **item,
                    "independent_human_labels": [
                        first["reviews"][identifier]["label"],
                        second["reviews"][identifier]["label"],
                    ],
                }
            )
    adjudication = {
        "schema_version": QUEUE_SCHEMA,
        "rubric_version": queue["rubric_version"],
        "created_from_queue_sha256": queue["queue_sha256"],
        "created_from_status": result["status"],
        "items": items,
    }
    adjudication["queue_sha256"] = canonical_sha256(items)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(adjudication, ensure_ascii=False, indent=2) + "\n")
    temporary.chmod(0o600)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--review", type=Path, action="append", required=True)
    parser.add_argument("--adjudication-queue", type=Path)
    parser.add_argument("--adjudication", type=Path)
    parser.add_argument("--adjudication-out", type=Path)
    args = parser.parse_args()
    if len(args.review) not in {1, 2}:
        parser.error("pass one or two independent --review files")
    queue = _load(args.queue)
    if queue.get("schema_version") != QUEUE_SCHEMA or queue.get("queue_sha256") != canonical_sha256(
        queue.get("items", [])
    ):
        raise ValueError("invalid response-review queue")
    reviews = [_progress(queue, path) for path in args.review]
    if len(reviews) == 1:
        if args.adjudication or args.adjudication_queue or args.adjudication_out:
            parser.error("adjudication requires two independent reviews")
        print(json.dumps(analyse_single_review(queue, reviews[0]), ensure_ascii=False, indent=2))
        return
    first, second = reviews
    adjudication = None
    if args.adjudication or args.adjudication_queue:
        if not args.adjudication or not args.adjudication_queue:
            parser.error("pass --adjudication-queue and --adjudication together")
        adjudication_queue = _load(args.adjudication_queue)
        if adjudication_queue.get("created_from_queue_sha256") != queue["queue_sha256"]:
            raise ValueError("adjudication queue is not derived from the review queue")
        adjudication = _progress(adjudication_queue, args.adjudication)
    result = analyse_reviews(queue, first, second, adjudication)
    if args.adjudication_out and result["status"] != "incomplete":
        write_adjudication_queue(queue, result, first, second, args.adjudication_out)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

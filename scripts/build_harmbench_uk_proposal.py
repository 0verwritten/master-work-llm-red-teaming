"""Generate a private Ukrainian-HarmBench translation-review artifact.

Reads a frozen bilingual dataset, validates its selected identity and size,
applies any curated correction proposals, and writes the artifact into the
Git-ignored translation workspace. No provider, database, or network access is
used.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from apps.harmbench_uk_review.proposal import (
    EXPECTED_COUNT,
    SUBSET_PLAN_SHA256,
    build_proposal,
    write_json_0600,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(".translation-workspace/bilingual-dataset-v1.json"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(".translation-workspace/harmbench-uk-correction-proposal-v1.json"),
    )
    parser.add_argument("--expected-count", type=int, default=EXPECTED_COUNT)
    parser.add_argument(
        "--subset-plan-sha256",
        default=SUBSET_PLAN_SHA256,
        help="expected frozen collection identity",
    )
    args = parser.parse_args()
    if args.expected_count < 1:
        parser.error("--expected-count must be positive")
    proposal = build_proposal(
        args.dataset,
        expected_count=args.expected_count,
        expected_subset_plan_sha256=args.subset_plan_sha256,
    )
    write_json_0600(args.out, proposal)
    corrections = sum(
        1 for record in proposal["records"] if record["disposition"] == "propose_correction"
    )
    print(
        f"wrote {args.out} : {proposal['count']} records, {corrections} proposed "
        f"corrections, collection {proposal['collection_sha256'][:12]}"
    )


if __name__ == "__main__":
    main()

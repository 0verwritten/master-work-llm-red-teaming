"""Finalize the Ukrainian HarmBench human review and report the outcome.

Refuses unless all 120 items carry an explicit, hash-bound decision. Writes a
new reviewed-translation artifact (never overwrites the DeepL source) and prints
the provenance split. Regenerating the runnable bilingual dataset and freezing
the separate 100-item Ukrainian manifest are deliberately left as later,
explicitly gated steps per the handover.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from apps.harmbench_uk_review.workspace import ReviewWorkspace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--proposal",
        type=Path,
        default=Path(".translation-workspace/harmbench-uk-correction-proposal-v1.json"),
    )
    parser.add_argument(
        "--overlay",
        type=Path,
        default=Path(".translation-workspace/harmbench-uk-human-review-v2.json"),
    )
    parser.add_argument(
        "--legacy-overlay",
        type=Path,
        default=Path(".translation-workspace/harmbench-uk-human-review-v1.json"),
    )
    args = parser.parse_args()
    workspace = ReviewWorkspace(
        args.proposal, args.overlay, legacy_overlay_path=args.legacy_overlay
    )
    path = workspace.finalize()
    artifact = json.loads(path.read_text())
    print(f"wrote {path}")
    print(json.dumps(artifact["counts"], indent=2))
    print(f"collection {artifact['collection_sha256'][:12]}")


if __name__ == "__main__":
    main()

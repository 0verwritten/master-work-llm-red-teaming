#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root"

queue=".translation-workspace/harmbench-uk-second-review-queue-v1.json"
if [[ ! -f "$queue" ]]; then
  .venv/bin/python scripts/build_harmbench_uk_second_review.py --out "$queue"
fi

exec .venv/bin/python -m apps.harmbench_uk_second_review.main "$@"

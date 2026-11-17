#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 REVIEWER_PSEUDONYM [PORT] [QUEUE]" >&2
  exit 2
fi

reviewer_id="$1"
port="${2:-8766}"
queue="${3:-.translation-workspace/harmbench-response-review-queue-v1.json}"

if [[ ! -f "$queue" ]]; then
  .venv/bin/python scripts/build_harmbench_response_review.py --out "$queue"
fi

exec .venv/bin/python -m apps.harmbench_response_review.main \
  --queue "$queue" \
  --reviewer-id "$reviewer_id" \
  --port "$port"

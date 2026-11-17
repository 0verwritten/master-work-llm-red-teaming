#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root"

dataset=".translation-workspace/bilingual-dataset-v1-harmbench-full-510.json"
proposal=".translation-workspace/harmbench-uk-full-510-review-proposal-v1.json"
overlay=".translation-workspace/harmbench-uk-human-review-all-510-v1.json"
seed=".translation-workspace/harmbench-uk-reviewed-translations-v1.json"
output=".translation-workspace/harmbench-uk-reviewed-translations-all-510-v1.json"
subset_sha256="8d5419d8bd8fcba2d5177281f3488937ec6091dd05b9fa116a2cdb93ec11a2cd"

.venv/bin/python scripts/build_harmbench_uk_proposal.py \
  --dataset "$dataset" \
  --out "$proposal" \
  --expected-count 510 \
  --subset-plan-sha256 "$subset_sha256"

exec .venv/bin/python -m apps.harmbench_uk_review.main \
  --proposal "$proposal" \
  --overlay "$overlay" \
  --reviewed-seed "$seed" \
  --reviewed-output "$output" \
  "$@"

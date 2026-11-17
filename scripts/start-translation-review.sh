#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root"

./deploy/podman/postgres.sh start >/dev/null
for attempt in $(seq 1 20); do
  if .venv/bin/research-harness db status >/dev/null 2>&1; then
    break
  fi
  if [[ "$attempt" == "20" ]]; then
    echo "PostgreSQL did not become ready" >&2
    exit 1
  fi
  sleep 1
done

.venv/bin/alembic upgrade head
exec .venv/bin/python -m apps.translation_review.main "$@"

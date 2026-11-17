#!/usr/bin/env bash
set -euo pipefail

container_name="research-harness-postgres"
volume_name="research-harness-postgres-data"
env_file="${POSTGRES_ENV_FILE:-$(dirname "$0")/../../.env.postgres}"
image="${POSTGRES_IMAGE:-docker.io/library/postgres@sha256:af194ccf3e2d7fe367012c7b88ce8b816c5c889b18a5b316799a1f0d7eac746a}"
host_port="${POSTGRES_PORT:-5432}"

usage() {
  echo "usage: $0 {start|stop|status|logs}"
}

require_env_file() {
  if [[ ! -f "$env_file" ]]; then
    echo "missing $env_file; copy deploy/podman/postgres.env.example to .env.postgres" >&2
    exit 2
  fi
  chmod 600 "$env_file"
}

case "${1:-}" in
  start)
    require_env_file
    if podman container exists "$container_name"; then
      podman start "$container_name"
    else
      podman volume create "$volume_name" >/dev/null
      podman run --detach \
        --name "$container_name" \
        --env-file "$env_file" \
        --publish "127.0.0.1:${host_port}:5432" \
        --volume "${volume_name}:/var/lib/postgresql/data:Z" \
        --health-cmd "pg_isready -U \$POSTGRES_USER -d \$POSTGRES_DB" \
        --health-interval 5s \
        --health-timeout 3s \
        --health-retries 20 \
        --security-opt no-new-privileges \
        "$image"
    fi
    ;;
  stop)
    podman stop "$container_name"
    ;;
  status)
    podman inspect "$container_name" \
      --format '{{.State.Status}} health={{if .State.Health}}{{.State.Health.Status}}{{else}}n/a{{end}}'
    ;;
  logs)
    podman logs "$container_name"
    ;;
  *)
    usage
    exit 2
    ;;
esac

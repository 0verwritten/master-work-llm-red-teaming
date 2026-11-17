#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PY=.venv/bin/python
WS=.translation-workspace
LOGS=$WS/logs
mkdir -p "$LOGS"

SPEND_CAP_USD=${SPEND_CAP_USD:-25}
MAX_HOURS=${MAX_HOURS:-7}
MAX_H100_DPH=${MAX_H100_DPH:-3.60}
MAX_JUDGE_DPH=${MAX_JUDGE_DPH:-0.95}
TREATMENT_CONCURRENCY=${TREATMENT_CONCURRENCY:-10}
REPLICATION_CONCURRENCY=${REPLICATION_CONCURRENCY:-4}

LABEL=$($PY -c 'from harness.spend.vast import OWNER_LABEL; print(OWNER_LABEL)')
EVIDENCE=docs/adr/0037-test-attack-repetition-as-a-cause-of-ukrainian-attacker-failure.md
FULL_DATASET=$WS/bilingual-dataset-v1-harmbench-full-510.json
REVIEWED_DATASET=$WS/bilingual-dataset-v1-harmbench-reviewed.json
TREATMENT_CHECKPOINT=$WS/vast-harmbench-uk-510-reject-repeats-r2-checkpoint.jsonl
REPLICATION_CHECKPOINT=$WS/vast-harmbench-uk-120-replication-2026-09-checkpoint.jsonl
REPLICATION_NAMESPACE=vast-harmbench-uk-qwen-h100-gptoss-rtx6000ada-v1-replication-2026-09

log() { printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*"; }

log "preflight"
$PY - <<'EOF'
import asyncio
from sqlalchemy import text
from harness.persistence.database import create_engine
from harness.settings import Settings

settings = Settings()
if not settings.vast_vllm_api_key_value():
    raise SystemExit("VAST_VLLM_API_KEY is not set in .env; refusing to rent")

async def check() -> None:
    engine = create_engine(settings.database_url)
    async with engine.connect() as connection:
        version = (await connection.execute(text("select version_num from alembic_version"))).scalar()
    await engine.dispose()
    print(f"postgres ok, alembic {version}")

asyncio.run(check())
EOF
VLLM_API_KEY=$($PY -c 'from harness.settings import Settings; print(Settings().vast_vllm_api_key_value())')
for dataset in "$FULL_DATASET" "$REVIEWED_DATASET"; do
  [[ -f $dataset ]] || { log "missing $dataset"; exit 1; }
done
$PY -m harness.cli spend vast >/dev/null  # vastai CLI authenticated

plan() {
  $PY scripts/run_vast_harmbench.py "$@" | $PY -c 'import json,sys; d=json.loads(sys.stdin.read().split("\n\nPlan only")[0]); print(d["manifest_sha256"])'
}
TREATMENT_SHA=$(plan --dataset "$FULL_DATASET" --language uk --records 510 --allow-ai-reviewed --max-attack-regenerations 2)
[[ $TREATMENT_SHA == b4cb25499740d9bd860c7fe4fb95c636ab32d7233c34605c0963f5b6c40ba6f7 ]] \
  || { log "treatment manifest $TREATMENT_SHA does not match ADR-0037"; exit 1; }
log "treatment manifest matches ADR-0037"

cheapest_offer() {  # gpu_query max_dph
  vastai search offers "gpu_name=$1 num_gpus=1 verified=true rentable=true reliability>0.98 disk_space>=150 cuda_max_good>=13.0" -o dph --raw \
    | $PY -c '
import json, sys
limit = float(sys.argv[1])
offers = [o for o in json.load(sys.stdin) if o["dph_total"] <= limit]
if not offers:
    raise SystemExit("no offer under the price limit")
print(offers[0]["id"], offers[0]["dph_total"])' "$2"
}

create_instance() {  # offer_id profile
  vastai create instance "$1" \
    --image "$($PY scripts/vast_serve_command.py "$2" image)" \
    --disk 150 --ssh --direct \
    --label "$LABEL" \
    --env "-e VLLM_API_KEY=$VLLM_API_KEY" \
    --onstart-cmd "$($PY scripts/vast_serve_command.py "$2" onstart)" \
    --raw | $PY -c 'import json,sys; print(json.load(sys.stdin)["new_contract"])'
}

destroy_owned_now() {  # instance ids...; label re-verified inside finish_arm
  local ids; ids=$(IFS=,; echo "$*")
  $PY - "$ids" <<'EOF'
import sys
from pathlib import Path
sys.path.insert(0, "scripts")
from watch_vast_batch import Arm, finish_arm
ids = tuple(int(value) for value in sys.argv[1].split(","))
finish_arm(
    Arm(name="uk-adr0037-abort", pid=1, instance_ids=ids),
    repository_root=Path.cwd(),
    reason="launch_abort",
    purpose="ADR-0037 Ukrainian launch aborted before experiment",
    evidence="docs/adr/0037-test-attack-repetition-as-a-cause-of-ukrainian-attacker-failure.md",
)
EOF
}

read -r H100_OFFER H100_DPH < <(cheapest_offer H100_SXM "$MAX_H100_DPH")
read -r JUDGE_OFFER JUDGE_DPH < <(cheapest_offer RTX_6000Ada "$MAX_JUDGE_DPH")
[[ -n ${H100_OFFER:-} && -n ${JUDGE_OFFER:-} ]] || { log "no suitable offer under the price limits; nothing rented"; exit 1; }
log "offers: H100 $H100_OFFER at $H100_DPH/h, judge $JUDGE_OFFER at $JUDGE_DPH/h"

H100_ID=$(create_instance "$H100_OFFER" qwen-h100)
log "created H100 instance $H100_ID"
if ! JUDGE_ID=$(create_instance "$JUDGE_OFFER" gptoss-rtx6000ada); then
  log "judge creation failed; destroying H100 $H100_ID"
  destroy_owned_now "$H100_ID"
  exit 1
fi
log "created judge instance $JUDGE_ID"

HOURS=$($PY -c "print(min($MAX_HOURS, $SPEND_CAP_USD / ($H100_DPH + $JUDGE_DPH)))")
DEADLINE=$($PY -c "import time; print(time.time() + $HOURS * 3600)")
log "spend cap USD $SPEND_CAP_USD -> hard deadline in $HOURS h"

WATCHDOG=(
  "$PWD/$PY" scripts/watch_vast_batch.py
  --arm "uk-adr0037:$$:$H100_ID,$JUDGE_ID"
  --deadline-epoch "$DEADLINE"
  --repository-root "$PWD"
  --purpose "ADR-0037 Ukrainian repeat-rejection 510 + replication 120 ({arm})"
  --evidence "$EVIDENCE"
)
if command -v systemd-run >/dev/null && systemd-run --user --unit="vast-adr0037-watchdog-$$" \
     --setenv=PATH="$PATH" \
     --working-directory="$PWD" "${WATCHDOG[@]}" >/dev/null 2>&1; then
  log "watchdog: systemd user unit vast-adr0037-watchdog-$$"
else
  setsid nohup "${WATCHDOG[@]}" > "$LOGS/adr0037-watchdog.log" 2>&1 &
  log "watchdog: detached PID $!"
fi

wait_running() {
  for _ in $(seq 1 120); do
    status=$(vastai show instance "$1" --raw | $PY -c 'import json,sys; d=json.load(sys.stdin); print(d.get("actual_status"), d.get("label"))')
    [[ ${status#* } == "$LABEL" ]] || { log "instance $1 label mismatch: $status"; exit 1; }
    [[ ${status%% *} == running ]] && return 0
    sleep 15
  done
  log "instance $1 did not reach running"; exit 1
}
wait_running "$H100_ID"
wait_running "$JUDGE_ID"
log "both instances running"

SSH_KEY=${VAST_SSH_KEY:-$HOME/.ssh/id_vastai}
[[ -f $SSH_KEY && -f $SSH_KEY.pub ]] || { log "missing SSH key $SSH_KEY"; exit 1; }
vastai attach ssh "$H100_ID" "$(cat "$SSH_KEY.pub")" >/dev/null
vastai attach ssh "$JUDGE_ID" "$(cat "$SSH_KEY.pub")" >/dev/null
log "attached $(ssh-keygen -lf "$SSH_KEY.pub" | awk '{print $2}') to both instances"

tunnel() {  # instance local_port
  local url host port
  url=$(vastai ssh-url "$1")
  host=${url#ssh://root@}; port=${host##*:}; host=${host%%:*}
  ( while true; do
      ssh -N -i "$SSH_KEY" -o IdentitiesOnly=yes \
          -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=4 \
          -o StrictHostKeyChecking=accept-new -p "$port" -L "$2:127.0.0.1:8000" "root@$host" || true
      sleep 5
    done ) > "$LOGS/adr0037-tunnel-$2.log" 2>&1 &
  echo $!
}
ssh_auth_ok() {  # instance; attached keys take a short while to propagate
  local url host port
  url=$(vastai ssh-url "$1")
  host=${url#ssh://root@}; port=${host##*:}; host=${host%%:*}
  for _ in $(seq 1 12); do
    ssh -i "$SSH_KEY" -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=15 \
        -o StrictHostKeyChecking=accept-new -p "$port" "root@$host" true 2>/dev/null && return 0
    sleep 15
  done
  log "SSH authentication to instance $1 failed for 3 minutes; aborting"; exit 1
}
ssh_auth_ok "$H100_ID"
ssh_auth_ok "$JUDGE_ID"
log "SSH authentication ok on both instances"

TUNNEL_PIDS="$(tunnel "$H100_ID" 18000) $(tunnel "$JUDGE_ID" 18001)"
trap 'kill $TUNNEL_PIDS 2>/dev/null || true' EXIT

wait_healthy() {  # local_port
  for _ in $(seq 1 120); do
    curl -sf "http://127.0.0.1:$1/health" >/dev/null && break
    sleep 15
  done
  curl -sf "http://127.0.0.1:$1/health" >/dev/null || { log "port $1 never became healthy"; exit 1; }
  code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$1/v1/models")
  [[ $code == 401 ]] || { log "port $1 answered /v1/models with $code, expected 401 (API key not enforced)"; exit 1; }
}
wait_healthy 18000
wait_healthy 18001
log "both servers healthy and key-protected"

TREATMENT=(
  "$PY" scripts/run_vast_harmbench.py --execute
  --dataset "$FULL_DATASET" --language uk --records 510 --allow-ai-reviewed
  --max-attack-regenerations 2 --checkpoint "$TREATMENT_CHECKPOINT"
  --target-instance-id "$H100_ID" --judge-instance-id "$JUDGE_ID"
)
REPLICATION=(
  "$PY" scripts/run_vast_harmbench.py --execute
  --dataset "$REVIEWED_DATASET" --language uk --records 120
  --cell-namespace "$REPLICATION_NAMESPACE" --checkpoint "$REPLICATION_CHECKPOINT"
  --target-instance-id "$H100_ID" --judge-instance-id "$JUDGE_ID"
)

smoke_ok() {  # checkpoint
  $PY - "$1" <<'EOF'
import sys
from harness.experiments.checkpoint import CheckpointLog
records = CheckpointLog(__import__("pathlib").Path(sys.argv[1])).records()
if not records:
    raise SystemExit("smoke produced no checkpoint record")
last = records[-1]
print(f"smoke: status={last.status} stop={last.stop_reason} label={last.label} lang={last.judge_language_adherence}")
if last.status not in {"completed", "stopped"} or last.stop_reason == "role_request_error":
    raise SystemExit("smoke failed")
EOF
}

log "smoke: one unseen cell per arm"
"${TREATMENT[@]}" --max-cells 1 --cell-concurrency 1 > "$LOGS/adr0037-smoke-treatment.log" 2>&1
smoke_ok "$TREATMENT_CHECKPOINT"
"${REPLICATION[@]}" --max-cells 1 --cell-concurrency 1 > "$LOGS/adr0037-smoke-replication.log" 2>&1
smoke_ok "$REPLICATION_CHECKPOINT"

log "releasing both arms"
"${TREATMENT[@]}" --cell-concurrency "$TREATMENT_CONCURRENCY" > "$LOGS/adr0037-treatment.log" 2>&1 &
TREATMENT_PID=$!
"${REPLICATION[@]}" --cell-concurrency "$REPLICATION_CONCURRENCY" > "$LOGS/adr0037-replication.log" 2>&1 &
REPLICATION_PID=$!
wait "$REPLICATION_PID" || log "replication runner exited non-zero"
wait "$TREATMENT_PID" || log "treatment runner exited non-zero"

log "resume pass for transient failures"
"${REPLICATION[@]}" --cell-concurrency "$REPLICATION_CONCURRENCY" >> "$LOGS/adr0037-replication.log" 2>&1 || true
"${TREATMENT[@]}" --cell-concurrency "$TREATMENT_CONCURRENCY" >> "$LOGS/adr0037-treatment.log" 2>&1 || true

log "done; exiting so the watchdog records spend and destroys $H100_ID and $JUDGE_ID"

#!/usr/bin/env bash
# scripts/run_matrix_safe.sh — только Malysh, безопасные параметры после BSOD.
set -uo pipefail
cd "$(dirname "$0")/.."

PROMPTS="prompts/prompts_matrix_1k.jsonl"
MAX_TOKENS=2000
REPEATS=5
GRID=(2 4 6 10)
LOG="results/_matrix_safe.log"
mkdir -p results/matrix
exec > >(tee -a "$LOG") 2>&1

log() { echo "[$(date -u +%H:%M:%SZ)] $*"; }
stop_all() {
  pkill -f "gpu_monitor.py" 2>/dev/null || true
  pkill -f "vllm serve" 2>/dev/null || true
  for i in $(seq 1 30); do
    curl -sS -m 1 -o /dev/null http://127.0.0.1:8001/health 2>/dev/null || return 0
    sleep 1
  done
  return 1
}
wait_health() {
  local t=${1:-900}
  for i in $(seq 1 "$t"); do
    if curl -sS -m 2 -o /dev/null -w '%{http_code}' http://127.0.0.1:8001/health 2>/dev/null | grep -q 200; then
      log "healthy after ${i}s"; return 0
    fi
    sleep 1
  done
  log "ERROR: not healthy after ${t}s"; return 1
}

NAME="rusmind-malysh-mx"
HF="DICPlay/RusMind-Malysh-4.7"
MAX_SEQS=10
MAX_LEN=3072
MEM=0.70
RUNDIR="results/matrix/$NAME"
mkdir -p "$RUNDIR"

log "########## MODEL $NAME (max_num_seqs=$MAX_SEQS, max_model_len=$MAX_LEN, mem=$MEM) ##########"
stop_all || { log "ERROR: port busy"; exit 1; }

MODEL_HF_ID="$HF" SERVED_MODEL_NAME="$NAME" \
  MAX_MODEL_LEN="$MAX_LEN" MAX_NUM_SEQS="$MAX_SEQS" GPU_MEM_UTIL="$MEM" \
  HEALTH_TIMEOUT_S=900 bash scripts/launch_vllm_server.sh >/dev/null 2>&1 &
if ! wait_health 900; then log "SKIP $NAME: server failed"; exit 1; fi

for c in "${GRID[@]}"; do
  log "--- $NAME c=$c ---"
  mkdir -p "$RUNDIR/c$c"
  python scripts/gpu_monitor.py --run-id "$NAME-c$c" --concurrency "$c" \
    --out "$RUNDIR/gpu_c$c.csv" --duration-s 900 &
  local_mon=$!
  sleep 2
  python scripts/inference_manager.py --base-url http://127.0.0.1:8001/v1 \
    --model "$NAME" --prompts "$PROMPTS" --buckets mx \
    --concurrency "$c" --repeats "$REPEATS" --max-tokens "$MAX_TOKENS" \
    --run-id "$NAME" --out "$RUNDIR/c$c/per_request.csv" || log "WARN: c=$c failed"
  kill -INT "$local_mon" 2>/dev/null || true
  wait "$local_mon" 2>/dev/null || true
  python scripts/aggregate_gpu.py --timeseries "$RUNDIR/gpu_c$c.csv" \
    --gpu-mem-util "$MEM" --out "$RUNDIR/gpu_summary_c$c.json" 2>/dev/null || true

  PEAK=$(python -c "import json; print(json.load(open('$RUNDIR/gpu_summary_c$c.json'))['peak_memory_used_pct'])" 2>/dev/null || echo 0)
  log "c=$c peak_vram=${PEAK}%"
  if [ "${PEAK%.*}" -gt 95 ] 2>/dev/null; then
    log "STOP: peak VRAM ${PEAK}% > 95% — прекращаю, чтобы избежать BSOD"
    break
  fi
  sleep 20
 done

INPUTS=()
for c in "${GRID[@]}"; do [ -f "$RUNDIR/c$c/per_request.csv" ] && INPUTS+=("$RUNDIR/c$c/per_request.csv"); done
if [ ${#INPUTS[@]} -gt 0 ]; then
  python scripts/aggregate_results.py --inputs "${INPUTS[@]}" --model "$NAME" --out "$RUNDIR/aggregated.csv" || true
fi
stop_all || true
log "=== safe matrix done ==="

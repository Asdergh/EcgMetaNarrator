#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."

NAME="qwen25-3b-mx"
HF="Qwen/Qwen2.5-3B-Instruct-AWQ"
MAX_SEQS=12
MAX_LEN=3072
MEM=0.80
PROMPTS="prompts/prompts_matrix_1k.jsonl"
MAX_TOKENS=2000
REPEATS=5
GRID=(2 4 6 8 10 12)
RUNDIR="results/matrix/$NAME"
mkdir -p "$RUNDIR"
LOG="results/_3b_matrix.log"
exec > >(tee -a "$LOG") 2>&1

log() { echo "[$(date -u +%H:%M:%SZ)] $*"; }

log "=== 3B matrix start ==="
log "waiting for server on :8001 ..."
for i in $(seq 1 300); do
  if curl -sS -m 2 -o /dev/null -w '%{http_code}' http://127.0.0.1:8001/health 2>/dev/null | grep -q 200; then
    log "server healthy"; break
  fi
  sleep 1
  [ "$i" -eq 300 ] && { log "ERROR: server not ready"; exit 1; }
done

for c in "${GRID[@]}"; do
  log "--- $NAME c=$c ---"
  mkdir -p "$RUNDIR/c$c"
  python scripts/gpu_monitor.py --run-id "$NAME-c$c" --concurrency "$c" \
    --out "$RUNDIR/gpu_c$c.csv" --duration-s 900 &
  MON=$!
  sleep 2
  python scripts/inference_manager.py --base-url http://127.0.0.1:8001/v1 \
    --model "$NAME" --prompts "$PROMPTS" --buckets mx \
    --concurrency "$c" --repeats "$REPEATS" --max-tokens "$MAX_TOKENS" \
    --run-id "$NAME" --out "$RUNDIR/c$c/per_request.csv" || log "WARN: c=$c failed"
  kill -INT "$MON" 2>/dev/null; wait "$MON" 2>/dev/null || true
  python scripts/aggregate_gpu.py --timeseries "$RUNDIR/gpu_c$c.csv" \
    --gpu-mem-util "$MEM" --out "$RUNDIR/gpu_summary_c$c.json" 2>/dev/null || true
  PEAK=$(python -c "import json;print(json.load(open('$RUNDIR/gpu_summary_c$c.json'))['peak_memory_used_pct'])" 2>/dev/null || echo 0)
  log "c=$c peak_vram=${PEAK}%"
  if [ "${PEAK%.*}" -gt 95 ] 2>/dev/null; then
    log "STOP: peak VRAM ${PEAK}% > 95% — чтобы не повторить BSOD"
    break
  fi
  sleep 20
 done

INPUTS=()
for c in "${GRID[@]}"; do [ -f "$RUNDIR/c$c/per_request.csv" ] && INPUTS+=("$RUNDIR/c$c/per_request.csv"); done
[ ${#INPUTS[@]} -gt 0 ] && python scripts/aggregate_results.py --inputs "${INPUTS[@]}" --model "$NAME" --out "$RUNDIR/aggregated.csv"
log "=== 3B matrix done ==="

#!/usr/bin/env bash
# scripts/run_matrix.sh — матрица: 3 модели × 5 точек × промпт 1K / ответ 2K.
# GPU-монитор запускается автоматически на каждой точке.
set -uo pipefail

cd "$(dirname "$0")/.."

PROMPTS="prompts/prompts_matrix_1k.jsonl"
MAX_TOKENS=2000
REPEATS=10
GRID=(2 4 6 10 15)
LOG="results/_matrix_run.log"
mkdir -p results/matrix
exec > >(tee -a "$LOG") 2>&1

log() { echo "[$(date -u +%H:%M:%SZ)] $*"; }

stop_all() {
  pkill -f "gpu_monitor.py" 2>/dev/null || true
  pkill -f "vllm serve" 2>/dev/null || true
  for i in $(seq 1 30); do
    if ! curl -sS -m 1 -o /dev/null http://127.0.0.1:8001/health 2>/dev/null; then
      return 0
    fi
    sleep 1
  done
  return 1
}

wait_health() {
  local timeout=${1:-900}
  for i in $(seq 1 "$timeout"); do
    if curl -sS -m 2 -o /dev/null -w '%{http_code}' http://127.0.0.1:8001/health 2>/dev/null | grep -q 200; then
      log "healthy after ${i}s"; return 0
    fi
    sleep 1
  done
  log "ERROR: not healthy after ${timeout}s"; return 1
}

run_point() {
  local model="$1" rundir="$2" c="$3"
  log "--- $model c=$c ---"
  mkdir -p "$rundir/c$c"

  # Запускаем монитор параллельно; duration = 900s (с запасом)
  python scripts/gpu_monitor.py \
    --run-id "$model-c$c" --concurrency "$c" \
    --out "$rundir/gpu_c$c.csv" --duration-s 900 &
  local mon_pid=$!
  sleep 2  # дать монитору стартовать до нагрузки

  python scripts/inference_manager.py \
    --base-url http://127.0.0.1:8001/v1 \
    --model "$model" \
    --prompts "$PROMPTS" \
    --buckets mx \
    --concurrency "$c" \
    --repeats "$REPEATS" \
    --max-tokens "$MAX_TOKENS" \
    --run-id "$model" \
    --out "$rundir/c$c/per_request.csv" || log "WARN: c=$c failed for $model"

  kill -INT "$mon_pid" 2>/dev/null || true
  wait "$mon_pid" 2>/dev/null || true

  if [ -f "$rundir/gpu_c$c.csv" ]; then
    python scripts/aggregate_gpu.py \
      --timeseries "$rundir/gpu_c$c.csv" \
      --gpu-mem-util 0.85 \
      --out "$rundir/gpu_summary_c$c.json" 2>/dev/null || log "WARN: gpu agg failed c=$c"
  fi
}

run_model() {
  local name="$1" hf_id="$2" max_seqs="$3" max_len="$4" mem_util="$5"
  local rundir="results/matrix/$name"
  mkdir -p "$rundir"

  log "########## MODEL $name (max_num_seqs=$max_seqs, max_model_len=$max_len) ##########"
  stop_all || { log "ERROR: could not free port"; return 1; }

  MODEL_HF_ID="$hf_id" SERVED_MODEL_NAME="$name" \
    MAX_MODEL_LEN="$max_len" MAX_NUM_SEQS="$max_seqs" GPU_MEM_UTIL="$mem_util" \
    HEALTH_TIMEOUT_S=900 \
    bash scripts/launch_vllm_server.sh >/dev/null 2>&1 &

  if ! wait_health 900; then
    log "SKIP $name: server failed to start"
    echo "$name,FAILED_TO_START,,,,," >> results/matrix/matrix_summary.csv
    return 0
  fi

  for c in "${GRID[@]}"; do
    run_point "$name" "$rundir" "$c"
  done

  local inputs=()
  for c in "${GRID[@]}"; do
    [ -f "$rundir/c$c/per_request.csv" ] && inputs+=("$rundir/c$c/per_request.csv")
  done
  if [ ${#inputs[@]} -gt 0 ]; then
    python scripts/aggregate_results.py \
      --inputs "${inputs[@]}" --model "$name" --out "$rundir/aggregated.csv" \
      2>/dev/null || log "WARN: aggregate_results failed for $name"
  fi
}

log "=== matrix start $(date -u +%FT%TZ) ==="
run_model "qwen25-1.5b-mx" "Qwen/Qwen2.5-1.5B-Instruct-AWQ" 16 3072 0.85
run_model "qwen25-3b-mx"   "Qwen/Qwen2.5-3B-Instruct-AWQ"   12 3072 0.85
run_model "rusmind-malysh-mx" "DICPlay/RusMind-Malysh-4.7" 16 3072 0.85
stop_all || true
log "=== matrix done $(date -u +%FT%TZ) ==="

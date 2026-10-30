#!/usr/bin/env bash
# scripts/launch_vllm_server.sh — запуск vLLM-сервера (актуализировано под vLLM 0.32.0).
# Проверено на: WSL2, RTX 4060 8GB, Ubuntu 22.04, Python 3.12.
set -euo pipefail

# ---------- параметры прогона (переопределяются через env) ----------
MODEL_HF_ID="${MODEL_HF_ID:-Qwen/Qwen2.5-3B-Instruct-AWQ}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen25-3b-awq}"
PORT="${PORT:-8001}"          # 8000 занят docker-proxy на этой машине
HOST="${HOST:-0.0.0.0}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.75}"   # 8GB минус Xwayland (~3.7GB)
TP_SIZE="${TP_SIZE:-1}"
VLLM_VERSION="${VLLM_VERSION:-$(vllm --version 2>/dev/null || echo unknown)}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
HEALTH_TIMEOUT_S="${HEALTH_TIMEOUT_S:-180}"
LOG_DIR="${LOG_DIR:-results/_server_logs}"

# Обход отсутствия nvcc: без этого FlashInfer пытается JIT-собрать сэмплер и падает.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/vllm_server_${RUN_ID}.log"

log() { printf '%s | %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "${LOG_FILE}" >&2; }

log "=== run_config (server side) ==="
log "run_id=${RUN_ID}"
log "model=${MODEL_HF_ID} served_as=${SERVED_MODEL_NAME}"
log "vllm_version=${VLLM_VERSION}"
log "max_num_seqs=${MAX_NUM_SEQS} max_model_len=${MAX_MODEL_LEN}"
log "gpu_memory_utilization=${GPU_MEM_UTIL} tensor_parallel_size=${TP_SIZE}"
log "host=${HOST} port=${PORT}"
log "VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER}"

if [ -n "${HF_TOKEN:-}" ]; then
  log "HF_TOKEN: set"
else
  log "HF_TOKEN: unset -> anonymous HF download (public repo; ok)"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  log "--- nvidia-smi (hardware-поля для run_config.yaml) ---"
  nvidia-smi --query-gpu=name,memory.total,driver_version \
    --format=csv >>"${LOG_FILE}" 2>&1 || log "WARN: nvidia-smi query failed"
else
  log "WARN: nvidia-smi не найден"
fi

log "starting vLLM server ..."
vllm serve "${MODEL_HF_ID}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --enable-per-request-metrics \
  >>"${LOG_FILE}" 2>&1 &
SERVER_PID=$!
log "server pid=${SERVER_PID}"

HEALTH_URL="http://127.0.0.1:${PORT}/health"
READY=0
for i in $(seq 1 "${HEALTH_TIMEOUT_S}"); do
  if curl -fsS "${HEALTH_URL}" >/dev/null 2>&1; then
    log "server is healthy after ${i}s"
    READY=1
    break
  fi
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    log "ERROR: server process died; see ${LOG_FILE}"
    exit 1
  fi
  sleep 1
done

if [ "${READY}" -ne 1 ]; then
  log "ERROR: server not healthy after ${HEALTH_TIMEOUT_S}s; see ${LOG_FILE}"
  exit 1
fi

log "--- /v1/models ---"
curl -fsS "http://127.0.0.1:${PORT}/v1/models" | tee -a "${LOG_FILE}" >&2
log "ready; log file: ${LOG_FILE}"

wait "${SERVER_PID}"

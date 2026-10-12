#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-transnas}"
RUN_ID="${RUN_ID:-dual_gpu_smoke_$(date -u +%Y%m%dT%H%M%SZ)_$$}"
RUN_ROOT="${PROJECT_ROOT}/runs/parallel/${RUN_ID}"

mkdir -p "${PROJECT_ROOT}/runs/parallel"
if ! mkdir "${RUN_ROOT}" 2>/dev/null; then
  echo "Cannot create run directory: ${RUN_ROOT}" >&2
  exit 2
fi

echo "RUN_ROOT=${RUN_ROOT}"
status0=0
status1=0

for gpu in 0 1; do
  mkdir -p "${RUN_ROOT}/gpu${gpu}/results" "${RUN_ROOT}/gpu${gpu}/logs"
  (
    cd "${PROJECT_ROOT}"
    export CUDA_VISIBLE_DEVICES="${gpu}"
    conda run --no-capture-output -n "${CONDA_ENV}" \
      python scripts/run_training_smoke.py \
        --manifest data/processed/citb_formal_4tasks/data_manifest.json \
        --model-dir model/qwen3-0.6b \
        --output-dir "runs/parallel/${RUN_ID}/gpu${gpu}/results" \
        --steps 10 --seed "$((314159 + gpu))" \
      >"${RUN_ROOT}/gpu${gpu}/logs/training.log" 2>&1
  ) &
  eval "pid${gpu}=$!"
done

wait "${pid0}" || status0=$?
wait "${pid1}" || status1=$?
printf 'GPU 0 exit: %s\nGPU 1 exit: %s\n' "${status0}" "${status1}"
printf 'GPU 0 log: %s\nGPU 1 log: %s\n' "${RUN_ROOT}/gpu0/logs/training.log" "${RUN_ROOT}/gpu1/logs/training.log"
if [[ "${status0}" -ne 0 || "${status1}" -ne 0 ]]; then
  exit 1
fi
echo "DUAL_GPU_SMOKE=PASS"

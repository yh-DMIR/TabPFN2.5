#!/usr/bin/env bash
set -euo pipefail

mkdir -p /tmp/$USER/comgr
export TMPDIR=/tmp/$USER
export TEMP=/tmp/$USER
export TMP=/tmp/$USER
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TABPFN_DISABLE_TELEMETRY=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
unset HIP_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES
unset ROCR_VISIBLE_DEVICES
unset GPU_DEVICE_ORDINAL

PYTHON=${PYTHON:-python}
SCRIPT=${SCRIPT:-benchmark_tabpfn_classification_amd_skip.py}
ROOT=${ROOT:-.}
BENCHMARKS=${BENCHMARKS:-openml_cc18_csv=dataset/openml_cc18_72,tabarena_cls=dataset/tabarena/cls,tabzilla_csv=dataset/tabzilla35,talent_cls=dataset/talent_cls}
MODEL_PATH=${MODEL_PATH:-ckpt/TabPFN-2.5/tabpfn-v2.5-classifier-v2.5_default.ckpt}
OUT_DIR=${OUT_DIR:-result/TabPFN_2_5_all_classification_8gpu}
WORKERS=${WORKERS:-8}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
SKIP_FILE=${SKIP_FILE:-skip.txt}

${PYTHON} ${SCRIPT} \
  --root "${ROOT}" \
  --benchmarks "${BENCHMARKS}" \
  --model-path "${MODEL_PATH}" \
  --out-dir "${OUT_DIR}" \
  --skip-file "${SKIP_FILE}" \
  --workers "${WORKERS}" \
  --gpus "${GPUS}" \
  --n-estimators 8 \
  --fit-mode fit_preprocessors \
  --inference-precision auto \
  --memory-saving-mode auto \
  --ignore-pretraining-limits \
  --softmax-temperature 0.9 \
  --verbose

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
SCRIPT=${SCRIPT:-test.py}
ROOT=${ROOT:-.}
BENCHMARKS=${BENCHMARKS:-test=dataset/test}
#BENCHMARKS=${BENCHMARKS:-tabarena_cls=dataset/tabarena/cls,talent_binclass=dataset/talent_cls/binclass,talent_multiclass=dataset/talent_cls/multiclass}
MODEL_PATH=${MODEL_PATH:-ckpt/TabPFN-2.5/tabpfn-v2.5-classifier-v2.5_default.ckpt}
OUT_DIR=${OUT_DIR:-result/TabPFN_2_5_official_classification_8gpu}
WORKERS=${WORKERS:-8}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}

${PYTHON} ${SCRIPT} \
  --root "${ROOT}" \
  --benchmarks "${BENCHMARKS}" \
  --model-path "${MODEL_PATH}" \
  --out-dir "${OUT_DIR}" \
  --workers "${WORKERS}" \
  --gpus "${GPUS}" \
  --n-estimators 8 \
  --fit-mode fit_preprocessors \
  --inference-precision auto \
  --memory-saving-mode auto \
  --ignore-pretraining-limits \
  --softmax-temperature 0.9 \
  --verbose

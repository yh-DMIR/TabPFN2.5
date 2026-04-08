#!/usr/bin/env bash
set -euo pipefail

mkdir -p /tmp/$USER/comgr
export TMPDIR=/tmp/$USER
export TEMP=/tmp/$USER
export TMP=/tmp/$USER
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TABPFN_DISABLE_TELEMETRY=1

PYTHON=${PYTHON:-python}
SCRIPT=${SCRIPT:-benchmark_tabpfn_classification_amd.py}
ROOT=${ROOT:-.}
BENCHMARKS=${BENCHMARKS:-tabarena_cls=dataset/tabarena/cls,talent_binclass=dataset/talent_cls/binclass,talent_multiclass=dataset/talent_cls/multiclass}
MODEL_PATH=${MODEL_PATH:-ckpt/TabPFN-2.5/tabpfn-v2-classifier.ckpt}
OUT_DIR=${OUT_DIR:-result/TabPFN_v2_official_classification_8gpu}
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

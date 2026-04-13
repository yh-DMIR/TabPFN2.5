#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python}
SCRIPT=${SCRIPT:-aggregate_classification_workers.py}
RESULT_DIR=${RESULT_DIR:-result/TabPFN_2_5_all_classification_8gpu}
WORKER_GLOB=${WORKER_GLOB:-worker_*.csv}

${PYTHON} ${SCRIPT} \
  --result-dir "${RESULT_DIR}" \
  --worker-glob "${WORKER_GLOB}"

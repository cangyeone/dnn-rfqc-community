#!/usr/bin/env bash
# Portable launcher. Install with install.sh, or select an installed Python.
set -euo pipefail
RFQC_APP_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${RFQC_PYTHON:-}" ]]; then
  RFQC_RUN_PYTHON="$RFQC_PYTHON"
elif [[ -x "$RFQC_APP_ROOT/.venv/bin/python" ]]; then
  RFQC_RUN_PYTHON="$RFQC_APP_ROOT/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  RFQC_RUN_PYTHON=python3
else
  RFQC_RUN_PYTHON=python
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
exec "$RFQC_RUN_PYTHON" -m rfqc_bench.cli screen-eqr "$@"

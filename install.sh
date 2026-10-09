#!/usr/bin/env bash
# Per-user installation in this checkout; no sudo or fixed home directory.
set -euo pipefail
RFQC_APP_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RFQC_SETUP_PYTHON="${RFQC_PYTHON:-python3}"
"$RFQC_SETUP_PYTHON" -c 'import sys; assert sys.version_info >= (3,10), "Python 3.10 or newer is required"'
"$RFQC_SETUP_PYTHON" -m venv "$RFQC_APP_ROOT/.venv"
RFQC_RUN_PYTHON="$RFQC_APP_ROOT/.venv/bin/python"
"$RFQC_RUN_PYTHON" -m pip install --upgrade pip
# Optional official CPU/CUDA wheel index; normal PyTorch defaults otherwise.
if [[ -n "${RFQC_TORCH_INDEX:-}" ]]; then
  "$RFQC_RUN_PYTHON" -m pip install 'torch>=2.8,<3' --index-url "$RFQC_TORCH_INDEX"
fi
"$RFQC_RUN_PYTHON" -m pip install --upgrade "$RFQC_APP_ROOT[api]"
"$RFQC_RUN_PYTHON" -m rfqc_bench.cli doctor
printf '%s\n' "Installed. Run: bash \"$RFQC_APP_ROOT/screen_eqr.sh\" \"/path/to/eqr\" --threshold 0.7 --resample"

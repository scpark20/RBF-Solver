#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1
RF_PYTHON="${PYTHON_BIN:-python}"
"$RF_PYTHON" "$PWD/rbf_pipeline.py" --cfg 1 --stage target
"$RF_PYTHON" "$PWD/rbf_pipeline.py" --cfg 1 --stage fit
"$RF_PYTHON" "$PWD/plot_rbf_coefficients.py" --cfg 1
exec "$RF_PYTHON" "$PWD/launch.py"

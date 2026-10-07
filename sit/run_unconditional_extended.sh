#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export SIT_SAMPLING_PLAN=unconditional_extended_sampling.json
export SIT_TRAINING_PLAN=unconditional_extended_training.json
export PYTHONDONTWRITEBYTECODE=1
exec "${PYTHON_BIN:-python}" rbf_followup.py --run

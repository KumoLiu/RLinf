#!/usr/bin/env bash
# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE_ROOT="$(dirname "${REPO_ROOT}")"
export COSMOS3_ROOT="${COSMOS3_ROOT:?Set COSMOS3_ROOT to your Cosmos source checkout}"
export GR00T_ROOT="${GR00T_ROOT:-${WORKSPACE_ROOT}/Isaac-GR00T}"
export EMBODIED_PATH="${SCRIPT_DIR}"
export PYTHONPATH="${REPO_ROOT}:${GR00T_ROOT}:${COSMOS3_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1

cd "${REPO_ROOT}"
ENTRYPOINT="${SCRIPT_DIR}/train_embodied_agent.py"
if [[ "${1:-}" == "--eval" ]]; then
    shift
    ENTRYPOINT="${REPO_ROOT}/evaluations/eval_embodied_agent.py"
    set -- runner.only_eval=true "$@"
fi
exec "${RLINF_PYTHON:-python}" -u "${ENTRYPOINT}" \
    --config-path "${SCRIPT_DIR}/config" \
    --config-name cosmos3_medical_pads_grpo_gr00t_n1d7 "$@"

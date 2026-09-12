#!/usr/bin/env bash
#
# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"

# Source a site-specific file explicitly; never auto-load files from the cwd.
# This file is optional on the current host (sibling repos/data/models).
if [[ -n "${DREAMDOJO_SITE_ENV:-}" ]]; then
    source "${DREAMDOJO_SITE_ENV}"
fi

export DREAMDOJO_ROOT="${DREAMDOJO_ROOT:-${WORKSPACE_ROOT}/DreamDojo}"
export DREAMDOJO_DATA_ROOT="${DREAMDOJO_DATA_ROOT:-${WORKSPACE_ROOT}/data}"
export GR00T_MODEL_PATH="${GR00T_MODEL_PATH:-${WORKSPACE_ROOT}/models/s2r_models_dev/yunl/sft/g1/pick_trocar/g1_pick_trocar_head_10k_bs32_lr1e-4}"
export DREAMDOJO_WM_CHECKPOINT="${DREAMDOJO_WM_CHECKPOINT:-${WORKSPACE_ROOT}/models/dreamdojo/lora_r32_lr3e-4_r64_18k/checkpoints/iter_000018000/model_ema_bf16.pt}"

# NCCL P2P hangs on the current H200 NVL host. Shared-memory transport passed
# the previously validated collective/GRPO checks. GPU placement is in YAML.
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# GR00T, DreamDojo and TensorFlow are imported in every actor/rollout/env
# process. Limit host thread pools to avoid oversubscribing the 224 CPU cores.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TF_NUM_INTRAOP_THREADS="${TF_NUM_INTRAOP_THREADS:-1}"
export TF_NUM_INTEROP_THREADS="${TF_NUM_INTEROP_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export NO_ALBUMENTATIONS_UPDATE="${NO_ALBUMENTATIONS_UPDATE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTHONDONTWRITEBYTECODE=1

# Respect an explicitly selected interpreter or an activated environment.
if [[ -n "${RLINF_PYTHON:-}" ]]; then
    PYTHON_BIN="${RLINF_PYTHON}"
elif [[ -n "${VIRTUAL_ENV:-}" || -n "${CONDA_PREFIX:-}" ]]; then
    PYTHON_BIN="$(command -v python)"
elif [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
else
    PYTHON_BIN="$(command -v python)"
fi
PYTHON_BIN="$(command -v "${PYTHON_BIN}")"
export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"

export EMBODIED_PATH="${SCRIPT_DIR}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# Standard RLinf runner: no external eval supervisor or YAML overlay.
cd "${REPO_ROOT}"
exec "${PYTHON_BIN}" -u "${SCRIPT_DIR}/train_embodied_agent.py" \
    --config-name dreamdojo_trocar_grpo_gr00t_n1d7 "$@"

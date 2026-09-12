#!/usr/bin/env bash
#SBATCH --account=healthcareeng_isaac
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=8
#SBATCH --time=00:30:00
#SBATCH --job-name=rlinf-dd-smoke
#SBATCH --output=rlinf-dd-%j.out
#SBATCH --error=rlinf-dd-%j.err
# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

# This entry never resubmits. train_chain.slurm optionally wraps native training.
MODE="${1:-smoke}"
if (( $# )); then shift; fi
BASE="${CLUSTER_BASE:-/lustre/fsw/portfolios/healthcareeng/users/yunl}"
IMAGE="${CONTAINER_IMAGE:-${BASE}/docker/rlinf-dreamdojo-gr00t2d9a9fc-cu128-v1.sqsh}"
ASSETS="${RLINF_ASSETS:-${BASE}/rlinf_assets/20260911}"
CHECKPOINTS="${RLINF_CHECKPOINT_ROOT:-${BASE}/checkpoints}"
OUTPUTS="${RLINF_OUTPUTS:-${BASE}/outputs/rlinf}"
HF_CACHE="${HF_CACHE_ROOT:-${BASE}/cache/huggingface}"
# These are host paths, mounted read-only at the SAME paths inside the container.
# Never copy existing checkpoints into an experiment-specific assets directory.
export GR00T_MODEL_PATH="${GR00T_MODEL_PATH:-${CHECKPOINTS}/gr00t_ft/g1_pick_trocar_head_10k_bs32_lr1e-4/976127}"
export DREAMDOJO_WM_CHECKPOINT="${DREAMDOJO_WM_CHECKPOINT:-${CHECKPOINTS}/DreamDojo/lora_r32_lr3e-4_r64_18k/checkpoints/iter_000018000/model_ema_bf16.pt}"
export DREAMDOJO_REWARD_CHECKPOINT="${DREAMDOJO_REWARD_CHECKPOINT:-${CHECKPOINTS}/reward/milestone_v2/best.pt}"
export DREAMDOJO_LAM_CHECKPOINT="${DREAMDOJO_LAM_CHECKPOINT:-${CHECKPOINTS}/DreamDojo/LAM_400k.ckpt}"
[[ -f "${IMAGE}" ]] || { echo "Missing image: ${IMAGE}" >&2; exit 1; }
mkdir -p "${OUTPUTS}"
MOUNTS="${OUTPUTS}:/outputs"
# Versioned, explicit source supplement for v1 images predating budget support.
# Only these two files are overlaid; never mount a mutable checkout wholesale.
if [[ -n "${RLINF_CHAIN_SOURCE_ROOT:-}" ]]; then
    for relative in rlinf/runners/embodied_runner.py rlinf/utils/train_budget.py; do
        [[ -f "${RLINF_CHAIN_SOURCE_ROOT}/${relative}" ]] || {
            echo "Missing chain runtime source: ${relative}" >&2; exit 1;
        }
        MOUNTS+=",${RLINF_CHAIN_SOURCE_ROOT}/${relative}:/opt/src/RLinf/${relative}:ro"
    done
    echo "RLinf chain source supplement (read-only): ${RLINF_CHAIN_SOURCE_ROOT}"
fi
# Explicit, read-only supplement for v1's missing external/lam source package.
# A rebuilt image includes it. The strict supplement changes checkpoint loading
# only; the G1 action conditioning and WM inference implementation stay unchanged.
if [[ -n "${DREAMDOJO_EXTERNAL_ROOT:-}" ]]; then
    [[ -f "${DREAMDOJO_EXTERNAL_ROOT}/lam/model.py" ]] || {
        echo "Missing DreamDojo external source: ${DREAMDOJO_EXTERNAL_ROOT}" >&2; exit 1;
    }
    MOUNTS+=",${DREAMDOJO_EXTERNAL_ROOT}:/opt/src/DreamDojo/external:ro"
    echo "DreamDojo external source supplement (read-only): ${DREAMDOJO_EXTERNAL_ROOT}"
fi
if [[ "${MODE}" != smoke ]]; then
    [[ -d "${ASSETS}/data" && -d "${CHECKPOINTS}" && -d "${HF_CACHE}/hub" ]] || {
        echo "Training/eval assets are not ready: ${ASSETS}/data, ${CHECKPOINTS}, ${HF_CACHE}/hub" >&2; exit 1;
    }
    for path in "${GR00T_MODEL_PATH}" "${DREAMDOJO_WM_CHECKPOINT}" "${DREAMDOJO_REWARD_CHECKPOINT}" "${DREAMDOJO_LAM_CHECKPOINT}"; do
        [[ "${path}" == "${CHECKPOINTS}/"* && -e "${path}" ]] || {
            echo "Missing checkpoint, or path outside mounted checkpoint root: ${path}" >&2; exit 1;
        }
    done
    [[ -s "${DREAMDOJO_LAM_CHECKPOINT}" ]] || {
        echo "Empty LAM checkpoint: ${DREAMDOJO_LAM_CHECKPOINT}" >&2; exit 1;
    }
    MOUNTS+=",${ASSETS}/data:/assets/data:ro,${CHECKPOINTS}:${CHECKPOINTS}:ro,${HF_CACHE}/hub:/cache/huggingface/hub:ro"
fi
if [[ "${MODE}" == verify ]]; then
    # Slurm runs a spool copy of this script, not its original checkout path.
    VERIFY_SCRIPT="${DREAMDOJO_VERIFY_SCRIPT:-${SLURM_SUBMIT_DIR:-$PWD}/docker/dreamdojo/verify_runtime.py}"
    [[ -f "${VERIFY_SCRIPT}" ]] || { echo "Missing verification script: ${VERIFY_SCRIPT}" >&2; exit 1; }
    MOUNTS+=",${VERIFY_SCRIPT}:/opt/rlinf-build/verify_runtime.py:ro"
fi

export RLINF_RUN_MODE="${MODE}"
export RLINF_RUN_LOG="/outputs/${SLURM_JOB_ID}-${MODE}"
export RLINF_PYTHON=/opt/venv/bin/python
export DREAMDOJO_DATA_ROOT=/assets/data
export HF_HOME=/cache/huggingface
# Use the GPUs assigned to this node; do not reuse the local-host GPU 4 exclusion.
export DREAMDOJO_GPUS="${DREAMDOJO_GPUS:-0-7}"
# Explicit 0 also prevents the v1 image's local train wrapper falling back to 1.
# Do not restrict NCCL_P2P_LEVEL: let NCCL select the cluster's normal transport.
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
CONTAINER_ENV="RLINF_RUN_MODE,RLINF_RUN_LOG,RLINF_PYTHON,DREAMDOJO_DATA_ROOT,GR00T_MODEL_PATH,DREAMDOJO_WM_CHECKPOINT,DREAMDOJO_REWARD_CHECKPOINT,DREAMDOJO_LAM_CHECKPOINT,HF_HOME,DREAMDOJO_GPUS,NCCL_P2P_DISABLE"
if [[ -n "${RLINF_TRAIN_DEADLINE:-}" ]]; then
    [[ "${MODE}" == train ]] || { echo "Chaining is train-only" >&2; exit 2; }
    CONTAINER_ENV+=",RLINF_TRAIN_DEADLINE,RLINF_CHAIN_DIR,RLINF_CHAIN_JOB_ID,RLINF_CHAIN_STEP_SECONDS,RLINF_CHAIN_EVAL_SECONDS,RLINF_CHAIN_SAVE_SECONDS"
fi

srun --container-image="${IMAGE}" --container-mounts="${MOUNTS}" \
    --container-workdir=/opt/src/RLinf --container-writable \
    --container-env="${CONTAINER_ENV}" \
    bash -c '
        set -euo pipefail
        export PYTHONPATH=/opt/src/RLinf:/opt/src/DreamDojo
        export DREAMDOJO_ROOT=/opt/src/DreamDojo
        export EMBODIED_PATH=/opt/src/RLinf/examples/embodiment
        echo "NCCL_P2P_DISABLE=$NCCL_P2P_DISABLE NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL:-unset}"
        if [[ "$RLINF_RUN_MODE" != smoke ]]; then
            # Require the strict loader, not an older supplement that skips LAM.
            /opt/venv/bin/python -c "from external.lam.model import DEFAULT_LAM_CHECKPOINT, resolve_lam_checkpoint; print(resolve_lam_checkpoint(DEFAULT_LAM_CHECKPOINT))"
        fi
        case "$RLINF_RUN_MODE" in
            smoke)
                exec /opt/venv/bin/python /opt/rlinf-build/smoke.py --gpu
                ;;
            verify)
                /opt/venv/bin/python /opt/rlinf-build/verify_runtime.py lam
                exec /opt/venv/bin/torchrun --standalone --nnodes=1 --nproc-per-node=8 \
                    /opt/rlinf-build/verify_runtime.py nccl
                ;;
            eval)
                exec /opt/venv/bin/python evaluations/eval_embodied_agent.py \
                    --config-path /opt/src/RLinf/examples/embodiment/config \
                    --config-name dreamdojo_trocar_grpo_gr00t_n1d7 \
                    runner.only_eval=true "rollout.model=\${actor.model}" \
                    "runner.logger.log_path=$RLINF_RUN_LOG" "$@"
                ;;
            train)
                exec bash examples/embodiment/run_dreamdojo_trocar.sh \
                    "runner.logger.log_path=$RLINF_RUN_LOG" \
                    env.train.total_num_envs=64 actor.global_batch_size=64 "$@"
                ;;
            *) echo "Unknown mode: $RLINF_RUN_MODE" >&2; exit 2 ;;
        esac
    ' bash "$@"

#!/usr/bin/env bash
# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
# One native recipe, explicit Hydra overrides. Dry-run unless --submit is given.
set -euo pipefail

MODE="${1:---dry-run}"
[[ $# -le 1 && ( "$MODE" == --dry-run || "$MODE" == --submit ) ]] || {
    echo "Usage: bash $0 [--dry-run|--submit]" >&2; exit 2;
}
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export REPO_ROOT="${REPO_ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"
source "${REPO_ROOT}/docs/dreamdojo/cluster.env.example"
BASE="${CLUSTER_BASE:-/lustre/fsw/portfolios/healthcareeng/users/yunl}"
export RLINF_CHAIN_SOURCE_ROOT="${BASE}/code/RLinf-runtime/20260911-chain-v1"
export CHAIN_TIMEOUT=3.9h MAX_RUNS=18
unset CHAIN_ID RUN_COUNT NCCL_P2P_LEVEL

# Lists may be narrowed for follow-up controlled comparisons.
read -r -a WMS <<< "${WM_VARIANTS:-r64 scratch_r32}"
read -r -a NOISES <<< "${NOISE_LEVELS:-0.1 0.3 0.5}"
read -r -a BATCHES <<< "${GLOBAL_BATCH_SIZES:-128}"
read -r -a LRS <<< "${ACTOR_LRS:-5e-6}"
# Replicate 0 preserves the original actor=1234 / train-env=0 pairing.
read -r -a SEEDS <<< "${SEED_IDS:-0}"
TRAIN_WM_STEPS="${TRAIN_WM_STEPS:-35}"
for wm in "${WMS[@]}"; do
    [[ "$wm" == r64 || "$wm" == scratch_r32 ]] || { echo "Invalid WM: $wm" >&2; exit 2; }
done
for noise in "${NOISES[@]}"; do
    [[ "$noise" == 0.1 || "$noise" == 0.2 || "$noise" == 0.3 || "$noise" == 0.5 ]] || { echo "Invalid noise: $noise" >&2; exit 2; }
done
for batch in "${BATCHES[@]}"; do
    [[ "$batch" == 128 || "$batch" == 256 || "$batch" == 512 ]] || { echo "Invalid global batch: $batch" >&2; exit 2; }
done
for lr in "${LRS[@]}"; do
    [[ "$lr" == 2.5e-6 || "$lr" == 5e-6 || "$lr" == 1e-5 ]] || { echo "Invalid actor LR: $lr" >&2; exit 2; }
done
for seed in "${SEEDS[@]}"; do
    [[ "$seed" =~ ^(0|[1-9][0-9]{0,3})$ ]] || { echo "Invalid seed ID: $seed" >&2; exit 2; }
done
[[ "$TRAIN_WM_STEPS" == 15 || "$TRAIN_WM_STEPS" == 35 ]] || { echo "Invalid train WM steps: $TRAIN_WM_STEPS" >&2; exit 2; }
wm_suffix=""
if [[ "$TRAIN_WM_STEPS" != 35 ]]; then wm_suffix="_wm${TRAIN_WM_STEPS}"; fi
SWEEP_TAG="${SWEEP_TAG:-20260911_wm_noise}"
[[ "$SWEEP_TAG" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "Invalid SWEEP_TAG" >&2; exit 2; }
SUBMISSIONS="${RLINF_OUTPUTS:-${BASE}/outputs/rlinf}/sweeps/${SWEEP_TAG}"

if [[ "$MODE" == --submit ]]; then
    command -v sbatch >/dev/null
    for path in \
        "${BASE}/docker/rlinf-dreamdojo-gr00t2d9a9fc-cu128-v1.sqsh" \
        "${RLINF_CHAIN_SOURCE_ROOT}/rlinf/runners/embodied_runner.py" \
        "${RLINF_CHAIN_SOURCE_ROOT}/rlinf/utils/train_budget.py" \
        "${DREAMDOJO_EXTERNAL_ROOT}/lam/model.py" \
        "${DREAMDOJO_LAM_CHECKPOINT}" "${DREAMDOJO_REWARD_CHECKPOINT}"; do
        [[ -s "$path" ]] || { echo "Missing asset: $path" >&2; exit 1; }
    done
    [[ -d "$GR00T_MODEL_PATH" ]] || { echo "Missing SFT: $GR00T_MODEL_PATH" >&2; exit 1; }
    for directory in lora_r32_lr3e-4_r64_18k lora_r32_scratch_lr3e-4_18k; do
        [[ -s "${RLINF_CHECKPOINT_ROOT}/DreamDojo/${directory}/checkpoints/iter_000018000/model_ema_bf16.pt" ]] || {
            echo "Missing WM: $directory" >&2; exit 1;
        }
    done
    mkdir -p "$SUBMISSIONS"
    exec 9>"${SUBMISSIONS}/submit.lock"
    flock -n 9 || { echo "Another submitter holds the sweep lock" >&2; exit 1; }
fi

for wm in "${WMS[@]}"; do
    if [[ "$wm" == r64 ]]; then
        directory=lora_r32_lr3e-4_r64_18k
        experiment=dreamdojo_2b_480_640_g1_hf_teleop_rollout_posttrain_lora_lr3e-4_r64
    else
        directory=lora_r32_scratch_lr3e-4_18k
        experiment=dreamdojo_2b_480_640_g1_hf_teleop_rollout_posttrain_lora
    fi
    export DREAMDOJO_WM_CHECKPOINT="${RLINF_CHECKPOINT_ROOT}/DreamDojo/${directory}/checkpoints/iter_000018000/model_ema_bf16.pt"
    for noise in "${NOISES[@]}"; do
        for batch in "${BATCHES[@]}"; do
          for lr in "${LRS[@]}"; do
           for seed in "${SEEDS[@]}"; do
            actor_seed=$((1234 + seed))
            lr_suffix=""
            if [[ "$lr" != 5e-6 ]]; then lr_suffix="_lr${lr/-/}"; fi
            lr_suffix="${lr_suffix/./p}"
            seed_suffix=""
            if [[ "$seed" != 0 ]]; then seed_suffix="-s${actor_seed}"; fi
            name="wm_${wm}_n${noise/./}_gb${batch}${lr_suffix}${wm_suffix}_s${actor_seed}"
            args=(
                --parsable --export=ALL --partition=batch --time=04:00:00
                "--job-name=dd-${wm}-n${noise/./}-b${batch}${lr_suffix}${wm_suffix}${seed_suffix}"
                "--chdir=${REPO_ROOT}"
                "--output=${SUBMISSIONS}/${name}-%j.out"
                "--error=${SUBMISSIONS}/${name}-%j.err"
                "${REPO_ROOT}/docker/dreamdojo/train_chain.slurm"
                "runner.logger.experiment_name=${name}"
                runner.max_epochs=1000 runner.max_steps=-1
                runner.resume_dir=null runner.ckpt_path=null
                runner.val_check_interval=5 runner.save_interval=5
                env.train.total_num_envs=64 env.train.rollout_epoch=2
                env.eval.total_num_envs=56 env.eval.eval_unique_episodes=true
                "env.train.num_inference_steps=${TRAIN_WM_STEPS}" env.eval.num_inference_steps=35
                "env.train.experiment_name=${experiment}" "env.eval.experiment_name=${experiment}"
                "actor.seed=${actor_seed}" "env.train.seed=${seed}" env.eval.seed=0
                "actor.global_batch_size=${batch}" actor.micro_batch_size=8 "actor.optim.lr=${lr}"
                "actor.model.rl_head_config.noise_level=${noise}"
                actor.model.rl_head_config.action_noise_scale=0.0
                actor.model.rl_head_config.noise_anneal=false
            )
            if [[ "$MODE" == --dry-run ]]; then
                printf 'MAX_RUNS=18 CHAIN_TIMEOUT=3.9h DREAMDOJO_WM_CHECKPOINT=%q sbatch ' "$DREAMDOJO_WM_CHECKPOINT"
                printf '%q ' "${args[@]}"
                printf '\n'
            elif [[ -e "${SUBMISSIONS}/${name}.jobid" ]]; then
                echo "Already submitted ${name}: $(<"${SUBMISSIONS}/${name}.jobid")"
            else
                job_id="$(sbatch "${args[@]}")"
                job_id="${job_id%%;*}"
                [[ "$job_id" =~ ^[0-9]+$ ]] || { echo "Unexpected sbatch response: $job_id" >&2; exit 1; }
                printf '%s\n' "$job_id" >"${SUBMISSIONS}/${name}.jobid"
                echo "Submitted ${name}: ${job_id} (MAX_RUNS=18, max_epochs=1000)"
            fi
           done
          done
        done
    done
done

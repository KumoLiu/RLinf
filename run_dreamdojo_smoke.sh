#!/usr/bin/env bash
# One-step RLinf interface smoke test on host GPUs 4-7.
set -euo pipefail
cd /localhome/local-yunl/RLinf

export CUDA_VISIBLE_DEVICES="${RLINF_SMOKE_CUDA_VISIBLE_DEVICES:-4,5,6,7}"
source /localhome/local-yunl/DreamDojo/env_local.sh
unset NVTE_PROJECT_BUILDING
export EMBODIED_PATH=/localhome/local-yunl/RLinf/examples/embodiment
export PYTHONPATH="/localhome/local-yunl/RLinf/.smoke-deps:\
/localhome/local-yunl/RLinf:\
/localhome/local-yunl/Isaac-GR00T:\
/localhome/local-yunl/DreamDojo${PYTHONPATH:+:$PYTHONPATH}"

LOG_DIR=/localhome/local-yunl/RLinf/logs/dreamdojo_smoke
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/run.log") 2>&1

/localhome/local-yunl/DreamDojo/.venv/bin/python \
  examples/embodiment/train_embodied_agent.py \
  --config-name dreamdojo_g1_pick_trocar_ppo_gr00t_n1d7 \
  '~cluster.component_placement' \
  '+cluster.component_placement={actor:0-2,rollout:0-2,env:3}' \
  runner.max_epochs=1 \
  runner.max_steps=1 \
  runner.val_check_interval=-1 \
  runner.save_interval=-1 \
  runner.logger.log_path="$LOG_DIR" \
  'runner.logger.logger_backends=[]' \
  env.train.total_num_envs=3 \
  env.train.max_episode_steps=25 \
  env.train.max_steps_per_rollout_epoch=25 \
  env.eval.total_num_envs=3 \
  env.eval.max_episode_steps=25 \
  env.eval.max_steps_per_rollout_epoch=25 \
  actor.global_batch_size=3 \
  actor.model.model_path=/localhome/local-yunl/Isaac-GR00T/g1_pick_trocar_200_finetune/head_30k

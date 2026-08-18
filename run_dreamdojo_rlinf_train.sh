#!/usr/bin/env bash
# Train the horizon-25 GR00T policy against the DreamDojo world model.
set -euo pipefail
cd /localhome/local-yunl/RLinf

source /localhome/local-yunl/DreamDojo/env_local.sh
unset NVTE_PROJECT_BUILDING
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export EMBODIED_PATH=/localhome/local-yunl/RLinf/examples/embodiment
export PYTHONPATH="/localhome/local-yunl/RLinf/.smoke-deps:\
/localhome/local-yunl/RLinf:\
/localhome/local-yunl/Isaac-GR00T:\
/localhome/local-yunl/DreamDojo${PYTHONPATH:+:$PYTHONPATH}"

LOG_DIR=/localhome/local-yunl/RLinf/logs/dreamdojo_g1_pick_trocar_rl
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/run.log") 2>&1

/localhome/local-yunl/DreamDojo/.venv/bin/python \
  examples/embodiment/train_embodied_agent.py \
  --config-name dreamdojo_g1_pick_trocar_ppo_gr00t_n1d7 \
  runner.logger.log_path="$LOG_DIR"

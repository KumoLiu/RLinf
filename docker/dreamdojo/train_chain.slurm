#!/usr/bin/env bash
#SBATCH --account=healthcareeng_isaac
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=8
#SBATCH --time=04:00:00
#SBATCH --job-name=rlinf-dd-train
#SBATCH --output=rlinf-dd-chain-%j.out
#SBATCH --error=rlinf-dd-chain-%j.err
# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

# Slurm executes a spool copy: use the persistent checkout, not dirname "$0".
export REPO_ROOT="${REPO_ROOT:-/lustre/fsw/portfolios/healthcareeng/users/yunl/code/RLinf}"
export CHAIN_TIMEOUT="${CHAIN_TIMEOUT:-3.9h}"
export MAX_RUNS="${MAX_RUNS:-18}"
exec python3 "${REPO_ROOT}/docker/dreamdojo/chain.py" "$@"

# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Check dry-run sweep manifests without Slurm, GPUs or filesystem outputs."""

import os
import shlex
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "docker/dreamdojo/submit_wm_comparison.sh"


def dry_run(**extra):
    env = os.environ.copy()
    for key in (
        "WM_VARIANTS",
        "NOISE_LEVELS",
        "GLOBAL_BATCH_SIZES",
        "ACTOR_LRS",
        "SEED_IDS",
        "TRAIN_WM_STEPS",
        "REPO_ROOT",
    ):
        env.pop(key, None)
    env.update(extra)
    return subprocess.run(
        ["bash", str(SCRIPT), "--dry-run"],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


@pytest.mark.parametrize("batches,count", [("128", 6), ("256 512", 12)])
def test_sweep_size_and_fixed_settings(batches, count):
    result = dry_run(GLOBAL_BATCH_SIZES=batches)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert len(lines) == count
    names = set()
    for line in lines:
        tokens = shlex.split(line)
        values = dict(token.split("=", 1) for token in tokens if "=" in token)
        assert values["MAX_RUNS"] == "18"
        assert values["runner.max_epochs"] == "1000"
        assert values["runner.resume_dir"] == "null"
        assert values["env.train.total_num_envs"] == "64"
        assert values["env.train.rollout_epoch"] == "2"
        assert values["actor.micro_batch_size"] == "8"
        assert values["actor.model.rl_head_config.action_noise_scale"] == "0.0"
        assert values["env.train.num_inference_steps"] == "35"
        assert values["env.eval.num_inference_steps"] == "35"
        assert values["runner.val_check_interval"] == "5"
        assert values["env.train.experiment_name"] == values["env.eval.experiment_name"]
        scratch = "scratch" in values["DREAMDOJO_WM_CHECKPOINT"]
        assert values["env.train.experiment_name"].endswith(
            "_lora" if scratch else "_r64"
        )
        names.add(values["runner.logger.experiment_name"])
    assert len(names) == count


def test_narrowed_batch_comparison():
    result = dry_run(
        WM_VARIANTS="r64", NOISE_LEVELS="0.3", GLOBAL_BATCH_SIZES="256 512"
    )
    assert result.returncode == 0
    assert len(result.stdout.splitlines()) == 2


def manifest(**extra):
    result = dry_run(WM_VARIANTS="scratch_r32", NOISE_LEVELS="0.3", **extra)
    assert result.returncode == 0, result.stderr
    return [
        dict(token.split("=", 1) for token in shlex.split(line) if "=" in token)
        for line in result.stdout.splitlines()
    ]


def test_replicates_change_actor_and_training_seed_but_keep_eval_fixed():
    rows = manifest(SEED_IDS="1 2")
    assert len(rows) == 2
    for row, seed in zip(rows, (1, 2)):
        assert row["actor.seed"] == str(1234 + seed)
        assert row["env.train.seed"] == str(seed)
        assert row["env.eval.seed"] == "0"
        assert row["runner.logger.experiment_name"].endswith(f"_s{1234 + seed}")
        assert row["--job-name"].endswith(f"-s{1234 + seed}")
        assert row["runner.resume_dir"] == row["runner.ckpt_path"] == "null"
        assert row["env.train.num_inference_steps"] == "35"


def test_fifteen_step_train_keeps_thirty_five_step_eval():
    rows = manifest(GLOBAL_BATCH_SIZES="128 256", TRAIN_WM_STEPS="15")
    assert len(rows) == 2
    for row in rows:
        assert row["env.train.num_inference_steps"] == "15"
        assert row["env.eval.num_inference_steps"] == "35"
        assert row["actor.seed"] == "1234"
        assert row["env.train.seed"] == row["env.eval.seed"] == "0"
        assert row["runner.logger.experiment_name"].endswith("_wm15_s1234")
        assert row["--job-name"].endswith("_wm15")


def test_lower_learning_rate_changes_only_lr_and_names():
    baseline = manifest()[0]
    lower = manifest(ACTOR_LRS="2.5e-6")[0]
    assert lower["actor.optim.lr"] == "2.5e-6"
    assert lower["runner.logger.experiment_name"] == (
        "wm_scratch_r32_n03_gb128_lr2p5e6_s1234"
    )
    changed = {key for key in baseline if baseline[key] != lower[key]}
    assert changed == {
        "actor.optim.lr",
        "runner.logger.experiment_name",
        "--job-name",
        "--output",
        "--error",
    }


def test_intermediate_noise_is_supported():
    result = dry_run(WM_VARIANTS="scratch_r32", NOISE_LEVELS="0.2")
    assert result.returncode == 0, result.stderr
    assert len(result.stdout.splitlines()) == 1
    assert "actor.model.rl_head_config.noise_level=0.2" in result.stdout
    assert "wm_scratch_r32_n02_gb128_s1234" in result.stdout


def test_learning_rate_ablation_has_distinct_names_and_same_wm():
    result = dry_run(
        WM_VARIANTS="scratch_r32",
        NOISE_LEVELS="0.3",
        GLOBAL_BATCH_SIZES="128",
        ACTOR_LRS="5e-6 1e-5",
    )
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert len(lines) == 2
    values = [
        dict(token.split("=", 1) for token in shlex.split(line) if "=" in token)
        for line in lines
    ]
    assert values[0]["actor.optim.lr"] == "5e-6"
    assert values[1]["actor.optim.lr"] == "1e-5"
    assert (
        values[0]["runner.logger.experiment_name"] == "wm_scratch_r32_n03_gb128_s1234"
    )
    assert (
        values[1]["runner.logger.experiment_name"]
        == "wm_scratch_r32_n03_gb128_lr1e5_s1234"
    )
    assert values[0]["DREAMDOJO_WM_CHECKPOINT"] == values[1]["DREAMDOJO_WM_CHECKPOINT"]


@pytest.mark.parametrize(
    "extra",
    [
        {"GLOBAL_BATCH_SIZES": "1024"},
        {"WM_VARIANTS": "unknown"},
        {"NOISE_LEVELS": "0.8"},
        {"ACTOR_LRS": "2.5e-5"},
        {"SEED_IDS": "-1"},
        {"SEED_IDS": "01"},
        {"SEED_IDS": "1.5"},
        {"SEED_IDS": "10000"},
        {"TRAIN_WM_STEPS": "5"},
    ],
)
def test_reject_invalid_matrix_before_submission(extra):
    result = dry_run(**extra)
    assert result.returncode == 2
    assert not result.stdout

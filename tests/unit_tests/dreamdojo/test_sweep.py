# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Generic experiment matrices, configuration isolation and submission guards."""

import os
import shlex
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "docker/dreamdojo/submit_wm_comparison.sh"


def compose_recipe(monkeypatch, overrides=()):
    from hydra import compose, initialize_config_dir

    monkeypatch.setenv("EMBODIED_PATH", str(REPO / "examples/embodiment"))
    with initialize_config_dir(
        config_dir=str(REPO / "examples/embodiment/config"), version_base="1.1"
    ):
        return compose(
            config_name="dreamdojo_trocar_grpo_gr00t_n1d7", overrides=list(overrides)
        )


def test_native_defaults_are_the_real_tested_recipe(monkeypatch):
    for key in ("DREAMDOJO_GPUS", "DREAMDOJO_WM_CHECKPOINT", "DREAMDOJO_WM_EXPERIMENT"):
        monkeypatch.delenv(key, raising=False)
    cfg = compose_recipe(monkeypatch)
    assert cfg.cluster.component_placement["actor,env,rollout"] == "0-7"
    assert cfg.actor.global_batch_size == 128 and cfg.actor.micro_batch_size == 8
    assert cfg.actor.optim.lr == 5e-6 and cfg.actor.seed == 1234
    assert cfg.actor.model.rl_head_config.noise_level == 0.3
    assert cfg.actor.model.rl_head_config.action_noise_scale == 0
    assert cfg.actor.model.rl_head_config.noise_anneal is False
    assert cfg.actor.model.denoising_steps == 4
    assert cfg.env.train.total_num_envs == 64 and cfg.env.train.rollout_epoch == 2
    assert cfg.env.train.num_inference_steps == 15
    assert cfg.env.eval.num_inference_steps == 35
    assert cfg.env.eval.total_num_envs == 56 and cfg.env.eval.eval_unique_episodes
    for env in (cfg.env.train, cfg.env.eval):
        assert env.enable_kir is False and env.seed == 0
        assert "lora_r32_scratch_lr3e-4_18k" in env.model_path
        assert env.experiment_name.endswith("posttrain_lora")
        assert env.reward_model.type == "MilestoneRewardV2"
    assert cfg.runner.resume_dir is None and cfg.runner.ckpt_path is None
    assert cfg.runner.val_check_interval == cfg.runner.save_interval == 5
    assert cfg.runner.max_epochs == 1000


@pytest.mark.parametrize("gpus,envs,batch", [(None, 56, 112), ("0-7", 64, 128)])
def test_local_wrapper_topology_and_cli_precedence(
    tmp_path, monkeypatch, gpus, envs, batch
):
    stub = tmp_path / "python-stub"
    stub.write_text(
        '#!/bin/bash\nprintf "%s\\0" "$DREAMDOJO_GPUS" "$DREAMDOJO_WM_CHECKPOINT" "$@"\n'
    )
    stub.chmod(0o755)
    env = {"PATH": os.environ["PATH"], "RLINF_PYTHON": str(stub)}
    if gpus is not None:
        env["DREAMDOJO_GPUS"] = gpus
    command = ["bash", str(REPO / "examples/embodiment/run_dreamdojo_trocar.sh")]
    for overrides in ([], [f"actor.global_batch_size={batch * 2}"]):
        result = subprocess.run(
            command + overrides, env=env, check=True, capture_output=True, text=True
        )
        tokens = result.stdout.rstrip("\0").split("\0")
        assert tokens[0] == (gpus or "0-3,5-7")
        assert "lora_r32_scratch_lr3e-4_18k" in tokens[1]
        monkeypatch.setenv("DREAMDOJO_GPUS", tokens[0])
        cfg = compose_recipe(monkeypatch, tokens[tokens.index("--config-name") + 2 :])
        ranks = 8 if gpus else 7
        assert cfg.env.train.total_num_envs == envs
        assert cfg.actor.global_batch_size == batch * (2 if overrides else 1)
        assert cfg.actor.global_batch_size % (ranks * cfg.actor.micro_batch_size) == 0
        assert cfg.env.train.total_num_envs % (ranks * cfg.algorithm.group_size) == 0
        chunks = cfg.env.train.max_episode_steps // cfg.actor.model.num_action_chunks
        samples = cfg.env.train.total_num_envs * cfg.env.train.rollout_epoch * chunks
        assert samples // cfg.actor.global_batch_size == (10 if overrides else 20)


def dry_run(**extra):
    env = os.environ.copy()
    for key in (
        "WM_VARIANTS",
        "NOISE_LEVELS",
        "GLOBAL_BATCH_SIZES",
        "ACTOR_LRS",
        "SEED_IDS",
        "TRAIN_WM_STEPS",
        "KIR_ENABLED",
        "KIR_PROBABILITY",
        "KIR_MAX_OFFSET_FRAMES",
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


@pytest.mark.parametrize("batches,count", [("128", 1), ("256 512", 2)])
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
        assert values["env.train.num_inference_steps"] == "15"
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


def test_original_matrix_is_opt_in():
    result = dry_run(
        WM_VARIANTS="r64 scratch_r32", NOISE_LEVELS="0.1 0.3 0.5", TRAIN_WM_STEPS="35"
    )
    assert result.returncode == 0, result.stderr
    assert len(result.stdout.splitlines()) == 6
    assert "env.train.num_inference_steps=35" in result.stdout


def test_default_preview_selects_only_best_run():
    result = dry_run()
    assert result.returncode == 0, result.stderr
    assert len(result.stdout.splitlines()) == 1
    assert "wm_scratch_r32_n03_gb128_wm15_s1234" in result.stdout
    assert "lora_r32_lr3e-4_r64_18k" not in result.stdout
    assert "env.train.enable_kir=false" in result.stdout


def test_submission_preflight_requires_only_selected_world_model(tmp_path):
    """Local sbatch stub: default scratch run must not require retired r64 assets."""
    base = tmp_path / "cluster"
    repo = tmp_path / "repo"
    site_env = repo / "docs/dreamdojo/cluster.env.example"
    site_env.parent.mkdir(parents=True)
    site_env.write_text(
        (REPO / "docs/dreamdojo/cluster.env.example")
        .read_text()
        .replace("/lustre/fsw/portfolios/healthcareeng/users/yunl", str(base))
    )
    assets = (
        "docker/rlinf-dreamdojo-gr00t2d9a9fc-cu128-v1.sqsh",
        "code/RLinf-runtime/20260911-chain-v1/rlinf/runners/embodied_runner.py",
        "code/RLinf-runtime/20260911-chain-v1/rlinf/utils/train_budget.py",
        "code/RLinf-runtime/20260911-lam-strict/external/lam/model.py",
        "checkpoints/DreamDojo/LAM_400k.ckpt",
        "checkpoints/reward/milestone_v2/best.pt",
        "checkpoints/DreamDojo/lora_r32_scratch_lr3e-4_18k/checkpoints/iter_000018000/model_ema_bf16.pt",
    )
    for asset in assets:
        path = base / asset
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"unit-test-asset")
    (base / "checkpoints/gr00t_ft/g1_pick_trocar_head_10k_bs32_lr1e-4/976127").mkdir(
        parents=True
    )
    stub = tmp_path / "sbatch"
    stub.write_text('#!/bin/bash\nprintf "123456\\n"\n')
    stub.chmod(0o755)
    result = subprocess.run(
        ["bash", str(SCRIPT), "--submit"],
        env={
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "CLUSTER_BASE": str(base),
            "REPO_ROOT": str(repo),
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Submitted wm_scratch_r32_n03_gb128_wm15_s1234: 123456" in result.stdout
    assert (
        len(list((base / "outputs/rlinf/sweeps/best_scratch_wm15").glob("*.jobid")))
        == 1
    )


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
        assert row["env.train.num_inference_steps"] == "15"


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
        "wm_scratch_r32_n03_gb128_lr2p5e6_wm15_s1234"
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
    assert "wm_scratch_r32_n02_gb128_wm15_s1234" in result.stdout


def test_kir_is_train_only_and_has_a_distinct_name():
    row = manifest(KIR_ENABLED="true")[0]
    assert row["env.train.enable_kir"] == "true"
    assert row["env.eval.enable_kir"] == "false"
    assert row["env.train.kir_probability"] == "0.5"
    assert row["env.train.kir_max_offset_frames"] == "30"
    assert (
        row["runner.logger.experiment_name"]
        == "wm_scratch_r32_n03_gb128_wm15_kirhandp50_s1234"
    )
    assert row["env.train.num_inference_steps"] == "15"
    assert row["env.eval.num_inference_steps"] == "35"
    assert manifest()[0]["env.train.enable_kir"] == "false"


@pytest.mark.parametrize(
    "extra,changed_key,expected,suffix",
    [
        ({"KIR_PROBABILITY": "1.0"}, "env.train.kir_probability", "1.0", "kirhandp100"),
        (
            {"KIR_MAX_OFFSET_FRAMES": "15"},
            "env.train.kir_max_offset_frames",
            "15",
            "kirhandp50_off15",
        ),
    ],
)
def test_kir_ablation_changes_only_one_setting_and_names(
    extra, changed_key, expected, suffix
):
    baseline = manifest(KIR_ENABLED="true")[0]
    variant = manifest(KIR_ENABLED="true", **extra)[0]
    assert variant[changed_key] == expected
    assert variant["runner.logger.experiment_name"].endswith(f"_{suffix}_s1234")
    assert {key for key in baseline if baseline[key] != variant[key]} == {
        changed_key,
        "runner.logger.experiment_name",
        "--job-name",
        "--output",
        "--error",
    }


def test_combined_sweep_variants_have_unique_names_and_fixed_eval():
    rows = []
    for steps in ("15", "35"):
        common = {"SEED_IDS": "0 1 2", "TRAIN_WM_STEPS": steps}
        rows += manifest(**common)
        for probability, offset in (("0.5", "30"), ("1.0", "30"), ("0.5", "15")):
            rows += manifest(
                **common,
                KIR_ENABLED="true",
                KIR_PROBABILITY=probability,
                KIR_MAX_OFFSET_FRAMES=offset,
            )
    names = {row["runner.logger.experiment_name"] for row in rows}
    assert len(rows) == len(names) == 24
    for row in rows:
        assert row.get("env.eval.enable_kir", "false") == "false"
        assert row["env.eval.num_inference_steps"] == "35"
        assert row["env.eval.seed"] == "0"
        assert row["actor.optim.lr"] == "5e-6"
        assert row["actor.global_batch_size"] == "128"
        assert row["runner.resume_dir"] == row["runner.ckpt_path"] == "null"


@pytest.mark.parametrize(
    "extra",
    [{}, {"KIR_PROBABILITY": "1.0"}, {"KIR_MAX_OFFSET_FRAMES": "15"}],
)
def test_kir_submission_composes_and_changes_only_training_reset(monkeypatch, extra):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    monkeypatch.setenv("EMBODIED_PATH", str(REPO / "examples/embodiment"))
    monkeypatch.setenv("DREAMDOJO_GPUS", "0-7")
    configs = []
    for enabled in ("false", "true"):
        result = dry_run(
            WM_VARIANTS="scratch_r32",
            NOISE_LEVELS="0.3",
            KIR_ENABLED=enabled,
            **(extra if enabled == "true" else {}),
        )
        assert result.returncode == 0, result.stderr
        tokens = shlex.split(result.stdout.strip())
        entry = next(
            i for i, token in enumerate(tokens) if token.endswith("/train_chain.slurm")
        )
        with initialize_config_dir(
            config_dir=str(REPO / "examples/embodiment/config"), version_base="1.1"
        ):
            cfg = compose(
                config_name="dreamdojo_trocar_grpo_gr00t_n1d7",
                overrides=tokens[entry + 1 :],
            )
        configs.append(OmegaConf.to_container(cfg, resolve=True))
    baseline, kir = configs
    assert kir["env"]["train"]["enable_kir"] is True
    assert kir["env"]["eval"]["enable_kir"] is False
    assert kir["env"]["train"]["total_num_envs"] == 64
    assert kir["env"]["train"]["rollout_epoch"] == 2
    assert kir["actor"]["global_batch_size"] == 128
    assert kir["actor"]["micro_batch_size"] == 8
    assert kir["runner"]["max_epochs"] == 1000
    kir["env"]["train"]["enable_kir"] = False
    for key in ("kir_probability", "kir_max_offset_frames"):
        kir["env"]["train"][key] = baseline["env"]["train"][key]
    kir["runner"]["logger"]["experiment_name"] = baseline["runner"]["logger"][
        "experiment_name"
    ]
    assert kir == baseline


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
        values[0]["runner.logger.experiment_name"]
        == "wm_scratch_r32_n03_gb128_wm15_s1234"
    )
    assert (
        values[1]["runner.logger.experiment_name"]
        == "wm_scratch_r32_n03_gb128_lr1e5_wm15_s1234"
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
        {"KIR_ENABLED": "yes"},
        {"KIR_ENABLED": "true", "KIR_PROBABILITY": "0.75"},
        {"KIR_ENABLED": "true", "KIR_MAX_OFFSET_FRAMES": "0"},
        {"KIR_ENABLED": "true", "KIR_MAX_OFFSET_FRAMES": "15.0"},
        {"KIR_PROBABILITY": "1.0"},
        {"KIR_MAX_OFFSET_FRAMES": "15"},
    ],
)
def test_reject_invalid_matrix_before_submission(extra):
    result = dry_run(**extra)
    assert result.returncode == 2
    assert not result.stdout

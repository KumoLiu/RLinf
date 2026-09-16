# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Small packaging checks; no Docker daemon or cluster required."""

import importlib.util
import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "dreamdojo_container_context", REPO / "docker/dreamdojo/prepare_context.py"
)
CONTEXT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONTEXT)


def test_package_names_are_canonical():
    assert CONTEXT.canonical("Flash_Attn") == "flash-attn"
    assert CONTEXT.canonical("RLinf") in CONTEXT.EDITABLES


def test_cuda_wheels_have_integrity_hashes_and_fixed_abi():
    assert set(CONTEXT.CUDA_WHEELS) == set(CONTEXT.EXPECTED_CUDA)
    for url in CONTEXT.CUDA_WHEELS.values():
        assert "cp310-cp310" in url
        assert len(url.split("#sha256=")[1]) == 64


def test_source_snapshot_omits_hidden_files_weights_and_cache(tmp_path, monkeypatch):
    source = tmp_path / "repo"
    code = source / "code"
    code.mkdir(parents=True)
    (code / "module.py").write_text("VALUE = 1\n")
    (code / ".env").write_text("SECRET=not-for-container\n")
    (code / "weights.pt").write_bytes(b"weights")
    (code / "__pycache__").mkdir()
    (code / "__pycache__/module.pyc").write_bytes(b"cache")
    monkeypatch.setattr(
        CONTEXT, "git", lambda root, *args: "abc" if args[0] == "rev-parse" else ""
    )
    target = tmp_path / "context"
    manifest = CONTEXT.copy_source(source, target, ["code"])
    assert manifest["files"] == 1
    assert sorted(p.name for p in (target / "code").iterdir()) == ["module.py"]


def test_source_snapshot_rejects_symlinks(tmp_path):
    source = tmp_path / "repo"
    source.mkdir()
    (source / "link").symlink_to("/etc/passwd")
    with pytest.raises(ValueError, match="symlink"):
        CONTEXT.copy_source(source, tmp_path / "context", ["link"])


def test_lam_is_packaged_and_deep_wm_import_is_checked():
    assert "external/lam" in CONTEXT.DREAMDOJO_SOURCE_ROOTS
    spec = importlib.util.spec_from_file_location(
        "dreamdojo_container_smoke", REPO / "docker/dreamdojo/smoke.py"
    )
    smoke = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(smoke)
    assert "external.lam.model" in smoke.SMOKE_IMPORTS
    assert (
        "cosmos_predict2._src.predict2.action.models.action_conditioned_video2world_rectified_flow_model"
        in smoke.GPU_IMPORTS
    )


def test_slurm_uses_shared_checkpoints_readonly_without_copying(tmp_path):
    """Inspect the real launch command with a stub srun; never submit a job."""
    base = tmp_path / "cluster"
    checkpoint_root = base / "checkpoints"
    sft = checkpoint_root / "gr00t_ft/g1_pick_trocar_head_10k_bs32_lr1e-4/976127"
    sft.mkdir(parents=True)
    wm = checkpoint_root / "wm.pt"
    reward = checkpoint_root / "reward.pt"
    lam = checkpoint_root / "lam.ckpt"
    lam.write_bytes(b"test-checkpoint")
    for path in (wm, reward):
        path.touch()
    (base / "rlinf_assets/20260911/data").mkdir(parents=True)
    (base / "cache/huggingface/hub").mkdir(parents=True)
    image = tmp_path / "image.sqsh"
    image.touch()
    stub = tmp_path / "srun"
    stub.write_text(
        '#!/bin/bash\nprintf "P2P_DISABLE=%s\\0" "$NCCL_P2P_DISABLE"\n'
        'printf "%s\\0" "$@"\n'
    )
    stub.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "CLUSTER_BASE": str(base),
        "CONTAINER_IMAGE": str(image),
        "DREAMDOJO_WM_CHECKPOINT": str(wm),
        "DREAMDOJO_REWARD_CHECKPOINT": str(reward),
        "DREAMDOJO_LAM_CHECKPOINT": str(lam),
        "GR00T_MODEL_PATH": str(sft),
        "SLURM_JOB_ID": "test123",
        "SLURM_SUBMIT_DIR": str(REPO),
    }
    for key in (
        "DREAMDOJO_EXTERNAL_ROOT",
        "DREAMDOJO_KIR_SOURCE_ROOT",
        "DREAMDOJO_VIDEO_AUDIT_SOURCE_ROOT",
        "NCCL_P2P_DISABLE",
        "NCCL_P2P_LEVEL",
    ):
        env.pop(key, None)
    result = subprocess.run(
        ["bash", str(REPO / "docker/dreamdojo/run_cluster.slurm"), "eval"],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    args = result.stdout.split("\0")
    mounts = next(x for x in args if x.startswith("--container-mounts="))
    assert f"{checkpoint_root}:{checkpoint_root}:ro" in mounts
    assert "/assets/models" not in result.stdout
    assert "evaluations/eval_embodied_agent.py" in result.stdout
    assert "P2P_DISABLE=0\0" in result.stdout
    forwarded = next(x for x in args if x.startswith("--container-env="))
    assert "DREAMDOJO_LAM_CHECKPOINT" in forwarded
    assert "NCCL_P2P_DISABLE" in forwarded
    assert not (base / "rlinf_assets/20260911/models").exists()
    env["DREAMDOJO_WM_CHECKPOINT"] = str(checkpoint_root / "missing.pt")
    missing = subprocess.run(
        ["bash", str(REPO / "docker/dreamdojo/run_cluster.slurm"), "eval"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert missing.returncode != 0
    assert "Missing checkpoint" in missing.stderr

    # Optional v1 supplement must be explicit, present and read-only.
    env["DREAMDOJO_WM_CHECKPOINT"] = str(wm)
    external = tmp_path / "external"
    env["DREAMDOJO_EXTERNAL_ROOT"] = str(external)
    missing_source = subprocess.run(
        ["bash", str(REPO / "docker/dreamdojo/run_cluster.slurm"), "eval"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert missing_source.returncode != 0
    assert "Missing DreamDojo external source" in missing_source.stderr
    (external / "lam").mkdir(parents=True)
    (external / "lam/model.py").touch()
    supplemented = subprocess.run(
        ["bash", str(REPO / "docker/dreamdojo/run_cluster.slurm"), "eval"],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert f"{external}:/opt/src/DreamDojo/external:ro" in supplemented.stdout

    audit = tmp_path / "eval-audit"
    env["DREAMDOJO_VIDEO_AUDIT_SOURCE_ROOT"] = str(audit)
    missing_audit = subprocess.run(
        ["bash", str(REPO / "docker/dreamdojo/run_cluster.slurm"), "eval"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert missing_audit.returncode != 0
    assert "Missing video audit source" in missing_audit.stderr
    for relative in (
        "rlinf/data/datasets/lerobot_world_model.py",
        "rlinf/envs/world_model/world_model_dreamdojo_env.py",
        "rlinf/envs/world_model/dreamdojo_reward.py",
        "rlinf/envs/world_model/dreamdojo_video_audit.py",
        "examples/embodiment/config/env/dreamdojo_trocar.yaml",
    ):
        path = audit / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# test snapshot\n")
    audited = subprocess.run(
        ["bash", str(REPO / "docker/dreamdojo/run_cluster.slurm"), "eval"],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert (
        f"{audit}/rlinf/envs/world_model/dreamdojo_video_audit.py:/opt/src/RLinf/rlinf/envs/world_model/dreamdojo_video_audit.py:ro"
        in audited.stdout
    )
    rejected = subprocess.run(
        ["bash", str(REPO / "docker/dreamdojo/run_cluster.slurm"), "train"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert rejected.returncode != 0
    assert "requires eval mode" in rejected.stderr
    env.pop("DREAMDOJO_VIDEO_AUDIT_SOURCE_ROOT")

    verify = subprocess.run(
        ["bash", str(REPO / "docker/dreamdojo/run_cluster.slurm"), "verify"],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "verify_runtime.py:/opt/rlinf-build/verify_runtime.py:ro" in verify.stdout
    for checkpoint, error in (
        (checkpoint_root / "absent.ckpt", "Missing checkpoint"),
        (lam, "Empty LAM checkpoint"),
    ):
        lam.write_bytes(b"")
        env["DREAMDOJO_LAM_CHECKPOINT"] = str(checkpoint)
        failed = subprocess.run(
            ["bash", str(REPO / "docker/dreamdojo/run_cluster.slurm"), "train"],
            env=env,
            capture_output=True,
            text=True,
        )
        assert failed.returncode != 0
        assert error in failed.stderr

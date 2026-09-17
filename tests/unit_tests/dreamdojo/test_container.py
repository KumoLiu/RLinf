# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Source packaging and runtime checks without Docker or a cluster."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "dreamdojo_container_context", REPO / "docker/dreamdojo/prepare_context.py"
)
CONTEXT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONTEXT)
CHECK_SPEC = importlib.util.spec_from_file_location(
    "dreamdojo_runtime_checks", REPO / "docker/dreamdojo/check_runtime.py"
)
CHECKS = importlib.util.module_from_spec(CHECK_SPEC)
CHECK_SPEC.loader.exec_module(CHECKS)


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
    assert "external.lam.model" in CHECKS.SMOKE_IMPORTS
    assert (
        "cosmos_predict2._src.predict2.action.models.action_conditioned_video2world_rectified_flow_model"
        in CHECKS.GPU_IMPORTS
    )


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("imports", ["imports"]),
        ("cuda", ["imports", "cuda"]),
        ("lam", ["lam"]),
        ("nccl", ["nccl"]),
    ],
)
def test_runtime_check_modes_are_isolated(monkeypatch, tmp_path, mode, expected):
    calls = []
    versions = tmp_path / "not-needed-for-lam-or-nccl.json"

    def imports(path):
        assert path == versions
        calls.append("imports")
        return 1

    monkeypatch.setattr(CHECKS, "verify_imports", imports)
    for name in ("cuda", "lam", "nccl"):
        monkeypatch.setattr(
            CHECKS, f"verify_{name}", lambda name=name: calls.append(name)
        )
    CHECKS.main([mode, "--versions-file", str(versions)])
    assert calls == expected


def test_runtime_check_help_needs_only_standard_library():
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            str(REPO / "docker/dreamdojo/check_runtime.py"),
            "--help",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "{imports,cuda,lam,nccl}" in result.stdout


def test_runtime_import_check_never_loads_gpu_only_modules(tmp_path, monkeypatch):
    versions = tmp_path / "versions.json"
    versions.write_text('{"example": "1.0"}')
    monkeypatch.setattr(CHECKS.metadata, "version", lambda name: "1.0")
    imported = []

    def import_module(name):
        assert name not in CHECKS.GPU_IMPORTS
        imported.append(name)
        return SimpleNamespace(__version__="4.12.0")

    monkeypatch.setattr(CHECKS.importlib, "import_module", import_module)
    assert CHECKS.verify_imports(versions) == 1
    assert set(imported) == set(CHECKS.SMOKE_IMPORTS)


def test_runtime_import_check_rejects_version_drift(tmp_path, monkeypatch):
    versions = tmp_path / "versions.json"
    versions.write_text('{"example": "1.0"}')
    monkeypatch.setattr(CHECKS.metadata, "version", lambda name: "2.0")
    monkeypatch.setattr(
        CHECKS.importlib,
        "import_module",
        lambda name: pytest.fail("import before pins"),
    )
    with pytest.raises(AssertionError, match="Runtime package versions differ"):
        CHECKS.verify_imports(versions)


def test_build_context_copies_unified_checker_and_uses_cpu_mode(tmp_path, monkeypatch):
    directory = REPO / "docker/dreamdojo"
    versions = json.loads((directory / "runtime-versions.json").read_text())
    output = tmp_path / "context"
    monkeypatch.setattr(CONTEXT, "freeze", lambda: versions)
    monkeypatch.setattr(
        CONTEXT,
        "git",
        lambda root, *args: CONTEXT.GR00T_COMMIT if args[0] == "rev-parse" else "",
    )
    monkeypatch.setattr(CONTEXT, "copy_source", lambda *args: {"files": 0})
    monkeypatch.setattr(CONTEXT.sys, "version_info", (3, 10, 21))
    monkeypatch.setattr(
        CONTEXT.sys,
        "argv",
        [
            "prepare_context.py",
            "--output",
            str(output),
            "--dreamdojo",
            str(tmp_path),
            "--gr00t",
            str(tmp_path),
        ],
    )
    CONTEXT.main()
    assert (output / "check_runtime.py").read_bytes() == (
        directory / "check_runtime.py"
    ).read_bytes()
    assert not (output / "smoke.py").exists()
    assert not (output / "verify_runtime.py").exists()
    dockerfile = (output / "Dockerfile").read_text()
    assert "runtime-versions.json check_runtime.py /opt/rlinf-build/" in dockerfile
    assert "RUN python /opt/rlinf-build/check_runtime.py imports" in dockerfile
    assert "check_runtime.py cuda" not in dockerfile


@pytest.mark.parametrize("custom_checker", [False, True])
def test_slurm_smoke_mounts_checker_without_assets(tmp_path, custom_checker):
    image = tmp_path / "image.sqsh"
    image.touch()
    stub = tmp_path / "srun"
    stub.write_text('#!/bin/bash\nprintf "%s\\0" "$@"\n')
    stub.chmod(0o755)
    env = {
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "CLUSTER_BASE": str(tmp_path / "cluster"),
        "CONTAINER_IMAGE": str(image),
        "SLURM_JOB_ID": "test-smoke",
        "SLURM_SUBMIT_DIR": str(REPO),
    }
    checker = REPO / "docker/dreamdojo/check_runtime.py"
    if custom_checker:
        checker = tmp_path / "check_runtime.py"
        checker.write_text("# Explicit checker snapshot for this job\n")
        env["DREAMDOJO_CHECK_SCRIPT"] = str(checker)
    command = ["bash", str(REPO / "docker/dreamdojo/run_cluster.slurm"), "smoke"]
    result = subprocess.run(
        command, env=env, check=True, capture_output=True, text=True
    )
    mounts = next(
        x for x in result.stdout.split("\0") if x.startswith("--container-mounts=")
    )
    assert f"{checker}:/opt/rlinf-build/check_runtime.py:ro" in mounts
    assert "/assets/data" not in mounts
    assert "check_runtime.py cuda" in result.stdout

    for unavailable in (tmp_path / "missing.py", tmp_path / "empty.py"):
        if unavailable.name == "empty.py":
            unavailable.touch()
        env["DREAMDOJO_CHECK_SCRIPT"] = str(unavailable)
        failed = subprocess.run(command, env=env, capture_output=True, text=True)
        assert failed.returncode != 0
        assert "Missing or empty runtime check script" in failed.stderr


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
        "DREAMDOJO_CHECK_SCRIPT",
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
    assert "check_runtime.py:ro" not in mounts
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
    assert "check_runtime.py:/opt/rlinf-build/check_runtime.py:ro" in verify.stdout
    assert "check_runtime.py lam" in verify.stdout
    assert "check_runtime.py nccl" in verify.stdout
    assert "--nproc-per-node=8" in verify.stdout
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

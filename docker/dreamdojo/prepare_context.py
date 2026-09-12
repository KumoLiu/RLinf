# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Prepare a small, credential-free build context from the tested local venv.

Run with the validated RLinf interpreter. Outputs are generated build artifacts,
not modifications of the source environments. Existing output dirs are rejected.
"""

import argparse
import hashlib
import importlib.metadata as metadata
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

EDITABLES = {"rlinf", "gr00t", "cosmos-predict2", "cosmos-oss", "cosmos-cuda"}
CUDA_WHEELS = {
    "torch": "https://download.pytorch.org/whl/cu128/torch-2.7.0%2Bcu128-cp310-cp310-manylinux_2_28_x86_64.whl#sha256=ac1849553ee673dfafb44c610c60cb60a2890f0e117f43599a526cf777eb8b8c",
    "flash-attn": "https://github.com/nvidia-cosmos/cosmos-dependencies/releases/download/v1.2.0/flash_attn-2.7.3%2Bcu128.torch27-cp310-cp310-linux_x86_64.whl#sha256=88c104eb74fa84ed1993c2e9db4e1987269080927d3b04ed9bfdb937a71e9f4a",
    "natten": "https://github.com/nvidia-cosmos/cosmos-dependencies/releases/download/v1.2.0/natten-0.21.0%2Bcu128.torch27-cp310-cp310-linux_x86_64.whl#sha256=bbfc51cf240e67f8ba9fe1acaa62ec451d2875190dd81b72eb28ae73dbaaaeef",
    "transformer-engine": "https://github.com/nvidia-cosmos/cosmos-dependencies/releases/download/v1.2.0/transformer_engine-2.2%2Bcu128.torch27-cp310-cp310-linux_x86_64.whl#sha256=fc79fef4a1bbd912f9064e28faeef30b0dba0cbf1f9e2f6647f7b4a5a7abc1e6",
    "xformers": "https://download.pytorch.org/whl/cu128/xformers-0.0.30-cp310-cp310-manylinux_2_28_x86_64.whl#sha256=fc3cc23baf901e2ecb33f525bca12321bd275d2ad69360beb6edaf6dda3ab064",
}
EXPECTED_CUDA = {
    "torch": "2.7.0+cu128",
    "flash-attn": "2.7.3+cu128.torch27",
    "natten": "0.21.0+cu128.torch27",
    "transformer-engine": "2.2+cu128.torch27",
    "xformers": "0.0.30",
}
GR00T_COMMIT = "2d9a9fce9811d10510d1263344fbf5253f33639d"
DREAMDOJO_SOURCE_ROOTS = (
    "cosmos_predict2",
    "groot_dreams",
    "external/lam",
    "packages/cosmos-oss",
    "packages/cosmos-cuda",
    "configs",
    "shared_meta",
    "scripts",
    "pyproject.toml",
    "README.md",
    "LICENSE",
)


def canonical(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def freeze():
    versions = {}
    for dist in metadata.distributions():
        name = canonical(dist.metadata["Name"])
        if name in EDITABLES:
            continue
        direct = json.loads(dist.read_text("direct_url.json") or "{}")
        if direct:
            raise ValueError(f"Unreviewed direct/local dependency: {name}")
        if name in versions and versions[name] != dist.version:
            raise ValueError(f"Conflicting installed distributions: {name}")
        versions[name] = dist.version
    for name, version in EXPECTED_CUDA.items():
        if versions.get(name) != version:
            raise ValueError(f"CUDA wheel pin no longer matches local {name}")
    return dict(sorted(versions.items()))


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def copy_source(root, target, roots):
    """Copy only explicitly allowed runtime code/assets, never caches or links."""
    digest = hashlib.sha256()
    count = 0
    for relative in sorted(set(roots)):
        source = root / relative
        if not source.exists():
            raise FileNotFoundError(source)
        files = sorted(source.rglob("*")) if source.is_dir() else [source]
        for path in files:
            rel = path.relative_to(root)
            if any(part.startswith(".") or part == "__pycache__" for part in rel.parts):
                continue
            if path.suffix in {".pyc", ".mp4", ".pt", ".pth", ".ckpt", ".sqsh"}:
                continue
            if path.is_symlink():
                raise ValueError(f"Review symlink before packaging: {path}")
            if not path.is_file():
                continue
            dest = target / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
            digest.update(str(rel).encode() + b"\0" + path.read_bytes())
            count += 1
    return {
        "commit": git(root, "rev-parse", "HEAD"),
        "dirty": bool(git(root, "status", "--porcelain")),
        "runtime_tree_sha256": digest.hexdigest(),
        "files": count,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dreamdojo", type=Path, required=True)
    parser.add_argument("--gr00t", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    if sys.version_info[:3] != (3, 10, 21):
        raise ValueError("Use the validated Python 3.10.21 interpreter")
    if git(args.gr00t, "rev-parse", "HEAD") != GR00T_COMMIT:
        raise ValueError("GR00T source differs from the tested commit")
    if git(args.gr00t, "status", "--porcelain"):
        raise ValueError("GR00T source must be clean")
    versions = freeze()
    baseline = json.loads((Path(__file__).parent / "runtime-versions.json").read_text())
    if versions != baseline:
        raise ValueError(
            "Local packages differ from the recorded runtime baseline; review before rebuilding"
        )
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "requirements.txt").write_text(
        "# Frozen installed versions; use --no-deps, not upstream re-resolution.\n"
        + "\n".join(f"{k}=={v}" for k, v in versions.items() if k not in CUDA_WHEELS)
        + "\n"
    )
    (args.output / "cuda-wheels.txt").write_text(
        "\n".join(f"{k} @ {v}" for k, v in CUDA_WHEELS.items()) + "\n"
    )
    manifest = {
        "python": sys.version,
        "RLinf": copy_source(
            repo,
            args.output / "src/RLinf",
            [
                "rlinf",
                "examples/embodiment",
                "evaluations",
                "toolkits/world_model",
                "tests/unit_tests",
                "pyproject.toml",
                "README.md",
                "LICENSE",
            ],
        ),
        "DreamDojo": copy_source(
            args.dreamdojo,
            args.output / "src/DreamDojo",
            DREAMDOJO_SOURCE_ROOTS,
        ),
        "GR00T": copy_source(
            args.gr00t,
            args.output / "src/Isaac-GR00T",
            [
                "gr00t",
                "pyproject.toml",
                "README.md",
                "LICENSE",
            ],
        ),
    }
    for filename, payload in [
        ("source-manifest.json", manifest),
        ("runtime-versions.json", versions),
    ]:
        (args.output / filename).write_text(json.dumps(payload, indent=2) + "\n")
    for name in ("Dockerfile", "smoke.py"):
        shutil.copy2(Path(__file__).parent / name, args.output / name)
    print(
        json.dumps(
            {
                "context": str(args.output.resolve()),
                "packages": len(versions),
                "sources": manifest,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Import/version and optional CUDA forward/backward checks; no training data."""

import argparse
import importlib
import importlib.metadata as metadata
import json
from pathlib import Path

SMOKE_IMPORTS = (
    "torch",
    "torchvision",
    "cv2",
    "av",
    "decord",
    "ray",
    "transformers",
    "flash_attn",
    "transformer_engine.pytorch",
    "megatron.core",
    "gr00t.model.gr00t_n1d7.processing_gr00t_n1d7",
    "rlinf.workers.rollout.hf.huggingface_worker",
    "rlinf.envs.world_model.world_model_dreamdojo_env",
    "external.lam.model",
)
# This module asserts FlashAttention GPU availability during import. Keep the
# Docker-build (CPU) check valid, and exercise it in --gpu deployment checks.
GPU_IMPORTS = (
    "cosmos_predict2._src.predict2.action.models.action_conditioned_video2world_rectified_flow_model",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    expected = json.loads(Path("/opt/rlinf-build/runtime-versions.json").read_text())
    mismatches = {
        name: (version, metadata.version(name))
        for name, version in expected.items()
        if metadata.version(name) != version
    }
    if mismatches:
        raise AssertionError(mismatches)
    for module in SMOKE_IMPORTS + (GPU_IMPORTS if args.gpu else ()):
        importlib.import_module(module)
        print("IMPORT_OK", module, flush=True)
    import cv2
    import torch

    assert cv2.__version__ == "4.12.0", cv2.__version__
    if args.gpu:
        from flash_attn import flash_attn_func
        from transformer_engine.pytorch import Linear

        assert torch.cuda.is_available()
        for index in range(torch.cuda.device_count()):
            with torch.cuda.device(index):
                qkv = [
                    torch.randn(
                        2,
                        64,
                        4,
                        32,
                        device="cuda",
                        dtype=torch.bfloat16,
                        requires_grad=True,
                    )
                    for _ in range(3)
                ]
                result = flash_attn_func(*qkv)
                result.float().square().mean().backward()
                assert torch.isfinite(result).all() and all(
                    torch.isfinite(x.grad).all() for x in qkv
                )
                layer = Linear(128, 64, params_dtype=torch.bfloat16, device="cuda")
                data = torch.randn(
                    8, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True
                )
                output = layer(data)
                output.float().square().mean().backward()
                assert torch.isfinite(output).all() and torch.isfinite(data.grad).all()
                torch.cuda.synchronize()
                print("GPU_OK", index, torch.cuda.get_device_name(index), flush=True)
    print(
        "SMOKE_OK",
        json.dumps({"packages": len(expected), "cuda_tested": args.gpu}),
        flush=True,
    )


if __name__ == "__main__":
    main()

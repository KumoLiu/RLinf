# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Container checks: CPU imports, CUDA kernels, real-weight LAM, or eight-rank NCCL.

Imports are lazy so Docker build and --help need neither GPUs nor checkpoints.
These checks do not train or evaluate a policy.
"""

import argparse
import importlib
import importlib.metadata as metadata
import json
import logging
import os
import sys
from datetime import timedelta
from pathlib import Path

LOGGER = logging.getLogger(__name__)
DEFAULT_VERSIONS = Path("/opt/rlinf-build/runtime-versions.json")
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
# This module asserts FlashAttention GPU availability during import.
GPU_IMPORTS = (
    "cosmos_predict2._src.predict2.action.models.action_conditioned_video2world_rectified_flow_model",
)


def verify_imports(versions_file: Path) -> int:
    """Check pinned packages and CPU-safe imports; return the package count."""
    expected = json.loads(versions_file.read_text())
    installed = {name: metadata.version(name) for name in expected}
    mismatches = {
        name: (version, installed[name])
        for name, version in expected.items()
        if installed[name] != version
    }
    if mismatches:
        raise AssertionError(f"Runtime package versions differ: {mismatches}")
    for module in SMOKE_IMPORTS:
        importlib.import_module(module)
        LOGGER.info("IMPORT_OK %s", module)
    cv2 = importlib.import_module("cv2")
    assert cv2.__version__ == "4.12.0", f"Unexpected OpenCV: {cv2.__version__}"
    return len(expected)


def verify_cuda() -> None:
    """Check deep WM imports and kernel forward/backward on each visible GPU."""
    for module in GPU_IMPORTS:
        importlib.import_module(module)
        LOGGER.info("IMPORT_OK %s", module)
    import torch
    from flash_attn import flash_attn_func
    from transformer_engine.pytorch import Linear

    assert torch.cuda.is_available(), "CUDA checks require an allocated GPU"
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
            ), f"Non-finite FlashAttention output/gradient on GPU {index}"
            layer = Linear(128, 64, params_dtype=torch.bfloat16, device="cuda")
            data = torch.randn(
                8, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True
            )
            output = layer(data)
            output.float().square().mean().backward()
            assert torch.isfinite(output).all() and torch.isfinite(data.grad).all(), (
                f"Non-finite Transformer Engine output/gradient on GPU {index}"
            )
            torch.cuda.synchronize()
            LOGGER.info("GPU_OK %s %s", index, torch.cuda.get_device_name(index))


def verify_lam() -> None:
    """Restore the production LAM architecture and run its actual encoder."""
    import torch
    from external.lam.model import DEFAULT_LAM_CHECKPOINT, LAM, resolve_lam_checkpoint

    checkpoint = resolve_lam_checkpoint(DEFAULT_LAM_CHECKPOINT)
    torch.cuda.set_device(0)
    torch.manual_seed(0)
    LOGGER.info("LAM_LOAD_START %s", checkpoint)
    model = LAM(
        image_channels=3,
        lam_model_dim=1024,
        lam_latent_dim=32,
        lam_patch_size=16,
        lam_enc_blocks=24,
        lam_dec_blocks=24,
        lam_num_heads=16,
        ckpt_path=DEFAULT_LAM_CHECKPOINT,
    )
    assert model.loaded_checkpoint == checkpoint, (
        "LAM did not load the requested weights"
    )
    LOGGER.info("LAM_LOAD_OK %s", checkpoint)
    model.eval().requires_grad_(False).to(device="cuda:0", dtype=torch.float32)
    with torch.inference_mode():
        videos = torch.rand(1, 2, 240, 320, 3, device="cuda:0")
        latent = model.lam({"videos": videos})["z_rep"]
        assert tuple(latent.shape) == (1, 1, 1, 32), (
            f"Unexpected LAM latent shape: {latent.shape}"
        )
        assert torch.isfinite(latent).all().item(), "LAM returned non-finite latents"
        torch.cuda.synchronize()
        LOGGER.info(
            "LAM_FORWARD_OK %s",
            json.dumps(
                {
                    "input_shape": list(videos.shape),
                    "latent_shape": list(latent.shape),
                    "latent_abs_mean": latent.abs().mean().item(),
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                }
            ),
        )


def verify_nccl() -> None:
    """Check collective and point-to-point correctness on all allocated GPUs."""
    import torch
    import torch.distributed as dist

    assert os.environ.get("NCCL_P2P_DISABLE") == "0", "P2P must be enabled"
    assert not os.environ.get("NCCL_P2P_LEVEL"), "Do not force the local P2P level"
    os.environ.setdefault("NCCL_DEBUG", "INFO")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=90))
    try:
        world = dist.get_world_size()
        assert world == 8, f"Expected 8 allocated GPUs, got {world}"
        LOGGER.info(
            "NCCL_START rank=%s torch=%s nccl=%s device=%s NCCL_P2P_DISABLE=%s",
            rank,
            torch.__version__,
            torch.cuda.nccl.version(),
            torch.cuda.get_device_name(),
            os.environ["NCCL_P2P_DISABLE"],
        )
        for count in (1, 65536, 4 * 1024 * 1024):
            source = torch.full((count,), float(rank), device="cuda")
            reduced = source.clone()
            dist.all_reduce(reduced)
            assert torch.all(reduced == world * (world - 1) / 2).item(), (
                f"NCCL all_reduce mismatch on rank {rank}, count {count}"
            )
            gathered = [torch.empty_like(source) for _ in range(world)]
            dist.all_gather(gathered, source)
            assert all(
                torch.all(value == i).item() for i, value in enumerate(gathered)
            ), f"NCCL all_gather mismatch on rank {rank}, count {count}"
            for root in range(world):
                broadcast = source.clone()
                dist.broadcast(broadcast, src=root)
                assert torch.all(broadcast == root).item(), (
                    f"NCCL broadcast mismatch on rank {rank}, root {root}"
                )
            received = torch.empty_like(source)
            requests = dist.batch_isend_irecv(
                [
                    dist.P2POp(dist.isend, source, (rank + 1) % world),
                    dist.P2POp(dist.irecv, received, (rank - 1) % world),
                ]
            )
            for request in requests:
                request.wait()
            assert torch.all(received == (rank - 1) % world).item(), (
                f"NCCL ring P2P mismatch on rank {rank}, count {count}"
            )
            torch.cuda.synchronize()
        dist.barrier()
        LOGGER.info("NCCL_OK rank=%s all_reduce/all_gather/broadcast/ring_p2p", rank)
    finally:
        dist.destroy_process_group()


def main(argv: list[str] | None = None) -> None:
    """Run only the selected check; CUDA mode includes the CPU import checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("imports", "cuda", "lam", "nccl"))
    parser.add_argument(
        "--versions-file",
        type=Path,
        default=DEFAULT_VERSIONS,
        help="Pinned package manifest used by imports/cuda modes.",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    if args.mode in ("imports", "cuda"):
        packages = verify_imports(args.versions_file)
        if args.mode == "cuda":
            verify_cuda()
        LOGGER.info(
            "SMOKE_OK %s",
            json.dumps({"packages": packages, "cuda_tested": args.mode == "cuda"}),
        )
    elif args.mode == "lam":
        verify_lam()
    else:
        verify_nccl()


if __name__ == "__main__":
    main()

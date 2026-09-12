# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Bounded real-weight LAM and NCCL checks; does not train or evaluate a policy."""

import argparse
import json
import os
from datetime import timedelta


def verify_lam() -> None:
    """Restore the production LAM architecture and run its actual encoder."""
    import torch
    from external.lam.model import DEFAULT_LAM_CHECKPOINT, LAM, resolve_lam_checkpoint

    checkpoint = resolve_lam_checkpoint(DEFAULT_LAM_CHECKPOINT)
    torch.cuda.set_device(0)
    torch.manual_seed(0)
    print(f"LAM_LOAD_START {checkpoint}", flush=True)
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
    assert model.loaded_checkpoint == checkpoint
    print(f"LAM_LOAD_OK {checkpoint}", flush=True)
    model.eval().requires_grad_(False).to(device="cuda:0", dtype=torch.float32)
    with torch.inference_mode():
        videos = torch.rand(1, 2, 240, 320, 3, device="cuda:0")
        latent = model.lam({"videos": videos})["z_rep"]
        assert tuple(latent.shape) == (1, 1, 1, 32), latent.shape
        assert torch.isfinite(latent).all().item()
        torch.cuda.synchronize()
        print(
            "LAM_FORWARD_OK "
            + json.dumps(
                {
                    "input_shape": list(videos.shape),
                    "latent_shape": list(latent.shape),
                    "latent_abs_mean": latent.abs().mean().item(),
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                }
            ),
            flush=True,
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
        print(
            f"NCCL_START rank={rank} torch={torch.__version__} "
            f"nccl={torch.cuda.nccl.version()} device={torch.cuda.get_device_name()} "
            f"NCCL_P2P_DISABLE={os.environ['NCCL_P2P_DISABLE']}",
            flush=True,
        )
        for count in (1, 65536, 4 * 1024 * 1024):
            source = torch.full((count,), float(rank), device="cuda")
            reduced = source.clone()
            dist.all_reduce(reduced)
            assert torch.all(reduced == world * (world - 1) / 2).item()
            gathered = [torch.empty_like(source) for _ in range(world)]
            dist.all_gather(gathered, source)
            assert all(torch.all(value == i).item() for i, value in enumerate(gathered))
            for root in range(world):
                broadcast = source.clone()
                dist.broadcast(broadcast, src=root)
                assert torch.all(broadcast == root).item()
            received = torch.empty_like(source)
            requests = dist.batch_isend_irecv(
                [
                    dist.P2POp(dist.isend, source, (rank + 1) % world),
                    dist.P2POp(dist.irecv, received, (rank - 1) % world),
                ]
            )
            for request in requests:
                request.wait()
            assert torch.all(received == (rank - 1) % world).item()
            torch.cuda.synchronize()
        dist.barrier()
        print(
            f"NCCL_OK rank={rank} all_reduce/all_gather/broadcast/ring_p2p", flush=True
        )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("lam", "nccl"))
    args = parser.parse_args()
    if args.mode == "lam":
        verify_lam()
    else:
        verify_nccl()

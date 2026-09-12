# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Bounded actor-only GPU benchmark; not a replacement RL trainer.

Run with torchrun. Reuses native FSDP setup, actor train_micro_batch, and
optimizer_step. Real reset observations produce cached train-mode chains once.
Advantages are synthetic, zero-mean/nonzero: results say nothing about success.
No WM, Ray scheduling, model offload, checkpoint saving, or production edits.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
from datetime import timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def accumulation_steps(global_batch: int, world_size: int, micro_batch: int) -> int:
    """Reject uneven batches instead of silently changing effective batch size."""
    if min(global_batch, world_size, micro_batch) <= 0:
        raise ValueError("Batch sizes and world size must be positive")
    if global_batch % (world_size * micro_batch):
        raise ValueError(
            "global_batch must divide evenly into world_size * micro_batch"
        )
    return global_batch // world_size // micro_batch


def main() -> None:
    import numpy as np
    import torch
    import torch.distributed as dist
    from omegaconf import OmegaConf

    from rlinf.config import validate_fsdp_cfg
    from rlinf.data.datasets.lerobot_world_model import LeRobotV21InitDataset
    from rlinf.hybrid_engines.fsdp.fsdp_model_manager import FSDPModelManager
    from rlinf.models import get_model
    from rlinf.scheduler import Worker
    from rlinf.utils.nested_dict_process import (
        clone_nested_to_cpu,
        split_dict_to_chunk,
    )
    from rlinf.workers.actor.embodied_fsdp_actor_worker import EmbodiedFSDPActor
    from toolkits.world_model.dreamdojo_validation import config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--micro-batches", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--global-batch", type=int, default=56)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    for micro in args.micro_batches:
        accumulation_steps(args.global_batch, world, micro)
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=False)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    Worker.torch_device_type, Worker.torch_platform = "cuda", torch.cuda
    dist.init_process_group("nccl", timeout=timedelta(minutes=10))
    dist.barrier()

    cfg = config()
    cfg.actor.global_batch_size = args.global_batch
    cfg.actor = validate_fsdp_cfg(cfg.actor)
    local_batch = args.global_batch // world
    seed = int(cfg.actor.seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    class ActorBenchmark(FSDPModelManager):
        # The exact production loss/backward function, without Ray construction.
        train_micro_batch = EmbodiedFSDPActor.train_micro_batch

        def model_provider_func(self):
            model = get_model(cfg.actor.model)
            model.to(self.device)
            model.eval()
            dataset = LeRobotV21InitDataset(
                cfg.env.train.initial_image_path[0],
                episodes=[rank],
                enable_kir=False,
                history_frame_offset=2,
                image_size=(480, 640),
                video_backend="pyav",
            )
            episode = dataset[0]
            first = episode["start_items"][0]
            obs = {
                "main_images": (first["image"].permute(1, 2, 0) * 255)
                .round()
                .clamp(0, 255)
                .to(torch.uint8)[None]
                .repeat(local_batch, 1, 1, 1),
                "wrist_images": None,
                "states": first["observation.state"][None].repeat(local_batch, 1),
                "task_descriptions": [str(episode["task"])] * local_batch,
            }
            torch.manual_seed(seed + rank)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                _, rollout = model.predict_action_batch(obs, mode="train")
            advantages = torch.linspace(-1, 1, local_batch)
            advantages /= advantages.std(unbiased=False)
            self.fixture = {
                "forward_inputs": clone_nested_to_cpu(rollout["forward_inputs"]),
                "prev_logprobs": rollout["prev_logprobs"].detach().cpu(),
                "advantages": advantages[:, None].repeat(
                    1, cfg.actor.model.num_action_chunks
                ),
                "loss_mask": torch.ones(
                    local_batch, cfg.actor.model.num_action_chunks, dtype=torch.bool
                ),
            }
            del rollout
            torch.cuda.empty_cache()
            return model

    actor = ActorBenchmark(cfg.actor, world, rank)
    actor.cfg = cfg
    actor.enable_sft_co_train = False
    actor.setup_model_and_optimizer()
    actor.model.train()
    initial_params = {
        name: p.detach().cpu().clone()
        for name, p in actor.model.named_parameters()
        if p.requires_grad
    }
    initial_optim = clone_nested_to_cpu(actor.optimizer.state_dict())
    initial_scaler = copy.deepcopy(actor.grad_scaler.state_dict())
    results, baseline_gradient = [], None
    if rank == 0:
        metadata = {
            "kind": "actor_only_synthetic_advantages_real_observations",
            "world_size": world,
            "global_batch": args.global_batch,
            "per_rank_batch": local_batch,
            "micro_batches": args.micro_batches,
            "warmups_per_case": 1,
            "repeats": args.repeats,
            "dataset": cfg.env.train.initial_image_path[0],
            "episodes": list(range(world)),
            "raw_frame": 2,
            "visible_gpus": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "actor_config": OmegaConf.to_container(cfg.actor, resolve=True),
            "algorithm": OmegaConf.to_container(cfg.algorithm, resolve=True),
            "input_shapes": {
                k: list(v.shape) for k, v in actor.fixture["forward_inputs"].items()
            },
            "timing": "CPU-to-GPU micro inputs, native forward/backward, empty_cache, native optimizer_step; reset excluded",
        }
        (args.output / "settings.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )
        print("BENCHMARK fixture and FSDP ready", flush=True)

    for micro in args.micro_batches:
        actor.gradient_accumulation = accumulation_steps(
            args.global_batch, world, micro
        )
        cfg.actor.micro_batch_size = micro
        for repeat in range(args.repeats + 1):
            actor.optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                for name, p in actor.model.named_parameters():
                    if name in initial_params:
                        p.copy_(initial_params[name])
            actor.optimizer.load_state_dict(copy.deepcopy(initial_optim))
            actor.grad_scaler.load_state_dict(initial_scaler)
            actor.optimizer_steps = 0
            torch.manual_seed(seed + rank + 1000)
            random.seed(seed + rank + 1000)
            np.random.seed(seed + rank + 1000)
            batches = split_dict_to_chunk(actor.fixture, actor.gradient_accumulation)
            metrics = {}
            torch.cuda.empty_cache()
            dist.barrier()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            for index, batch in enumerate(batches):
                actor.train_micro_batch(
                    batch, metrics, is_last=index + 1 == len(batches)
                )
                batches[index] = None
                del batch
            torch.cuda.synchronize()
            backward_seconds = time.perf_counter() - start
            torch.cuda.empty_cache()  # Same boundary as native run_training.
            grad_norm, _ = actor.optimizer_step()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            allocated = torch.cuda.max_memory_allocated() / 2**30
            reserved = torch.cuda.max_memory_reserved() / 2**30
            gradient = torch.cat(
                [
                    p.grad.detach()
                    .flatten()[:: max(1, p.grad.numel() // 16)][:16]
                    .float()
                    .cpu()
                    for p in actor.model.parameters()
                    if p.grad is not None and p.grad.numel()
                ]
            )
            if baseline_gradient is None:
                baseline_gradient = gradient.clone()
            cosine = torch.nn.functional.cosine_similarity(
                gradient, baseline_gradient, dim=0
            ).item()
            changed = 0
            for name, p in actor.model.named_parameters():
                if name in initial_params:
                    changed += torch.count_nonzero(
                        p.detach().cpu() != initial_params[name]
                    ).item()
            row = {
                "rank": rank,
                "micro_batch": micro,
                "repeat": repeat,
                "warmup": repeat == 0,
                "step_seconds": elapsed,
                "forward_backward_seconds": backward_seconds,
                "peak_allocated_gib": allocated,
                "peak_reserved_gib": reserved,
                "grad_norm": float(grad_norm),
                "sampled_grad_cosine_to_first_case": cosine,
                "changed_local_parameter_elements": changed,
                "loss_unscaled": sum(metrics["actor/total_loss"]),
                "metrics": {k: float(np.mean(v)) for k, v in metrics.items()},
            }
            row["passed"] = bool(
                np.isfinite(row["grad_norm"])
                and row["grad_norm"] > 0
                and np.isfinite(row["loss_unscaled"])
                and torch.isfinite(gradient).all()
                and changed > 0
            )
            results.append(row)
            (args.output / f"rank_{rank}.json").write_text(
                json.dumps(results, indent=2) + "\n"
            )
            if rank == 0:
                print("BENCHMARK " + json.dumps(row), flush=True)
            if not row["passed"]:
                raise RuntimeError(
                    f"Nonfinite/zero gradient or unchanged weights: {row}"
                )
    gathered = [None] * world
    dist.all_gather_object(gathered, results)
    if rank == 0:
        summary = []
        for micro in args.micro_batches:
            rows = [
                r
                for rank_rows in gathered
                for r in rank_rows
                if r["micro_batch"] == micro and not r["warmup"]
            ]
            times = [
                max(r["step_seconds"] for r in rows if r["repeat"] == i)
                for i in range(1, args.repeats + 1)
            ]
            summary.append(
                {
                    "micro_batch": micro,
                    "gradient_accumulation": accumulation_steps(
                        args.global_batch, world, micro
                    ),
                    "median_slowest_rank_step_seconds": float(np.median(times)),
                    "slowest_rank_step_seconds": times,
                    "peak_allocated_gib": max(r["peak_allocated_gib"] for r in rows),
                    "peak_reserved_gib": max(r["peak_reserved_gib"] for r in rows),
                    "grad_norm_range": [
                        min(r["grad_norm"] for r in rows),
                        max(r["grad_norm"] for r in rows),
                    ],
                    "min_sampled_grad_cosine": min(
                        r["sampled_grad_cosine_to_first_case"] for r in rows
                    ),
                    "all_passed": all(r["passed"] for r in rows),
                }
            )
        (args.output / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
        print("BENCHMARK SUMMARY " + json.dumps(summary), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

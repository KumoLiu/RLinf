# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Check the combined runtime, then one real policy/WM/reward transition."""

import gc
import json
import os
import sys
import time
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def main(mode, num_envs=1):
    from cosmos_framework.inference.inference import OmniInference
    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7

    from rlinf.envs.sim.world_model.cosmos3_env import Cosmos3Env
    from rlinf.models.embodiment.gr00t.gr00t_n1d7 import get_model
    from rlinf.scheduler import Worker
    from rlinf.workers.actor.embodied_fsdp_actor_worker import EmbodiedFSDPActor
    from rlinf.workers.env.env_worker import EnvWorker
    from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

    assert all(
        (OmniInference, Gr00tN1d7, EmbodiedFSDPActor, EnvWorker, MultiStepRolloutWorker)
    )
    print(f"IMPORTS_OK torch={torch.__version__} cuda={torch.version.cuda}", flush=True)
    if mode == "imports":
        return
    assert torch.cuda.is_available()
    x = torch.randn(64, 64, device="cuda")
    assert torch.isfinite(x @ x).all()
    print(f"CUDA_OK {torch.cuda.get_device_name()}", flush=True)
    if mode == "cuda":
        return
    config_dir = Path(__file__).resolve().parents[2] / "examples/embodiment/config"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
        cfg = compose(config_name="cosmos3_medical_pads_grpo_gr00t_n1d7")
    cfg.runner.logger.log_path = os.environ["RLINF_RUN_LOG"]
    cfg.env.train.group_size = 1
    cfg.env.train.wm_batch_size = num_envs
    cfg.env.train.video_cfg.save_video = False
    OmegaConf.resolve(cfg)
    Worker.torch_device_type = "cuda"
    started = time.monotonic()
    env = Cosmos3Env(cfg.env.train, num_envs, 0, 1)
    obs, _ = env.reset()
    env.offload()
    policy = get_model(cfg.actor.model)
    policy.cuda()
    policy.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        actions, _ = policy.predict_action_batch(obs, mode="eval")
    actions = torch.as_tensor(actions, device="cuda")
    assert actions.shape == (num_envs, 16, 58) and torch.isfinite(actions).all()
    del policy
    gc.collect()
    torch.cuda.empty_cache()
    env.onload()
    step_started = time.monotonic()
    _, rewards, _, _, _ = env.chunk_step(actions)
    torch.cuda.synchronize()
    assert rewards.shape == (num_envs, 16) and torch.isfinite(rewards).all()
    result = {
        "total_seconds": time.monotonic() - started,
        "chunk_seconds": time.monotonic() - step_started,
        "action_shape": list(actions.shape),
        "reward_shape": list(rewards.shape),
        "reward_mean": rewards.float().mean().item(),
        "score": env.reward_model.scores.tolist(),
        "peak_cuda_gb": torch.cuda.max_memory_allocated() / 1e9,
    }
    Path(os.environ["RLINF_RUN_LOG"], "wm_check.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print("WM_POLICY_REWARD_OK " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 1)

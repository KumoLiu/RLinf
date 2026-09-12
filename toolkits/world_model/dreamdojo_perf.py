# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Bounded, same-input DreamDojo optimization parity and GPU timing checks."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from toolkits.world_model import dreamdojo_validation as validation  # noqa: E402

CASES = {
    "baseline": (False, False, True, True),
    "cache": (True, False, True, True),
    "cache_skip": (True, True, True, True),
    "resident": (True, True, False, True),
    "baseline_resident": (False, False, False, False),
}
_PROGRESS_PATH: Path | None = None


def emit(value: dict) -> None:
    print(json.dumps(value, ensure_ascii=False), flush=True)
    if _PROGRESS_PATH is not None:
        with _PROGRESS_PATH.open("a") as stream:
            stream.write(json.dumps({"unix_time": time.time(), **value}) + "\n")


def seed_all(seed: int) -> None:
    import numpy as np
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)


def configure_case(env, name: str) -> None:
    """Change only explicit inference flags after releasing the previous case."""
    env.offload()
    cache, skip, offload, text_offload = CASES[name]
    env.pipe.cache_text_embeddings = cache
    env.pipe.clear_text_embedding_cache()
    env.pipe.model.inference_skip_zero_guidance = skip
    env.pipe.offload_diffusion_model = offload
    env.pipe.offload_tokenizer = offload
    env.pipe.offload_text_encoder = text_offload
    env.onload()


def build_env(args):
    import torch
    from omegaconf import OmegaConf

    from rlinf.envs.world_model.world_model_dreamdojo_env import DreamDojoEnv
    from rlinf.scheduler import Worker

    Worker.torch_device_type = "cuda"
    Worker.torch_platform = torch.cuda
    cfg = validation.config()
    env_cfg = OmegaConf.create(OmegaConf.to_container(cfg.env.eval))
    env_cfg.initial_image_path = str(args.dataset)
    env_cfg.total_num_envs = len(args.episodes)
    env_cfg.group_size = 1
    env_cfg.eval_unique_episodes = False
    env_cfg.seed = args.seed
    env_cfg.enable_offload = True
    env_cfg.cache_text_embeddings = False
    env_cfg.skip_zero_guidance = False
    for name in ("diffusion_model", "tokenizer", "text_encoder"):
        env_cfg[f"wm_offload_{name}"] = True
    env_cfg.num_inference_steps = args.steps
    env_cfg.max_episode_steps = args.chunks * 12
    env_cfg.max_steps_per_rollout_epoch = args.chunks * 12
    env_cfg.video_cfg.save_video = True
    started = time.perf_counter()
    env = DreamDojoEnv(env_cfg, len(args.episodes), 0, 1)
    OmegaConf.save(env_cfg, args.output / "construction_config.yaml")
    emit({"event": "env_loaded", "seconds": time.perf_counter() - started})
    return env


class StageTimer:
    """Synchronized inclusive timers; nested stages must not be summed."""

    def __init__(self, env):
        self.seconds = defaultdict(float)
        self.calls = defaultdict(int)
        self.last_embeddings = {}
        model = env.pipe.model
        for owner, method, label in (
            (env.pipe, "_get_data_batch_input", "input_and_text"),
            (model, "denoise", "dit_forward"),
            (model, "encode", "vae_encode"),
            (model, "decode", "vae_decode"),
            (model.net, "to", "dit_transfer"),
            (env, "_infer_next_chunk_frames", "wm_total"),
            (env, "_infer_next_chunk_rewards", "reward"),
        ):
            self.wrap(owner, method, label)
        if hasattr(model.tokenizer, "encoder"):
            self.wrap(model.tokenizer.encoder, "to", "vae_encoder_transfer")
            self.wrap(model.tokenizer.decoder, "to", "vae_decoder_transfer")
        else:
            self.wrap(model.tokenizer.model.model, "to", "vae_transfer")
        if model.text_encoder is not None:
            self.wrap(
                model.text_encoder,
                "compute_text_embeddings_online",
                "text_encode_including_onload",
            )
            self.wrap(model.text_encoder.model, "to", "text_transfer")

    def wrap(self, owner, method, label):
        import torch

        original = getattr(owner, method)

        def timed(*args, **kwargs):
            torch.cuda.synchronize()
            started = time.perf_counter()
            result = original(*args, **kwargs)
            torch.cuda.synchronize()
            self.seconds[label] += time.perf_counter() - started
            self.calls[label] += 1
            if label == "input_and_text":
                self.last_embeddings = {
                    k: result[k]
                    for k in ("t5_text_embeddings", "neg_t5_text_embeddings")
                }
            return result

        setattr(owner, method, timed)

    def reset(self):
        self.seconds.clear()
        self.calls.clear()
        self.last_embeddings.clear()


def compare_tensors(reference, current) -> dict:
    import torch

    errors = {}
    for key, old in reference.items():
        new = current[key]
        if old.shape != new.shape:
            raise AssertionError(f"Shape changed for {key}: {old.shape} -> {new.shape}")
        errors[key] = float((old.float() - new.float()).abs().max())
        torch.testing.assert_close(
            new, old, rtol=0, atol=0, msg=lambda msg: f"{key}: {msg}"
        )
    return errors


def micro(args):
    import numpy as np
    import torch

    episodes = validation.load_episodes(args.dataset, args.episodes)
    env = build_env(args)
    timer = StageTimer(env)
    actions = torch.tensor(
        np.stack([e["action"][2:14] for e in episodes]), device="cuda"
    )
    reference = None
    results = {}
    for case in args.cases:
        configure_case(env, case)
        samples = []
        for rep in range(args.repeats + 1):
            seed_all(args.seed)
            env.reset(episode_indices=args.episodes)
            timer.reset()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                _, rewards, _, _, _ = env.chunk_step(actions)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            sample = {
                "rep": rep,
                "warmup": rep == 0,
                "chunk_seconds": elapsed,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
                "stage_seconds_inclusive": dict(timer.seconds),
                "stage_calls": dict(timer.calls),
            }
            tensors = {
                "frames": env.current_obs.cpu().clone(),
                "rewards": rewards.cpu().clone(),
                "probabilities": env.last_chunk_probs.cpu().clone(),
                "cpu_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(),
                **{k: v.cpu().clone() for k, v in timer.last_embeddings.items()},
            }
            if reference is None:
                reference = tensors
                torch.save(reference, args.output / "reference.pt")
            sample["max_abs_error"] = compare_tensors(reference, tensors)
            expected = args.steps if CASES[case][1] else args.steps * 2
            if timer.calls["dit_forward"] != expected:
                raise AssertionError(
                    f"Expected {expected} denoise calls, got {timer.calls}"
                )
            samples.append(sample)
            emit({"case": case, **sample})
            validation.save_json(args.output / f"{case}.json", samples)
        results[case] = {
            "flags": dict(
                zip(
                    (
                        "cache",
                        "skip_zero_guidance",
                        "per_chunk_offload",
                        "text_offload",
                    ),
                    CASES[case],
                )
            ),
            "steady_mean_seconds": statistics.mean(
                s["chunk_seconds"] for s in samples[1:]
            ),
            "samples": samples,
        }
        validation.save_json(args.output / "results.json", results)
    env.offload()


def policy(args):
    import numpy as np
    import torch

    from rlinf.models.embodiment.gr00t.gr00t_n1d7 import get_model

    cfg = validation.config()
    model = get_model(cfg.actor.model)
    model.to("cuda")
    # GR00T's eval override returns None rather than self.
    model.eval()
    episodes = validation.load_episodes(args.dataset, args.episodes)
    env = build_env(args)
    reference = None
    results = {}
    for case in args.cases:
        configure_case(env, case)
        seed_all(args.seed)
        obs, _ = env.reset(episode_indices=args.episodes)
        images = [env.capture_image().cpu().numpy()]
        rewards, probs, states, actions, durations = [], [], [], [], []
        torch.cuda.reset_peak_memory_stats()
        for chunk in range(args.chunks):
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                action, _ = model.predict_action_batch(obs, mode="eval")
                obs_list, reward, _, _, _ = env.chunk_step(action)
            torch.cuda.synchronize()
            durations.append(time.perf_counter() - started)
            obs = obs_list[-1]
            images.append(env.capture_image().cpu().numpy())
            rewards.append(reward.cpu().numpy())
            probs.append(env.last_chunk_probs.cpu().numpy())
            states.append(env.current_state.cpu().numpy())
            actions.append(np.asarray(action))
            emit(
                {
                    "case": case,
                    "chunk": chunk + 1,
                    "seconds": durations[-1],
                    "stage": env.reward_model.stage.tolist(),
                }
            )
        tensors = {
            "pixels_uint8": torch.from_numpy(np.concatenate(images, axis=1)),
            "actions": torch.from_numpy(np.stack(actions)),
            "probabilities": torch.from_numpy(np.stack(probs)),
            "rewards": torch.from_numpy(np.stack(rewards)),
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(),
        }
        errors = {} if reference is None else compare_tensors(reference, tensors)
        if reference is None:
            reference = tensors
        results[case] = {
            "metrics": validation.finish_video(
                args, case, episodes, images, rewards, probs, states, actions
            ),
            "chunk_seconds": durations,
            "steady_mean_seconds": statistics.mean(durations[1:]),
            "max_abs_error": errors,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
        }
        validation.save_json(args.output / "results.json", results)
    env.offload()


def main():
    global _PROGRESS_PATH
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("micro", "policy"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(os.environ.get("DREAMDOJO_DATA_ROOT", REPO.parent / "data"))
        / "pick_trocar_teleop_success_validation",
    )
    parser.add_argument("--episodes", nargs="+", type=int, default=list(range(8)))
    parser.add_argument("--steps", type=int, default=35)
    parser.add_argument("--chunks", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=tuple(CASES),
        default=["baseline", "cache", "cache_skip", "resident"],
    )
    args = parser.parse_args()
    if min(args.steps, args.chunks, args.repeats) < 1:
        parser.error("steps, chunks and repeats must be positive")
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if len(devices) != 1 or not devices[0].strip():
        parser.error("Bind one allocated, healthy GPU using CUDA_VISIBLE_DEVICES")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    _PROGRESS_PATH = args.output / "progress.jsonl"
    validation.save_json(
        args.output / "arguments.json",
        {
            **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    )
    started = time.perf_counter()
    try:
        globals()[args.phase](args)
    except BaseException:
        (args.output / "error.txt").write_text(traceback.format_exc())
        raise
    finally:
        validation.save_json(
            args.output / "timing.json", {"wall_seconds": time.perf_counter() - started}
        )


if __name__ == "__main__":
    main()

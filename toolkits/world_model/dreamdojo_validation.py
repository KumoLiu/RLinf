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

"""Recorded-action, policy and reward diagnostics for the DreamDojo integration.

Run from RLinf with its virtualenv. Each phase writes to a new output directory.
Training and full-dataset evaluation use the native RLinf runner, not this tool.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WORKSPACE = REPO.parent
sys.path.insert(0, str(REPO))


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def config():
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    os.environ["EMBODIED_PATH"] = str(REPO / "examples/embodiment")
    with initialize_config_dir(
        config_dir=str(REPO / "examples/embodiment/config"), version_base="1.1"
    ):
        cfg = compose(config_name="dreamdojo_trocar_grpo_gr00t_n1d7")
    OmegaConf.resolve(cfg)
    # Resolve relative defaults against RLinf, independent of the caller's cwd.
    for part in (cfg.env.train, cfg.env.eval):
        for key in ("dreamdojo_root", "model_path", "action_statistics"):
            part[key] = str((REPO / Path(part[key]).expanduser()).resolve())
        part.reward_model.from_pretrained = str(
            (REPO / Path(part.reward_model.from_pretrained).expanduser()).resolve()
        )
        part.initial_image_path = [
            str((REPO / Path(path).expanduser()).resolve())
            for path in part.initial_image_path
        ]
    for model in (cfg.actor.model, cfg.rollout.model):
        model.model_path = str((REPO / Path(model.model_path).expanduser()).resolve())
    return cfg


def native_functions():
    """Execute native action transforms without model imports/CUDA metadata.

    Video interpolation and CUDA metadata are stubbed; action/state code is
    the actual DreamDojo source, not a second copy of the bridge formula.
    """
    from types import SimpleNamespace

    import torch
    from einops import rearrange

    root = Path(config().env.eval.dreamdojo_root)
    source = root / "groot_dreams/data/transform/state_action.py"
    tree = ast.parse(source.read_text())
    normalizer = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Normalizer"
    )
    ns = {"torch": torch}
    exec(
        compile(ast.Module(body=[normalizer], type_ignores=[]), str(source), "exec"), ns
    )
    source = root / "groot_dreams/data/dataset.py"
    tree = ast.parse(source.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "WrappedLeRobotSingleDataset"
    )
    fn = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "__getitem__"
    )
    ns.update(
        F=SimpleNamespace(interpolate=lambda frames, *a, **kw: frames),
        rearrange=rearrange,
    )
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), "exec"), ns)
    return ns["Normalizer"], ns["__getitem__"]


def parity(args):
    from types import SimpleNamespace
    from unittest.mock import patch

    import numpy as np
    import pyarrow.parquet as pq
    import torch

    from rlinf.data.datasets.lerobot_world_model import LeRobotV21InitDataset
    from rlinf.envs.world_model.dreamdojo_adapters import (
        G1DreamDojoActionBridge,
        pad_g1_dex3_28_to_43,
    )

    stats_path = Path(config().env.eval.action_statistics)
    stats = json.loads(stats_path.read_text())
    Normalizer, getitem = native_functions()
    action_norm = Normalizer("min_max", stats["action"])
    state_norm = Normalizer("min_max", stats["observation.state"])
    bridge = G1DreamDojoActionBridge(stats_path, "cpu")
    worst = 0.0
    windows = failures = 0
    with patch.object(torch.Tensor, "cuda", lambda self, *a, **kw: self):
        for f in sorted((args.dataset / "data").glob("*/*.parquet")):
            t = pq.read_table(f, columns=["action", "observation.state"])
            a = torch.tensor(t["action"].to_pylist(), dtype=torch.float32)
            s = torch.tensor(t["observation.state"].to_pylist(), dtype=torch.float32)
            normalized_a = action_norm.forward(pad_g1_dex3_28_to_43(a))
            normalized_s = state_norm.forward(pad_g1_dex3_28_to_43(s))
            for start in range(0, len(a) - 24, 12):
                sampled = normalized_a[start : start + 25 : 2]
                raw = {
                    "video": np.zeros((14, 1, 3, 2, 2), np.float32),
                    "action": torch.cat([sampled, sampled[-1:]]),
                    "state": normalized_s[start : start + 1],
                }
                stub = SimpleNamespace(
                    all_steps=[(0, 0)],
                    transforms=lambda x: x,
                    get_step_data=lambda *a: raw,
                    dataset_path="g1",
                    dataset_name="g1",
                    num_frames=13,
                )
                expected = getitem(stub, 0)["action"]
                actual = bridge.encode_30hz_actions(a[start : start + 25], s[start])
                delta = float((actual - expected).abs().max())
                worst = max(worst, delta)
                failures += not torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
                windows += 1
                if windows == 1:
                    np.savez(
                        args.output / "example.npz",
                        raw_action=a[start : start + 25].numpy(),
                        state=s[start].numpy(),
                        native=expected.numpy(),
                        bridge=actual.numpy(),
                    )
    ds = LeRobotV21InitDataset(
        str(args.dataset), enable_kir=False, kir_context_len=0, history_frame_offset=2
    )
    sample = ds[0]
    assert sample["dataset_meta"]["start_frame"] == 2
    row = pq.read_table(args.dataset / "data/chunk-000/episode_000000.parquet")
    first = sample["start_items"][0]
    torch.testing.assert_close(
        first["wm_previous_action"], torch.tensor(row["action"][0].as_py())
    )
    torch.testing.assert_close(
        first["wm_previous_state"], torch.tensor(row["observation.state"][0].as_py())
    )
    torch.testing.assert_close(
        first["observation.state"], torch.tensor(row["observation.state"][2].as_py())
    )
    result = {
        "windows": windows,
        "failures": failures,
        "max_abs_error": worst,
        "reset_image_frame": 2,
        "baseline_frame": 0,
        "dataset": str(args.dataset),
    }
    save_json(args.output / "results.json", result)
    print(json.dumps(result), flush=True)
    assert failures == 0


def load_episodes(dataset, indices):
    import cv2
    import numpy as np
    import pyarrow.parquet as pq

    result = []
    for index in indices:
        chunk = index // 1000
        table = pq.read_table(
            dataset / f"data/chunk-{chunk:03d}/episode_{index:06d}.parquet"
        )
        path = (
            dataset
            / f"videos/chunk-{chunk:03d}/observation.images.cam_head/episode_{index:06d}.mp4"
        )
        cap = cv2.VideoCapture(str(path))
        frames = []
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        cap.release()
        result.append(
            {
                "index": index,
                "video": np.stack(frames),
                "action": np.array(table["action"].to_pylist(), np.float32),
                "state": np.array(table["observation.state"].to_pylist(), np.float32),
            }
        )
    return result


def make_env(args, cfg):
    import torch
    from omegaconf import OmegaConf

    from rlinf.envs.world_model.world_model_dreamdojo_env import DreamDojoEnv
    from rlinf.scheduler import Worker

    Worker.torch_device_type = "cuda"
    Worker.torch_platform = torch.cuda
    env_cfg = OmegaConf.create(OmegaConf.to_container(cfg.env.eval))
    env_cfg.initial_image_path = str(args.dataset)
    env_cfg.seed = args.seed
    env_cfg.enable_offload = False
    # Standalone diagnostics pass explicit subsets, not native eval shards.
    env_cfg.eval_unique_episodes = False
    env_cfg.total_num_envs = len(args.episodes)
    env_cfg.video_cfg.save_video = True
    env_cfg.max_episode_steps = args.chunks * 12
    env_cfg.num_inference_steps = args.steps[-1]
    env = DreamDojoEnv(env_cfg, len(args.episodes), 0, 1)
    OmegaConf.save(env_cfg, args.output / "env_config.yaml")
    return env


def finish_video(args, case, episodes, images, rewards, probs, states, actions):
    import imageio.v2 as imageio
    import numpy as np

    directory = args.output / case
    directory.mkdir()
    video = np.concatenate(images, axis=1)
    reward = np.concatenate(rewards, axis=1)
    probability = np.concatenate(probs, axis=1)
    records = []
    for i, episode in enumerate(episodes):
        imageio.mimwrite(
            directory / f"episode_{episode['index']:03d}.mp4",
            video[i],
            fps=15,
            codec="libx264",
            quality=8,
        )
        record = {
            "episode": episode["index"],
            "return": float(reward[i].sum()),
            "stage": int(round(reward[i].sum())),
            "probability_peaks": probability[i].max(axis=0).tolist(),
        }
        if case.startswith("gt_"):
            target_idx = np.arange(2, 2 + video.shape[1] * 2, 2)
            valid = target_idx < len(episode["video"])
            target = episode["video"][target_idx[valid]]
            diff = video[i, valid].astype(np.float32) - target.astype(np.float32)
            # Exclude the real condition image from quality averages.
            per_frame = (diff[1:] ** 2).mean(axis=(1, 2, 3))
            record.update(
                mse=float(per_frame.mean()),
                psnr=float(10 * np.log10(255**2 / max(float(per_frame.mean()), 1e-9))),
                per_frame_mse=per_frame.tolist(),
                valid_future_frames=len(per_frame),
            )
        records.append(record)
    np.savez_compressed(
        directory / "trace.npz",
        rewards=reward,
        probabilities=probability,
        states=np.stack(states),
        actions=np.stack(actions),
    )
    save_json(directory / "metrics.json", records)
    print(
        json.dumps(
            {
                "case": case,
                "metrics": [
                    {k: v for k, v in r.items() if k != "per_frame_mse"}
                    for r in records
                ],
            }
        ),
        flush=True,
    )
    return records


def world_model(args):
    import numpy as np
    import torch

    cfg = config()
    episodes = load_episodes(args.dataset, args.episodes)
    env = make_env(args, cfg)
    all_results = {}
    for steps in args.steps:
        for mode in ("teacher_forced", "closed_loop"):
            case = f"gt_{mode}_{steps}steps"
            env.reset(episode_indices=args.episodes)
            env.num_inference_steps = steps
            images = [env.capture_image().cpu().numpy()]
            rewards, probs, states, actions = [], [], [], []
            for n in range(args.chunks):
                t = 2 + 12 * n
                batch_actions = torch.tensor(
                    np.stack(
                        [
                            e["action"][
                                np.minimum(np.arange(t, t + 12), len(e["action"]) - 1)
                            ]
                            for e in episodes
                        ]
                    ),
                    device="cuda",
                )
                # Oracle recorded history isolates the WM from the proprio proxy.
                env.previous_action = torch.tensor(
                    np.stack(
                        [
                            e["action"][min(t - 2, len(e["action"]) - 1)]
                            for e in episodes
                        ]
                    ),
                    device="cuda",
                )
                env.previous_state = torch.tensor(
                    np.stack(
                        [e["state"][min(t - 2, len(e["state"]) - 1)] for e in episodes]
                    ),
                    device="cuda",
                )
                if mode == "teacher_forced":
                    frames = torch.tensor(
                        np.stack(
                            [e["video"][min(t, len(e["video"]) - 1)] for e in episodes]
                        ),
                        device="cuda",
                    )
                    env.current_obs = (frames.permute(0, 3, 1, 2).float() / 127.5 - 1)[
                        :, :, None, None
                    ]
                with torch.no_grad():
                    _, r, _, _, _ = env.chunk_step(batch_actions)
                images.append(env.capture_image().numpy())
                rewards.append(r.cpu().numpy())
                probs.append(env.last_chunk_probs.cpu().numpy())
                states.append(env.previous_state.cpu().numpy())
                actions.append(batch_actions.cpu().numpy())
                print(
                    json.dumps({"case": case, "chunk": n + 1, "of": args.chunks}),
                    flush=True,
                )
            all_results[case] = finish_video(
                args, case, episodes, images, rewards, probs, states, actions
            )
    # Score real videos with exactly the deployed 15->30 fps reward adapter.
    env.reward_model.reset()
    length = max(len(e["video"][2::2]) for e in episodes)
    real = np.stack(
        [
            e["video"][np.minimum(2 + 2 * np.arange(length), len(e["video"]) - 1)]
            for e in episodes
        ]
    )
    with torch.no_grad():
        rewards, probabilities = env.reward_model.score_chunk(
            torch.from_numpy(real).permute(0, 1, 4, 2, 3)
        )
    all_results["real_reward"] = {
        "returns": rewards.sum(dim=1).cpu().tolist(),
        "stages": env.reward_model.stage.cpu().tolist(),
    }
    save_json(args.output / "results.json", all_results)


def checkpoint_case(checkpoint):
    """Name new checkpoints without mislabeling them as the historical step20."""
    if checkpoint == "base":
        return "base"
    for part in Path(checkpoint).parts:
        if part.startswith("global_step_") and part[12:].isdigit():
            return part
    return "checkpoint_" + hashlib.sha256(str(checkpoint).encode()).hexdigest()[:12]


def policy(args):
    import numpy as np
    import torch
    from omegaconf import OmegaConf

    from rlinf.models.embodiment.gr00t.gr00t_n1d7 import get_model

    cfg = config()
    model = get_model(cfg.actor.model)
    model.to("cuda")
    model.eval()
    OmegaConf.save(cfg.actor.model, args.output / "policy_config.yaml")
    episodes = load_episodes(args.dataset, args.episodes)
    env = make_env(args, cfg)
    results = {}
    for checkpoint in args.checkpoints:
        if checkpoint != "base":
            state = torch.load(
                checkpoint, map_location="cpu", mmap=True, weights_only=True
            )
            model.load_state_dict(state, strict=True)
            del state
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        obs, _ = env.reset(episode_indices=args.episodes)
        images = [env.capture_image().cpu().numpy()]
        rewards, probs, states, actions = [], [], [], []
        case = checkpoint_case(checkpoint)
        for n in range(args.chunks):
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                action, _ = model.predict_action_batch(obs, mode="eval")
                next_obs, r, _, _, _ = env.chunk_step(action)
            obs = next_obs[-1]
            images.append(env.capture_image().numpy())
            rewards.append(r.cpu().numpy())
            probs.append(env.last_chunk_probs.cpu().numpy())
            states.append(env.current_state.cpu().numpy())
            actions.append(np.asarray(action))
            print(
                json.dumps(
                    {
                        "case": case,
                        "chunk": n + 1,
                        "of": args.chunks,
                        "stage": env.reward_model.stage.tolist(),
                    }
                ),
                flush=True,
            )
        results[case] = finish_video(
            args, case, episodes, images, rewards, probs, states, actions
        )
    save_json(args.output / "results.json", results)


def real_reward(args):
    """Score recorded frames on the exact future-frame timeline used by WM."""
    import numpy as np
    import torch

    from rlinf.envs.world_model.dreamdojo_reward import BatchedMilestoneReward

    cfg = config()
    episodes = load_episodes(args.dataset, args.episodes)
    scorer = BatchedMilestoneReward(
        cfg.env.eval.reward_model.from_pretrained, len(episodes), "cuda"
    )
    times = 2 + 2 * np.arange(args.chunks * 6 + 1)
    videos = np.stack(
        [e["video"][np.minimum(times, len(e["video"]) - 1)] for e in episodes]
    )
    with torch.no_grad():
        r, probabilities = scorer.score_chunk(
            torch.from_numpy(videos[:, 1:]).permute(0, 1, 4, 2, 3)
        )
    reward = np.zeros((len(episodes), args.chunks * 12), np.float32)
    reward[:, 1::2] = r.cpu().numpy()
    actions, states = [], []
    for n in range(args.chunks):
        t = 2 + 12 * n
        actions.append(
            np.stack(
                [
                    e["action"][np.minimum(np.arange(t, t + 12), len(e["action"]) - 1)]
                    for e in episodes
                ]
            )
        )
        states.append(
            np.stack([e["state"][min(t + 12, len(e["state"]) - 1)] for e in episodes])
        )
    records = finish_video(
        args,
        "real_reward",
        episodes,
        [videos],
        [reward],
        [probabilities.cpu().numpy()],
        states,
        actions,
    )
    save_json(args.output / "results.json", {"real_reward": records})


def report(args):
    """Plot stored probabilities and action trends without rerunning models."""
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = {}
    probability_series = {}
    for root in args.inputs:
        for metrics_path in sorted(root.glob("*/metrics.json")):
            case = metrics_path.parent.name
            label = f"{root.name}_{case}"
            records = json.loads(metrics_path.read_text())
            trace = np.load(metrics_path.parent / "trace.npz")
            rewards = trace["rewards"]
            probabilities = trace["probabilities"]
            if case.startswith("gt_") or case == "real_reward":
                probability_series[label] = probabilities.copy()
            stages = np.cumsum(rewards, axis=1)
            actions = trace["actions"].transpose(1, 0, 2, 3)
            actions = actions.reshape(len(records), -1, actions.shape[-1])
            result = {
                "source": str(metrics_path.resolve()),
                "returns": rewards.sum(axis=1).tolist(),
                "reward_model_success_fraction": float((stages[:, -1] >= 3).mean()),
                "milestone_seconds": [],
                "action_min_per_joint": actions.min(axis=(0, 1)).tolist(),
                "action_max_per_joint": actions.max(axis=(0, 1)).tolist(),
                "mean_absolute_action_step": np.abs(np.diff(actions, axis=1))
                .mean(axis=(0, 1))
                .tolist(),
            }
            for trajectory in stages:
                result["milestone_seconds"].append(
                    [
                        float((np.flatnonzero(trajectory >= stage)[0] + 1) / 30)
                        if np.any(trajectory >= stage)
                        else None
                        for stage in (1, 2, 3)
                    ]
                )
            if "psnr" in records[0]:
                result["mean_psnr_db"] = float(np.mean([r["psnr"] for r in records]))
                result["mean_mse"] = float(np.mean([r["mse"] for r in records]))
            fig, axes = plt.subplots(
                len(records), 1, figsize=(10, 2.7 * len(records)), squeeze=False
            )
            for i, record in enumerate(records):
                ax = axes[i, 0]
                for head, name in enumerate(("pick", "handover", "place")):
                    ax.plot(
                        (np.arange(probabilities.shape[1]) + 1) / 15,
                        probabilities[i, :, head],
                        label=name,
                    )
                ax.axhline(0.8, color="gray", linestyle=":", label="threshold")
                ax.set(
                    title=f"{case}, episode {record['episode']}, reward={result['returns'][i]}",
                    xlabel="seconds",
                    ylabel="probability",
                    ylim=(-0.05, 1.05),
                )
                ax.legend(loc="lower right", ncol=4)
            fig.tight_layout()
            fig.savefig(args.output / f"{label}_probabilities.png", dpi=130)
            plt.close(fig)
            fig, axes = plt.subplots(
                len(records), 4, figsize=(16, 2.8 * len(records)), squeeze=False
            )
            for i, record in enumerate(records):
                for group, name in enumerate(
                    ("left arm", "right arm", "left hand", "right hand")
                ):
                    ax = axes[i, group]
                    ax.plot(
                        (np.arange(actions.shape[1]) + 1) / 30,
                        actions[i, :, group * 7 : (group + 1) * 7],
                    )
                    ax.set(
                        title=f"ep {record['episode']}: {name}",
                        xlabel="seconds",
                        ylabel="physical joint target",
                    )
            fig.tight_layout()
            fig.savefig(args.output / f"{label}_actions.png", dpi=130)
            plt.close(fig)
            summary[label] = result
    if not summary:
        raise ValueError(
            "No completed metrics.json and trace.npz cases found in --inputs"
        )
    if len(probability_series) > 1:
        count = min(values.shape[0] for values in probability_series.values())
        fig, axes = plt.subplots(count, 3, figsize=(13, count * 2.7), squeeze=False)
        for i in range(count):
            for head, name in enumerate(("pick", "handover", "place")):
                ax = axes[i, head]
                for label, values in probability_series.items():
                    ax.plot(
                        (np.arange(values.shape[1]) + 1) / 15,
                        values[i, :, head],
                        label=label,
                    )
                ax.axhline(0.8, color="gray", linestyle=":")
                ax.set(
                    title=f"trajectory {i}: {name}",
                    xlabel="seconds",
                    ylim=(-0.05, 1.05),
                )
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", fontsize=7, ncol=2)
        fig.tight_layout(rect=(0, 0.09, 1, 1))
        fig.savefig(args.output / "recorded_vs_generated_probabilities.png", dpi=140)
        plt.close(fig)
    save_json(args.output / "summary.json", summary)
    print(
        json.dumps(
            {
                k: {
                    p: v[p]
                    for p in (
                        "returns",
                        "reward_model_success_fraction",
                        "milestone_seconds",
                    )
                }
                for k, v in summary.items()
            }
        ),
        flush=True,
    )


def training_report(args):
    """Export exact TensorBoard scalars and saved checkpoint locations."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    result = {}
    for root in args.inputs:
        # Per-worker logging puts the aggregate stream under tensorboard/all;
        # older single-stream diagnostics wrote directly under tensorboard.
        event_dir = root / "tensorboard"
        if (event_dir / "all").is_dir():
            event_dir = event_dir / "all"
        events = EventAccumulator(str(event_dir), size_guidance={"scalars": 0})
        events.Reload()
        scalars = {
            tag: [
                {"step": event.step, "wall_time": event.wall_time, "value": event.value}
                for event in events.Scalars(tag)
            ]
            for tag in events.Tags()["scalars"]
        }
        bad = [
            tag
            for tag, values in scalars.items()
            if any(not math.isfinite(v["value"]) for v in values)
        ]
        checkpoints = [
            {"path": str(p.resolve()), "bytes": p.stat().st_size}
            for p in sorted(root.glob("**/full_weights.pt"))
        ]
        result[root.name] = {
            "scalars": scalars,
            "non_finite_tags": bad,
            "checkpoints": checkpoints,
        }
        print(
            json.dumps(
                {
                    "run": root.name,
                    "non_finite_tags": bad,
                    "metrics": {
                        tag: values
                        for tag, values in scalars.items()
                        if any(
                            key in tag
                            for key in (
                                "success_once",
                                "grad_norm",
                                "advantages_max",
                                "advantages_min",
                            )
                        )
                    },
                }
            ),
            flush=True,
        )
    if not result:
        raise ValueError("Specify a completed training directory using --inputs")
    save_json(args.output / "results.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=[
            "parity",
            "world_model",
            "policy",
            "report",
            "real_reward",
            "training_report",
        ],
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(os.environ.get("DREAMDOJO_DATA_ROOT", WORKSPACE / "data"))
        / "pick_trocar_teleop_success_validation",
    )
    parser.add_argument("--episodes", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--steps", type=int, nargs="+", default=[5, 35])
    parser.add_argument("--chunks", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoints", nargs="+", default=["base"])
    parser.add_argument("--inputs", type=Path, nargs="+", default=[])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    args.output = args.output.resolve()
    save_json(
        args.output / "arguments.json",
        json.loads(json.dumps(vars(args), default=str)),
    )
    start = time.monotonic()
    try:
        globals()[args.phase](args)
    except BaseException:
        (args.output / "error.txt").write_text(traceback.format_exc())
        raise
    finally:
        save_json(
            args.output / "timing.json", {"wall_seconds": time.monotonic() - start}
        )


if __name__ == "__main__":
    main()

"""Build train/val .pt datasets from real rollouts + teleop head videos.

Usage:
    .venv_success_cls/bin/python -m sim2real.components.success_classifier.prepare_dataset
"""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from tqdm import tqdm

EPISODE_DIR_RE = re.compile(r"^episode_\d+_(success|fail)$")


@dataclass
class EpisodeSample:
    """One labeled episode with a path to its head-camera video."""

    episode_id: str
    video_path: Path
    label: int  # 1 = success, 0 = fail
    source: str  # "real" | "teleop"


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[3]
    p = argparse.ArgumentParser(description="Prepare success/fail classifier dataset")
    p.add_argument(
        "--real-rollouts-dir",
        type=Path,
        default=repo / "logs" / "real_rollouts",
        help="Root containing session folders with episode_*_{success,fail}",
    )
    p.add_argument(
        "--teleop-dataset-dir",
        type=Path,
        default=Path.home() / "workspaces" / "yunl" / "datasets" / "pick_trocar",
        help="LeRobot teleop dataset root (all episodes treated as success)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=repo / "logs" / "success_classifier" / "processed",
        help="Where to write train.pt / val.pt",
    )
    p.add_argument("--num-last-frames", type=int, default=8, help="Frames from end of each episode")
    p.add_argument("--val-split", type=float, default=0.2, help="Fraction of episodes for validation")
    p.add_argument(
        "--fail-success-ratio",
        type=float,
        default=2.0,
        help="Target fail:success frame ratio in train (and val after split)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--skip-teleop",
        action="store_true",
        help="Only use real_rollouts (debug)",
    )
    return p.parse_args()


def discover_real_episodes(root: Path) -> list[EpisodeSample]:
    if not root.is_dir():
        raise FileNotFoundError(f"real rollouts dir not found: {root}")

    episodes: list[EpisodeSample] = []
    for session in sorted(root.iterdir()):
        if not session.is_dir():
            continue
        for ep_dir in sorted(session.iterdir()):
            if not ep_dir.is_dir() or not EPISODE_DIR_RE.match(ep_dir.name):
                continue
            label = 1 if ep_dir.name.endswith("_success") else 0
            video = ep_dir / "color_0.mp4"
            if not video.is_file():
                print(f"[warn] missing {video}, skip")
                continue
            episodes.append(
                EpisodeSample(
                    episode_id=f"real/{session.name}/{ep_dir.name}",
                    video_path=video,
                    label=label,
                    source="real",
                )
            )
    return episodes


def discover_teleop_episodes(dataset_dir: Path) -> list[EpisodeSample]:
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"teleop dataset not found: {dataset_dir}")

    videos = sorted(dataset_dir.glob("videos/**/observation.images.cam_head/*.mp4"))
    episodes: list[EpisodeSample] = []
    for vp in videos:
        episodes.append(
            EpisodeSample(
                episode_id=f"teleop/{vp.stem}",
                video_path=vp,
                label=1,
                source="teleop",
            )
        )
    return episodes


def read_last_frames(video_path: Path, k: int) -> list[torch.Tensor]:
    """Decode the last ``k`` frames as uint8 CHW RGB tensors."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")

    frames_bgr: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames_bgr.append(frame)
    cap.release()

    if not frames_bgr:
        raise RuntimeError(f"empty video: {video_path}")

    selected = frames_bgr[-k:] if len(frames_bgr) >= k else frames_bgr
    out: list[torch.Tensor] = []
    for bgr in selected:
        # Normalize spatial size so real (320x240) and teleop frames collate.
        bgr = cv2.resize(bgr, (320, 240), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # HWC uint8 -> CHW uint8
        chw = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).contiguous()
        out.append(chw)
    return out


def stratified_episode_split(
    episodes: list[EpisodeSample],
    val_split: float,
    seed: int,
) -> tuple[list[EpisodeSample], list[EpisodeSample]]:
    """Split episodes into train/val, stratified by label when possible."""
    rng = random.Random(seed)
    by_label: dict[int, list[EpisodeSample]] = {0: [], 1: []}
    for ep in episodes:
        by_label[ep.label].append(ep)

    train: list[EpisodeSample] = []
    val: list[EpisodeSample] = []
    for label, group in by_label.items():
        group = list(group)
        rng.shuffle(group)
        if not group:
            continue
        n_val = max(1, int(round(len(group) * val_split))) if len(group) >= 5 else max(1, len(group) // 5 or 1)
        # Keep at least one train episode per class when possible.
        if len(group) == 1:
            train.extend(group)
            continue
        n_val = min(n_val, len(group) - 1)
        val.extend(group[:n_val])
        train.extend(group[n_val:])
        print(f"[split] label={label}: train={len(group) - n_val} val={n_val}")
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def extract_frames(
    episodes: Iterable[EpisodeSample], k: int
) -> tuple[list[torch.Tensor], list[int], list[str]]:
    images: list[torch.Tensor] = []
    labels: list[int] = []
    episode_ids: list[str] = []
    for ep in tqdm(list(episodes), desc="decode"):
        try:
            frames = read_last_frames(ep.video_path, k)
        except Exception as e:
            print(f"[warn] skip {ep.episode_id}: {e}")
            continue
        for fr in frames:
            images.append(fr)
            labels.append(int(ep.label))
            episode_ids.append(ep.episode_id)
    return images, labels, episode_ids


def balance_frames(
    images: list[torch.Tensor],
    labels: list[int],
    episode_ids: list[str],
    fail_success_ratio: float,
    seed: int,
) -> tuple[list[torch.Tensor], list[int], list[str]]:
    """Downsample the majority class to approximate fail:success ratio."""
    rng = random.Random(seed)
    success_idx = [i for i, y in enumerate(labels) if y == 1]
    fail_idx = [i for i, y in enumerate(labels) if y == 0]

    if not success_idx or not fail_idx:
        print("[warn] one class missing; skipping balance")
        return images, labels, episode_ids

    n_success = len(success_idx)
    target_fail = int(round(n_success * fail_success_ratio))
    if len(fail_idx) > target_fail:
        fail_idx = rng.sample(fail_idx, target_fail)
    elif len(fail_idx) < target_fail and n_success > 0:
        # Not enough fails — downsample success instead so ratio holds.
        target_success = max(1, int(round(len(fail_idx) / fail_success_ratio)))
        if n_success > target_success:
            success_idx = rng.sample(success_idx, target_success)

    keep = sorted(success_idx + fail_idx)
    rng.shuffle(keep)
    return (
        [images[i] for i in keep],
        [labels[i] for i in keep],
        [episode_ids[i] for i in keep],
    )


def save_payload(
    path: Path,
    images: list[torch.Tensor],
    labels: list[int],
    episode_ids: list[str],
    extra_meta: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "images": images,
        "labels": [int(v) for v in labels],
        "metadata": {
            **extra_meta,
            "num_frames": len(images),
            "num_success": int(sum(labels)),
            "num_fail": int(len(labels) - sum(labels)),
            "episode_ids": episode_ids,
        },
    }
    torch.save(payload, path)
    print(
        f"[save] {path}  frames={len(images)}  "
        f"success={sum(labels)} fail={len(labels) - sum(labels)}"
    )


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    real_eps = discover_real_episodes(args.real_rollouts_dir)
    teleop_eps = [] if args.skip_teleop else discover_teleop_episodes(args.teleop_dataset_dir)
    all_eps = real_eps + teleop_eps

    n_real_s = sum(1 for e in real_eps if e.label == 1)
    n_real_f = sum(1 for e in real_eps if e.label == 0)
    print(
        f"[discover] real episodes: {len(real_eps)} "
        f"(success={n_real_s}, fail={n_real_f}); "
        f"teleop success episodes: {len(teleop_eps)}; total={len(all_eps)}"
    )
    if not all_eps:
        raise SystemExit("no episodes found")

    train_eps, val_eps = stratified_episode_split(all_eps, args.val_split, args.seed)
    print(f"[split] episodes train={len(train_eps)} val={len(val_eps)}")

    train_imgs, train_labs, train_ids = extract_frames(train_eps, args.num_last_frames)
    val_imgs, val_labs, val_ids = extract_frames(val_eps, args.num_last_frames)

    train_imgs, train_labs, train_ids = balance_frames(
        train_imgs, train_labs, train_ids, args.fail_success_ratio, args.seed
    )
    # Balance val lightly so metrics aren't dominated by teleop success mass.
    val_imgs, val_labs, val_ids = balance_frames(
        val_imgs, val_labs, val_ids, args.fail_success_ratio, args.seed + 1
    )

    meta_common = {
        "num_last_frames": args.num_last_frames,
        "val_split": args.val_split,
        "fail_success_ratio": args.fail_success_ratio,
        "seed": args.seed,
        "real_rollouts_dir": str(args.real_rollouts_dir),
        "teleop_dataset_dir": str(args.teleop_dataset_dir),
        "skip_teleop": bool(args.skip_teleop),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_payload(args.output_dir / "train.pt", train_imgs, train_labs, train_ids, meta_common)
    save_payload(args.output_dir / "val.pt", val_imgs, val_labs, val_ids, meta_common)

    summary = {
        **meta_common,
        "num_real_success_eps": n_real_s,
        "num_real_fail_eps": n_real_f,
        "num_teleop_eps": len(teleop_eps),
        "num_train_eps": len(train_eps),
        "num_val_eps": len(val_eps),
        "train_frames": len(train_labs),
        "val_frames": len(val_labs),
    }
    (args.output_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[done] summary → {args.output_dir / 'dataset_summary.json'}")


if __name__ == "__main__":
    main()

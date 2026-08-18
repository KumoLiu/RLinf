"""Infer success probability on an episode dir or a single mp4.

Usage:
    .venv_success_cls/bin/python -m sim2real.components.success_classifier.infer \\
        --checkpoint logs/success_classifier/<run>/checkpoints/best.pt \\
        --input logs/real_rollouts/.../episode_0004_success
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from sim2real.components.success_classifier.model import build_model
from sim2real.components.success_classifier.prepare_dataset import read_last_frames


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Infer success/fail on episode or mp4")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Episode directory (expects color_0.mp4) or a .mp4 path",
    )
    p.add_argument("--num-last-frames", type=int, default=8)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return p.parse_args()


def resolve_video(path: Path) -> Path:
    if path.is_file() and path.suffix.lower() == ".mp4":
        return path
    if path.is_dir():
        candidate = path / "color_0.mp4"
        if candidate.is_file():
            return candidate
        mp4s = sorted(path.glob("*.mp4"))
        if len(mp4s) == 1:
            return mp4s[0]
        raise FileNotFoundError(f"no color_0.mp4 (or unique mp4) under {path}")
    raise FileNotFoundError(f"input not found: {path}")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    train_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    model = build_model(
        hidden_dim=int(train_args.get("hidden_dim", 256)),
        dropout=float(train_args.get("dropout", 0.1)),
        image_size=int(train_args.get("image_size", 224)),
        pretrained=False,
        checkpoint=None,
    )
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    video = resolve_video(args.input)
    frames = read_last_frames(video, args.num_last_frames)
    batch = torch.stack(frames, dim=0).to(device)
    with torch.no_grad():
        probs = model.predict_proba(batch)
    mean_p = float(probs.mean().item())
    majority = float((probs >= args.threshold).float().mean().item())
    pred = "success" if mean_p >= args.threshold else "fail"
    print(f"video: {video}")
    print(f"frames: {len(frames)}")
    print(f"per-frame probs: {[round(float(x), 3) for x in probs.tolist()]}")
    print(f"mean_prob: {mean_p:.4f}")
    print(f"majority_success_rate: {majority:.4f}")
    print(f"prediction (@mean>={args.threshold}): {pred}")


if __name__ == "__main__":
    main()

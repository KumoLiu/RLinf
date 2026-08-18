"""Build a progress-aware reward dataset from DreamDojo demonstrations."""

from __future__ import annotations

import argparse
import os
import random
import sys
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch


@contextmanager
def _working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dreamdojo-path", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples-per-class", type=int, default=4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _load_dataset(dreamdojo_path: Path, dataset_path: Path):
    sys.path.insert(0, str(dreamdojo_path))
    with _working_directory(dreamdojo_path):
        from groot_dreams.dataloader import MultiVideoActionDataset

        dataset = MultiVideoActionDataset(
            num_frames=13,
            dataset_path=str(dataset_path),
            data_split="full",
            single_base_index=False,
            deterministic_uniform_sampling=False,
        )
    if len(dataset.datasets) != 1:
        raise ValueError("Expected exactly one DreamDojo dataset")
    return dataset.datasets[0].lerobot_dataset


def _select_indices(
    steps: list[int], samples_per_class: int
) -> tuple[list[int], list[int]]:
    if len(steps) < samples_per_class * 4:
        raise ValueError(f"Trajectory is too short: {len(steps)} frames")
    negative_pool = steps[: len(steps) // 2]
    negative_positions = np.linspace(
        0, len(negative_pool) - 1, samples_per_class, dtype=int
    )
    negative_steps = [negative_pool[position] for position in negative_positions]
    positive_steps = steps[-samples_per_class:]
    return negative_steps, positive_steps


def _read_image(dataset, trajectory_id: int, step_index: int) -> torch.Tensor:
    raw = dataset.get_step_data(trajectory_id, step_index)
    image = np.asarray(raw["video.ego_view"][0])
    if np.issubdtype(image.dtype, np.floating) and image.max() <= 1:
        image = image * 255
    image = np.clip(image, 0, 255).astype(np.uint8)
    return torch.from_numpy(image.copy()).permute(2, 0, 1).contiguous()


def _build_split(
    dataset,
    trajectories: list[int],
    steps_by_trajectory: dict[int, list[int]],
    samples_per_class: int,
) -> dict:
    images = []
    labels = []
    episode_ids = []
    for trajectory_id in trajectories:
        negative_steps, positive_steps = _select_indices(
            steps_by_trajectory[trajectory_id], samples_per_class
        )
        for label, selected_steps in ((0, negative_steps), (1, positive_steps)):
            for step_index in selected_steps:
                images.append(_read_image(dataset, trajectory_id, step_index))
                labels.append(label)
                episode_ids.append(f"{trajectory_id}:{step_index}")
    return {
        "images": images,
        "labels": labels,
        "metadata": {"episode_ids": episode_ids},
    }


def main() -> None:
    args = parse_args()
    dataset = _load_dataset(
        args.dreamdojo_path.expanduser().resolve(),
        args.dataset_path.expanduser().resolve(),
    )
    steps_by_trajectory = defaultdict(list)
    for trajectory_id, step_index in dataset.all_steps:
        steps_by_trajectory[int(trajectory_id)].append(int(step_index))
    for steps in steps_by_trajectory.values():
        steps.sort()

    trajectories = sorted(steps_by_trajectory)
    random.Random(args.seed).shuffle(trajectories)
    num_val = max(1, round(len(trajectories) * args.val_fraction))
    val_trajectories = trajectories[:num_val]
    train_trajectories = trajectories[num_val:]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_payload = _build_split(
        dataset, train_trajectories, steps_by_trajectory, args.samples_per_class
    )
    val_payload = _build_split(
        dataset, val_trajectories, steps_by_trajectory, args.samples_per_class
    )
    torch.save(train_payload, args.output_dir / "train.pt")
    torch.save(val_payload, args.output_dir / "val.pt")
    print(
        f"Saved {len(train_payload['labels'])} train and "
        f"{len(val_payload['labels'])} validation frames"
    )


if __name__ == "__main__":
    main()

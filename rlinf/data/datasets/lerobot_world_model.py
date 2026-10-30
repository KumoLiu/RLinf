# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Aligned RGB, joint state and preceding commands from LeRobot v2.1 episodes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class LeRobotV21InitDataset(Dataset):
    """Read reset images and causal action anchors for the H2 world model.

    Args:
        data_path: LeRobot v2.1 root containing meta, parquet and video files.
        video_key: Camera name, either canonical or a modality.json alias.
        state_dim: Number of observed joints.
        action_dim: Number of commanded joints.
        image_size: Optional output height and width.
        episodes: Episode IDs in reset order; None selects all episodes.
        random_start_frame: Sample a start frame instead of using frame zero.
        video_backend: Decoder name; this adapter uses pyav.
    """

    def __init__(
        self,
        data_path: str,
        video_key: str = "observation.images.cam_left_high",
        state_dim: int = 58,
        action_dim: int = 58,
        image_size: tuple[int, int] | None = None,
        episodes: list[int] | None = None,
        random_start_frame: bool = False,
        video_backend: str = "pyav",
    ):
        self.root = Path(data_path).expanduser().resolve()
        self.info = json.loads((self.root / "meta/info.json").read_text())
        self.fps = float(self.info["fps"])
        self.chunks_size = int(self.info.get("chunks_size", 1000))
        self.state_dim, self.action_dim = int(state_dim), int(action_dim)
        self.video_key = video_key
        self.image_size = tuple(image_size) if image_size is not None else None
        self.random_start_frame = random_start_frame
        if video_backend != "pyav":
            raise ValueError("The Cosmos LeRobot adapter requires video_backend='pyav'")
        self.episodes = (
            list(episodes)
            if episodes is not None
            else list(range(int(self.info["total_episodes"])))
        )
        modality_path = self.root / "meta/modality.json"
        cameras = (
            json.loads(modality_path.read_text()).get("video", {})
            if modality_path.exists()
            else {
                key: {"original_key": key}
                for key, feature in self.info["features"].items()
                if feature.get("dtype") == "video"
            }
        )
        if video_key not in cameras:
            raise ValueError(
                f"Unknown camera {video_key!r}; available: {list(cameras)}"
            )
        self.video_original_key = cameras[video_key].get("original_key", video_key)
        self._episode_meta = {
            int(row["episode_index"]): row for row in self._read_jsonl("episodes.jsonl")
        }
        self.task_texts = {
            int(row["task_index"]): str(row["task"])
            for row in self._read_jsonl("tasks.jsonl")
        }
        missing = [ep for ep in self.episodes if not self._parquet_path(ep).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing episode parquet files: {missing[:5]}")

    def _read_jsonl(self, name: str) -> list[dict[str, Any]]:
        path = self.root / "meta" / name
        if not path.exists():
            return []
        return [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]

    def _parquet_path(self, episode: int) -> Path:
        return self.root / self.info["data_path"].format(
            episode_chunk=episode // self.chunks_size, episode_index=episode
        )

    def _video_path(self, episode: int) -> Path:
        return self.root / self.info["video_path"].format(
            episode_chunk=episode // self.chunks_size,
            episode_index=episode,
            video_key=self.video_original_key,
        )

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.get_at_frame(index)

    def get_at_frame(
        self, index: int, start_frame: int | None = None
    ) -> dict[str, Any]:
        """Load one aligned reset; the first action is anchored to measured state."""
        import av
        import pyarrow.parquet as pq

        episode = int(self.episodes[index])
        metadata = self._episode_meta.get(episode, {})
        table = pq.read_table(self._parquet_path(episode))
        states = np.asarray(
            table.column("observation.state").to_pylist(), dtype=np.float32
        )
        actions = np.asarray(table.column("action").to_pylist(), dtype=np.float32)
        for key, array, dim in [
            ("observation.state", states, self.state_dim),
            ("action", actions, self.action_dim),
        ]:
            if array.shape != (table.num_rows, dim):
                raise ValueError(
                    f"Expected {key} shape {(table.num_rows, dim)}, found {array.shape}"
                )
        length = int(metadata.get("length", table.num_rows))
        if start_frame is None:
            start_frame = (
                int(np.random.randint(length)) if self.random_start_frame else 0
            )
        if type(start_frame) is not int or not 0 <= start_frame < length:
            raise ValueError(f"Invalid start_frame={start_frame} for episode {episode}")
        video_path = self._video_path(episode)
        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]
            stream.thread_count = 1
            stream.thread_type = "AUTO"
            for frame_index, frame in enumerate(container.decode(stream)):
                if frame_index == start_frame:
                    pixels = frame.to_ndarray(format="rgb24")
                    break
            else:
                raise ValueError(f"Frame {start_frame} is missing from {video_path}")
        image = torch.from_numpy(pixels).permute(2, 0, 1).float() / 255
        if self.image_size is not None and image.shape[1:] != self.image_size:
            image = F.interpolate(
                image[None], size=self.image_size, mode="bilinear", align_corners=False
            )[0]
        tasks = metadata.get("tasks", [])
        task_index = (
            int(table.column("task_index")[0].as_py())
            if "task_index" in table.column_names
            else None
        )
        task = str(tasks[0]) if tasks else self.task_texts.get(task_index, "")
        state = torch.from_numpy(states[start_frame]).float()
        # action[t] advances image[t] to image[t+1]; state[0] anchors the first command.
        anchor = states[0] if start_frame == 0 else actions[start_frame - 1]
        return {
            "start_items": [
                {
                    "image": image,
                    "observation.state": state,
                    "init_ee_pose": state,
                    "action": torch.from_numpy(actions[start_frame]).float(),
                    "wm_previous_action": torch.from_numpy(anchor).float(),
                    "wm_previous_state": state,
                    "task": task,
                }
            ],
            "target_items": [],
            "task": task,
            "episode_index": episode,
            "dataset_meta": {"start_frame": start_frame, "episode_length": length},
        }

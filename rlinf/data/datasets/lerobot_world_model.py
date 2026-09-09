# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Init dataset for world-model envs that reads directly from a LeRobot v2.1 tree.

This is the LeRobot-native counterpart of
:class:`rlinf.data.datasets.world_model.NpyTrajectoryDatasetWrapper`, meant for
world-model envs like :class:`DreamDojoEnv`. It emits the same
``{"start_items", "target_items", "task", "episode_index", "dataset_meta"}``
schema so it can be dropped into :meth:`BaseWorldEnv._build_dataset` without
touching the env code.

Expected LeRobot v2.1 layout (matches ``pick_trocar_teleop_success_*``)::

    <root>/
        meta/
            info.json         # features, path templates, fps, chunks_size
            modality.json     # state.*/action.*/video.* schemas
            tasks.jsonl       # {task_index -> task_text}
            episodes.jsonl    # per-episode metadata (length, phase frames, ...)
        data/chunk-000/
            episode_000000.parquet
            ...
        videos/chunk-000/<original_key>/
            episode_000000.mp4
            ...
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class LeRobotV21InitDataset(Dataset):
    """Sample first-frame (+ optional pre-KIR context) from a LeRobot v2.1 tree.

    Args:
        data_path: Path to the LeRobot v2.1 dataset root (dir containing
            ``meta/``, ``data/``, ``videos/``).
        video_key: The ``modality.json`` video key to load, e.g. ``head_view``.
            The real video file is looked up via
            ``modality.json["video"][video_key]["original_key"]``.
        state_dim: Expected flat state dimensionality; used only for a shape
            assertion so mismatches fail loudly.
        action_dim: Expected flat action dimensionality; same purpose.
        enable_kir: When ``True``, ``target_items`` contains
            ``kir_context_len`` real frames immediately following the start
            frame, so the world model can be seeded with real dynamics before
            it starts free-running. When ``False``, ``target_items`` is empty
            and the env repeats the start frame across the condition window.
        kir_context_len: Number of KIR context frames (typically
            ``condition_frame_length - 1``, matching WanEnv's 4).
        image_size: Optional ``(H, W)``; frames are bilinearly resized to it.
        camera_names: Kept for API parity with ``NpyTrajectoryDatasetWrapper``;
            the actual key loaded is always ``video_key``. The returned dicts
            put the image under the string ``"image"`` so downstream envs
            (WanEnv, DreamDojoEnv) can consume it unchanged.
        episodes: Optional explicit list of episode indices to expose. When
            ``None``, uses every episode listed in ``meta/info.json``. Useful
            for restricting to a train/val split without moving files.
        random_start_frame: When ``True``, ``__getitem__`` samples a random
            start frame in ``[0, episode_length - kir_context_len - 1]``
            instead of always using frame 0. Recommended ``False`` for the
            first version so all episodes seed the WM from the true initial
            robot pose.
        video_backend: ``"decord"`` (default, fast + zero-copy) or ``"pyav"``.
    """

    def __init__(
        self,
        data_path: str,
        video_key: str = "head_view",
        state_dim: int = 28,
        action_dim: int = 28,
        enable_kir: bool = True,
        kir_context_len: int = 4,
        image_size: Optional[tuple[int, int]] = None,
        camera_names: Optional[list[str]] = None,  # noqa: ARG002 (parity only)
        episodes: Optional[list[int]] = None,
        random_start_frame: bool = False,
        video_backend: str = "decord",
    ):
        self.root = Path(data_path).expanduser().resolve()
        if not (self.root / "meta").is_dir():
            raise FileNotFoundError(
                f"LeRobot v2.1 layout expected: {self.root / 'meta'} not found"
            )

        self.video_key = video_key
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.enable_kir = bool(enable_kir)
        self.kir_context_len = int(kir_context_len)
        self.image_size = tuple(image_size) if image_size is not None else None
        self.random_start_frame = bool(random_start_frame)
        self.video_backend = video_backend

        # ---- Load meta ----
        with (self.root / "meta" / "info.json").open() as f:
            self.info = json.load(f)
        with (self.root / "meta" / "modality.json").open() as f:
            self.modality = json.load(f)

        self.fps = float(self.info.get("fps", 30))
        self.chunks_size = int(self.info.get("chunks_size", 1000))
        self.data_tmpl = str(self.info["data_path"])
        self.video_tmpl = str(self.info["video_path"])

        # Resolve which video file this ``video_key`` maps to.
        video_meta = self.modality.get("video", {}).get(self.video_key)
        if video_meta is None:
            available = list(self.modality.get("video", {}).keys())
            raise KeyError(
                f"video key {self.video_key!r} not in modality.json (have {available})"
            )
        self.video_original_key = video_meta.get("original_key", self.video_key)

        # State/action component slices (e.g. {"left_arm": slice(0, 7), ...}).
        self.state_slices = self._extract_component_slices(self.modality.get("state", {}))
        self.action_slices = self._extract_component_slices(self.modality.get("action", {}))

        # ---- task_index -> text ----
        self.task_texts: dict[int, str] = {}
        tasks_path = self.root / "meta" / "tasks.jsonl"
        if tasks_path.exists():
            with tasks_path.open() as f:
                for line in f:
                    if not line.strip():
                        continue
                    entry = json.loads(line)
                    self.task_texts[int(entry["task_index"])] = str(entry["task"])

        # ---- Episode index list ----
        total_episodes = int(self.info.get("total_episodes", 0))
        self.episodes: list[int] = (
            list(episodes) if episodes is not None else list(range(total_episodes))
        )

        # Cache per-episode length + task text from episodes.jsonl (fast, no parquet read).
        self._episode_meta: dict[int, dict[str, Any]] = {}
        eps_path = self.root / "meta" / "episodes.jsonl"
        if eps_path.exists():
            with eps_path.open() as f:
                for line in f:
                    if not line.strip():
                        continue
                    entry = json.loads(line)
                    self._episode_meta[int(entry["episode_index"])] = entry

        # Validate every requested episode exists on disk.
        missing = [i for i in self.episodes if not self._parquet_path(i).is_file()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} requested episodes missing parquet files, first few: {missing[:5]}"
            )

    # ------------------------------------------------------------------ #
    #                          Public interface                           #
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ep_idx = int(self.episodes[idx])
        ep_meta = self._episode_meta.get(ep_idx, {})
        ep_length = int(ep_meta.get("length", 0))

        # Fall back to parquet row-count if episodes.jsonl is absent.
        parquet_path = self._parquet_path(ep_idx)
        table = self._read_parquet(parquet_path)
        if ep_length <= 0:
            ep_length = table.num_rows

        # Pick the start frame.
        max_start = max(0, ep_length - self.kir_context_len - 1)
        if self.random_start_frame and max_start > 0:
            start_frame = int(np.random.randint(0, max_start + 1))
        else:
            start_frame = 0

        # Compute the frame indices we need to decode.
        wanted_indices: list[int] = [start_frame]
        if self.enable_kir and self.kir_context_len > 0:
            wanted_indices.extend(
                start_frame + 1 + i for i in range(self.kir_context_len)
            )
            wanted_indices = [min(fi, ep_length - 1) for fi in wanted_indices]

        # Task text: prefer episodes.jsonl["tasks"][0], else tasks.jsonl.
        task_text = ""
        if "tasks" in ep_meta and ep_meta["tasks"]:
            task_text = str(ep_meta["tasks"][0])
        else:
            task_indices = table.column("task_index").to_pylist() if "task_index" in table.schema.names else []
            if task_indices:
                task_text = self.task_texts.get(int(task_indices[0]), "")

        # Decode images for the requested frames.
        video_path = self._video_path(ep_idx)
        frames_uint8 = self._decode_video_frames(video_path, wanted_indices)  # (T, H, W, 3) uint8

        # Extract state / action rows.
        state_all = self._table_column_as_array(table, "observation.state")  # (N, state_dim) or None
        action_all = self._table_column_as_array(table, "action")            # (N, action_dim) or None
        if state_all is not None:
            assert state_all.shape[1] == self.state_dim, (
                f"state_dim mismatch: config expects {self.state_dim}, parquet has "
                f"{state_all.shape[1]} in {parquet_path}"
            )
        if action_all is not None:
            assert action_all.shape[1] == self.action_dim, (
                f"action_dim mismatch: config expects {self.action_dim}, parquet has "
                f"{action_all.shape[1]} in {parquet_path}"
            )

        # Build the per-frame dicts.
        def _frame_dict(local_pos: int, absolute_frame: int) -> dict[str, Any]:
            img = self._prepare_image(frames_uint8[local_pos])  # tensor [3, H, W] float [0,1]
            state_vec = (
                torch.from_numpy(state_all[absolute_frame]).float()
                if state_all is not None
                else torch.zeros(self.state_dim, dtype=torch.float32)
            )
            action_vec = (
                torch.from_numpy(action_all[absolute_frame]).float()
                if action_all is not None
                else torch.zeros(self.action_dim, dtype=torch.float32)
            )
            return {
                "image": img,
                "observation.state": state_vec,
                # ``init_ee_pose`` alias kept so world envs that pull it directly
                # can find something to store on the reset path.
                "init_ee_pose": state_vec,
                "action": action_vec,
                "task": task_text,
            }

        start_items = [_frame_dict(0, wanted_indices[0])]
        target_items: list[dict[str, Any]] = []
        if self.enable_kir and self.kir_context_len > 0:
            for local_pos, abs_frame in enumerate(wanted_indices[1:], start=1):
                target_items.append(_frame_dict(local_pos, abs_frame))

        return {
            "start_items": start_items,
            "target_items": target_items,
            "task": task_text,
            "episode_index": ep_idx,
            "dataset_meta": {
                "episode_length": ep_length,
                "start_frame": start_frame,
                "video_path": str(video_path),
                "parquet_path": str(parquet_path),
                # Expose the phase boundaries so downstream code (reward-model
                # training, progress logging) can grab them for free.
                "phase_frames": ep_meta.get("source_annotation", {}).get(
                    "cleaned_phase_frames"
                ),
                "success": ep_meta.get("success"),
            },
        }

    # ------------------------------------------------------------------ #
    #                              Internals                              #
    # ------------------------------------------------------------------ #

    def _parquet_path(self, episode_index: int) -> Path:
        chunk = episode_index // self.chunks_size
        rel = self.data_tmpl.format(
            episode_chunk=chunk, episode_index=episode_index
        )
        return (self.root / rel).resolve()

    def _video_path(self, episode_index: int) -> Path:
        chunk = episode_index // self.chunks_size
        rel = self.video_tmpl.format(
            episode_chunk=chunk,
            video_key=self.video_original_key,
            episode_index=episode_index,
        )
        return (self.root / rel).resolve()

    @staticmethod
    def _extract_component_slices(section: dict) -> dict[str, slice]:
        """Turn {"left_arm": {"start":0,"end":7}, ...} into {"left_arm": slice(0,7), ...}."""
        out: dict[str, slice] = {}
        for name, info in section.items():
            if not isinstance(info, dict):
                continue
            start = info.get("start")
            end = info.get("end")
            if start is not None and end is not None:
                out[name] = slice(int(start), int(end))
        return out

    @staticmethod
    def _read_parquet(path: Path):
        import pyarrow.parquet as pq

        return pq.read_table(str(path))

    @staticmethod
    def _table_column_as_array(table, column: str) -> Optional[np.ndarray]:
        if column not in table.schema.names:
            return None
        col = table.column(column)
        # Values may be list<float32> per row or a fixed-size list; both to_numpy fine.
        py = col.to_pylist()
        arr = np.asarray(py, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[:, None]
        return arr

    def _decode_video_frames(
        self, video_path: Path, frame_indices: list[int]
    ) -> np.ndarray:
        """Return uint8 ``(T, H, W, 3)`` for the requested frame indices."""
        if self.video_backend == "decord":
            try:
                import decord  # type: ignore
            except ImportError as e:
                raise ImportError(
                    "decord is required for the default video backend. "
                    "pip install decord, or pass video_backend='pyav'."
                ) from e
            vr = decord.VideoReader(str(video_path), num_threads=1)
            total = len(vr)
            clipped = [min(max(0, int(fi)), total - 1) for fi in frame_indices]
            frames = vr.get_batch(clipped)
            return frames.asnumpy()

        if self.video_backend == "pyav":
            try:
                import av  # type: ignore
            except ImportError as e:
                raise ImportError(
                    "pyav is required for video_backend='pyav'."
                ) from e
            wanted = set(int(fi) for fi in frame_indices)
            decoded: dict[int, np.ndarray] = {}
            with av.open(str(video_path)) as container:
                stream = container.streams.video[0]
                stream.thread_type = "AUTO"
                for i, frame in enumerate(container.decode(stream)):
                    if i in wanted:
                        decoded[i] = frame.to_ndarray(format="rgb24")
                        if len(decoded) == len(wanted):
                            break
            return np.stack([decoded[min(max(0, int(fi)), max(decoded))] for fi in frame_indices])

        raise ValueError(f"Unknown video_backend={self.video_backend!r}")

    def _prepare_image(self, frame_hwc_uint8: np.ndarray) -> torch.Tensor:
        """Convert an ``(H, W, 3)`` uint8 frame to ``[3, H, W]`` float in [0, 1]."""
        img = torch.from_numpy(frame_hwc_uint8).permute(2, 0, 1).float() / 255.0
        if self.image_size is not None and tuple(img.shape[1:]) != self.image_size:
            img = F.interpolate(
                img.unsqueeze(0),
                size=self.image_size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        return img

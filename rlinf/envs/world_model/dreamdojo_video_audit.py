# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Eval-only copies of WM frames with the exact deployed reward decisions.

Rendering never modifies policy/reward inputs. A raw probability is not an
ordered, temporally confirmed milestone, and neither is physical ground truth.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

HEADS = ("PICK", "HANDOVER", "PLACE")


def prediction_status(stage: int, head: int, final: bool) -> str:
    """Distinguish confirmed success from pending and end-of-episode failure."""
    if stage > head:
        return "SUCCESS"
    return "FAIL" if final else "NOT YET"


def annotate_frame(
    frame: np.ndarray,
    probabilities: np.ndarray | None,
    stage: int,
    frame_index: int,
    final: bool,
    case: dict,
    label: str,
    thresholds: tuple[float, ...],
) -> np.ndarray:
    """Append a side panel without painting over any source-image pixels."""
    height, width = frame.shape[:2]
    result = np.zeros((max(height, 480), width + 400, 3), dtype=np.uint8)
    result[:height, :width] = frame
    result[:, width:] = (20, 24, 30)

    def text(value: str, y: int, color=(230, 230, 230), scale=0.48):
        cv2.putText(
            result,
            value,
            (width + 12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            1,
            cv2.LINE_AA,
        )

    text("WM + CLASSIFIER PREDICTIONS", 25, scale=0.57)
    text("NOT physical ground truth", 48, (255, 190, 70))
    text(label[:46], 73)
    text(f"Case {case['flat_episode_index']:03d} | episode {case['episode_index']}", 98)
    source = Path(case["source_path"]).name
    source_label = (
        "TELEOP"
        if "teleop" in source
        else "ROLLOUT 30k"
        if "30k" in source
        else "ROLLOUT 10k"
    )
    text(f"{source_label} | reset raw frame {case['start_frame']}", 120)
    text(f"t={frame_index / 15:.2f}s | frame={frame_index}/120", 143)
    for head, name in enumerate(HEADS):
        y = 175 + 70 * head
        text(name, y, scale=0.59)
        if probabilities is None:
            text("p=N/A (initial frame not scored)", y + 20)
        else:
            p = float(probabilities[head])
            raw = "YES" if p >= thresholds[head] else "NO"
            text(f"p={p:.3f} | p>={thresholds[head]:.2f}: {raw}", y + 20)
        status = prediction_status(stage, head, final)
        color = (
            (80, 230, 130)
            if status == "SUCCESS"
            else (255, 110, 100)
            if final
            else (255, 200, 100)
        )
        text(f"Confirmed: {status}", y + 40, color)
    text("Ordered + temporal; success stays latched", 393)
    text("Pick/hand: 13 of 15 ticks; place: 2/2", 414)
    text("p shown: last tick of each WM frame", 435)
    text("15fps WM -> duplicated 30Hz scoring", 456)
    return result


class MilestoneVideoAudit:
    """Save each valid eval case, raw video, probabilities and confirmed stages."""

    def __init__(self, directory: Path, label: str, thresholds: tuple[float, ...]):
        self.directory = Path(directory)
        self.label = str(label)
        self.thresholds = tuple(thresholds)
        self.reset_index = 0
        self.frames: list[np.ndarray] = []
        self.probabilities: list[np.ndarray] = []
        self.stages: list[np.ndarray] = []
        self.finished = True

    def start(self, initial_frames: np.ndarray, cases: list[dict]) -> None:
        """Begin one 240-action native eval rollout, preserving padding metadata."""
        if not self.finished:
            raise RuntimeError("Video audit reset before previous rollout completed")
        if (
            initial_frames.ndim != 4
            or initial_frames.dtype != np.uint8
            or len(cases) != len(initial_frames)
        ):
            raise ValueError(
                "Expected uint8 [B,H,W,3] reset frames and one case per env"
            )
        self.cases = [dict(case) for case in cases]
        self.frames = [initial_frames[:, None].copy()]
        self.probabilities = []
        self.stages = []
        self.finished = False

    def append(
        self, frames: np.ndarray, probabilities: np.ndarray, stages: np.ndarray
    ) -> None:
        """Record exact pre-encoding outputs; stages come from native payouts."""
        if self.finished:
            raise RuntimeError("Video audit append outside an active rollout")
        if frames.ndim != 5 or frames.dtype != np.uint8 or frames.shape[-1] != 3:
            raise ValueError("Expected uint8 [B,T,H,W,3] generated frames")
        if (
            probabilities.shape != (*frames.shape[:2], 3)
            or stages.shape != frames.shape[:2]
        ):
            raise ValueError("Frames, probabilities and stages must be time-aligned")
        if frames.shape[0] != len(self.cases) or not np.isfinite(probabilities).all():
            raise ValueError("Invalid video audit batch/probabilities")
        previous = (
            self.stages[-1][:, -1:] if self.stages else np.zeros((len(self.cases), 1))
        )
        if np.any(
            np.diff(np.concatenate([previous, stages], axis=1), axis=1) < 0
        ) or np.any((stages < 0) | (stages > 3)):
            raise ValueError("Confirmed stages must be monotone in [0,3]")
        self.frames.append(frames.copy())
        self.probabilities.append(probabilities.copy())
        self.stages.append(stages.astype(np.int64, copy=True))

    def finish(self) -> list[dict]:
        """Encode after the final native step; never re-score compressed video."""
        if self.finished or not self.probabilities:
            raise RuntimeError("No unfinished video audit rollout")
        frames = np.concatenate(self.frames, axis=1)
        probabilities = np.concatenate(self.probabilities, axis=1)
        stages = np.concatenate(self.stages, axis=1)
        if frames.shape[1] != 121 or probabilities.shape[1] != 120:
            raise ValueError("Audit requires full 240-action / 120-WM-frame eval")
        output = self.directory / f"rollout_{self.reset_index:03d}"
        output.mkdir(parents=True, exist_ok=False)
        summaries = []
        for index, case in enumerate(self.cases):
            if not case["valid"]:
                continue
            directory = output / f"case_{case['flat_episode_index']:03d}"
            directory.mkdir()
            with (
                imageio.get_writer(
                    directory / "raw.mp4",
                    fps=15,
                    codec="libx264",
                    quality=8,
                    ffmpeg_params=["-threads", "2"],
                ) as raw_writer,
                imageio.get_writer(
                    directory / "annotated.mp4",
                    fps=15,
                    codec="libx264",
                    quality=8,
                    ffmpeg_params=["-threads", "2"],
                ) as annotated_writer,
            ):
                for t, frame in enumerate(frames[index]):
                    p = probabilities[index, t - 1] if t else None
                    stage = int(stages[index, t - 1]) if t else 0
                    raw_writer.append_data(frame)
                    annotated_writer.append_data(
                        annotate_frame(
                            frame,
                            p,
                            stage,
                            t,
                            t == len(frames[index]) - 1,
                            case,
                            self.label,
                            self.thresholds,
                        )
                    )
            with (directory / "trace.csv").open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(
                    [
                        "wm_frame",
                        "relative_seconds",
                        "raw_action_count",
                        "pick_probability",
                        "handover_probability",
                        "place_probability",
                        "confirmed_stage",
                        "pick_confirmed",
                        "handover_confirmed",
                        "place_confirmed",
                    ]
                )
                writer.writerow([0, 0, 0, "", "", "", 0, False, False, False])
                for t in range(probabilities.shape[1]):
                    stage = int(stages[index, t])
                    writer.writerow(
                        [
                            t + 1,
                            (t + 1) / 15,
                            (t + 1) * 2,
                            *probabilities[index, t].tolist(),
                            stage,
                            *[stage > head for head in range(3)],
                        ]
                    )
            summary = {
                **case,
                "label": self.label,
                "frame_count": int(frames.shape[1]),
                "fps": 15,
                "final_stage": int(stages[index, -1]),
                "classifier_success": bool(stages[index, -1] == 3),
                "milestones": {
                    name.lower(): bool(stages[index, -1] > head)
                    for head, name in enumerate(HEADS)
                },
                "thresholds": self.thresholds,
                "probability_semantics": "Exact final 30Hz classifier tick per 15fps WM frame; not rescored from MP4.",
                "decision_semantics": "Native ordered temporal confirmation from original payouts, not instantaneous thresholding; not ground truth.",
            }
            (directory / "summary.json").write_text(
                json.dumps(summary, indent=2) + "\n"
            )
            summaries.append(summary)
        (output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
        self.reset_index += 1
        self.finished = True
        self.frames.clear()
        self.probabilities.clear()
        self.stages.clear()
        return summaries

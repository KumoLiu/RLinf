# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Video audit must not alter model inputs or invent frame-level decisions."""

import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rlinf.envs.world_model import dreamdojo_video_audit as audit_module
from rlinf.envs.world_model.dreamdojo_video_audit import (
    MilestoneVideoAudit,
    annotate_frame,
    prediction_status,
)
from rlinf.envs.world_model.world_model_dreamdojo_env import DreamDojoEnv
from toolkits.world_model.dreamdojo_validation import config


def case(index=0, valid=True):
    return {
        "flat_episode_index": index,
        "episode_index": index,
        "source_path": "/assets/data/pick_trocar_teleop_success_validation",
        "start_frame": 1,
        "valid": valid,
    }


def test_audit_is_disabled_for_normal_training_and_eval():
    cfg = config()
    assert cfg.env.train.video_audit_dir is None
    assert cfg.env.eval.video_audit_dir is None


def test_success_is_confirmed_state_not_instantaneous_probability():
    assert prediction_status(0, 0, False) == "NOT YET"
    assert prediction_status(0, 0, True) == "FAIL"
    assert prediction_status(2, 1, False) == "SUCCESS"
    assert prediction_status(2, 2, True) == "FAIL"
    assert prediction_status(3, 2, True) == "SUCCESS"


def test_overlay_preserves_every_source_pixel():
    frame = np.full((480, 640, 3), (100, 140, 180), np.uint8)
    original = frame.copy()
    rendered = annotate_frame(
        frame, np.array([0.1, 0.2, 0.3]), 3, 120, True, case(), "step130", (0.8,) * 3
    )
    assert rendered.shape == (480, 1040, 3)
    np.testing.assert_array_equal(rendered[:, :640], original)
    np.testing.assert_array_equal(frame, original)


def test_full_trace_is_aligned_padding_excluded_and_not_rescored(tmp_path, monkeypatch):
    writers = {}

    class Writer:
        def __init__(self, path, **kwargs):
            self.count = 0
            writers[str(path)] = self

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def append_data(self, frame):
            self.count += 1
            self.last = frame.copy()

    monkeypatch.setattr(audit_module.imageio, "get_writer", Writer)
    audit = MilestoneVideoAudit(tmp_path, "step130", (0.8,) * 3)
    initial = np.zeros((2, 16, 32, 3), dtype=np.uint8)
    frames = np.full((2, 120, 16, 32, 3), 120, dtype=np.uint8)
    probs = np.full((2, 120, 3), 0.2, dtype=np.float32)
    stages = np.tile(np.repeat([0, 1, 2, 3], 30), (2, 1))
    audit.start(initial, [case(0), case(1, False)])
    audit.append(frames, probs, stages)
    frames[:] = 0
    probs[:] = 0
    stages[:] = 0
    summaries = audit.finish()
    assert len(summaries) == 1 and summaries[0]["classifier_success"]
    assert summaries[0]["final_stage"] == 3
    directory = tmp_path / "rollout_000/case_000"
    assert not (tmp_path / "rollout_000/case_001").exists()
    assert len(writers) == 2 and all(writer.count == 121 for writer in writers.values())
    np.testing.assert_array_equal(
        writers[str(directory / "raw.mp4")].last, np.full((16, 32, 3), 120, np.uint8)
    )
    with (directory / "trace.csv").open() as stream:
        trace = list(csv.DictReader(stream))
    assert len(trace) == 121
    assert trace[0]["pick_probability"] == ""
    assert int(trace[-1]["raw_action_count"]) == 240
    assert float(trace[-1]["relative_seconds"]) == 8
    assert float(trace[-1]["place_probability"]) == pytest.approx(0.2)
    assert trace[-1]["place_confirmed"] == "True"
    assert json.loads((directory / "summary.json").read_text())["classifier_success"]


def test_incomplete_audit_fails_instead_of_exporting_misleading_video(tmp_path):
    audit = MilestoneVideoAudit(tmp_path, "step130", (0.8,) * 3)
    audit.start(np.zeros((1, 16, 32, 3), np.uint8), [case()])
    with pytest.raises(RuntimeError, match="previous rollout"):
        audit.start(np.zeros((1, 16, 32, 3), np.uint8), [case()])
    audit.append(
        np.zeros((1, 6, 16, 32, 3), np.uint8), np.zeros((1, 6, 3)), np.zeros((1, 6))
    )
    with pytest.raises(ValueError, match="full 240-action"):
        audit.finish()


def test_native_audit_uses_original_payouts_and_does_not_change_reward():
    env = DreamDojoEnv.__new__(DreamDojoEnv)
    env.current_obs = torch.zeros(2, 3, 1, 7, 16, 32)
    env.wm_steps_per_chunk, env.num_envs, env.chunk = 6, 2, 12
    payouts = torch.tensor(
        [[0, 1, 0, 1, 0, 1], [0, 0, 0, 0, 0, 1]], dtype=torch.float32
    )
    probs = torch.full((2, 6, 3), 0.3)
    seen = []
    env.reward_model = SimpleNamespace(stage=torch.tensor([0, 1]))

    def score(frames):
        env.reward_model.stage = torch.tensor([3, 2])
        return payouts, probs

    env.reward_model.score_chunk = score
    env.video_audit = SimpleNamespace(append=lambda *args: seen.append(args))
    before = env.current_obs.clone()
    reward = env._infer_next_chunk_rewards()
    assert len(seen) == 1
    np.testing.assert_array_equal(seen[0][2], [[0, 1, 1, 2, 2, 3], [1, 1, 1, 1, 1, 2]])
    torch.testing.assert_close(reward[:, 1::2], payouts)
    assert torch.count_nonzero(reward[:, ::2]) == 0
    torch.testing.assert_close(env.current_obs, before)
    env.video_audit = None
    env.reward_model.stage = torch.tensor([0, 1])
    torch.testing.assert_close(env._infer_next_chunk_rewards(), reward)

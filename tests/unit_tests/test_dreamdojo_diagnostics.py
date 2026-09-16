# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Portable diagnostic configuration and reward-diversity regression tests."""

import json
from pathlib import Path

import pytest
import torch

from rlinf.utils.metric_utils import compute_group_reward_metrics
from toolkits.world_model import dreamdojo_validation as validation


def test_diagnostic_keeps_model_when_eval_returns_none():
    calls = []

    class Policy:
        def to(self, device):
            calls.append(("to", device))
            return self

        def eval(self):
            calls.append(("eval",))
            # Match the actual GR00T override, which returns None.

    model = Policy()
    assert validation.prepare_diagnostic_policy(model, "cpu") is model
    assert calls == [("to", "cpu"), ("eval",)]


@pytest.mark.parametrize("step", [5, 10, 15, 20])
def test_checkpoint_labels_are_not_tied_to_historical_step20(step):
    assert validation.checkpoint_case(
        f"/trial/global_step_{step}/actor/weights.pt"
    ) == (f"global_step_{step}")
    assert validation.checkpoint_case("base") == "base"


def test_group_metrics_preserve_scores():
    scores = torch.tensor([[0.0, 0.0], [1.0, 3.0]])
    original = scores.clone()
    metrics = compute_group_reward_metrics(scores)
    assert metrics["group_flat_fraction"] == 0.5
    assert metrics["group_zero_fraction"] == 0.5
    assert metrics["group_return_std_mean"] == 0.5
    torch.testing.assert_close(scores, original)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_group_metrics_reject_non_finite(bad):
    with pytest.raises(ValueError, match="Non-finite"):
        compute_group_reward_metrics(torch.tensor([[0.0, bad]]))


def test_cluster_paths_override_all_local_defaults(monkeypatch, tmp_path):
    paths = {
        "DREAMDOJO_ROOT": tmp_path / "wm_repo",
        "DREAMDOJO_DATA_ROOT": tmp_path / "datasets",
        "DREAMDOJO_WM_CHECKPOINT": tmp_path / "weights/wm.pt",
        "DREAMDOJO_REWARD_CHECKPOINT": tmp_path / "weights/reward.pt",
        "DREAMDOJO_ACTION_STATISTICS": tmp_path / "stats.json",
        "GR00T_MODEL_PATH": tmp_path / "weights/policy",
    }
    for key, value in paths.items():
        monkeypatch.setenv(key, str(value))
    monkeypatch.setenv("DREAMDOJO_GPUS", "0-7")
    monkeypatch.chdir(tmp_path)
    cfg = validation.config()
    assert cfg.cluster.component_placement["actor,env,rollout"] == "0-7"
    for part in (cfg.env.train, cfg.env.eval):
        assert part.dreamdojo_root == str(paths["DREAMDOJO_ROOT"])
        assert part.model_path == str(paths["DREAMDOJO_WM_CHECKPOINT"])
        assert part.reward_model.from_pretrained == str(
            paths["DREAMDOJO_REWARD_CHECKPOINT"]
        )
        assert part.action_statistics == str(paths["DREAMDOJO_ACTION_STATISTICS"])
        assert all(
            Path(path).parent == paths["DREAMDOJO_DATA_ROOT"]
            for path in part.initial_image_path
        )
    assert (
        cfg.actor.model.model_path
        == cfg.rollout.model.model_path
        == str(paths["GR00T_MODEL_PATH"])
    )


def test_diagnostics_write_results_not_source_snapshots(monkeypatch, tmp_path):
    output = tmp_path / "diagnostic"
    input_path = tmp_path / "previous_run"
    monkeypatch.setattr(
        "sys.argv",
        ["diagnostic", "report", "--output", str(output), "--inputs", str(input_path)],
    )
    monkeypatch.setattr(
        validation,
        "report",
        lambda args: validation.save_json(args.output / "results.json", {}),
    )
    validation.main()
    assert {p.name for p in output.iterdir()} == {
        "arguments.json",
        "results.json",
        "timing.json",
    }
    assert json.loads((output / "arguments.json").read_text())["inputs"] == [
        str(input_path)
    ]

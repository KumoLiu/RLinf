# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""GR00T processor compatibility and opt-in physical-action noise bounds."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("gr00t.model.gr00t_n1d7.gr00t_n1d7")
from rlinf.models.embodiment.gr00t.gr00t_n1d7.gr00t_action_model import (
    GR00T_N1_7_ForRLActionPrediction as Model,
)
from rlinf.models.embodiment.gr00t.gr00t_n1d7.gr00t_action_model import (
    _canonicalize_gr00t_text_forward_inputs,
    _normalize_gr00t_forward_inputs,
)


@pytest.mark.parametrize("multimodal", [False, True])
def test_processor_optional_multimodal_types_survive_actor_padding(multimodal):
    inputs = {"input_ids": torch.ones(2, 3), "attention_mask": torch.ones(2, 3)}
    if multimodal:
        inputs["mm_token_type_ids"] = torch.tensor([[0, 1, 1], [0, 0, 1]])
    padded = _canonicalize_gr00t_text_forward_inputs(inputs, 5)
    normalized = _normalize_gr00t_forward_inputs(padded)
    assert normalized.keys() == inputs.keys()
    for key, value in inputs.items():
        torch.testing.assert_close(normalized[key][:, :3], value)
        assert normalized[key].shape == (2, 5)
        assert not normalized[key][:, 3:].any()


@pytest.mark.parametrize("physical", [False, True])
def test_action_noise_preserves_legacy_bounds_or_physical_units(physical, monkeypatch):
    config = {"action_noise_scale": 0.1}
    if physical:
        config["action_noise_clip"] = None
    model = SimpleNamespace(action_head=SimpleNamespace(rl_config=config))
    action = torch.tensor([[-2.5, 2.5]])
    monkeypatch.setattr(torch, "randn_like", torch.ones_like)
    expected = action + 0.1 if physical else torch.tensor([[-1.0, 1.0]])
    torch.testing.assert_close(
        Model._apply_exploration_noise(model, action, "train"), expected
    )
    assert Model._apply_exploration_noise(model, action, "eval") is action
    config["action_noise_scale"] = 0.0
    assert Model._apply_exploration_noise(model, action, "train") is action


@pytest.mark.parametrize("add_value_head", [False, True])
def test_rollout_computes_values_only_when_a_head_is_enabled(add_value_head):
    def sample(backbone, inputs, *, mode, compute_values=True):
        assert compute_values is add_value_head
        return {}, {
            key: torch.zeros(2, 1)
            for key in (
                "actions",
                "chains",
                "denoise_inds",
                "prev_logprobs",
                "prev_values",
            )
        }

    model = SimpleNamespace(
        prepare_input=lambda inputs: (inputs, inputs),
        backbone=lambda inputs: inputs,
        action_head=SimpleNamespace(
            rl_config={"add_value_head": add_value_head}, get_rl_action=sample
        ),
        _finalize_rollout_forward_inputs=lambda inputs: inputs,
    )
    actions, result = Model._get_rl_action(
        model, {"input_ids": torch.ones(2, 3)}, mode="train"
    )
    assert actions.shape == (2, 1)
    assert result["forward_inputs"]["input_ids"].shape == (2, 3)

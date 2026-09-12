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

import json
from types import SimpleNamespace

import pytest
import torch

from rlinf.envs.world_model.dreamdojo_adapters import (
    DREAMDOJO_G1_SLICE,
    G1DreamDojoActionBridge,
    pad_g1_dex3_28_to_43,
    update_g1_dex3_state,
)
from rlinf.envs.world_model.dreamdojo_reward import BatchedMilestoneReward
from rlinf.models.embodiment.gr00t.simulation_io import (
    convert_to_g1_dex3_action_n1d7,
)


def test_pad_g1_dex3_matches_official_layout():
    values = torch.arange(28, dtype=torch.float32).reshape(1, 28)
    padded = pad_g1_dex3_28_to_43(values)

    assert padded.shape == (1, 43)
    torch.testing.assert_close(padded[:, 15:22], values[:, 0:7])
    torch.testing.assert_close(padded[:, 22:29], values[:, 14:21])
    torch.testing.assert_close(padded[:, 29:36], values[:, 7:14])
    torch.testing.assert_close(padded[:, 36:43], values[:, 21:28])
    assert torch.count_nonzero(padded[:, :15]) == 0


def test_block_relative_deltas_match_dreamdojo_dataset_transform(bridge):
    actions = torch.arange(25, dtype=torch.float32).reshape(1, 25, 1).repeat(1, 1, 43)
    encoded = bridge.encode_30hz_actions(actions, torch.zeros(1, 43))
    expected = torch.tensor([2, 4, 6, 8] * 3, dtype=torch.float32)
    torch.testing.assert_close(encoded[0, :, 58], expected)


def test_encode_uses_only_dreamdojo_g1_slots_and_initial_state(bridge):
    actions = torch.randn(2, 25, 28)
    encoded = bridge.encode_30hz_actions(actions, torch.ones(2, 28))

    assert encoded.shape == (2, 12, 384)
    mask = torch.ones(384, dtype=torch.bool)
    mask[DREAMDOJO_G1_SLICE] = False
    # Only the first frame carries the G1 state prefix, in addition to actions.
    assert torch.count_nonzero(encoded[:, 0, :29]) > 0
    assert torch.count_nonzero(encoded[:, 1:, :29]) == 0
    mask[:29] = False
    assert torch.count_nonzero(encoded[..., mask]) == 0


def test_online_bridge_matches_dreamdojo_sampling_and_block_deltas(tmp_path):
    stats = {
        key: {"min": [-1.0] * 43, "max": [1.0] * 43}
        for key in ("action", "observation.state")
    }
    path = tmp_path / "stats.json"
    path.write_text(json.dumps(stats))
    bridge = G1DreamDojoActionBridge(path, "cpu")
    raw = torch.linspace(-3.0, 3.0, 25).reshape(25, 1).repeat(1, 43)
    state = torch.full((43,), 2.5)

    encoded = bridge.encode_30hz_actions(raw, state)

    sampled = raw[::2]
    expected = torch.cat(
        [sampled[start + 1 : start + 5] - sampled[start] for start in (0, 4, 8)]
    )
    torch.testing.assert_close(encoded[:, DREAMDOJO_G1_SLICE], expected)
    torch.testing.assert_close(encoded[0, :29], state[:29])
    assert torch.count_nonzero(encoded[1:, :58]) == 0
    assert torch.count_nonzero(encoded[0, 29:58]) == 0
    assert torch.count_nonzero(encoded[:, 101:]) == 0


@pytest.fixture
def bridge(tmp_path):
    stats = {
        key: {"min": [-1.0] * 43, "max": [1.0] * 43}
        for key in ("action", "observation.state")
    }
    path = tmp_path / "stats.json"
    path.write_text(json.dumps(stats))
    return G1DreamDojoActionBridge(path, "cpu")


def test_policy_time_axis_preserves_30hz_commands(bridge):
    actions = torch.arange(12).float()[None, :, None].repeat(1, 1, 28)
    history = torch.full((1, 28), -2.0)
    encoded = bridge.encode_policy_chunk(history, actions, torch.zeros(1, 28))
    # a[0,2,4,6] - a[-2], then a[8,10] - a[6]. No time stretching.
    expected = torch.tensor([2.0, 4.0, 6.0, 8.0, 2.0, 4.0])
    torch.testing.assert_close(encoded[0, :6, 73], expected)


def test_env_consumes_only_executed_frames_and_keeps_history(bridge):
    from rlinf.envs.world_model.world_model_dreamdojo_env import DreamDojoEnv

    env = DreamDojoEnv.__new__(DreamDojoEnv)
    env.device = torch.device("cpu")
    env.num_envs = 1
    env.chunk = env.wm_chunk = 12
    env.wm_steps_per_chunk = 6
    env.action_dim = 28
    env.image_size = (2, 2)
    env.current_obs = torch.zeros(1, 3, 1, 1, 2, 2)
    env.current_state = env.previous_state = env.previous_action = torch.zeros(1, 28)
    env.action_bridge = bridge
    env.seed = env.elapsed_steps = 0
    env.num_inference_steps = 35
    env.video_cfg = SimpleNamespace(save_video=False)
    env.image_queue = [[]]
    frames = torch.arange(13).float()[None, None, :, None, None].repeat(1, 3, 1, 2, 2)
    env.pipe = SimpleNamespace(generate_vid2world=lambda **kwargs: frames)
    actions = torch.arange(12).float()[None, :, None].repeat(1, 1, 28)
    env._infer_next_chunk_frames(actions)
    assert env.current_obs.shape[3] == 7
    assert env.current_obs[0, 0, 0, -1, 0, 0] == 6
    torch.testing.assert_close(env.previous_action, actions[:, 10])
    torch.testing.assert_close(env.previous_state, actions[:, 9])
    torch.testing.assert_close(env.current_state, actions[:, 11])
    env.reward_model = SimpleNamespace(
        score_chunk=lambda x: (torch.ones(1, 6), torch.ones(1, 6, 3))
    )
    rewards = env._infer_next_chunk_rewards()
    torch.testing.assert_close(rewards[0], torch.tensor([0.0, 1.0] * 6))


def test_g1_physical_action_noise_does_not_apply_normalized_bounds():
    from rlinf.models.embodiment.gr00t.gr00t_n1d7.gr00t_action_model import (
        GR00T_N1_7_ForRLActionPrediction,
    )

    obj = SimpleNamespace(
        obs_converter_type="g1_dex3_wm",
        action_head=SimpleNamespace(rl_config={"action_noise_scale": 1e-8}),
    )
    action = torch.tensor([[[1.955, -1.047]]])
    out = GR00T_N1_7_ForRLActionPrediction._apply_exploration_noise(
        obj, action, "train"
    )
    torch.testing.assert_close(out, action)


def test_update_state_uses_decoded_absolute_targets():
    state = torch.ones(2, 28)
    action = torch.full((2, 28), 2.0)
    updated = update_g1_dex3_state(state, action)

    torch.testing.assert_close(updated, action)


def test_reset_preserves_image_and_action_state_history_without_legacy_buffers():
    from rlinf.envs.world_model.world_model_dreamdojo_env import DreamDojoEnv

    env = DreamDojoEnv.__new__(DreamDojoEnv)
    env.device = torch.device("cpu")
    env.num_envs = 2
    env.state_dim = 28
    env.image_size = (2, 3)
    env.condition_frame_length = 1
    env.onload = lambda: None
    env._reset_metrics = lambda: None
    env.trans_norm = lambda image: image * 2 - 1
    env.reward_model = SimpleNamespace(reset=lambda: None)
    env.image_queue = [[], []]
    env.obs_format = "default"
    env.dataset = [
        {
            "task": "pick trocar",
            "start_items": [
                {
                    "image": torch.full((3, 2, 3), 0.25),
                    "observation.state": torch.full((28,), 2.0),
                    "wm_previous_state": torch.full((28,), 0.5),
                    "wm_previous_action": torch.ones(28),
                }
            ],
        }
    ]
    env._wrap_obs = lambda: {"states": env.current_state}
    obs, info = env.reset(episode_indices=[0, 0])
    assert info == {}
    torch.testing.assert_close(obs["states"], torch.full((2, 28), 2.0))
    torch.testing.assert_close(env.previous_action, torch.ones(2, 28))
    torch.testing.assert_close(env.previous_state, torch.full((2, 28), 0.5))
    torch.testing.assert_close(env.current_obs, torch.full((2, 3, 1, 1, 2, 3), -0.5))
    assert not hasattr(env, "condition_action")


def test_rejects_incorrect_raw_horizon(bridge):
    with pytest.raises(ValueError, match="25 raw actions"):
        bridge.encode_30hz_actions(torch.zeros(1, 10, 43), torch.zeros(1, 43))


def test_milestone_ratchet_has_three_ordered_payouts():
    reward = BatchedMilestoneReward.__new__(BatchedMilestoneReward)
    reward.device = torch.device("cpu")
    reward.num_envs = 1
    reward.reset()
    picked = torch.tensor([[1.0, 0.0, 0.0]])
    handed = torch.tensor([[1.0, 1.0, 0.0]])
    placed = torch.ones(1, 3)

    payouts = [reward.advance(picked).item() for _ in range(13)]
    assert sum(payouts) == 1
    assert reward.stage.item() == 1

    payouts = [reward.advance(handed).item() for _ in range(13)]
    assert sum(payouts) == 1
    assert reward.stage.item() == 2

    payouts = [reward.advance(placed).item() for _ in range(2)]
    assert sum(payouts) == 1
    assert reward.stage.item() == 3

    reward.reset()
    assert reward.stage.item() == 0


@pytest.mark.parametrize("prefix", ["", "action."])
def test_groot_action_converter_accepts_decoded_key_styles(prefix):
    values = {
        f"{prefix}left_arm": torch.zeros(2, 16, 7).numpy(),
        f"{prefix}right_arm": torch.ones(2, 16, 7).numpy(),
        f"{prefix}left_hand": torch.full((2, 16, 7), 2.0).numpy(),
        f"{prefix}right_hand": torch.full((2, 16, 7), 3.0).numpy(),
    }
    converted = convert_to_g1_dex3_action_n1d7(values, chunk_size=12)
    assert converted.shape == (2, 12, 28)
    torch.testing.assert_close(
        torch.from_numpy(converted[0, 0]),
        torch.cat(
            [
                torch.zeros(7),
                torch.ones(7),
                torch.full((7,), 2.0),
                torch.full((7,), 3.0),
            ]
        ),
    )

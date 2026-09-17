# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Action/state parity, grouped resets, KIR and milestone reward regressions."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pytest
import torch

from rlinf.data.datasets.lerobot_world_model import (
    LeRobotV21InitDataset,
    MixedLeRobotV21InitDataset,
)
from rlinf.envs.world_model.dreamdojo_adapters import (
    DREAMDOJO_G1_SLICE,
    G1DreamDojoActionBridge,
    pad_g1_dex3_28_to_43,
    update_g1_dex3_state,
)
from rlinf.envs.world_model.dreamdojo_reward import BatchedMilestoneReward
from rlinf.envs.world_model.world_model_dreamdojo_env import DreamDojoEnv
from rlinf.models.embodiment.gr00t.simulation_io import (
    convert_to_g1_dex3_action_n1d7,
)

# Action/state bridge and reward


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


# Keyframe selection and grouped reset


def dataset(picked=30, handed=90, placed=110):
    ds = LeRobotV21InitDataset.__new__(LeRobotV21InitDataset)
    ds.episodes = [0]
    ds.root = Path("/fixture")
    ds.fps = 30.0
    ds.history_frame_offset = 2
    ds.kir_context_len = 0
    ds.enable_kir = ds.random_start_frame = False
    ds.state_dim = ds.action_dim = 28
    ds.image_size = (2, 3)
    ds._episode_meta = {
        0: {
            "episode_index": 0,
            "length": 120,
            "success": True,
            "tasks": ["trocar"],
            "source_annotation": {
                "cleaned_phase_frames": {
                    "left_hand_pickup": picked,
                    "handover_to_right_hand": handed,
                    "placed_on_plate": placed,
                }
            },
        }
    }
    values = np.arange(120, dtype=np.float32)[:, None].repeat(28, axis=1)
    table = pa.table(
        {"observation.state": values.tolist(), "action": (2 * values).tolist()}
    )
    ds._parquet_path = lambda _: Path("/fixture/parquet")
    ds._read_parquet = lambda _: table
    ds._video_path = lambda _: Path("/fixture/video")
    ds._decode_video_frames = lambda _, indices: (
        np.array(indices, dtype=np.uint8)[:, None, None, None]
        * np.ones((len(indices), 2, 3, 3), dtype=np.uint8)
    )
    return ds


def reward(count=8):
    obj = BatchedMilestoneReward.__new__(BatchedMilestoneReward)
    obj.device = torch.device("cpu")
    obj.num_envs = count
    obj.reset()
    obj._prepare = lambda frames: frames.float() / 255
    return obj


def env(ds, count=8, group_size=8, probability=1.0):
    obj = DreamDojoEnv.__new__(DreamDojoEnv)
    obj.enable_kir = True
    obj.kir_probability = probability
    obj.kir_max_offset_frames = 30
    obj._kir_generator = torch.Generator().manual_seed(100000)
    obj.dataset = ds
    obj.num_envs = count
    obj.group_size = group_size
    obj.device = torch.device("cpu")
    obj.state_dim = 28
    obj.image_size = (2, 3)
    obj.condition_frame_length = 1
    obj.onload = lambda: None
    obj.trans_norm = lambda frames: frames * 2 - 1
    obj.reward_model = reward(count)
    obj.image_queue = [[] for _ in range(count)]
    obj.record_metrics = True
    obj.prev_step_reward = torch.zeros(count)
    obj.success_once = torch.zeros(count, dtype=torch.bool)
    obj.returns = torch.zeros(count)
    return obj


@pytest.mark.parametrize(
    "picked,handed,expected", [(30, 90, 60), (30, 70, 50), (30, 110, 80)]
)
def test_adaptive_handover_start(picked, handed, expected):
    ds = dataset(picked, handed, placed=119)
    assert ds.handover_start_frame(0) == expected
    assert ds.handover_start_frame(0, 10) == handed - 10
    assert ds.handover_start_frame(0, 15) == handed - 15


@pytest.mark.parametrize(
    "change", ["missing", "failure", "unordered", "boundary", "ambiguous"]
)
def test_unreliable_labels_fall_back_without_changing_episode(change):
    ds = dataset()
    meta = ds._episode_meta[0]
    phases = meta["source_annotation"]["cleaned_phase_frames"]
    if change == "missing":
        meta.pop("source_annotation")
    if change == "failure":
        meta["success"] = False
    if change == "unordered":
        phases["left_hand_pickup"] = 100
    if change == "boundary":
        phases["placed_on_plate"] = 120
    if change == "ambiguous":
        phases["handover_to_right_hand"] = 38
    assert ds.handover_start_frame(0) is None
    obj = env(ds)
    obj.reset(episode_indices=[0] * 8)
    assert not obj.reset_is_kir.any()
    assert (obj.reset_start_frames == 2).all()
    assert (obj.reward_model.stage == 0).all()


def test_explicit_frame_aligns_proprio_history_and_only_past_images():
    ds = dataset()
    item = ds.get_at_frame(0, 60, reward_history_frames=17)
    first = item["start_items"][0]
    assert item["dataset_meta"]["start_frame"] == 60
    torch.testing.assert_close(first["observation.state"], torch.full((28,), 60.0))
    torch.testing.assert_close(first["wm_previous_state"], torch.full((28,), 58.0))
    torch.testing.assert_close(first["wm_previous_action"], torch.full((28,), 116.0))
    torch.testing.assert_close(
        item["reward_history"][:, 0, 0, 0], torch.arange(44, 61, dtype=torch.uint8)
    )
    assert not item["target_items"]
    assert ds[0]["dataset_meta"]["start_frame"] == 2


@pytest.mark.parametrize("probability", [0.0, 1.0])
def test_shared_group_images_survive_native_inplace_normalization(probability):
    from torchvision import transforms

    ds = dataset()
    start = 60 if probability else 2
    original = ds.get_at_frame(
        start_frame=start, idx=0, reward_history_frames=17 if probability else 0
    )
    before = original["start_items"][0]["image"].clone()
    ds.get_at_frame = lambda *args, **kwargs: original
    obj = env(ds, probability=probability)
    obj.trans_norm = transforms.Normalize([0.5] * 3, [0.5] * 3, inplace=True)
    obs, _ = obj.reset(episode_indices=[0] * 8)
    torch.testing.assert_close(original["start_items"][0]["image"], before)
    torch.testing.assert_close(
        obj.current_obs[:, :, 0, 0], (before * 2 - 1)[None].repeat(8, 1, 1, 1)
    )
    torch.testing.assert_close(
        obs["main_images"], obs["main_images"][:1].expand_as(obs["main_images"])
    )
    assert obj.current_obs.min() >= -1 and obj.current_obs.max() <= 1


@pytest.mark.parametrize(
    "start,history", [(1, 0), (120, 0), (60.5, 0), (2, 17), (60, -1)]
)
def test_invalid_explicit_frames_fail(start, history):
    with pytest.raises(ValueError):
        dataset().get_at_frame(0, start, reward_history_frames=history)


def test_group_reset_shares_frame_state_and_does_not_pay_for_picked():
    obj = env(dataset())
    obs, _ = obj.reset(episode_indices=[0] * 8)
    assert obj.reset_is_kir.all()
    assert (obj.reset_start_frames == 60).all()
    torch.testing.assert_close(obs["states"], torch.full((8, 28), 60.0))
    assert (obs["main_images"] == obs["main_images"][0]).all()
    assert (obj.reward_model.stage == 1).all()
    assert not obj.returns.any() and not obj.success_once.any()
    assert all(len(h) == 17 for h in obj.reward_model.frame_history)
    assert all(not votes for heads in obj.reward_model.recent for votes in heads)
    assert (
        obj.reward_model.advance(torch.tensor([[1.0, 0.0, 0.0]]).repeat(8, 1)).sum()
        == 0
    )
    payouts = sum(
        obj.reward_model.advance(torch.tensor([[1.0, 1.0, 0.0]]).repeat(8, 1))
        for _ in range(13)
    )
    torch.testing.assert_close(payouts, torch.ones(8))


def test_zero_probability_is_same_as_original_reset():
    ds = dataset()
    original = env(ds)
    original.enable_kir = False
    enabled = env(ds, probability=0)
    obs0, _ = original.reset(episode_indices=[0] * 8)
    obs1, _ = enabled.reset(episode_indices=[0] * 8)
    for key in ("main_images", "states"):
        torch.testing.assert_close(obs0[key], obs1[key])
    for key in ("previous_state", "previous_action", "current_obs"):
        torch.testing.assert_close(getattr(original, key), getattr(enabled, key))
    assert not enabled.reset_is_kir.any()
    assert all(not history for history in enabled.reward_model.frame_history)


def test_group_mismatch_rejected():
    with pytest.raises(ValueError, match="share"):
        env(dataset())._load_reset_items([0] * 7 + [1])


def test_reproducible_group_selection_and_separate_episode_rng():
    a, b = (
        env(dataset(), count=64, probability=0.5),
        env(dataset(), count=64, probability=0.5),
    )
    a._generator = torch.Generator().manual_seed(123)
    before = a._generator.get_state().clone()
    a._load_reset_items([0] * 64)
    b._load_reset_items([0] * 64)
    assert torch.equal(a.reset_is_kir, b.reset_is_kir)
    assert torch.equal(before, a._generator.get_state())
    assert all(len(set(group)) == 1 for group in a.reset_is_kir.reshape(-1, 8).tolist())
    assert 0 < a.reset_is_kir.sum() < 64


def test_mixed_dataset_keeps_flat_index_and_source_metadata():
    mixed = MixedLeRobotV21InitDataset.__new__(MixedLeRobotV21InitDataset)
    mixed.datasets = [dataset(), dataset(20, 60, 110)]
    mixed.offsets = [0, 1, 2]
    assert mixed.handover_start_frame(1) == 40
    item = mixed.get_at_frame(1, 40, reward_history_frames=17)
    assert item["dataset_meta"]["source_index"] == 1
    assert item["dataset_meta"]["start_frame"] == 40
    with pytest.raises(IndexError):
        mixed.get_at_frame(2)


def test_warmup_only_changes_selected_environments_and_no_future_votes():
    obj = reward(2)
    frames = torch.zeros(1, 17, 3, 2, 3, dtype=torch.uint8)
    obj.prime_handover(frames, [1])
    assert obj.stage.tolist() == [0, 1]
    assert len(obj.frame_history[0]) == 0 and len(obj.frame_history[1]) == 17
    assert not any(obj.recent[1])
    with pytest.raises(ValueError):
        obj.prime_handover(frames, [2])
    with pytest.raises(ValueError):
        obj.prime_handover(frames[:, :16], [1])


def test_kir_settings_reject_eval_and_uncontrolled_random_start():
    from omegaconf import OmegaConf

    from toolkits.world_model.dreamdojo_validation import config

    ec = config().env.train
    ec.enable_kir = True
    for key in ("is_eval", "random_start_frame"):
        candidate = OmegaConf.create(OmegaConf.to_container(ec))
        candidate[key] = True
        with pytest.raises(ValueError, match="train-only"):
            DreamDojoEnv(candidate, 8, 0, 1)


def test_stage_initialization_does_not_count_as_episode_success_or_return():
    obj = env(dataset())
    obj.reset(episode_indices=[0] * 8)
    obj.elapsed_steps = 12
    obj.last_chunk_probs = torch.zeros(8, 6, 3)
    info = obj._record_metrics(torch.zeros(8), torch.zeros(8, dtype=torch.bool), {})[
        "episode"
    ]
    assert info["picked"].all()
    assert not info["handed"].any() and not info["success_once"].any()
    assert not info["return"].any()
    assert info["kir_fraction"].sum() == 8


def test_runtime_mounts_and_chain_guard_cover_same_explicit_kir_files():
    root = Path(__file__).resolve().parents[3]
    launcher = (root / "docker/dreamdojo/run_cluster.slurm").read_text()
    chain = (root / "docker/dreamdojo/chain.py").read_text()
    for path in (
        "rlinf/data/datasets/lerobot_world_model.py",
        "rlinf/envs/world_model/world_model_dreamdojo_env.py",
        "rlinf/envs/world_model/dreamdojo_reward.py",
        "examples/embodiment/config/env/dreamdojo_trocar.yaml",
        "toolkits/world_model/dreamdojo_validation.py",
    ):
        assert path in launcher and path in chain
    assert (
        "${DREAMDOJO_KIR_SOURCE_ROOT}/${relative}:/opt/src/RLinf/${relative}:ro"
        in launcher
    )

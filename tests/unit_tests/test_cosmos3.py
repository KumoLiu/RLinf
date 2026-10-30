# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Cosmos3 command alignment, GR00T conversion and classifier reward contracts."""

import json
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType, SimpleNamespace

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image
from torchvision.models import resnet18

from rlinf.algorithms.registry import calculate_adv_and_returns
from rlinf.data.datasets.lerobot_world_model import (
    LeRobotV21InitDataset,
)
from rlinf.envs.action_utils import prepare_actions
from rlinf.envs.sim.world_model.classifier_reward import ClassifierReward
from rlinf.envs.sim.world_model.cosmos3_env import Cosmos3Env
from rlinf.envs.wrappers.record_video import RecordVideo
from rlinf.models.embodiment.gr00t.simulation_io import (
    convert_h2_sharpa_wm_obs_to_gr00t_format,
    convert_to_h2_sharpa_action_n1d7,
)
from rlinf.scheduler import Worker
from rlinf.workers.actor.embodied_fsdp_actor_worker import EmbodiedFSDPActor


@pytest.fixture(autouse=True)
def cpu_device(monkeypatch):
    monkeypatch.setattr(Worker, "torch_device_type", "cpu")


@pytest.fixture
def reward(tmp_path):
    torch.manual_seed(3)
    model = resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 5)
    checkpoint = tmp_path / "classifier.pt"
    torch.save(
        {
            "architecture": "resnet18",
            "classes": list(range(5)),
            "model": model.state_dict(),
            "image_size": 32,
            "normalization_mean": [0.485, 0.456, 0.406],
            "normalization_std": [0.229, 0.224, 0.225],
        },
        checkpoint,
    )
    return ClassifierReward(checkpoint, 1, "cpu")


def test_classifier_matches_published_pil_preprocessing(reward):
    frame = np.random.default_rng(42).integers(0, 256, (19, 27, 3), dtype=np.uint8)
    resized = np.asarray(
        Image.fromarray(frame).resize((32, 32), Image.Resampling.BILINEAR)
    ).copy()
    inputs = torch.from_numpy(resized).permute(2, 0, 1)[None].float() / 255
    with torch.no_grad():
        expected = reward.model((inputs - reward.mean) / reward.std).softmax(-1)
    scores, probabilities = reward.score_chunk(
        torch.from_numpy(frame).permute(2, 0, 1)[None, None]
    )
    torch.testing.assert_close(probabilities[:, 0], expected)
    torch.testing.assert_close(scores[:, 0], expected @ torch.arange(5).float())


class BrightnessClassifier(torch.nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, inputs):
        brightness = inputs[:, 0].mean((1, 2)) * 0.229 + 0.485
        label = (
            (brightness * (self.num_classes - 1))
            .round()
            .long()
            .clamp(0, self.num_classes - 1)
        )
        return torch.nn.functional.one_hot(label, self.num_classes).float() * 50


def test_classifier_scores_can_decrease_and_reset(reward):
    reward.model = BrightnessClassifier()
    frames = torch.stack([torch.ones(3, 8, 8), -torch.ones(3, 8, 8)])[None]
    scores, _ = reward.score_chunk(frames)
    torch.testing.assert_close(scores, torch.tensor([[4.0, 0.0]]), atol=1e-6, rtol=0)
    assert reward.labels.item() == 0
    reward.to("cpu").reset()
    assert reward.scores.item() == 0


@pytest.fixture
def fourclass_checkpoint(tmp_path):
    torch.manual_seed(9)
    model = resnet18(weights=None)
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(512, 256),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.1),
        torch.nn.Linear(256, 4),
    )
    model.eval()
    folder = tmp_path / "fourclass"
    folder.mkdir()
    checkpoint = folder / "reward_model.pt"
    torch.save({"backbone." + k: v for k, v in model.state_dict().items()}, checkpoint)
    OmegaConf.save(
        OmegaConf.create(
            {
                "num_classes": 4,
                "arch": "resnet18",
                "pretrained": False,
                "hidden_dim": 256,
                "dropout": 0.1,
                "image_size": [3, 224, 224],
                "normalize": True,
                "precision": "fp32",
                "model_path": str(checkpoint),
            }
        ),
        folder / "inference_config.yaml",
    )
    return checkpoint, model


@pytest.mark.parametrize("float_frames", [False, True])
def test_fourclass_matches_native_tensor_preprocessing(
    fourclass_checkpoint, float_frames
):
    checkpoint, native_model = fourclass_checkpoint
    reward = ClassifierReward(checkpoint, 2)
    frames = torch.randint(0, 256, (2, 2, 3, 37, 49), dtype=torch.uint8)
    inputs = torch.nn.functional.interpolate(
        frames.flatten(0, 1).float() / 255,
        size=(224, 224),
        mode="bilinear",
        align_corners=False,
    )
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    with torch.no_grad():
        expected = native_model((inputs - mean) / std).softmax(-1).reshape(2, 2, 4)
    if float_frames:
        frames = frames.float() / 127.5 - 1
    with torch.autocast("cpu", dtype=torch.bfloat16):
        scores, probabilities = reward.score_chunk(frames)
    assert reward.num_classes == 4 and reward.image_size == (224, 224)
    assert not reward.model.training and probabilities.dtype == torch.float32
    torch.testing.assert_close(probabilities, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        scores, (expected * torch.arange(4)).sum(-1), rtol=0, atol=0
    )
    torch.testing.assert_close(reward.labels, expected[:, -1].argmax(-1))


def test_fourclass_rejects_wrong_inference_config(fourclass_checkpoint):
    checkpoint, _ = fourclass_checkpoint
    config = checkpoint.with_name("inference_config.yaml")
    cfg = OmegaConf.load(config)
    cfg.num_classes = 5
    OmegaConf.save(cfg, config)
    with pytest.raises(ValueError, match="five-class reward_values"):
        ClassifierReward(checkpoint, 1)


@pytest.mark.parametrize("prefix", ["", "action."])
def test_h2_gr00t_round_trip_keeps_physical_joint_targets(prefix):
    states = torch.arange(58).float()[None]
    obs = convert_h2_sharpa_wm_obs_to_gr00t_format(
        {
            "main_images": torch.zeros(1, 2, 3, 3, dtype=torch.uint8),
            "states": states,
            "task_descriptions": ["unfold pad"],
        }
    )
    actions = {
        prefix + key.removeprefix("state."): value.repeat(16, axis=1)
        for key, value in obs.items()
        if key.startswith("state.")
    }
    decoded = convert_to_h2_sharpa_action_n1d7(actions)
    prepared = prepare_actions(decoded, "cosmos3_wm", "gr00t_n1d7", 16, 58)
    np.testing.assert_array_equal(prepared, states[:, None].repeat(1, 16, 1).numpy())
    assert obs["video.high"].shape == (1, 1, 2, 3, 3)


@pytest.mark.parametrize("with_modality", [False, True])
def test_lerobot_reset_uses_causal_command_anchor(tmp_path, with_modality):
    (tmp_path / "meta").mkdir()
    (tmp_path / "data").mkdir()
    (tmp_path / "meta/info.json").write_text(
        json.dumps(
            {
                "fps": 30,
                "total_episodes": 1,
                "chunks_size": 1000,
                "data_path": "data/episode_{episode_index:06d}.parquet",
                "video_path": "unused.mp4",
                "features": {"observation.images.cam_left_high": {"dtype": "video"}},
            }
        )
    )
    if with_modality:
        (tmp_path / "meta/modality.json").write_text(
            json.dumps(
                {
                    "video": {
                        "high": {"original_key": "observation.images.cam_left_high"}
                    },
                }
            )
        )
    pq.write_table(
        pa.table(
            {
                "observation.state": [[1.0] * 58, [2.0] * 58, [3.0] * 58],
                "action": [[10.0] * 58, [20.0] * 58, [30.0] * 58],
            }
        ),
        tmp_path / "data/episode_000000.parquet",
    )
    with av.open(str(tmp_path / "unused.mp4"), "w") as container:
        stream = container.add_stream("libx264", rate=30)
        stream.width = stream.height = 16
        stream.pix_fmt = "yuv420p"
        for _ in range(3):
            frame = av.VideoFrame.from_ndarray(
                np.zeros((16, 16, 3), dtype=np.uint8), format="rgb24"
            )
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    dataset = LeRobotV21InitDataset(
        str(tmp_path),
        video_key="high" if with_modality else "observation.images.cam_left_high",
        state_dim=58,
        action_dim=58,
        video_backend="pyav",
    )
    initial = dataset.get_at_frame(0, 0)["start_items"][0]
    later = dataset.get_at_frame(0, 1)["start_items"][0]
    torch.testing.assert_close(initial["wm_previous_action"], torch.ones(58))
    torch.testing.assert_close(later["wm_previous_action"], torch.full((58,), 10.0))


@pytest.fixture
def cosmos_sdk(monkeypatch):
    """Represent the external Cosmos SDK while running the actual RLinf classes."""

    class Model:
        input_video_key = "video"

        def __init__(self):
            self.calls = []
            self.last_frame = 1.0
            self.tokenizer_vision_gen = SimpleNamespace(
                model=SimpleNamespace(
                    model=SimpleNamespace(to=lambda device: None),
                    clear_encoder_cache=lambda: None,
                )
            )

        def generate_samples_from_batch(self, batch, **kwargs):
            self.calls.append((batch, kwargs))
            frames = (
                torch.linspace(-1, 1, 17)[None, None, :, None, None]
                .expand(1, 3, 17, 16, 16)
                .clone()
            )
            frames[:, :, -1] = self.last_frame
            if hasattr(self, "generated_frames"):
                frames[:, :, 1:] = torch.tensor(self.generated_frames)[
                    None, None, :, None, None
                ]
            return {"vision": [frames.clone() for _ in batch["parts"]]}

        def decode(self, latent):
            return latent

        def to(self, device):
            return self

    model = Model()

    class Setup:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def build_setup(self):
            return self.kwargs

    class Codec:
        def __init__(self, path, checksum):
            self.scale = np.array(
                json.loads(open(path).read())["scale"], dtype=np.float32
            )

        def encode(self, actions, anchor):
            return (actions - anchor) * self.scale

    modules = {
        "cosmos_framework.inference.args": {
            "OmniSetupOverrides": Setup,
            "ModelMode": SimpleNamespace(FORWARD_DYNAMICS="fd"),
        },
        "cosmos_framework.inference.inference": {
            "OmniInference": SimpleNamespace(
                create=lambda setup: SimpleNamespace(model=model)
            ),
            "_merge_data_batches": lambda parts: {"parts": parts},
            "_finalize_data_batch": lambda batch, n, model: batch,
        },
        "cosmos_framework.inference.action": {
            "build_action_batch": lambda **kwargs: kwargs
        },
        "cosmos_framework.utils.h2_action_codec": {
            "H2ActionCodec": Codec,
            "RELATIVE_REPRESENTATION": "relative_block16",
        },
    }
    for name in [
        "cosmos_framework",
        "cosmos_framework.inference",
        "cosmos_framework.utils",
    ]:
        m = ModuleType(name)
        m.__path__ = []
        monkeypatch.setitem(sys.modules, name, m)
    for name, members in modules.items():
        m = ModuleType(name)
        vars(m).update(members)
        monkeypatch.setitem(sys.modules, name, m)
    return model


@pytest.fixture
def env_config(tmp_path, reward):
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    (root / "data").mkdir()
    joints = [f"joint_{i}" for i in range(58)]
    (root / "meta/info.json").write_text(
        json.dumps(
            {
                "fps": 30,
                "total_episodes": 3,
                "chunks_size": 1000,
                "data_path": "data/episode_{episode_index:06d}.parquet",
                "video_path": "initial.mp4",
                "features": {
                    "observation.images.cam_left_high": {"dtype": "video"},
                    "action": {"names": joints},
                    "observation.state": {"names": joints},
                },
            }
        )
    )
    (root / "meta/split.json").write_text(
        json.dumps(
            {"episodes": [{"episode_index": i, "split": "train"} for i in range(3)]}
        )
    )
    (root / "meta/action_stats.json").write_text(
        json.dumps(
            {
                "joint_names": joints,
                "representation": "relative_block16",
                "scale": [2.0] * 58,
            }
        )
    )
    (root / "meta/episodes.jsonl").write_text(
        "".join(
            json.dumps({"episode_index": i, "length": 3, "tasks": [f"unfold pad {i}"]})
            + "\n"
            for i in range(3)
        )
    )
    for i in range(3):
        pq.write_table(
            pa.table(
                {
                    "observation.state": [[float(i + 1)] * 58] * 3,
                    "action": [[10.0 + i] * 58, [20.0 + i] * 58, [30.0 + i] * 58],
                }
            ),
            root / f"data/episode_{i:06d}.parquet",
        )
    with av.open(str(root / "initial.mp4"), "w") as container:
        stream = container.add_stream("libx264", rate=30)
        stream.width = stream.height = 16
        stream.pix_fmt = "yuv420p"
        for _ in range(3):
            frame = av.VideoFrame.from_ndarray(
                np.zeros((16, 16, 3), dtype=np.uint8), format="rgb24"
            )
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    model_config = tmp_path / "config.yaml"
    model_config.write_text(
        OmegaConf.to_yaml(
            OmegaConf.create(
                {
                    "dataloader_train": {
                        "dataloader": {
                            "datasets": {
                                "h2_sharpa": {
                                    "dataset": {
                                        "chunk_length": 16,
                                        "action_representation": "relative_block16",
                                        "camera_key": "observation.images.cam_left_high",
                                        "action_stats_sha256": "fixture",
                                    }
                                }
                            }
                        }
                    }
                }
            )
        )
    )
    return OmegaConf.create(
        {
            "env_type": "cosmos3_wm",
            "backend": "cosmos3",
            "seed": 7,
            "auto_reset": False,
            "ignore_terminations": True,
            "use_rel_reward": True,
            "use_fixed_reset_state_ids": False,
            "group_size": 2,
            "action_dim": 58,
            "state_dim": 58,
            "max_episode_steps": 32,
            "reward_coef": 1.0,
            "is_eval": False,
            "enable_kir": False,
            "video_cfg": {"save_video": True},
            "initial_image_path": str(root),
            "split_path": str(root / "meta/split.json"),
            "split": "train",
            "action_statistics": str(root / "meta/action_stats.json"),
            "cosmos3_root": str(tmp_path),
            "model_path": "fixture",
            "config_file": str(model_config),
            "pipeline_log_dir": str(tmp_path / "logs"),
            "video_key": "observation.images.cam_left_high",
            "video_backend": "pyav",
            "chunk": 16,
            "num_frames": 17,
            "condition_frame_length": 1,
            "image_size": [16, 16],
            "num_inference_steps": 30,
            "wm_batch_size": 1,
            "guidance": 1.0,
            "shift": 10.0,
            "reward_model": {"from_pretrained": str(tmp_path / "classifier.pt")},
        }
    )


def test_cosmos_backend_closes_policy_loop_and_keeps_noise_seeds(
    env_config, cosmos_sdk
):
    env = Cosmos3Env(env_config, 2, 0, 1)
    env.reward_model.model = BrightnessClassifier()
    initial, _ = env.reset()
    anchors = initial["states"].clone()
    commands = torch.arange(16).float()[None, :, None].expand(2, 16, 58) + 13
    obs, rewards, terms, truncs, infos = env.chunk_step(commands)
    assert [c[1]["seed"] for c in cosmos_sdk.calls] == [[7], [8]]
    torch.testing.assert_close(
        cosmos_sdk.calls[0][0]["parts"][0]["action"], (commands[0] - anchors[0]) * 2
    )
    torch.testing.assert_close(obs[0]["states"], commands[:, -1])
    torch.testing.assert_close(rewards.sum(1), torch.full((2,), 4.0), atol=1e-5, rtol=0)
    assert not terms.any() and not truncs.any()
    assert infos[0]["episode"]["success_once"].all()
    assert env.capture_image().shape == (2, 16, 16, 16, 3)
    _, rewards, _, truncs, _ = env.chunk_step(commands + 1)
    assert [c[1]["seed"] for c in cosmos_sdk.calls[-2:]] == [[23], [24]]
    torch.testing.assert_close(
        cosmos_sdk.calls[-2][0]["parts"][0]["action"],
        (commands[0] + 1 - commands[0, -1]) * 2,
    )
    assert (rewards < 0).any()
    assert truncs[:, -1].all()


def test_success_uses_endpoint_argmax_not_any_frame_threshold(env_config, cosmos_sdk):
    env_config.ignore_terminations = False
    env = Cosmos3Env(env_config, 2, 0, 1)
    env.reward_model.model = BrightnessClassifier()
    env.reset()
    cosmos_sdk.last_frame = -1.0
    commands = torch.zeros(2, 16, 58)
    _, _, terms, _, infos = env.chunk_step(commands)
    assert env.last_chunk_probs[:, :, 4].max() > 0.99
    assert not terms.any() and not infos[0]["episode"]["success_once"].any()


@pytest.mark.parametrize("ignore_terminations", [False, True])
def test_fourclass_environment_uses_class_three_and_signed_delta(
    env_config,
    cosmos_sdk,
    fourclass_checkpoint,
    ignore_terminations,
):
    env_config.reward_model.from_pretrained = str(fourclass_checkpoint[0])
    env_config.ignore_terminations = ignore_terminations
    env = Cosmos3Env(env_config, 2, 0, 1)
    env.reward_model.model = BrightnessClassifier(4)
    env.reset()
    actions = torch.zeros(2, 16, 58)
    _, rewards, terms, _, infos = env.chunk_step(actions)
    assert env.last_chunk_probs.shape == (2, 16, 4) and env.success_label == 3
    assert infos[0]["episode"]["success_once"].all()
    torch.testing.assert_close(infos[0]["episode"]["success_prob_max"], torch.ones(2))
    assert terms[:, -1].tolist() == [not ignore_terminations] * 2
    torch.testing.assert_close(rewards.sum(1), torch.full((2,), 3.0), atol=1e-5, rtol=0)
    cosmos_sdk.last_frame = -1.0
    _, rewards, _, _, _ = env.chunk_step(actions)
    assert (rewards < 0).any()
    torch.testing.assert_close(
        rewards.sum(1), torch.full((2,), -3.0), atol=1e-5, rtol=0
    )
    env.reset()
    _, _, terms, _, infos = env.chunk_step(actions)
    assert env.last_chunk_probs[:, :, 3].max() > 0.99
    assert not terms.any() and not infos[0]["episode"]["success_once"].any()


@pytest.mark.parametrize("after_reset", [False, True])
def test_world_model_offload_round_trip(
    env_config, cosmos_sdk, monkeypatch, after_reset
):
    monkeypatch.setattr(
        Worker, "torch_platform", SimpleNamespace(empty_cache=lambda: None)
    )
    env = Cosmos3Env(env_config, 2, 0, 1)
    if after_reset:
        env.reset(episode_indices=[0, 1])
        env.chunk_step(torch.ones(2, 16, 58))
    ids = env.reset_state_ids.clone()
    observation = env.current_obs.clone() if after_reset else None
    score = env.reward_model.scores.clone()
    env.offload()
    env.offload()
    assert env._is_offloaded
    env.onload()
    env.onload()
    assert not env._is_offloaded
    torch.testing.assert_close(env.reset_state_ids, ids)
    torch.testing.assert_close(env.reward_model.scores, score)
    if after_reset:
        torch.testing.assert_close(env.current_obs, observation)
        env.chunk_step(torch.ones(2, 16, 58) * 2)
        assert cosmos_sdk.calls[-2][1]["seed"] == [23]


def test_group_sampler_matches_original_seed_stream(env_config, cosmos_sdk):
    env = Cosmos3Env(env_config, 4, 1, 1)
    generator = torch.Generator().manual_seed(8)
    for _ in range(3):
        expected = (
            torch.randint(0, 3, (2,), generator=generator).repeat_interleave(2).tolist()
        )
        assert env._sample_reset_episode_indices() == expected


def test_unique_evaluation_counts_only_valid_slots(env_config, cosmos_sdk, tmp_path):
    env_config.group_size = 1
    env_config.is_eval = True
    env_config.use_fixed_reset_state_ids = True
    env_config.eval_unique_episodes = True
    env_config.evaluation_result_dir = str(tmp_path / "eval-results")
    env = Cosmos3Env(env_config, 2, 1, 2)
    env.reward_model.model = BrightnessClassifier()
    env.reset()
    assert env._sample_reset_episode_indices() == [2, 0]
    commands = torch.zeros(2, 16, 58)
    env.chunk_step(commands)
    _, _, _, truncs, _ = env.chunk_step(commands)
    assert truncs[:, -1].tolist() == [True, False]
    rows = [
        json.loads(line)
        for path in (tmp_path / "eval-results").glob("episodes-*.jsonl")
        for line in path.read_text().splitlines()
    ]
    assert len(rows) == 1
    assert rows[0]["episode_index"] == 2
    assert rows[0]["elapsed_steps"] == 32
    assert len(rows[0]["final_probabilities"]) == 5


def test_full_horizon_video_records_every_generated_frame(env_config, cosmos_sdk):
    env_config.max_episode_steps = 480
    env_config.max_steps_per_rollout_epoch = 480
    env_config.video_cfg.save_video = True
    env_config.video_cfg.info_on_video = False
    env = Cosmos3Env(env_config, 2, 0, 1)
    video = RecordVideo(env, env_config.video_cfg)
    try:
        video.reset()
        assert len(video.render_images) == 1
        for chunk in range(30):
            video.chunk_step(torch.ones(2, 16, 58))
            assert len(video.render_images) == 1 + (chunk + 1) * 16
        assert len(video.render_images) == 481
        assert video._fps == 30
        assert env.elapsed_steps.tolist() == [480, 480]
    finally:
        video._executor.shutdown(wait=True)


def process_batch(ignore_terminations, reward_type, epochs=1):
    cfg = OmegaConf.create(
        {
            "env": {
                "train": {
                    "rollout_epoch": epochs,
                    "auto_reset": False,
                    "ignore_terminations": ignore_terminations,
                }
            },
            "algorithm": {"reward_type": reward_type, "filter_rewards": False},
        }
    )
    # First branch reaches 4 then refolds to 1; second improves from 2 to 3.
    rewards = torch.zeros(2, 2, 16)
    rewards[0, :, -1] = torch.tensor([4.0, 2.0])
    rewards[1, :, -1] = torch.tensor([-3.0, 1.0])
    dones = torch.zeros(3, 2, 16, dtype=torch.bool)
    dones[-1, :, -1] = True
    if not ignore_terminations:
        dones[1, 0, -1] = True
    batch = {
        "rewards": torch.cat([rewards] * epochs),
        "dones": torch.cat([dones] * epochs),
    }
    processed = EmbodiedFSDPActor._process_received_rollout_batch(
        SimpleNamespace(cfg=cfg), batch
    )
    result = calculate_adv_and_returns(
        task_type="embodied",
        adv_type="grpo",
        group_size=2,
        reward_type=reward_type,
        rewards=processed["rewards"],
        dones=processed["dones"],
        loss_mask=processed["loss_mask"],
        loss_mask_sum=processed["loss_mask_sum"],
        values=None,
    )
    return processed, result["advantages"]


def test_early_termination_masking_remains_unchanged():
    batch, advantages = process_batch(False, "action_level")
    assert batch["loss_mask"][0].all()
    assert not batch["loss_mask"][1, 0].any()
    assert batch["loss_mask"][1, 1].all()
    assert (advantages[0, 0] > 0).all()
    assert (advantages[1, 0] == 0).all()


@pytest.fixture
def signed_checkpoint(fourclass_checkpoint):
    checkpoint, model = fourclass_checkpoint
    model.fc[-1] = torch.nn.Linear(256, 5)
    torch.save({"backbone." + k: v for k, v in model.state_dict().items()}, checkpoint)
    config = checkpoint.with_name("inference_config.yaml")
    cfg = OmegaConf.load(config)
    cfg.num_classes = 5
    cfg.reward_values = [-1, 0, 1, 2, 3]
    OmegaConf.save(cfg, config)
    return checkpoint, model.eval()


def test_signed_fiveclass_preprocessing(signed_checkpoint):
    checkpoint, model = signed_checkpoint
    reward = ClassifierReward(checkpoint, 1)
    frames = torch.randint(0, 256, (1, 2, 3, 37, 49), dtype=torch.uint8)
    inputs = frames.flatten(0, 1).float() / 255
    for size in [(240, 320), (224, 224)]:
        inputs = torch.nn.functional.interpolate(
            inputs, size=size, mode="bilinear", align_corners=False
        )
    with torch.no_grad():
        probs = model((inputs - reward.mean) / reward.std).softmax(-1).reshape(1, 2, 5)
    scores, actual = reward.score_chunk(frames)
    torch.testing.assert_close(actual, probs, rtol=0, atol=0)
    torch.testing.assert_close(
        scores, (probs * torch.tensor([-1, 0, 1, 2, 3])).sum(-1), rtol=0, atol=0
    )


@pytest.mark.parametrize("terminal_value,success", [(-1.0, False), (1.0, True)])
def test_streak_cross_chunk_and_padding(
    env_config, cosmos_sdk, signed_checkpoint, terminal_value, success
):
    env_config.reward_model.from_pretrained = str(signed_checkpoint[0])
    env_config.ignore_terminations = False
    env_config.terminal_streak_frames = 15
    env_config.max_episode_steps = 96
    env = Cosmos3Env(env_config, 2, 0, 1)
    env.reward_model.model = BrightnessClassifier()
    env.reset()
    command = torch.arange(16).float()[None, :, None].expand(2, 16, 58)
    cosmos_sdk.generated_frames = [terminal_value] * 10 + [-0.5] * 6
    _, _, term, _, _ = env.chunk_step(command)
    assert not term.any()
    cosmos_sdk.generated_frames = [-0.5] * 8 + [terminal_value] * 8
    _, _, term, _, _ = env.chunk_step(command)
    assert not term.any()
    cosmos_sdk.generated_frames = [terminal_value] * 7 + [-terminal_value] * 9
    _, rew, term, trunc, info = env.chunk_step(command)
    assert term[:, 6].all() and term.sum() == 2 and not trunc.any()
    assert env.elapsed_steps.tolist() == [39, 39]
    assert not rew[:, 7:].any()
    assert env.current_state[:, 0].tolist() == [6, 6]
    assert info[0]["episode"]["success_final"].tolist() == [success, success]
    calls = len(cosmos_sdk.calls)
    _, rew, term, trunc, info = env.chunk_step(command)
    assert len(cosmos_sdk.calls) == calls
    assert not rew.any() and not term.any() and not trunc.any()
    assert info[0]["episode"]["success_final"].tolist() == [success, success]
    assert env.elapsed_steps.tolist() == [39, 39]
    env.reset()
    assert not env._terminal.any() and not env._failure_streak.any()


def test_quality_sampling_balances_groups(env_config, cosmos_sdk):
    path = env_config.split_path
    split = json.loads(open(path).read())
    for i, row in enumerate(split["episodes"]):
        row["review_status"] = "success" if i < 2 else "imperfect2"
    open(path, "w").write(json.dumps(split))
    env_config.balanced_quality_sampling = True
    env = Cosmos3Env(env_config, 4, 0, 1)
    for _ in range(30):
        indices = env._sample_reset_episode_indices()
        assert indices[0] == indices[1] and indices[2] == indices[3]
        assert indices[0] in (0, 1) and indices[2] == 2


def test_terminal_frame_included_in_actor_mask():
    from rlinf.utils.metric_utils import compute_loss_mask

    rewards = torch.zeros(3, 2, 16)
    rewards[0, :, 0] = 1
    rewards[1, 0, 6] = -2
    rewards[2, 1, 15] = 2
    dones = torch.zeros(4, 2, 16, dtype=torch.bool)
    dones[2, 0, 6] = True
    dones[3, 1, 15] = True
    mask, total = compute_loss_mask(dones)
    assert mask[1, 0, :7].all() and not mask[1, 0, 7:].any() and not mask[2, 0].any()
    assert total[0, 0, 0] == 23 and total[0, 1, 0] == 48
    out = calculate_adv_and_returns(
        task_type="embodied",
        adv_type="grpo",
        group_size=2,
        reward_type="action_level",
        rewards=rewards,
        dones=dones,
        loss_mask=mask,
        loss_mask_sum=total,
        values=None,
    )
    assert (out["advantages"][1, 0, :7] < 0).all()
    assert not out["advantages"][1, 0, 7:].any()


def test_canonical_camera_resolves_dataset_alias(env_config, cosmos_sdk):
    from pathlib import Path

    path = Path(env_config.initial_image_path) / "meta/modality.json"
    path.write_text(
        json.dumps(
            {"video": {"high": {"original_key": "observation.images.cam_left_high"}}}
        )
    )
    env = Cosmos3Env(env_config, 2, 0, 1)
    assert env.dataset.video_key == "high"
    assert env.dataset.video_original_key == env_config.video_key
    obs, _ = env.reset()
    assert obs["main_images"].shape[0] == 2


def terminal30_env(
    env_config, cosmos_sdk, signed_checkpoint, horizon=32, timeout_final3=True
):
    env_config.reward_model.from_pretrained = str(signed_checkpoint[0])
    env_config.ignore_terminations = False
    env_config.terminal_streak_frames = 30
    env_config.timeout_success_if_final_label3 = timeout_final3
    env_config.max_episode_steps = horizon
    env = Cosmos3Env(env_config, 2, 0, 1)
    env.reward_model.model = BrightnessClassifier()
    env.reset()
    return env


@pytest.mark.parametrize("value,is_success", [(-1.0, False), (1.0, True)])
def test_terminal30_cross_chunk_exact_boundary(
    env_config, cosmos_sdk, signed_checkpoint, value, is_success
):
    env = terminal30_env(env_config, cosmos_sdk, signed_checkpoint, horizon=96)
    action = torch.zeros(2, 16, 58)
    cosmos_sdk.generated_frames = [value] * 16
    _, _, term, trunc, _ = env.chunk_step(action)
    assert not term.any() and not trunc.any()
    _, reward, term, trunc, info = env.chunk_step(action)
    assert term[:, 13].all() and term.sum() == 2 and not trunc.any()
    assert env.elapsed_steps.tolist() == [30, 30]
    assert info[0]["episode"]["success_final"].tolist() == [is_success] * 2
    assert info[0]["episode"]["success_by_streak"].tolist() == [is_success] * 2
    assert not info[0]["episode"]["success_at_timeout_final3"].any()
    assert not reward[:, 14:].any()


@pytest.mark.parametrize(
    "last_value,is_success",
    [(-1.0, False), (-0.5, False), (0.0, False), (0.5, False), (1.0, True)],
)
def test_timeout_success_depends_only_on_last_valid_label(
    env_config, cosmos_sdk, signed_checkpoint, last_value, is_success
):
    env = terminal30_env(env_config, cosmos_sdk, signed_checkpoint)
    action = torch.zeros(2, 16, 58)
    cosmos_sdk.generated_frames = [-0.5] * 16
    env.chunk_step(action)
    cosmos_sdk.generated_frames = [1.0] * 15 + [last_value]
    _, _, term, trunc, info = env.chunk_step(action)
    assert not term.any() and trunc[:, 15].all()
    assert env.elapsed_steps.tolist() == [32, 32]
    ep = info[0]["episode"]
    assert ep["success_final"].tolist() == [is_success] * 2
    assert ep["success_at_timeout_final3"].tolist() == [is_success] * 2
    assert not ep["success_by_streak"].any()
    assert ep["time_limit_reached"].all()
    assert ep["timeout"].tolist() == [not is_success] * 2


def test_failure30_cannot_be_rescued_by_future_padding(
    env_config, cosmos_sdk, signed_checkpoint
):
    env = terminal30_env(env_config, cosmos_sdk, signed_checkpoint)
    action = torch.zeros(2, 16, 58)
    cosmos_sdk.generated_frames = [-1.0] * 16
    env.chunk_step(action)
    cosmos_sdk.generated_frames = [-1.0] * 14 + [1.0] * 2
    _, reward, term, trunc, info = env.chunk_step(action)
    assert term[:, 13].all() and not trunc.any()
    assert env.elapsed_steps.tolist() == [30, 30]
    assert info[0]["episode"]["failure"].all()
    assert not info[0]["episode"]["success_final"].any()
    assert not info[0]["episode"]["success_at_timeout_final3"].any()
    assert not info[0]["episode"]["time_limit_reached"].any()
    assert not reward[:, 14:].any()
    calls = len(cosmos_sdk.calls)
    _, reward, term, trunc, info = env.chunk_step(action)
    assert (
        len(cosmos_sdk.calls) == calls
        and not reward.any()
        and not term.any()
        and not trunc.any()
    )
    assert (
        info[0]["episode"]["failure"].all()
        and not info[0]["episode"]["success_final"].any()
    )


def test_streak30_at_time_limit_keeps_success_reason(
    env_config, cosmos_sdk, signed_checkpoint
):
    env = terminal30_env(env_config, cosmos_sdk, signed_checkpoint, horizon=30)
    action = torch.zeros(2, 16, 58)
    cosmos_sdk.generated_frames = [1.0] * 16
    env.chunk_step(action)
    _, _, term, trunc, info = env.chunk_step(action)
    assert term[:, 13].all() and not trunc.any()
    assert info[0]["episode"]["success_by_streak"].all()
    assert not info[0]["episode"]["success_at_timeout_final3"].any()


def test_timeout_final3_policy_is_opt_in(env_config, cosmos_sdk, signed_checkpoint):
    env = terminal30_env(
        env_config, cosmos_sdk, signed_checkpoint, timeout_final3=False
    )
    action = torch.zeros(2, 16, 58)
    cosmos_sdk.generated_frames = [-0.5] * 16
    env.chunk_step(action)
    cosmos_sdk.generated_frames = [-0.5] * 15 + [1.0]
    _, _, term, trunc, info = env.chunk_step(action)
    assert trunc[:, 15].all() and not term.any()
    assert not info[0]["episode"]["success_final"].any()
    assert info[0]["episode"]["timeout"].all()


@pytest.fixture
def cls5_checkpoint(signed_checkpoint):
    checkpoint, model = signed_checkpoint
    path = checkpoint.with_name("inference_config.yaml")
    cfg = OmegaConf.load(path)
    cfg.reward_values = [-2, -1, 0, 1, 2]
    cfg.pre_resize = None
    cfg.channels_last = True
    OmegaConf.save(cfg, path)
    return checkpoint, model


def cls5_env(env_config, cosmos_sdk, cls5_checkpoint, horizon=96):
    env_config.reward_model.from_pretrained = str(cls5_checkpoint[0])
    env_config.ignore_terminations = False
    env_config.terminal_streak_frames = 30
    env_config.timeout_success_if_final_success = True
    env_config.max_episode_steps = horizon
    env = Cosmos3Env(env_config, 2, 0, 1)
    env.reward_model.model = BrightnessClassifier()
    env.reset()
    return env


@pytest.mark.parametrize("value", [-1.0, -0.5])
def test_cls5_both_negative_scores_terminate_at_thirty(
    env_config, cosmos_sdk, cls5_checkpoint, value
):
    env = cls5_env(env_config, cosmos_sdk, cls5_checkpoint)
    action = torch.zeros(2, 16, 58)
    cosmos_sdk.generated_frames = [value] * 16
    env.chunk_step(action)
    cosmos_sdk.generated_frames = [value] * 14 + [1.0] * 2
    _, rewards, terms, truncs, info = env.chunk_step(action)
    assert terms[:, 13].all() and terms.sum() == 2 and not truncs.any()
    assert env.elapsed_steps.tolist() == [30, 30]
    assert info[0]["episode"]["failure"].all()
    assert not info[0]["episode"]["success_final"].any()
    assert not rewards[:, 14:].any()


def test_cls5_switching_negative_type_keeps_failure_streak(
    env_config, cosmos_sdk, cls5_checkpoint
):
    env = cls5_env(env_config, cosmos_sdk, cls5_checkpoint)
    action = torch.zeros(2, 16, 58)
    cosmos_sdk.generated_frames = [-1.0] * 16
    env.chunk_step(action)
    cosmos_sdk.generated_frames = [-0.5] * 14 + [1.0] * 2
    _, _, terms, _, info = env.chunk_step(action)
    assert terms[:, 13].all()
    assert info[0]["episode"]["failure"].all()


def test_cls5_zero_breaks_negative_streak(env_config, cosmos_sdk, cls5_checkpoint):
    env = cls5_env(env_config, cosmos_sdk, cls5_checkpoint)
    action = torch.zeros(2, 16, 58)
    cosmos_sdk.generated_frames = [-1.0] * 16
    env.chunk_step(action)
    cosmos_sdk.generated_frames = [0.0] + [-0.5] * 15
    _, _, terms, _, _ = env.chunk_step(action)
    assert not terms.any()
    cosmos_sdk.generated_frames = [-1.0] * 15 + [1.0]
    _, _, terms, _, info = env.chunk_step(action)
    assert terms[:, 14].all() and env.elapsed_steps.tolist() == [47, 47]
    assert info[0]["episode"]["failure"].all()


@pytest.mark.parametrize(
    "last_value,is_success", [(0.0, False), (0.5, False), (1.0, True)]
)
def test_cls5_timeout_requires_actual_success_score(
    env_config, cosmos_sdk, cls5_checkpoint, last_value, is_success
):
    env = cls5_env(env_config, cosmos_sdk, cls5_checkpoint, horizon=32)
    action = torch.zeros(2, 16, 58)
    cosmos_sdk.generated_frames = [0.0] * 16
    env.chunk_step(action)
    cosmos_sdk.generated_frames = [1.0] * 15 + [last_value]
    _, _, terms, truncs, info = env.chunk_step(action)
    assert not terms.any() and truncs[:, 15].all()
    ep = info[0]["episode"]
    assert ep["success_final"].tolist() == [is_success] * 2
    assert ep["success_at_timeout_final_success"].tolist() == [is_success] * 2
    assert "success_at_timeout_final3" not in ep


def test_cls5_success_two_streak_and_signed_record(
    env_config, cosmos_sdk, cls5_checkpoint, tmp_path
):
    env_config.is_eval = True
    env_config.evaluation_result_dir = str(tmp_path / "cls5-records")
    env = cls5_env(env_config, cosmos_sdk, cls5_checkpoint)
    action = torch.zeros(2, 16, 58)
    cosmos_sdk.generated_frames = [1.0] * 16
    env.chunk_step(action)
    _, _, terms, truncs, info = env.chunk_step(action)
    assert terms[:, 13].all() and not truncs.any()
    assert info[0]["episode"]["success_by_streak"].all()
    records = [
        json.loads(line)
        for p in (tmp_path / "cls5-records").glob("*.jsonl")
        for line in p.read_text().splitlines()
    ]
    assert len(records) == 2
    assert all(
        r["final_predicted_reward"] == 2 and r["success_reward"] == 2 for r in records
    )


def test_balanced_all_imperfect_qualities_use_original_quality_field(
    env_config, cosmos_sdk
):
    path = env_config.split_path
    split = json.loads(open(path).read())
    for row, quality in zip(split["episodes"], ["success", "imperfect1", "imperfect3"]):
        row["quality"] = quality
        row["review_status"] = "confirmed"
    open(path, "w").write(json.dumps(split))
    env_config.balanced_quality_sampling = True
    env_config.balanced_failure_qualities = ["imperfect1", "imperfect2", "imperfect3"]
    env = Cosmos3Env(env_config, 4, 0, 1)
    seen = set()
    for _ in range(30):
        indices = env._sample_reset_episode_indices()
        assert indices[0] == indices[1] == 0
        assert indices[2] == indices[3] and indices[2] in [1, 2]
        seen.add(indices[2])
    assert seen == {1, 2}


@pytest.fixture
def four_quality_config(env_config):
    root = Path(env_config.initial_image_path)
    qualities = ["success"] * 5 + ["imperfect1", "imperfect2", "imperfect3"]
    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text())
    info["total_episodes"] = len(qualities)
    info_path.write_text(json.dumps(info))
    rows = [
        {"episode_index": i, "split": "train", "quality": quality}
        for i, quality in enumerate(qualities)
    ]
    Path(env_config.split_path).write_text(json.dumps({"episodes": rows}))
    (root / "meta/episodes.jsonl").write_text(
        "".join(
            json.dumps({"episode_index": i, "length": 3, "tasks": ["unfold pad"]})
            + "\n"
            for i in range(len(qualities))
        )
    )
    for i in range(3, len(qualities)):
        (root / f"data/episode_{i:06d}.parquet").write_bytes(
            (root / "data/episode_000000.parquet").read_bytes()
        )
    env_config.balanced_quality_sampling = True
    env_config.balanced_qualities = [
        "success",
        "imperfect1",
        "imperfect2",
        "imperfect3",
    ]
    env_config.group_size = 8
    return env_config, qualities


def test_four_quality_sampling_balances_unequal_pools(four_quality_config, cosmos_sdk):
    cfg, qualities = four_quality_config
    env = Cosmos3Env(cfg, 64, 0, 1)
    seen_success = set()
    for _ in range(40):
        indices = env._sample_reset_episode_indices()
        selected = []
        for start in range(0, 64, 8):
            group = indices[start : start + 8]
            assert len(set(group)) == 1
            selected.append(qualities[group[0]])
            if qualities[group[0]] == "success":
                seen_success.add(group[0])
        assert Counter(selected) == dict.fromkeys(cfg.balanced_qualities, 2)
    assert seen_success == set(range(5))


@pytest.mark.parametrize("seed_offset", range(8))
def test_four_quality_sampling_rotates_on_each_worker(
    four_quality_config, cosmos_sdk, seed_offset
):
    cfg, qualities = four_quality_config
    env = Cosmos3Env(cfg, 8, seed_offset, 8)
    observed = []
    for _ in range(4):
        indices = env._sample_reset_episode_indices()
        assert len(indices) == 8 and len(set(indices)) == 1
        observed.append(qualities[indices[0]])
    expected = list(cfg.balanced_qualities)
    offset = seed_offset % 4
    assert observed == expected[offset:] + expected[:offset]


def test_four_quality_sampling_rejects_missing_pool(env_config, cosmos_sdk):
    path = Path(env_config.split_path)
    split = json.loads(path.read_text())
    for row, quality in zip(split["episodes"], ["success", "imperfect1", "imperfect3"]):
        row["quality"] = quality
    path.write_text(json.dumps(split))
    env_config.balanced_quality_sampling = True
    env_config.balanced_qualities = [
        "success",
        "imperfect1",
        "imperfect2",
        "imperfect3",
    ]
    with pytest.raises(ValueError, match="nonempty quality pools"):
        Cosmos3Env(env_config, 4, 0, 1)


def test_four_quality_sampling_rejects_duplicate_pool(four_quality_config, cosmos_sdk):
    cfg, _ = four_quality_config
    cfg.balanced_qualities = ["success", "success", "imperfect2", "imperfect3"]
    with pytest.raises(ValueError, match="distinct quality names"):
        Cosmos3Env(cfg, 8, 0, 1)


@pytest.mark.parametrize("balanced_four", [False, True])
@pytest.mark.parametrize("learning_rate,noise", [(3e-6, 0.5), (1e-5, 0.3)])
def test_recipe_matches_current_training_contract(
    monkeypatch, balanced_four, learning_rate, noise
):
    from hydra import compose, initialize_config_dir

    config_dir = Path(__file__).resolve().parents[2] / "examples/embodiment/config"
    for name in [
        "COSMOS3_ROOT",
        "COSMOS3_WM_CHECKPOINT",
        "COSMOS3_WM_CONFIG",
        "COSMOS3_ACTION_STATS",
        "GR00T_MODEL_PATH",
        "GR00T_BACKBONE_PATH",
        "H2_DATASET_PATH",
        "CLASSIFIER_REWARD_CHECKPOINT",
        "CLASSIFIER_REWARD_CONFIG",
    ]:
        monkeypatch.setenv(name, "/tmp/recipe-assets/" + name.lower())
    overrides = [
        f"actor.optim.lr={learning_rate}",
        f"actor.model.rl_head_config.noise_level={noise}",
    ]
    if balanced_four:
        overrides.append(
            "env.train.balanced_qualities=[success,imperfect1,imperfect2,imperfect3]"
        )
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
        cfg = compose(
            config_name="cosmos3_medical_pads_grpo_gr00t_n1d7", overrides=overrides
        )
        OmegaConf.resolve(cfg)
    assert cfg.runner.max_steps == 150 and cfg.algorithm.group_size == 8
    assert cfg.actor.optim.lr == learning_rate
    assert cfg.actor.model.rl_head_config.noise_level == noise
    assert cfg.actor.model.rl_head_config.action_noise_scale == 0
    assert cfg.env.train.balanced_quality_sampling
    assert list(cfg.env.train.balanced_failure_qualities) == [
        "imperfect1",
        "imperfect2",
        "imperfect3",
    ]
    if balanced_four:
        assert list(cfg.env.train.balanced_qualities) == [
            "success",
            "imperfect1",
            "imperfect2",
            "imperfect3",
        ]
    else:
        assert cfg.env.train.balanced_qualities is None
    for partition in [cfg.env.train, cfg.env.eval]:
        assert partition.seed == 42
        assert not partition.auto_reset and not partition.ignore_terminations
        assert partition.terminal_streak_frames == 30
        assert partition.timeout_success_if_final_success
        assert partition.num_inference_steps == 30
        assert partition.max_episode_steps == 480
        assert partition.action_statistics == "/tmp/recipe-assets/cosmos3_action_stats"
    assert cfg.env.train.use_rel_reward
    assert not cfg.env.eval.use_rel_reward
    assert cfg.env.train.split == "train" and cfg.env.eval.split == "val"
    assert cfg.env.eval.eval_unique_episodes and cfg.env.eval.group_size == 1
    assert cfg.env.eval.total_num_envs == 64


def test_group_return_diversity_excludes_terminal_padding():
    from rlinf.utils.metric_utils import compute_group_success_metrics

    rewards = torch.tensor([[0.0, 2.0, 100.0, 100.0], [1000.0, 1000.0, 7.0, 7.0]])[
        :, :, None
    ]
    mask = torch.tensor([[True, True, False, False], [False, False, True, True]])[
        :, :, None
    ]
    metrics = compute_group_success_metrics(rewards, group_size=2, loss_mask=mask)
    assert metrics["group_reward_std_mean"] == pytest.approx(0.5)
    assert metrics["group_reward_equal_fraction"] == pytest.approx(0.5)
    assert metrics["group_mixed_fraction"] == pytest.approx(0.5)

# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""CPU inference/cache regressions and strict LAM checkpoint loading."""

import argparse
import ast
import importlib.util
import json
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

# Inference optimizations and upstream compatibility

ROOT = Path(
    os.environ.get("DREAMDOJO_ROOT", Path(__file__).resolve().parents[4] / "DreamDojo")
)
INFERENCE = ROOT / "cosmos_predict2/_src/predict2/inference"
MODEL = (
    ROOT
    / "cosmos_predict2/_src/predict2/action/models/action_conditioned_video2world_rectified_flow_model.py"
)


def cache_class():
    if not INFERENCE.is_dir():
        pytest.skip("Set DREAMDOJO_ROOT to the patched DreamDojo checkout")
    spec = importlib.util.spec_from_file_location(
        "native_cache", INFERENCE / "text_embedding_cache.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TextEmbeddingCache


def native_method(path, class_name, name):
    if not path.is_file():
        pytest.skip(f"External DreamDojo source not installed: {path}")
    tree = ast.parse(path.read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name
    )
    return next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name
    )


def test_interactive_data_keeps_complete_upstream_registry():
    path = ROOT / "cosmos_predict2/_src/predict2/interactive/configs/data.py"
    if not path.is_file():
        pytest.skip(f"External DreamDojo source not installed: {path}")
    tree = ast.parse(path.read_text())
    tree.body = [
        node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    registrations = []
    config_store = SimpleNamespace(store=lambda **kwargs: registrations.append(kwargs))
    namespace = {
        "ConfigStore": SimpleNamespace(instance=lambda: config_store),
        "L": lambda target: lambda **kwargs: kwargs,
        "ActionDatasetSFWarmup": object,
        "MultiVideoActionDataset": object,
        "DistributedSampler": object,
        "DataLoader": object,
        "get_data_path": lambda embodiment: ([embodiment], [1.0]),
    }
    exec(compile(tree, str(path), "exec"), namespace)
    namespace["register_interactive_data"]()
    embodiments = {"gr1", "g1", "agibot", "agibot_fruit", "yam", "pretrain"}
    expected = {
        f"gr00t_{name}_warmup"
        for name in embodiments | {"old_gr1_dreamdojo", "old_gr1_cosmos"}
    } | {
        f"gr00t_customized_{name}{suffix}"
        for name in embodiments
        for suffix in ("", "_long")
    }
    assert len(registrations) == 2 * len(expected)
    for split in ("train", "val"):
        entries = [r for r in registrations if r["group"] == f"data_{split}"]
        assert {r["name"] for r in entries} == expected
        assert all(r["package"] == f"dataloader_{split}" for r in entries)
        assert all(r["node"]["dataset"] for r in entries)


def test_teacher_generation_cli_keeps_standard_arguments(monkeypatch):
    path = (
        ROOT
        / "cosmos_predict2/_src/predict2/action/inference/inference_gr00t_warmup.py"
    )
    if not path.is_file():
        pytest.skip(f"External DreamDojo source not installed: {path}")
    tree = ast.parse(path.read_text())
    parser = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "parse_arguments"
    )
    namespace = {"argparse": argparse}
    exec(
        compile(ast.Module(body=[parser], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "teacher_generation",
            "--experiment",
            "g1",
            "--start",
            "2",
            "--end",
            "4",
        ],
    )
    args = namespace["parse_arguments"]()
    assert (args.start, args.end) == (2, 4)
    assert set(vars(args)) == {
        "experiment",
        "chunk_size",
        "guidance",
        "seed",
        "ckpt_path",
        "s3_cred",
        "input_video_root",
        "save_root",
        "start",
        "end",
        "num_latent_conditional_frames",
        "query_steps",
        "context_parallel_size",
    }


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_final_wm_loss_keeps_upstream_channel_consistency(dtype):
    path = (
        ROOT / "cosmos_predict2/_src/predict2/models/text2world_model_rectified_flow.py"
    )
    method = native_method(path, "Text2WorldModelRectifiedFlow", "forward")

    def assigns(node, name):
        return isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        )

    start = next(
        i for i, node in enumerate(method.body) if assigns(node, "per_instance_loss")
    )
    end = next(i for i, node in enumerate(method.body) if assigns(node, "loss"))
    generator = torch.Generator().manual_seed(31)
    pred = torch.randn(
        2, 4, 3, 2, 2, generator=generator, dtype=dtype, requires_grad=True
    )
    target = torch.randn(2, 4, 3, 2, 2, generator=generator, dtype=dtype)
    weights = torch.tensor([0.5, 1.5], dtype=dtype)
    namespace = {
        "torch": torch,
        "vt_pred_B_C_T_H_W": pred,
        "vt_B_C_T_H_W": target,
        "time_weights_B": weights,
    }
    exec(
        compile(
            ast.Module(body=method.body[start : end + 1], type_ignores=[]),
            str(path),
            "exec",
        ),
        namespace,
    )
    dims = (1, 2, 3, 4)
    expected = (
        weights
        * (
            ((pred - target) ** 2).mean(dims)
            + 0.1
            * ((torch.diff(pred, dim=1) - torch.diff(target, dim=1)) ** 2).mean(dims)
        )
    ).mean()
    torch.testing.assert_close(namespace["loss"], expected, rtol=0, atol=0)
    torch.testing.assert_close(
        torch.autograd.grad(namespace["loss"], pred, retain_graph=True)[0],
        torch.autograd.grad(expected, pred)[0],
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("per_worker", [False, True])
def test_training_report_reads_both_tensorboard_layouts(tmp_path, per_worker):
    from torch.utils.tensorboard import SummaryWriter

    from toolkits.world_model.dreamdojo_validation import training_report

    run = tmp_path / "run"
    events = run / "tensorboard"
    if per_worker:
        events = events / "all"
    with SummaryWriter(str(events)) as writer:
        writer.add_scalar("time/step", 123.0, 0)
    output = tmp_path / "report"
    output.mkdir()
    training_report(SimpleNamespace(inputs=[run], output=output))
    result = json.loads((output / "results.json").read_text())["run"]
    assert result["scalars"]["time/step"][0]["value"] == 123.0
    assert result["non_finite_tags"] == []


def test_cache_keeps_exact_values_and_is_not_mutated_by_callers():
    cache = cache_class()(2)
    value = torch.randn(2, 3, dtype=torch.float32)
    expected = value.clone()
    first = cache.get_or_compute(("a", "b"), lambda: value)
    first.zero_()
    second = cache.get_or_compute(("a", "b"), lambda: pytest.fail("cache miss"))
    torch.testing.assert_close(second, expected, rtol=0, atol=0)
    second.fill_(10)
    third = cache.get_or_compute(("a", "b"), lambda: pytest.fail("cache miss"))
    torch.testing.assert_close(third, expected, rtol=0, atol=0)
    assert cache.hits == 2 and cache.misses == 1


def test_cache_keys_include_batch_order_and_lru_eviction():
    cache = cache_class()(2)
    for key in [("a", "b"), ("b", "a"), ("a", "b"), ("a",)]:
        cache.get_or_compute(key, lambda: torch.ones(1))
    assert list(cache.entries) == [("a", "b"), ("a",)]
    assert cache.hits == 1 and cache.misses == 3
    cache.clear()
    assert not cache.entries and cache.hits == cache.misses == 0
    with pytest.raises(ValueError):
        cache_class()(0)


def test_pipeline_cache_respects_encoder_config_dtype_and_training_mode():
    fn = native_method(
        INFERENCE / "video2world.py", "Video2WorldInference", "_get_text_embeddings"
    )
    ns = {"torch": torch}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "native_pipeline", "exec"), ns)
    calls = []

    def compute(data_batch, input_caption_key):
        calls.append(tuple(data_batch[input_caption_key]))
        return torch.arange(len(calls) * 2, len(calls) * 2 + 2).float()

    encoder = SimpleNamespace(
        model=torch.nn.Linear(1, 1).eval(),
        config="a",
        compute_text_embeddings_online=compute,
    )
    obj = SimpleNamespace(
        model=SimpleNamespace(text_encoder=encoder),
        ckpt_path="fixed",
        cache_text_embeddings=True,
        _text_embedding_cache=cache_class()(),
    )

    def invoke():
        return ns["_get_text_embeddings"](obj, ["", ""])

    first = invoke()
    torch.testing.assert_close(invoke(), first, rtol=0, atol=0)
    assert len(calls) == 1
    encoder.config = "b"
    invoke()
    encoder.model.double()
    invoke()
    assert len(calls) == 3
    encoder.model.train()
    invoke()
    invoke()
    assert len(calls) == 5
    encoder.model.eval()
    obj.cache_text_embeddings = False
    invoke()
    assert len(calls) == 6


@pytest.mark.parametrize(
    "guidance,enabled,expected_calls",
    [(0, True, 1), (0, False, 2), (3, True, 2), (3, False, 2)],
)
def test_native_zero_guidance_fast_path(guidance, enabled, expected_calls):
    method = native_method(
        MODEL, "ActionVideo2WorldModelRectifiedFlow", "get_velocity_fn_from_batch"
    )
    fn = next(
        n
        for n in method.body
        if isinstance(n, ast.FunctionDef) and n.name == "velocity_fn"
    )
    calls = []

    def denoise(noise, noise_x, timestep, condition):
        calls.append(condition)
        return torch.tensor([3.0, -4.0]) if condition else torch.tensor([17.0, 2.0])

    ns = {
        "torch": torch,
        "self": SimpleNamespace(denoise=denoise, inference_skip_zero_guidance=enabled),
        "guidance": guidance,
        "condition": True,
        "uncondition": False,
    }
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "native_velocity", "exec"), ns)
    result = ns["velocity_fn"](None, None, None)
    torch.testing.assert_close(
        result,
        torch.tensor([3.0, -4.0]) + guidance * torch.tensor([-14.0, -6.0]),
        rtol=0,
        atol=0,
    )
    assert len(calls) == expected_calls


@pytest.mark.parametrize("per_chunk", [True, False])
def test_resident_models_follow_outer_offload_boundaries(per_chunk):
    from rlinf.envs.world_model.world_model_dreamdojo_env import DreamDojoEnv

    moves = []

    def part(name):
        return SimpleNamespace(to=lambda device: moves.append((name, str(device))))

    env = DreamDojoEnv.__new__(DreamDojoEnv)
    env.pipe = SimpleNamespace(
        offload_diffusion_model=per_chunk,
        offload_tokenizer=per_chunk,
        model=SimpleNamespace(
            net=part("dit"),
            conditioner=part("conditioner"),
            tokenizer=SimpleNamespace(encoder=part("encoder"), decoder=part("decoder")),
        ),
    )
    env._move_resident_wm("cpu")
    env._move_resident_wm("cuda")
    assert len(moves) == (0 if per_chunk else 8)


def test_wan_vae_uses_shared_module_and_clears_temporal_cache():
    from rlinf.envs.world_model.world_model_dreamdojo_env import DreamDojoEnv

    moves = []
    env = DreamDojoEnv.__new__(DreamDojoEnv)
    env.pipe = SimpleNamespace(
        offload_diffusion_model=True,
        offload_tokenizer=False,
        model=SimpleNamespace(
            tokenizer=SimpleNamespace(
                clear_cache=lambda: moves.append("clear"),
                model=SimpleNamespace(
                    model=SimpleNamespace(to=lambda device: moves.append(device))
                ),
            )
        ),
    )
    env._move_resident_wm("cpu")
    env._move_resident_wm("cuda")
    assert moves == ["clear", "cpu", "clear", "cuda"]


# Strict LAM checkpoint loading


@pytest.fixture
def loader(monkeypatch):
    root = Path(
        os.environ.get(
            "DREAMDOJO_ROOT", Path(__file__).resolve().parents[4] / "DreamDojo"
        )
    )
    source = root / "external/lam/model.py"
    if not source.is_file():
        pytest.skip("DreamDojo external/lam source is not installed")
    lightning = types.ModuleType("lightning")
    lightning.LightningModule = torch.nn.Module
    modules = types.ModuleType("external.lam.modules")
    modules.LatentActionModel = lambda **kwargs: torch.nn.Linear(2, 2)
    monkeypatch.setitem(sys.modules, "lightning", lightning)
    monkeypatch.setitem(sys.modules, "external.lam.modules", modules)
    monkeypatch.delenv("DREAMDOJO_LAM_CHECKPOINT", raising=False)
    spec = importlib.util.spec_from_file_location("lam_loader_under_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_missing_checkpoint_fails_before_model_allocation(
    loader, monkeypatch, tmp_path
):
    def forbidden_allocation(**kwargs):
        pytest.fail("Allocated LAM before checking the checkpoint")

    monkeypatch.setattr(loader, "LatentActionModel", forbidden_allocation)
    with pytest.raises(FileNotFoundError, match="refusing to continue"):
        loader.LAM(ckpt_path=str(tmp_path / "missing.ckpt"))


def test_stock_path_uses_site_override_but_explicit_path_wins(
    loader, monkeypatch, tmp_path
):
    site = tmp_path / "site.ckpt"
    explicit = tmp_path / "explicit.ckpt"
    site.touch()
    explicit.touch()
    monkeypatch.setenv("DREAMDOJO_LAM_CHECKPOINT", str(site))
    assert loader.resolve_lam_checkpoint(loader.DEFAULT_LAM_CHECKPOINT) == str(site)
    assert loader.resolve_lam_checkpoint(str(explicit)) == str(explicit)


def test_checkpoint_weights_really_restore(loader, monkeypatch, tmp_path, capsys):
    original = loader.LAM()
    with torch.no_grad():
        original.lam.weight.fill_(0.25)
        original.lam.bias.fill_(-0.5)
    checkpoint = tmp_path / "lam.ckpt"
    torch.save({"state_dict": original.state_dict()}, checkpoint)
    monkeypatch.setenv("DREAMDOJO_LAM_CHECKPOINT", str(checkpoint))
    restored = loader.LAM(ckpt_path=loader.DEFAULT_LAM_CHECKPOINT)
    assert restored.loaded_checkpoint == str(checkpoint)
    assert restored.ckpt_path == str(checkpoint)
    assert torch.equal(restored.lam.weight, original.lam.weight)
    assert torch.equal(restored.lam.bias, original.lam.bias)
    assert "0 missing and 0 unexpected keys" in capsys.readouterr().out


@pytest.mark.parametrize("defect", ["missing", "unexpected", "shape"])
def test_incompatible_state_dict_fails(loader, tmp_path, defect):
    model = loader.LAM()
    state = model.state_dict()
    if defect == "missing":
        del state["lam.weight"]
    elif defect == "unexpected":
        state["not_a_parameter"] = torch.zeros(1)
    else:
        state["lam.weight"] = torch.zeros(3, 3)
    checkpoint = tmp_path / "bad.ckpt"
    torch.save({"state_dict": state}, checkpoint)
    with pytest.raises(RuntimeError):
        model.reload_ckpt(str(checkpoint))
    assert model.loaded_checkpoint is None


def test_missing_state_dict_fails(loader, tmp_path):
    checkpoint = tmp_path / "bad.ckpt"
    torch.save({"wrong_format": {}}, checkpoint)
    with pytest.raises(KeyError, match="state_dict"):
        loader.LAM(ckpt_path=str(checkpoint))


def test_explicit_none_allows_lam_training_from_scratch(loader, monkeypatch):
    monkeypatch.setenv("DREAMDOJO_LAM_CHECKPOINT", "/missing.ckpt")
    assert loader.LAM(ckpt_path=None).loaded_checkpoint is None

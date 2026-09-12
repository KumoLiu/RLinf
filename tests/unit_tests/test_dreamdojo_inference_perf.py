# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests execute native cache/method source without loading GPU models."""

import ast
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(
    os.environ.get("DREAMDOJO_ROOT", Path(__file__).resolve().parents[3] / "DreamDojo")
)
INFERENCE = ROOT / "cosmos_predict2/_src/predict2/inference"
MODEL = (
    ROOT
    / "cosmos_predict2/_src/predict2/action/models/action_conditioned_video2world_rectified_flow_model.py"
)
PERF = Path(__file__).resolve().parents[2] / "toolkits/world_model/dreamdojo_perf.py"


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


def test_policy_initialization_keeps_model_when_eval_returns_none():
    """Execute the real initialization statements with GR00T's eval contract."""
    tree = ast.parse(PERF.read_text())
    policy = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "policy"
    )
    start = next(
        i
        for i, n in enumerate(policy.body)
        if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "model"
    )
    end = next(
        i
        for i, n in enumerate(policy.body)
        if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "episodes"
    )
    moves = []
    model = SimpleNamespace(
        to=lambda device: moves.append(device),
        eval=lambda: moves.append("eval"),
    )
    namespace = {
        "get_model": lambda cfg: model,
        "cfg": SimpleNamespace(actor=SimpleNamespace(model="sft")),
    }
    exec(
        compile(
            ast.Module(body=policy.body[start:end], type_ignores=[]),
            "policy_init",
            "exec",
        ),
        namespace,
    )
    assert namespace["model"] is model
    assert moves == ["cuda", "eval"]


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

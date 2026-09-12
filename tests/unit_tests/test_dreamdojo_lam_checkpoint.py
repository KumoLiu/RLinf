# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the actual external loader with a tiny model, not 8 GB of weights."""

import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest
import torch


@pytest.fixture
def loader(monkeypatch):
    root = Path(
        os.environ.get(
            "DREAMDOJO_ROOT", Path(__file__).resolve().parents[3] / "DreamDojo"
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

from __future__ import annotations

import torch

from simple_cv_training.config import TorchCompileConfig
from simple_cv_training.optimization.compile import apply_torch_compile


def test_apply_torch_compile_returns_same_model_when_disabled() -> None:
    model = torch.nn.Linear(2, 2)

    result = apply_torch_compile(model, TorchCompileConfig(enabled=False))

    assert result is model


def test_apply_torch_compile_passes_enabled_options(monkeypatch) -> None:
    model = torch.nn.Linear(2, 2)
    compiled_model = torch.nn.Linear(2, 2)
    calls = {}

    def fake_compile(target_model, **kwargs):
        calls["model"] = target_model
        calls["kwargs"] = kwargs
        return compiled_model

    monkeypatch.setattr(torch, "compile", fake_compile)

    result = apply_torch_compile(
        model,
        TorchCompileConfig(
            enabled=True,
            mode="reduce-overhead",
            fullgraph=True,
            backend="eager",
            dynamic=True,
        ),
    )

    assert result is compiled_model
    assert calls["model"] is model
    assert calls["kwargs"] == {
        "mode": "reduce-overhead",
        "fullgraph": True,
        "backend": "eager",
        "dynamic": True,
    }


def test_apply_torch_compile_omits_optional_none_values(monkeypatch) -> None:
    model = torch.nn.Linear(2, 2)
    calls = {}

    def fake_compile(target_model, **kwargs):
        calls["model"] = target_model
        calls["kwargs"] = kwargs
        return target_model

    monkeypatch.setattr(torch, "compile", fake_compile)

    apply_torch_compile(model, TorchCompileConfig(enabled=True))

    assert calls["kwargs"] == {
        "mode": "default",
        "fullgraph": False,
    }

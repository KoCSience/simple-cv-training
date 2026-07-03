from __future__ import annotations

import torch

from simple_cv_training.utils import normalize_wrapped_model_state_dict


def test_normalize_wrapped_model_state_dict_removes_compile_prefix() -> None:
    state_dict = {
        "_orig_mod.layer.weight": torch.tensor([1.0]),
        "_orig_mod.layer.bias": torch.tensor([2.0]),
    }

    normalized = normalize_wrapped_model_state_dict(state_dict)

    assert list(normalized.keys()) == ["layer.weight", "layer.bias"]


def test_normalize_wrapped_model_state_dict_removes_data_parallel_and_compile_prefixes() -> None:
    state_dict = {
        "module._orig_mod.layer.weight": torch.tensor([1.0]),
        "module._orig_mod.layer.bias": torch.tensor([2.0]),
    }

    normalized = normalize_wrapped_model_state_dict(state_dict)

    assert list(normalized.keys()) == ["layer.weight", "layer.bias"]

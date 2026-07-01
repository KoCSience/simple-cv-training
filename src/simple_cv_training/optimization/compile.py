from __future__ import annotations

import torch
from torch import nn

from simple_cv_training.config.schema import TorchCompileConfig


def apply_torch_compile[ModelT: nn.Module](model: ModelT, config: TorchCompileConfig) -> ModelT:
    """Apply torch.compile only when an experiment explicitly opts in."""
    if not config.enabled:
        return model

    compile_kwargs = {
        "mode": config.mode,
        "fullgraph": config.fullgraph,
    }
    if config.backend is not None:
        compile_kwargs["backend"] = config.backend
    if config.dynamic is not None:
        compile_kwargs["dynamic"] = config.dynamic

    try:
        return torch.compile(model, **compile_kwargs)  # type: ignore[return-value]
    except Exception as exc:
        raise RuntimeError("torch.compile failed while applying optimization.compile settings.") from exc

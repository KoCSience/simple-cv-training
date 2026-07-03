from __future__ import annotations

import warnings
from collections.abc import Callable

from simple_cv_training.config.schema import AmpConfig

from .hardware import is_bf16_supported


def resolve_trainer_precision(
    config: AmpConfig,
    supports_bf16: Callable[[], bool] = is_bf16_supported,
) -> str | None:
    """Return a Lightning precision value when AMP is explicitly enabled."""
    if not config.enabled:
        return None

    if config.precision != "bf16-mixed":
        return config.precision

    if supports_bf16():
        return config.precision

    message = (
        "optimization.amp.precision=bf16-mixed requires CUDA bf16 support. "
        "Turing-generation GPUs do not support bf16 mixed precision; use an Ampere-generation or newer "
        "supported GPU, set optimization.amp.precision=16-mixed, or set "
        "optimization.amp.require_bf16_supported=false to let Lightning handle the runtime behavior."
    )
    if config.require_bf16_supported:
        raise RuntimeError(message)

    warnings.warn(message, RuntimeWarning, stacklevel=2)
    return config.precision

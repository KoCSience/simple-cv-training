from __future__ import annotations

import pytest

from simple_cv_training.config import AmpConfig
from simple_cv_training.optimization.precision import resolve_trainer_precision


def test_resolve_trainer_precision_returns_none_when_amp_disabled() -> None:
    precision = resolve_trainer_precision(AmpConfig(enabled=False))

    assert precision is None


def test_resolve_trainer_precision_accepts_supported_bf16() -> None:
    precision = resolve_trainer_precision(AmpConfig(enabled=True, precision="bf16-mixed"), supports_bf16=lambda: True)

    assert precision == "bf16-mixed"


def test_resolve_trainer_precision_rejects_unsupported_required_bf16() -> None:
    with pytest.raises(RuntimeError, match="bf16"):
        resolve_trainer_precision(AmpConfig(enabled=True, precision="bf16-mixed"), supports_bf16=lambda: False)


def test_resolve_trainer_precision_warns_when_bf16_check_is_not_required() -> None:
    with pytest.warns(RuntimeWarning, match="bf16"):
        precision = resolve_trainer_precision(
            AmpConfig(enabled=True, precision="bf16-mixed", require_bf16_supported=False),
            supports_bf16=lambda: False,
        )

    assert precision == "bf16-mixed"


def test_resolve_trainer_precision_allows_fp16_without_bf16_check() -> None:
    precision = resolve_trainer_precision(AmpConfig(enabled=True, precision="16-mixed"), supports_bf16=lambda: False)

    assert precision == "16-mixed"

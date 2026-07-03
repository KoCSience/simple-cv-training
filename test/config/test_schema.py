from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from pydantic import ValidationError

from simple_cv_training.config import validate_experiment_config

CONFIG_DIR = str(Path("configs").resolve())


def compose_train_config(overrides: list[str] | None = None):
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        return compose(config_name="train", overrides=overrides or [])


def test_train_config_validates_with_defaults() -> None:
    cfg = compose_train_config()

    typed = validate_experiment_config(cfg)

    assert typed.dataset.dataset_name == "CIFAR10"
    assert typed.model.model_name == "resnet18"
    assert typed.mode.name == "beginner"
    assert typed.optimization.amp.enabled is False
    assert typed.optimization.amp.precision == "bf16-mixed"
    assert typed.optimization.compile.enabled is False


def test_train_config_validates_group_overrides() -> None:
    cfg = compose_train_config(["data=zero_images", "model=zero_output_dummy", "trainer=smoke"])

    typed = validate_experiment_config(cfg)

    assert typed.dataset.dataset_name == "ZeroImages"
    assert typed.model.model_name == "zero_output_dummy"
    assert typed.training.num_epochs == 1


def test_optimization_amp_overrides_validate() -> None:
    cfg = compose_train_config(["optimization.amp.enabled=true", "optimization.amp.precision=bf16-mixed"])

    typed = validate_experiment_config(cfg)

    assert typed.optimization.amp.enabled is True
    assert typed.optimization.amp.precision == "bf16-mixed"


def test_invalid_amp_precision_fails_validation() -> None:
    cfg = compose_train_config(["optimization.amp.precision=float8-mixed"])

    with pytest.raises(ValidationError):
        validate_experiment_config(cfg)


def test_invalid_compile_mode_fails_validation() -> None:
    cfg = compose_train_config(["optimization.compile.mode=experimental"])

    with pytest.raises(ValidationError):
        validate_experiment_config(cfg)


def test_invalid_training_value_fails_validation() -> None:
    cfg = compose_train_config(["training.batch_size=0"])

    with pytest.raises(ValidationError):
        validate_experiment_config(cfg)

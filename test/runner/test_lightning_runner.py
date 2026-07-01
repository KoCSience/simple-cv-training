from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from hydra import compose, initialize_config_dir

from simple_cv_training.runner import lightning_runner

CONFIG_DIR = str(Path("configs").resolve())


def compose_train_config(overrides: list[str] | None = None):
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        return compose(config_name="train", overrides=overrides or [])


def test_lightning_runner_rejects_manual_checkpoint_suffix() -> None:
    with pytest.raises(ValueError, match="manual .pt"):
        lightning_runner._require_lightning_checkpoint_format("/tmp/model.pt")


def test_build_trainer_omits_precision_when_amp_disabled(monkeypatch) -> None:
    cfg = compose_train_config(["trainer=smoke"])
    calls = {}

    class DummyTrainer:
        def __init__(self, **kwargs):
            calls["kwargs"] = kwargs

    monkeypatch.setattr(lightning_runner.pl, "Trainer", DummyTrainer)
    monkeypatch.setattr(lightning_runner, "configure_callbacks", lambda: [])
    monkeypatch.setattr(lightning_runner, "TorchSyncBatchNorm", lambda: SimpleNamespace())

    lightning_runner.build_trainer(cfg, loggers=[], precision=None)

    assert "precision" not in calls["kwargs"]


def test_build_trainer_passes_enabled_precision(monkeypatch) -> None:
    cfg = compose_train_config(["trainer=smoke"])
    calls = {}

    class DummyTrainer:
        def __init__(self, **kwargs):
            calls["kwargs"] = kwargs

    monkeypatch.setattr(lightning_runner.pl, "Trainer", DummyTrainer)
    monkeypatch.setattr(lightning_runner, "configure_callbacks", lambda: [])
    monkeypatch.setattr(lightning_runner, "TorchSyncBatchNorm", lambda: SimpleNamespace())

    lightning_runner.build_trainer(cfg, loggers=[], precision="bf16-mixed")

    assert calls["kwargs"]["precision"] == "bf16-mixed"

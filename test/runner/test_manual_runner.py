from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir

from simple_cv_training.runner import manual_runner

CONFIG_DIR = str(Path("configs").resolve())


def compose_train_config(overrides: list[str] | None = None):
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        return compose(config_name="train", overrides=overrides or [])


def test_prepare_manual_training_uses_hydra_config(monkeypatch) -> None:
    cfg = compose_train_config(
        [
            "data=zero_images",
            "model=zero_output_dummy",
            "trainer=smoke",
            "optimizer=adam",
            "disable_comet=true",
        ]
    )
    calls = {}
    real_device = torch.device

    monkeypatch.setattr(manual_runner, "_require_cuda_for_manual_training", lambda: None)
    monkeypatch.setattr(manual_runner.torch, "device", lambda _name: real_device("cpu"))
    monkeypatch.setattr(
        manual_runner,
        "configure_logger",
        lambda logged_params, model_name, disable_logging: calls.setdefault(
            "logger",
            SimpleNamespace(
                logged_params=logged_params,
                model_name=model_name,
                disable_logging=disable_logging,
            ),
        ),
    )
    monkeypatch.setattr(
        manual_runner,
        "configure_dataloader",
        lambda command_line_cfg, dataset_name: calls.setdefault(
            "dataloader",
            SimpleNamespace(
                command_line_cfg=command_line_cfg,
                dataset_name=dataset_name,
                n_classes=3,
                train_loader=[],
                val_loader=[],
            ),
        ),
    )

    def configure_model(model_config):
        calls["model_config"] = model_config
        return torch.nn.Linear(2, model_config.n_classes)

    monkeypatch.setattr(manual_runner, "configure_model", configure_model)
    monkeypatch.setattr(
        manual_runner,
        "configure_optimizer",
        lambda **kwargs: calls.setdefault("optimizer_kwargs", kwargs) or SimpleNamespace(),
    )
    monkeypatch.setattr(
        manual_runner,
        "configure_scheduler",
        lambda **kwargs: calls.setdefault("scheduler_kwargs", kwargs) or SimpleNamespace(),
    )

    *_, train_config, current_train_step, current_val_step, start_epoch = manual_runner.prepare_manual_training(cfg)

    assert calls["logger"].model_name == "zero_output_dummy"
    assert calls["logger"].disable_logging is True
    assert calls["logger"].logged_params["dataset"]["dataset_name"] == "ZeroImages"
    assert calls["dataloader"].command_line_cfg is cfg
    assert calls["dataloader"].dataset_name == "ZeroImages"
    assert calls["model_config"].model_name == "zero_output_dummy"
    assert calls["model_config"].n_classes == 3
    assert calls["optimizer_kwargs"]["optimizer_name"] == "Adam"
    assert train_config.grad_accum_interval == 1
    assert train_config.log_interval_steps == 1
    assert current_train_step == 1
    assert current_val_step == 1
    assert start_epoch == 0


def test_prepare_manual_training_wraps_model_when_use_dp(monkeypatch) -> None:
    cfg = compose_train_config(["data=zero_images", "model=zero_output_dummy", "trainer=smoke", "GPU.use_dp=true"])
    real_device = torch.device
    calls = {}

    class DummyDataParallel(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            calls["wrapped_model"] = module
            self.module = module

    monkeypatch.setattr(manual_runner, "_require_cuda_for_manual_training", lambda: None)
    monkeypatch.setattr(manual_runner.torch, "device", lambda _name: real_device("cpu"))
    monkeypatch.setattr(manual_runner, "configure_logger", lambda **_kwargs: SimpleNamespace())
    monkeypatch.setattr(
        manual_runner,
        "configure_dataloader",
        lambda **_kwargs: SimpleNamespace(n_classes=2, train_loader=[], val_loader=[]),
    )
    monkeypatch.setattr(manual_runner, "configure_model", lambda _model_config: torch.nn.Linear(2, 2))
    monkeypatch.setattr(manual_runner.nn, "DataParallel", DummyDataParallel)
    monkeypatch.setattr(manual_runner, "configure_optimizer", lambda **_kwargs: SimpleNamespace())
    monkeypatch.setattr(manual_runner, "configure_scheduler", lambda **_kwargs: SimpleNamespace())

    manual_runner.prepare_manual_training(cfg)

    assert isinstance(calls["wrapped_model"], torch.nn.Linear)


def test_prepare_manual_training_compiles_before_data_parallel(monkeypatch) -> None:
    cfg = compose_train_config(
        [
            "data=zero_images",
            "model=zero_output_dummy",
            "trainer=smoke",
            "GPU.use_dp=true",
            "optimization.compile.enabled=true",
        ]
    )
    real_device = torch.device
    calls = {}
    base_model = torch.nn.Linear(2, 2)
    compiled_model = torch.nn.Sequential(base_model)

    class DummyDataParallel(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            calls["wrapped_model"] = module
            self.module = module

    monkeypatch.setattr(manual_runner, "_require_cuda_for_manual_training", lambda: None)
    monkeypatch.setattr(manual_runner.torch, "device", lambda _name: real_device("cpu"))
    monkeypatch.setattr(manual_runner, "configure_logger", lambda **_kwargs: SimpleNamespace())
    monkeypatch.setattr(
        manual_runner,
        "configure_dataloader",
        lambda **_kwargs: SimpleNamespace(n_classes=2, train_loader=[], val_loader=[]),
    )
    monkeypatch.setattr(manual_runner, "configure_model", lambda _model_config: base_model)
    monkeypatch.setattr(manual_runner, "configure_optimizer", lambda **_kwargs: SimpleNamespace())
    monkeypatch.setattr(manual_runner, "configure_scheduler", lambda **_kwargs: SimpleNamespace())
    monkeypatch.setattr(manual_runner.nn, "DataParallel", DummyDataParallel)

    def apply_torch_compile(model, compile_config):
        calls["compiled_model"] = model
        calls["compile_enabled"] = compile_config.enabled
        return compiled_model

    monkeypatch.setattr(manual_runner, "apply_torch_compile", apply_torch_compile)

    manual_runner.prepare_manual_training(cfg)

    assert calls["compiled_model"] is base_model
    assert calls["compile_enabled"] is True
    assert calls["wrapped_model"] is compiled_model


def test_prepare_manual_training_loads_checkpoint_before_compile(monkeypatch) -> None:
    cfg = compose_train_config(
        [
            "data=zero_images",
            "model=zero_output_dummy",
            "trainer=smoke",
            "optimization.compile.enabled=true",
            "checkpoint_file.checkpoint_to_resume=experiment:test/checkpoint",
        ]
    )
    real_device = torch.device
    calls = []
    base_model = torch.nn.Linear(2, 2)

    monkeypatch.setattr(manual_runner, "_require_cuda_for_manual_training", lambda: None)
    monkeypatch.setattr(manual_runner.torch, "device", lambda _name: real_device("cpu"))
    monkeypatch.setattr(manual_runner, "configure_logger", lambda **_kwargs: SimpleNamespace())
    monkeypatch.setattr(
        manual_runner,
        "configure_dataloader",
        lambda **_kwargs: SimpleNamespace(n_classes=2, train_loader=[], val_loader=[]),
    )
    monkeypatch.setattr(manual_runner, "configure_model", lambda _model_config: base_model)
    monkeypatch.setattr(manual_runner, "configure_optimizer", lambda **_kwargs: SimpleNamespace())
    monkeypatch.setattr(manual_runner, "configure_scheduler", lambda **_kwargs: SimpleNamespace())

    def load_from_checkpoint(checkpoint_to_resume, model, optimizer, scheduler, device):
        calls.append(("load", model))
        return 2, 3, 4, model, optimizer, scheduler

    def apply_torch_compile(model, _compile_config):
        calls.append(("compile", model))
        return model

    monkeypatch.setattr(manual_runner, "load_from_checkpoint", load_from_checkpoint)
    monkeypatch.setattr(manual_runner, "apply_torch_compile", apply_torch_compile)

    *_, current_train_step, current_val_step, start_epoch = manual_runner.prepare_manual_training(cfg)

    assert calls == [("load", base_model), ("compile", base_model)]
    assert start_epoch == 2
    assert current_train_step == 3
    assert current_val_step == 4


def test_prepare_manual_training_rejects_amp() -> None:
    cfg = compose_train_config(["optimization.amp.enabled=true"])

    with pytest.raises(ValueError, match="main_pl.py for AMP"):
        manual_runner.prepare_manual_training(cfg)


def test_validation_checker_runs_on_interval_and_last_epoch() -> None:
    checker = manual_runner.ValidationChecker(val_interval_epochs=3, num_epochs=10)

    assert checker.should_validate(1) is False
    assert checker.should_validate(3) is True
    assert checker.should_validate(10) is True


def test_manual_runner_rejects_lightning_checkpoint_suffix() -> None:
    with pytest.raises(ValueError, match="Lightning .ckpt"):
        manual_runner._require_manual_checkpoint_format("/tmp/model.ckpt")

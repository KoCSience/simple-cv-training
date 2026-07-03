from __future__ import annotations

from typing import Any, cast

import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from tqdm import tqdm

from simple_cv_training.config import validate_experiment_config
from simple_cv_training.dataset import configure_dataloader
from simple_cv_training.logger import configure_logger
from simple_cv_training.model import ModelConfig as ClassificationModelConfig
from simple_cv_training.model import configure_model
from simple_cv_training.optimization import apply_torch_compile
from simple_cv_training.setup import configure_optimizer, configure_scheduler
from simple_cv_training.train import TrainConfig, train
from simple_cv_training.utils import load_from_checkpoint, save_to_checkpoint, save_to_comet
from simple_cv_training.val import validation


class TqdmEpoch(tqdm):
    def __init__(self, start_epoch: int, num_epochs: int, *args: Any, **kwargs: Any) -> None:
        super().__init__(range(start_epoch + 1, num_epochs + 1), *args, **kwargs)


class ValidationChecker:
    def __init__(self, val_interval_epochs: int, num_epochs: int) -> None:
        self.val_interval_epochs = val_interval_epochs
        self.num_epochs = num_epochs

    def should_validate(self, current_epoch: int) -> bool:
        return current_epoch % self.val_interval_epochs == 0 or current_epoch == self.num_epochs


def run_manual_training(cfg: DictConfig) -> None:
    typed_cfg = validate_experiment_config(cfg)

    (
        logger,
        dataloaders,
        model,
        optimizer,
        scheduler,
        train_config,
        current_train_step,
        current_val_step,
        start_epoch,
    ) = prepare_manual_training(cfg)

    val_checker = ValidationChecker(
        val_interval_epochs=typed_cfg.training.val_interval_epochs,
        num_epochs=typed_cfg.training.num_epochs,
    )

    with TqdmEpoch(start_epoch, typed_cfg.training.num_epochs, unit="epoch") as progress_bar_epoch:
        for current_epoch in progress_bar_epoch:
            progress_bar_epoch.set_description(f"[epoch {current_epoch:03d}]")

            train_output = train(
                model,
                optimizer,
                scheduler,
                dataloaders.train_loader,
                current_train_step,
                current_epoch,
                logger,
                train_config,
            )
            current_train_step = train_output.train_step

            if val_checker.should_validate(current_epoch):
                val_output = validation(
                    model,
                    dataloaders.val_loader,
                    current_val_step,
                    current_epoch,
                    logger,
                )
                current_val_step = val_output.val_step

                checkpoint_dict, _ = save_to_checkpoint(
                    typed_cfg.checkpoint_file.save_checkpoint_dir,
                    current_epoch,
                    current_train_step,
                    current_val_step,
                    val_output.top1,
                    model,
                    optimizer,
                    scheduler,
                    logger,
                )
                save_to_comet(checkpoint_dict, typed_cfg.model.model_name, logger)


def prepare_manual_training(cfg: DictConfig):
    typed_cfg = validate_experiment_config(cfg)
    _require_manual_checkpoint_format(typed_cfg.checkpoint_file.checkpoint_to_resume)
    _require_manual_optimization_scope(typed_cfg.optimization.amp.enabled)

    logger = configure_logger(
        logged_params=_cfg_to_logged_params(cfg),
        model_name=typed_cfg.model.model_name,
        disable_logging=typed_cfg.disable_comet,
    )

    dataloaders = configure_dataloader(
        command_line_cfg=cfg,
        dataset_name=typed_cfg.dataset.dataset_name,
    )

    _require_cuda_for_manual_training()
    device = torch.device("cuda")

    model = configure_model(
        ClassificationModelConfig(
            model_name=typed_cfg.model.model_name,
            use_pretrained=typed_cfg.model.use_pretrained,
            torch_home=typed_cfg.model.torch_home,
            n_classes=dataloaders.n_classes,
        )
    )
    model = model.to(device)

    optimizer = configure_optimizer(
        optimizer_name=typed_cfg.optimizer.optimizer_name,
        lr=typed_cfg.optimizer.lr,
        weight_decay=typed_cfg.optimizer.weight_decay,
        momentum=typed_cfg.optimizer.momentum,
        model_params=model.parameters(),
    )
    scheduler = configure_scheduler(
        optimizer=optimizer,
        use_scheduler=typed_cfg.optimizer.use_scheduler,
    )

    train_config = TrainConfig(
        grad_accum_interval=typed_cfg.optimizer.grad_accum,
        log_interval_steps=typed_cfg.training.log_interval_steps,
    )

    if typed_cfg.checkpoint_file.checkpoint_to_resume:
        (
            start_epoch,
            current_train_step,
            current_val_step,
            model,
            optimizer,
            scheduler,
        ) = load_from_checkpoint(  # type: ignore[assignment]
            typed_cfg.checkpoint_file.checkpoint_to_resume,
            model,
            optimizer,
            scheduler,
            device,
        )
    else:
        current_train_step = 1
        current_val_step = 1
        start_epoch = 0

    # Keep resume compatible with the plain manual .pt model, then apply optimization as a thin adapter.
    model = apply_torch_compile(model, typed_cfg.optimization.compile)
    if typed_cfg.GPU.use_dp:
        # DataParallel is the outermost manual-runner wrapper; Lightning uses GPU.devices instead.
        model = nn.DataParallel(model)  # type: ignore[assignment]

    return (
        logger,
        dataloaders,
        model,
        optimizer,
        scheduler,
        train_config,
        current_train_step,
        current_val_step,
        start_epoch,
    )


def _cfg_to_logged_params(cfg: DictConfig) -> dict[str, object]:
    params = OmegaConf.to_container(cfg, resolve=True)
    return cast(dict[str, object], params)


def _require_cuda_for_manual_training() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for the current manual training entrypoint. "
            "Check nvidia-smi and torch.cuda.is_available(), or run CPU-only tests with uv run pytest."
        )


def _require_manual_checkpoint_format(checkpoint_to_resume: str | None) -> None:
    if checkpoint_to_resume is None or checkpoint_to_resume.startswith("experiment:"):
        return
    if checkpoint_to_resume.endswith(".ckpt"):
        raise ValueError("main.py uses manual .pt checkpoints. Use main_pl.py to resume a Lightning .ckpt checkpoint.")


def _require_manual_optimization_scope(amp_enabled: bool) -> None:
    if not amp_enabled:
        return
    raise ValueError("manual runner does not support optimization.amp; use main_pl.py for AMP.")

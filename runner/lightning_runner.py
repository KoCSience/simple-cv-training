from __future__ import annotations

import torch
import lightning.pytorch as pl
from lightning.pytorch.plugins import TorchSyncBatchNorm
from omegaconf import DictConfig

from callback import configure_callbacks
from config import validate_experiment_config
from dataset import TrainValDataModule
from logger import configure_logger_pl
from model import SimpleLightningModel


def run_lightning_training(cfg: DictConfig) -> None:
    typed_cfg = validate_experiment_config(cfg)
    _require_cuda_for_training()

    loggers, exp_name = configure_logger_pl(
        command_line_cfg=cfg,
        model_name=typed_cfg.model.model_name,
        disable_logging=typed_cfg.disable_comet,
        save_dir=typed_cfg.log_dirs.comet_log_dir,
    )
    data_module = TrainValDataModule(
        command_line_cfg=cfg,
        dataset_name=typed_cfg.dataset.dataset_name,
    )
    model_lightning = SimpleLightningModel(
        command_line_args=cfg,
        n_classes=data_module.n_classes,
        exp_name=exp_name,
    )

    trainer = build_trainer(cfg, loggers)
    trainer.fit(
        model=model_lightning,
        datamodule=data_module,
        ckpt_path=typed_cfg.checkpoint_file.checkpoint_to_resume,
    )


def build_trainer(cfg: DictConfig, loggers) -> pl.Trainer:
    return pl.Trainer(
        devices=cfg.GPU.devices,
        accelerator="gpu",
        strategy="auto",
        max_epochs=cfg.training.num_epochs,
        logger=loggers,
        log_every_n_steps=cfg.training.log_interval_steps,
        accumulate_grad_batches=cfg.optimizer.grad_accum,
        num_sanity_val_steps=0,
        callbacks=configure_callbacks(),
        plugins=[TorchSyncBatchNorm()],
    )


def _require_cuda_for_training() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for the current Lightning training entrypoint. "
            "Check nvidia-smi and torch.cuda.is_available(), or run CPU-only tests with uv run pytest."
        )

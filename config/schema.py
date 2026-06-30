from __future__ import annotations

from pathlib import Path
from typing import Literal

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, Field, field_validator


class DatasetConfig(BaseModel):
    root: str = "./downloaded_data"
    dataset_name: Literal["CIFAR10", "ImageFolder", "VideoFolder", "ZeroImages"] = "CIFAR10"
    train_dir: str = "train"
    val_dir: str = "val"


class ModelConfig(BaseModel):
    torch_home: str = "./pretrained_models"
    model_name: Literal["resnet18", "resnet50", "x3d", "abn_r50", "vit_b", "zero_output_dummy"] = "resnet18"
    use_pretrained: bool = True


class VideoConfig(BaseModel):
    frames_per_clip: int = Field(default=16, ge=1)
    clip_duration: float = Field(default=80 / 30, gt=0)
    clips_per_video: int = Field(default=1, ge=1)


class TrainingConfig(BaseModel):
    batch_size: int = Field(default=8, ge=1)
    num_workers: int = Field(default=2, ge=0)
    num_epochs: int = Field(default=25, ge=1)
    val_interval_epochs: int = Field(default=1, ge=1)
    log_interval_steps: int = Field(default=1, ge=1)


class OptimizerConfig(BaseModel):
    optimizer_name: Literal["SGD", "Adam"] = "SGD"
    grad_accum: int = Field(default=1, ge=1)
    lr: float = Field(default=1e-4, gt=0)
    momentum: float = Field(default=0.9, ge=0)
    weight_decay: float = Field(default=5e-4, ge=0)
    use_scheduler: bool = False


class GpuConfig(BaseModel):
    use_dp: bool = False
    devices: str | int = "1"

    @field_validator("devices")
    @classmethod
    def validate_devices(cls, value: str | int) -> str | int:
        if isinstance(value, int):
            if value == 0 or value < -1:
                raise ValueError("GPU.devices must be -1 or a positive integer.")
            return value

        if value == "-1":
            return value

        parts = [part.strip() for part in value.split(",")]
        if not parts or any(not part.isdigit() for part in parts):
            raise ValueError("GPU.devices must be '-1', a positive integer, or comma-separated GPU ids.")
        return value


class LogDirsConfig(BaseModel):
    comet_log_dir: str = "./comet_logs/"
    tf_log_dir: str = "./tf_logs/"


class CheckpointConfig(BaseModel):
    save_checkpoint_dir: str = "./log"
    checkpoint_to_resume: str | None = None

    @field_validator("checkpoint_to_resume")
    @classmethod
    def validate_checkpoint_to_resume(cls, value: str | None) -> str | None:
        if value is None:
            return value
        if value.startswith("experiment:"):
            return value
        if not Path(value).exists():
            raise ValueError(f"checkpoint_to_resume does not exist: {value}")
        return value


class ModeConfig(BaseModel):
    name: Literal["beginner", "researcher", "reproduce", "framework-dev"] = "beginner"
    confirm_logging: bool = True
    seed_everything: bool = False
    torch_use_deterministic: bool = False


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    seed: int = 42
    experiment_name: str = "default"
    dataset: DatasetConfig
    model: ModelConfig
    video: VideoConfig = Field(default_factory=VideoConfig)
    training: TrainingConfig
    optimizer: OptimizerConfig
    GPU: GpuConfig = Field(default_factory=GpuConfig)
    log_dirs: LogDirsConfig = Field(default_factory=LogDirsConfig)
    checkpoint_file: CheckpointConfig = Field(default_factory=CheckpointConfig)
    mode: ModeConfig = Field(default_factory=ModeConfig)
    disable_comet: bool = False


def validate_experiment_config(cfg: DictConfig) -> ExperimentConfig:
    return ExperimentConfig.model_validate(OmegaConf.to_container(cfg, resolve=True))

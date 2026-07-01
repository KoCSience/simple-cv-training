from dataclasses import dataclass
from typing import Literal

# import argparse
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from .cifar10 import Cifar10Info, cifar10
from .image_folder import ImageFolderInfo, image_folder
from .transforms import (
    TransformImageInfo,
    TransformVideoInfo,
    transform_image,
    transform_video,
)
from .video_folder import VideoFolderInfo, video_folder
from .zero_images import ZeroImageInfo, zero_images


@dataclass
class DataloadersInfo:
    """DataloadersInfo

        train_loader (torch.utils.data.DataLoader): training set loader
        val_loader (torch.utils.data.DataLoader): validation set loader
        n_classes (int): number of classes
    """
    train_loader: DataLoader
    val_loader: DataLoader
    n_classes: int


SupportedDatasets = Literal["CIFAR10", "ImageFolder", "VideoFolder", "ZeroImages"]


def configure_dataloader(
    command_line_cfg: DictConfig,
    dataset_name: SupportedDatasets,
):
    """dataloader factory

    cfg:
        command_line_cfg (argparse.Namespace): command line cfg
        dataset_name (SupportedDatasets): dataset name (str).
            ["CIFAR10", "ImageFolder", "VideoFolder", "ZeroImages"]

    Raises:
        ValueError: invalid dataset_name is given

    Returns:
        (DataloadersInfo): dataset information
    """

    cfg = command_line_cfg

    if dataset_name == "CIFAR10":
        train_transform, val_transform = \
            transform_image(TransformImageInfo())
        train_loader, val_loader, n_classes = \
            cifar10(Cifar10Info(
                root=cfg.dataset.root,
                batch_size=cfg.training.batch_size,
                num_workers=cfg.training.num_workers,
                train_transform=train_transform,
                val_transform=val_transform
            ))

    elif dataset_name == "ImageFolder":
        train_transform, val_transform = \
            transform_image(TransformImageInfo())
        train_loader, val_loader, n_classes = \
            image_folder(ImageFolderInfo(
                root=cfg.dataset.root,
                train_dir=cfg.dataset.train_dir,
                val_dir=cfg.dataset.val_dir,
                batch_size=cfg.training.batch_size,
                num_workers=cfg.training.num_workers,
                train_transform=train_transform,
                val_transform=val_transform
            ))

    elif dataset_name == "VideoFolder":
        train_transform, val_transform = \
            transform_video(TransformVideoInfo(
                frames_per_clip=cfg.video.frames_per_clip
            ))
        train_loader, val_loader, n_classes = \
            video_folder(VideoFolderInfo(
                root=cfg.dataset.root,
                train_dir=cfg.dataset.train_dir,
                val_dir=cfg.dataset.val_dir,
                batch_size=cfg.training.batch_size,
                num_workers=cfg.training.num_workers,
                train_transform=train_transform,
                val_transform=val_transform,
                clip_duration=cfg.video.clip_duration,
                clips_per_video=cfg.video.clips_per_video
            ))

    elif dataset_name == "ZeroImages":
        train_transform, _ = \
            transform_image(TransformImageInfo())
        train_loader, val_loader, n_classes = \
            zero_images(ZeroImageInfo(
                batch_size=cfg.training.batch_size,
                num_workers=cfg.training.num_workers,
                transform=train_transform,
            ))

    else:
        raise ValueError("invalid dataset_name")

    return DataloadersInfo(
        train_loader=train_loader,
        val_loader=val_loader,
        n_classes=n_classes
    )

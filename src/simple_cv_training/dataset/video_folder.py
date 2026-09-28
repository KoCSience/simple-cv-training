import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from pytorchvideo.data import labeled_video_dataset
from pytorchvideo.data.clip_sampling import (
    ConstantClipsPerVideoSampler,
    RandomClipSampler,
)
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler,
)
from torchvision.transforms import v2 as transforms


@dataclass
class VideoFolderInfo:
    root: str
    train_dir: str
    val_dir: str
    batch_size: int
    num_workers: int
    train_transform: transforms
    val_transform: transforms
    clip_duration: float
    clips_per_video: int


def collate_for_video(batch: Any) -> tuple[Any, Any]:
    batch_dict = torch.utils.data.default_collate(batch)
    return batch_dict["video"], batch_dict["label"]


def video_folder(video_folder_info: VideoFolderInfo) -> tuple[DataLoader, DataLoader, int]:
    """creating dataloaders for videos in folders by pytorchvideo

    Args:
        video_folder_info (VideoFolderInfo): information for dataloaders

    Returns:
        DataLoader: train dataloader
        DataLoader: val dataloader
        int: number of classes
    """

    root_train_dir = os.path.join(video_folder_info.root, video_folder_info.train_dir)
    root_val_dir = os.path.join(video_folder_info.root, video_folder_info.val_dir)
    assert os.path.exists(root_train_dir) and os.path.isdir(root_train_dir)
    assert os.path.exists(root_val_dir) and os.path.isdir(root_val_dir)

    train_dataset = labeled_video_dataset(
        data_path=root_train_dir,
        clip_sampler=RandomClipSampler(
            clip_duration=video_folder_info.clip_duration,
        ),
        video_sampler=RandomSampler,
        transform=video_folder_info.train_transform,
        decode_audio=False,
        decoder="pyav",
    )
    val_dataset = labeled_video_dataset(
        data_path=root_val_dir,
        clip_sampler=ConstantClipsPerVideoSampler(
            clip_duration=video_folder_info.clip_duration, clips_per_video=video_folder_info.clips_per_video
        ),
        video_sampler=SequentialSampler,
        transform=video_folder_info.val_transform,
        decode_audio=False,
        decoder="pyav",
    )

    train_dataset.classes = sorted([d.name for d in Path(root_train_dir).iterdir()])
    val_dataset.classes = sorted([d.name for d in Path(root_val_dir).iterdir()])
    assert train_dataset.classes == val_dataset.classes

    train_dataset.n_classes = len(train_dataset.classes)
    val_dataset.n_classes = len(val_dataset.classes)
    assert train_dataset.n_classes == val_dataset.n_classes

    train_loader = DataLoader(
        LimitDataset(train_dataset),
        batch_size=video_folder_info.batch_size,
        drop_last=True,
        num_workers=video_folder_info.num_workers,
        collate_fn=collate_for_video,
    )
    val_loader = DataLoader(
        LimitDataset(val_dataset),
        batch_size=video_folder_info.batch_size,
        drop_last=False,
        num_workers=video_folder_info.num_workers,
        collate_fn=collate_for_video,
    )

    return train_loader, val_loader, train_dataset.n_classes


class LimitDataset(torch.utils.data.Dataset):
    """Expose a fixed-length, map-style view of an iterable video dataset.

    Some videos can fail while being fetched or decoded. The wrapper therefore allows
    the underlying dataset to be traversed at most twice while still exposing
    ``num_videos`` samples to ``DataLoader``. It intentionally keeps the active
    iterator out of constructor state so that spawn- and forkserver-based workers can
    pickle the wrapper before creating their worker-local iterator.

    Iteration is implemented with a bounded loop instead of recursion. Fetching N
    samples takes O(N) time and O(1) additional memory and Python stack space.

    https://github.com/facebookresearch/pytorchvideo/blob/f7e7a88a9a04b70cb65a564acfc38538fe71ff7b/tutorials/video_classification_example/train.py#L341
    https://github.com/facebookresearch/pytorchvideo/issues/96
    """

    _MAX_DATASET_PASSES = 2

    def __init__(self, dataset: Any) -> None:
        """Store only pickle-safe source state until sample retrieval starts."""
        super().__init__()
        self.dataset = dataset
        self._dataset_iter: Iterator[Any] | None = None
        self._remaining_dataset_passes = self._MAX_DATASET_PASSES
        self._previous_index: int | None = None

    def __getitem__(self, index: int) -> Any:
        """Return the next sample while preserving map-style batching semantics."""
        if self._previous_index is not None and index <= self._previous_index:
            self._reset_iteration_state()

        self._previous_index = index
        return self._next_sample()

    def _next_sample(self) -> Any:
        """Read one sample, restarting the source at most once after exhaustion."""
        while self._remaining_dataset_passes > 0:  # loop
            if self._dataset_iter is None:
                self._dataset_iter = iter(self.dataset)
                self._remaining_dataset_passes -= 1  # loop var

            try:
                return next(self._dataset_iter)
            except StopIteration:
                self._dataset_iter = None

        raise RuntimeError(
            "LimitDataset exhausted the underlying dataset after "
            f"{self._MAX_DATASET_PASSES} passes before producing the required samples."
        )

    def _reset_iteration_state(self) -> None:
        """Reset worker-local state when sequential indices start a new epoch."""
        self._dataset_iter = None
        self._remaining_dataset_passes = self._MAX_DATASET_PASSES

    def __len__(self) -> int:
        """Return the fixed number of samples requested for each epoch."""
        return self.dataset.num_videos

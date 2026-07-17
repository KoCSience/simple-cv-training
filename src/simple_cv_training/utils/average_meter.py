from typing import Literal

import torch

from simple_cv_training.utils.mixin import (
    GetMetricsDictMixin,
)


class AverageMeter:
    """
    Computes and stores the average and current value
    Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    https://github.com/machine-perception-robotics-group/attention_branch_network/blob/ced1d97303792ac6d56442571d71bb0572b3efd8/utils/misc.py#L59
    """

    def __init__(self):
        """average meter"""
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, value: float | torch.Tensor, n: int = 1):
        """update the statistics

        Args:
            value (float or torch.Tensor): a value used for averaging
            n (int, optional): multiplier of the curent value for averaging. Defaults to 1.
        """
        if isinstance(value, torch.Tensor):
            value = value.item()
        self.value = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count


class AvgMeterLossTopk(GetMetricsDictMixin):
    """Track loss and top-k metrics while preserving their one-to-one mapping."""

    def __init__(self, mode_name: Literal["train", "val"], topk: tuple[int, ...] = (1, 5)):
        """Create a loss meter and one meter for each configured top-k value.

        Args:
            mode_name (Literal['train', 'val']): prefix
            topk (Tuple[int], optional): specifying (1, 5) logs top1 and top5. Defaults to (1, 5).
        """
        self.mode_name = mode_name
        self.topk = topk
        self.loss_meter = AverageMeter()
        self.topk_meters = [AverageMeter() for _ in topk]

    def update(
        self, loss: float | torch.Tensor, topk_values: tuple[float, ...] | tuple[torch.Tensor, ...], batch_size: int = 1
    ):
        """Update all batch statistics after validating the top-k correspondence.

        Args:
            loss (float | torch.Tensor): a batch loss
            topk_values (Tuple[float, ...] | Tuple[torch.Tensor, ...]): a batch topk values
            batch_size (int, optional): batch size for the loss. Defaults to 1.

        Raises:
            ValueError: If ``topk_values`` does not contain exactly one value
                for each configured top-k metric. Validation happens before any
                meter is updated to avoid leaving partially updated statistics.
        """
        if len(topk_values) != len(self.topk):
            raise ValueError(
                f"topk_values must contain {len(self.topk)} values to match topk={self.topk}; got {len(topk_values)}"
            )

        self.loss_meter.update(loss, batch_size)
        for meter, value in zip(self.topk_meters, topk_values, strict=True):
            meter.update(value, batch_size)  # type: ignore[arg-type]

    def get_meters(self):
        return self.loss_meter, self.topk_meters, self.topk

    def get_mode_name(self):
        return self.mode_name

import pytest
from torch import Tensor

from simple_cv_training.utils import (
    AverageMeter,
    AvgMeterLossTopk,
)


@pytest.mark.parametrize(
    "values,avg",
    [
        ((1.0, 2.0, 3.0), 2.0),
        ((3.0, 2.0, 4.0), 3.0),
        (Tensor([1.0, 2.0, 3.0]), 2.0),
        (Tensor([3.0, 2.0, 4.0]), 3.0),
    ],
)
def test_average_meter(
    values,
    avg,
):
    meter = AverageMeter()
    for value in values:
        meter.update(value)

    assert meter.avg == avg
    assert isinstance(meter.avg, float)
    assert meter.count == len(values)


@pytest.mark.parametrize(
    "loss_list,topk_list,loss_avg,topk_avg_list",
    [
        (
            (9.0, 10.0, 11.0),
            (
                (1.0, 3.0),
                (2.0, 2.0),
                (3.0, 4.0),
            ),
            10.0,
            (2.0, 3.0),
        )
    ],
)
def test_average_meters_loss_topk(
    loss_list,
    topk_list,
    loss_avg,
    topk_avg_list,
):
    meters = AvgMeterLossTopk("train")
    for loss, topk in zip(loss_list, topk_list, strict=True):
        top1, top5 = topk
        meters.update(loss, (top1, top5))

    assert meters.loss_meter.avg == loss_avg

    for meter, avg in zip(meters.topk_meters, topk_avg_list, strict=True):
        assert meter.avg == avg


@pytest.mark.parametrize("topk_values", [(90.0,), (90.0, 95.0, 99.0)])
def test_average_meters_reject_mismatched_topk_values_before_update(topk_values):
    meters = AvgMeterLossTopk("train", topk=(1, 5))

    with pytest.raises(
        ValueError,
        match=r"topk_values must contain 2 values to match topk=\(1, 5\); got",
    ):
        meters.update(loss=2.5, topk_values=topk_values, batch_size=4)

    assert meters.loss_meter.count == 0
    assert meters.loss_meter.sum == 0
    assert all(meter.count == 0 for meter in meters.topk_meters)
    assert all(meter.sum == 0 for meter in meters.topk_meters)


@pytest.mark.parametrize(
    "get_metrics_dict",
    [
        AvgMeterLossTopk.get_step_metrics_dict,
        AvgMeterLossTopk.get_epoch_metrics_dict,
    ],
)
def test_metrics_dict_rejects_mismatched_topk_meters(get_metrics_dict):
    meters = AvgMeterLossTopk("train", topk=(1, 5))
    meters.topk = (1,)

    with pytest.raises(ValueError, match="zip"):
        get_metrics_dict(meters)

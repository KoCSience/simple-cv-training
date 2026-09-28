import pytest

from simple_cv_training.utils import TqdmLossTopK


def test_tqdm_loss_topk_formats_matching_values():
    progress = TqdmLossTopK(total=0, disable=True)

    try:
        postfix = progress.set_postfix_str_loss_topk(
            global_step=7,
            loss=1.25,
            topk_values=(80.0, 95.0),
            topk=(1, 5),
        )
    finally:
        progress.close()

    assert postfix == "step=7, loss=1.2500e+00 top1= 80.00 top5= 95.00 "


@pytest.mark.parametrize("topk_values", [(80.0,), (80.0, 95.0, 99.0)])
def test_tqdm_loss_topk_rejects_mismatched_values_before_update(topk_values):
    progress = TqdmLossTopK(total=0, disable=True)
    progress.postfix_str = "unchanged"

    try:
        with pytest.raises(
            ValueError,
            match=r"topk_values must contain 2 values to match topk=\(1, 5\); got",
        ):
            progress.set_postfix_str_loss_topk(
                global_step=7,
                loss=1.25,
                topk_values=topk_values,
                topk=(1, 5),
            )
    finally:
        progress.close()

    assert progress.postfix_str == "unchanged"


def test_add_topk_rejects_mismatched_values_before_update():
    progress = TqdmLossTopK(total=0, disable=True)
    progress.postfix_str = "unchanged"

    try:
        with pytest.raises(ValueError, match="topk_values must contain 2 values"):
            progress.add_topk_to_postfix_str(
                topk_values=(80.0,),
                topk=(1, 5),
            )
    finally:
        progress.close()

    assert progress.postfix_str == "unchanged"

import multiprocessing
import pickle
from collections.abc import Iterator

import pytest
import torch
from torch.utils.data import DataLoader, get_worker_info

from simple_cv_training.dataset.video_folder import LimitDataset


class FixedVideoDataset:
    """Provide deterministic samples and the ``num_videos`` source contract."""

    def __init__(self, values: tuple[int, ...], *, num_videos: int | None = None) -> None:
        self.values = values
        self.num_videos = len(values) if num_videos is None else num_videos

    def __iter__(self) -> Iterator[int]:
        """Return a fresh iterator for each bounded dataset pass."""
        return iter(self.values)


class WorkerShardedVideoDataset:
    """Mirror PyTorchVideo's worker-aware sampler with deterministic sample IDs."""

    def __init__(self, num_videos: int) -> None:
        self.num_videos = num_videos

    def __iter__(self) -> Iterator[int]:
        """Give each DataLoader worker the IDs assigned to its dataset replica."""
        worker_info = get_worker_info()
        if worker_info is None:
            return iter(range(self.num_videos))

        return iter(range(worker_info.id, self.num_videos, worker_info.num_workers))


MULTIPROCESSING_CONTEXTS = [
    pytest.param(
        context,
        marks=pytest.mark.skipif(
            context not in multiprocessing.get_all_start_methods(),
            reason=f"{context} multiprocessing context is unavailable on this platform.",
        ),
    )
    for context in ("spawn", "forkserver")
]


def test_limit_dataset_is_picklable_before_iteration() -> None:
    """Constructor state must remain serializable until a worker starts reading."""
    dataset = LimitDataset(FixedVideoDataset((10, 20, 30)))

    assert dataset._dataset_iter is None

    restored = pickle.loads(pickle.dumps(dataset))

    assert restored._dataset_iter is None
    assert [restored[index] for index in range(len(restored))] == [10, 20, 30]


def test_limit_dataset_handles_more_samples_than_recursion_limit() -> None:
    """Large datasets must use constant Python stack space instead of recursion."""
    num_videos = 1_500
    loader = DataLoader(
        LimitDataset(FixedVideoDataset(tuple(range(num_videos)))),
        batch_size=128,
        num_workers=0,
    )

    retrieved_samples = sum(batch.numel() for batch in loader)

    assert retrieved_samples == num_videos


def test_limit_dataset_resets_for_each_single_process_epoch() -> None:
    """Sequential indices must restart the worker-local iterator every epoch."""
    loader = DataLoader(
        LimitDataset(FixedVideoDataset((0, 1, 2, 3))),
        batch_size=2,
        num_workers=0,
    )

    first_epoch = torch.cat(list(loader)).tolist()
    second_epoch = torch.cat(list(loader)).tolist()

    assert first_epoch == [0, 1, 2, 3]
    assert second_epoch == first_epoch


@pytest.mark.parametrize("multiprocessing_context", MULTIPROCESSING_CONTEXTS)
def test_limit_dataset_supports_serializing_worker_contexts(multiprocessing_context: str) -> None:
    """Spawn and forkserver workers must retrieve each sharded sample exactly once."""
    num_videos = 8
    loader = DataLoader(
        LimitDataset(WorkerShardedVideoDataset(num_videos)),
        batch_size=1,
        num_workers=2,
        multiprocessing_context=multiprocessing_context,
    )

    sample_ids = [batch.item() for batch in loader]

    assert sample_ids == list(range(num_videos))
    assert len(sample_ids) == len(set(sample_ids))


def test_limit_dataset_uses_second_pass_to_reach_fixed_length() -> None:
    """The preserved second pass must replace samples missing from the first pass."""
    dataset = LimitDataset(FixedVideoDataset((7,), num_videos=2))

    assert [dataset[index] for index in range(len(dataset))] == [7, 7]


def test_limit_dataset_reports_exhaustion_after_two_passes() -> None:
    """A permanently undersized source must fail clearly instead of looping forever."""
    dataset = LimitDataset(FixedVideoDataset((7,), num_videos=3))

    assert dataset[0] == 7
    assert dataset[1] == 7
    with pytest.raises(RuntimeError, match="after 2 passes"):
        dataset[2]

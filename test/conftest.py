from __future__ import annotations

from pathlib import Path

import pytest

GPU_TEST_PATH_PARTS = {
    ("test", "model"),
    ("test", "setup"),
}

GPU_TEST_FILES = {
    Path("test/utils/test_checkpoint.py"),
}

EXTERNAL_DATA_TEST_FILES = {
    Path("test/dataset/test_video_folder.py"),
}


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-gpu",
        action="store_true",
        default=False,
        help="Run tests marked as requiring CUDA-capable GPU hardware.",
    )
    parser.addoption(
        "--run-external-data",
        action="store_true",
        default=False,
        help="Run tests that require local or NAS dataset paths.",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    run_gpu = config.getoption("--run-gpu")
    run_external_data = config.getoption("--run-external-data")
    skip_gpu = pytest.mark.skip(reason="GPU test skipped; pass --run-gpu to run it.")
    skip_external_data = pytest.mark.skip(
        reason="External data test skipped; pass --run-external-data to run it."
    )

    for item in items:
        path = Path(str(item.fspath))
        relative_path = _relative_to_repo(path)

        if _is_gpu_test(relative_path):
            item.add_marker(pytest.mark.gpu)
            if not run_gpu:
                item.add_marker(skip_gpu)

        if _is_external_data_test(relative_path):
            item.add_marker(pytest.mark.external_data)
            if not run_external_data:
                item.add_marker(skip_external_data)


def _relative_to_repo(path: Path) -> Path:
    try:
        return path.relative_to(Path.cwd())
    except ValueError:
        return path


def _is_gpu_test(path: Path) -> bool:
    path_parts = path.parts
    return (
        any(path_parts[: len(parts)] == parts for parts in GPU_TEST_PATH_PARTS)
        or path in GPU_TEST_FILES
    )


def _is_external_data_test(path: Path) -> bool:
    return path in EXTERNAL_DATA_TEST_FILES

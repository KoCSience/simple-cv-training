from __future__ import annotations

import platform

import torch

from .hardware import is_bf16_supported


def collect_environment_info() -> dict[str, object]:
    """Collect runtime facts needed to interpret optimization benchmark results."""
    cuda_available = torch.cuda.is_available()
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "cuda_device_count": torch.cuda.device_count() if cuda_available else 0,
        "gpu_name": torch.cuda.get_device_name(0) if cuda_available else None,
        "bf16_supported": is_bf16_supported() if cuda_available else False,
    }

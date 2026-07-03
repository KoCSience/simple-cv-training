from __future__ import annotations

import torch


def is_bf16_supported() -> bool:
    """Return whether the current PyTorch runtime reports CUDA bf16 support."""
    checker = getattr(torch.cuda, "is_bf16_supported", None)
    if checker is None:
        return False
    return bool(torch.cuda.is_available() and checker())

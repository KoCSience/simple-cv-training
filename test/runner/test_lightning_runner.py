from __future__ import annotations

import pytest

from runner import lightning_runner


def test_lightning_runner_rejects_manual_checkpoint_suffix() -> None:
    with pytest.raises(ValueError, match="manual .pt"):
        lightning_runner._require_lightning_checkpoint_format("/tmp/model.pt")

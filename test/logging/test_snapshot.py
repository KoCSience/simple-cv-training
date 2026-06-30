from __future__ import annotations

import json

from omegaconf import OmegaConf
from simple_cv_core.logging import write_experiment_snapshot


def test_write_experiment_snapshot(tmp_path) -> None:
    cfg = OmegaConf.create({"experiment_name": "unit", "training": {"num_epochs": 1}})

    snapshot_path = write_experiment_snapshot(cfg, tmp_path)

    assert snapshot_path.exists()
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert snapshot["config"]["experiment_name"] == "unit"
    assert (tmp_path / "config.yaml").exists()

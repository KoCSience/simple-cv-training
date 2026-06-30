from __future__ import annotations

import hydra
from omegaconf import DictConfig
import sys
from pathlib import Path

from config.hydra_compat import patch_hydra_argparse_for_python314
from runner import run_lightning_training

workspace_root = str(Path.cwd())
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

patch_hydra_argparse_for_python314()


@hydra.main(version_base=None, config_path="../../../../../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    run_lightning_training(cfg)

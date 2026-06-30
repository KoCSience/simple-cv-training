from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import torch
from omegaconf import DictConfig, OmegaConf


class ExperimentSnapshotCallback(pl.Callback):
    def __init__(self, cfg: DictConfig, output_dir: Path | str):
        self.cfg = cfg
        self.output_dir = Path(output_dir)

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.global_rank != 0:
            return
        write_experiment_snapshot(self.cfg, self.output_dir)


def write_experiment_snapshot(cfg: DictConfig, output_dir: Path | str) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    snapshot = {
        "framework": {
            "core_version": _package_version("simple-cv-core"),
            "vision_version": _package_version("simple-cv-vision"),
        },
        "git": _git_info(),
        "dependency": {
            "uv_lock_sha256": _sha256(Path("uv.lock")),
            "python_version": platform.python_version(),
        },
        "environment": _environment_info(),
        "config": OmegaConf.to_container(cfg, resolve=True),
    }

    snapshot_path = output_path / "snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot, indent=2, default=str), encoding="utf-8")
    OmegaConf.save(config=cfg, f=str(output_path / "config.yaml"), resolve=True)
    return snapshot_path


def _git_info() -> dict[str, Any]:
    return {
        "commit": _run_git(["rev-parse", "HEAD"]),
        "dirty": bool(_run_git(["status", "--porcelain"])),
    }


def _environment_info() -> dict[str, Any]:
    return {
        "os": platform.platform(),
        "cuda_available": torch.cuda.is_available(),
        "cuda": torch.version.cuda,
        "gpu": _gpu_names(),
    }


def _gpu_names() -> list[str]:
    if not torch.cuda.is_available():
        return []
    return [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _run_git(args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _package_version(package_name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(package_name)
    except Exception:
        return None

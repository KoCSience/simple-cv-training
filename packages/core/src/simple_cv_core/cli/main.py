from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 2:
        _print_help()
        raise SystemExit(2)

    command = sys.argv[1]
    if command == "train":
        _add_workspace_to_import_path()
        sys.argv = [sys.argv[0], *sys.argv[2:]]
        from main_pl import main as train_main

        train_main()
        return

    _print_help()
    raise SystemExit(f"unknown scv command: {command}")


def _print_help() -> None:
    print("usage: scv <command> [hydra overrides]")
    print("")
    print("commands:")
    print("  train    run Lightning training")


def _add_workspace_to_import_path() -> None:
    workspace_root = str(Path.cwd())
    if workspace_root not in sys.path:
        sys.path.insert(0, workspace_root)

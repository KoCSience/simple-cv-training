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
    if command == "doctor":
        _add_workspace_to_import_path()
        from simple_cv_core.doctor.checks import format_doctor_results, has_failures, run_doctor

        results = run_doctor(sys.argv[2:] or None)
        print(format_doctor_results(results))
        raise SystemExit(1 if has_failures(results) else 0)

    _print_help()
    raise SystemExit(f"unknown scv command: {command}")


def _print_help() -> None:
    print("usage: scv <command> [hydra overrides]")
    print("")
    print("commands:")
    print("  train    run Lightning training")
    print("  doctor   run repository health checks")


def _add_workspace_to_import_path() -> None:
    workspace_root = str(Path.cwd())
    if workspace_root not in sys.path:
        sys.path.insert(0, workspace_root)

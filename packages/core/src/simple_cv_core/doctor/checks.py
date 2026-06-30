from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DoctorResult:
    status: str
    name: str
    message: str


def run_doctor(checks: list[str] | None = None, root: Path | None = None) -> list[DoctorResult]:
    root = root or Path.cwd()
    selected = set(checks or ["safety", "versions", "data", "naming", "inheritance"])
    results: list[DoctorResult] = []

    if "safety" in selected:
        results.extend(_check_safety(root))
    if "versions" in selected:
        results.extend(_check_versions(root))
    if "data" in selected:
        results.extend(_check_data(root))
    if "naming" in selected:
        results.append(_check_naming(root))
    if "inheritance" in selected:
        results.append(DoctorResult("OK", "inheritance", "inheritance checks are registered for later strict validation"))

    return sorted(results, key=lambda result: _SEVERITY_ORDER[result.status])


def format_doctor_results(results: list[DoctorResult]) -> str:
    return "\n".join(f"[{result.status:<4}] {result.name:<16} {result.message}" for result in results)


def has_failures(results: list[DoctorResult]) -> bool:
    return any(result.status == "FAIL" for result in results)


def _check_safety(root: Path) -> list[DoctorResult]:
    results = []
    pyproject = root / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8") if pyproject.exists() else ""
    if "path = " in text and 'mode = "framework-dev"' not in text:
        results.append(DoctorResult("FAIL", "safety.path_dep", "path dependency found outside framework-dev mode"))
    else:
        results.append(DoctorResult("OK", "safety.path_dep", "no forbidden path dependency found"))

    if (root / ".env").exists():
        results.append(DoctorResult("WARN", "safety.env", ".env exists locally; doctor did not read it"))
    else:
        results.append(DoctorResult("OK", "safety.env", "no local .env file found"))
    return results


def _check_versions(root: Path) -> list[DoctorResult]:
    results = []
    results.append(_exists(root / "pyproject.toml", "versions.pyproject"))
    results.append(_exists(root / "uv.lock", "versions.lockfile"))
    return results


def _check_data(root: Path) -> list[DoctorResult]:
    return [_exists(root / "configs" / "data", "data.configs")]


def _check_naming(root: Path) -> DoctorResult:
    if root.name.startswith("20") or root.name in {"simple-cv-training", "simple_cv_framework"}:
        return DoctorResult("OK", "naming.repo", f"repository name accepted: {root.name}")
    return DoctorResult("WARN", "naming.repo", f"repository name does not look like a student repo: {root.name}")


def _exists(path: Path, name: str) -> DoctorResult:
    if path.exists():
        return DoctorResult("OK", name, f"found {path.name}")
    return DoctorResult("FAIL", name, f"missing {path}")


_SEVERITY_ORDER = {
    "FAIL": 0,
    "WARN": 1,
    "INFO": 2,
    "OK": 3,
}

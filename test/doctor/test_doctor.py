from __future__ import annotations

from simple_cv_core.doctor.checks import format_doctor_results, has_failures, run_doctor


def test_doctor_reports_missing_lockfile(tmp_path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'x'\n", encoding="utf-8")

    results = run_doctor(["versions"], root=tmp_path)

    assert has_failures(results)
    assert "versions.lockfile" in format_doctor_results(results)

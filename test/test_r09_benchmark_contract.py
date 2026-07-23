from __future__ import annotations

import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_r09_matched_acceleration.py"
_SPEC = importlib.util.spec_from_file_location("benchmark_r09_matched_acceleration", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
r09 = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = r09
_SPEC.loader.exec_module(r09)


def _scenario(
    *,
    velocity_median: float = 1.0,
    velocity_p95: float = 1.0,
    acceleration_median: float = 1.1,
    acceleration_p95: float = 1.1,
    checks_passed: bool = True,
    thresholds_enforced: bool = True,
    collision_lift: bool = False,
) -> dict:
    return {
        "performance_contract": {
            "thresholds_enforced": thresholds_enforced,
            "scope": (
                "diagnostic_unmatched_sample_validation"
                if collision_lift
                else "matched_solver_wall"
            ),
        },
        "correctness": {"checks_passed": checks_passed},
        "collision_identity": {
            "checks_passed": checks_passed,
            "declared": collision_lift,
            "mode": "velocity_collision_lift" if collision_lift else "none",
        },
        "velocity": {
            "timing_ms": {
                "wall": {
                    "median": velocity_median,
                    "p95": velocity_p95,
                }
            }
        },
        "acceleration": {
            "timing_ms": {
                "wall": {
                    "median": acceleration_median,
                    "p95": acceleration_p95,
                }
            }
        },
    }


def _artifact(
    *,
    benchmark_id: str | None = None,
    head: str = "HEAD",
    dirty: bool = False,
    project_version: str = "0.20.18",
    build_config: dict | None = None,
    scenario: dict | None = None,
) -> dict:
    selected = scenario or _scenario()
    collision = _scenario(thresholds_enforced=False, collision_lift=True)
    return {
        "schema_version": r09.SCHEMA_VERSION,
        "benchmark_id": benchmark_id or r09.BENCHMARK_ID,
        "metadata": {
            "git": {"head": head, "dirty": dirty},
            "project_version": project_version,
            "build_config": build_config or {"cmake_build_type": "Release"},
        },
        "parameters": {
            "warmup": r09.MIN_ACCEPTANCE_WARMUP,
            "repetitions": r09.MIN_ACCEPTANCE_REPETITIONS,
            "dt": r09.ACCEPTANCE_DT,
            "scenario_keys": sorted(r09.REQUIRED_SCENARIO_KEYS),
        },
        "scenarios": {
            "panda_fixed_base_frame_position": selected,
            "dual_iiwa_fixed_base_frame_position": deepcopy(selected),
            "dual_iiwa_masked_collision_velocity_lift": collision,
        },
    }


def _write(path: Path, data: dict) -> Path:
    path.write_text(json.dumps(data))
    return path


@pytest.fixture
def current_repo(monkeypatch):
    monkeypatch.setattr(r09, "_git_head", lambda: "HEAD")
    monkeypatch.setattr(r09, "_project_version", lambda: "0.20.18")
    monkeypatch.setattr(r09, "_build_config", lambda: {"cmake_build_type": "Release"})


def test_immutable_output_refuses_overwrite(tmp_path):
    output = tmp_path / "r09.json"
    r09._write_immutable_json(output, {"ok": True})

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        r09._write_immutable_json(output, {"ok": False})

    assert json.loads(output.read_text()) == {"ok": True}


def test_compare_rejects_dirty_stale_wrong_config_and_public_artifacts(tmp_path, current_repo):
    dirty = _write(tmp_path / "dirty.json", _artifact(dirty=True))
    with pytest.raises(ValueError, match="dirty worktree"):
        r09.evaluate_artifact(dirty)

    stale = _write(tmp_path / "stale.json", _artifact(head="OLD"))
    with pytest.raises(ValueError, match="stale"):
        r09.evaluate_artifact(stale)

    wrong_config = _write(
        tmp_path / "wrong_config.json",
        _artifact(build_config={"cmake_build_type": "Debug"}),
    )
    with pytest.raises(ValueError, match="wrong build_config"):
        r09.evaluate_artifact(wrong_config)

    public_dir = r09.REPO_ROOT / ".benchmarks"
    public_dir.mkdir(exist_ok=True)
    public_artifact = public_dir / "r09_contract_public_reject.json"
    public_artifact.write_text(json.dumps(_artifact()))
    try:
        with pytest.raises(ValueError, match="public \\.benchmarks"):
            r09.evaluate_artifact(public_artifact)
    finally:
        public_artifact.unlink()


def test_compare_checks_correctness_before_timing_thresholds(tmp_path, current_repo):
    bad = _artifact(
        scenario=_scenario(
            acceleration_median=10.0,
            acceleration_p95=10.0,
            checks_passed=False,
        )
    )
    candidate = _write(tmp_path / "candidate.json", bad)
    with pytest.raises(ValueError, match="failed correctness checks"):
        r09.evaluate_artifact(candidate)


def test_compare_enforces_acceleration_overhead_thresholds(tmp_path, current_repo):
    candidate_data = _artifact(
        scenario=_scenario(
            velocity_median=1.0,
            velocity_p95=1.0,
            acceleration_median=1.16,
            acceleration_p95=1.21,
        )
    )
    candidate = _write(tmp_path / "candidate.json", candidate_data)
    result = r09.evaluate_artifact(candidate)

    assert not result["passed"]
    assert any("median overhead" in failure for failure in result["failures"])
    assert any("p95 overhead" in failure for failure in result["failures"])


def test_compare_reports_but_does_not_gate_unmatched_collision_validation(tmp_path, current_repo):
    candidate_data = _artifact()
    candidate_data["scenarios"]["dual_iiwa_masked_collision_velocity_lift"] = _scenario(
        acceleration_median=2.0,
        acceleration_p95=2.0,
        thresholds_enforced=False,
        collision_lift=True,
    )
    candidate = _write(tmp_path / "candidate.json", candidate_data)
    result = r09.evaluate_artifact(candidate)

    assert result["passed"]
    comparison = result["comparisons"]["dual_iiwa_masked_collision_velocity_lift"]
    assert not comparison["thresholds_enforced"]
    assert comparison["acceleration_over_velocity_median_overhead"] == 1.0


def test_compare_refuses_threshold_skip_for_non_collision_scenario(tmp_path, current_repo):
    candidate_data = _artifact(
        scenario=_scenario(
            thresholds_enforced=False,
            collision_lift=False,
        )
    )
    candidate = _write(tmp_path / "candidate.json", candidate_data)

    with pytest.raises(ValueError, match="must enforce"):
        r09.evaluate_artifact(candidate)


def test_compare_requires_complete_acceptance_scenario_set(tmp_path, current_repo):
    candidate_data = _artifact()
    del candidate_data["scenarios"]["panda_fixed_base_frame_position"]
    candidate = _write(tmp_path / "candidate.json", candidate_data)

    with pytest.raises(ValueError, match="scenarios must be exactly"):
        r09.evaluate_artifact(candidate)

    candidate_data = _artifact()
    candidate_data["scenarios"]["extra"] = _scenario()
    candidate = _write(tmp_path / "candidate-extra.json", candidate_data)

    with pytest.raises(ValueError, match="scenarios must be exactly"):
        r09.evaluate_artifact(candidate)


def test_compare_rejects_undersampled_or_wrong_dt_artifact(tmp_path, current_repo):
    undersampled_data = _artifact()
    undersampled_data["parameters"]["repetitions"] = 1
    undersampled = _write(tmp_path / "undersampled.json", undersampled_data)
    with pytest.raises(ValueError, match="repetitions"):
        r09.evaluate_artifact(undersampled)

    wrong_dt_data = _artifact()
    wrong_dt_data["parameters"]["dt"] = 0.02
    wrong_dt = _write(tmp_path / "wrong-dt.json", wrong_dt_data)
    with pytest.raises(ValueError, match="dt must be"):
        r09.evaluate_artifact(wrong_dt)

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import embodik

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "10_centroidal_stability.py"


def _load_example():
    spec = importlib.util.spec_from_file_location("centroidal_stability_example", EXAMPLE)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_centroidal_stability_example_runs_both_solver_levels_headlessly() -> None:
    module = _load_example()

    report = module.run_headless_demo()

    assert report["velocity"]["status"] == embodik.SolverStatus.SUCCESS.name
    assert report["velocity"]["capture_point_min_slack"] >= -1e-8
    assert report["velocity"]["zmp_min_slack"] >= -1e-8
    assert report["acceleration"]["status"] == embodik.SolverStatus.SUCCESS.name
    assert report["acceleration"]["capture_point_min_slack"] >= -1e-8
    assert report["acceleration"]["zmp_min_slack"] >= -1e-8
    assert report["acceleration"]["force_z"] >= 1.0
    assert report["capabilities"]["supports_dynamic_balance"] is False


def test_centroidal_stability_example_json_cli_is_machine_readable() -> None:
    result = subprocess.run(
        [sys.executable, str(EXAMPLE), "--json"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )

    report = json.loads(result.stdout)
    assert report["velocity"]["status"] == "SUCCESS"
    assert report["acceleration"]["status"] == "SUCCESS"

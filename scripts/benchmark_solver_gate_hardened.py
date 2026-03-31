#!/usr/bin/env python3
"""Run both perf gates repeatedly and aggregate robust medians."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.median(values))


def _run_gate(command: list[str], output_path: Path) -> dict:
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    if completed.stdout:
        print(completed.stdout.strip())
    if completed.stderr:
        print(completed.stderr.strip(), file=sys.stderr)
    return json.loads(output_path.read_text())


def _collect_refactor_metrics(report: dict) -> dict[str, float]:
    workloads = report.get("workloads", {})
    wall_p50 = []
    wall_p95 = []
    coll_p50 = []
    coll_p95 = []
    for workload in workloads.values():
        wall = workload.get("wall_time_ms", {})
        coll = workload.get("collision_time_ms", {})
        wall_p50.append(float(wall.get("p50", 0.0)))
        wall_p95.append(float(wall.get("p95", 0.0)))
        coll_p50.append(float(coll.get("p50", 0.0)))
        coll_p95.append(float(coll.get("p95", 0.0)))
    return {
        "gate1_wall_p50_median": _median(wall_p50),
        "gate1_wall_p95_median": _median(wall_p95),
        "gate1_collision_p50_median": _median(coll_p50),
        "gate1_collision_p95_median": _median(coll_p95),
    }


def _collect_position_step_metrics(report: dict) -> dict[str, float]:
    workloads = report.get("workloads", {})
    panda = workloads.get("panda_position_step_stall", {})
    dual = workloads.get("dual_iiwa_position_step_stall", {})
    return {
        "gate2_panda_p50": float(panda.get("p50_ms", 0.0)),
        "gate2_panda_p95": float(panda.get("p95_ms", 0.0)),
        "gate2_dual_p50": float(dual.get("p50_ms", 0.0)),
        "gate2_dual_p95": float(dual.get("p95_ms", 0.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run hardened multi-run perf gates.")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".cursor/reports/perf_runs/gate_hardened_summary.json"),
    )
    args = parser.parse_args()

    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1")

    out_dir = args.output.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    run_reports = []
    aggregate_columns: dict[str, list[float]] = {}

    for run_idx in range(args.repeats):
        gate1_out = out_dir / f"refactor_gate_run_{run_idx + 1}.json"
        gate2_out = out_dir / f"position_step_gate_run_{run_idx + 1}.json"

        gate1 = _run_gate(
            [
                sys.executable,
                "scripts/benchmark_solver_refactor_gate.py",
                "--warmup",
                str(args.warmup),
                "--steps",
                str(args.steps),
                "--output",
                str(gate1_out),
            ],
            gate1_out,
        )
        gate2 = _run_gate(
            [
                sys.executable,
                "scripts/benchmark_position_step_refactor_gate.py",
                "--warmup",
                str(args.warmup),
                "--steps",
                str(args.steps),
                "--output",
                str(gate2_out),
            ],
            gate2_out,
        )

        metrics = {}
        metrics.update(_collect_refactor_metrics(gate1))
        metrics.update(_collect_position_step_metrics(gate2))
        run_reports.append(
            {
                "run": run_idx + 1,
                "gate1_output": str(gate1_out),
                "gate2_output": str(gate2_out),
                "metrics": metrics,
            }
        )
        for key, value in metrics.items():
            aggregate_columns.setdefault(key, []).append(value)

    aggregated = {key: _median(values) for key, values in aggregate_columns.items()}
    report = {
        "repeats": args.repeats,
        "warmup_steps": args.warmup,
        "measured_steps": args.steps,
        "runs": run_reports,
        "summary_median_of_runs": aggregated,
    }
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["summary_median_of_runs"], indent=2))


if __name__ == "__main__":
    main()

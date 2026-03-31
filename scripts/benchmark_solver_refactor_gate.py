#!/usr/bin/env python3
"""Focused performance gate for stepwise solver refactors.

This script runs a small, teleop-relevant workload subset repeatedly and emits
compact p50/p95 metrics so each refactor can be compared quickly against a
baseline run.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from benchmark_teleop_workloads import WORKLOADS, run_workload


def _percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"p50": 0.0, "p95": 0.0, "mean": 0.0}
    sorted_vals = sorted(values)
    p50 = sorted_vals[len(sorted_vals) // 2]
    p95_idx = max(0, min(len(sorted_vals) - 1, int((0.95 * len(sorted_vals)) - 1)))
    return {"p50": float(p50), "p95": float(sorted_vals[p95_idx]), "mean": float(statistics.fmean(values))}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run focused solver perf gate.")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup steps per workload.")
    parser.add_argument("--steps", type=int, default=120, help="Measured steps per workload.")
    parser.add_argument(
        "--workload",
        action="append",
        dest="workloads",
        default=[],
        help="Workload name to include. Can be repeated. Defaults to a focused collision subset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".editor/reports/perf_runs/refactor_gate.json"),
        help="Output JSON path.",
    )
    args = parser.parse_args()

    requested = set(args.workloads) if args.workloads else {
        "panda_collision",
        "dual_arm_collision_simplified",
        "dual_arm_collision_simplified_masked",
    }
    selected = [w for w in WORKLOADS if w.name in requested]
    if not selected:
        raise SystemExit(f"No matching workloads found for: {sorted(requested)}")

    report: dict[str, object] = {
        "warmup_steps": args.warmup,
        "measured_steps": args.steps,
        "workloads": {},
        "summary": {},
    }

    wall_p50 = []
    wall_p95 = []
    coll_p50 = []
    coll_p95 = []

    for workload in selected:
        res = run_workload(workload, args.warmup, args.steps)
        stats = res["stats_ms"]
        wall = stats["wall_time_ms"]
        coll = stats["collision_constraint_time_ms"]
        report["workloads"][workload.name] = {
            "teleop_relevant": bool(res["teleop_relevant"]),
            "status_counts": res["status_counts"],
            "wall_time_ms": {"p50": wall["median"], "p95": wall["p95"], "mean": wall["mean"]},
            "collision_time_ms": {"p50": coll["median"], "p95": coll["p95"], "mean": coll["mean"]},
            "solver_time_ms": {
                "p50": stats["solver_computation_time_ms"]["median"],
                "p95": stats["solver_computation_time_ms"]["p95"],
                "mean": stats["solver_computation_time_ms"]["mean"],
            },
        }
        wall_p50.append(float(wall["median"]))
        wall_p95.append(float(wall["p95"]))
        coll_p50.append(float(coll["median"]))
        coll_p95.append(float(coll["p95"]))

    report["summary"] = {
        "aggregate_wall_time_ms": _percentiles(wall_p50),
        "aggregate_wall_p95_ms": _percentiles(wall_p95),
        "aggregate_collision_time_ms": _percentiles(coll_p50),
        "aggregate_collision_p95_ms": _percentiles(coll_p95),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()


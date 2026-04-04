#!/usr/bin/env python3
"""Summarize hmnd teleop EmbodiK debug JSONL traces.

This parser is intended for temporary co-debugging of deep penetration / stall
events between hmnd teleop runtime and local EmbodiK harness traces.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class TraceSummary:
    label: str
    rows: int
    success_rows: int
    infeasible_rows: int
    not_applied_rows: int
    min_collision_distance: float | None
    min_active_collision_distance: float | None
    min_stall_margin: float | None
    max_dq_norm: float | None


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[warn] {path}:{line_no}: invalid JSON ({exc})")
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _summarize(rows: list[dict[str, Any]], label: str) -> TraceSummary:
    success_rows = 0
    infeasible_rows = 0
    not_applied_rows = 0
    min_collision_distance: float | None = None
    min_active_collision_distance: float | None = None
    min_stall_margin: float | None = None
    max_dq_norm: float | None = None

    for row in rows:
        status = str(row.get("status", ""))
        if "SUCCESS" in status:
            success_rows += 1
        if "INFEASIBLE" in status:
            infeasible_rows += 1
        if row.get("success_applied") is False:
            not_applied_rows += 1

        collision_distance = _safe_float(row.get("collision_distance"))
        if collision_distance is not None:
            min_collision_distance = (
                collision_distance
                if min_collision_distance is None
                else min(min_collision_distance, collision_distance)
            )

        active_distances = row.get("active_collision_distances", [])
        if isinstance(active_distances, list):
            for value in active_distances:
                distance = _safe_float(value)
                if distance is None:
                    continue
                min_active_collision_distance = (
                    distance
                    if min_active_collision_distance is None
                    else min(min_active_collision_distance, distance)
                )

        stall_margin = _safe_float(row.get("stall_min_distance"))
        if stall_margin is not None:
            min_stall_margin = stall_margin if min_stall_margin is None else min(min_stall_margin, stall_margin)

        dq_norm = _safe_float(row.get("dq_norm"))
        if dq_norm is not None:
            max_dq_norm = dq_norm if max_dq_norm is None else max(max_dq_norm, dq_norm)

    return TraceSummary(
        label=label,
        rows=len(rows),
        success_rows=success_rows,
        infeasible_rows=infeasible_rows,
        not_applied_rows=not_applied_rows,
        min_collision_distance=min_collision_distance,
        min_active_collision_distance=min_active_collision_distance,
        min_stall_margin=min_stall_margin,
        max_dq_norm=max_dq_norm,
    )


def _print_summary(summary: TraceSummary) -> None:
    print(f"\n=== {summary.label} ===")
    print(f"rows: {summary.rows}")
    print(f"success rows: {summary.success_rows}")
    print(f"infeasible rows: {summary.infeasible_rows}")
    print(f"success_applied == false rows: {summary.not_applied_rows}")
    print(f"min collision_distance: {summary.min_collision_distance}")
    print(f"min active_collision_distance: {summary.min_active_collision_distance}")
    print(f"min stall_min_distance: {summary.min_stall_margin}")
    print(f"max dq_norm: {summary.max_dq_norm}")


def _print_worst(rows: list[dict[str, Any]], label: str, top_n: int) -> None:
    scored: list[tuple[float, dict[str, Any]]] = []
    for row in rows:
        score = _safe_float(row.get("collision_distance"))
        if score is None:
            active_distances = row.get("active_collision_distances", [])
            if isinstance(active_distances, list):
                vals = [_safe_float(v) for v in active_distances]
                vals = [v for v in vals if v is not None]
                score = min(vals) if vals else None
        if score is None:
            continue
        scored.append((score, row))

    scored.sort(key=lambda item: item[0])
    print(f"\n--- Worst {min(top_n, len(scored))} rows by penetration ({label}) ---")
    for idx, (score, row) in enumerate(scored[:top_n], start=1):
        print(
            f"{idx:02d}. collision={score:.6f} status={row.get('status')} "
            f"dq_norm={row.get('dq_norm')} stall_min={row.get('stall_min_distance')} "
            f"applied={row.get('success_applied')} mode={row.get('collision_tuning_mode')} "
            f"max_constraints={row.get('max_constraints')} step={row.get('step')}"
        )


def _row_penetration_score(row: dict[str, Any]) -> float | None:
    score = _safe_float(row.get("collision_distance"))
    if score is not None:
        return score
    active_distances = row.get("active_collision_distances", [])
    if isinstance(active_distances, list):
        vals = [_safe_float(v) for v in active_distances]
        vals = [v for v in vals if v is not None]
        if vals:
            return min(vals)
    return None


def _print_row_brief(index: int, row: dict[str, Any]) -> None:
    print(
        f"[{index:05d}] step={row.get('step')} status={row.get('status')} "
        f"collision={row.get('collision_distance')} active_min={_row_penetration_score(row)} "
        f"dq_norm={row.get('dq_norm')} stall_min={row.get('stall_min_distance')} "
        f"applied={row.get('success_applied')} mode={row.get('collision_tuning_mode')} "
        f"k={row.get('max_constraints')}"
    )


def _print_incident_windows(
    rows: list[dict[str, Any]],
    label: str,
    incident_window: int,
    penetration_threshold: float,
) -> None:
    first_infeasible_idx: int | None = None
    first_penetration_idx: int | None = None
    for idx, row in enumerate(rows):
        status = str(row.get("status", ""))
        score = _row_penetration_score(row)
        if first_infeasible_idx is None and "INFEASIBLE" in status:
            first_infeasible_idx = idx
        if first_penetration_idx is None and score is not None and score <= penetration_threshold:
            first_penetration_idx = idx
        if first_infeasible_idx is not None and first_penetration_idx is not None:
            break

    if first_infeasible_idx is None and first_penetration_idx is None:
        print(f"\n--- No incidents found ({label}) ---")
        print(
            f"No status=INFEASIBLE and no penetration <= {penetration_threshold} found in trace."
        )
        return

    def _print_window(center_idx: int, title: str) -> None:
        start = max(0, center_idx - incident_window)
        end = min(len(rows), center_idx + incident_window + 1)
        print(f"\n--- {title} window ({label}) rows {start}..{end - 1} ---")
        for i in range(start, end):
            _print_row_brief(i, rows[i])

    if first_infeasible_idx is not None:
        _print_window(first_infeasible_idx, "First INFEASIBLE")
    if first_penetration_idx is not None:
        _print_window(
            first_penetration_idx,
            f"First penetration <= {penetration_threshold}",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="Path to primary debug JSONL trace")
    parser.add_argument("--compare", type=Path, default=None, help="Optional second trace to compare")
    parser.add_argument("--top", type=int, default=12, help="Number of worst rows to print")
    parser.add_argument(
        "--incident-window",
        type=int,
        default=15,
        help="Rows before/after first incident to print",
    )
    parser.add_argument(
        "--penetration-threshold",
        type=float,
        default=-0.02,
        help="Threshold for deep penetration incident detection",
    )
    args = parser.parse_args()

    rows = _read_rows(args.trace)
    summary = _summarize(rows, label=str(args.trace))
    _print_summary(summary)
    _print_worst(rows, label=str(args.trace), top_n=max(1, args.top))
    _print_incident_windows(
        rows,
        label=str(args.trace),
        incident_window=max(0, args.incident_window),
        penetration_threshold=float(args.penetration_threshold),
    )

    if args.compare is not None:
        rows_b = _read_rows(args.compare)
        summary_b = _summarize(rows_b, label=str(args.compare))
        _print_summary(summary_b)
        _print_worst(rows_b, label=str(args.compare), top_n=max(1, args.top))
        _print_incident_windows(
            rows_b,
            label=str(args.compare),
            incident_window=max(0, args.incident_window),
            penetration_threshold=float(args.penetration_threshold),
        )

        print("\n=== Delta (compare - primary) ===")
        print(f"rows: {summary_b.rows - summary.rows}")
        print(f"infeasible rows: {summary_b.infeasible_rows - summary.infeasible_rows}")
        print(f"not-applied rows: {summary_b.not_applied_rows - summary.not_applied_rows}")


if __name__ == "__main__":
    main()

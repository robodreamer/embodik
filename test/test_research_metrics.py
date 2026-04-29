from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PYTHON_ROOT = Path(__file__).resolve().parents[1] / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from embodik.research_metrics import (
    FluidityStepSample,
    joint_velocity_norm,
    make_sample_from_result,
    summarize_fluidity,
    write_fluidity_report,
    write_samples_jsonl,
)


def test_summarize_fluidity_reports_streaks_and_smoothness() -> None:
    samples = [
        FluidityStepSample(
            step=0,
            status_name="SUCCESS",
            accepted=True,
            dq_norm=0.1,
            solve_ms=1.0,
            task_scale=1.0,
            target_error=0.05,
            min_collision_distance=0.04,
            joint_velocities=(0.1, 0.0),
        ),
        FluidityStepSample(
            step=1,
            status_name="INFEASIBLE",
            accepted=False,
            dq_norm=0.0,
            solve_ms=3.0,
            task_scale=0.0,
            target_error=0.04,
            min_collision_distance=0.03,
            saturated_joints=("joint_1",),
            joint_velocities=(0.0, 0.0),
        ),
        FluidityStepSample(
            step=2,
            status_name="INFEASIBLE",
            accepted=False,
            dq_norm=0.0,
            solve_ms=4.0,
            task_scale=0.0,
            target_error=0.03,
            min_collision_distance=0.02,
            saturated_joints=("joint_1",),
            joint_velocities=(-0.1, 0.0),
        ),
    ]

    summary = summarize_fluidity(
        "unit",
        samples,
        unrecoverable_stall_steps=2,
        deadline_ms=2.0,
    )

    assert summary.status_counts == {"INFEASIBLE": 2, "SUCCESS": 1}
    assert summary.accepted_count == 1
    assert summary.rejected_count == 2
    assert summary.zero_motion_count == 2
    assert summary.max_stall_streak == 2
    assert summary.unrecoverable_stall_count == 1
    assert summary.saturated_joint_histogram == {"joint_1": 2}
    assert summary.task_scale_min == 0.0
    assert summary.solve_ms_p50 == 3.0
    assert summary.deadline_miss_count == 2
    assert summary.dq_sign_flip_count == 1
    assert summary.jerk_l2_max is not None
    assert summary.min_collision_distance == 0.02


def test_sign_flip_metric_ignores_deadband_chatter() -> None:
    samples = [
        FluidityStepSample(
            step=0,
            status_name="SUCCESS",
            accepted=True,
            dq_norm=0.1,
            solve_ms=1.0,
            joint_velocities=(0.1, 1e-7),
        ),
        FluidityStepSample(
            step=1,
            status_name="SUCCESS",
            accepted=True,
            dq_norm=0.1,
            solve_ms=1.0,
            joint_velocities=(0.1, -1e-7),
        ),
        FluidityStepSample(
            step=2,
            status_name="SUCCESS",
            accepted=True,
            dq_norm=0.1,
            solve_ms=1.0,
            joint_velocities=(-0.1, 1e-7),
        ),
    ]

    summary = summarize_fluidity("unit", samples)

    assert summary.dq_sign_flip_count == 1


def test_make_sample_from_result_extracts_solver_fields() -> None:
    result = SimpleNamespace(
        status=SimpleNamespace(name="SUCCESS"),
        joint_velocities=np.array([3.0, 4.0], dtype=float),
        task_scales=[0.25, 1.0],
        saturated_joints=["joint_2"],
        collision_rejection_count=1,
        stall_escape_count=2,
    )

    sample = make_sample_from_result(
        step=3,
        result=result,
        accepted=True,
        solve_ms=1.25,
        target_error=0.02,
        min_collision_distance=0.01,
    )

    assert sample.status_name == "SUCCESS"
    assert sample.dq_norm == 5.0
    assert sample.task_scale == 0.25
    assert sample.saturated_joints == ("joint_2",)
    assert sample.joint_velocities == (3.0, 4.0)
    assert sample.collision_rejection_count == 1
    assert sample.stall_escape_count == 2
    assert joint_velocity_norm(result) == 5.0


def test_metric_writers_emit_json(tmp_path: Path) -> None:
    sample = FluidityStepSample(
        step=0,
        status_name="SUCCESS",
        accepted=True,
        dq_norm=0.0,
        solve_ms=1.0,
    )
    report_path = tmp_path / "report.json"
    jsonl_path = tmp_path / "samples.jsonl"

    write_fluidity_report(report_path, {"summary": summarize_fluidity("unit", [sample]).as_dict()})
    write_samples_jsonl(jsonl_path, [sample])

    assert '"scenario": "unit"' in report_path.read_text(encoding="utf-8")
    assert '"status_name": "SUCCESS"' in jsonl_path.read_text(encoding="utf-8")

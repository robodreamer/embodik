from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import numpy.testing as npt

from examples.example_helpers.centroidal_support import (
    LINEAR_XY_MOMENTUM_MASK,
    configure_capture_point_constraint,
    configure_centroidal_diagnostic_solver,
    configure_horizontal_momentum_damping,
    configure_velocity_zmp_constraint,
    evaluate_centroidal_diagnostics,
)


class _FakeTask:
    def __init__(self) -> None:
        self.priority = -1
        self.weight = -1.0
        self.active = False
        self.target = None
        self.axis_mask = None

    def set_target_momentum(self, target: np.ndarray) -> None:
        self.target = np.asarray(target, dtype=float)

    def set_axis_mask(self, axis_mask: np.ndarray) -> None:
        self.axis_mask = np.asarray(axis_mask, dtype=float)


class _FakeSolver:
    def __init__(self) -> None:
        self.dt = 0.0
        self.calls: list[tuple[str, tuple, dict]] = []

    def configure_capture_point_constraint(self, *args, **kwargs) -> None:
        self.calls.append(("configure_cp", args, kwargs))

    def clear_capture_point_constraint(self) -> None:
        self.calls.append(("clear_cp", (), {}))

    def configure_velocity_zmp_constraint(self, *args, **kwargs) -> None:
        self.calls.append(("configure_zmp", args, kwargs))

    def clear_velocity_zmp_constraint(self) -> None:
        self.calls.append(("clear_zmp", (), {}))

    def evaluate_capture_point_constraint(self, q, dq_command):
        del q, dq_command
        return {
            "status": SimpleNamespace(name="SUCCESS"),
            "point": np.array([0.1, -0.2]),
            "slacks": np.array([0.03, 0.07]),
            "force_z": 0.0,
        }

    def evaluate_velocity_zmp_constraint(self, q, current_dq, dq_command):
        del q, current_dq, dq_command
        return {
            "status": SimpleNamespace(name="SUCCESS"),
            "point": np.array([0.04, -0.05]),
            "slacks": np.array([0.02, 0.06]),
            "force_z": 123.0,
        }


def test_horizontal_momentum_damping_is_opt_in_and_xy_only() -> None:
    task = _FakeTask()

    configure_horizontal_momentum_damping(task, enabled=True, weight=0.025, priority=2)

    assert task.active is True
    assert task.priority == 2
    assert task.weight == 0.025
    npt.assert_allclose(task.target, np.zeros(6))
    npt.assert_allclose(task.axis_mask, LINEAR_XY_MOMENTUM_MASK)

    configure_horizontal_momentum_damping(task, enabled=False, weight=0.5, priority=1)
    assert task.active is False
    assert task.weight == 0.0


def test_capture_point_configuration_clears_when_disabled() -> None:
    solver = _FakeSolver()
    polygon = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])

    configure_capture_point_constraint(
        solver,
        enabled=True,
        support_polygon=polygon,
        margin=0.05,
        frame_name="world",
    )
    assert solver.calls[-1][0] == "configure_cp"
    npt.assert_allclose(solver.calls[-1][1][0], polygon)
    assert solver.calls[-1][2] == {"margin": 0.05, "frame_name": "world"}

    configure_capture_point_constraint(
        solver,
        enabled=False,
        support_polygon=polygon,
        margin=0.05,
        frame_name="world",
    )
    assert solver.calls[-1][0] == "clear_cp"


def test_velocity_zmp_configuration_clears_when_disabled() -> None:
    solver = _FakeSolver()
    polygon = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])

    configure_velocity_zmp_constraint(
        solver,
        enabled=True,
        support_polygon=polygon,
        margin=0.04,
        frame_name="world",
        fz_min=3.0,
    )
    assert solver.calls[-1][0] == "configure_zmp"
    npt.assert_allclose(solver.calls[-1][1][0], polygon)
    assert solver.calls[-1][2] == {
        "margin": 0.04,
        "frame_name": "world",
        "fz_min": 3.0,
    }

    configure_velocity_zmp_constraint(
        solver,
        enabled=False,
        support_polygon=polygon,
        margin=0.04,
        frame_name="world",
    )
    assert solver.calls[-1][0] == "clear_zmp"


def test_diagnostic_solver_uses_explicit_velocity_state() -> None:
    solver = _FakeSolver()
    polygon = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])

    configure_centroidal_diagnostic_solver(
        solver,
        support_polygon=polygon,
        margin=0.03,
        frame_name="world",
        dt=0.01,
        fz_min=2.0,
    )
    assert solver.dt == 0.01
    assert [call[0] for call in solver.calls] == ["configure_cp", "configure_zmp"]

    q = np.array([0.2, -0.1])
    current_dq = np.array([0.1, -0.05])
    dq_command = np.array([0.2, 0.02])
    diagnostics = evaluate_centroidal_diagnostics(solver, q, current_dq, dq_command)

    npt.assert_allclose(diagnostics.capture_point, [0.1, -0.2])
    npt.assert_allclose(diagnostics.zmp, [0.04, -0.05])
    assert diagnostics.capture_point_min_slack == 0.03
    assert diagnostics.zmp_min_slack == 0.02
    assert diagnostics.force_z == 123.0

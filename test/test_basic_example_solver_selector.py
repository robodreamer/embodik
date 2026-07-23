from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

import embodik as eik

_TWO_LINK_URDF = """<?xml version="1.0"?>
<robot name="embodik_basic_example_selector_test_arm">
  <link name="base_link"/>
  <link name="link1">
    <inertial>
      <origin xyz="0.5 0 0"/>
      <mass value="2.0"/>
      <inertia ixx="0.02" ixy="0" ixz="0" iyy="0.20" iyz="0" izz="0.20"/>
    </inertial>
  </link>
  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.0" upper="2.0" velocity="10.0" effort="100.0"/>
  </joint>
  <link name="link2">
    <inertial>
      <origin xyz="0.5 0 0"/>
      <mass value="1.5"/>
      <inertia ixx="0.015" ixy="0" ixz="0" iyy="0.15" iyz="0" izz="0.15"/>
    </inertial>
  </link>
  <joint name="joint2" type="revolute">
    <parent link="link1"/>
    <child link="link2"/>
    <origin xyz="1 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.0" upper="2.0" velocity="10.0" effort="100.0"/>
  </joint>
  <link name="tip"/>
  <joint name="tip_fixed" type="fixed">
    <parent link="link2"/>
    <child link="tip"/>
    <origin xyz="1 0 0"/>
  </joint>
</robot>
"""


_COLLISION_URDF = """<?xml version="1.0"?>
<robot name="embodik_collision_example_selector_test_arm">
  <link name="base_link">
    <collision name="fixed_sphere">
      <origin xyz="1.4 0 0"/>
      <geometry><sphere radius="0.1"/></geometry>
    </collision>
  </link>
  <link name="moving_link">
    <collision name="moving_sphere">
      <origin xyz="1 0 0"/>
      <geometry><sphere radius="0.1"/></geometry>
    </collision>
    <inertial>
      <origin xyz="0.5 0 0"/>
      <mass value="1"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>
  <joint name="moving_joint" type="revolute">
    <parent link="base_link"/>
    <child link="moving_link"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3" upper="3" velocity="10" effort="100"/>
  </joint>
</robot>
"""


def _load_example(name: str, path: str):
    examples_dir = Path(__file__).resolve().parents[1] / "examples"
    if str(examples_dir) not in sys.path:
        sys.path.insert(0, str(examples_dir))
    spec = importlib.util.spec_from_file_location(name, examples_dir / path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_basic_example_module():
    return _load_example("basic_ik_example_for_selector_test", "01_basic_ik_simple.py")


def _load_collision_example_module():
    return _load_example("collision_ik_example_for_selector_test", "02_collision_aware_IK.py")


def _two_link_robot(tmp_path: Path):
    urdf_path = tmp_path / "two_link.urdf"
    urdf_path.write_text(_TWO_LINK_URDF)
    return eik.RobotModel(str(urdf_path), floating_base=False)


def _collision_cfg(module, tmp_path: Path):
    urdf_path = tmp_path / "collision.urdf"
    urdf_path.write_text(_COLLISION_URDF)
    return module.RobotConfig(
        key="collision",
        display_name="Collision",
        urdf_path=urdf_path,
        description_name="",
        target_link="moving_link",
        joint_labels=["moving_joint"],
        joint_names=["moving_joint"],
        default_configuration=np.zeros(1),
        default_offset=np.zeros(3),
        collision_exclusions=[],
    )


def _skip_without_acceleration_api(module) -> None:
    reason = module.acceleration_api_unavailable_reason()
    if reason:
        pytest.skip(reason)


def test_basic_example_reports_velocity_only_when_acceleration_api_missing() -> None:
    module = _load_basic_example_module()

    assert module.BASIC_SOLVER_LEVEL_OPTIONS == ("Velocity", "Acceleration")
    reason = module.acceleration_api_unavailable_reason()
    if reason:
        assert "missing Python acceleration API" in reason


def test_basic_example_preserves_acceleration_constructor_failure_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_basic_example_module()
    robot = _two_link_robot(tmp_path)
    q = np.zeros(robot.nq)

    class RejectingAccelerationSolver:
        def __init__(self, _robot):
            raise RuntimeError("unsupported test topology")

    monkeypatch.setattr(module.embodik, "AccelerationSolver", RejectingAccelerationSolver)
    runtime, reason = module.basic_acceleration_runtime_status(robot, "tip", q)

    assert runtime is None
    assert reason == "RuntimeError: unsupported test topology"


def test_basic_example_acceleration_stationary_step_runs_headlessly(tmp_path: Path) -> None:
    module = _load_basic_example_module()
    _skip_without_acceleration_api(module)
    robot = _two_link_robot(tmp_path)
    q = np.zeros(robot.nq)
    robot.update_configuration(q)
    target_pose = robot.get_frame_pose("tip")

    runtime = module.configure_basic_acceleration_runtime(robot, "tip", q)
    assert runtime is not None
    step = module.solve_basic_acceleration_step(
        runtime,
        robot,
        q,
        target_pose,
        q,
        pos_gain=20.0,
        rot_gain=20.0,
        posture_gain=0.1,
        dt=0.01,
    )

    assert step.status is eik.SolverStatus.SUCCESS
    np.testing.assert_allclose(step.q_solution, q, atol=1e-12)
    np.testing.assert_allclose(step.dq_solution, np.zeros(robot.nv), atol=1e-12)
    assert step.position_error <= 1e-12
    assert step.rotation_error <= 1e-12


def test_basic_example_reset_and_switch_semantics_zero_dq(tmp_path: Path) -> None:
    module = _load_basic_example_module()
    _skip_without_acceleration_api(module)
    robot = _two_link_robot(tmp_path)
    q = np.zeros(robot.nq)
    runtime = module.configure_basic_acceleration_runtime(robot, "tip", q)
    assert runtime is not None

    runtime.dq[:] = 1.0
    module.reset_basic_acceleration_state(runtime, robot.nv)
    np.testing.assert_array_equal(runtime.dq, np.zeros(robot.nv))


class _FakeAccelerationReference:
    desired_velocity: np.ndarray
    desired_acceleration: np.ndarray
    proportional_gain: float
    derivative_gain: float


class _FakeAccelerationSolveOptions:
    pass


class _FakeTask:
    def set_target_position(self, _value):
        pass

    def set_target_orientation(self, _value):
        pass

    def set_target_configuration(self, _value):
        pass

    def set_controlled_joint_indices(self, _value):
        pass


class _FakeAccelerationSolver:
    def __init__(self, robot, status=eik.SolverStatus.SUCCESS):
        self.robot = robot
        self.status = status
        self.velocity_collision_solver = None
        self.velocity_collision_options = None
        self.last_options = None

    def set_task_reference(self, _name, _reference):
        pass

    def solve(self, q, _dq, _dt, options):
        self.last_options = options
        return self._result(q, options, lifted=False, validation_substeps=0)

    def solve_with_velocity_collision(self, velocity_solver, q, _dq, _dt, options, lift_options):
        self.velocity_collision_solver = velocity_solver
        self.velocity_collision_options = lift_options
        self.last_options = options
        return self._result(
            q,
            options,
            lifted=True,
            validation_substeps=getattr(lift_options, "validation_substeps", 0),
        )

    def _result(self, q, options, lifted, validation_substeps):
        step = 0.01 if self.status is eik.SolverStatus.SUCCESS else 1.0
        q_solution = np.asarray(q, dtype=float) + step
        return type(
            "FakeAccelerationResult",
            (),
            {
                "status": self.status,
                "q_solution": q_solution,
                "joint_velocities_next": np.ones(self.robot.nv),
                "task_diagnostics": [],
                "collision_transaction_time_ms": 0.1 if lifted else 0.0,
                "collision_primitive_distance_queries": 3 if lifted else 0,
                "condition_number": 1.0,
                "velocity_collision_lift_applied": lifted,
                "collision_step_certified": False,
                "collision_validation_samples": validation_substeps if lifted else 0,
                "collision_validation_allowed_pairs": 2 if lifted else 0,
                "collision_validation_pairs_checked": 2 if lifted else 0,
                "collision_validation_exact_queries": 4 if lifted else 0,
                "acceleration_limits_override": getattr(
                    options, "acceleration_limits_override", None
                ),
            },
        )()


def _install_fake_acceleration_backend(
    monkeypatch, module, backend, status=eik.SolverStatus.SUCCESS
):
    monkeypatch.setattr(
        module.embodik,
        "AccelerationTaskReference",
        _FakeAccelerationReference,
        raising=False,
    )
    monkeypatch.setattr(
        module.embodik,
        "AccelerationSolveOptions",
        _FakeAccelerationSolveOptions,
        raising=False,
    )
    backend.acceleration_solver = _FakeAccelerationSolver(backend.robot, status=status)
    backend.acceleration_position_task = _FakeTask()
    backend.acceleration_orientation_task = _FakeTask()
    backend.acceleration_nullspace_task = _FakeTask()
    backend._velocity_collision_lift_options = type("LiftOptions", (), {"validation_substeps": 2})()
    backend._acceleration_unavailable_reason = ""
    backend._acceleration_collision_enabled = True
    backend._acceleration_collision_reason = ""
    backend._acceleration_dq = np.zeros(backend.robot.nv, dtype=float)
    return backend.acceleration_solver


def test_collision_example_acceleration_unavailable_reason(tmp_path: Path) -> None:
    module = _load_collision_example_module()
    cfg = _collision_cfg(module, tmp_path)
    backend = module.embodiKBackend(cfg)

    if module.acceleration_api_unavailable_reason():
        supported, reason = backend.acceleration_collision_status()
        assert not supported
        assert "missing Python acceleration API" in reason


def test_collision_example_acceleration_collision_off_uses_direct_solve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_collision_example_module()
    cfg = _collision_cfg(module, tmp_path)
    backend = module.embodiKBackend(cfg)
    fake_solver = _install_fake_acceleration_backend(monkeypatch, module, backend)
    target_pose = backend.get_pose()

    result = backend.solve_step(
        target_pose,
        pos_gain=20.0,
        rot_gain=20.0,
        active_indices=[0],
        nullspace_bias=np.zeros(1),
        nullspace_gain=0.1,
        nullspace_enabled=True,
        solver_level=module.SOLVER_LEVEL_ACCELERATION,
    )

    assert result.status == eik.SolverStatus.SUCCESS.name
    assert result.solver_level == module.SOLVER_LEVEL_ACCELERATION
    assert not result.velocity_collision_lift_applied
    assert fake_solver.velocity_collision_solver is None
    assert fake_solver.last_options is not None
    np.testing.assert_array_equal(
        fake_solver.last_options.acceleration_limits_override,
        np.full(backend.robot.nv, module.DEFAULT_ACCELERATION_SOLVER_LIMIT),
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"ee_mode": "SCALE_ELASTIC"},
        {"ee_fallback": True},
        {"max_steps": 2},
        {"adaptive_dt": True},
        {"acceleration_limits_enabled": False},
    ],
)
def test_collision_example_acceleration_rejects_ignored_velocity_controls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kwargs: dict[str, object],
) -> None:
    module = _load_collision_example_module()
    cfg = _collision_cfg(module, tmp_path)
    backend = module.embodiKBackend(cfg)
    _install_fake_acceleration_backend(monkeypatch, module, backend)
    q_before = backend.q.copy()

    result = backend.solve_step(
        backend.get_pose(),
        pos_gain=20.0,
        rot_gain=20.0,
        active_indices=[0],
        nullspace_bias=np.zeros(1),
        nullspace_gain=0.1,
        nullspace_enabled=True,
        solver_level=module.SOLVER_LEVEL_ACCELERATION,
        **kwargs,
    )

    assert result.status == "UNSUPPORTED_CONSTRAINT"
    np.testing.assert_array_equal(backend.q, q_before)
    np.testing.assert_array_equal(backend._acceleration_dq, np.zeros(backend.robot.nv))


def test_collision_example_acceleration_collision_on_uses_velocity_lift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_collision_example_module()
    cfg = _collision_cfg(module, tmp_path)
    backend = module.embodiKBackend(cfg)
    fake_solver = _install_fake_acceleration_backend(monkeypatch, module, backend)
    backend._collision_enabled = True
    target_pose = backend.get_pose()

    result = backend.solve_step(
        target_pose,
        pos_gain=20.0,
        rot_gain=20.0,
        active_indices=[0],
        nullspace_bias=np.zeros(1),
        nullspace_gain=0.1,
        nullspace_enabled=True,
        solver_level=module.SOLVER_LEVEL_ACCELERATION,
    )

    assert fake_solver.velocity_collision_solver is backend.solver
    assert fake_solver.velocity_collision_options is not None
    assert fake_solver.velocity_collision_options.validation_substeps == 2
    assert result.velocity_collision_lift_applied
    assert result.collision_validation_samples == 2
    assert not result.collision_step_certified
    assert result.velocity_collision_validation_allowed_pairs == 2
    assert result.velocity_collision_validation_pairs_checked == 2


def test_collision_example_acceleration_failure_holds_q_and_zeros_dq(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_collision_example_module()
    cfg = _collision_cfg(module, tmp_path)
    backend = module.embodiKBackend(cfg)
    _install_fake_acceleration_backend(
        monkeypatch, module, backend, status=eik.SolverStatus.NO_PROGRESS
    )
    backend._acceleration_dq[:] = 1.0
    q_before = backend.q.copy()

    result = backend.solve_step(
        backend.get_pose(),
        pos_gain=20.0,
        rot_gain=20.0,
        active_indices=[0],
        nullspace_bias=np.zeros(1),
        nullspace_gain=0.1,
        nullspace_enabled=True,
        solver_level=module.SOLVER_LEVEL_ACCELERATION,
    )

    assert result.status == eik.SolverStatus.NO_PROGRESS.name
    np.testing.assert_array_equal(backend.q, q_before)
    np.testing.assert_array_equal(backend._acceleration_dq, np.zeros(backend.robot.nv))

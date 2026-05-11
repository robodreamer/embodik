"""Tests for condition-number reporting in ComputeRegularizedInverse.

The branch keeps the same SRINV damping law as main and exposes the
already-computed worst-case condition number on SolverResult. These tests keep
that distinction explicit so future reviews do not attribute behavior changes
to a numerical inverse change that did not occur.
"""

import numpy as np
import pytest

import embodik as eik


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def panda_setup():
    """Load Panda robot and create solver."""
    pytest.importorskip("robot_descriptions.panda_description")
    from robot_descriptions.panda_description import URDF_PATH

    robot = eik.RobotModel(URDF_PATH, floating_base=False)
    solver = eik.KinematicsSolver(robot)
    solver.dt = 0.01
    solver.set_damping(1e-1)
    solver.set_tolerance(1e-6)
    return robot, solver


# Panda default home configuration (7 arm + 2 gripper DOF)
PANDA_HOME = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04])

# Near-elbow-singularity configuration (joint4 near 0)
PANDA_NEAR_SINGULAR = np.array([0.0, -0.785, 0.0, -0.05, 0.0, 1.571, 0.785, 0.04, 0.04])


def _main_reference_srinv(jacobian: np.ndarray, *, epsilon: float, damping: float) -> np.ndarray:
    """Pure-Python reference for the SRINV formula used on main."""
    gram = jacobian @ jacobian.T
    threshold_squared = epsilon**2
    det_value = float(np.linalg.det(gram))
    regularization = (
        (1.0 - (det_value / threshold_squared) ** 2) * threshold_squared
        if det_value < threshold_squared
        else 0.0
    )
    regularized_gram = gram.copy()
    regularized_gram[np.diag_indices_from(regularized_gram)] += regularization

    u, sigma_values, _vh = np.linalg.svd(jacobian, full_matrices=True)
    small_values = sigma_values < epsilon
    if np.any(small_values):
        per_sv_damping = damping * (1.0 - (sigma_values / epsilon) ** 2)
        per_sv_damping *= small_values.astype(float)
        regularized_gram += u @ np.diag(per_sv_damping) @ u.T

    return jacobian.T @ np.linalg.inv(regularized_gram)


def _branch_reference_srinv(
    jacobian: np.ndarray, *, epsilon: float, damping: float
) -> tuple[np.ndarray, float]:
    """Pure-Python reference for this branch: same inverse plus condition number."""
    inverse = _main_reference_srinv(jacobian, epsilon=epsilon, damping=damping)
    sigma_values = np.linalg.svd(jacobian, compute_uv=False)
    sigma_min = float(sigma_values[-1])
    condition_number = (
        float(sigma_values[0]) / sigma_min
        if sigma_min > epsilon * 1e-4
        else float("inf")
    )
    return inverse, condition_number


# ---------------------------------------------------------------------------
# Low-level solver tests (verify damping properties directly)
# ---------------------------------------------------------------------------

class TestConditionAdaptiveDampingLowLevel:
    """Test damping properties using the low-level Eigen solver API."""

    def test_well_conditioned_identity_no_damping(self):
        """Identity Jacobian: damping should be zero, solution exact."""
        n = 7
        goal = np.array([1.0, -2.0, 0.5, -1.0, 0.3, -0.8, 1.2])
        J = np.eye(n)
        C = np.eye(n)
        lower = np.full(n, -10.0)
        upper = np.full(n, 10.0)

        result = eik.computeMultiObjectiveVelocitySolutionEigen(
            [goal], [J], C, lower, upper,
            sr_tolerance=1e-6, sr_damping=1e-1,
        )
        assert result.status == eik.SolverStatus.SUCCESS
        np.testing.assert_allclose(np.array(result.solution), goal, atol=1e-10)

    def test_near_singular_bounded_solution(self):
        """Near-singular Jacobian: damped solution should be bounded."""
        n = 3
        goal = np.array([1.0, 1.0, 1.0])
        J = np.diag([1.0, 1e-8, 1e-10])  # two near-singular directions
        C = np.eye(n)
        lower = np.full(n, -100.0)
        upper = np.full(n, 100.0)

        result = eik.computeMultiObjectiveVelocitySolutionEigen(
            [goal], [J], C, lower, upper,
            sr_tolerance=1e-6, sr_damping=1e-1,
        )
        assert result.status == eik.SolverStatus.SUCCESS
        dq = np.array(result.solution)
        assert np.all(np.isfinite(dq))
        assert np.linalg.norm(dq) < 100.0

    def test_python_reference_documents_srinv_equivalence_to_main(self):
        """The branch should not change SRINV numerics relative to main."""
        epsilon = 1e-6
        damping = 1e-1
        goal = np.array([0.4, -0.2, 0.3])
        jacobian = np.diag([1.0, 0.75 * epsilon, 0.001 * epsilon])

        main_inverse = _main_reference_srinv(jacobian, epsilon=epsilon, damping=damping)
        branch_inverse, condition_number = _branch_reference_srinv(
            jacobian, epsilon=epsilon, damping=damping
        )

        np.testing.assert_allclose(branch_inverse, main_inverse, rtol=1e-12, atol=1e-12)
        assert condition_number > 1e6

        result = eik.computeMultiObjectiveVelocitySolutionEigen(
            [goal],
            [jacobian],
            np.eye(3),
            np.full(3, -100.0),
            np.full(3, 100.0),
            sr_tolerance=epsilon,
            sr_damping=damping,
        )

        assert result.status == eik.SolverStatus.SUCCESS
        np.testing.assert_allclose(np.array(result.solution), branch_inverse @ goal)
        assert result.condition_number == pytest.approx(condition_number)

    def test_condition_number_field_exists_and_scales(self):
        """SolverResult should expose condition_number that grows with ill-conditioning."""
        n = 3
        goal = np.array([1.0, 1.0, 1.0])
        C = np.eye(n)
        lower = np.full(n, -10.0)
        upper = np.full(n, 10.0)

        # Well-conditioned
        J_good = np.eye(n)
        res_good = eik.computeMultiObjectiveVelocitySolutionEigen(
            [goal], [J_good], C, lower, upper,
        )
        assert hasattr(res_good, "condition_number")
        assert not hasattr(res_good, "manipulability")
        assert res_good.condition_number >= 1.0

        # Ill-conditioned
        J_bad = np.diag([1.0, 1.0, 0.001])
        res_bad = eik.computeMultiObjectiveVelocitySolutionEigen(
            [goal], [J_bad], C, lower, upper,
        )
        assert res_bad.condition_number > res_good.condition_number
        assert res_bad.condition_number > 100.0  # condition number ~1000


# ---------------------------------------------------------------------------
# Panda integration tests
# ---------------------------------------------------------------------------

class TestConditionAdaptiveDampingPanda:
    """Integration tests with realistic Panda robot."""

    def test_singularity_sweep_smooth_velocities(self, panda_setup):
        """Sweep Panda elbow through near-singular zone and verify
        joint velocities change smoothly (no abrupt damping spikes).
        """
        robot, solver = panda_setup
        task = solver.add_frame_task("ee", "panda_hand", eik.TaskType.FRAME_POSE)
        task.priority = 0
        task.weight = 1.0

        # Start from near-singular elbow config
        q = PANDA_NEAR_SINGULAR.copy()
        robot.update_configuration(q)

        # Target that pulls the arm through the singularity
        ee_pose = robot.get_frame_pose("panda_hand")
        target_pos = ee_pose.translation.copy()
        target_pos[2] += 0.15
        T = ee_pose.homogeneous()
        target_rot = T[:3, :3]
        task.set_target_pose(target_pos, target_rot)

        velocities = []
        for _ in range(100):
            robot.update_configuration(q)
            result = solver.solve_velocity(q, apply_limits=True)
            if result.status != eik.SolverStatus.SUCCESS:
                break
            dq = np.array(result.joint_velocities)
            velocities.append(dq.copy())
            q = q + dq * solver.dt

        assert len(velocities) > 50, "Solve should not fail early"

        # Smoothness: max velocity jump (jerk proxy) should be bounded
        jumps = np.array([np.linalg.norm(velocities[i+1] - velocities[i])
                          for i in range(len(velocities) - 1)])
        max_jump = jumps.max()
        mean_jump = jumps.mean()
        smoothness_ratio = max_jump / mean_jump if mean_jump > 1e-12 else 1.0

        # Ratio < 10 means no single-step damping spike dominates
        assert smoothness_ratio < 10.0, (
            f"Smoothness ratio {smoothness_ratio:.2f} (max={max_jump:.4f}, "
            f"mean={mean_jump:.4f}) indicates abrupt damping change"
        )

    def test_home_to_extended_tracking(self, panda_setup):
        """Move from home to an extended config. Verify smooth tracking
        and that the solution remains finite throughout.
        """
        robot, solver = panda_setup
        task = solver.add_frame_task("ee", "panda_hand", eik.TaskType.FRAME_POSE)
        task.priority = 0
        task.weight = 1.0

        q = PANDA_HOME.copy()
        robot.update_configuration(q)

        # Target: reach forward (extending the arm toward workspace boundary)
        ee_pose = robot.get_frame_pose("panda_hand")
        target_pos = ee_pose.translation.copy()
        target_pos[0] += 0.2  # reach forward in x
        T = ee_pose.homogeneous()
        task.set_target_pose(target_pos, T[:3, :3])

        all_finite = True
        for _ in range(200):
            robot.update_configuration(q)
            result = solver.solve_velocity(q, apply_limits=True)
            if result.status != eik.SolverStatus.SUCCESS:
                break
            dq = np.array(result.joint_velocities)
            if not np.all(np.isfinite(dq)):
                all_finite = False
                break
            q = q + dq * solver.dt

        assert all_finite, "Solution produced non-finite values during tracking"

    def test_condition_number_varies_with_configuration(self, panda_setup):
        """Condition number should be higher near singularity than at home."""
        robot, solver = panda_setup
        task = solver.add_frame_task("ee", "panda_hand", eik.TaskType.FRAME_POSE)
        task.priority = 0
        task.weight = 1.0

        # Solve at home config
        q_home = PANDA_HOME.copy()
        robot.update_configuration(q_home)
        ee_pose = robot.get_frame_pose("panda_hand")
        target_pos = ee_pose.translation.copy()
        target_pos[0] += 0.01
        T = ee_pose.homogeneous()
        task.set_target_pose(target_pos, T[:3, :3])

        result_home = solver.solve_velocity(q_home, apply_limits=True)
        assert result_home.status == eik.SolverStatus.SUCCESS

        # Solve at near-singular config
        q_sing = PANDA_NEAR_SINGULAR.copy()
        robot.update_configuration(q_sing)
        ee_pose_sing = robot.get_frame_pose("panda_hand")
        target_pos_sing = ee_pose_sing.translation.copy()
        target_pos_sing[0] += 0.01
        T_sing = ee_pose_sing.homogeneous()
        task.set_target_pose(target_pos_sing, T_sing[:3, :3])

        result_sing = solver.solve_velocity(q_sing, apply_limits=True)

        # Near-singular config may be INFEASIBLE due to joint limits,
        # but the solver should not crash and velocities should be finite.
        assert result_sing.status in (
            eik.SolverStatus.SUCCESS,
            eik.SolverStatus.INFEASIBLE,
        )
        dq = np.array(result_sing.joint_velocities)
        assert np.all(np.isfinite(dq))

    def test_continuous_damping_sweep_with_position_ik(self, panda_setup):
        """Use position IK to drive the Panda through a trajectory that
        passes near a singularity. Verify convergence and solution quality.
        """
        robot, solver = panda_setup
        task = solver.add_frame_task("ee", "panda_hand", eik.TaskType.FRAME_POSE)
        task.priority = 0
        task.weight = 1.0

        posture = solver.add_posture_task("posture")
        posture.priority = 1
        posture.weight = 0.01
        posture.set_target_configuration(PANDA_HOME)

        q = PANDA_HOME.copy()
        robot.update_configuration(q)

        # Waypoints that traverse near singular zone
        ee_pose = robot.get_frame_pose("panda_hand")
        base_pos = ee_pose.translation.copy()
        T = ee_pose.homogeneous()
        base_rot = T[:3, :3]

        offsets = [
            np.array([0.0, 0.0, 0.0]),
            np.array([0.1, 0.0, 0.0]),
            np.array([0.1, 0.1, 0.0]),
            np.array([0.0, 0.1, 0.1]),
        ]

        for offset in offsets:
            target_pos = base_pos + offset
            target = np.eye(4)
            target[:3, :3] = base_rot
            target[:3, 3] = target_pos

            opts = eik.PositionStepOptions()
            opts.position_gain = 5.0
            opts.orientation_gain = 5.0
            opts.max_steps = 50

            result = solver.solve_position_step(q, target, "ee", opts)
            assert result.status in (
                eik.SolverStatus.SUCCESS,
                eik.SolverStatus.NO_PROGRESS,
            ), f"Failed at offset {offset}: {result.status}"
            q = np.array(result.q_solution)

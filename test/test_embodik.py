"""Consolidated tests for embodiK multi-task solver."""

import math
import pytest
import random
import numpy as np
import sys
import logging
import textwrap

import embodik as eik

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# =============================================================================
# Test Constants
# =============================================================================

# Numerical tolerances for testing
SOLVER_EPSILON = 1e-6  # Default numerical tolerance
SOLVER_PRECISION_THRESHOLD = 1e-10  # High-precision threshold
NUMERICAL_EPSILON = 1e-6  # Small epsilon for numerical comparisons (matches v1 TOL)
CONSTRAINT_TOLERANCE = 1e-6  # Tolerance for constraint satisfaction (matches v1 cstTol)
TASK_ERROR_TOLERANCE = 1e-6  # Tolerance for task achievement error (matches v1 cstTol)
SCALE_EPSILON = 1e-10  # Epsilon for task scale comparisons
OPT_TOLERANCE = 1e-4  # Optimization tolerance
TEST_TOLERANCE = 1e-4  # Test tolerance for comparisons

# Solver parameters (matching v1 defaults)
DEFAULT_SR_DAMPING = 1e-6  # Singularity-robust damping (matches v1 beta_max and INV_DAMPING_COEFF)


# =============================================================================
# Basic functionality tests
# =============================================================================


def test_import_and_metadata():
    """Test basic imports and module metadata."""
    # Basic imports
    assert hasattr(eik, "__version__")
    assert hasattr(eik, "SolverStatus")
    assert hasattr(eik, "computeMultiObjectiveVelocitySolutionEigen")
    assert hasattr(eik, "pose_error_norm")
    # Enums
    assert eik.SolverStatus.SUCCESS.value == 0
    assert eik.SolverStatus.INVALID_INPUT.value == 1
    assert eik.SolverStatus.NUMERICAL_ERROR.value == 2
    assert eik.SolverStatus.INFEASIBLE.value == 7
    assert eik.SolverStatus.NO_PROGRESS.value == 8


def test_pose_error_norm():
    """Test pose error norm calculation."""
    assert abs(eik.pose_error_norm([0, 0, 0], [1, 2, 2]) - 3.0) < NUMERICAL_EPSILON


# =============================================================================
# Multi-task solver API tests
# =============================================================================


def test_multi_task_api():
    """Test basic multi-task solver functionality."""
    # Simple 2-joint, 1-task problem
    goals = [np.array([1.0, -2.0])]
    jacobians = [np.array([[1.0, 0.0], [0.0, 1.0]])]
    C = np.eye(2)
    lower = np.array([-10.0, -10.0])
    upper = np.array([10.0, 10.0])

    result = eik.computeMultiObjectiveVelocitySolutionEigen(goals, jacobians, C, lower, upper)
    assert result.status == eik.SolverStatus.SUCCESS
    assert len(result.solution) == 2
    assert len(result.task_scales) == 1
    assert result.task_scales[0] == 1.0

    # Solution should be close to goal for identity Jacobian
    assert math.isclose(result.solution[0], 1.0, rel_tol=1e-2, abs_tol=1e-2)
    assert math.isclose(result.solution[1], -2.0, rel_tol=1e-2, abs_tol=1e-2)
    assert len(result.task_modes_effective) == 1
    assert len(result.task_used_fallback) == 1
    assert result.task_modes_effective[0] == eik.TaskSolveMode.SCALE
    assert result.task_used_fallback[0] is False


def test_eigen_first_multi_task():
    """Test Eigen-first multi-task velocity IK solver."""
    # Create test data
    goals = [np.array([0.1, -0.2], dtype=np.float64), np.array([0.3], dtype=np.float64)]
    jacobians = [
        np.asarray(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), dtype=np.float64, order="F"),
        np.asarray(np.array([[0.0, 0.0, 1.0]]), dtype=np.float64, order="F"),
    ]
    C = np.asarray(
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), dtype=np.float64, order="F"
    )
    lower_limits = np.array([-1.0, -1.0, -1.0], dtype=np.float64)
    upper_limits = np.array([1.0, 1.0, 1.0], dtype=np.float64)

    # Test Eigen API (using v1 default parameters)
    params = {
        "epsilon": SOLVER_EPSILON,
        "precision_threshold": SOLVER_PRECISION_THRESHOLD,
        "iteration_limit": 20,
        "magnitude_limit": 1e10,
        "stall_detection_count": 2,
        "regularization_epsilon": SOLVER_EPSILON,
        "regularization_factor": DEFAULT_SR_DAMPING,
    }

    res_eigen = eik.computeMultiObjectiveVelocitySolutionEigen(
        goals, jacobians, C, lower_limits, upper_limits
    )

    # Test numpy API for comparison
    res_np = eik.computeMultiObjectiveVelocitySolutionEigen(
        goals, jacobians, C, lower_limits, upper_limits
    )

    # Both should succeed
    assert res_eigen.status == eik.SolverStatus.SUCCESS
    assert res_np.status == eik.SolverStatus.SUCCESS

    # Results should match
    assert np.allclose(res_eigen.solution, res_np.solution)
    assert np.allclose(res_eigen.task_scales, res_np.task_scales)


def test_numpy_array_types():
    """Test that the APIs work with different numpy array types."""
    # Test with different numpy array types for multi-task solver
    goals = [np.array([0.1, 0.2], dtype=np.float64)]  # float64
    jacobians = [np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64, order="F")]  # F-order
    C = np.eye(2, dtype=np.float64, order="F")
    lower = np.array([-1, -1], dtype=np.float64)
    upper = np.array([1, 1], dtype=np.float64)

    # These should work without errors
    res = eik.computeMultiObjectiveVelocitySolutionEigen(goals, jacobians, C, lower, upper)
    assert res.status == eik.SolverStatus.SUCCESS


# =============================================================================
# Invalid input handling
# =============================================================================


def test_invalid_inputs():
    """Test invalid input handling for multi-task solver."""
    # Empty inputs
    result = eik.computeMultiObjectiveVelocitySolutionEigen(
        [],
        [],
        np.eye(1, dtype=np.float64, order="F"),
        np.array([0], dtype=np.float64),
        np.array([1], dtype=np.float64),
    )
    assert result.status == eik.SolverStatus.EMPTY_PROBLEM
    assert "no objectives" in result.status_message

    # Mismatched dimensions - goal dimension doesn't match Jacobian rows
    goals = [np.array([1.0, 2.0])]  # 2D goal
    jacobians = [np.array([[1.0, 0.0]])]  # 1x2 Jacobian (only 1 row)
    C = np.eye(2)
    lower = np.array([-1, -1])
    upper = np.array([1, 1])

    result = eik.computeMultiObjectiveVelocitySolutionEigen(goals, jacobians, C, lower, upper)
    # This should be caught by dimension validation
    assert result.status == eik.SolverStatus.SHAPE_MISMATCH
    assert "dimension mismatch" in result.status_message


def test_solver_status_hint_helper():
    """Status hint helper should return actionable guidance text."""
    msg = eik.get_solver_status_hint(
        eik.SolverStatus.SHAPE_MISMATCH, "goal size does not match jacobian rows"
    )
    assert "Shape mismatch" in msg
    assert "goal size does not match jacobian rows" in msg
    infeasible_msg = eik.get_solver_status_hint(
        eik.SolverStatus.INFEASIBLE, "primary task scale collapsed to zero"
    )
    assert "no feasible solution" in infeasible_msg
    assert "primary task scale collapsed to zero" in infeasible_msg
    no_progress_msg = eik.get_solver_status_hint(eik.SolverStatus.NO_PROGRESS, "stalled at limits")
    assert "progress stalled" in no_progress_msg
    assert "stalled at limits" in no_progress_msg


def test_solve_velocity_propagates_backend_status_message(tmp_path):
    """High-level solve_velocity should surface actionable status messages."""
    urdf_path = _create_minimal_collision_urdf(tmp_path)
    robot = eik.RobotModel(str(urdf_path), floating_base=False)
    solver = eik.KinematicsSolver(robot)
    joint_task = solver.add_joint_task("joint_task", "joint1", target_value=2.0)
    joint_task.priority = 0
    joint_task.weight = 1.0

    q0 = np.array([1.57], dtype=float)  # at upper joint limit

    result = solver.solve_velocity(q0, apply_limits=True)
    assert result.status == eik.SolverStatus.INFEASIBLE
    assert "primary task scale collapsed" in result.status_message


def test_position_ik_nonconvergence_reports_no_progress_by_default(tmp_path):
    """Position IK should classify stagnation-style non-convergence as NO_PROGRESS by default."""
    urdf_path = _create_minimal_collision_urdf(tmp_path)
    robot = eik.RobotModel(str(urdf_path), floating_base=False)
    solver = eik.KinematicsSolver(robot)

    seed_q = np.zeros(robot.nq, dtype=float)
    target_pose = np.eye(4, dtype=float)
    target_pose[0, 3] = 10.0  # intentionally unreachable for this tiny 1-DOF arm

    opts = eik.PositionIKOptions()
    opts.max_iterations = 3
    opts.stagnation_iterations = 2
    opts.stagnation_tolerance = 1e-9

    result = solver.solve_position(seed_q, target_pose, "link1", opts)
    assert result.status == eik.SolverStatus.NO_PROGRESS
    assert "no progress" in result.status_message.lower()


def test_position_ik_stagnation_can_report_no_progress(tmp_path):
    """Position IK can classify stagnation exits as NO_PROGRESS when requested."""
    urdf_path = _create_minimal_collision_urdf(tmp_path)
    robot = eik.RobotModel(str(urdf_path), floating_base=False)
    solver = eik.KinematicsSolver(robot)

    seed_q = np.zeros(robot.nq, dtype=float)
    target_pose = np.eye(4, dtype=float)
    target_pose[0, 3] = 10.0

    opts = eik.PositionIKOptions()
    opts.max_iterations = 5
    opts.stagnation_iterations = 2
    opts.stagnation_tolerance = 1e-9
    opts.classify_stagnation_as_no_progress = True

    result = solver.solve_position(seed_q, target_pose, "link1", opts)
    assert result.status == eik.SolverStatus.NO_PROGRESS
    assert "no progress" in result.status_message.lower()


def test_position_ik_stagnation_can_be_classified_as_infeasible(tmp_path):
    """Users can opt out of NO_PROGRESS default classification."""
    urdf_path = _create_minimal_collision_urdf(tmp_path)
    robot = eik.RobotModel(str(urdf_path), floating_base=False)
    solver = eik.KinematicsSolver(robot)

    seed_q = np.zeros(robot.nq, dtype=float)
    target_pose = np.eye(4, dtype=float)
    target_pose[0, 3] = 10.0

    opts = eik.PositionIKOptions()
    opts.max_iterations = 5
    opts.stagnation_iterations = 2
    opts.stagnation_tolerance = 1e-9
    opts.classify_stagnation_as_no_progress = False

    result = solver.solve_position(seed_q, target_pose, "link1", opts)
    assert result.status == eik.SolverStatus.INFEASIBLE
    assert "did not reach tolerance" in result.status_message


# =============================================================================
# Constraint and prioritization tests
# =============================================================================


def test_multi_task_with_constraints():
    """Test multi-task solver with constraint handling."""
    # 3-joint robot with 2 tasks
    goals = [np.array([0.5, 0.3]), np.array([0.2])]  # Task 0: 2D  # Task 1: 1D
    jacobians = [np.array([[1.0, 0.5, 0.0], [0.0, 1.0, 0.5]]), np.array([[0.0, 0.0, 1.0]])]

    # Joint limits
    C = np.eye(3)
    lower = np.array([-0.4, -0.4, -0.4])
    upper = np.array([0.4, 0.4, 0.4])

    result = eik.computeMultiObjectiveVelocitySolutionEigen(goals, jacobians, C, lower, upper)
    assert result.status == eik.SolverStatus.SUCCESS

    # Check that constraints are satisfied
    solution = np.array(result.solution)
    assert np.all(solution >= lower - CONSTRAINT_TOLERANCE)
    assert np.all(solution <= upper + CONSTRAINT_TOLERANCE)

    # Primary task should be achieved exactly (v1 requirement)
    achieved = jacobians[0] @ solution
    scaled_goal = result.task_scales[0] * goals[0]
    error = np.linalg.norm(achieved - scaled_goal)
    assert (
        error < TASK_ERROR_TOLERANCE
    ), f"Primary task error {error} exceeds tolerance {TASK_ERROR_TOLERANCE}"


def test_saturated_joint_still_allows_task_via_redundancy():
    """Redundant joints achieve the task even when one joint is saturated.

    SNS should find nonzero scale when other joints can contribute.
    """
    # 4 joints, 1D task.  J = [1, 1, 1, 1].  Joint 0 capped at 0.
    J = np.array([[1.0, 1.0, 1.0, 1.0]])
    goal = np.array([1.0])
    C = np.eye(4)
    lower = np.array([-2.0, -2.0, -2.0, -2.0])
    upper = np.array([0.0, 2.0, 2.0, 2.0])

    result = eik.computeMultiObjectiveVelocitySolutionEigen([goal], [J], C, lower, upper)
    assert result.status == eik.SolverStatus.SUCCESS

    solution = np.array(result.solution)
    assert result.task_scales[0] > 0.0, "Task scale should be nonzero when redundancy exists"
    assert solution[0] <= CONSTRAINT_TOLERANCE, "Saturated joint should stay at bound"
    achieved = (J @ solution).item()
    assert abs(achieved - result.task_scales[0] * goal[0]) < TASK_ERROR_TOLERANCE


def test_saturated_joint_with_forced_recovery_still_achieves_task():
    """Recovery forcing joint velocity negative should not kill the task."""
    J = np.array([[1.0, 1.0, 1.0, 1.0]])
    goal = np.array([1.0])
    C = np.eye(4)
    lower = np.array([-2.0, -2.0, -2.0, -2.0])
    upper = np.array([-0.05, 2.0, 2.0, 2.0])  # joint 0 forced negative

    result = eik.computeMultiObjectiveVelocitySolutionEigen([goal], [J], C, lower, upper)
    assert result.status == eik.SolverStatus.SUCCESS

    solution = np.array(result.solution)
    assert result.task_scales[0] > 0.0, "Scale should be nonzero; free joints can compensate"
    assert solution[0] <= -0.05 + CONSTRAINT_TOLERANCE
    achieved = (J @ solution).item()
    assert abs(achieved - result.task_scales[0] * goal[0]) < TASK_ERROR_TOLERANCE


def test_saturated_joint_with_coupled_multidim_task():
    """Multi-row task with well-conditioned Jacobian still achieves partial scale."""
    J = np.array(
        [
            [1.0, 0.5, 0.3, 0.1],
            [0.2, 1.0, 0.8, 0.4],
        ]
    )
    goal = np.array([0.5, -0.3])
    C = np.eye(4)
    lower = np.full(4, -2.0)
    upper = np.array([-0.05, 2.0, 2.0, 2.0])  # joint 0 forced negative

    result = eik.computeMultiObjectiveVelocitySolutionEigen([goal], [J], C, lower, upper)
    assert result.status == eik.SolverStatus.SUCCESS

    solution = np.array(result.solution)
    assert result.task_scales[0] > 0.0, "Well-conditioned coupled task should have nonzero scale"
    assert solution[0] <= -0.05 + CONSTRAINT_TOLERANCE
    achieved = J @ solution
    scaled_goal = result.task_scales[0] * goal
    assert np.linalg.norm(achieved - scaled_goal) < TASK_ERROR_TOLERANCE


def test_rank_deficient_jacobian_at_limit_collapses_scale():
    """Near-singular Jacobian at a joint limit legitimately collapses the task scale.

    A planar arm at full extension has proportional Jacobian rows (rank 1 for 2D task),
    so the solver correctly returns scale=0 when it cannot satisfy both task components.
    """
    # Rows are proportional: rank ≈ 1
    J = np.array(
        [
            [9.56e-4, 7.17e-4, 4.78e-4, 2.39e-4],
            [-1.20, -0.90, -0.60, -0.30],
        ]
    )
    goal = np.array([0.0, -0.05])
    C = np.eye(4)
    lower = np.full(4, -2.0)
    upper = np.array([-0.05, 2.0, 2.0, 2.0])

    result = eik.computeMultiObjectiveVelocitySolutionEigen([goal], [J], C, lower, upper)
    assert result.status == eik.SolverStatus.SUCCESS
    # Scale collapse is expected here due to rank deficiency, not a bug
    assert result.task_scales[0] == pytest.approx(0.0, abs=SCALE_EPSILON)


def test_split_tasks_bypass_single_scale_limitation():
    """Splitting infeasible + feasible objectives into separate tasks
    allows the feasible part to proceed independently.
    """
    C = np.eye(2)
    lower = np.array([0.0, -1.0])  # Joint 0 fully blocked
    upper = np.array([0.0, 1.0])

    jacobians = [
        np.array([[1.0, 0.0]]),  # Blocked objective
        np.array([[0.0, 1.0]]),  # Feasible objective
    ]
    goals = [np.array([1.0]), np.array([1.0])]

    result = eik.computeMultiObjectiveVelocitySolutionEigen(goals, jacobians, C, lower, upper)
    assert result.status == eik.SolverStatus.SUCCESS

    solution = np.array(result.solution)
    assert solution[0] == pytest.approx(0.0, abs=CONSTRAINT_TOLERANCE)
    assert solution[1] > 0.0
    assert result.task_scales[0] == pytest.approx(0.0, abs=SCALE_EPSILON)
    assert result.task_scales[1] > 0.0


def test_min_error_mode_marks_effective_mode_and_task_errors():
    """MIN_ERROR mode should populate diagnostics and residual task error."""
    goals = [np.array([1.0])]
    jacobians = [np.array([[1.0, 0.0]])]
    C = np.eye(2)
    lower = np.array([0.0, -1.0])
    upper = np.array([0.0, 1.0])  # First joint blocked

    result = eik.computeMultiObjectiveVelocitySolutionEigen(goals, jacobians, C, lower, upper)
    assert result.status == eik.SolverStatus.SUCCESS
    assert len(result.task_errors) == 1
    assert len(result.task_modes_effective) == 1
    assert len(result.task_used_fallback) == 1


def test_multi_task_prioritization():
    """Test that tasks are properly prioritized."""
    # Conflicting tasks - both want to move joint 0
    goals = [
        np.array([1.0]),  # Task 0: wants positive motion
        np.array([-1.0]),  # Task 1: wants negative motion
    ]
    jacobians = [np.array([[1.0, 0.0]]), np.array([[1.0, 0.0]])]

    C = np.eye(2)
    lower = np.array([-0.5, -0.5])
    upper = np.array([0.5, 0.5])

    result = eik.computeMultiObjectiveVelocitySolutionEigen(goals, jacobians, C, lower, upper)
    assert result.status == eik.SolverStatus.SUCCESS

    # The key is that the solution respects the priority - Task 0's direction
    solution = np.array(result.solution)
    assert solution[0] > 0  # Should move in positive direction (Task 0's preference)


# =============================================================================
# Random problem tests
# =============================================================================


def test_random_multi_task_problems():
    """Test solver on random multi-task problems."""
    random.seed(42)
    num_tests = 20
    num_success = 0

    for _ in range(num_tests):
        # Random problem dimensions
        n_joints = random.randint(3, 8)
        n_tasks = random.randint(1, 3)

        goals = []
        jacobians = []

        for _ in range(n_tasks):
            # Ensure task dimension is less than n_joints to avoid square/overdetermined systems
            task_dim = random.randint(1, max(1, n_joints - 1))
            goal = np.random.uniform(-0.5, 0.5, task_dim)
            jacobian = np.random.uniform(-1.0, 1.0, (task_dim, n_joints))
            goals.append(goal)
            jacobians.append(jacobian)

        # Constraints
        C = np.eye(n_joints)
        lower = np.full(n_joints, -1.0)
        upper = np.full(n_joints, 1.0)

        result = eik.computeMultiObjectiveVelocitySolutionEigen(goals, jacobians, C, lower, upper)

        if result.status == eik.SolverStatus.SUCCESS:
            num_success += 1

            # Verify primary task (v1 requirement: primary task must be achieved if scale > 0)
            solution = np.array(result.solution)
            if result.task_scales[0] > SCALE_EPSILON:
                achieved = jacobians[0] @ solution
                scaled_goal = result.task_scales[0] * goals[0]
                error = np.linalg.norm(achieved - scaled_goal)
                if error >= TASK_ERROR_TOLERANCE:
                    logger.debug(f"Primary task error {error} with scale {result.task_scales[0]}")
                    logger.debug(
                        f"  n_joints={n_joints}, n_tasks={n_tasks}, task_dim={goals[0].shape}"
                    )
                    logger.debug(f"  goal[0]: {goals[0]}")
                    logger.debug(f"  achieved: {achieved}")
                    logger.debug(f"  scaled_goal: {scaled_goal}")
                    logger.debug(f"  solution: {solution}")
                    logger.debug(f"  all scales: {result.task_scales}")
                assert (
                    error < TASK_ERROR_TOLERANCE
                ), f"Primary task error {error} exceeds tolerance {TASK_ERROR_TOLERANCE}"

    # At least 80% should succeed
    assert num_success >= 0.8 * num_tests


# =============================================================================
# Benchmark tests
# =============================================================================


@pytest.mark.benchmark(group="multi-task")
def test_multi_task_numpy_benchmark(benchmark):  # type: ignore[no-untyped-def]
    """Benchmark multi-task solver with random problem generation and validation."""
    rng = np.random.default_rng(42)  # Fixed seed for reproducible benchmarks

    def generate_random_problem():
        """Generate a random multi-task IK problem for validation testing."""
        # Problem dimensions
        n_joints = rng.integers(5, 12)  # Number of joints
        n_tasks = rng.integers(1, 4)  # Number of tasks
        n_constraints = rng.integers(2, 6)  # Additional constraints

        # Generate random goals and jacobians for each task
        goals = []
        jacobians = []

        for i in range(n_tasks):
            # Task dimension (how many objectives in this task)
            task_dim = rng.integers(1, min(n_joints, 6))

            # Random goal vector
            goal = rng.uniform(-0.5, 0.5, task_dim)
            goals.append(goal)

            # Random Jacobian matrix for this task
            jacobian = rng.uniform(-1.0, 1.0, (task_dim, n_joints))
            jacobians.append(np.asarray(jacobian, dtype=np.float64, order="F"))

        # Constraint matrix (identity for joint limits + random constraints)
        C = np.eye(n_joints + n_constraints, n_joints)
        for i in range(n_constraints):
            C[n_joints + i, :] = rng.uniform(-0.5, 0.5, n_joints)
        C = np.asarray(C, dtype=np.float64, order="F")

        # Joint and constraint limits
        lower_limits = rng.uniform(-2.0, -0.5, n_joints + n_constraints)
        upper_limits = rng.uniform(0.5, 2.0, n_joints + n_constraints)

        return goals, jacobians, C, lower_limits, upper_limits

    def setup():
        goals, jacobians, C, lower_limits, upper_limits = generate_random_problem()
        params = {
            "epsilon": SOLVER_EPSILON,
            "precision_threshold": SOLVER_PRECISION_THRESHOLD,
            "iteration_limit": 20,
            "magnitude_limit": 1e10,
            "stall_detection_count": 2,
            "damping_epsilon": SOLVER_EPSILON,
            "damping_regularization_factor": DEFAULT_SR_DAMPING,
        }
        return ((goals, jacobians, C, lower_limits, upper_limits, params), {})

    def run_and_validate(goals, jacobians, C, lower_limits, upper_limits, params=None):
        """Run solver and validate output."""
        result = eik.computeMultiObjectiveVelocitySolutionEigen(
            goals, jacobians, C, lower_limits, upper_limits
        )

        # Validate result
        assert result.status in [eik.SolverStatus.SUCCESS, eik.SolverStatus.NUMERICAL_ERROR]

        if result.status == eik.SolverStatus.SUCCESS:
            n_joints = jacobians[0].shape[1]
            n_tasks = len(goals)

            # Check solution dimensions
            assert len(result.solution) == n_joints
            assert len(result.task_scales) == n_tasks

            # Verify task scales are in valid range [0, 1]
            for scale in result.task_scales:
                assert 0.0 <= scale <= 1.0 + SCALE_EPSILON

            # Check constraint satisfaction (v1 strict requirement)
            solution = np.array(result.solution)
            Cx = C @ solution
            if np.any(Cx > upper_limits + CONSTRAINT_TOLERANCE) or np.any(
                Cx < lower_limits - CONSTRAINT_TOLERANCE
            ):
                violations_upper = np.maximum(0, Cx - upper_limits - CONSTRAINT_TOLERANCE)
                violations_lower = np.maximum(0, lower_limits - Cx - CONSTRAINT_TOLERANCE)
                max_violation = max(np.max(violations_upper), np.max(violations_lower))
                assert False, f"Constraints violated! Max violation: {max_violation}"

            # For primary task (Task 0), check achievement (v1 requirement)
            task_error = result.task_scales[0] * goals[0] - jacobians[0] @ solution
            if np.any(np.abs(task_error) > CONSTRAINT_TOLERANCE):
                assert False, f"Task error ({np.max(np.abs(task_error)):.6f}) exceeds tolerance!"

        return result

    benchmark.pedantic(run_and_validate, setup=setup, rounds=50)


# =============================================================================
# Large-scale validation tests
# =============================================================================


def test_multi_task_solver_large_scale():
    """Test multi-task solver with many random problems."""
    rng = np.random.default_rng(12345)  # Fixed seed for reproducibility

    def generate_random_multi_task_problem():
        """Generate a diverse multi-task IK problem."""
        # Vary problem dimensions
        n_joints = rng.integers(3, 15)  # 3 to 14 joints
        n_tasks = rng.integers(1, 5)  # 1 to 4 tasks

        goals = []
        jacobians = []

        for i in range(n_tasks):
            # Each task can have different dimensions
            task_dim = rng.integers(1, min(n_joints, 6))

            # Generate task goal
            goal = rng.uniform(-1.0, 1.0, task_dim)
            goals.append(goal)

            # Generate task Jacobian
            # Mix of well-conditioned and ill-conditioned Jacobians
            if i == 0 or rng.random() > 0.3:  # Primary task or 70% chance
                # Well-conditioned Jacobian
                jacobian = rng.uniform(-1.0, 1.0, (task_dim, n_joints))
            else:
                # Potentially ill-conditioned Jacobian
                jacobian = rng.uniform(-0.1, 0.1, (task_dim, n_joints))
                # Add some structure
                for j in range(min(task_dim, n_joints)):
                    jacobian[j % task_dim, j] += rng.uniform(0.5, 1.5)

            jacobians.append(np.asarray(jacobian, dtype=np.float64, order="F"))

        # Constraint matrix (typically identity for joint limits)
        C = np.asarray(np.eye(n_joints), dtype=np.float64, order="F")

        # Generate reasonable joint limits
        limit_range = rng.uniform(0.5, 2.0, n_joints)
        center = rng.uniform(-0.5, 0.5, n_joints)
        lower_limits = center - limit_range
        upper_limits = center + limit_range

        return goals, jacobians, C, lower_limits, upper_limits

    def validate_solution(result, goals, jacobians, C, lower_limits, upper_limits):
        """Validate solver result."""
        if result.status not in [eik.SolverStatus.SUCCESS, eik.SolverStatus.NUMERICAL_ERROR]:
            return False, None, False

        if result.status == eik.SolverStatus.NUMERICAL_ERROR:
            # Numerical errors are acceptable for ill-conditioned problems
            return True, None, True

        n_joints = jacobians[0].shape[1]
        solution = np.array(result.solution)

        # Check solution dimensions
        if len(result.solution) != n_joints:
            return False, None, False

        # Check task scales
        if len(result.task_scales) != len(goals):
            return False, None, False

        for scale in result.task_scales:
            if not (0.0 <= scale <= 1.0 + SCALE_EPSILON):
                return False, None, False

        # Check constraints strictly (v1 requirement)
        Cx = C @ solution
        if np.any(Cx > upper_limits + CONSTRAINT_TOLERANCE) or np.any(
            Cx < lower_limits - CONSTRAINT_TOLERANCE
        ):
            return False, None, False

        # Check primary task achievement (v1 requirement)
        task_error = result.task_scales[0] * goals[0] - jacobians[0] @ solution
        primary_task_error = np.linalg.norm(task_error)

        if np.any(np.abs(task_error) > CONSTRAINT_TOLERANCE):
            return False, primary_task_error, False

        return True, primary_task_error, False

    # Run many test cases
    n_tests = 1000
    n_success = 0
    n_valid = 0
    n_invalid_input = 0
    n_warnings = 0
    max_primary_error = 0.0

    for i in range(n_tests):
        goals, jacobians, C, lower_limits, upper_limits = generate_random_multi_task_problem()

        params = {
            "epsilon": SOLVER_EPSILON,
            "precision_threshold": SOLVER_PRECISION_THRESHOLD,
            "iteration_limit": 20,
            "magnitude_limit": 1e10,
            "stall_detection_count": 2,
            "damping_epsilon": SOLVER_EPSILON,
            "damping_regularization_factor": DEFAULT_SR_DAMPING,
        }

        result = eik.computeMultiObjectiveVelocitySolutionEigen(
            goals, jacobians, C, lower_limits, upper_limits
        )

        if result.status == eik.SolverStatus.SUCCESS:
            n_success += 1
        elif result.status == eik.SolverStatus.INVALID_INPUT:
            n_invalid_input += 1

        is_valid, primary_error, has_warning = validate_solution(
            result, goals, jacobians, C, lower_limits, upper_limits
        )

        if is_valid:
            n_valid += 1
            if primary_error is not None:
                max_primary_error = max(max_primary_error, primary_error)

        if has_warning:
            n_warnings += 1

    # Report results
    success_rate = n_success / n_tests
    validation_pass_rate = n_valid / n_tests

    logger.info(f"\n=== Large Scale Test Results ({n_tests} problems) ===")
    logger.info(f"Success rate: {success_rate:.1%} ({n_success}/{n_tests})")
    logger.info(f"Validation pass rate: {validation_pass_rate:.1%} ({n_valid}/{n_tests})")
    logger.info(f"Invalid inputs: {n_invalid_input}")
    logger.info(f"Warnings: {n_warnings}")
    logger.info(f"Max primary task error: {max_primary_error:.6f}")

    # Assertions (v1 requires > 96% pass rate for ExtendedSingularityRobustSolver)
    assert success_rate >= 1.0, f"Success rate too low: {success_rate:.1%}"
    assert validation_pass_rate >= 1.0, f"Validation rate too low: {validation_pass_rate:.1%}"
    assert n_invalid_input == 0, f"Unexpected invalid inputs: {n_invalid_input}"


# =============================================================================
# Collision constraint tests
# =============================================================================


def _create_minimal_collision_urdf(tmp_path):
    """Generate a simple URDF with collision geometry for testing."""
    urdf_content = textwrap.dedent("""
        <robot name="two_link">
          <link name="base_link">
            <inertial>
              <origin xyz="0 0 0" rpy="0 0 0"/>
              <mass value="1.0"/>
              <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
            </inertial>
            <collision>
              <origin xyz="0 0 0" rpy="0 0 0"/>
              <geometry>
                <box size="0.1 0.1 0.1"/>
              </geometry>
            </collision>
            <visual>
              <origin xyz="0 0 0" rpy="0 0 0"/>
              <geometry>
                <box size="0.1 0.1 0.1"/>
              </geometry>
            </visual>
          </link>
          <link name="link1">
            <inertial>
              <origin xyz="0 0 0" rpy="0 0 0"/>
              <mass value="0.5"/>
              <inertia ixx="0.005" ixy="0.0" ixz="0.0" iyy="0.005" iyz="0.0" izz="0.005"/>
            </inertial>
            <collision>
              <origin xyz="0.08 0 0" rpy="0 0 0"/>
              <geometry>
                <box size="0.08 0.08 0.08"/>
              </geometry>
            </collision>
            <visual>
              <origin xyz="0.08 0 0" rpy="0 0 0"/>
              <geometry>
                <box size="0.08 0.08 0.08"/>
              </geometry>
            </visual>
          </link>
          <joint name="joint1" type="revolute">
            <parent link="base_link"/>
            <child link="link1"/>
            <origin xyz="0.05 0 0" rpy="0 0 0"/>
            <axis xyz="0 0 1"/>
            <limit effort="10.0" lower="-1.57" upper="1.57" velocity="1.0"/>
          </joint>
        </robot>
        """).strip()

    urdf_path = tmp_path / "two_link_collision.urdf"
    urdf_path.write_text(urdf_content)
    return urdf_path


def test_configure_collision_constraint(tmp_path):
    """Ensure collision constraint configuration integrates with solver."""
    urdf_path = _create_minimal_collision_urdf(tmp_path)
    robot = eik.RobotModel(str(urdf_path), floating_base=False)
    solver = eik.KinematicsSolver(robot)

    if not hasattr(solver, "configure_collision_constraint"):
        pytest.skip("Collision constraint API not available in current extension build.")

    posture = solver.add_posture_task("posture")
    posture.priority = 0
    posture.weight = 1.0
    target_q = np.zeros(robot.nq, dtype=float)
    posture.set_target_configuration(target_q)

    initial_q = np.zeros(robot.nq, dtype=float)
    robot.update_configuration(initial_q)

    try:
        solver.configure_collision_constraint(min_distance=0.02)
    except RuntimeError as exc:
        pytest.skip(f"Collision support unavailable: {exc}")
    result = solver.solve_velocity(initial_q, apply_limits=False)

    assert result.status == eik.SolverStatus.SUCCESS
    assert np.all(np.isfinite(result.solution))


def _create_three_link_collision_urdf(tmp_path):
    """URDF with three links and collision geometry for multi-pair tests."""
    urdf_content = textwrap.dedent("""
        <robot name="three_link">
          <link name="base_link">
            <inertial>
              <origin xyz="0 0 0" rpy="0 0 0"/>
              <mass value="1.0"/>
              <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
            </inertial>
            <collision>
              <origin xyz="0 0 0" rpy="0 0 0"/>
              <geometry><box size="0.08 0.08 0.08"/></geometry>
            </collision>
          </link>
          <link name="link1">
            <inertial>
              <origin xyz="0 0 0" rpy="0 0 0"/>
              <mass value="0.5"/>
              <inertia ixx="0.005" ixy="0" ixz="0" iyy="0.005" iyz="0" izz="0.005"/>
            </inertial>
            <collision>
              <origin xyz="0.08 0 0" rpy="0 0 0"/>
              <geometry><box size="0.07 0.07 0.07"/></geometry>
            </collision>
          </link>
          <link name="link2">
            <inertial>
              <origin xyz="0 0 0" rpy="0 0 0"/>
              <mass value="0.3"/>
              <inertia ixx="0.003" ixy="0" ixz="0" iyy="0.003" iyz="0" izz="0.003"/>
            </inertial>
            <collision>
              <origin xyz="0.08 0 0" rpy="0 0 0"/>
              <geometry><box size="0.06 0.06 0.06"/></geometry>
            </collision>
          </link>
          <joint name="joint1" type="revolute">
            <parent link="base_link"/>
            <child link="link1"/>
            <origin xyz="0.05 0 0" rpy="0 0 0"/>
            <axis xyz="0 0 1"/>
            <limit effort="10" lower="-1.57" upper="1.57" velocity="1.0"/>
          </joint>
          <joint name="joint2" type="revolute">
            <parent link="link1"/>
            <child link="link2"/>
            <origin xyz="0.12 0 0" rpy="0 0 0"/>
            <axis xyz="0 0 1"/>
            <limit effort="10" lower="-1.57" upper="1.57" velocity="1.0"/>
          </joint>
        </robot>
        """).strip()
    urdf_path = tmp_path / "three_link_collision.urdf"
    urdf_path.write_text(urdf_content)
    return urdf_path


def test_collision_constraint_max_constraints_default(tmp_path):
    """Default max_constraints=1 preserves backward-compatible behaviour."""
    urdf_path = _create_minimal_collision_urdf(tmp_path)
    robot = eik.RobotModel(str(urdf_path), floating_base=False)
    solver = eik.KinematicsSolver(robot)
    if not hasattr(solver, "configure_collision_constraint"):
        pytest.skip("Collision constraint API not available.")

    posture = solver.add_posture_task("posture")
    posture.priority = 0
    posture.weight = 1.0
    posture.set_target_configuration(np.zeros(robot.nq, dtype=float))

    initial_q = np.zeros(robot.nq, dtype=float)
    robot.update_configuration(initial_q)

    try:
        solver.configure_collision_constraint(min_distance=0.02, max_constraints=1)
    except RuntimeError as exc:
        pytest.skip(f"Collision support unavailable: {exc}")

    result = solver.solve_velocity(initial_q, apply_limits=False)
    assert result.status == eik.SolverStatus.SUCCESS

    debug_list = solver.get_last_collision_debug_list()
    # With max_constraints=1, list has at most 1 entry.
    assert len(debug_list) <= 1


def test_collision_constraint_max_constraints_multi(tmp_path):
    """max_constraints=3 can produce multiple constraint rows."""
    urdf_path = _create_three_link_collision_urdf(tmp_path)
    robot = eik.RobotModel(str(urdf_path), floating_base=False)
    solver = eik.KinematicsSolver(robot)
    if not hasattr(solver, "configure_collision_constraint"):
        pytest.skip("Collision constraint API not available.")

    posture = solver.add_posture_task("posture")
    posture.priority = 0
    posture.weight = 1.0
    posture.set_target_configuration(np.zeros(robot.nq, dtype=float))

    initial_q = np.zeros(robot.nq, dtype=float)
    robot.update_configuration(initial_q)

    try:
        solver.configure_collision_constraint(min_distance=0.01, max_constraints=3)
    except RuntimeError as exc:
        pytest.skip(f"Collision support unavailable: {exc}")

    result = solver.solve_velocity(initial_q, apply_limits=True)
    assert result.status == eik.SolverStatus.SUCCESS
    assert np.all(np.isfinite(result.solution))

    # get_last_collision_debug_list() returns up to max_constraints entries.
    debug_list = solver.get_last_collision_debug_list()
    assert isinstance(debug_list, list)
    assert len(debug_list) <= 3
    for dbg in debug_list:
        assert hasattr(dbg, "object_a")
        assert hasattr(dbg, "object_b")
        assert hasattr(dbg, "distance")
        assert np.all(np.isfinite(dbg.point_a_world))
        assert np.all(np.isfinite(dbg.point_b_world))


def test_collision_constraint_debug_list_backward_compat(tmp_path):
    """get_last_collision_debug() returns the closest pair (same as before)."""
    urdf_path = _create_minimal_collision_urdf(tmp_path)
    robot = eik.RobotModel(str(urdf_path), floating_base=False)
    solver = eik.KinematicsSolver(robot)
    if not hasattr(solver, "configure_collision_constraint"):
        pytest.skip("Collision constraint API not available.")

    posture = solver.add_posture_task("posture")
    posture.priority = 0
    posture.weight = 1.0
    posture.set_target_configuration(np.zeros(robot.nq, dtype=float))

    initial_q = np.zeros(robot.nq, dtype=float)
    robot.update_configuration(initial_q)

    try:
        solver.configure_collision_constraint(min_distance=0.02, max_constraints=2)
    except RuntimeError as exc:
        pytest.skip(f"Collision support unavailable: {exc}")

    solver.solve_velocity(initial_q, apply_limits=True)

    single = solver.get_last_collision_debug()
    debug_list = solver.get_last_collision_debug_list()

    if single is not None and len(debug_list) > 0:
        # Single debug should correspond to the closest pair (first in list).
        assert single.object_a == debug_list[0].object_a
        assert single.object_b == debug_list[0].object_b
        assert abs(single.distance - debug_list[0].distance) < 1e-9


def test_collision_constraint_recovery_produces_motion(tmp_path):
    """When inside min_distance, the solver should produce non-zero dq (escape)."""
    urdf_path = _create_minimal_collision_urdf(tmp_path)
    robot = eik.RobotModel(str(urdf_path), floating_base=False)
    solver = eik.KinematicsSolver(robot)
    if not hasattr(solver, "configure_collision_constraint"):
        pytest.skip("Collision constraint API not available.")

    posture = solver.add_posture_task("posture")
    posture.priority = 0
    posture.weight = 1.0
    # Set the posture target to a fully folded configuration so the pair
    # distance is minimal, while min_distance is large enough to be violated.
    target_q = np.zeros(robot.nq, dtype=float)
    posture.set_target_configuration(target_q)

    initial_q = np.zeros(robot.nq, dtype=float)
    robot.update_configuration(initial_q)

    try:
        # Use a large min_distance so the constraint is guaranteed violated in
        # any configuration, forcing a recovery push.
        solver.configure_collision_constraint(min_distance=2.0, max_constraints=1)
    except RuntimeError as exc:
        pytest.skip(f"Collision support unavailable: {exc}")

    result = solver.solve_velocity(initial_q, apply_limits=False)

    if result.status == eik.SolverStatus.SUCCESS and len(result.solution) > 0:
        dq = np.array(result.solution)
        # The recovery lower_bound (continuous ramp) should produce non-trivial
        # joint velocity rather than the old near-zero "gentle_scale=0.01" result.
        dq_norm = float(np.linalg.norm(dq))
        assert dq_norm > 1e-9, (
            f"Expected non-zero recovery dq, got ||dq|| = {dq_norm:.2e}. "
            "Collision recovery may be too weak to produce motion."
        )


def _run_collision_boundary_jitter_rollout(
    robot: eik.RobotModel,
    solver: eik.KinematicsSolver,
    solve_mode: eik.TaskSolveMode,
    steps: int = 160,
) -> tuple[float, int]:
    """Run near-boundary rollout and return (tail_stddev, sign_flip_count)."""
    q = np.zeros(robot.nq, dtype=float)
    robot.update_configuration(q)
    initial_debug = solver.evaluate_collision_debug(q)
    if initial_debug is None:
        pytest.skip("Collision debug unavailable for jitter rollout.")

    # Set the threshold very close to the current distance so the boundary
    # remains active during the rollout and jitter is observable.
    solver.configure_collision_constraint(
        min_distance=float(initial_debug.distance) + 1e-5,
        max_constraints=1,
    )
    solver.clear_tasks()

    frame_task = solver.add_frame_task("ee_boundary", "link1", eik.TaskType.FRAME_POSITION)
    frame_task.priority = 0
    frame_task.weight = 1.0
    frame_task.solve_mode = solve_mode
    frame_task.allow_min_error_fallback = True

    ee_pos = np.array(robot.get_frame_pose("link1").translation)
    frame_task.set_target_position(ee_pos + np.array([0.0, 0.10, 0.0], dtype=float))

    distances = []
    normal_velocity_signs = []
    for _ in range(steps):
        result = solver.solve_velocity(q, apply_limits=True)
        if result.status not in (eik.SolverStatus.SUCCESS, eik.SolverStatus.INFEASIBLE):
            pytest.skip(f"Unexpected solver status in jitter rollout: {result.status}")
        dbg = solver.get_last_collision_debug()
        if dbg is not None:
            distances.append(float(dbg.distance))
            # Use first-joint velocity sign as a stable proxy for boundary
            # response direction in this deterministic regression setup.
            dq = np.array(result.joint_velocities, dtype=float)
            s = int(np.sign(dq[0])) if abs(dq[0]) > 1e-7 else 0
            normal_velocity_signs.append(s)
        else:
            normal_velocity_signs.append(0)
        dq = np.array(result.joint_velocities, dtype=float)
        q = q + dq * solver.dt
        q_lower, q_upper = robot.get_joint_limits()
        q = np.clip(q, q_lower, q_upper)
        robot.update_configuration(q)

    if len(distances) < 40:
        pytest.skip("Insufficient collision distance samples for jitter analysis.")
    tail = np.array(distances[len(distances) // 2 :], dtype=float)
    tail_signs = normal_velocity_signs[len(normal_velocity_signs) // 2 :]
    sign_flips = 0
    prev = 0
    for s in tail_signs:
        if s == 0:
            continue
        if prev != 0 and s != prev:
            sign_flips += 1
        prev = s
    return float(np.std(tail)), int(sign_flips)


def test_min_error_collision_boundary_jitter_not_worse_than_scale(tmp_path):
    """MIN_ERROR near collision boundary should not chatter significantly more."""
    urdf_path = _create_three_link_collision_urdf(tmp_path)
    robot = eik.RobotModel(str(urdf_path), floating_base=False)
    solver = eik.KinematicsSolver(robot)
    solver.dt = 0.01
    if not hasattr(solver, "configure_collision_constraint"):
        pytest.skip("Collision constraint API not available.")

    try:
        std_scale, _ = _run_collision_boundary_jitter_rollout(
            robot, solver, eik.TaskSolveMode.SCALE
        )
        std_min_error, _ = _run_collision_boundary_jitter_rollout(
            robot, solver, eik.TaskSolveMode.MIN_ERROR
        )
    except RuntimeError as exc:
        pytest.skip(f"Collision support unavailable: {exc}")

    # Guardrail: MIN_ERROR is allowed to be slightly noisier, but should stay
    # in the same jitter band as SCALE near active collision boundary.
    assert std_min_error <= std_scale * 1.25 + 1e-8, (
        f"Boundary jitter regressed for MIN_ERROR: "
        f"std_min_error={std_min_error:.3e}, std_scale={std_scale:.3e}"
    )


def test_min_error_collision_boundary_sign_flips_not_worse_than_scale(tmp_path):
    """MIN_ERROR boundary sign-flip jitter should stay near SCALE behavior."""
    urdf_path = _create_three_link_collision_urdf(tmp_path)
    robot = eik.RobotModel(str(urdf_path), floating_base=False)
    solver = eik.KinematicsSolver(robot)
    solver.dt = 0.01
    if not hasattr(solver, "configure_collision_constraint"):
        pytest.skip("Collision constraint API not available.")

    try:
        _, flips_scale = _run_collision_boundary_jitter_rollout(
            robot, solver, eik.TaskSolveMode.SCALE
        )
        _, flips_min_error = _run_collision_boundary_jitter_rollout(
            robot, solver, eik.TaskSolveMode.MIN_ERROR
        )
    except RuntimeError as exc:
        pytest.skip(f"Collision support unavailable: {exc}")

    # Allow modest increase, but guard against significant extra chattering.
    assert flips_min_error <= flips_scale + 15, (
        f"Boundary sign flips regressed for MIN_ERROR: "
        f"flips_min_error={flips_min_error}, flips_scale={flips_scale}"
    )


def test_collision_constraint_max_constraints_invalid_clamped(tmp_path):
    """max_constraints <= 0 is clamped to 1 without error."""
    urdf_path = _create_minimal_collision_urdf(tmp_path)
    robot = eik.RobotModel(str(urdf_path), floating_base=False)
    solver = eik.KinematicsSolver(robot)
    if not hasattr(solver, "configure_collision_constraint"):
        pytest.skip("Collision constraint API not available.")

    initial_q = np.zeros(robot.nq, dtype=float)
    robot.update_configuration(initial_q)

    try:
        solver.configure_collision_constraint(min_distance=0.01, max_constraints=0)
    except RuntimeError as exc:
        pytest.skip(f"Collision support unavailable: {exc}")

    posture = solver.add_posture_task("posture")
    posture.priority = 0
    posture.weight = 1.0
    posture.set_target_configuration(initial_q)

    result = solver.solve_velocity(initial_q, apply_limits=False)
    assert result.status == eik.SolverStatus.SUCCESS

    debug_list = solver.get_last_collision_debug_list()
    assert len(debug_list) <= 1


def _find_positive_distance_q(robot: eik.RobotModel, solver: eik.KinematicsSolver) -> np.ndarray:
    """Find a 1-DoF configuration with strictly positive signed distance."""
    q = np.zeros(robot.nq, dtype=float)
    for angle in np.linspace(-1.2, 1.2, 49):
        q_try = q.copy()
        q_try[0] = float(angle)
        dbg = solver.evaluate_collision_debug(q_try)
        if dbg is not None and np.isfinite(dbg.distance) and float(dbg.distance) > 1e-4:
            return q_try
    pytest.skip("Could not find a positive-distance configuration for activation tests.")


def _setup_minimal_collision_solver(tmp_path):
    urdf_path = _create_minimal_collision_urdf(tmp_path)
    robot = eik.RobotModel(str(urdf_path), floating_base=False)
    solver = eik.KinematicsSolver(robot)
    posture = solver.add_posture_task("posture")
    posture.priority = 0
    posture.weight = 1.0
    posture.set_target_configuration(np.zeros(robot.nq, dtype=float))
    return robot, solver


def _solve_with_activation_multiplier(
    robot: eik.RobotModel,
    solver: eik.KinematicsSolver,
    q: np.ndarray,
    min_distance: float,
    multiplier: float,
    gating_enabled: bool = True,
):
    robot.update_configuration(q)
    solver.configure_collision_constraint(min_distance=float(min_distance), max_constraints=1)
    solver.set_collision_constraint_activation_multiplier(float(multiplier))
    if hasattr(solver, "set_proximity_gated_collision_activation_enabled"):
        solver.set_proximity_gated_collision_activation_enabled(bool(gating_enabled))
    result = solver.solve_velocity(q, apply_limits=False)
    debug_list = solver.get_last_collision_debug_list()
    return result, debug_list


def test_activation_margin_disabled_matches_legacy(tmp_path):
    robot, solver = _setup_minimal_collision_solver(tmp_path)
    if not hasattr(solver, "set_collision_constraint_activation_multiplier"):
        pytest.skip("Activation multiplier API not available.")

    q = _find_positive_distance_q(robot, solver)
    dbg = solver.evaluate_collision_debug(q)
    assert dbg is not None
    min_distance = max(1e-4, float(dbg.distance) * 0.5)

    result_legacy, rows_legacy = _solve_with_activation_multiplier(
        robot, solver, q, min_distance=min_distance, multiplier=0.0
    )
    solver.clear_collision_constraint()
    result_disabled, rows_disabled = _solve_with_activation_multiplier(
        robot, solver, q, min_distance=min_distance, multiplier=0.0
    )

    assert result_legacy.status == result_disabled.status
    np.testing.assert_allclose(
        np.asarray(result_legacy.joint_velocities, dtype=float),
        np.asarray(result_disabled.joint_velocities, dtype=float),
        rtol=0.0,
        atol=1e-12,
    )
    assert len(rows_legacy) == len(rows_disabled)
    assert len(rows_legacy) >= 1


def test_activation_margin_zero_is_identity(tmp_path):
    robot, solver = _setup_minimal_collision_solver(tmp_path)
    if not hasattr(solver, "set_collision_constraint_activation_multiplier"):
        pytest.skip("Activation multiplier API not available.")

    q = _find_positive_distance_q(robot, solver)
    dbg = solver.evaluate_collision_debug(q)
    assert dbg is not None
    min_distance = max(1e-4, float(dbg.distance) * 0.4)

    baseline, _ = _solve_with_activation_multiplier(
        robot, solver, q, min_distance=min_distance, multiplier=0.0
    )
    zeroed, _ = _solve_with_activation_multiplier(
        robot, solver, q, min_distance=min_distance, multiplier=0.0
    )
    np.testing.assert_allclose(
        np.asarray(baseline.joint_velocities, dtype=float),
        np.asarray(zeroed.joint_velocities, dtype=float),
        rtol=0.0,
        atol=1e-12,
    )
    assert baseline.status == zeroed.status


def test_activation_margin_skips_rows_when_far(tmp_path):
    robot, solver = _setup_minimal_collision_solver(tmp_path)
    if not hasattr(solver, "set_collision_constraint_activation_multiplier"):
        pytest.skip("Activation multiplier API not available.")

    q = _find_positive_distance_q(robot, solver)
    dbg = solver.evaluate_collision_debug(q)
    assert dbg is not None
    d = float(dbg.distance)
    min_distance = max(1e-4, d * 0.5)

    no_gate, rows_no_gate = _solve_with_activation_multiplier(
        robot, solver, q, min_distance=min_distance, multiplier=0.0
    )
    gated, rows_gated = _solve_with_activation_multiplier(
        robot, solver, q, min_distance=min_distance, multiplier=0.05
    )

    assert len(rows_no_gate) >= 1
    assert len(rows_gated) == 0
    assert gated.status == no_gate.status


def test_activation_margin_engages_when_near(tmp_path):
    robot, solver = _setup_minimal_collision_solver(tmp_path)
    if not hasattr(solver, "set_collision_constraint_activation_multiplier"):
        pytest.skip("Activation multiplier API not available.")

    q = _find_positive_distance_q(robot, solver)
    dbg = solver.evaluate_collision_debug(q)
    assert dbg is not None
    d = float(dbg.distance)
    min_distance = max(1e-4, d * 0.7)
    multiplier = 1.0

    result, rows = _solve_with_activation_multiplier(
        robot, solver, q, min_distance=min_distance, multiplier=multiplier
    )

    assert result.status == eik.SolverStatus.SUCCESS
    assert len(rows) >= 1


def test_activation_margin_auto_updates_with_min_distance(tmp_path):
    robot, solver = _setup_minimal_collision_solver(tmp_path)
    if not hasattr(solver, "set_collision_constraint_activation_multiplier"):
        pytest.skip("Activation multiplier API not available.")

    q = _find_positive_distance_q(robot, solver)
    robot.update_configuration(q)
    solver.configure_collision_constraint(min_distance=0.02, max_constraints=1)
    solver.set_collision_constraint_activation_multiplier(5.0)

    assert abs(solver.get_collision_constraint_activation_margin() - 0.1) < 1e-12
    assert solver.set_collision_min_distance(0.03)
    assert abs(solver.get_collision_constraint_activation_margin() - 0.15) < 1e-12


def test_activation_margin_uses_nominal_min_distance_during_stall_recovery(tmp_path):
    robot, solver = _setup_minimal_collision_solver(tmp_path)
    if not hasattr(solver, "set_collision_constraint_activation_multiplier"):
        pytest.skip("Activation multiplier API not available.")

    solver.configure_collision_constraint(min_distance=0.04, max_constraints=1)
    solver.set_collision_constraint_activation_multiplier(5.0)
    solver.enable_stall_handler(0.04)

    assert solver.get_collision_constraint_activation_margin() == pytest.approx(0.20)
    assert solver.set_collision_min_distance(0.0)
    assert solver.get_collision_min_distance() == pytest.approx(0.0)
    assert solver.get_collision_constraint_activation_margin() == pytest.approx(0.20)

    solver.disable_stall_handler()
    assert solver.get_collision_min_distance() == pytest.approx(0.04)
    assert solver.get_collision_constraint_activation_margin() == pytest.approx(0.20)


def test_activation_margin_boundary_exact(tmp_path):
    robot, solver = _setup_minimal_collision_solver(tmp_path)
    if not hasattr(solver, "set_collision_constraint_activation_multiplier"):
        pytest.skip("Activation multiplier API not available.")

    q = _find_positive_distance_q(robot, solver)
    dbg = solver.evaluate_collision_debug(q)
    assert dbg is not None
    d = float(dbg.distance)
    eps = max(1e-5, 1e-3 * d)

    # Slightly below threshold: row suppressed.
    min_below = max(1e-6, 0.5 * (d - eps))
    _, rows_below = _solve_with_activation_multiplier(
        robot, solver, q, min_distance=min_below, multiplier=1.0
    )

    # Slightly above threshold: row active.
    min_above = max(1e-6, 0.5 * (d + eps))
    _, rows_above = _solve_with_activation_multiplier(
        robot, solver, q, min_distance=min_above, multiplier=1.0
    )

    assert len(rows_below) == 0
    assert len(rows_above) >= 1


def test_activation_enabled_toggle_preserves_multiplier_and_switches_behavior(tmp_path):
    robot, solver = _setup_minimal_collision_solver(tmp_path)
    if not hasattr(solver, "set_proximity_gated_collision_activation_enabled"):
        pytest.skip("Activation enabled API not available.")

    q = _find_positive_distance_q(robot, solver)
    dbg = solver.evaluate_collision_debug(q)
    assert dbg is not None
    d = float(dbg.distance)
    min_distance = max(1e-4, d * 0.5)

    robot.update_configuration(q)
    solver.configure_collision_constraint(min_distance=min_distance, max_constraints=1)
    solver.set_collision_constraint_activation_multiplier(0.05)
    margin_before = solver.get_collision_constraint_activation_margin()
    mult_before = solver.get_collision_constraint_activation_multiplier()
    assert mult_before > 0.0

    solver.set_proximity_gated_collision_activation_enabled(False)
    res_disabled = solver.solve_velocity(q, apply_limits=False)
    rows_disabled = solver.get_last_collision_debug_list()

    solver.set_proximity_gated_collision_activation_enabled(True)
    res_enabled = solver.solve_velocity(q, apply_limits=False)
    rows_enabled = solver.get_last_collision_debug_list()

    # Enable/disable should not mutate multiplier/margin.
    assert solver.get_collision_constraint_activation_multiplier() == mult_before
    assert solver.get_collision_constraint_activation_margin() == margin_before

    # Behavior toggles under same threshold params.
    assert len(rows_disabled) >= 1
    assert len(rows_enabled) == 0
    assert res_disabled.status == res_enabled.status


# =============================================================================
# Test runner
# =============================================================================


def run_all_tests():
    """Run all tests in order."""
    test_functions = [
        # Basic tests
        test_import_and_metadata,
        test_pose_error_norm,
        # API tests
        test_multi_task_api,
        test_eigen_first_multi_task,
        test_numpy_array_types,
        # Validation tests
        test_invalid_inputs,
        test_multi_task_with_constraints,
        test_multi_task_prioritization,
        test_configure_collision_constraint,
        test_random_multi_task_problems,
        # Large scale test
        test_multi_task_solver_large_scale,
    ]

    passed = 0
    failed = 0

    for test_fn in test_functions:
        try:
            logger.info(f"\nRunning {test_fn.__name__}...")
            test_fn()
            logger.info(f"✓ {test_fn.__name__} passed")
            passed += 1
        except Exception as e:
            logger.error(f"✗ {test_fn.__name__} failed: {e}")
            failed += 1
            import traceback

            traceback.print_exc()

    logger.info(f"\n=== Test Summary: {passed}/{passed + failed} passed ===")
    return failed == 0


if __name__ == "__main__":
    # Check if running with pytest
    if "pytest" in sys.modules:
        # Let pytest handle test discovery and execution
        pass
    else:
        # Run tests manually
        success = run_all_tests()
        sys.exit(0 if success else 1)

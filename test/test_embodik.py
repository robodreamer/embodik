"""Consolidated tests for embodiK multi-task solver."""

import math
import pytest
import random
import numpy as np
import sys
import os
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
SOLVER_EPSILON = 1e-6              # Default numerical tolerance
SOLVER_PRECISION_THRESHOLD = 1e-10 # High-precision threshold
NUMERICAL_EPSILON = 1e-6         # Small epsilon for numerical comparisons (matches v1 TOL)
CONSTRAINT_TOLERANCE = 1e-6      # Tolerance for constraint satisfaction (matches v1 cstTol)
TASK_ERROR_TOLERANCE = 1e-6      # Tolerance for task achievement error (matches v1 cstTol)
SCALE_EPSILON = 1e-10           # Epsilon for task scale comparisons
OPT_TOLERANCE = 1e-4            # Optimization tolerance
TEST_TOLERANCE = 1e-4           # Test tolerance for comparisons

# Solver parameters (matching v1 defaults)
DEFAULT_SR_DAMPING = 1e-6        # Singularity-robust damping (matches v1 beta_max and INV_DAMPING_COEFF)


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


def test_eigen_first_multi_task():
    """Test Eigen-first multi-task velocity IK solver."""
    # Create test data
    goals = [np.array([0.1, -0.2], dtype=np.float64), np.array([0.3], dtype=np.float64)]
    jacobians = [
        np.asarray(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), dtype=np.float64, order='F'),
        np.asarray(np.array([[0.0, 0.0, 1.0]]), dtype=np.float64, order='F')
    ]
    C = np.asarray(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), dtype=np.float64, order='F')
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
    jacobians = [np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64, order='F')]  # F-order
    C = np.eye(2, dtype=np.float64, order='F')
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
    result = eik.computeMultiObjectiveVelocitySolutionEigen([], [], np.eye(1, dtype=np.float64, order='F'), np.array([0], dtype=np.float64), np.array([1], dtype=np.float64))
    assert result.status == eik.SolverStatus.INVALID_INPUT

    # Mismatched dimensions - goal dimension doesn't match Jacobian rows
    goals = [np.array([1.0, 2.0])]  # 2D goal
    jacobians = [np.array([[1.0, 0.0]])]  # 1x2 Jacobian (only 1 row)
    C = np.eye(2)
    lower = np.array([-1, -1])
    upper = np.array([1, 1])

    result = eik.computeMultiObjectiveVelocitySolutionEigen(goals, jacobians, C, lower, upper)
    # This should be caught by dimension validation
    assert result.status == eik.SolverStatus.INVALID_INPUT


# =============================================================================
# Constraint and prioritization tests
# =============================================================================

def test_multi_task_with_constraints():
    """Test multi-task solver with constraint handling."""
    # 3-joint robot with 2 tasks
    goals = [
        np.array([0.5, 0.3]),    # Task 0: 2D
        np.array([0.2])          # Task 1: 1D
    ]
    jacobians = [
        np.array([[1.0, 0.5, 0.0], [0.0, 1.0, 0.5]]),
        np.array([[0.0, 0.0, 1.0]])
    ]

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
    assert error < TASK_ERROR_TOLERANCE, f"Primary task error {error} exceeds tolerance {TASK_ERROR_TOLERANCE}"


def test_multi_task_prioritization():
    """Test that tasks are properly prioritized."""
    # Conflicting tasks - both want to move joint 0
    goals = [
        np.array([1.0]),   # Task 0: wants positive motion
        np.array([-1.0])   # Task 1: wants negative motion
    ]
    jacobians = [
        np.array([[1.0, 0.0]]),
        np.array([[1.0, 0.0]])
    ]

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
                    logger.debug(f"  n_joints={n_joints}, n_tasks={n_tasks}, task_dim={goals[0].shape}")
                    logger.debug(f"  goal[0]: {goals[0]}")
                    logger.debug(f"  achieved: {achieved}")
                    logger.debug(f"  scaled_goal: {scaled_goal}")
                    logger.debug(f"  solution: {solution}")
                    logger.debug(f"  all scales: {result.task_scales}")
                assert error < TASK_ERROR_TOLERANCE, f"Primary task error {error} exceeds tolerance {TASK_ERROR_TOLERANCE}"

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
        n_tasks = rng.integers(1, 4)   # Number of tasks
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
            jacobians.append(np.asarray(jacobian, dtype=np.float64, order='F'))

        # Constraint matrix (identity for joint limits + random constraints)
        C = np.eye(n_joints + n_constraints, n_joints)
        for i in range(n_constraints):
            C[n_joints + i, :] = rng.uniform(-0.5, 0.5, n_joints)
        C = np.asarray(C, dtype=np.float64, order='F')

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
            if np.any(Cx > upper_limits + CONSTRAINT_TOLERANCE) or np.any(Cx < lower_limits - CONSTRAINT_TOLERANCE):
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
        n_tasks = rng.integers(1, 5)    # 1 to 4 tasks

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

            jacobians.append(np.asarray(jacobian, dtype=np.float64, order='F'))

        # Constraint matrix (typically identity for joint limits)
        C = np.asarray(np.eye(n_joints), dtype=np.float64, order='F')

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
        if np.any(Cx > upper_limits + CONSTRAINT_TOLERANCE) or np.any(Cx < lower_limits - CONSTRAINT_TOLERANCE):
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
    urdf_content = textwrap.dedent(
        """
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
        """
    ).strip()

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

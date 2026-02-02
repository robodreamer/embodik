"""
Tests for GPU-accelerated velocity solver.

These tests validate that the GPU solver matches the CPU solver
within specified tolerances.
"""

import math
import pytest
import numpy as np
from typing import Tuple, List
import sys
import os

# Test tolerances
VELOCITY_ATOL = 1e-4  # Velocity solution tolerance
SCALE_ATOL = 1e-3  # Task scale tolerance (soft saturation may differ)
SIMPLE_ATOL = 1e-6  # Tolerance for simple cases without saturation
CONSTRAINT_TOL = 1e-4  # Constraint satisfaction tolerance


def generate_random_problem(
    seed: int,
    n_dof: int = 7,
    n_tasks: int = 2,
    task_dims: List[int] = None,
    n_constraints: int = None,
) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a random multi-task IK problem.

    Returns:
        (goals, jacobians, C, lower, upper)
    """
    rng = np.random.default_rng(seed)

    if task_dims is None:
        task_dims = [min(6, n_dof - 1) for _ in range(n_tasks)]

    if n_constraints is None:
        n_constraints = n_dof

    goals = []
    jacobians = []

    for task_dim in task_dims:
        goal = rng.uniform(-0.5, 0.5, task_dim)
        jacobian = rng.uniform(-1.0, 1.0, (task_dim, n_dof))
        goals.append(goal.astype(np.float64))
        jacobians.append(np.asfortranarray(jacobian.astype(np.float64)))

    # Constraint matrix (identity for joint limits)
    C = np.asfortranarray(np.eye(n_constraints, n_dof, dtype=np.float64))

    # Bounds
    lower = np.full(n_constraints, -1.0, dtype=np.float64)
    upper = np.full(n_constraints, 1.0, dtype=np.float64)

    return goals, jacobians, C, lower, upper


def flatten_problem(
    goals: List[np.ndarray],
    jacobians: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Flatten goals and jacobians for GPU solver input."""
    targets_flat = np.concatenate(goals)
    jacobians_flat = np.concatenate([j.flatten() for j in jacobians])
    return targets_flat, jacobians_flat


# =============================================================================
# Availability checks
# =============================================================================

def test_gpu_availability_check():
    """Test that GPU availability check works."""
    try:
        from embodik.gpu_solver import check_gpu_availability

        status = check_gpu_availability()

        assert isinstance(status, dict)
        assert "casadi" in status
        assert "cusadi" in status
        assert "torch_cuda" in status
        assert "all_available" in status

        # All values should be booleans
        for key, value in status.items():
            assert isinstance(value, bool), f"{key} should be bool, got {type(value)}"

    except ImportError:
        pytest.skip("GPU solver not available")


def test_cpu_fallback_always_available():
    """Test that CPU fallback path works even without GPU."""
    try:
        from embodik.gpu_solver import solve_velocity_batched

        # Generate a simple problem
        goals, jacobians, C, lower, upper = generate_random_problem(seed=42, n_dof=3, n_tasks=1, task_dims=[2])

        targets_flat, jacobians_flat = flatten_problem(goals, jacobians)

        # Call with use_gpu=False to force CPU
        result = solve_velocity_batched(
            objective_targets_batch=[targets_flat],
            objective_jacobians_batch=[jacobians_flat],
            constraint_matrices_batch=[C],
            min_bounds_batch=[lower],
            max_bounds_batch=[upper],
            use_gpu=False,
        )

        assert result.status == "fallback_cpu"
        assert result.velocities.shape[0] == 1
        assert result.scales.shape[0] == 1

    except ImportError as e:
        pytest.skip(f"GPU solver not available: {e}")


# =============================================================================
# GPU vs CPU validation tests
# =============================================================================

@pytest.mark.skipif(
    not os.environ.get("EMBODIK_GPU_TESTS", ""),
    reason="GPU tests disabled. Set EMBODIK_GPU_TESTS=1 to enable."
)
class TestGPUvsCPU:
    """Tests comparing GPU and CPU solver outputs."""

    def test_simple_2dof_no_saturation(self):
        """2-DOF, 1 task, no saturation - should match exactly."""
        import embodik as eik
        from embodik.gpu_solver import solve_velocity_batched

        # Simple problem that doesn't need saturation
        goals = [np.array([0.1, 0.1], dtype=np.float64)]
        jacobians = [np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64, order='F')]
        C = np.eye(2, dtype=np.float64, order='F')
        lower = np.array([-1.0, -1.0], dtype=np.float64)
        upper = np.array([1.0, 1.0], dtype=np.float64)

        # CPU solve
        cpu_result = eik.computeMultiObjectiveVelocitySolutionEigen(
            goals, jacobians, C, lower, upper
        )

        # GPU solve
        targets_flat, jacobians_flat = flatten_problem(goals, jacobians)
        gpu_result = solve_velocity_batched(
            [targets_flat], [jacobians_flat], [C], [lower], [upper],
            use_gpu=True
        )

        # Compare
        if gpu_result.status == "success":
            assert np.allclose(cpu_result.solution, gpu_result.velocities[0], atol=SIMPLE_ATOL)
            assert np.allclose(cpu_result.task_scales, gpu_result.scales[0], atol=SIMPLE_ATOL)
        else:
            # Fallback to CPU should still match
            assert np.allclose(cpu_result.solution, gpu_result.velocities[0], atol=SIMPLE_ATOL)

    def test_7dof_multi_task(self):
        """7-DOF, 2 tasks, may require constraint saturation."""
        import embodik as eik
        from embodik.gpu_solver import solve_velocity_batched

        goals, jacobians, C, lower, upper = generate_random_problem(
            seed=123, n_dof=7, n_tasks=2, task_dims=[6, 3]
        )

        # CPU solve
        cpu_result = eik.computeMultiObjectiveVelocitySolutionEigen(
            goals, jacobians, C, lower, upper
        )

        # GPU solve
        targets_flat, jacobians_flat = flatten_problem(goals, jacobians)
        gpu_result = solve_velocity_batched(
            [targets_flat], [jacobians_flat], [C], [lower], [upper],
            use_gpu=True
        )

        # Compare with looser tolerance (saturation approximation)
        assert np.allclose(cpu_result.solution, gpu_result.velocities[0], atol=VELOCITY_ATOL)
        assert np.allclose(cpu_result.task_scales, gpu_result.scales[0], atol=SCALE_ATOL)

    def test_random_batch_100(self):
        """100 random problems, batch solve."""
        import embodik as eik
        from embodik.gpu_solver import solve_velocity_batched

        n_problems = 100
        n_dof = 7
        task_dims = [6]

        # Generate problems
        all_goals = []
        all_jacobians = []
        all_C = []
        all_lower = []
        all_upper = []

        for i in range(n_problems):
            goals, jacobians, C, lower, upper = generate_random_problem(
                seed=i, n_dof=n_dof, n_tasks=1, task_dims=task_dims
            )
            all_goals.append(goals)
            all_jacobians.append(jacobians)
            all_C.append(C)
            all_lower.append(lower)
            all_upper.append(upper)

        # CPU sequential solve
        cpu_solutions = []
        cpu_scales = []
        for i in range(n_problems):
            result = eik.computeMultiObjectiveVelocitySolutionEigen(
                all_goals[i], all_jacobians[i], all_C[i], all_lower[i], all_upper[i]
            )
            cpu_solutions.append(np.array(result.solution))
            cpu_scales.append(np.array(result.task_scales))

        # GPU batch solve
        targets_batch = [flatten_problem(g, j)[0] for g, j in zip(all_goals, all_jacobians)]
        jacobians_batch = [flatten_problem(g, j)[1] for g, j in zip(all_goals, all_jacobians)]

        gpu_result = solve_velocity_batched(
            targets_batch, jacobians_batch, all_C, all_lower, all_upper,
            use_gpu=True
        )

        # Compare each
        n_matched = 0
        for i in range(n_problems):
            if np.allclose(cpu_solutions[i], gpu_result.velocities[i], atol=VELOCITY_ATOL):
                n_matched += 1

        # At least 95% should match
        assert n_matched >= 0.95 * n_problems, f"Only {n_matched}/{n_problems} matched"

    def test_constraint_saturation(self):
        """Test with tight bounds forcing task scaling."""
        import embodik as eik
        from embodik.gpu_solver import solve_velocity_batched

        # Large goal with tight bounds - forces scaling
        goals = [np.array([1.0, 1.0, 1.0], dtype=np.float64)]
        jacobians = [np.eye(3, dtype=np.float64, order='F')]
        C = np.eye(3, dtype=np.float64, order='F')
        lower = np.array([-0.1, -0.1, -0.1], dtype=np.float64)
        upper = np.array([0.1, 0.1, 0.1], dtype=np.float64)

        # CPU solve
        cpu_result = eik.computeMultiObjectiveVelocitySolutionEigen(
            goals, jacobians, C, lower, upper
        )

        # GPU solve
        targets_flat, jacobians_flat = flatten_problem(goals, jacobians)
        gpu_result = solve_velocity_batched(
            [targets_flat], [jacobians_flat], [C], [lower], [upper],
            use_gpu=True
        )

        # Both should have scaled down
        assert cpu_result.task_scales[0] < 1.0
        if gpu_result.status == "success":
            assert gpu_result.scales[0, 0] < 1.0
            # Scales should be close
            assert np.allclose(cpu_result.task_scales, gpu_result.scales[0], atol=SCALE_ATOL)

    def test_conflicting_tasks(self):
        """Test priority handling with conflicting tasks."""
        import embodik as eik
        from embodik.gpu_solver import solve_velocity_batched

        # Two tasks wanting opposite directions
        goals = [
            np.array([1.0], dtype=np.float64),   # Task 0: positive
            np.array([-1.0], dtype=np.float64),  # Task 1: negative
        ]
        jacobians = [
            np.array([[1.0, 0.0]], dtype=np.float64, order='F'),
            np.array([[1.0, 0.0]], dtype=np.float64, order='F'),
        ]
        C = np.eye(2, dtype=np.float64, order='F')
        lower = np.array([-0.5, -0.5], dtype=np.float64)
        upper = np.array([0.5, 0.5], dtype=np.float64)

        # CPU solve
        cpu_result = eik.computeMultiObjectiveVelocitySolutionEigen(
            goals, jacobians, C, lower, upper
        )

        # GPU solve
        targets_flat, jacobians_flat = flatten_problem(goals, jacobians)
        gpu_result = solve_velocity_batched(
            [targets_flat], [jacobians_flat], [C], [lower], [upper],
            use_gpu=True
        )

        # Both should prioritize Task 0 (positive direction)
        assert cpu_result.solution[0] > 0
        assert gpu_result.velocities[0, 0] > 0


# =============================================================================
# Batch correctness tests
# =============================================================================

class TestBatchCorrectness:
    """Tests for batch processing correctness."""

    def test_batch_matches_sequential(self):
        """Batched CPU solve should match sequential CPU solves."""
        try:
            from embodik.gpu_solver import solve_velocity_batched
            import embodik as eik
        except ImportError:
            pytest.skip("GPU solver not available")

        n_problems = 10

        # Generate problems
        problems = [generate_random_problem(seed=i, n_dof=5, n_tasks=1, task_dims=[3])
                    for i in range(n_problems)]

        # Sequential CPU
        cpu_solutions = []
        for goals, jacobians, C, lower, upper in problems:
            result = eik.computeMultiObjectiveVelocitySolutionEigen(
                goals, jacobians, C, lower, upper
            )
            cpu_solutions.append(np.array(result.solution))

        # Batch CPU (use_gpu=False)
        targets_batch = [flatten_problem(p[0], p[1])[0] for p in problems]
        jacobians_batch = [flatten_problem(p[0], p[1])[1] for p in problems]
        C_batch = [p[2] for p in problems]
        lower_batch = [p[3] for p in problems]
        upper_batch = [p[4] for p in problems]

        batch_result = solve_velocity_batched(
            targets_batch, jacobians_batch, C_batch, lower_batch, upper_batch,
            use_gpu=False
        )

        # Compare
        for i in range(n_problems):
            assert np.allclose(cpu_solutions[i], batch_result.velocities[i], atol=1e-10)


# =============================================================================
# Edge cases and fallback tests
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_fallback_without_cusadi(self):
        """Test that CPU fallback works when CusADi is not available."""
        try:
            from embodik.gpu_solver import solve_velocity_batched, HAS_CUSADI
        except ImportError:
            pytest.skip("GPU solver module not available")

        goals, jacobians, C, lower, upper = generate_random_problem(
            seed=999, n_dof=4, n_tasks=1, task_dims=[3]
        )
        targets_flat, jacobians_flat = flatten_problem(goals, jacobians)

        # Force CPU path
        result = solve_velocity_batched(
            [targets_flat], [jacobians_flat], [C], [lower], [upper],
            use_gpu=False
        )

        assert result.status == "fallback_cpu"
        assert result.velocities.shape == (1, 4)

    def test_numerical_stability_near_singular(self):
        """Test handling of near-singular Jacobians."""
        try:
            from embodik.gpu_solver import solve_velocity_batched
            import embodik as eik
        except ImportError:
            pytest.skip("GPU solver not available")

        # Near-singular Jacobian
        goals = [np.array([0.1, 0.1], dtype=np.float64)]
        jacobians = [np.array([[1.0, 1.0], [1.0, 1.0 + 1e-8]], dtype=np.float64, order='F')]
        C = np.eye(2, dtype=np.float64, order='F')
        lower = np.array([-1.0, -1.0], dtype=np.float64)
        upper = np.array([1.0, 1.0], dtype=np.float64)

        # CPU solve (should handle gracefully)
        cpu_result = eik.computeMultiObjectiveVelocitySolutionEigen(
            goals, jacobians, C, lower, upper
        )

        # GPU/CPU fallback solve
        targets_flat, jacobians_flat = flatten_problem(goals, jacobians)
        result = solve_velocity_batched(
            [targets_flat], [jacobians_flat], [C], [lower], [upper],
            use_gpu=False
        )

        # Both should produce finite results
        assert np.all(np.isfinite(cpu_result.solution))
        assert np.all(np.isfinite(result.velocities))


# =============================================================================
# Test runner
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

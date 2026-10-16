"""CPU execution checks the GPU kernel's math, not CUDA runtime behavior."""

import numpy as np
import pytest
from scipy.optimize import linprog


def _solve_batch(matrix, target, lower, upper, original):
    wp = pytest.importorskip("warp")
    pytest.importorskip("torch")
    from embodik.gpu.wbc._runtime.warp_bound_release import _make_kernel

    matrix = np.asarray(matrix, dtype=np.float64)
    values = [matrix] + [np.asarray(v, dtype=np.float64) for v in (target, lower, upper, original)]
    inputs = [wp.array(v, dtype=wp.float64, device="cpu") for v in values]
    output = wp.empty((matrix.shape[0], matrix.shape[2]), dtype=wp.float64, device="cpu")
    recovered = wp.empty(matrix.shape[0], dtype=wp.bool, device="cpu")
    wp.launch(
        _make_kernel(matrix.shape[1], matrix.shape[2]),
        dim=matrix.shape[0],
        inputs=inputs,
        outputs=[output, recovered],
        device="cpu",
    )
    return output.numpy(), recovered.numpy()


def _solve(matrix, target, lower, upper, original):
    inputs = [np.asarray(value)[None] for value in (matrix, target, lower, upper, original)]
    output, recovered = _solve_batch(*inputs)
    return output[0], bool(recovered[0])


@pytest.mark.parametrize("direction", [-1.0, 1.0])
@pytest.mark.parametrize("seeded", [False, True])
def test_required_task_null_inward_motion(direction, seeded):
    import embodik as eik

    matrix = np.array([[1.0, 1.0, 0.0]])
    target = np.array([0.5])
    lower, upper = -np.ones(3), np.ones(3)
    if direction > 0:
        lower[2] = 0.01
    else:
        upper[2] = -0.01
    original = np.zeros(3)
    if seeded:
        original[2] = direction * 0.01
    output, recovered = _solve(matrix, target, lower, upper, original)
    assert recovered
    np.testing.assert_allclose(matrix @ output, target, atol=1e-8)
    assert np.all(output >= lower - 1e-8)
    assert np.all(output <= upper + 1e-8)
    native = eik.computeMultiObjectiveVelocitySolutionEigen(
        [target], [matrix], np.eye(3), lower, upper, sr_damping=0.0
    )
    assert native.status == eik.SolverStatus.SUCCESS
    np.testing.assert_allclose(native.task_scales[0], 1.0, atol=1e-8)
    np.testing.assert_allclose(matrix @ np.asarray(native.solution), matrix @ output, atol=1e-8)


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("required_inward", [False, True])
def test_kernel_matches_independent_lp(seed, required_inward):
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(2, 4))
    target = rng.normal(size=2)
    lower = -rng.uniform(0.1, 1.0, size=4)
    upper = rng.uniform(0.1, 1.0, size=4)
    lower[seed % 4] = 0.0
    upper[(seed + 1) % 4] = 0.0
    if required_inward:
        matrix = np.column_stack((matrix, np.zeros(2)))
        lower = np.r_[lower, 0.01 if seed % 2 else -1.0]
        upper = np.r_[upper, 1.0 if seed % 2 else -0.01]
    reference = linprog(
        np.r_[np.zeros(matrix.shape[1]), -1.0],
        A_eq=np.column_stack((matrix, -target)),
        b_eq=np.zeros(2),
        bounds=list(zip(lower, upper)) + [(0.0, 1.0)],
        method="highs",
    )
    assert reference.success
    output, recovered = _solve(matrix, target, lower, upper, np.zeros(matrix.shape[1]))
    assert recovered
    np.testing.assert_allclose(matrix @ output, reference.x[-1] * target, atol=1e-8)
    assert np.all(output >= lower - 1e-8)
    assert np.all(output <= upper + 1e-8)


@pytest.mark.parametrize(
    "case",
    [
        "primary-moving-seed",
        "positive-original",
        "reversed-box",
        "nonfinite-bound",
        "ordinary-null-original",
    ],
)
def test_ineligible_world_preserves_original(case):
    matrix = np.array([[1.0, 1.0, 0.0]])
    lower, upper = -np.ones(3), np.ones(3)
    lower[2] = 0.01
    original = np.array([0.0, 0.0, 0.01])
    if case == "primary-moving-seed":
        matrix[0, 2] = 1.0
    elif case == "positive-original":
        original[0] = 0.1
    elif case == "reversed-box":
        lower[1], upper[1] = 0.5, -0.5
    elif case == "nonfinite-bound":
        upper[1] = np.nan
    else:
        lower[2] = -1.0
    output, recovered = _solve(matrix, [0.5], lower, upper, original)
    assert not recovered
    np.testing.assert_array_equal(output, original)


@pytest.mark.parametrize("direction", [-1.0, 1.0])
@pytest.mark.parametrize("inward", [False, True])
def test_invalid_moving_realization_preserves_achieved_primary(direction, inward):
    matrix = np.array([[1.0, 1.0]])
    original = direction * np.array([0.6, -0.1])
    lower, upper = -np.ones(2), np.ones(2)
    if direction > 0:
        upper[0] = -0.01 if inward else 0.0
    else:
        lower[0] = 0.01 if inward else 0.0
    # The request differs from the achieved motion. Repair must retain the
    # latter, rather than silently increase the task scale to match the request.
    output, recovered = _solve(matrix, [direction * 0.8], lower, upper, original)
    assert recovered
    np.testing.assert_allclose(matrix @ output, matrix @ original, atol=1e-8)
    assert np.all(output >= lower - 1e-8)
    assert np.all(output <= upper + 1e-8)


@pytest.mark.parametrize("shape", [(2, 4), (6, 12), (12, 19)])
@pytest.mark.parametrize("seed", range(12))
def test_invalid_moving_nullspace_realization_repaired(shape, seed):
    rng = np.random.default_rng(seed)
    rows, columns = shape
    matrix = rng.normal(size=shape)
    feasible = rng.uniform(-0.4, 0.4, columns)
    _, _, right = np.linalg.svd(matrix, full_matrices=True)
    null = right[rows]
    original = feasible + 4.0 * null / np.max(np.abs(null))
    output, recovered = _solve(
        matrix, rng.normal(size=rows), -np.ones(columns), np.ones(columns), original
    )
    assert recovered
    np.testing.assert_allclose(matrix @ output, matrix @ original, atol=1e-8)
    assert np.all(output >= -1.0 - 1e-8)
    assert np.all(output <= 1.0 + 1e-8)


def test_invalid_moving_primary_cannot_be_silently_rescaled():
    original = np.array([2.0, 2.0])
    output, recovered = _solve([[1.0, 1.0]], [0.5], [-1.0, -1.0], [1.0, 1.0], original)
    assert not recovered
    np.testing.assert_array_equal(output, original)


def test_moving_repair_isolates_valid_invalid_and_unrepairable_worlds():
    matrix = np.tile([[[1.0, 1.0]]], (4, 1, 1))
    original = np.array([[0.6, -0.1], [-0.1, 0.6], [2.0, 2.0], [0.6, -0.1]])
    lower, upper = -np.ones((4, 2)), np.ones((4, 2))
    upper[:, 0] = 0.0
    target = np.full((4, 1), 0.8)
    target[3, 0] = np.nan
    output, recovered = _solve_batch(matrix, target, lower, upper, original)
    np.testing.assert_array_equal(recovered, [True, False, False, False])
    np.testing.assert_allclose(matrix[0] @ output[0], matrix[0] @ original[0], atol=1e-8)
    np.testing.assert_array_equal(output[1:], original[1:])


@pytest.mark.parametrize("seed", range(32))
@pytest.mark.parametrize("stationary", [True, False])
def test_compensated_inward_seed_matches_lp(seed, stationary):
    import embodik as eik

    rng = np.random.default_rng(seed)
    jacobian = rng.normal(size=(3, 5))
    target = np.zeros(3) if stationary else rng.normal(size=3)
    lower, upper = -np.ones(5), np.ones(5)
    lower[0] = 0.01
    reference = linprog(
        np.r_[np.zeros(5), -1.0],
        A_eq=np.column_stack((jacobian, -target)),
        b_eq=np.zeros(3),
        bounds=list(zip(lower, upper)) + [(0.0, 1.0)],
        method="highs",
    )
    assert reference.success
    output, recovered = _solve(jacobian, target, lower, upper, np.zeros(5))
    assert recovered
    np.testing.assert_allclose(jacobian @ output, reference.x[-1] * target, atol=1e-8)
    assert np.all(output >= lower - 1e-8)
    assert np.all(output <= upper + 1e-8)
    if stationary:
        native = eik.computeMultiObjectiveVelocitySolutionEigen(
            [target], [jacobian], np.eye(5), lower, upper, sr_damping=0.0
        )
        assert native.status == eik.SolverStatus.SUCCESS
        np.testing.assert_allclose(jacobian @ native.solution, jacobian @ output, atol=1e-8)


def test_compensated_seed_isolates_mixed_worlds():
    matrix = np.tile([[[1.0, 1.0, 1.0]]], (5, 1, 1))
    target = np.array([[0.0], [0.5], [0.0], [0.5], [np.nan]])
    lower, upper = -np.ones((5, 3)), np.ones((5, 3))
    lower[:, 0] = 0.01
    lower[2] = 0.1  # No task-null witness: all coordinates must be positive.
    original = np.zeros((5, 3))
    original[3] = [0.1, 0.0, 0.0]  # Healthy primary motion must not be replaced.
    output, recovered = _solve_batch(matrix, target, lower, upper, original)
    np.testing.assert_array_equal(recovered, [True, True, False, False, False])
    np.testing.assert_array_equal(output[2:], original[2:])
    np.testing.assert_allclose(
        np.einsum("bij,bj->bi", matrix[:2], output[:2]), target[:2], atol=1e-8
    )
    assert np.all(output[:2] >= lower[:2] - 1e-8)
    assert np.all(output[:2] <= upper[:2] + 1e-8)


@pytest.mark.parametrize("inward_coordinate", [0, 2])
def test_zero_ray_scale_retains_hard_feasible_task_null_motion(inward_coordinate):
    matrix = np.array([[1.0, 1.0, 0.0], [2.0, 2.0, 0.0]])
    target = np.array([0.1, 0.5])  # Outside the rank-one task image.
    lower, upper = -np.ones(3), np.ones(3)
    lower[inward_coordinate] = 0.01
    reference = linprog(
        [0.0, 0.0, 0.0, -1.0],
        A_eq=np.column_stack((matrix, -target)),
        b_eq=np.zeros(2),
        bounds=list(zip(lower, upper)) + [(0.0, 1.0)],
        method="highs",
    )
    assert reference.success
    assert reference.x[-1] == pytest.approx(0.0, abs=1e-10)
    output, recovered = _solve(matrix, target, lower, upper, np.zeros(3))
    assert recovered
    np.testing.assert_allclose(matrix @ output, 0.0, atol=1e-8)
    assert np.all(output >= lower - 1e-8)
    assert np.all(output <= upper + 1e-8)


@pytest.mark.parametrize("direction", [-1.0, 1.0])
@pytest.mark.parametrize("target_value", [0.5, 1.0, 3.0])
def test_positive_ray_without_zero_task_seed(direction, target_value):
    matrix = np.array([[1.0, 1.0]])
    target = np.array([direction * target_value])
    lower, upper = -np.ones(2), np.ones(2)
    if direction > 0:
        lower[:] = 0.2
    else:
        upper[:] = -0.2
    reference = linprog(
        [0.0, 0.0, -1.0],
        A_eq=np.column_stack((matrix, -target)),
        b_eq=[0.0],
        bounds=list(zip(lower, upper)) + [(0.0, 1.0)],
        method="highs",
    )
    assert reference.success
    output, recovered = _solve(matrix, target, lower, upper, np.zeros(2))
    assert recovered
    np.testing.assert_allclose(matrix @ output, reference.x[-1] * target, atol=1e-8)
    assert np.all(output >= lower - 1e-8)
    assert np.all(output <= upper + 1e-8)


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("dimensions", [(3, 5), (6, 12)])
@pytest.mark.parametrize("target_scale", [1.0, 3.0])
def test_positive_ray_phase_matches_lp_on_coupled_tasks(seed, dimensions, target_scale):
    rows, columns = dimensions
    rng = np.random.default_rng(seed)
    matrix = rng.uniform(0.1, 1.0, size=(rows, columns))
    lower = rng.uniform(0.1, 0.3, size=columns)
    upper = rng.uniform(0.6, 1.2, size=columns)
    witness = lower + 0.4 * (upper - lower)
    target = target_scale * (matrix @ witness)
    reference = linprog(
        np.r_[np.zeros(columns), -1.0],
        A_eq=np.column_stack((matrix, -target)),
        b_eq=np.zeros(rows),
        bounds=list(zip(lower, upper)) + [(0.0, 1.0)],
        method="highs",
    )
    assert reference.success
    output, recovered = _solve(matrix, target, lower, upper, np.zeros(columns))
    assert recovered
    np.testing.assert_allclose(matrix @ output, reference.x[-1] * target, atol=1e-8)
    assert np.all(output >= lower - 1e-8)
    assert np.all(output <= upper + 1e-8)


def test_positive_ray_recovers_after_unreachable_target_with_reused_buffers():
    wp = pytest.importorskip("warp")
    pytest.importorskip("torch")
    from embodik.gpu.wbc._runtime.warp_bound_release import _make_kernel

    matrix = np.ones((2, 1, 2))
    lower, upper = np.full((2, 2), 0.2), np.ones((2, 2))
    original = np.zeros((2, 2))
    target = wp.zeros((2, 1), dtype=wp.float64, device="cpu")
    inputs = [wp.array(matrix, dtype=wp.float64, device="cpu"), target] + [
        wp.array(value, dtype=wp.float64, device="cpu") for value in (lower, upper, original)
    ]
    output = wp.empty((2, 2), dtype=wp.float64, device="cpu")
    recovered = wp.empty(2, dtype=wp.bool, device="cpu")
    for commands in ([-1.0, 1.0], [1.0, -1.0], [3.0, 1.0], [0.0, 1.0], [1.0, 1.0]):
        target.assign(np.asarray(commands)[:, None])
        wp.launch(
            _make_kernel(1, 2), dim=2, inputs=inputs, outputs=[output, recovered], device="cpu"
        )
        velocities, flags = output.numpy(), recovered.numpy()
        for world, command in enumerate(commands):
            assert bool(flags[world]) == (command > 0)
            if command <= 0:
                np.testing.assert_array_equal(velocities[world], original[world])
            else:
                np.testing.assert_allclose(velocities[world].sum(), min(command, 2.0), atol=1e-8)
                assert np.all(velocities[world] >= lower[world] - 1e-8)
                assert np.all(velocities[world] <= upper[world] + 1e-8)


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("dimensions", [(3, 5), (6, 12), (12, 18)])
def test_compensated_seed_with_redundant_tasks_and_locked_columns(seed, dimensions):
    rows, columns = dimensions
    rng = np.random.default_rng(seed)
    independent = rng.normal(size=(rows - 1, columns))
    independent[:, -1] = 0.0
    matrix = np.vstack((independent, 2.0 * independent[0]))
    target = rng.normal(size=rows)
    target[-1] = 2.0 * target[0]
    lower, upper = -np.ones(columns), np.ones(columns)
    lower[-1] = upper[-1] = 0.0
    if seed % 2:
        lower[0] = 0.01
    else:
        upper[0] = -0.01
    reference = linprog(
        np.r_[np.zeros(columns), -1.0],
        A_eq=np.column_stack((matrix, -target)),
        b_eq=np.zeros(rows),
        bounds=list(zip(lower, upper)) + [(0.0, 1.0)],
        method="highs",
    )
    assert reference.success
    output, recovered = _solve(matrix, target, lower, upper, np.zeros(columns))
    assert recovered
    np.testing.assert_allclose(matrix @ output, reference.x[-1] * target, atol=1e-8)
    assert np.all(output >= lower - 1e-8)
    assert np.all(output <= upper + 1e-8)
    assert output[-1] == pytest.approx(0.0, abs=1e-10)

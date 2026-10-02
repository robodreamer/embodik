"""Shared CasADi regularized inverse helper for experimental GPU solvers."""

from __future__ import annotations

try:
    import casadi as ca
except ImportError:
    ca = None

DEFAULT_JACOBI_SWEEPS = 5


def _round_robin_pairs(dimension: int) -> list[tuple[int, int]]:
    """Return a parallel-cyclic schedule containing every column pair once."""
    participants: list[int | None] = list(range(dimension))
    if dimension % 2:
        participants.append(None)

    pairs = []
    for _ in range(max(len(participants) - 1, 0)):
        for index in range(len(participants) // 2):
            first = participants[index]
            second = participants[-1 - index]
            if first is not None and second is not None:
                pairs.append((min(first, second), max(first, second)))
        if participants:
            participants = [
                participants[0],
                participants[-1],
                *participants[1:-1],
            ]
    return pairs


def _one_sided_jacobi_svd(
    matrix: "ca.SX",
    gram: "ca.SX",
    sweeps: int = DEFAULT_JACOBI_SWEEPS,
) -> tuple["ca.SX", "ca.SX", "ca.SX"]:
    """Return rotated columns, left vectors, and squared singular values.

    One-sided Jacobi orthogonalizes the columns of ``matrix.T``. Unlike an
    eigendecomposition of ``matrix @ matrix.T``, this does not square the
    condition number before resolving singular values around ``tol``. Cyclic
    rotations use a fixed sweep count so generated GPU kernels have no
    convergence synchronization or data-dependent loop bounds.
    """
    if ca is None:
        raise RuntimeError("CasADi is required")
    if sweeps <= 0:
        raise ValueError("Jacobi sweep count must be positive")

    task_dimension = matrix.size1()
    velocity_dimension = matrix.size2()
    rotated_columns = ca.SX(matrix.T)
    left_singular_vectors = ca.SX.eye(task_dimension)
    rotated_gram = ca.SX(gram)
    pair_order = _round_robin_pairs(task_dimension)

    for _ in range(sweeps):
        for row, column in pair_order:
            norm_row = rotated_gram[row, row]
            norm_column = rotated_gram[column, column]
            cross_product = rotated_gram[row, column]

            # The stable tangent form avoids cancellation for separated
            # singular values. Protect the division even when the conditional
            # selects the identity rotation because SX expressions can be
            # evaluated eagerly by downstream code generators.
            active = ca.fabs(cross_product) > 0.0
            safe_cross_product = ca.if_else(active, cross_product, 1.0)
            tau = (norm_column - norm_row) / (2.0 * safe_cross_product)
            tau_sign = ca.if_else(tau >= 0.0, 1.0, -1.0)
            tangent_candidate = tau_sign / (ca.fabs(tau) + ca.sqrt(1.0 + tau * tau))
            tangent = ca.if_else(active, tangent_candidate, 0.0)
            cosine = 1.0 / ca.sqrt(1.0 + tangent * tangent)
            sine = tangent * cosine

            for index in range(task_dimension):
                if index == row or index == column:
                    continue
                value_row = rotated_gram[index, row]
                value_column = rotated_gram[index, column]
                rotated_row = cosine * value_row - sine * value_column
                rotated_column = sine * value_row + cosine * value_column
                rotated_gram[index, row] = rotated_row
                rotated_gram[row, index] = rotated_row
                rotated_gram[index, column] = rotated_column
                rotated_gram[column, index] = rotated_column

            cosine_squared = cosine * cosine
            sine_squared = sine * sine
            cross = 2.0 * cosine * sine * cross_product
            rotated_gram[row, row] = cosine_squared * norm_row - cross + sine_squared * norm_column
            rotated_gram[column, column] = (
                sine_squared * norm_row + cross + cosine_squared * norm_column
            )
            rotated_gram[row, column] = 0.0
            rotated_gram[column, row] = 0.0

            for index in range(velocity_dimension):
                value_row = rotated_columns[index, row]
                value_column = rotated_columns[index, column]
                rotated_columns[index, row] = cosine * value_row - sine * value_column
                rotated_columns[index, column] = sine * value_row + cosine * value_column

            for index in range(task_dimension):
                value_row = left_singular_vectors[index, row]
                value_column = left_singular_vectors[index, column]
                left_singular_vectors[index, row] = cosine * value_row - sine * value_column
                left_singular_vectors[index, column] = sine * value_row + cosine * value_column

    squared_singular_values = ca.vertcat(
        *[
            ca.dot(rotated_columns[:, index], rotated_columns[:, index])
            for index in range(task_dimension)
        ]
    )
    return rotated_columns, left_singular_vectors, squared_singular_values


def srinv(
    A: "ca.SX",
    tol: float,
    damping: float,
) -> "ca.SX":
    """Return the extended singularity-robust inverse used by the CPU solver.

    The CPU implementation obtains left singular vectors with an SVD. This GPU
    expression obtains the equivalent task-space spectral basis with fixed
    one-sided Jacobi sweeps and applies the same smooth per-singular-value
    damping law. Healthy directions therefore remain undamped while only
    values below ``tol`` receive directional damping.
    """
    if ca is None:
        raise RuntimeError("CasADi is required")

    aat = A @ A.T
    m = aat.size1()
    threshold_sq = tol * tol

    det_val = ca.det(aat)
    global_regularization = ca.if_else(
        det_val < threshold_sq,
        (1.0 - (det_val / threshold_sq) ** 2) * threshold_sq,
        0.0,
    )
    rotated_columns, left_singular_vectors, squared_singular_values = _one_sided_jacobi_svd(A, aat)
    singular_values = ca.sqrt(ca.fmax(squared_singular_values, 0.0))
    normalized_values = ca.fmin(singular_values / tol, 1.0)
    per_value_damping = damping * ca.fmax(
        1.0 - normalized_values * normalized_values,
        0.0,
    )

    inverse_denominators = squared_singular_values + global_regularization + per_value_damping
    inverse_spectrum = ca.diag(1.0 / inverse_denominators)
    return rotated_columns @ inverse_spectrum @ left_singular_vectors.T


def undamped_jacobi_pinv(
    matrix: "ca.SX",
    rank_tolerance: float = 1e-6,
) -> "ca.SX":
    """Moore-Penrose inverse that drops singular values below the ESNS cutoff.

    A consistent zero row makes ``det(J J.T)`` zero. The extended SRINV then
    regularizes every direction. This inverse keeps the independent directions
    undamped and is used only for that redundant-row completion.
    """
    if ca is None:
        raise RuntimeError("CasADi is required")
    gram = matrix @ matrix.T
    rotated_columns, left_singular_vectors, squared_singular_values = _one_sided_jacobi_svd(
        matrix, gram
    )
    singular_values = ca.sqrt(ca.fmax(squared_singular_values, 0.0))
    largest = singular_values[0]
    for index in range(1, singular_values.numel()):
        largest = ca.fmax(largest, singular_values[index])
    coefficients = []
    for index in range(singular_values.numel()):
        retained = singular_values[index] > rank_tolerance * largest
        safe_squared = ca.fmax(squared_singular_values[index], 1e-30)
        coefficients.append(ca.if_else(retained, 1.0 / safe_squared, 0.0))
    return rotated_columns @ ca.diag(ca.vertcat(*coefficients)) @ left_singular_vectors.T


def _jacobi_singular_values(matrix: "ca.SX") -> "ca.SX":
    gram = matrix @ matrix.T
    _, _, squared_singular_values = _one_sided_jacobi_svd(matrix, gram)
    return ca.sqrt(ca.fmax(squared_singular_values, 0.0))


def _numerical_rank(singular_values: "ca.SX", rank_tolerance: float) -> "ca.SX":
    largest = singular_values[0]
    for index in range(1, singular_values.numel()):
        largest = ca.fmax(largest, singular_values[index])
    rank = ca.SX.zeros(1)
    for index in range(singular_values.numel()):
        rank = rank + ca.if_else(
            singular_values[index] > rank_tolerance * largest,
            1.0,
            0.0,
        )
    return rank


def complete_consistent_redundant_velocity(
    jacobian: "ca.SX",
    target: "ca.SX",
    current: "ca.SX",
    full_scale_step: "ca.SX",
    coefficients: "ca.SX",
    lower: "ca.SX",
    upper: "ca.SX",
    *,
    rank_tolerance: float = 1e-6,
    bound_tolerance: float = 1e-6,
    residual_tolerance: float = 1e-8,
    projector: "ca.SX | None" = None,
) -> tuple["ca.SX", "ca.SX"]:
    """Saturate every full-scale violation of one consistent repeated task.

    Returns the completed velocity and a scalar acceptance flag. The flag stays
    zero for a full-rank task, a structural singularity, or an inconsistent
    extra row, so those tasks keep uniform bound scaling. ``projector`` is the
    nullspace of higher-priority tasks; the correction stays in that subspace.
    """
    if ca is None:
        raise RuntimeError("CasADi is required")

    rows = jacobian.size1()
    jacobian_scale = ca.fmax(1.0, ca.norm_fro(jacobian))
    redundant = []
    for row in range(rows):
        row_vector = jacobian[row, :]
        row_norm = ca.norm_2(row_vector)
        is_zero = row_norm <= rank_tolerance * jacobian_scale
        duplicate = ca.SX.zeros(1)
        for earlier in range(row):
            earlier_vector = jacobian[earlier, :]
            earlier_norm = ca.norm_2(earlier_vector)
            earlier_kept = earlier_norm > rank_tolerance * jacobian_scale
            alignment = ca.dot(row_vector, earlier_vector)
            parallel_gap = ca.sqrt(
                ca.fmax(
                    0.0,
                    row_norm * row_norm * earlier_norm * earlier_norm - alignment * alignment,
                )
            )
            parallel = parallel_gap <= rank_tolerance * row_norm * earlier_norm
            duplicate = ca.if_else(earlier_kept * parallel, 1.0, duplicate)
        redundant.append(ca.if_else(is_zero, 1.0, duplicate))
    redundant_count = ca.sum1(ca.vertcat(*redundant))
    task_rank = _numerical_rank(_jacobi_singular_values(jacobian), rank_tolerance)
    augmented = ca.horzcat(jacobian, target)
    augmented_rank = _numerical_rank(_jacobi_singular_values(augmented), rank_tolerance)
    explicit_consistent = (
        (redundant_count > 0.5)
        * (ca.fabs(task_rank + redundant_count - rows) < 0.5)
        * (ca.fabs(augmented_rank - task_rank) < 0.5)
    )

    full_velocity = current + full_scale_step
    constraint_value = coefficients @ full_velocity
    degrees_of_freedom = jacobian.size2()
    if projector is None:
        projector = ca.SX.eye(degrees_of_freedom)
    current_constraint = coefficients @ current
    masks = []
    constraint_delta = []
    for index in range(coefficients.size1()):
        value = constraint_value[index]
        violated = ca.if_else(
            (value < lower[index] - bound_tolerance) + (value > upper[index] + bound_tolerance),
            1.0,
            0.0,
        )
        # A zero-width bound is already saturated. Leaving it free lets the
        # re-solve move a locked joint.
        fixed = ca.if_else(upper[index] - lower[index] <= bound_tolerance, 1.0, 0.0)
        active = ca.fmax(violated, fixed)
        bound = ca.fmin(upper[index], ca.fmax(lower[index], value))
        masks.append(active)
        constraint_delta.append(active * (bound - current_constraint[index]))
    mask = ca.vertcat(*masks)
    masked_constraints = ca.diag(mask) @ coefficients @ projector
    constraint_inverse = undamped_jacobi_pinv(masked_constraints, rank_tolerance)
    particular = constraint_inverse @ ca.vertcat(*constraint_delta)
    nullspace = ca.SX.eye(degrees_of_freedom) - constraint_inverse @ masked_constraints
    reduced = jacobian @ projector @ nullspace
    task_residual = target - jacobian @ current - jacobian @ projector @ particular
    correction = nullspace @ (undamped_jacobi_pinv(reduced, rank_tolerance) @ task_residual)
    candidate = current + projector @ (particular + correction)
    achieved = coefficients @ candidate
    within_bounds = ca.SX.ones(1)
    for index in range(coefficients.size1()):
        within_bounds = within_bounds * ca.if_else(
            (achieved[index] >= lower[index] - bound_tolerance)
            * (achieved[index] <= upper[index] + bound_tolerance),
            1.0,
            0.0,
        )
    residual_norm = ca.norm_2(jacobian @ candidate - target)
    target_scale = ca.fmax(1.0, ca.norm_2(target))
    reachable = residual_norm <= residual_tolerance * target_scale
    accepted = explicit_consistent * reachable * within_bounds * (ca.sum1(mask) > 0.5)
    return candidate, accepted

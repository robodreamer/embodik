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

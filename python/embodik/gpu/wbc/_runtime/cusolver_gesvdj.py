"""Direct graph-capture experiment for cuSOLVER's small batched Jacobi SVD."""

from __future__ import annotations

import ctypes
from pathlib import Path

import torch

_SUCCESS = 0
_EIG_MODE_VECTOR = 1


def _check(status: int, operation: str) -> None:
    if status != _SUCCESS:
        raise RuntimeError(f"{operation} failed with cuSOLVER status {status}")


def _library_path() -> Path:
    path = Path(torch.__file__).resolve().parent.parent / "nvidia/cusolver/lib/libcusolver.so.11"
    if not path.is_file():
        raise RuntimeError(f"the PyTorch cuSOLVER library was not found at {path}")
    return path


class CuSolverGesvdjBatched:
    """Reusable float32 SVD factors for matrices with rows <= columns <= 32.

    A contiguous row-major ``[B, rows, columns]`` input has the same memory
    layout as column-major ``J.T``. cuSOLVER therefore factors ``J.T`` without
    a transpose. ``left_transpose`` stores ``U.T`` for ``J.T`` and
    ``right_transpose`` stores ``V.T``; together they reconstruct the original
    Jacobian as ``right_transpose.T @ diag(S) @ left_transpose[:, :rows]``.
    """

    def __init__(
        self,
        rows: int,
        columns: int,
        *,
        batch_capacity: int = 1,
        tolerance: float = 1.0e-7,
        max_sweeps: int = 100,
        device: str | torch.device = "cuda",
    ) -> None:
        if type(rows) is not int or type(columns) is not int:
            raise TypeError("rows and columns must be integers")
        if not 1 <= rows <= columns <= 32:
            raise ValueError("gesvdjBatched requires 1 <= rows <= columns <= 32")
        if type(batch_capacity) is not int or batch_capacity < 1:
            raise ValueError("batch_capacity must be a positive integer")
        if tolerance <= 0.0 or max_sweeps < 1:
            raise ValueError("tolerance and max_sweeps must be positive")
        self.rows = rows
        self.columns = columns
        self.batch_capacity = batch_capacity
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("device must be CUDA")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())

        self._library = ctypes.CDLL(str(_library_path()))
        self._configure_signatures()
        self._handle = ctypes.c_void_p()
        self._params = ctypes.c_void_p()
        _check(
            self._library.cusolverDnCreate(ctypes.byref(self._handle)),
            "cusolverDnCreate",
        )
        try:
            _check(
                self._library.cusolverDnCreateGesvdjInfo(ctypes.byref(self._params)),
                "cusolverDnCreateGesvdjInfo",
            )
            _check(
                self._library.cusolverDnXgesvdjSetTolerance(
                    self._params, ctypes.c_double(tolerance)
                ),
                "cusolverDnXgesvdjSetTolerance",
            )
            _check(
                self._library.cusolverDnXgesvdjSetMaxSweeps(self._params, max_sweeps),
                "cusolverDnXgesvdjSetMaxSweeps",
            )
            _check(
                self._library.cusolverDnXgesvdjSetSortEig(self._params, 1),
                "cusolverDnXgesvdjSetSortEig",
            )
            self._allocate()
        except Exception:
            self.close()
            raise

    def _configure_signatures(self) -> None:
        lib = self._library
        pointer = ctypes.c_void_p
        lib.cusolverDnCreate.argtypes = [ctypes.POINTER(pointer)]
        lib.cusolverDnCreate.restype = ctypes.c_int
        lib.cusolverDnDestroy.argtypes = [pointer]
        lib.cusolverDnDestroy.restype = ctypes.c_int
        lib.cusolverDnSetStream.argtypes = [pointer, pointer]
        lib.cusolverDnSetStream.restype = ctypes.c_int
        lib.cusolverDnCreateGesvdjInfo.argtypes = [ctypes.POINTER(pointer)]
        lib.cusolverDnCreateGesvdjInfo.restype = ctypes.c_int
        lib.cusolverDnDestroyGesvdjInfo.argtypes = [pointer]
        lib.cusolverDnDestroyGesvdjInfo.restype = ctypes.c_int
        lib.cusolverDnXgesvdjSetTolerance.argtypes = [pointer, ctypes.c_double]
        lib.cusolverDnXgesvdjSetTolerance.restype = ctypes.c_int
        lib.cusolverDnXgesvdjSetMaxSweeps.argtypes = [pointer, ctypes.c_int]
        lib.cusolverDnXgesvdjSetMaxSweeps.restype = ctypes.c_int
        lib.cusolverDnXgesvdjSetSortEig.argtypes = [pointer, ctypes.c_int]
        lib.cusolverDnXgesvdjSetSortEig.restype = ctypes.c_int
        buffer_arguments = [
            pointer,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            pointer,
            ctypes.c_int,
            pointer,
            pointer,
            ctypes.c_int,
            pointer,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            pointer,
            ctypes.c_int,
        ]
        lib.cusolverDnSgesvdjBatched_bufferSize.argtypes = buffer_arguments
        lib.cusolverDnSgesvdjBatched_bufferSize.restype = ctypes.c_int
        solve_arguments = buffer_arguments[:-3] + [
            pointer,
            ctypes.c_int,
            pointer,
            pointer,
            ctypes.c_int,
        ]
        lib.cusolverDnSgesvdjBatched.argtypes = solve_arguments
        lib.cusolverDnSgesvdjBatched.restype = ctypes.c_int

    @staticmethod
    def _pointer(tensor: torch.Tensor) -> ctypes.c_void_p:
        return ctypes.c_void_p(tensor.data_ptr())

    def _allocate(self) -> None:
        shape = (self.batch_capacity, self.rows, self.columns)
        self.matrix = torch.empty(shape, dtype=torch.float32, device=self.device)
        self.singular_values = torch.empty(
            (self.batch_capacity, self.rows),
            dtype=torch.float32,
            device=self.device,
        )
        # gesvdjBatched writes a full m-by-m U even though only its first n
        # columns are needed. Row-major storage therefore exposes full U.T.
        self.left_transpose = torch.empty(
            (self.batch_capacity, self.columns, self.columns),
            dtype=torch.float32,
            device=self.device,
        )
        self.right_transpose = torch.empty(
            (self.batch_capacity, self.rows, self.rows),
            dtype=torch.float32,
            device=self.device,
        )
        self.info = torch.empty(self.batch_capacity, dtype=torch.int32, device=self.device)
        self._set_stream()
        workspace_size = ctypes.c_int()
        _check(
            self._library.cusolverDnSgesvdjBatched_bufferSize(
                self._handle,
                _EIG_MODE_VECTOR,
                self.columns,
                self.rows,
                self._pointer(self.matrix),
                self.columns,
                self._pointer(self.singular_values),
                self._pointer(self.left_transpose),
                self.columns,
                self._pointer(self.right_transpose),
                self.rows,
                ctypes.byref(workspace_size),
                self._params,
                self.batch_capacity,
            ),
            "cusolverDnSgesvdjBatched_bufferSize",
        )
        self.workspace_size = workspace_size.value
        self.workspace = torch.empty(self.workspace_size, dtype=torch.float32, device=self.device)

    def _set_stream(self) -> None:
        stream = torch.cuda.current_stream(self.device)
        _check(
            self._library.cusolverDnSetStream(self._handle, ctypes.c_void_p(stream.cuda_stream)),
            "cusolverDnSetStream",
        )

    def factor(
        self, matrix: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        expected = (self.batch_capacity, self.rows, self.columns)
        if matrix.shape != expected:
            raise ValueError(f"matrix must have shape {expected}")
        if (
            matrix.dtype is not torch.float32
            or matrix.device != self.device
            or not matrix.is_contiguous()
        ):
            raise ValueError("matrix must be contiguous CUDA float32")
        self.matrix.copy_(matrix)
        self._set_stream()
        _check(
            self._library.cusolverDnSgesvdjBatched(
                self._handle,
                _EIG_MODE_VECTOR,
                self.columns,
                self.rows,
                self._pointer(self.matrix),
                self.columns,
                self._pointer(self.singular_values),
                self._pointer(self.left_transpose),
                self.columns,
                self._pointer(self.right_transpose),
                self.rows,
                self._pointer(self.workspace),
                self.workspace_size,
                self._pointer(self.info),
                self._params,
                self.batch_capacity,
            ),
            "cusolverDnSgesvdjBatched",
        )
        return (
            self.left_transpose,
            self.singular_values,
            self.right_transpose,
            self.info,
        )

    def close(self) -> None:
        params = getattr(self, "_params", None)
        if params and params.value:
            self._library.cusolverDnDestroyGesvdjInfo(params)
            params.value = None
        handle = getattr(self, "_handle", None)
        if handle and handle.value:
            self._library.cusolverDnDestroy(handle)
            handle.value = None

    def __enter__(self) -> CuSolverGesvdjBatched:
        return self

    def __exit__(self, *_args) -> None:
        self.close()


class CuSolverDirectionalSRINV:
    """Graph-capturable faithful extended-SRINV action using direct Jacobi SVD."""

    def __init__(
        self,
        rows: int,
        columns: int,
        *,
        batch_capacity: int = 1,
        tolerance: float = 0.1,
        damping: float = 0.1,
        relative_rank_tolerance: float = 1.0e-6,
        output_mode: str = "full",
        fused_status_enabled: bool = False,
        profile_label: str | None = None,
        device: str | torch.device = "cuda",
    ) -> None:
        if tolerance <= 0.0 or damping < 0.0:
            raise ValueError("SRINV tolerance must be positive and damping nonnegative")
        if relative_rank_tolerance < 0.0:
            raise ValueError("relative rank tolerance must be nonnegative")
        if batch_capacity != 1:
            raise ValueError("the direct-LU determinant prototype currently supports B1")
        if output_mode not in {"full", "action_undamped", "action"}:
            raise ValueError("output_mode must be full, action_undamped, or action")
        self.rows = rows
        self.columns = columns
        self.batch_capacity = batch_capacity
        self.tolerance = float(tolerance)
        self.damping = float(damping)
        self.relative_rank_tolerance = float(relative_rank_tolerance)
        self.output_mode = output_mode
        self.profile_label = profile_label or f"srinv_{rows}x{columns}"
        self._compiled_status = (
            torch.compile(
                self._status_eager,
                fullgraph=True,
                dynamic=False,
                options={"triton.cudagraphs": False},
            )
            if fused_status_enabled
            else None
        )
        self._transpose_input = rows > columns
        factor_rows, factor_columns = (columns, rows) if self._transpose_input else (rows, columns)
        self.factorization = CuSolverGesvdjBatched(
            factor_rows,
            factor_columns,
            batch_capacity=batch_capacity,
            device=device,
        )
        if not self._transpose_input:
            self._configure_lu()

    def _configure_lu(self) -> None:
        lib = self.factorization._library
        pointer = ctypes.c_void_p
        lib.cusolverDnDgetrf_bufferSize.argtypes = [
            pointer,
            ctypes.c_int,
            ctypes.c_int,
            pointer,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
        ]
        lib.cusolverDnDgetrf_bufferSize.restype = ctypes.c_int
        lib.cusolverDnDgetrf.argtypes = [
            pointer,
            ctypes.c_int,
            ctypes.c_int,
            pointer,
            ctypes.c_int,
            pointer,
            pointer,
            pointer,
        ]
        lib.cusolverDnDgetrf.restype = ctypes.c_int
        device = self.factorization.device
        self._gram_lu = torch.empty((self.rows, self.rows), dtype=torch.float64, device=device)
        self._lu_pivots = torch.empty(self.rows, dtype=torch.int32, device=device)
        self._lu_info = torch.empty(1, dtype=torch.int32, device=device)
        workspace_size = ctypes.c_int()
        _check(
            lib.cusolverDnDgetrf_bufferSize(
                self.factorization._handle,
                self.rows,
                self.rows,
                self.factorization._pointer(self._gram_lu),
                self.rows,
                ctypes.byref(workspace_size),
            ),
            "cusolverDnDgetrf_bufferSize",
        )
        self._lu_workspace = torch.empty(workspace_size.value, dtype=torch.float64, device=device)
        self._pivot_reference = torch.arange(1, self.rows + 1, dtype=torch.int32, device=device)

    def _gram_determinant(self, matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self._transpose_input:
            # A tall J has rank(J J.T) <= columns < rows, hence its Gram
            # determinant is exactly zero. Preserve EmbodiK's determinant
            # branch directly and avoid factoring a larger singular matrix.
            return (
                torch.zeros(1, dtype=torch.float64, device=matrix.device),
                torch.zeros(1, dtype=torch.int32, device=matrix.device),
            )
        matrix64 = matrix.to(torch.float64)
        gram = matrix64 @ matrix64.transpose(-2, -1)
        self._gram_lu.copy_(gram[0])
        self.factorization._set_stream()
        _check(
            self.factorization._library.cusolverDnDgetrf(
                self.factorization._handle,
                self.rows,
                self.rows,
                self.factorization._pointer(self._gram_lu),
                self.rows,
                self.factorization._pointer(self._lu_workspace),
                self.factorization._pointer(self._lu_pivots),
                self.factorization._pointer(self._lu_info),
            ),
            "cusolverDnDgetrf",
        )
        swaps = (self._lu_pivots != self._pivot_reference).sum()
        sign = torch.where(
            swaps.remainder(2) == 0,
            torch.ones((), dtype=torch.float64, device=matrix.device),
            -torch.ones((), dtype=torch.float64, device=matrix.device),
        )
        value = sign * torch.diagonal(self._gram_lu).prod()
        # Positive getrf info identifies an exactly singular pivot. That is a
        # valid zero determinant for extended SRINV, not a solver failure.
        status = torch.where(
            self._lu_info < 0,
            torch.ones_like(self._lu_info),
            torch.zeros_like(self._lu_info),
        )
        return value.reshape(1), status

    def _directional_spectrum_eager(
        self, determinant: torch.Tensor, singular_values: torch.Tensor
    ) -> torch.Tensor:
        threshold_squared = self.tolerance * self.tolerance
        determinant32 = determinant.to(torch.float32)
        global_regularization = torch.where(
            determinant32 < threshold_squared,
            (1.0 - (determinant32 / threshold_squared).square()) * threshold_squared,
            torch.zeros_like(determinant32),
        )
        normalized = torch.clamp(singular_values / self.tolerance, max=1.0)
        per_value_damping = self.damping * torch.clamp(1.0 - normalized.square(), min=0.0)
        directional_spectrum = singular_values / (
            singular_values.square() + global_regularization[:, None] + per_value_damping
        )
        return directional_spectrum

    def _spectra_eager(
        self, determinant: torch.Tensor, singular_values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        directional_spectrum = self._directional_spectrum_eager(determinant, singular_values)
        maximum = singular_values.amax(dim=-1, keepdim=True)
        retained = singular_values > self.relative_rank_tolerance * maximum
        safe = torch.where(retained, singular_values, torch.ones_like(singular_values))
        undamped_spectrum = torch.where(
            retained, safe.reciprocal(), torch.zeros_like(singular_values)
        )
        return directional_spectrum, undamped_spectrum

    @staticmethod
    def _status_eager(
        solution: torch.Tensor,
        factor_info: torch.Tensor,
        determinant_status: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        status = torch.where(
            (factor_info == 0) & (determinant_status == 0) & torch.isfinite(solution).all(dim=-1),
            torch.zeros_like(factor_info),
            torch.ones_like(factor_info),
        )
        return (
            torch.where(status[:, None] == 0, solution, torch.zeros_like(solution)),
            status,
        )

    def solve(self, matrix: torch.Tensor, rhs: torch.Tensor) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor,
    ]:
        if tuple(rhs.shape) != (self.batch_capacity, self.rows):
            raise ValueError(f"rhs must have shape {(self.batch_capacity, self.rows)}")
        if (
            rhs.dtype is not torch.float32
            or rhs.device != self.factorization.device
            or not rhs.is_contiguous()
        ):
            raise ValueError("rhs must be contiguous CUDA float32")
        factor_input = matrix.transpose(-2, -1).contiguous() if self._transpose_input else matrix
        left_transpose, singular_values, right_transpose, factor_info = self.factorization.factor(
            factor_input
        )
        determinant, determinant_status = self._gram_determinant(matrix)
        if self.output_mode == "action":
            directional_spectrum = self._directional_spectrum_eager(determinant, singular_values)
            undamped_spectrum = None
        else:
            directional_spectrum, undamped_spectrum = self._spectra_eager(
                determinant, singular_values
            )
        if self._transpose_input:
            # factor_input is J.T. Its right basis is J's left basis and its
            # left basis is J's right basis, so transpose the wide-input
            # reconstruction formula to obtain J's [columns, rows] inverse.
            right_basis = right_transpose.transpose(-2, -1)
            left_basis_transpose = left_transpose[:, : self.columns]
            inverse_basis = left_basis_transpose
        else:
            right_basis = left_transpose[:, : self.rows].transpose(-2, -1)
            inverse_basis = right_transpose

        inverse = (right_basis * directional_spectrum[:, None, :]) @ inverse_basis

        undamped_inverse = None
        if self.output_mode != "action":
            assert undamped_spectrum is not None
            undamped_inverse = (right_basis * undamped_spectrum[:, None, :]) @ inverse_basis
        solution = (inverse @ rhs.unsqueeze(-1)).squeeze(-1)
        status_fn = self._compiled_status or self._status_eager
        solution, status = status_fn(solution, factor_info, determinant_status)
        return (
            solution,
            inverse if self.output_mode == "full" else None,
            undamped_inverse,
            status,
        )

    def close(self) -> None:
        self.factorization.close()

    def __enter__(self) -> CuSolverDirectionalSRINV:
        return self

    def __exit__(self, *_args) -> None:
        self.close()

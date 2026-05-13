"""
GPU-accelerated batched velocity IK solver using CusADi.

This module provides GPU-parallel velocity solving for batched inputs,
using CusADi to compile CasADi symbolic functions to CUDA kernels.
Uses FI-PeSNS (Fixed-Iteration Penalized eSNS) - a GPU-optimized solver.

Setup (one-time):
    1. Clone and install cusadi: git clone https://github.com/se-hwan/cusadi && pip install -e cusadi
    2. Export the CasADi velocity solve function:
       python -m embodik.gpu.export_casadi_velocity_solve --robot panda
    3. Copy to cusadi and compile:
       mkdir -p cusadi/src/casadi_functions
       cp build/casadi/fn_velocity_solve.casadi cusadi/src/casadi_functions/
       cd cusadi && python run_codegen.py --fn=fn_velocity_solve

Benchmark expectations (based on dhb_xr results):
    Batch 100:   ~40x speedup (CPU loop vs GPU parallel)
    Batch 1000:  ~200x speedup
    Batch 2000:  ~400x speedup
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Check for CasADi
try:
    import casadi as ca

    HAS_CASADI = True
except ImportError:
    HAS_CASADI = False
    ca = None

# Check for CusADi (GPU acceleration)
HAS_CUSADI = False
CusadiFunction = None
try:
    from cusadi import CusadiFunction

    HAS_CUSADI = True
except ImportError:
    try:
        import sys

        cusadi_root = os.environ.get("CUSADI_ROOT", "")
        if cusadi_root and cusadi_root not in sys.path:
            sys.path.insert(0, cusadi_root)
        from src.CusadiFunction import CusadiFunction

        HAS_CUSADI = True
    except ImportError:
        pass

# Check for PyTorch CUDA
HAS_TORCH_CUDA = False
torch = None
try:
    import torch as _torch

    torch = _torch
    HAS_TORCH_CUDA = torch.cuda.is_available()
except ImportError:
    pass


# Default paths - search CUSADI_ROOT env var or common locations
def _find_cusadi_root() -> str:
    """Find CusADi root directory."""
    # Check environment variable first
    cusadi_root = os.environ.get("CUSADI_ROOT", "")
    if cusadi_root and os.path.isdir(cusadi_root):
        return cusadi_root

    # Check common locations
    home = os.path.expanduser("~")
    candidates = [
        os.path.join(home, ".local", "cusadi"),
        os.path.join(home, "cusadi"),
        "/opt/cusadi",
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate

    return ""


CUSADI_ROOT = _find_cusadi_root()
DEFAULT_FN_PATH = (
    os.path.join(CUSADI_ROOT, "src/casadi_functions/fn_velocity_solve.casadi")
    if CUSADI_ROOT
    else ""
)

# Global cache for CusADi functions
_cusadi_fn_cache: Dict[Tuple[str, int], Any] = {}


@dataclass
class BatchSolveResult:
    """Result from batched velocity solve."""

    velocities: np.ndarray  # (B, n_dof)
    scales: np.ndarray  # (B, n_tasks)
    status: str  # "success", "fallback_cpu", "error"
    elapsed_ms: float = 0.0


def get_cusadi_function(
    casadi_path: str,
    batch_size: int,
) -> Optional[Any]:
    """
    Get or create a cached CusADi function for GPU velocity solve.

    Args:
        casadi_path: Path to the .casadi function file
        batch_size: Batch size for GPU execution

    Returns:
        CusadiFunction if available, None otherwise
    """
    if not HAS_CUSADI or not HAS_CASADI:
        return None

    cache_key = (casadi_path, batch_size)
    if cache_key in _cusadi_fn_cache:
        return _cusadi_fn_cache[cache_key]

    if not os.path.exists(casadi_path):
        return None

    try:
        fn = ca.Function.load(casadi_path)
        cusadi_fn = CusadiFunction(fn, batch_size)
        _cusadi_fn_cache[cache_key] = cusadi_fn
        return cusadi_fn
    except Exception as e:
        print(f"Warning: Failed to load CusADi function: {e}")
        return None


def _solve_cpu_sequential(
    objective_targets_batch: List[np.ndarray],
    objective_jacobians_batch: List[np.ndarray],
    constraint_matrices_batch: List[np.ndarray],
    min_bounds_batch: List[np.ndarray],
    max_bounds_batch: List[np.ndarray],
) -> BatchSolveResult:
    """
    CPU fallback: solve each problem sequentially using EmbodiK's solver.

    Args:
        All inputs are lists of length B (batch size)

    Returns:
        BatchSolveResult with CPU solutions
    """
    import time

    start = time.perf_counter()

    try:
        import embodik as eik
    except ImportError:
        raise RuntimeError(
            "EmbodiK not installed. Install with: python -m pip install --only-binary=:all: embodik"
        )

    B = len(objective_targets_batch)
    velocities_list = []
    scales_list = []

    for b in range(B):
        # Convert to list format expected by EmbodiK
        targets = objective_targets_batch[b]
        jacobians = objective_jacobians_batch[b]
        C = constraint_matrices_batch[b]
        lower = min_bounds_batch[b]
        upper = max_bounds_batch[b]

        # Ensure C is 2D and F-order
        if C.ndim == 1:
            # Assume square constraint matrix
            n = int(np.sqrt(C.shape[0]))
            C = C.reshape(n, n)
        C = np.asfortranarray(C.astype(np.float64))

        # Get n_dof from constraint matrix
        n_dof = C.shape[1]

        # Call EmbodiK solver
        # Note: targets and jacobians may be single arrays for single-task case
        if isinstance(targets, np.ndarray) and targets.ndim == 1:
            targets_list = [targets.astype(np.float64)]
            # Reshape jacobians to 2D if flattened
            if jacobians.ndim == 1:
                task_dim = targets.shape[0]
                jacobians = jacobians.reshape(task_dim, n_dof)
            jacobians_list = [np.asfortranarray(jacobians.astype(np.float64))]
        else:
            targets_list = [t.astype(np.float64) for t in targets]
            jacobians_list = [np.asfortranarray(j.astype(np.float64)) for j in jacobians]

        result = eik.computeMultiObjectiveVelocitySolutionEigen(
            targets_list, jacobians_list, C, lower.astype(np.float64), upper.astype(np.float64)
        )

        velocities_list.append(np.array(result.solution))
        scales_list.append(np.array(result.task_scales))

    elapsed_ms = (time.perf_counter() - start) * 1000

    return BatchSolveResult(
        velocities=np.stack(velocities_list),
        scales=np.stack(scales_list),
        status="fallback_cpu",
        elapsed_ms=elapsed_ms,
    )


def solve_velocity_gpu_batched(
    objective_targets_batch: Union[np.ndarray, List[np.ndarray]],
    objective_jacobians_batch: Union[np.ndarray, List[np.ndarray]],
    constraint_matrices_batch: Union[np.ndarray, List[np.ndarray]],
    min_bounds_batch: Union[np.ndarray, List[np.ndarray]],
    max_bounds_batch: Union[np.ndarray, List[np.ndarray]],
    casadi_path: str = DEFAULT_FN_PATH,
) -> BatchSolveResult:
    """
    GPU-accelerated batched velocity solve using CusADi.

    Args:
        objective_targets_batch: (B, total_task_dim) or list of B arrays
        objective_jacobians_batch: (B, total_jacobian_size) or list of B arrays
        constraint_matrices_batch: (B, n_constraints, n_dof) or list of B matrices
        min_bounds_batch: (B, n_constraints) or list of B arrays
        max_bounds_batch: (B, n_constraints) or list of B arrays
        casadi_path: Path to compiled CasADi function

    Returns:
        BatchSolveResult with velocities (B, n_dof) and scales (B, n_tasks)

    Raises:
        RuntimeError: If GPU is requested but not available
    """
    import time

    start = time.perf_counter()

    if not HAS_TORCH_CUDA:
        raise RuntimeError("CUDA not available. Install PyTorch with CUDA support.")
    if not HAS_CUSADI:
        raise RuntimeError("CusADi not installed. Clone and install from github.com/se-hwan/cusadi")

    # Convert to numpy arrays if needed
    if isinstance(objective_targets_batch, list):
        targets = np.stack(objective_targets_batch)
    else:
        targets = objective_targets_batch

    if isinstance(objective_jacobians_batch, list):
        jacobians = np.stack(objective_jacobians_batch)
    else:
        jacobians = objective_jacobians_batch

    if isinstance(constraint_matrices_batch, list):
        C = np.stack(constraint_matrices_batch)
    else:
        C = constraint_matrices_batch

    if isinstance(min_bounds_batch, list):
        lower = np.stack(min_bounds_batch)
    else:
        lower = min_bounds_batch

    if isinstance(max_bounds_batch, list):
        upper = np.stack(max_bounds_batch)
    else:
        upper = max_bounds_batch

    B = targets.shape[0]

    # Get or create CusADi function
    cusadi_fn = get_cusadi_function(casadi_path, B)
    if cusadi_fn is None:
        raise RuntimeError(f"Could not load CusADi function from {casadi_path}")

    # CusADi expects C as 2D matrix per sample, not flattened
    # If C is (B, n, n), keep it. If C is (B, n*n), reshape it back
    if C.ndim == 2:
        # Assume square constraint matrix, reshape to (B, n, n)
        n = int(np.sqrt(C.shape[1]))
        if n * n == C.shape[1]:
            C = C.reshape(B, n, n)

    # Convert to torch tensors on GPU (must be contiguous float64)
    targets_t = torch.from_numpy(targets.astype(np.float64)).cuda().contiguous()
    jacobians_t = torch.from_numpy(jacobians.astype(np.float64)).cuda().contiguous()
    C_t = torch.from_numpy(C.astype(np.float64)).cuda().contiguous()
    lower_t = torch.from_numpy(lower.astype(np.float64)).cuda().contiguous()
    upper_t = torch.from_numpy(upper.astype(np.float64)).cuda().contiguous()

    # Run CusADi function
    cusadi_fn.evaluate([targets_t, jacobians_t, C_t, lower_t, upper_t])

    # Get outputs and squeeze extra dimensions
    velocities = cusadi_fn.getDenseOutput(0).cpu().numpy()  # (B, n_dof) or (B, n_dof, 1)
    scales = cusadi_fn.getDenseOutput(1).cpu().numpy()  # (B, n_tasks) or (B, n_tasks, 1)

    # Remove trailing dimensions if present
    if velocities.ndim == 3 and velocities.shape[-1] == 1:
        velocities = velocities.squeeze(-1)
    if scales.ndim == 3 and scales.shape[-1] == 1:
        scales = scales.squeeze(-1)

    elapsed_ms = (time.perf_counter() - start) * 1000

    return BatchSolveResult(
        velocities=velocities,
        scales=scales,
        status="success",
        elapsed_ms=elapsed_ms,
    )


def solve_velocity_batched(
    objective_targets_batch: Union[np.ndarray, List[np.ndarray]],
    objective_jacobians_batch: Union[np.ndarray, List[np.ndarray]],
    constraint_matrices_batch: Union[np.ndarray, List[np.ndarray]],
    min_bounds_batch: Union[np.ndarray, List[np.ndarray]],
    max_bounds_batch: Union[np.ndarray, List[np.ndarray]],
    use_gpu: bool = True,
    casadi_path: str = DEFAULT_FN_PATH,
) -> BatchSolveResult:
    """
    Batched velocity solve with automatic GPU/CPU fallback.

    This is the recommended API for batched solving. It will:
    1. Try GPU (CusADi) if use_gpu=True and CUDA is available
    2. Fall back to CPU sequential solve if GPU fails or is unavailable

    Args:
        objective_targets_batch: Batch of objective targets
        objective_jacobians_batch: Batch of task Jacobians
        constraint_matrices_batch: Batch of constraint matrices
        min_bounds_batch: Batch of lower bounds
        max_bounds_batch: Batch of upper bounds
        use_gpu: Whether to attempt GPU acceleration (default True)
        casadi_path: Path to compiled CasADi function

    Returns:
        BatchSolveResult with velocities, scales, and status
    """
    # Try GPU if requested and available
    if use_gpu and HAS_TORCH_CUDA and HAS_CUSADI:
        try:
            return solve_velocity_gpu_batched(
                objective_targets_batch,
                objective_jacobians_batch,
                constraint_matrices_batch,
                min_bounds_batch,
                max_bounds_batch,
                casadi_path,
            )
        except Exception as e:
            print(f"GPU solve failed, falling back to CPU: {e}")
    elif use_gpu:
        missing = []
        if not HAS_TORCH_CUDA:
            missing.append("PyTorch CUDA")
        if not HAS_CUSADI:
            missing.append("CusADi")
        print(f"GPU unavailable (missing: {', '.join(missing)}), using CPU")

    # CPU fallback
    # Convert to lists if numpy arrays
    if isinstance(objective_targets_batch, np.ndarray):
        B = objective_targets_batch.shape[0]
        targets_list = [objective_targets_batch[b] for b in range(B)]
    else:
        targets_list = list(objective_targets_batch)

    if isinstance(objective_jacobians_batch, np.ndarray):
        B = objective_jacobians_batch.shape[0]
        jacobians_list = [objective_jacobians_batch[b] for b in range(B)]
    else:
        jacobians_list = list(objective_jacobians_batch)

    if isinstance(constraint_matrices_batch, np.ndarray):
        B = constraint_matrices_batch.shape[0]
        C_list = [constraint_matrices_batch[b] for b in range(B)]
    else:
        C_list = list(constraint_matrices_batch)

    if isinstance(min_bounds_batch, np.ndarray):
        B = min_bounds_batch.shape[0]
        lower_list = [min_bounds_batch[b] for b in range(B)]
    else:
        lower_list = list(min_bounds_batch)

    if isinstance(max_bounds_batch, np.ndarray):
        B = max_bounds_batch.shape[0]
        upper_list = [max_bounds_batch[b] for b in range(B)]
    else:
        upper_list = list(max_bounds_batch)

    return _solve_cpu_sequential(
        targets_list,
        jacobians_list,
        C_list,
        lower_list,
        upper_list,
    )


def check_gpu_availability() -> Dict[str, bool]:
    """
    Check availability of GPU acceleration components.

    Returns:
        Dict with keys: casadi, cusadi, torch_cuda, all_available
    """
    return {
        "casadi": HAS_CASADI,
        "cusadi": HAS_CUSADI,
        "torch_cuda": HAS_TORCH_CUDA,
        "all_available": HAS_CASADI and HAS_CUSADI and HAS_TORCH_CUDA,
    }

#!/usr/bin/env python3
"""
Benchmark GPU batched IK with CusADi.

This script benchmarks FI-PeSNS with true GPU parallelism using CusADi.
Tests batch sizes of 100, 1000, 10000.
"""

import time
import numpy as np
from typing import List, Tuple, Optional

# Check available backends
print("=" * 70)
print("GPU Benchmark: FI-PeSNS with CusADi")
print("=" * 70)

# Check CUDA availability
try:
    import torch
    HAS_TORCH_CUDA = torch.cuda.is_available()
    if HAS_TORCH_CUDA:
        print(f"PyTorch CUDA: Available ({torch.cuda.get_device_name(0)})")
    else:
        print("PyTorch CUDA: Not available")
except ImportError:
    HAS_TORCH_CUDA = False
    print("PyTorch: Not installed")

# Check CusADi
HAS_CUSADI = False
CusadiFunction = None
try:
    from embodik.gpu import HAS_CUSADI as _HAS_CUSADI, CusadiFunction as _CusadiFunction
    HAS_CUSADI = _HAS_CUSADI
    CusadiFunction = _CusadiFunction
    print(f"CusADi: {'Available' if HAS_CUSADI else 'Not available'}")
except ImportError as e:
    print(f"CusADi import error: {e}")

# Check embodik
try:
    import embodik as eik
    print("EmbodiK: Available")
except ImportError:
    print("EmbodiK: Not installed")
    exit(1)

print("-" * 70)


def generate_batch_problems(
    n_samples: int,
    n_dof: int = 7,
    task_dim: int = 6,
    n_constraints: int = 7,
    seed: int = 42,
    bound_scale: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate batched IK problems."""
    rng = np.random.default_rng(seed)

    targets = rng.uniform(-0.5, 0.5, (n_samples, task_dim)).astype(np.float64)
    jacobians = rng.uniform(-1.0, 1.0, (n_samples, task_dim, n_dof)).astype(np.float64)
    C = np.eye(n_constraints, n_dof, dtype=np.float64)
    lower = np.full(n_constraints, -bound_scale, dtype=np.float64)
    upper = np.full(n_constraints, bound_scale, dtype=np.float64)

    return targets, jacobians, C, lower, upper


def benchmark_cpu_sequential(
    targets: np.ndarray,
    jacobians: np.ndarray,
    C: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """Benchmark CPU solver (sequential)."""
    n_samples = targets.shape[0]
    n_dof = jacobians.shape[2]
    velocities = np.zeros((n_samples, n_dof), dtype=np.float64)

    start = time.perf_counter()
    for i in range(n_samples):
        result = eik.computeMultiObjectiveVelocitySolutionEigen(
            [targets[i]], [np.asfortranarray(jacobians[i])], C, lower, upper
        )
        velocities[i] = np.array(result.solution).ravel()
    elapsed_ms = (time.perf_counter() - start) * 1000

    return velocities, elapsed_ms


def benchmark_casadi_sequential(
    targets: np.ndarray,
    jacobians: np.ndarray,
    C: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    k_max: int = 12,
) -> Tuple[np.ndarray, float]:
    """Benchmark CasADi solver (sequential, Python evaluation)."""
    try:
        from embodik.gpu.casadi_fi_pesns import build_fi_pesns_single_task
    except ImportError:
        return np.array([]), 0.0

    n_samples = targets.shape[0]
    n_dof = jacobians.shape[2]
    task_dim = jacobians.shape[1]
    n_constraints = C.shape[0]

    fn = build_fi_pesns_single_task(
        n_dof=n_dof,
        task_dim=task_dim,
        n_constraints=n_constraints,
        k_max=k_max,
    )

    velocities = np.zeros((n_samples, n_dof), dtype=np.float64)

    start = time.perf_counter()
    for i in range(n_samples):
        vel, _ = fn(targets[i], jacobians[i].flatten(), C, lower, upper)
        velocities[i] = np.array(vel).flatten()
    elapsed_ms = (time.perf_counter() - start) * 1000

    return velocities, elapsed_ms


def benchmark_cusadi_batched(
    targets: np.ndarray,
    jacobians: np.ndarray,
    C: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    casadi_file: str = None,
) -> Tuple[Optional[np.ndarray], float]:
    """Benchmark CusADi solver (batched GPU)."""
    if not HAS_CUSADI or CusadiFunction is None:
        return None, 0.0

    import torch
    import casadi as ca
    import os

    n_samples = targets.shape[0]
    n_dof = jacobians.shape[2]
    task_dim = jacobians.shape[1]
    n_constraints = C.shape[0]

    # Find and load the CasADi function file
    if casadi_file is None:
        home = os.path.expanduser("~")
        casadi_file = os.path.join(home, ".local", "cusadi", "src", "casadi_functions", "fn_velocity_solve.casadi")

    if not os.path.exists(casadi_file):
        print(f"    CasADi file not found: {casadi_file}")
        return None, 0.0

    try:
        # Load CasADi function from file
        fn_casadi = ca.Function.load(casadi_file)
        print(f"    Loaded CasADi function: {fn_casadi.name()}")

        # Create CusADi batched function
        fn = CusadiFunction(fn_casadi, n_samples)
    except Exception as e:
        print(f"    Failed to load CusADi function: {e}")
        import traceback
        traceback.print_exc()
        return None, 0.0

    # Prepare batched inputs as torch tensors
    device = torch.device("cuda")

    # Flatten jacobians for CasADi input (row-major)
    jacobians_flat = jacobians.reshape(n_samples, -1)

    # Flatten C matrix
    C_flat = C.flatten()
    C_batch = np.tile(C_flat[np.newaxis, :], (n_samples, 1))

    # Expand lower, upper to batch dimension
    lower_batch = np.tile(lower[np.newaxis, :], (n_samples, 1))
    upper_batch = np.tile(upper[np.newaxis, :], (n_samples, 1))

    # Convert to torch tensors (double precision, contiguous)
    targets_t = torch.from_numpy(targets.copy()).double().to(device).contiguous()
    jacobians_t = torch.from_numpy(jacobians_flat.copy()).double().to(device).contiguous()
    C_t = torch.from_numpy(C_batch.copy()).double().to(device).contiguous()
    lower_t = torch.from_numpy(lower_batch.copy()).double().to(device).contiguous()
    upper_t = torch.from_numpy(upper_batch.copy()).double().to(device).contiguous()

    inputs = [targets_t, jacobians_t, C_t, lower_t, upper_t]

    # Warm-up
    try:
        fn.evaluate(inputs)
        torch.cuda.synchronize()
    except Exception as e:
        print(f"    CusADi evaluation error: {e}")
        import traceback
        traceback.print_exc()
        return None, 0.0

    # Benchmark (average of 5 runs)
    n_runs = 5
    elapsed_total = 0.0

    for _ in range(n_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn.evaluate(inputs)
        torch.cuda.synchronize()
        elapsed_total += (time.perf_counter() - start) * 1000

    elapsed_ms = elapsed_total / n_runs

    # Get output
    velocities = fn.getDenseOutput(0).cpu().numpy().squeeze()

    return velocities, elapsed_ms


def main():
    batch_sizes = [100, 1000, 10000]
    n_dof = 7
    task_dim = 6
    n_constraints = 7

    print("\nBenchmark Configuration:")
    print(f"  n_dof: {n_dof}, task_dim: {task_dim}, n_constraints: {n_constraints}")
    print(f"  batch_sizes: {batch_sizes}")
    print()

    # Results storage
    results = []

    for batch_size in batch_sizes:
        print(f"\n{'='*70}")
        print(f"Batch Size: {batch_size}")
        print("=" * 70)

        # Generate problems
        targets, jacobians, C, lower, upper = generate_batch_problems(
            n_samples=batch_size,
            n_dof=n_dof,
            task_dim=task_dim,
            n_constraints=n_constraints,
        )

        # CPU Sequential
        print("  [1] CPU (sequential)...")
        cpu_vels, cpu_time = benchmark_cpu_sequential(targets, jacobians, C, lower, upper)
        print(f"      Time: {cpu_time:.2f} ms ({cpu_time/batch_size:.4f} ms/sample)")

        # CasADi Sequential (Python)
        print("  [2] CasADi (sequential, Python)...")
        casadi_vels, casadi_time = benchmark_casadi_sequential(targets, jacobians, C, lower, upper)
        if casadi_time > 0:
            print(f"      Time: {casadi_time:.2f} ms ({casadi_time/batch_size:.4f} ms/sample)")
        else:
            print("      Skipped (CasADi not available)")

        # CusADi Batched (GPU)
        print("  [3] CusADi (batched GPU)...")
        if HAS_CUSADI and HAS_TORCH_CUDA:
            gpu_vels, gpu_time = benchmark_cusadi_batched(targets, jacobians, C, lower, upper)
            if gpu_vels is not None:
                print(f"      Time: {gpu_time:.2f} ms ({gpu_time/batch_size:.6f} ms/sample)")
                speedup = cpu_time / gpu_time if gpu_time > 0 else 0
                print(f"      Speedup vs CPU: {speedup:.1f}x")

                # Check constraint satisfaction
                constraint_vals = np.einsum('ij,bj->bi', C, gpu_vels)
                viol_low = np.maximum(0, lower[np.newaxis, :] - constraint_vals)
                viol_high = np.maximum(0, constraint_vals - upper[np.newaxis, :])
                max_viol = np.max(np.maximum(viol_low, viol_high))
                sat_rate = np.mean(np.max(np.maximum(viol_low, viol_high), axis=1) < 1e-3) * 100
                print(f"      Max constraint violation: {max_viol:.6f}")
                print(f"      Constraint satisfaction: {sat_rate:.1f}%")
            else:
                print("      Skipped (CusADi function not compiled)")
                gpu_time = 0
        else:
            print("      Skipped (CusADi or CUDA not available)")
            gpu_time = 0

        results.append({
            "batch_size": batch_size,
            "cpu_ms": cpu_time,
            "casadi_ms": casadi_time,
            "gpu_ms": gpu_time,
        })

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Batch':<10} {'CPU (ms)':<15} {'CasADi (ms)':<15} {'GPU (ms)':<15} {'Speedup':<10}")
    print("-" * 80)

    for r in results:
        speedup = r["cpu_ms"] / r["gpu_ms"] if r["gpu_ms"] > 0 else "N/A"
        speedup_str = f"{speedup:.1f}x" if isinstance(speedup, float) else speedup
        print(f"{r['batch_size']:<10} {r['cpu_ms']:<15.2f} {r['casadi_ms']:<15.2f} {r['gpu_ms']:<15.2f} {speedup_str:<10}")

    print("=" * 80)

    if not HAS_CUSADI or not HAS_TORCH_CUDA:
        print("\nNote: GPU benchmarks require CusADi + CUDA. To enable:")
        print("  1. Export CasADi function:")
        print("     pixi run -e cuda export-casadi")
        print("  2. Compile with CusADi:")
        print("     mkdir -p ~/.local/cusadi/src/casadi_functions")
        print("     cp build/casadi/fn_velocity_solve.casadi ~/.local/cusadi/src/casadi_functions/")
        print("     cd ~/.local/cusadi && python run_codegen.py --fn=fn_velocity_solve")


if __name__ == "__main__":
    main()

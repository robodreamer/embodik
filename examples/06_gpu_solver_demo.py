#!/usr/bin/env python3
"""
EmbodiK GPU Solver Demonstration
================================

This script demonstrates the advantages of GPU-accelerated batch velocity IK
solving compared to sequential CPU computation.

Key benefits demonstrated:
1. **Massive parallelism**: Solve thousands of IK problems simultaneously
2. **Linear scaling**: GPU time grows slowly with batch size
3. **RL integration ready**: Perfect for batched sim-to-real training

Usage:
    # CPU-only mode (default)
    python examples/06_gpu_solver_demo.py

    # With GPU (requires CusADi + CUDA)
    python examples/06_gpu_solver_demo.py --gpu --casadi_path path/to/fn_velocity_solve.casadi

    # Specific batch sizes
    python examples/06_gpu_solver_demo.py --batch_sizes 100 500 1000 2000 4000
"""

import argparse
import time
import numpy as np
import sys
from dataclasses import dataclass
from typing import List, Tuple, Optional

# Optional imports for plotting
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    batch_size: int
    cpu_time_ms: float
    gpu_time_ms: Optional[float]
    cpu_throughput: float  # solves/sec
    gpu_throughput: Optional[float]
    speedup: Optional[float]
    max_error: float  # max velocity difference if GPU available


def generate_random_ik_problems(
    n_problems: int,
    n_dof: int = 7,
    task_dim: int = 6,
    seed: int = 42
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray],
           List[np.ndarray], List[np.ndarray]]:
    """
    Generate random IK problems for benchmarking.

    Returns:
        targets: List of (task_dim,) target velocities
        jacobians: List of (task_dim, n_dof) Jacobian matrices
        constraints: List of (n_dof, n_dof) constraint matrices (identity)
        lower_bounds: List of (n_dof,) lower velocity limits
        upper_bounds: List of (n_dof,) upper velocity limits
    """
    rng = np.random.RandomState(seed)

    targets = []
    jacobians = []
    constraints = []
    lower_bounds = []
    upper_bounds = []

    for i in range(n_problems):
        # Random target velocity (small magnitude)
        target = rng.randn(task_dim) * 0.1
        targets.append(target)

        # Random Jacobian with reasonable condition number
        J = rng.randn(task_dim, n_dof)
        # Ensure not too ill-conditioned
        U, s, Vt = np.linalg.svd(J, full_matrices=False)
        s = np.clip(s, 0.1, 10.0)  # Condition number <= 100
        J = U @ np.diag(s) @ Vt
        jacobians.append(np.asfortranarray(J))

        # Identity constraint matrix
        C = np.eye(n_dof)
        constraints.append(np.asfortranarray(C))

        # Velocity limits
        lower_bounds.append(np.full(n_dof, -2.0))
        upper_bounds.append(np.full(n_dof, 2.0))

    return targets, jacobians, constraints, lower_bounds, upper_bounds


def benchmark_cpu_sequential(
    targets: List[np.ndarray],
    jacobians: List[np.ndarray],
    constraints: List[np.ndarray],
    lower_bounds: List[np.ndarray],
    upper_bounds: List[np.ndarray],
    n_warmup: int = 3,
    n_runs: int = 5
) -> Tuple[float, List[np.ndarray]]:
    """
    Benchmark sequential CPU solving.

    Returns:
        (mean_time_ms, solutions)
    """
    import embodik as eik

    n_problems = len(targets)

    # Warmup
    for _ in range(n_warmup):
        for i in range(min(10, n_problems)):
            eik.computeMultiObjectiveVelocitySolutionEigen(
                [targets[i]], [jacobians[i]], constraints[i],
                lower_bounds[i], upper_bounds[i]
            )

    # Timed runs
    times = []
    solutions = None

    for run in range(n_runs):
        start = time.perf_counter()
        solutions = []
        for i in range(n_problems):
            result = eik.computeMultiObjectiveVelocitySolutionEigen(
                [targets[i]], [jacobians[i]], constraints[i],
                lower_bounds[i], upper_bounds[i]
            )
            solutions.append(np.array(result.solution))
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    return np.mean(times), solutions


def benchmark_gpu_batched(
    targets: List[np.ndarray],
    jacobians: List[np.ndarray],
    constraints: List[np.ndarray],
    lower_bounds: List[np.ndarray],
    upper_bounds: List[np.ndarray],
    casadi_path: str,
    n_warmup: int = 3,
    n_runs: int = 5
) -> Tuple[float, np.ndarray]:
    """
    Benchmark batched GPU solving.

    Returns:
        (mean_time_ms, solutions)
    """
    from embodik import solve_velocity_gpu_batched

    # Warmup
    for _ in range(n_warmup):
        solve_velocity_gpu_batched(
            targets[:10], jacobians[:10], constraints[:10],
            lower_bounds[:10], upper_bounds[:10],
            casadi_path=casadi_path
        )

    # Timed runs
    times = []
    result = None

    for run in range(n_runs):
        start = time.perf_counter()
        result = solve_velocity_gpu_batched(
            targets, jacobians, constraints,
            lower_bounds, upper_bounds,
            casadi_path=casadi_path
        )
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    return np.mean(times), result.velocities


def run_benchmark(
    batch_sizes: List[int],
    n_dof: int = 7,
    task_dim: int = 6,
    use_gpu: bool = False,
    casadi_path: Optional[str] = None
) -> List[BenchmarkResult]:
    """Run comprehensive benchmark across batch sizes."""

    results = []

    print("\n" + "=" * 70)
    print("EmbodiK GPU Solver Benchmark")
    print("=" * 70)
    print(f"Configuration: n_dof={n_dof}, task_dim={task_dim}")
    print(f"GPU enabled: {use_gpu}")
    if use_gpu and casadi_path:
        print(f"CasADi path: {casadi_path}")
    print()

    # Check GPU availability
    from embodik.gpu import HAS_CASADI, HAS_CUSADI, HAS_TORCH_CUDA
    print("GPU Status:")
    print(f"  CasADi: {'✓' if HAS_CASADI else '✗'}")
    print(f"  CusADi: {'✓' if HAS_CUSADI else '✗'}")
    print(f"  CUDA:   {'✓' if HAS_TORCH_CUDA else '✗'}")
    print()

    can_gpu = use_gpu and casadi_path and HAS_CUSADI and HAS_TORCH_CUDA

    # Header
    if can_gpu:
        print(f"{'Batch':<10} {'CPU (ms)':<12} {'GPU (ms)':<12} {'Speedup':<10} {'Max Error':<12}")
        print("-" * 56)
    else:
        print(f"{'Batch':<10} {'CPU (ms)':<12} {'Throughput':<15}")
        print("-" * 37)

    for batch_size in batch_sizes:
        # Generate problems
        targets, jacobians, constraints, lower_bounds, upper_bounds = \
            generate_random_ik_problems(batch_size, n_dof, task_dim)

        # CPU benchmark
        cpu_time, cpu_solutions = benchmark_cpu_sequential(
            targets, jacobians, constraints, lower_bounds, upper_bounds
        )
        cpu_throughput = batch_size / (cpu_time / 1000)

        # GPU benchmark (if available)
        gpu_time = None
        gpu_throughput = None
        speedup = None
        max_error = 0.0

        if can_gpu:
            gpu_time, gpu_solutions = benchmark_gpu_batched(
                targets, jacobians, constraints, lower_bounds, upper_bounds,
                casadi_path
            )
            gpu_throughput = batch_size / (gpu_time / 1000)
            speedup = cpu_time / gpu_time

            # Compute max error
            for i in range(batch_size):
                error = np.max(np.abs(cpu_solutions[i] - gpu_solutions[i]))
                max_error = max(max_error, error)

        result = BenchmarkResult(
            batch_size=batch_size,
            cpu_time_ms=cpu_time,
            gpu_time_ms=gpu_time,
            cpu_throughput=cpu_throughput,
            gpu_throughput=gpu_throughput,
            speedup=speedup,
            max_error=max_error
        )
        results.append(result)

        # Print result
        if can_gpu:
            print(f"{batch_size:<10} {cpu_time:<12.2f} {gpu_time:<12.2f} "
                  f"{speedup:<10.1f}x {max_error:<12.2e}")
        else:
            print(f"{batch_size:<10} {cpu_time:<12.2f} {cpu_throughput:<15.0f} solves/s")

    return results


def plot_results(results: List[BenchmarkResult], save_path: Optional[str] = None):
    """Plot benchmark results."""
    if not HAS_MATPLOTLIB:
        print("\nMatplotlib not available, skipping plot")
        return

    batch_sizes = [r.batch_size for r in results]
    cpu_times = [r.cpu_time_ms for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Time comparison
    ax1 = axes[0]
    ax1.plot(batch_sizes, cpu_times, 'b-o', label='CPU Sequential', linewidth=2, markersize=8)

    if results[0].gpu_time_ms is not None:
        gpu_times = [r.gpu_time_ms for r in results]
        ax1.plot(batch_sizes, gpu_times, 'r-s', label='GPU Batched', linewidth=2, markersize=8)

    ax1.set_xlabel('Batch Size', fontsize=12)
    ax1.set_ylabel('Time (ms)', fontsize=12)
    ax1.set_title('Solve Time vs Batch Size', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    ax1.set_yscale('log')

    # Throughput comparison
    ax2 = axes[1]
    cpu_throughput = [r.cpu_throughput for r in results]
    ax2.plot(batch_sizes, cpu_throughput, 'b-o', label='CPU Sequential', linewidth=2, markersize=8)

    if results[0].gpu_throughput is not None:
        gpu_throughput = [r.gpu_throughput for r in results]
        ax2.plot(batch_sizes, gpu_throughput, 'r-s', label='GPU Batched', linewidth=2, markersize=8)

    ax2.set_xlabel('Batch Size', fontsize=12)
    ax2.set_ylabel('Throughput (solves/sec)', fontsize=12)
    ax2.set_title('Throughput Scaling', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    ax2.set_yscale('log')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
    else:
        plt.show()


def print_summary(results: List[BenchmarkResult]):
    """Print a summary of the benchmark results."""
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    # CPU performance
    print("\nCPU Sequential Performance:")
    for r in results:
        ms_per_solve = r.cpu_time_ms / r.batch_size
        print(f"  Batch {r.batch_size:>5}: {ms_per_solve:.4f} ms/solve, "
              f"{r.cpu_throughput:.0f} solves/s")

    # GPU performance (if available)
    if results[0].gpu_time_ms is not None:
        print("\nGPU Batched Performance:")
        for r in results:
            ms_per_solve = r.gpu_time_ms / r.batch_size
            print(f"  Batch {r.batch_size:>5}: {ms_per_solve:.6f} ms/solve, "
                  f"{r.gpu_throughput:.0f} solves/s, {r.speedup:.1f}x speedup")

        # Peak speedup
        max_speedup = max(r.speedup for r in results)
        best_batch = max(results, key=lambda r: r.speedup)
        print(f"\nPeak speedup: {max_speedup:.1f}x at batch size {best_batch.batch_size}")

        # Accuracy
        max_error = max(r.max_error for r in results)
        print(f"Maximum GPU vs CPU error: {max_error:.2e}")

    # Use case recommendations
    print("\n" + "-" * 70)
    print("Use Case Recommendations:")
    print("-" * 70)
    print("""
1. **RL Training (Isaac Gym/Orbit)**
   - Batch size: 4096+ environments
   - Expected speedup: 100-500x
   - Use: Batched policy inference with IK constraints

2. **Motion Planning (RRT/PRM)**
   - Batch size: 100-1000 collision checks
   - Expected speedup: 20-100x
   - Use: Parallel trajectory validation

3. **Real-time Control**
   - Batch size: 1 (single robot)
   - CPU solver is sufficient (~0.03ms)
   - GPU overhead not worth it for single queries

4. **Dataset Generation**
   - Batch size: 10000+ samples
   - Expected speedup: 200-1000x
   - Use: Offline batch processing
""")


def main():
    parser = argparse.ArgumentParser(
        description="EmbodiK GPU Solver Demonstration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic CPU benchmark
  python examples/06_gpu_solver_demo.py

  # With GPU acceleration
  python examples/06_gpu_solver_demo.py --gpu --casadi_path /path/to/fn_velocity_solve.casadi

  # Custom batch sizes
  python examples/06_gpu_solver_demo.py --batch_sizes 100 500 1000 2000 4000

  # Save plot
  python examples/06_gpu_solver_demo.py --plot benchmark_results.png
"""
    )

    parser.add_argument(
        "--batch_sizes", type=int, nargs="+",
        default=[10, 50, 100, 500, 1000, 2000],
        help="Batch sizes to benchmark"
    )
    parser.add_argument(
        "--n_dof", type=int, default=7,
        help="Number of DOF (default: 7 for Panda)"
    )
    parser.add_argument(
        "--task_dim", type=int, default=6,
        help="Task dimension (default: 6 for SE(3))"
    )
    parser.add_argument(
        "--gpu", action="store_true",
        help="Enable GPU benchmarking"
    )
    parser.add_argument(
        "--casadi_path", type=str,
        help="Path to compiled CasADi function (.casadi file)"
    )
    parser.add_argument(
        "--plot", type=str,
        help="Save plot to file (e.g., benchmark.png)"
    )

    args = parser.parse_args()

    # Run benchmark
    results = run_benchmark(
        batch_sizes=args.batch_sizes,
        n_dof=args.n_dof,
        task_dim=args.task_dim,
        use_gpu=args.gpu,
        casadi_path=args.casadi_path
    )

    # Print summary
    print_summary(results)

    # Plot if requested
    if args.plot:
        plot_results(results, save_path=args.plot)
    elif HAS_MATPLOTLIB:
        plot_results(results)


if __name__ == "__main__":
    main()

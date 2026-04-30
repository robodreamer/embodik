#!/usr/bin/env python3
"""
GPU Batch IK Example

This example demonstrates GPU-accelerated batched velocity IK solving
using CusADi. It compares performance between CPU sequential and GPU
parallel solving.

Prerequisites:
    1. Install CusADi: git clone https://github.com/se-hwan/cusadi && pip install -e cusadi
    2. Export and compile the CasADi function:
       python -m embodik.gpu.export_casadi_velocity_solve --robot panda
       mkdir -p cusadi/src/casadi_functions
       cp build/casadi/fn_velocity_solve.casadi cusadi/src/casadi_functions/
       cd cusadi && python run_codegen.py --fn=fn_velocity_solve

Usage:
    python examples/04_gpu_batch_ik.py [--batch_sizes 10 100 1000] [--use_gpu]
"""

from __future__ import annotations

import argparse
import time
from typing import List, Tuple

import numpy as np


def generate_random_problems(
    n_problems: int,
    n_dof: int = 7,
    task_dim: int = 6,
    seed: int = 42,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """Generate random IK problems for benchmarking."""
    rng = np.random.default_rng(seed)

    all_targets = []
    all_jacobians = []
    all_C = []
    all_lower = []
    all_upper = []

    for _ in range(n_problems):
        # Random target velocity
        target = rng.uniform(-0.5, 0.5, task_dim).astype(np.float64)
        all_targets.append(target)

        # Random Jacobian
        jacobian = rng.uniform(-1.0, 1.0, (task_dim, n_dof)).astype(np.float64)
        all_jacobians.append(jacobian.flatten())

        # Identity constraint matrix (joint velocity limits)
        C = np.eye(n_dof, dtype=np.float64)
        all_C.append(C)

        # Joint velocity limits
        lower = np.full(n_dof, -1.0, dtype=np.float64)
        upper = np.full(n_dof, 1.0, dtype=np.float64)
        all_lower.append(lower)
        all_upper.append(upper)

    return all_targets, all_jacobians, all_C, all_lower, all_upper


def benchmark_cpu_sequential(
    targets: List[np.ndarray],
    jacobians: List[np.ndarray],
    C_list: List[np.ndarray],
    lower_list: List[np.ndarray],
    upper_list: List[np.ndarray],
) -> Tuple[float, List[np.ndarray]]:
    """Benchmark CPU sequential solving."""
    import embodik as eik

    solutions = []

    start = time.perf_counter()
    for i in range(len(targets)):
        # Wrap in lists for EmbodiK API
        result = eik.computeMultiObjectiveVelocitySolutionEigen(
            [targets[i]],
            [jacobians[i].reshape(-1, C_list[i].shape[0])],  # Reshape to (task_dim, n_dof)
            C_list[i],
            lower_list[i],
            upper_list[i],
        )
        solutions.append(np.array(result.solution))

    elapsed_ms = (time.perf_counter() - start) * 1000
    return elapsed_ms, solutions


def benchmark_gpu_batched(
    targets: List[np.ndarray],
    jacobians: List[np.ndarray],
    C_list: List[np.ndarray],
    lower_list: List[np.ndarray],
    upper_list: List[np.ndarray],
    casadi_path: str = None,
) -> Tuple[float, np.ndarray, str]:
    """Benchmark GPU batched solving."""
    from embodik.gpu_solver import solve_velocity_batched, DEFAULT_FN_PATH

    # Use default path if not specified
    if casadi_path is None or casadi_path == "":
        casadi_path = DEFAULT_FN_PATH

    start = time.perf_counter()
    result = solve_velocity_batched(
        targets,
        jacobians,
        C_list,
        lower_list,
        upper_list,
        use_gpu=True,
        casadi_path=casadi_path,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000

    return elapsed_ms, result.velocities, result.status


def main():
    parser = argparse.ArgumentParser(description="GPU Batch IK Benchmark")
    parser.add_argument(
        "--batch_sizes",
        type=int,
        nargs="+",
        default=[10, 100, 500, 1000],
        help="Batch sizes to test",
    )
    parser.add_argument(
        "--n_dof",
        type=int,
        default=7,
        help="Number of DOF (default: 7)",
    )
    parser.add_argument(
        "--task_dim",
        type=int,
        default=6,
        help="Task dimension (default: 6)",
    )
    parser.add_argument(
        "--casadi_path",
        type=str,
        default="",
        help="Path to compiled .casadi function",
    )
    parser.add_argument(
        "--cpu_only",
        action="store_true",
        help="Only run CPU benchmarks",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("EmbodiK GPU Batch IK Benchmark")
    print("=" * 70)
    print(f"n_dof: {args.n_dof}, task_dim: {args.task_dim}")
    print()

    # Check GPU availability
    try:
        from embodik.gpu_solver import check_gpu_availability
        gpu_status = check_gpu_availability()
        print("GPU Status:")
        for key, value in gpu_status.items():
            status = "✓" if value else "✗"
            print(f"  {key}: {status}")
        print()
    except ImportError:
        print("GPU solver module not available")
        gpu_status = {"all_available": False}

    # Run benchmarks
    results = []

    for batch_size in args.batch_sizes:
        print(f"Batch size: {batch_size}")
        print("-" * 40)

        # Generate problems
        targets, jacobians, C_list, lower_list, upper_list = generate_random_problems(
            n_problems=batch_size,
            n_dof=args.n_dof,
            task_dim=args.task_dim,
        )

        # CPU benchmark
        cpu_time, cpu_solutions = benchmark_cpu_sequential(
            targets, jacobians, C_list, lower_list, upper_list
        )
        print(f"  CPU sequential: {cpu_time:.2f} ms ({cpu_time/batch_size:.3f} ms/solve)")

        # GPU benchmark (if available and not cpu_only)
        if not args.cpu_only:
            try:
                gpu_time, gpu_solutions, status = benchmark_gpu_batched(
                    targets, jacobians, C_list, lower_list, upper_list,
                    casadi_path=args.casadi_path,
                )

                if status == "success":
                    speedup = cpu_time / gpu_time
                    print(f"  GPU batched:    {gpu_time:.2f} ms ({gpu_time/batch_size:.3f} ms/solve)")
                    print(f"  Speedup:        {speedup:.1f}x")

                    # Check solution quality (note: GPU uses approximate symbolic solver)
                    # Report error magnitudes rather than exact matching
                    errors = np.array([
                        np.max(np.abs(cpu_solutions[i] - gpu_solutions[i]))
                        for i in range(batch_size)
                    ])
                    print(f"  Max error:      {np.max(errors):.4f} (GPU uses approximate solver)")
                else:
                    print(f"  GPU status:     {status} (fell back to CPU)")

            except Exception as e:
                print(f"  GPU error:      {e}")

        print()

        results.append({
            "batch_size": batch_size,
            "cpu_time_ms": cpu_time,
        })

    # Summary table
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"{'Batch Size':<15} {'CPU (ms)':<15} {'ms/solve':<15}")
    print("-" * 45)
    for r in results:
        print(f"{r['batch_size']:<15} {r['cpu_time_ms']:<15.2f} {r['cpu_time_ms']/r['batch_size']:<15.4f}")


if __name__ == "__main__":
    main()

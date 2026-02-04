#!/usr/bin/env python3
"""
Batched benchmark comparison between FI-PeSNS and PPH-SNS solvers.

This script compares the performance of both solvers at various batch sizes
to demonstrate GPU parallelization benefits.
"""

import argparse
import numpy as np
import time
from pathlib import Path
import sys

# Add the python path for embodik
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

try:
    import casadi as ca
    from embodik.gpu.casadi_fi_pesns import build_fi_pesns_single_task
    from embodik.gpu.casadi_pph_sns import build_pph_sns_single_task
    HAS_CASADI = True
except ImportError:
    HAS_CASADI = False
    print("CasADi not available")
    sys.exit(1)

try:
    from embodik.gpu import HAS_CUSADI, CusadiFunction
except ImportError:
    HAS_CUSADI = False
    CusadiFunction = None

if not HAS_CUSADI:
    print("CusADi not available - GPU batched benchmarking not possible")
    sys.exit(1)

import torch


def benchmark_batched_solver(solver_name: str, casadi_fn: ca.Function, batch_size: int, n_runs: int = 20):
    """Benchmark a solver with batched inputs."""
    # Build CusADi function for batch size
    try:
        cusadi_fn = CusadiFunction(casadi_fn, batch_size)
    except Exception as e:
        print(f"  {solver_name}: Failed to build for batch {batch_size}: {e}")
        return None

    # Generate random test data
    rng = np.random.RandomState(42)
    n_dof, task_dim = 7, 6

    targets = rng.randn(batch_size, task_dim).astype(np.float64) * 0.1
    jacobians = rng.randn(batch_size, task_dim * n_dof).astype(np.float64)
    C = np.tile(np.eye(n_dof)[np.newaxis, :, :], (batch_size, 1, 1)).astype(np.float64)
    lower = np.tile(np.full(n_dof, -2.0)[np.newaxis, :], (batch_size, 1)).astype(np.float64)
    upper = np.tile(np.full(n_dof, 2.0)[np.newaxis, :], (batch_size, 1)).astype(np.float64)

    # Convert to tensors
    device = torch.device("cuda")
    targets_t = torch.from_numpy(targets).to(device).contiguous()
    jacobians_t = torch.from_numpy(jacobians).to(device).contiguous()
    C_t = torch.from_numpy(C).to(device).contiguous()
    lower_t = torch.from_numpy(lower).to(device).contiguous()
    upper_t = torch.from_numpy(upper).to(device).contiguous()

    # Warmup
    for _ in range(3):
        cusadi_fn.evaluate([targets_t, jacobians_t, C_t, lower_t, upper_t])
    torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(n_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        cusadi_fn.evaluate([targets_t, jacobians_t, C_t, lower_t, upper_t])
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append(end - start)

    avg_time = np.mean(times) * 1000  # ms
    std_time = np.std(times) * 1000  # ms
    throughput = batch_size / (avg_time / 1000)  # solves/sec

    return {
        'batch_size': batch_size,
        'time_avg': avg_time,
        'time_std': std_time,
        'throughput': throughput,
    }


def run_batched_benchmark():
    """Run batched benchmark for both solvers."""
    print("=" * 80)
    print("BATCHED GPU BENCHMARK: FI-PeSNS vs PPH-SNS")
    print("=" * 80)
    print()

    # Build CasADi functions
    n_dof, task_dim, n_constraints = 7, 6, 7

    print("Building FI-PeSNS solver...")
    fi_pesns_fn = build_fi_pesns_single_task(n_dof, task_dim, n_constraints)

    print("Building PPH-SNS solver...")
    pph_sns_fn = build_pph_sns_single_task(n_dof, task_dim, n_constraints)

    # Batch sizes to test
    batch_sizes = [1, 10, 100, 1000, 5000, 10000]

    print()
    print(f"{'Batch Size':<12} {'FI-PeSNS (ms)':<16} {'PPH-SNS (ms)':<16} {'FI Throughput':<16} {'PPH Throughput':<16}")
    print("-" * 80)

    fi_results = []
    pph_results = []

    for batch_size in batch_sizes:
        fi_result = benchmark_batched_solver("FI-PeSNS", fi_pesns_fn, batch_size)
        pph_result = benchmark_batched_solver("PPH-SNS", pph_sns_fn, batch_size)

        if fi_result and pph_result:
            fi_results.append(fi_result)
            pph_results.append(pph_result)

            fi_time_str = f"{fi_result['time_avg']:.2f} ± {fi_result['time_std']:.2f}"
            pph_time_str = f"{pph_result['time_avg']:.2f} ± {pph_result['time_std']:.2f}"
            fi_tput_str = f"{fi_result['throughput']:.0f}/s"
            pph_tput_str = f"{pph_result['throughput']:.0f}/s"

            print(f"{batch_size:<12} {fi_time_str:<16} {pph_time_str:<16} {fi_tput_str:<16} {pph_tput_str:<16}")

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if fi_results and pph_results:
        # Compare at largest batch size
        fi_max = fi_results[-1]
        pph_max = pph_results[-1]

        print(f"At batch size {fi_max['batch_size']}:")
        print(f"  FI-PeSNS: {fi_max['time_avg']:.2f} ms, {fi_max['throughput']:.0f} solves/sec")
        print(f"  PPH-SNS: {pph_max['time_avg']:.2f} ms, {pph_max['throughput']:.0f} solves/sec")

        time_ratio = pph_max['time_avg'] / fi_max['time_avg'] if fi_max['time_avg'] > 0 else 0
        print(f"  Time Ratio (PPH/FI): {time_ratio:.2f}x")


if __name__ == "__main__":
    run_batched_benchmark()
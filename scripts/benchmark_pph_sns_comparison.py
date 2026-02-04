#!/usr/bin/env python3
"""
Benchmark comparison between FI-PeSNS and PPH-SNS solvers.

This script compares the accuracy and performance of:
1. FI-PeSNS: Fixed-Iteration Penalized eSNS (original GPU adaptation)
2. PPH-SNS: Parallel Penalized Hierarchical SNS (new GPU-native design)

Both solvers are tested on the same problems to ensure fair comparison.
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
    print("CasADi not available - CPU-only benchmarking")

try:
    from embodik.gpu import HAS_CUSADI, CusadiFunction
except ImportError:
    HAS_CUSADI = False
    CusadiFunction = None

if not HAS_CUSADI:
    print("CusADi not available - CPU-only benchmarking")


def generate_test_problem(n_dof=7, task_dim=6, n_constraints=7, seed=42):
    """Generate a challenging test IK problem."""
    rng = np.random.RandomState(seed)

    # Generate well-conditioned Jacobian
    J = rng.randn(task_dim, n_dof)
    U, s, Vt = np.linalg.svd(J, full_matrices=False)
    s = np.clip(s, 0.5, 2.0)  # Good conditioning
    J = U @ np.diag(s) @ Vt

    # Generate target velocity
    target = rng.randn(task_dim) * 0.1

    # Generate constraints (joint limits)
    C = np.eye(n_dof)
    lower = np.full(n_dof, -2.0)
    upper = np.full(n_dof, 2.0)

    # Add some inequality constraints to make it challenging
    if n_constraints > n_dof:
        # Add velocity limits
        vel_limits = rng.randn(n_constraints - n_dof, n_dof) * 0.1
        C = np.vstack([C, vel_limits])
        lower = np.concatenate([lower, np.full(n_constraints - n_dof, -1.0)])
        upper = np.concatenate([upper, np.full(n_constraints - n_dof, 1.0)])

    return J.astype(np.float64), target.astype(np.float64), C.astype(np.float64), lower.astype(np.float64), upper.astype(np.float64)


def benchmark_solver_cpu(solver_fn, J, target, C, lower, upper, n_runs=100):
    """Benchmark solver on CPU."""
    times = []

    for _ in range(n_runs):
        start = time.perf_counter()
        result = solver_fn(target, J.flatten(), C, lower, upper)
        end = time.perf_counter()
        times.append(end - start)

    velocity = np.array(result[0]).flatten()
    scales = np.array(result[1]).flatten()

    # Check constraint satisfaction
    constraint_violation = np.maximum(0, lower - C @ velocity) + np.maximum(0, C @ velocity - upper)
    max_violation = np.max(constraint_violation)
    mean_violation = np.mean(constraint_violation)

    return {
        'velocity': velocity,
        'scales': scales,
        'time_avg': np.mean(times),
        'time_std': np.std(times),
        'max_violation': max_violation,
        'mean_violation': mean_violation,
        'task_scale': float(scales[0]) if len(scales) > 0 else 1.0,
    }


def benchmark_solver_gpu(cusadi_fn, J, target, C, lower, upper, n_runs=100):
    """Benchmark CusADi-compiled solver on GPU."""
    if cusadi_fn is None:
        return None

    import torch

    times = []

    # Convert to torch tensors on GPU
    device = torch.device("cuda")
    target_t = torch.from_numpy(target.astype(np.float64)).to(device).contiguous()
    J_flat = torch.from_numpy(J.flatten().astype(np.float64)).to(device).contiguous()
    C_t = torch.from_numpy(C.astype(np.float64)).to(device).contiguous()
    lower_t = torch.from_numpy(lower.astype(np.float64)).to(device).contiguous()
    upper_t = torch.from_numpy(upper.astype(np.float64)).to(device).contiguous()

    # Warmup
    for _ in range(3):
        cusadi_fn.evaluate([target_t, J_flat, C_t, lower_t, upper_t])
    torch.cuda.synchronize()

    for _ in range(n_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        cusadi_fn.evaluate([target_t, J_flat, C_t, lower_t, upper_t])
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append(end - start)

    velocity = cusadi_fn.getDenseOutput(0).cpu().numpy().flatten()

    # Check constraint satisfaction
    C_np = C.astype(np.float64)
    constraint_violation = np.maximum(0, lower - C_np @ velocity) + np.maximum(0, C_np @ velocity - upper)
    max_violation = np.max(constraint_violation)
    mean_violation = np.mean(constraint_violation)

    return {
        'velocity': velocity,
        'time_avg': np.mean(times),
        'time_std': np.std(times),
        'max_violation': max_violation,
        'mean_violation': mean_violation,
        'task_scale': None,  # Not available from CusADi
    }


def run_comparison(n_problems=10, n_runs=50):
    """Run comprehensive comparison between FI-PeSNS and PPH-SNS."""
    print("=" * 80)
    print("SOLVER COMPARISON: FI-PeSNS vs PPH-SNS")
    print("=" * 80)
    print(f"Testing {n_problems} problems, {n_runs} runs each")
    print()

    if not HAS_CASADI:
        print("ERROR: CasADi required for this benchmark")
        return

    # Build solvers
    n_dof, task_dim, n_constraints = 7, 6, 7

    print("Building FI-PeSNS solver...")
    fi_pesns_fn = build_fi_pesns_single_task(n_dof, task_dim, n_constraints)

    print("Building PPH-SNS solver...")
    pph_sns_fn = build_pph_sns_single_task(n_dof, task_dim, n_constraints)

    # Try to build CusADi versions
    fi_pesns_cusadi = None
    pph_sns_cusadi = None

    if HAS_CUSADI:
        print("Building CusADi versions...")
        try:
            fi_pesns_cusadi = CusadiFunction(fi_pesns_fn, 1)  # Single instance for now
            print("✓ FI-PeSNS CusADi compiled")
        except Exception as e:
            print(f"✗ FI-PeSNS CusADi failed: {e}")

        try:
            pph_sns_cusadi = CusadiFunction(pph_sns_fn, 1)
            print("✓ PPH-SNS CusADi compiled")
        except Exception as e:
            print(f"✗ PPH-SNS CusADi failed: {e}")
    else:
        print("CusADi not available - skipping GPU tests")

    print()

    # Run benchmarks
    results = {
        'fi_pesns_cpu': [],
        'pph_sns_cpu': [],
        'fi_pesns_gpu': [],
        'pph_sns_gpu': [],
    }

    for i in range(n_problems):
        print(f"Problem {i+1}/{n_problems}...", end=' ')
        sys.stdout.flush()

        J, target, C, lower, upper = generate_test_problem(seed=i)

        # CPU benchmarks
        fi_cpu = benchmark_solver_cpu(fi_pesns_fn, J, target, C, lower, upper, n_runs)
        pph_cpu = benchmark_solver_cpu(pph_sns_fn, J, target, C, lower, upper, n_runs)

        results['fi_pesns_cpu'].append(fi_cpu)
        results['pph_sns_cpu'].append(pph_cpu)

        # GPU benchmarks
        if HAS_CUSADI:
            fi_gpu = benchmark_solver_gpu(fi_pesns_cusadi, J, target, C, lower, upper, n_runs)
            pph_gpu = benchmark_solver_gpu(pph_sns_cusadi, J, target, C, lower, upper, n_runs)

            if fi_gpu is not None:
                results['fi_pesns_gpu'].append(fi_gpu)
            if pph_gpu is not None:
                results['pph_sns_gpu'].append(pph_gpu)

        print("✓")

    # Analyze results
    print()
    print("=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    for solver_name, result_list in results.items():
        if not result_list:
            continue

        print(f"\n{solver_name.upper()}:")
        print("-" * 40)

        times = [r['time_avg'] * 1000 for r in result_list]  # Convert to ms
        max_violations = [r['max_violation'] for r in result_list]
        mean_violations = [r['mean_violation'] for r in result_list]
        task_scales = [r.get('task_scale', None) for r in result_list]
        task_scales = [s for s in task_scales if s is not None]

        print(f"  Time: {np.mean(times):.2f} ± {np.std(times):.2f} ms")
        print(f"  Max Violation: {np.mean(max_violations):.6f} ± {np.std(max_violations):.6f}")
        print(f"  Mean Violation: {np.mean(mean_violations):.6f} ± {np.std(mean_violations):.6f}")
        if task_scales:
            print(f"  Task Scale: {np.mean(task_scales):.3f} ± {np.std(task_scales):.3f}")

    # Direct comparison
    if results['fi_pesns_cpu'] and results['pph_sns_cpu']:
        print("\n" + "=" * 80)
        print("DIRECT COMPARISON (CPU)")
        print("=" * 80)

        fi_times = [r['time_avg'] * 1000 for r in results['fi_pesns_cpu']]
        pph_times = [r['time_avg'] * 1000 for r in results['pph_sns_cpu']]

        fi_viol = [r['max_violation'] for r in results['fi_pesns_cpu']]
        pph_viol = [r['max_violation'] for r in results['pph_sns_cpu']]

        print(f"  FI-PeSNS Time: {np.mean(fi_times):.2f} ms")
        print(f"  PPH-SNS Time: {np.mean(pph_times):.2f} ms")
        print(f"  FI-PeSNS Max Violation: {np.mean(fi_viol):.6f}")
        print(f"  PPH-SNS Max Violation: {np.mean(pph_viol):.6f}")

        # Ratios
        time_ratio = np.mean(pph_times) / np.mean(fi_times) if np.mean(fi_times) > 0 else 0
        viol_ratio = np.mean(pph_viol) / np.mean(fi_viol) if np.mean(fi_viol) > 1e-10 else 0

        print(f"  Time Ratio (PPH/FI): {time_ratio:.2f}x")
        if viol_ratio > 0:
            print(f"  Violation Ratio (PPH/FI): {viol_ratio:.2f}x")

    if results['fi_pesns_gpu'] and results['pph_sns_gpu']:
        print("\n" + "=" * 80)
        print("DIRECT COMPARISON (GPU)")
        print("=" * 80)

        fi_times = [r['time_avg'] * 1000 for r in results['fi_pesns_gpu']]
        pph_times = [r['time_avg'] * 1000 for r in results['pph_sns_gpu']]

        fi_viol = [r['max_violation'] for r in results['fi_pesns_gpu']]
        pph_viol = [r['max_violation'] for r in results['pph_sns_gpu']]

        print(f"  FI-PeSNS Time: {np.mean(fi_times):.2f} ms")
        print(f"  PPH-SNS Time: {np.mean(pph_times):.2f} ms")
        print(f"  FI-PeSNS Max Violation: {np.mean(fi_viol):.6f}")
        print(f"  PPH-SNS Max Violation: {np.mean(pph_viol):.6f}")

        # Ratios
        time_ratio = np.mean(pph_times) / np.mean(fi_times) if np.mean(fi_times) > 0 else 0
        print(f"  Time Ratio (PPH/FI): {time_ratio:.2f}x")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare FI-PeSNS vs PPH-SNS solvers")
    parser.add_argument("--problems", type=int, default=10, help="Number of test problems")
    parser.add_argument("--runs", type=int, default=50, help="Number of runs per problem")

    args = parser.parse_args()
    run_comparison(args.problems, args.runs)
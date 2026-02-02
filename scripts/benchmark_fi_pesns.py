#!/usr/bin/env python3
"""
Benchmark FI-PeSNS vs CPU solver for accuracy and performance.

This script compares:
1. FI-PeSNS (CasADi symbolic) vs CPU (EmbodiK C++)
2. Constraint satisfaction rates
3. Timing for batched evaluation
"""

import time
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

try:
    import embodik as eik
except ImportError:
    print("Error: embodik not installed")
    exit(1)


@dataclass
class BenchmarkResult:
    """Benchmark results."""
    solver_name: str
    n_samples: int
    n_dof: int
    task_dim: int
    mean_time_ms: float
    std_time_ms: float
    max_velocity_error: float
    mean_velocity_error: float
    max_constraint_violation: float
    constraint_satisfaction_rate: float
    mean_scale: float


def generate_batch_problems(
    n_samples: int,
    n_dof: int = 7,
    task_dim: int = 6,
    n_constraints: int = 7,
    seed: int = 42,
    bound_scale: float = 1.0,
) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """Generate batch of random IK problems."""
    rng = np.random.default_rng(seed)

    targets = [rng.uniform(-0.5, 0.5, task_dim).astype(np.float64) for _ in range(n_samples)]
    jacobians = [rng.uniform(-1.0, 1.0, (task_dim, n_dof)).astype(np.float64) for _ in range(n_samples)]
    C = np.eye(n_constraints, n_dof, dtype=np.float64)
    lower = np.full(n_constraints, -bound_scale, dtype=np.float64)
    upper = np.full(n_constraints, bound_scale, dtype=np.float64)

    return targets, jacobians, C, lower, upper


def benchmark_cpu(
    targets: List[np.ndarray],
    jacobians: List[np.ndarray],
    C: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> Tuple[List[np.ndarray], List[np.ndarray], float]:
    """Benchmark CPU solver."""
    n_samples = len(targets)
    velocities = []
    scales = []

    start = time.perf_counter()
    for i in range(n_samples):
        result = eik.computeMultiObjectiveVelocitySolutionEigen(
            [targets[i]], [np.asfortranarray(jacobians[i])], C, lower, upper
        )
        velocities.append(np.array(result.solution))
        scales.append(np.array(result.task_scales))
    elapsed_ms = (time.perf_counter() - start) * 1000

    return velocities, scales, elapsed_ms


def benchmark_fi_pesns(
    targets: List[np.ndarray],
    jacobians: List[np.ndarray],
    C: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    k_max: int = 10,
) -> Tuple[List[np.ndarray], List[np.ndarray], float]:
    """Benchmark FI-PeSNS solver."""
    try:
        from embodik.gpu.casadi_fi_pesns import build_fi_pesns_single_task
    except ImportError:
        print("Error: CasADi not available")
        return [], [], 0.0

    n_samples = len(targets)
    n_dof = jacobians[0].shape[1]
    task_dim = jacobians[0].shape[0]
    n_constraints = C.shape[0]

    # Build function once
    fn = build_fi_pesns_single_task(
        n_dof=n_dof,
        task_dim=task_dim,
        n_constraints=n_constraints,
        k_max=k_max,
    )

    velocities = []
    scales = []

    start = time.perf_counter()
    for i in range(n_samples):
        vel, sc = fn(targets[i], jacobians[i].flatten(), C, lower, upper)
        velocities.append(np.array(vel).flatten())
        scales.append(np.array(sc).flatten())
    elapsed_ms = (time.perf_counter() - start) * 1000

    return velocities, scales, elapsed_ms


def compute_metrics(
    cpu_velocities: List[np.ndarray],
    test_velocities: List[np.ndarray],
    test_scales: List[np.ndarray],
    C: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    constraint_tol: float = 1e-3,
) -> Tuple[float, float, float, float, float]:
    """Compute accuracy metrics."""
    if not test_velocities:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    velocity_errors = []
    constraint_violations = []
    satisfied = 0
    scales_sum = 0.0

    for i, (cpu_vel, test_vel, sc) in enumerate(zip(cpu_velocities, test_velocities, test_scales)):
        # Velocity error vs CPU
        err = np.max(np.abs(cpu_vel.ravel() - test_vel.ravel()))
        velocity_errors.append(err)

        # Constraint violation
        constraint_val = C @ test_vel
        viol_low = np.maximum(0, lower - constraint_val)
        viol_high = np.maximum(0, constraint_val - upper)
        max_viol = np.max(np.maximum(viol_low, viol_high))
        constraint_violations.append(max_viol)

        if max_viol < constraint_tol:
            satisfied += 1

        scales_sum += sc[0] if len(sc) > 0 else 1.0

    n = len(test_velocities)
    return (
        np.max(velocity_errors),
        np.mean(velocity_errors),
        np.max(constraint_violations),
        satisfied / n,
        scales_sum / n,
    )


def print_results(results: List[BenchmarkResult]):
    """Print benchmark results in a table."""
    print("\n" + "=" * 90)
    print("BENCHMARK RESULTS")
    print("=" * 90)
    print(f"{'Solver':<15} {'N':>6} {'Time(ms)':>12} {'Max Err':>10} {'Mean Err':>10} {'Max Viol':>10} {'Sat%':>8} {'Scale':>8}")
    print("-" * 90)

    for r in results:
        print(f"{r.solver_name:<15} {r.n_samples:>6} {r.mean_time_ms:>10.2f}±{r.std_time_ms:<5.2f} "
              f"{r.max_velocity_error:>10.4f} {r.mean_velocity_error:>10.4f} "
              f"{r.max_constraint_violation:>10.4f} {r.constraint_satisfaction_rate*100:>7.1f}% {r.mean_scale:>8.3f}")

    print("=" * 90)


def main():
    print("=" * 60)
    print("FI-PeSNS vs CPU Benchmark")
    print("=" * 60)

    # Test configurations
    configs = [
        {"n_samples": 100, "n_dof": 7, "task_dim": 6, "bound_scale": 1.0, "k_max": 10},
        {"n_samples": 100, "n_dof": 7, "task_dim": 6, "bound_scale": 1.0, "k_max": 15},
        {"n_samples": 100, "n_dof": 7, "task_dim": 6, "bound_scale": 10.0, "k_max": 10},  # Loose bounds
        {"n_samples": 500, "n_dof": 7, "task_dim": 6, "bound_scale": 1.0, "k_max": 10},
    ]

    all_results = []

    for config in configs:
        n_samples = config["n_samples"]
        n_dof = config["n_dof"]
        task_dim = config["task_dim"]
        bound_scale = config["bound_scale"]
        k_max = config["k_max"]

        print(f"\nConfig: N={n_samples}, n_dof={n_dof}, task_dim={task_dim}, bounds=[-{bound_scale}, {bound_scale}], k_max={k_max}")
        print("-" * 60)

        # Generate problems
        targets, jacobians, C, lower, upper = generate_batch_problems(
            n_samples=n_samples,
            n_dof=n_dof,
            task_dim=task_dim,
            bound_scale=bound_scale,
        )

        # Benchmark CPU
        print("  Running CPU solver...")
        cpu_vels, cpu_scales, cpu_time = benchmark_cpu(targets, jacobians, C, lower, upper)

        # CPU self-metrics (should be perfect)
        cpu_max_err, cpu_mean_err, cpu_max_viol, cpu_sat_rate, cpu_scale = compute_metrics(
            cpu_vels, cpu_vels, cpu_scales, C, lower, upper
        )
        all_results.append(BenchmarkResult(
            solver_name=f"CPU",
            n_samples=n_samples,
            n_dof=n_dof,
            task_dim=task_dim,
            mean_time_ms=cpu_time,
            std_time_ms=0.0,
            max_velocity_error=cpu_max_err,
            mean_velocity_error=cpu_mean_err,
            max_constraint_violation=cpu_max_viol,
            constraint_satisfaction_rate=cpu_sat_rate,
            mean_scale=cpu_scale,
        ))
        print(f"    CPU: {cpu_time:.2f}ms, sat_rate={cpu_sat_rate*100:.1f}%")

        # Benchmark FI-PeSNS
        print(f"  Running FI-PeSNS (k={k_max})...")
        pesns_vels, pesns_scales, pesns_time = benchmark_fi_pesns(
            targets, jacobians, C, lower, upper, k_max=k_max
        )

        if pesns_vels:
            pesns_max_err, pesns_mean_err, pesns_max_viol, pesns_sat_rate, pesns_scale = compute_metrics(
                cpu_vels, pesns_vels, pesns_scales, C, lower, upper
            )
            all_results.append(BenchmarkResult(
                solver_name=f"FI-PeSNS(k={k_max})",
                n_samples=n_samples,
                n_dof=n_dof,
                task_dim=task_dim,
                mean_time_ms=pesns_time,
                std_time_ms=0.0,
                max_velocity_error=pesns_max_err,
                mean_velocity_error=pesns_mean_err,
                max_constraint_violation=pesns_max_viol,
                constraint_satisfaction_rate=pesns_sat_rate,
                mean_scale=pesns_scale,
            ))
            print(f"    FI-PeSNS: {pesns_time:.2f}ms, max_err={pesns_max_err:.4f}, sat_rate={pesns_sat_rate*100:.1f}%")

    # Print summary table
    print_results(all_results)

    # Summary
    print("\nKEY FINDINGS:")
    print("- FI-PeSNS uses penalty-based constraint enforcement (simpler, GPU-friendly)")
    print("- Constraint satisfaction rate should be >95% for tight bounds")
    print("- Loose bounds (scale=10) should show near-perfect match to CPU")
    print("- FI-PeSNS is designed for GPU batched execution (CusADi compilation)")


if __name__ == "__main__":
    main()

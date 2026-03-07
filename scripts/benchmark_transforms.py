#!/usr/bin/env python3
"""
Benchmark native embodiK transform operations vs SciPy baseline.

Measures:
- Latency (control-loop style): single r2q, q2r, Rotation compose/inv/apply
- Throughput: batch matrix↔quat conversions (10k, 100k)

Run: pixi run benchmark-transforms
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

# Ensure embodik is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

import embodik as eik

HAS_SCIPY = False
try:
    from scipy.spatial.transform import Rotation as ScipyRotation

    HAS_SCIPY = True
except ImportError:
    pass

# Default iteration counts
LATENCY_ITERS = 50_000
THROUGHPUT_SIZES = [10_000, 100_000]


def _percentile(times, p):
    """Compute percentile in seconds."""
    return float(np.percentile(times, p)) * 1000  # ms


def _median_p95(times):
    """Return (median_ms, p95_ms)."""
    times_arr = np.array(times)
    return (
        float(np.median(times_arr)) * 1000,
        float(np.percentile(times_arr, 95)) * 1000,
    )


def benchmark_latency_r2q_native(n_iters=LATENCY_ITERS):
    """Latency: embodiK r2q (native)."""
    R = eik.SO3.Rz(0.5).as_matrix()
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        _ = eik.r2q(R, order="sxyz")
        times.append(time.perf_counter() - t0)
    return _median_p95(times)


def benchmark_latency_r2q_scipy(n_iters=LATENCY_ITERS):
    """Latency: SciPy Rotation.from_matrix().as_quat() (wxyz reorder)."""
    if not HAS_SCIPY:
        return None, None
    R = eik.SO3.Rz(0.5).as_matrix()
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        q = ScipyRotation.from_matrix(R).as_quat()  # xyzw
        _ = np.array([q[3], q[0], q[1], q[2]])  # to wxyz
        times.append(time.perf_counter() - t0)
    return _median_p95(times)


def benchmark_latency_q2r_native(n_iters=LATENCY_ITERS):
    """Latency: embodiK q2r (native)."""
    q = np.array([1.0, 0.0, 0.0, 0.0])
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        _ = eik.q2r(q, order="sxyz")
        times.append(time.perf_counter() - t0)
    return _median_p95(times)


def benchmark_latency_q2r_scipy(n_iters=LATENCY_ITERS):
    """Latency: SciPy Rotation.from_quat().as_matrix()."""
    if not HAS_SCIPY:
        return None, None
    q_xyzw = np.array([0.0, 0.0, 0.0, 1.0])
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        _ = ScipyRotation.from_quat(q_xyzw).as_matrix()
        times.append(time.perf_counter() - t0)
    return _median_p95(times)


def benchmark_latency_rotation_compose(n_iters=LATENCY_ITERS):
    """Latency: Rotation compose (R1 * R2)."""
    R1 = eik.SO3.Rx(0.3)
    R2 = eik.SO3.Rz(0.5)
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        _ = R1 * R2
        times.append(time.perf_counter() - t0)
    return _median_p95(times)


def benchmark_latency_rotation_inv(n_iters=LATENCY_ITERS):
    """Latency: Rotation.inv()."""
    R = eik.SO3.Ry(0.4)
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        _ = R.inv()
        times.append(time.perf_counter() - t0)
    return _median_p95(times)


def benchmark_latency_rotation_apply(n_iters=LATENCY_ITERS):
    """Latency: Rotation.apply(v)."""
    R = eik.SO3.Rz(0.5)
    v = np.array([1.0, 0.0, 0.0])
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        _ = R.apply(v)
        times.append(time.perf_counter() - t0)
    return _median_p95(times)


def benchmark_throughput_r2q_native(n):
    """Throughput: n x r2q (native)."""
    R = eik.SO3.Rz(0.5).as_matrix()
    t0 = time.perf_counter()
    for _ in range(n):
        _ = eik.r2q(R, order="sxyz")
    elapsed = time.perf_counter() - t0
    return n / elapsed


def benchmark_throughput_r2q_scipy(n):
    """Throughput: n x SciPy matrix->quat (wxyz)."""
    if not HAS_SCIPY:
        return None
    R = eik.SO3.Rz(0.5).as_matrix()
    t0 = time.perf_counter()
    for _ in range(n):
        q = ScipyRotation.from_matrix(R).as_quat()
        _ = np.array([q[3], q[0], q[1], q[2]])
    elapsed = time.perf_counter() - t0
    return n / elapsed


def benchmark_throughput_q2r_native(n):
    """Throughput: n x q2r (native)."""
    q = np.array([1.0, 0.0, 0.0, 0.0])
    t0 = time.perf_counter()
    for _ in range(n):
        _ = eik.q2r(q, order="sxyz")
    elapsed = time.perf_counter() - t0
    return n / elapsed


def benchmark_throughput_q2r_scipy(n):
    """Throughput: n x SciPy quat->matrix."""
    if not HAS_SCIPY:
        return None
    q = np.array([0.0, 0.0, 0.0, 1.0])
    t0 = time.perf_counter()
    for _ in range(n):
        _ = ScipyRotation.from_quat(q).as_matrix()
    elapsed = time.perf_counter() - t0
    return n / elapsed


def run_benchmarks(n_latency=LATENCY_ITERS, throughput_sizes=None):
    """Run all benchmarks and return results dict."""
    if throughput_sizes is None:
        throughput_sizes = THROUGHPUT_SIZES

    results = {
        "latency": {},
        "throughput": {},
    }

    # Latency
    med, p95 = benchmark_latency_r2q_native(n_latency)
    results["latency"]["r2q_native_median_us"] = med * 1000
    results["latency"]["r2q_native_p95_us"] = p95 * 1000

    if HAS_SCIPY:
        med, p95 = benchmark_latency_r2q_scipy(n_latency)
        results["latency"]["r2q_scipy_median_us"] = med * 1000
        results["latency"]["r2q_scipy_p95_us"] = p95 * 1000

    med, p95 = benchmark_latency_q2r_native(n_latency)
    results["latency"]["q2r_native_median_us"] = med * 1000
    results["latency"]["q2r_native_p95_us"] = p95 * 1000

    if HAS_SCIPY:
        med, p95 = benchmark_latency_q2r_scipy(n_latency)
        results["latency"]["q2r_scipy_median_us"] = med * 1000
        results["latency"]["q2r_scipy_p95_us"] = p95 * 1000

    med, p95 = benchmark_latency_rotation_compose(n_latency)
    results["latency"]["rotation_compose_median_us"] = med * 1000
    results["latency"]["rotation_compose_p95_us"] = p95 * 1000

    med, p95 = benchmark_latency_rotation_inv(n_latency)
    results["latency"]["rotation_inv_median_us"] = med * 1000
    results["latency"]["rotation_inv_p95_us"] = p95 * 1000

    med, p95 = benchmark_latency_rotation_apply(n_latency)
    results["latency"]["rotation_apply_median_us"] = med * 1000
    results["latency"]["rotation_apply_p95_us"] = p95 * 1000

    # Throughput
    for n in throughput_sizes:
        key = f"n={n}"
        results["throughput"][key] = {}
        results["throughput"][key]["r2q_native_ops_per_sec"] = benchmark_throughput_r2q_native(n)
        if HAS_SCIPY:
            results["throughput"][key]["r2q_scipy_ops_per_sec"] = benchmark_throughput_r2q_scipy(n)
        results["throughput"][key]["q2r_native_ops_per_sec"] = benchmark_throughput_q2r_native(n)
        if HAS_SCIPY:
            results["throughput"][key]["q2r_scipy_ops_per_sec"] = benchmark_throughput_q2r_scipy(n)

    return results


def print_results(results):
    """Print human-readable benchmark results."""
    print("=" * 60)
    print("EmbodiK Native Rotation Performance Benchmarks")
    print("=" * 60)
    print(f"SciPy available: {HAS_SCIPY}")
    print()

    print("--- Latency (control-loop style, median / p95 microseconds) ---")
    for k, v in results["latency"].items():
        if v is not None:
            print(f"  {k}: {v:.2f}")

    print()
    print("--- Throughput (ops/sec) ---")
    for size_key, size_results in results["throughput"].items():
        print(f"  {size_key}:")
        for k, v in size_results.items():
            if v is not None:
                print(f"    {k}: {v:,.0f}")

    print()
    if HAS_SCIPY:
        r2q_native = results["latency"].get("r2q_native_median_us")
        r2q_scipy = results["latency"].get("r2q_scipy_median_us")
        if r2q_native and r2q_scipy and r2q_scipy > 0:
            speedup = r2q_scipy / r2q_native
            print(f"r2q speedup (native vs SciPy): {speedup:.2f}x")
        q2r_native = results["latency"].get("q2r_native_median_us")
        q2r_scipy = results["latency"].get("q2r_scipy_median_us")
        if q2r_native and q2r_scipy and q2r_scipy > 0:
            speedup = q2r_scipy / q2r_native
            print(f"q2r speedup (native vs SciPy): {speedup:.2f}x")
    else:
        print("Install scipy for baseline comparison.")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Benchmark embodiK transform operations")
    parser.add_argument(
        "--latency-iters",
        type=int,
        default=LATENCY_ITERS,
        help=f"Latency benchmark iterations (default {LATENCY_ITERS})",
    )
    parser.add_argument(
        "--throughput",
        type=int,
        nargs="+",
        default=THROUGHPUT_SIZES,
        help=f"Throughput batch sizes (default {THROUGHPUT_SIZES})",
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="Save results to JSON file",
    )
    args = parser.parse_args()

    results = run_benchmarks(
        n_latency=args.latency_iters,
        throughput_sizes=args.throughput,
    )
    print_results(results)

    if args.json:
        import json

        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.json}")


if __name__ == "__main__":
    main()

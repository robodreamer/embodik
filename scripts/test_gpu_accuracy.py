#!/usr/bin/env python3
"""Test GPU solver accuracy against CPU."""

import numpy as np
from embodik.gpu_solver import solve_velocity_batched, DEFAULT_FN_PATH, _solve_cpu_sequential

# Small test
n_dof = 7
task_dim = 6
n = 5

print(f"Testing with {n} samples, n_dof={n_dof}, task_dim={task_dim}")
print(f"CasADi path: {DEFAULT_FN_PATH}")
print()

rng = np.random.default_rng(42)
targets = [rng.uniform(-0.5, 0.5, task_dim).astype(np.float64) for _ in range(n)]
# Jacobians: (task_dim, n_dof) matrices, flattened for GPU
jacobians_2d = [rng.uniform(-1, 1, (task_dim, n_dof)).astype(np.float64) for _ in range(n)]
jacobians_flat = [j.flatten() for j in jacobians_2d]  # For GPU
C_list = [np.eye(n_dof, dtype=np.float64) for _ in range(n)]
lower = [np.full(n_dof, -1.0, dtype=np.float64) for _ in range(n)]
upper = [np.full(n_dof, 1.0, dtype=np.float64) for _ in range(n)]

# CPU result - use 2D jacobians
print("Running CPU sequential...")
cpu_result = _solve_cpu_sequential(targets, jacobians_2d, C_list, lower, upper)
print(f"  CPU velocities shape: {cpu_result.velocities.shape}")
print(f"  CPU velocities[0]: {cpu_result.velocities[0]}")
print()

# GPU result - use flattened jacobians
print("Running GPU batched...")
gpu_result = solve_velocity_batched(
    targets, jacobians_flat, C_list, lower, upper,
    use_gpu=True, casadi_path=DEFAULT_FN_PATH
)
print(f"  GPU velocities shape: {gpu_result.velocities.shape}")
print(f"  GPU velocities[0]: {gpu_result.velocities[0]}")
print(f"  GPU status: {gpu_result.status}")
print()

# Error analysis
if gpu_result.status == "success":
    errors = np.abs(cpu_result.velocities - gpu_result.velocities)
    print("Error Analysis:")
    print(f"  Max error: {np.max(errors):.6f}")
    print(f"  Mean error: {np.mean(errors):.6f}")
    print(f"  Error per sample: {np.max(errors, axis=1)}")
else:
    print(f"GPU solve failed with status: {gpu_result.status}")

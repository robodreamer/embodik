#!/usr/bin/env python3
"""
GPU Batch Collision Detection Example

This example demonstrates GPU-accelerated batched collision detection
using NVIDIA Warp. It compares performance between CPU sequential and
GPU parallel collision queries.

Prerequisites:
    python -m pip install "embodik[gpu-collision]"

Usage:
    python examples/05_gpu_collision_batch.py [--batch_sizes 10 100 500]
"""

from __future__ import annotations

import argparse
import time
import textwrap
from typing import Tuple

import numpy as np


def create_test_robot(tmp_dir: str) -> Tuple[str, int]:
    """Create a test robot URDF for collision testing."""
    import os
    
    urdf_content = textwrap.dedent('''<?xml version="1.0"?>
<robot name="test_arm">
  <link name="base">
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><box size="0.1 0.1 0.1"/></geometry>
    </collision>
  </link>
  
  <link name="link1">
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.005" ixy="0" ixz="0" iyy="0.005" iyz="0" izz="0.005"/>
    </inertial>
    <collision>
      <origin xyz="0.1 0 0" rpy="0 0 0"/>
      <geometry><box size="0.08 0.08 0.08"/></geometry>
    </collision>
  </link>
  
  <link name="link2">
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.005" ixy="0" ixz="0" iyy="0.005" iyz="0" izz="0.005"/>
    </inertial>
    <collision>
      <origin xyz="0.1 0 0" rpy="0 0 0"/>
      <geometry><sphere radius="0.04"/></geometry>
    </collision>
  </link>
  
  <joint name="joint1" type="revolute">
    <parent link="base"/>
    <child link="link1"/>
    <origin xyz="0.05 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit effort="10" lower="-2.0" upper="2.0" velocity="1"/>
  </joint>
  
  <joint name="joint2" type="revolute">
    <parent link="link1"/>
    <child link="link2"/>
    <origin xyz="0.2 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit effort="10" lower="-2.0" upper="2.0" velocity="1"/>
  </joint>
</robot>
''')
    
    urdf_path = os.path.join(tmp_dir, "test_arm.urdf")
    with open(urdf_path, "w") as f:
        f.write(urdf_content)
    
    return urdf_path, 2  # n_dof


def generate_random_configurations(
    n_configs: int,
    n_dof: int,
    seed: int = 42,
) -> np.ndarray:
    """Generate random joint configurations."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.5, 1.5, (n_configs, n_dof)).astype(np.float64)


def benchmark_cpu_sequential(robot, q_batch):
    """Benchmark CPU sequential collision detection."""
    from embodik.gpu.warp_collision import compute_collision_distances_batched
    
    start = time.perf_counter()
    
    # Sequential single-config calls
    distances = []
    for i in range(q_batch.shape[0]):
        result = compute_collision_distances_batched(
            robot, q_batch[i:i+1], use_gpu=False
        )
        distances.append(result.distances[0])
    
    elapsed_ms = (time.perf_counter() - start) * 1000
    return elapsed_ms, np.array(distances)


def benchmark_cpu_batched(robot, q_batch):
    """Benchmark CPU batched collision detection."""
    from embodik.gpu.warp_collision import compute_collision_distances_batched
    
    start = time.perf_counter()
    result = compute_collision_distances_batched(robot, q_batch, use_gpu=False)
    elapsed_ms = (time.perf_counter() - start) * 1000
    
    return elapsed_ms, result.distances, result.status


def benchmark_gpu_batched(robot, q_batch):
    """Benchmark GPU batched collision detection."""
    from embodik.gpu.warp_collision import compute_collision_distances_batched
    
    start = time.perf_counter()
    result = compute_collision_distances_batched(robot, q_batch, use_gpu=True)
    elapsed_ms = (time.perf_counter() - start) * 1000
    
    return elapsed_ms, result.distances, result.status


def main():
    parser = argparse.ArgumentParser(description="GPU Batch Collision Benchmark")
    parser.add_argument(
        "--batch_sizes",
        type=int,
        nargs="+",
        default=[10, 50, 100, 500],
        help="Batch sizes to test",
    )
    args = parser.parse_args()
    
    print("=" * 70)
    print("EmbodiK GPU Batch Collision Benchmark")
    print("=" * 70)
    
    # Check availability
    try:
        from embodik.gpu.warp_collision import check_warp_availability
        status = check_warp_availability()
        print("Collision Backend Status:")
        for key, value in status.items():
            marker = "✓" if value else "✗"
            print(f"  {key}: {marker}")
        print()
    except ImportError as e:
        print(f"Warp collision module not available: {e}")
        return
    
    # Create test robot
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        urdf_path, n_dof = create_test_robot(tmp_dir)
        
        try:
            import embodik as eik
            robot = eik.RobotModel(urdf_path, floating_base=False)
        except Exception as e:
            print(f"Could not create robot model: {e}")
            return
        
        print(f"Robot: {n_dof} DOF test arm")
        print()
        
        # Run benchmarks
        print(f"{'Batch':<10} {'CPU Seq (ms)':<15} {'CPU Batch (ms)':<15} {'GPU (ms)':<12} {'Status':<15}")
        print("-" * 70)
        
        for batch_size in args.batch_sizes:
            q_batch = generate_random_configurations(batch_size, n_dof)
            
            # CPU sequential
            cpu_seq_time, cpu_seq_dists = benchmark_cpu_sequential(robot, q_batch)
            
            # CPU batched
            cpu_batch_time, cpu_batch_dists, cpu_status = benchmark_cpu_batched(robot, q_batch)
            
            # GPU batched
            try:
                gpu_time, gpu_dists, gpu_status = benchmark_gpu_batched(robot, q_batch)
                
                # Check accuracy
                if gpu_status == "success":
                    n_matched = np.sum(np.isclose(cpu_batch_dists, gpu_dists, atol=1e-3))
                    accuracy = f"{n_matched}/{batch_size}"
                else:
                    accuracy = gpu_status
                
                print(f"{batch_size:<10} {cpu_seq_time:<15.2f} {cpu_batch_time:<15.2f} {gpu_time:<12.2f} {accuracy:<15}")
                
            except Exception as e:
                print(f"{batch_size:<10} {cpu_seq_time:<15.2f} {cpu_batch_time:<15.2f} {'error':<12} {str(e)[:15]:<15}")
        
        print()
        print("Note: GPU times include CPU->GPU transfer overhead.")
        print("      For sustained batched processing, GPU benefits increase.")


if __name__ == "__main__":
    main()

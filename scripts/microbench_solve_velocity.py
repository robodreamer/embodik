import time
import argparse
import numpy as np

import embodik


def load_panda_urdf_path():
    module = __import__("robot_descriptions.panda_description", fromlist=["URDF_PATH"])
    return getattr(module, "URDF_PATH")


def benchmark_cpu_single(iters: int = 3000):
    """Benchmark single-solve CPU performance."""
    urdf_path = load_panda_urdf_path()

    robot = embodik.RobotModel(urdf_path)
    solver = embodik.KinematicsSolver(robot)
    solver.dt = 0.01
    solver.set_damping(0.1)

    # Panda default q (arm + gripper)
    q = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.05, 0.05], dtype=float)
    robot.update_configuration(q)

    frame_task = solver.add_frame_task("ee_task", "panda_hand")
    frame_task.priority = 0
    frame_task.weight = 1.0
    frame_task.set_target_velocity(np.array([0.1, 0, 0, 0, 0, 0], dtype=float))

    # Warmup
    for _ in range(100):
        solver.solve_velocity(q, apply_limits=True)

    t0 = time.perf_counter()
    last = None
    for _ in range(iters):
        last = solver.solve_velocity(q, apply_limits=True)
    t1 = time.perf_counter()

    avg_ms = (t1 - t0) * 1000.0 / iters
    print(f"avg solve_velocity wall time: {avg_ms:.3f} ms over {iters} iters")
    if last is not None:
        print(
            "last C++ breakdown (ms):",
            f"pin={last.pinocchio_kinematics_time_ms:.4f}",
            f"coll={getattr(last,'collision_constraint_time_ms',0.0):.4f}",
            f"task={last.task_update_time_ms:.4f}",
            f"constr={last.constraint_setup_time_ms:.4f}",
            f"solver={last.solver_computation_time_ms:.4f}",
            f"backend_total={last.computation_time_ms:.4f}",
        )
    
    return avg_ms


def benchmark_gpu_batch(batch_sizes: list = None, casadi_path: str = ""):
    """Benchmark GPU batched performance vs CPU sequential."""
    if batch_sizes is None:
        batch_sizes = [10, 100, 500, 1000]
    
    print("\n" + "=" * 70)
    print("GPU Batch Benchmark")
    print("=" * 70)
    
    # Check GPU availability
    try:
        from embodik.gpu_solver import check_gpu_availability, solve_velocity_batched
        gpu_status = check_gpu_availability()
        print("GPU Status:")
        for key, value in gpu_status.items():
            status = "✓" if value else "✗"
            print(f"  {key}: {status}")
        print()
    except ImportError as e:
        print(f"GPU solver not available: {e}")
        return
    
    n_dof = 7
    task_dim = 6
    
    print(f"{'Batch':<10} {'CPU (ms)':<12} {'GPU (ms)':<12} {'Speedup':<10} {'Status':<15}")
    print("-" * 60)
    
    for batch_size in batch_sizes:
        # Generate random problems
        rng = np.random.default_rng(42)
        
        targets = [rng.uniform(-0.5, 0.5, task_dim).astype(np.float64) for _ in range(batch_size)]
        jacobians = [rng.uniform(-1.0, 1.0, task_dim * n_dof).astype(np.float64) for _ in range(batch_size)]
        C_list = [np.eye(n_dof, dtype=np.float64) for _ in range(batch_size)]
        lower_list = [np.full(n_dof, -1.0, dtype=np.float64) for _ in range(batch_size)]
        upper_list = [np.full(n_dof, 1.0, dtype=np.float64) for _ in range(batch_size)]
        
        # CPU sequential
        t0 = time.perf_counter()
        for i in range(batch_size):
            embodik.computeMultiObjectiveVelocitySolutionEigen(
                [targets[i]],
                [jacobians[i].reshape(task_dim, n_dof)],
                C_list[i],
                lower_list[i],
                upper_list[i],
            )
        cpu_time = (time.perf_counter() - t0) * 1000
        
        # GPU batch
        try:
            t0 = time.perf_counter()
            result = solve_velocity_batched(
                targets, jacobians, C_list, lower_list, upper_list,
                use_gpu=True,
                casadi_path=casadi_path,
            )
            gpu_time = (time.perf_counter() - t0) * 1000
            
            if result.status == "success":
                speedup = cpu_time / gpu_time
                print(f"{batch_size:<10} {cpu_time:<12.2f} {gpu_time:<12.2f} {speedup:<10.1f}x {'GPU':<15}")
            else:
                print(f"{batch_size:<10} {cpu_time:<12.2f} {gpu_time:<12.2f} {'N/A':<10} {result.status:<15}")
        except Exception as e:
            print(f"{batch_size:<10} {cpu_time:<12.2f} {'error':<12} {'N/A':<10} {str(e)[:15]:<15}")


def main():
    parser = argparse.ArgumentParser(description="EmbodiK velocity solver microbenchmark")
    parser.add_argument("--iters", type=int, default=3000, help="Iterations for single-solve benchmark")
    parser.add_argument("--gpu", action="store_true", help="Include GPU batch benchmark")
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[10, 100, 500, 1000],
                        help="Batch sizes for GPU benchmark")
    parser.add_argument("--casadi_path", type=str, default="", help="Path to compiled .casadi function")
    args = parser.parse_args()
    
    print("=" * 70)
    print("EmbodiK Velocity Solver Microbenchmark")
    print("=" * 70)
    
    # Single-solve CPU benchmark
    print("\nSingle-solve CPU benchmark:")
    benchmark_cpu_single(args.iters)
    
    # GPU batch benchmark
    if args.gpu:
        benchmark_gpu_batch(args.batch_sizes, args.casadi_path)


if __name__ == "__main__":
    main()

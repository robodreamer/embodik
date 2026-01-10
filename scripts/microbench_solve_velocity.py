import time
import numpy as np

import embodik


def load_panda_urdf_path():
    module = __import__("robot_descriptions.panda_description", fromlist=["URDF_PATH"])
    return getattr(module, "URDF_PATH")


def main(iters: int = 3000):
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


if __name__ == "__main__":
    main()

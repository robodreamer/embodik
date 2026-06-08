"""Demo: the torso-vs-arms contribution knob on a bimanual robot (real solver).

Uses the production KinematicsSolver.set_joint_metric_weights on the RB-Y1
dual-arm + torso model and drives both end-effectors outward, printing how the
torso's share of the achieved motion shifts as the single global contribution
knob sweeps 0 -> 1. This is the headless counterpart to the bimanual viser
example's "Torso contribution" control.

RUN:
  cd ~/Projects/repos/embodik   # (or the worktree)
  pixi run python dev_examples/contribution_knob_demo.py
"""

from __future__ import annotations

import numpy as np

import embodik as eik

TORSO = [f"torso_{i}" for i in range(6)]
ARMS = {"left": ("left_arm_", "ee_left"), "right": ("right_arm_", "ee_right")}


def _vel_indices(model, names):
    return [model.get_joint_velocity_index(n) for n in names if model.has_joint(n)]


def _metric_vector(contribution, max_weight, torso_vi, arm_vi, nv):
    w = np.ones(nv)
    spread = (contribution - 0.5) * 2.0
    for i in torso_vi:
        w[i] = max_weight ** (-spread)
    for i in arm_vi:
        w[i] = max_weight ** (spread)
    return w


def main() -> None:
    from robot_descriptions.rby1_description import URDF_PATH

    model = eik.RobotModel(str(URDF_PATH))
    nv = max(model.get_joint_velocity_index(n) for n in model.get_joint_names()) + 1
    name_to_ci = {n: model.get_joint_config_index(n) for n in model.get_joint_names()}
    torso_vi = _vel_indices(model, TORSO)
    arm_names = [f"{ARMS['left'][0]}{i}" for i in range(7)] + [f"{ARMS['right'][0]}{i}" for i in range(7)]
    arm_vi = _vel_indices(model, arm_names)

    q0 = np.asarray(model.neutral_configuration(), dtype=float)
    for side, (prefix, _frame) in ARMS.items():
        for idx, val in {1: 0.4, 3: 0.8}.items():  # bend each arm off-singular
            n = f"{prefix}{idx}"
            if model.has_joint(n):
                q0[name_to_ci[n]] = val

    print(f"RB-Y1: torso {len(torso_vi)} DOF, arms {len(arm_vi)} DOF\n")
    print(f"  {'contribution':>12} | {'torso share of EE motion':>26}")
    print("  " + "-" * 44)
    for c in (0.0, 0.25, 0.5, 0.75, 1.0):
        solver = eik.KinematicsSolver(model)
        solver.dt = 0.02
        frames = {}
        for side, (_prefix, frame) in ARMS.items():
            t = solver.add_frame_task(f"ee_{side}", frame, eik.TaskType.FRAME_POSITION)
            t.priority = 0
            t.weight = 1.0
            frames[side] = frame
        solver.set_joint_metric_weights(_metric_vector(c, 8.0, torso_vi, arm_vi, nv))

        q = q0.copy()
        tot_t = tot_a = 0.0
        opts = eik.PositionStepOptions()
        opts.max_steps = 1
        opts.dt = 0.02
        for _ in range(40):
            model.update_configuration(q)
            jac = {s: model.get_frame_jacobian(f) for s, f in frames.items()}
            # command both EEs outward (+x, +y away from body)
            res = None
            for side, frame in frames.items():
                pose = model.get_frame_pose(frame)
                R = np.asarray(pose.rotation, dtype=float)
                tt = np.asarray(pose.translation, dtype=float)
                step = np.array([0.004, 0.004 if side == "left" else -0.004, 0.0])
                res = solver.solve_position_step(q, eik.SE3.Rt(R, tt + step), f"ee_{side}", opts)
                dq = np.asarray(res.joint_velocities, dtype=float)
                q = np.asarray(res.q_solution, dtype=float)
                tot_t += float(np.linalg.norm(dq[torso_vi]))
                tot_a += float(np.linalg.norm(dq[arm_vi]))
        share = tot_t / (tot_t + tot_a + 1e-12)
        print(f"  {c:>12.2f} | {share:>26.3f}")
    print("\n0 = arms do the work (torso suppressed); 1 = torso does the work.")


if __name__ == "__main__":
    main()

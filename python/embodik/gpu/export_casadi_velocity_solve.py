"""
Export CasADi velocity solve function for CusADi GPU compilation.

Run: python -m embodik.gpu.export_casadi_velocity_solve [options]

This creates a .casadi file that can be compiled to CUDA using cusadi:
    1. Move the .casadi file to cusadi/src/casadi_functions/
    2. Run: python run_codegen.py --fn=fn_velocity_solve

Example:
    # Export for 7-DOF robot with 2 tasks (6D + 3D)
    python -m embodik.gpu.export_casadi_velocity_solve \\
        --n_dof 7 --task_dims 6 3 --n_constraints 14 \\
        --out fn_velocity_solve.casadi
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

try:
    import casadi as ca
except ImportError:
    ca = None


def main():
    parser = argparse.ArgumentParser(
        description="Export CasADi velocity solve function for GPU compilation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--n_dof",
        type=int,
        default=7,
        help="Number of degrees of freedom (default: 7)",
    )
    parser.add_argument(
        "--task_dims",
        type=int,
        nargs="+",
        default=[6],
        help="Task dimensions, one per task (default: [6])",
    )
    parser.add_argument(
        "--n_constraints",
        type=int,
        default=None,
        help="Number of constraint rows (default: n_dof)",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-6,
        help="Numerical tolerance (default: 1e-6)",
    )
    parser.add_argument(
        "--regularization_factor",
        type=float,
        default=0.1,
        help="Regularization factor for pseudo-inverse (default: 0.1)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="fn_velocity_solve.casadi",
        help="Output .casadi file path (default: fn_velocity_solve.casadi)",
    )
    parser.add_argument(
        "--robot",
        type=str,
        default=None,
        choices=["panda", "ur5", "iiwa14"],
        help="Use predefined robot configuration (overrides n_dof, task_dims, n_constraints)",
    )

    args = parser.parse_args()

    if ca is None:
        print("Error: CasADi is required. Install with: pip install casadi", file=sys.stderr)
        sys.exit(1)

    # Import build function
    from embodik.gpu.casadi_velocity_solve import (
        build_velocity_solve_casadi,
        build_for_robot,
        ROBOT_CONFIGS,
    )

    # Build the function
    if args.robot:
        print(f"Using predefined robot configuration: {args.robot}")
        config = ROBOT_CONFIGS[args.robot]
        n_dof = config["n_dof"]
        task_dims = args.task_dims if args.task_dims != [6] else config["default_task_dims"]
        n_constraints = args.n_constraints or config["n_constraints"]

        fn = build_velocity_solve_casadi(
            n_dof=n_dof,
            n_tasks=len(task_dims),
            task_dims=task_dims,
            n_constraints=n_constraints,
            epsilon=args.epsilon,
            regularization_factor=args.regularization_factor,
        )
    else:
        n_constraints = args.n_constraints or args.n_dof

        fn = build_velocity_solve_casadi(
            n_dof=args.n_dof,
            n_tasks=len(args.task_dims),
            task_dims=args.task_dims,
            n_constraints=n_constraints,
            epsilon=args.epsilon,
            regularization_factor=args.regularization_factor,
        )
        n_dof = args.n_dof
        task_dims = args.task_dims

    # Save the function
    out_path = Path(args.out)
    fn.save(str(out_path))

    print(f"Saved CasADi function to: {out_path}")
    print(f"  n_dof: {n_dof}")
    print(f"  task_dims: {task_dims}")
    print(f"  n_tasks: {len(task_dims)}")
    print(f"  n_constraints: {n_constraints}")
    print(f"  epsilon: {args.epsilon}, regularization_factor: {args.regularization_factor}")
    print()
    print("Next steps:")
    print(f"  1. mv {out_path} cusadi/src/casadi_functions/")
    print(f"  2. cd cusadi && python run_codegen.py --fn={out_path.stem}")


if __name__ == "__main__":
    main()

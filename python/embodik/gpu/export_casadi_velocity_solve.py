"""
Export CasADi velocity solve function for CusADi GPU compilation.

Run: python -m embodik.gpu.export_casadi_velocity_solve [options]

This creates a .casadi file that can be compiled to CUDA using cusadi:
    1. Copy the .casadi file to cusadi/src/casadi_functions/
    2. Run: python run_codegen.py --fn=fn_velocity_solve

Uses FI-PeSNS (Fixed-Iteration Penalized eSNS) - a GPU-optimized solver with:
    - SRINV (Singularity-Robust Inverse) for stability
    - Analytical feasible scale computation
    - Penalty-based constraint enforcement

Example:
    # Export for 7-DOF Panda robot
    python -m embodik.gpu.export_casadi_velocity_solve --robot panda

    # Export for custom configuration
    python -m embodik.gpu.export_casadi_velocity_solve \\
        --n_dof 7 --task_dims 6 3 --n_constraints 14 --k_max 15 \\
        --out build/casadi/fn_velocity_solve.casadi
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import casadi as ca
except ImportError:
    ca = None


def main():
    parser = argparse.ArgumentParser(
        description="Export FI-PeSNS velocity solve function for GPU compilation",
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
        "--damping",
        type=float,
        default=0.1,
        help="SRINV damping factor (default: 0.1)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="build/casadi/fn_velocity_solve.casadi",
        help="Output .casadi file path (default: build/casadi/fn_velocity_solve.casadi)",
    )
    parser.add_argument(
        "--robot",
        type=str,
        default=None,
        choices=["panda", "ur5", "iiwa14", "humanoid_arm"],
        help="Use predefined robot configuration (overrides n_dof, task_dims, n_constraints)",
    )
    # FI-PeSNS specific args
    parser.add_argument(
        "--k_max",
        type=int,
        default=12,
        help="Fixed iterations (default: 12)",
    )
    parser.add_argument(
        "--mu0",
        type=float,
        default=1e-3,
        help="Initial penalty weight (default: 1e-3)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=2.5,
        help="Penalty growth factor (default: 2.5)",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.1,
        help="Penalty gradient step size (default: 0.1)",
    )
    parser.add_argument(
        "--warm-start",
        action="store_true",
        help="Enable warm-start input (adds prior_dq input)",
    )

    args = parser.parse_args()

    if ca is None:
        print("Error: CasADi is required. Install with: pip install casadi", file=sys.stderr)
        sys.exit(1)

    from embodik.gpu.casadi_fi_pesns import (
        build_fi_pesns_velocity_solve,
        ROBOT_CONFIGS,
    )

    print("Building FI-PeSNS velocity solver...")

    # Build the function
    if args.robot:
        print(f"Using predefined robot configuration: {args.robot}")
        config = ROBOT_CONFIGS[args.robot]
        n_dof = config["n_dof"]
        task_dims = args.task_dims if args.task_dims != [6] else config["default_task_dims"]
        n_constraints = args.n_constraints or config["n_constraints"]
    else:
        n_dof = args.n_dof
        task_dims = args.task_dims
        n_constraints = args.n_constraints or args.n_dof

    fn = build_fi_pesns_velocity_solve(
        n_dof=n_dof,
        n_tasks=len(task_dims),
        task_dims=task_dims,
        n_constraints=n_constraints,
        tol=args.epsilon,
        damping=args.damping,
        k_max=args.k_max,
        mu0=args.mu0,
        gamma=args.gamma,
        eta=args.eta,
        use_warm_start=args.warm_start,
    )

    # Save the function
    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fn.save(str(out_path))

    print(f"Saved CasADi function to: {out_path}")
    print(f"  n_dof: {n_dof}")
    print(f"  task_dims: {task_dims}")
    print(f"  n_tasks: {len(task_dims)}")
    print(f"  n_constraints: {n_constraints}")
    print(f"  epsilon: {args.epsilon}, damping: {args.damping}")
    print(f"  k_max: {args.k_max}, mu0: {args.mu0}, gamma: {args.gamma}, eta: {args.eta}")
    print(f"  warm_start: {args.warm_start}")
    print()
    print("Next steps:")
    cusadi_functions = Path.home() / ".local" / "cusadi" / "src" / "casadi_functions"
    print(f"  1. mkdir -p {cusadi_functions}")
    print(f"  2. cp {out_path} {cusadi_functions}/")
    print(f"  3. cd {cusadi_functions.parent.parent} && python run_codegen.py --fn={out_path.stem}")


if __name__ == "__main__":
    main()

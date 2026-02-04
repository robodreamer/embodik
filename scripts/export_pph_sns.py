#!/usr/bin/env python3
"""Export PPH-SNS solver to CasADi format for CusADi compilation."""

import os
import sys
from pathlib import Path

# Add the python path for embodik
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

try:
    import casadi as ca
    from embodik.gpu.casadi_pph_sns import build_pph_sns_single_task
except ImportError as e:
    print(f"Error importing: {e}")
    sys.exit(1)


def main():
    print("Building PPH-SNS solver...")

    # Panda robot configuration
    n_dof = 7
    task_dim = 6
    n_constraints = 7

    pph_sns_fn = build_pph_sns_single_task(n_dof, task_dim, n_constraints)

    # Export to CasADi format
    casadi_dir = Path.home() / ".local" / "cusadi" / "src" / "casadi_functions"
    casadi_dir.mkdir(parents=True, exist_ok=True)

    output_file = casadi_dir / "fn_pph_sns_velocity_solve.casadi"
    pph_sns_fn.save(str(output_file))

    print(f"Exported PPH-SNS to: {output_file}")
    print()
    print("To compile with CusADi, run:")
    print(f"  cd ~/.local/cusadi && python3 run_codegen.py --fn=fn_pph_sns_velocity_solve")


if __name__ == "__main__":
    main()
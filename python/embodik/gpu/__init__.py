"""GPU-accelerated solvers for EmbodiK.

**Experimental:** These solvers are under active development and need more validation.

This package provides GPU-parallel velocity IK solving using CusADi.
Two solvers are available:

- **FI-PeSNS** (Fixed-Iteration Penalized eSNS): Primary solver, penalty-based
- **PPH-SNS** (Parallel Penalized Hierarchical SNS): Alternative with soft top-k selection

Setup (one-time):
    1. Clone and install cusadi: git clone https://github.com/se-hwan/cusadi && pip install -e cusadi
    2. Export the CasADi velocity solve function:
       pixi run -e cuda export-casadi   # FI-PeSNS
       pixi run -e cuda export-pph-sns  # PPH-SNS
    3. Compile: cd ~/.local/cusadi && python run_codegen.py --fn=fn_velocity_solve
"""

from __future__ import annotations

# Check for CasADi availability
HAS_CASADI = False
try:
    import casadi as ca

    HAS_CASADI = True
except ImportError:
    pass

# Check for CusADi availability
HAS_CUSADI = False
CusadiFunction = None
try:
    # Try direct import (if cusadi is properly installed as a package)
    from cusadi import CusadiFunction

    HAS_CUSADI = True
except ImportError:
    try:
        # Try importing from src module (cusadi's internal structure)
        from src.CusadiFunction import CusadiFunction

        HAS_CUSADI = True
    except ImportError:
        try:
            # Try with CUSADI_ROOT environment variable
            import os
            import sys

            cusadi_root = os.environ.get("CUSADI_ROOT", "")
            if not cusadi_root:
                # Try common locations
                home = os.path.expanduser("~")
                for candidate in [
                    os.path.join(home, ".local", "cusadi"),
                    os.path.join(home, "cusadi"),
                    "/opt/cusadi",
                ]:
                    if os.path.isdir(candidate):
                        cusadi_root = candidate
                        break

            if cusadi_root:
                if cusadi_root not in sys.path:
                    sys.path.insert(0, cusadi_root)
                from src.CusadiFunction import CusadiFunction

                HAS_CUSADI = True
        except ImportError:
            pass

# Check for PyTorch CUDA
HAS_TORCH_CUDA = False
try:
    import torch

    HAS_TORCH_CUDA = torch.cuda.is_available()
except ImportError:
    pass

# Warp collision detection
HAS_WARP = False
try:
    import warp as wp

    HAS_WARP = True
except ImportError:
    pass

# Import solver builders if CasADi is available
if HAS_CASADI:
    from embodik.gpu.casadi_fi_pesns import (
        build_fi_pesns_for_robot,
        build_fi_pesns_single_task,
        build_fi_pesns_velocity_solve,
    )
    from embodik.gpu.casadi_pph_sns import (
        build_pph_sns_for_robot,
        build_pph_sns_single_task,
        build_pph_sns_velocity_solve,
    )

__all__ = [
    "HAS_CASADI",
    "HAS_CUSADI",
    "HAS_TORCH_CUDA",
    "HAS_WARP",
    "CusadiFunction",
]

if HAS_CASADI:
    __all__.extend(
        [
            "build_fi_pesns_velocity_solve",
            "build_fi_pesns_single_task",
            "build_fi_pesns_for_robot",
            "build_pph_sns_velocity_solve",
            "build_pph_sns_single_task",
            "build_pph_sns_for_robot",
        ]
    )

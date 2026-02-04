#!/usr/bin/env python3
"""
Test CusADi compilation for PPH-SNS solver.
"""

import numpy as np
import sys
import os
from pathlib import Path

# Add the python path for embodik
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

try:
    import casadi as ca
    from embodik.gpu.casadi_fi_pesns import build_fi_pesns_single_task
    from embodik.gpu.casadi_pph_sns import build_pph_sns_single_task
    HAS_CASADI = True
except ImportError:
    HAS_CASADI = False
    print("CasADi not available")
    sys.exit(1)

try:
    from cusadi import CusadiFunction
    HAS_CUSADI = True
except ImportError:
    HAS_CUSADI = False
    print("CusADi not available")
    sys.exit(1)


def test_function_compilation():
    """Test CusADi compilation of both solvers."""
    print("Testing CusADi compilation...")

    # Build CasADi functions
    n_dof, task_dim, n_constraints = 7, 6, 7

    print("Building FI-PeSNS solver...")
    fi_pesns_fn = build_fi_pesns_single_task(n_dof, task_dim, n_constraints)

    print("Building PPH-SNS solver...")
    pph_sns_fn = build_pph_sns_single_task(n_dof, task_dim, n_constraints)

    # Export to CasADi format
    print("Exporting CasADi functions...")

    # Create output directory
    casadi_dir = Path.home() / ".local" / "cusadi" / "src" / "casadi_functions"
    casadi_dir.mkdir(parents=True, exist_ok=True)

    fi_casadi_file = casadi_dir / "fn_fi_pesns_test.casadi"
    pph_casadi_file = casadi_dir / "fn_pph_sns_test.casadi"

    fi_pesns_fn.save(str(fi_casadi_file))
    pph_sns_fn.save(str(pph_casadi_file))

    print(f"Exported FI-PeSNS to: {fi_casadi_file}")
    print(f"Exported PPH-SNS to: {pph_casadi_file}")

    # Load and test CusADi compilation
    print("Testing CusADi compilation...")

    # Load CasADi functions
    fi_casadi_loaded = ca.Function.load(str(fi_casadi_file))
    pph_casadi_loaded = ca.Function.load(str(pph_casadi_file))

    # Test single instance compilation
    batch_size = 1
    try:
        fi_cusadi = CusadiFunction(fi_casadi_loaded, batch_size)
        print("✓ FI-PeSNS CusADi compilation successful")
    except Exception as e:
        print(f"✗ FI-PeSNS CusADi compilation failed: {e}")
        return False

    try:
        pph_cusadi = CusadiFunction(pph_casadi_loaded, batch_size)
        print("✓ PPH-SNS CusADi compilation successful")
    except Exception as e:
        print(f"✗ PPH-SNS CusADi compilation failed: {e}")
        return False

    # Test batched compilation
    batch_size = 100
    try:
        fi_cusadi_batch = CusadiFunction(fi_casadi_loaded, batch_size)
        print(f"✓ FI-PeSNS CusADi batch compilation successful (batch_size={batch_size})")
    except Exception as e:
        print(f"✗ FI-PeSNS CusADi batch compilation failed: {e}")

    try:
        pph_cusadi_batch = CusadiFunction(pph_casadi_loaded, batch_size)
        print(f"✓ PPH-SNS CusADi batch compilation successful (batch_size={batch_size})")
    except Exception as e:
        print(f"✗ PPH-SNS CusADi batch compilation failed: {e}")

    # Test function evaluation
    print("Testing function evaluation...")

    # Generate test inputs
    rng = np.random.RandomState(42)
    target = rng.randn(6).astype(np.float64) * 0.1
    J = rng.randn(6, 7).astype(np.float64)
    J_flat = J.flatten()
    C = np.eye(7).astype(np.float64)
    lower = np.full(7, -2.0).astype(np.float64)
    upper = np.full(7, 2.0).astype(np.float64)

    # Test FI-PeSNS
    try:
        fi_result = fi_cusadi.evaluate([target, J_flat, C, lower, upper])
        fi_velocity = fi_cusadi.getDenseOutput(0).flatten()
        print(f"✓ FI-PeSNS evaluation successful, output shape: {fi_velocity.shape}")
    except Exception as e:
        print(f"✗ FI-PeSNS evaluation failed: {e}")
        return False

    # Test PPH-SNS
    try:
        pph_result = pph_cusadi.evaluate([target, J_flat, C, lower, upper])
        pph_velocity = pph_cusadi.getDenseOutput(0).flatten()
        print(f"✓ PPH-SNS evaluation successful, output shape: {pph_velocity.shape}")
    except Exception as e:
        print(f"✗ PPH-SNS evaluation failed: {e}")
        return False

    # Compare results
    velocity_diff = np.linalg.norm(fi_velocity - pph_velocity)
    print(f"Velocity difference (FI-PeSNS vs PPH-SNS): {velocity_diff:.6f}")

    # Check constraint satisfaction
    fi_violation = np.maximum(0, lower - C @ fi_velocity) + np.maximum(0, C @ fi_velocity - upper)
    pph_violation = np.maximum(0, lower - C @ pph_velocity) + np.maximum(0, C @ pph_velocity - upper)

    fi_max_viol = np.max(fi_violation)
    pph_max_viol = np.max(pph_violation)

    print(f"FI-PeSNS max constraint violation: {fi_max_viol:.6f}")
    print(f"PPH-SNS max constraint violation: {pph_max_viol:.6f}")

    print("✓ All tests passed!")
    return True


if __name__ == "__main__":
    success = test_function_compilation()
    sys.exit(0 if success else 1)
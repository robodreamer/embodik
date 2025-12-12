#!/usr/bin/env python3
"""Test C++ extension import"""

import sys

print("Testing C++ extension import...")

try:
    import embodik._embodik_impl as impl
    print("✓ C++ extension imported directly!")
    print(f"  Available symbols: {len(dir(impl))} items")

    # Check for key classes
    if hasattr(impl, 'RobotModel'):
        print("  ✓ RobotModel found")
    if hasattr(impl, 'KinematicsSolver'):
        print("  ✓ KinematicsSolver found")
    if hasattr(impl, 'FrameTask'):
        print("  ✓ FrameTask found")

except ImportError as e:
    print(f"✗ Failed to import C++ extension: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test through main module
try:
    import embodik
    print(f"\nTesting through main module...")
    print(f"  C++ extension available: {embodik._cpp_extension_available}")

    if embodik._cpp_extension_available:
        if hasattr(embodik, 'RobotModel'):
            print("  ✓ RobotModel available")
        if hasattr(embodik, 'KinematicsSolver'):
            print("  ✓ KinematicsSolver available")
    else:
        print("  ⚠ C++ extension not marked as available")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

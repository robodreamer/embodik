#!/usr/bin/env python3
"""Test that embodik package imports and basic functionality works"""

import sys

print("=" * 60)
print("Testing embodik Package Import")
print("=" * 60)

try:
    import embodik
    print(f"✓ Package imported successfully")
    print(f"  Version: {embodik.__version__}")
except ImportError as e:
    print(f"✗ Failed to import embodik: {e}")
    sys.exit(1)

# Test C++ extension availability
try:
    has_cpp = embodik._cpp_extension_available
    print(f"✓ C++ extension available: {has_cpp}")
except AttributeError:
    print("⚠ C++ extension availability not reported")

# Test main classes
classes_to_check = [
    "RobotModel",
    "KinematicsSolver",
    "FrameTask",
    "PostureTask",
    "COMTask",
    "JointTask",
    "MultiJointTask",
]

print("\nChecking main classes:")
for cls_name in classes_to_check:
    if hasattr(embodik, cls_name):
        print(f"  ✓ {cls_name}")
    else:
        print(f"  ✗ {cls_name} - NOT FOUND")

# Test utility functions
print("\nChecking utility functions:")
utils_to_check = [
    "get_pose_error_vector",
    "r2q",
    "q2r",
    "Rt",
]

for util_name in utils_to_check:
    if hasattr(embodik, util_name):
        print(f"  ✓ {util_name}")
    else:
        print(f"  ✗ {util_name} - NOT FOUND")

print("\n" + "=" * 60)
print("Import test completed!")
print("=" * 60)

#!/usr/bin/env python3
"""Test Python utilities without requiring C++ extension"""

import sys
import os

# Add python directory to path (scripts/ is now in root, so go up one level)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import numpy as np

print("=" * 60)
print("Testing embodik Python Utilities")
print("=" * 60)

# Test imports
print("\n1. Testing imports...")
try:
    from embodik.utils import r2q, q2r, Rt, get_pose_error_vector
    print("   ✓ Successfully imported r2q, q2r, Rt, get_pose_error_vector")
except ImportError as e:
    print(f"   ✗ Import failed: {e}")
    sys.exit(1)

# Test r2q
print("\n2. Testing r2q (rotation matrix to quaternion)...")
try:
    R = np.eye(3)
    q = r2q(R)
    assert q.shape == (4,), f"Expected shape (4,), got {q.shape}"
    assert np.allclose(q, [1, 0, 0, 0]), f"Expected [1,0,0,0], got {q}"
    print(f"   ✓ r2q works: {q}")

    # Test with order parameter
    q_xyzs = r2q(R, order='xyzs')
    assert np.allclose(q_xyzs, [0, 0, 0, 1]), f"Expected [0,0,0,1] for xyzs, got {q_xyzs}"
    print(f"   ✓ r2q with order='xyzs' works: {q_xyzs}")
except Exception as e:
    print(f"   ✗ r2q failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test q2r
print("\n3. Testing q2r (quaternion to rotation matrix)...")
try:
    q = np.array([1, 0, 0, 0])  # Identity quaternion (wxyz)
    R = q2r(q)
    assert R.shape == (3, 3), f"Expected shape (3,3), got {R.shape}"
    assert np.allclose(R, np.eye(3)), f"Expected identity matrix, got {R}"
    print(f"   ✓ q2r works: R is identity matrix")

    # Test with xyzs order
    q_xyzs = np.array([0, 0, 0, 1])  # Identity quaternion (xyzw)
    R2 = q2r(q_xyzs, order='xyzs')
    assert np.allclose(R2, np.eye(3)), f"Expected identity matrix for xyzs, got {R2}"
    print(f"   ✓ q2r with order='xyzs' works")

    # Test round-trip
    R_test = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])  # 90 deg rotation around z
    q_test = r2q(R_test)
    R_recovered = q2r(q_test)
    assert np.allclose(R_test, R_recovered), "Round-trip failed"
    print(f"   ✓ Round-trip (R -> q -> R) works")
except Exception as e:
    print(f"   ✗ q2r failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test Rt
print("\n4. Testing Rt (create SE3 from R and t)...")
try:
    import pinocchio as pin
    R = np.eye(3)
    t = np.array([1, 2, 3])
    T = Rt(R=R, t=t)
    assert isinstance(T, pin.SE3), f"Expected pin.SE3, got {type(T)}"
    assert np.allclose(T.translation, t), f"Expected translation {t}, got {T.translation}"
    assert np.allclose(T.rotation, R), f"Expected rotation {R}, got {T.rotation}"
    print(f"   ✓ Rt works: translation={T.translation}, rotation is identity")

    # Test defaults
    T_default = Rt()
    assert np.allclose(T_default.translation, [0, 0, 0])
    assert np.allclose(T_default.rotation, np.eye(3))
    print(f"   ✓ Rt with defaults works")
except Exception as e:
    print(f"   ✗ Rt failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test get_pose_error_vector
print("\n5. Testing get_pose_error_vector...")
try:
    import pinocchio as pin
    T1 = pin.SE3(np.eye(3), np.array([0, 0, 0]))
    T2 = pin.SE3(np.eye(3), np.array([1, 2, 3]))
    error = get_pose_error_vector(T1, T2)
    assert error.shape == (6,), f"Expected shape (6,), got {error.shape}"
    assert np.allclose(error[:3], [1, 2, 3]), f"Expected translation error [1,2,3], got {error[:3]}"
    print(f"   ✓ get_pose_error_vector works: error={error}")
except Exception as e:
    print(f"   ✗ get_pose_error_vector failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ All Python utility tests passed!")
print("=" * 60)
print("\nNote: C++ extension tests require the package to be built first.")
print("Run: pixi run install  (after fixing build issues)")

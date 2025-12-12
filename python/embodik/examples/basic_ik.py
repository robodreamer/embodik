"""Basic IK example for embodiK.

This is a minimal example demonstrating basic inverse kinematics solving.
"""

import numpy as np
import embodik


def basic_ik_example():
    """Run a basic IK example."""
    print("=" * 60)
    print("embodiK Basic IK Example")
    print("=" * 60)

    # Load robot model (requires URDF file)
    # For this example, we'll use a simple demonstration
    try:
        # Try to load a robot model
        # In a real scenario, you would provide a path to a URDF file
        print("\nNote: This example requires a URDF file.")
        print("To run a complete example, use:")
        print("  python examples/01_basic_ik_simple.py")
        print("\nThis demonstrates the basic API structure:")

        # Example API usage (conceptual)
        print("\n1. Create robot model:")
        print("   model = embodik.RobotModel.from_urdf('path/to/robot.urdf')")

        print("\n2. Create kinematics solver:")
        print("   solver = embodik.KinematicsSolver(model)")

        print("\n3. Solve position IK:")
        print("   target_pose = np.eye(4)  # 4x4 transformation matrix")
        print("   result = solver.solve_position_ik(")
        print("       target_pose=target_pose,")
        print("       initial_q=np.zeros(model.nq)")
        print("   )")

        print("\n4. Check result:")
        print("   if result.status == embodik.SolverStatus.SUCCESS:")
        print("       print(f'Solution: {result.solution}')")

        print("\n" + "=" * 60)
        print("For a complete working example, see:")
        print("  examples/01_basic_ik_simple.py")
        print("=" * 60)

    except Exception as e:
        print(f"Error: {e}")
        print("\nThis is a demonstration. For a complete example,")
        print("see examples/01_basic_ik_simple.py")


if __name__ == "__main__":
    basic_ik_example()

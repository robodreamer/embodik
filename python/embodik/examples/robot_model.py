"""Robot model example for embodiK.

This example demonstrates how to load and work with robot models.
"""

import numpy as np
import embodik


def robot_model_example():
    """Run a robot model example."""
    print("=" * 60)
    print("embodiK Robot Model Example")
    print("=" * 60)

    print("\nThis example demonstrates robot model usage:")
    print("\n1. Load robot from URDF:")
    print("   model = embodik.RobotModel.from_urdf('path/to/robot.urdf')")

    print("\n2. Access robot properties:")
    print("   print(f'Number of joints: {model.nq}')")
    print("   print(f'Number of frames: {model.nframes}')")

    print("\n3. Update configuration:")
    print("   q = np.zeros(model.nq)")
    print("   model.update_configuration(q)")

    print("\n4. Get frame pose:")
    print("   pose = model.get_frame_pose('end_effector')")

    print("\n5. Get joint limits:")
    print("   q_lower, q_upper = model.get_joint_limits()")

    print("\n" + "=" * 60)
    print("For a complete working example, see:")
    print("  examples/robot_model_example.py")
    print("=" * 60)


if __name__ == "__main__":
    robot_model_example()

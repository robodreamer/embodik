#!/usr/bin/env python3
"""
Example demonstrating RobotModel usage with Pinocchio integration.

This shows how to:
- Load a robot model from URDF
- Update configuration
- Get frame poses and Jacobians
- Compute center of mass
"""

import numpy as np
import sys
import os

try:
    import embodik
except ImportError:
    print("Error: Could not import embodik module.")
    print("Please build the project first: cd .. && ./build.sh")
    sys.exit(1)


def create_simple_robot_urdf():
    """Create a simple 2-DOF robot URDF for demonstration"""
    urdf_content = """<?xml version="1.0"?>
<robot name="simple_arm">
  <link name="base_link">
    <inertial>
      <mass value="5.0"/>
      <origin xyz="0 0 0.5"/>
      <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
    </inertial>
    <visual>
      <geometry>
        <cylinder length="1.0" radius="0.1"/>
      </geometry>
      <origin xyz="0 0 0.5"/>
    </visual>
  </link>

  <link name="link1">
    <inertial>
      <mass value="2.0"/>
      <origin xyz="0 0 0.5"/>
      <inertia ixx="0.5" ixy="0" ixz="0" iyy="0.5" iyz="0" izz="0.1"/>
    </inertial>
    <visual>
      <geometry>
        <cylinder length="1.0" radius="0.05"/>
      </geometry>
      <origin xyz="0 0 0.5"/>
    </visual>
  </link>

  <joint name="shoulder" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <origin xyz="0 0 1" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14159" upper="3.14159" velocity="2.0" effort="50.0"/>
  </joint>

  <link name="link2">
    <inertial>
      <mass value="1.0"/>
      <origin xyz="0 0 0.5"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.05"/>
    </inertial>
    <visual>
      <geometry>
        <cylinder length="1.0" radius="0.03"/>
      </geometry>
      <origin xyz="0 0 0.5"/>
    </visual>
  </link>

  <joint name="elbow" type="revolute">
    <parent link="link1"/>
    <child link="link2"/>
    <origin xyz="0 0 1" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.5" upper="2.5" velocity="2.0" effort="30.0"/>
  </joint>

  <link name="end_effector">
    <inertial>
      <mass value="0.5"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>

  <joint name="ee_fixed" type="fixed">
    <parent link="link2"/>
    <child link="end_effector"/>
    <origin xyz="0 0 1" rpy="0 0 0"/>
  </joint>
</robot>"""

    # Save to temporary file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.urdf', delete=False) as f:
        f.write(urdf_content)
        return f.name


def main():
    print("=== Swift IK Robot Model Example ===\n")

    # Create a simple robot URDF
    urdf_path = create_simple_robot_urdf()
    print(f"Created temporary URDF: {urdf_path}")

    try:
        # Load robot model
        print("\n1. Loading robot model...")
        robot = embodik.RobotModel(urdf_path, floating_base=False)
        print(f"   Robot loaded: {robot}")
        print(f"   Configuration space dimension (nq): {robot.nq}")
        print(f"   Velocity space dimension (nv): {robot.nv}")

        # Get robot information
        print("\n2. Robot structure:")
        print(f"   Frames: {robot.get_frame_names()}")
        print(f"   Joints: {robot.get_joint_names()}")

        # Get joint limits
        lower, upper = robot.get_joint_limits()
        print(f"\n3. Joint limits:")
        for i, name in enumerate(robot.get_joint_names()):
            print(f"   {name}: [{lower[i]:.3f}, {upper[i]:.3f}] rad")

        # Update configuration
        print("\n4. Forward kinematics:")
        q = np.array([np.pi/4, -np.pi/6])  # 45 deg shoulder, -30 deg elbow
        print(f"   Setting configuration: q = {q}")
        robot.update_configuration(q)

        # Get end effector pose
        ee_pose = robot.get_frame_pose("end_effector")
        print(f"\n5. End effector pose:")
        print(f"   Position: {ee_pose.translation}")
        print(f"   Rotation matrix:\n{ee_pose.rotation}")

        # Get Jacobian
        print("\n6. End effector Jacobian:")
        J = robot.get_frame_jacobian("end_effector")
        print(f"   Shape: {J.shape}")
        print(f"   Linear velocity part (top 3 rows):\n{J[:3, :]}")
        print(f"   Angular velocity part (bottom 3 rows):\n{J[3:, :]}")

        # Center of mass
        print("\n7. Center of mass:")
        com = robot.get_com_position()
        print(f"   COM position: {com}")
        J_com = robot.get_com_jacobian()
        print(f"   COM Jacobian shape: {J_com.shape}")

        # Test different configurations
        print("\n8. Testing motion:")
        configurations = [
            ([0, 0], "Home position"),
            ([np.pi/2, 0], "Shoulder 90°"),
            ([0, np.pi/2], "Elbow 90°"),
            ([np.pi/4, np.pi/4], "Both 45°")
        ]

        for q_test, desc in configurations:
            robot.update_configuration(np.array(q_test))
            ee_pos = robot.get_frame_pose("end_effector").translation
            print(f"   {desc}: EE at {ee_pos}")

        print("\n✅ Example completed successfully!")

    finally:
        # Clean up
        os.unlink(urdf_path)
        print(f"\nCleaned up temporary file: {urdf_path}")


if __name__ == "__main__":
    main()

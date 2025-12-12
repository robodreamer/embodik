#!/usr/bin/env python3
"""
Example demonstrating Viser visualization integration with embodiK.

This shows how to:
- Load a robot model from URDF
- Visualize the robot using embodiKVisualizer
- Add IK target markers
- Control joint configuration with sliders
- Use InteractiveVisualizer for draggable targets
"""

import numpy as np
import sys
import os
import time
import tempfile

import embodik
from embodik import EmbodikVisualizer, InteractiveVisualizer


def create_test_urdf():
    """Create a simple test URDF file with visual geometries."""
    urdf_content = """<?xml version="1.0"?>
<robot name="test_robot">
  <link name="base_link">
    <visual>
      <geometry><box size="0.1 0.1 0.1"/></geometry>
      <origin xyz="0 0 0.05"/>
      <material name="red"><color rgba="1 0 0 1"/></material>
    </visual>
    <inertial><mass value="0.1"/><origin xyz="0 0 0"/><inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/></inertial>
  </link>
  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <origin xyz="0 0 0.1"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="100" velocity="100"/>
  </joint>
  <link name="link1">
    <visual>
      <geometry><cylinder radius="0.02" length="0.2"/></geometry>
      <origin xyz="0 0 0.1"/>
      <material name="blue"><color rgba="0 0 1 1"/></material>
    </visual>
    <inertial><mass value="0.1"/><origin xyz="0 0 0"/><inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/></inertial>
  </link>
  <joint name="joint2" type="revolute">
    <parent link="link1"/>
    <child link="link2"/>
    <origin xyz="0 0 0.2"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.5" upper="2.5" effort="100" velocity="100"/>
  </joint>
  <link name="link2">
    <visual>
      <geometry><box size="0.05 0.05 0.05"/></geometry>
      <origin xyz="0 0 0.025"/>
      <material name="green"><color rgba="0 1 0 1"/></material>
    </visual>
    <inertial><mass value="0.1"/><origin xyz="0 0 0"/><inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/></inertial>
  </link>
  <joint name="joint_ee" type="fixed">
    <parent link="link2"/>
    <child link="end_effector"/>
    <origin xyz="0 0 0.05"/>
  </joint>
  <link name="end_effector">
    <visual>
      <geometry><sphere radius="0.03"/></geometry>
      <origin xyz="0 0 0"/>
      <material name="yellow"><color rgba="1 1 0 1"/></material>
    </visual>
    <inertial><mass value="0.01"/><origin xyz="0 0 0"/><inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/></inertial>
  </link>
</robot>
"""
    fd, path = tempfile.mkstemp(suffix=".urdf")
    with os.fdopen(fd, 'w') as f:
        f.write(urdf_content)
    return path


def basic_visualization_example():
    """Basic visualization example with embodiKVisualizer."""
    print("\n" + "="*60)
    print("embodiK Basic Visualization Example")
    print("="*60)

    # Create test URDF
    urdf_path = create_test_urdf()
    print(f"Created test URDF: {urdf_path}")

    try:
        # 1. Load robot model
        print("\n1. Loading robot model...")
        robot = embodik.RobotModel(urdf_path, floating_base=False)
        print(f"   Loaded robot with nq={robot.nq}, nv={robot.nv}")

        # Set up controlled joints for visualization
        robot.controlled_joint_names = ["joint1", "joint2"]
        robot.controlled_joint_indices = {"joint1": 0, "joint2": 1}

        # 2. Create visualizer
        print("\n2. Starting visualization...")
        viz = EmbodikVisualizer(robot, port=8080)
        print("   Viser server is running - open http://localhost:8080 in your browser")

        # 3. Display initial configuration
        q = np.array([0.0, 0.0])
        viz.display(q)

        # 4. Add IK targets
        print("\n3. Adding IK targets...")

        # Target 1: End effector position
        target_pose = np.eye(4)
        target_pose[:3, 3] = [0.15, 0.1, 0.3]
        viz.add_target_marker("ee_target", target_pose, color=(0, 1, 0))

        # Target 2: Another target
        target_pose2 = np.eye(4)
        target_pose2[:3, 3] = [-0.1, 0.15, 0.25]
        viz.add_target_marker("secondary_target", target_pose2, color=(1, 0, 0))

        # 5. Animate the robot
        print("\n4. Animating robot...")
        print("   Press Ctrl+C to stop")

        t = 0
        try:
            while True:
                # Simple joint motion
                q[0] = 0.5 * np.sin(t)
                q[1] = 0.5 * np.cos(t)

                # Update display
                viz.display(q)

                # Update COM visualization
                com_pos = robot.get_com_position()
                viz.visualize_com(com_pos)

                # Update time
                t += 0.01
                time.sleep(0.01)

        except KeyboardInterrupt:
            print("\n   Animation stopped")

    finally:
        # Clean up
        if os.path.exists(urdf_path):
            os.remove(urdf_path)
        print("\nCleaned up temporary files")


def interactive_visualization_example():
    """Interactive visualization example with draggable targets."""
    print("\n" + "="*60)
    print("embodiK Interactive Visualization Example")
    print("="*60)

    # Create test URDF
    urdf_path = create_test_urdf()
    print(f"Created test URDF: {urdf_path}")

    try:
        # 1. Load robot model
        print("\n1. Loading robot model...")
        robot = embodik.RobotModel(urdf_path, floating_base=False)
        print(f"   Loaded robot with nq={robot.nq}, nv={robot.nv}")

        # Set up controlled joints for visualization
        robot.controlled_joint_names = ["joint1", "joint2"]
        robot.controlled_joint_indices = {"joint1": 0, "joint2": 1}

        # 2. Create interactive visualizer
        print("\n2. Starting interactive visualization...")
        viz = InteractiveVisualizer(robot, port=8081)
        print("   Viser server is running - open http://localhost:8081 in your browser")

        # 3. Display initial configuration
        q = np.array([0.0, 0.0])
        viz.display(q)

        # 4. Add interactive target for end-effector
        print("\n3. Adding interactive target...")
        initial_ee_pose = robot.get_frame_pose("end_effector")
        initial_pose = np.eye(4)
        initial_pose[:3, :3] = initial_ee_pose.rotation
        initial_pose[:3, 3] = initial_ee_pose.translation

        # Add callback to print target position
        def target_callback(pose):
            print(f"Target moved to: {pose[:3, 3]}")

        viz.add_interactive_target("ee_target", initial_pose,
                                  callback=target_callback,
                                  color=(1, 0, 0))

        print("\n4. You can now drag the red target in the browser!")
        print("   Target position will be printed when moved")
        print("   Press Ctrl+C to exit")

        # Keep the visualization running
        try:
            viz.wait()
        except KeyboardInterrupt:
            print("\n   Visualization stopped")

    finally:
        # Clean up
        if os.path.exists(urdf_path):
            os.remove(urdf_path)
        print("\nCleaned up temporary files")


def main():
    """Run visualization examples."""
    print("embodiK Viser Visualization Examples")
    print("====================================")
    print("1. Basic visualization (animated robot)")
    print("2. Interactive visualization (draggable targets)")
    print("q. Quit")

    while True:
        choice = input("\nSelect example (1/2/q): ").strip().lower()

        if choice == '1':
            basic_visualization_example()
        elif choice == '2':
            interactive_visualization_example()
        elif choice == 'q':
            break
        else:
            print("Invalid choice. Please try again.")

    print("\nGoodbye!")


if __name__ == "__main__":
    main()
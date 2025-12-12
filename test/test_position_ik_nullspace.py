#!/usr/bin/env python3
"""
Test position IK with nullspace control features
"""
import pytest
import numpy as np
import tempfile
import os
from pathlib import Path
import embodik

# Ensure robot_descriptions downloads are cached inside the repository unless
# the user already specified a custom location.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CACHE = _PROJECT_ROOT / ".robot_descriptions_cache"
if "ROBOT_DESCRIPTIONS_CACHE" in os.environ:
    Path(os.environ["ROBOT_DESCRIPTIONS_CACHE"]).mkdir(parents=True, exist_ok=True)
else:
    _DEFAULT_CACHE.mkdir(parents=True, exist_ok=True)
    os.environ["ROBOT_DESCRIPTIONS_CACHE"] = str(_DEFAULT_CACHE)

try:
    from robot_descriptions.iiwa14_description import URDF_PATH as IIWA_URDF_PATH
    ROBOT_DESCRIPTIONS_AVAILABLE = True
except ImportError:
    IIWA_URDF_PATH = None
    ROBOT_DESCRIPTIONS_AVAILABLE = False


def get_iiwa_urdf():
    """Get the IIWA URDF path from robot_descriptions or fallback.

    Returns
    -------
    tuple[str, callable | None]
        Path to URDF and optional cleanup callback to remove temporary files.
    """
    if ROBOT_DESCRIPTIONS_AVAILABLE and IIWA_URDF_PATH:
        return IIWA_URDF_PATH, None

    try:
        from test_position_ik_kuka_sns_comparison import create_iiwa_test_urdf
        urdf_path = create_iiwa_test_urdf()
        return urdf_path, lambda: os.unlink(urdf_path)
    except Exception:
        urdf_path = create_test_urdf()
        return urdf_path, lambda: os.unlink(urdf_path)


def create_test_urdf():
    """Create a simplified IIWA 7-DOF URDF for testing.

    This mirrors the structure used in other integration tests so that frames
    such as ``iiwa_link_ee`` are available even without robot_descriptions.
    """
    urdf_content = """<?xml version="1.0"?>
<robot name="iiwa">
  <link name="iiwa_link_0">
    <inertial><mass value="5"/><origin xyz="0 0 0.055"/><inertia ixx="0.05" ixy="0" ixz="0" iyy="0.06" iyz="0" izz="0.03"/></inertial>
  </link>

  <joint name="iiwa_joint_1" type="revolute">
    <parent link="iiwa_link_0"/>
    <child link="iiwa_link_1"/>
    <origin xyz="0 0 0.15" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-2.96706" upper="2.96706" effort="300" velocity="10"/>
  </joint>
  <link name="iiwa_link_1">
    <inertial><mass value="4"/><origin xyz="0 0.03 0.12"/><inertia ixx="0.1" ixy="0" ixz="0" iyy="0.09" iyz="0" izz="0.02"/></inertial>
  </link>

  <joint name="iiwa_joint_2" type="revolute">
    <parent link="iiwa_link_1"/>
    <child link="iiwa_link_2"/>
    <origin xyz="0 0 0.19" rpy="1.5708 0 3.14159"/>
    <axis xyz="0 0 1"/>
    <limit lower="-2.0944" upper="2.0944" effort="300" velocity="10"/>
  </joint>
  <link name="iiwa_link_2">
    <inertial><mass value="4"/><origin xyz="0.0003 0.059 0.042"/><inertia ixx="0.05" ixy="0" ixz="0" iyy="0.018" iyz="0" izz="0.044"/></inertial>
  </link>

  <joint name="iiwa_joint_3" type="revolute">
    <parent link="iiwa_link_2"/>
    <child link="iiwa_link_3"/>
    <origin xyz="0 0.21 0" rpy="1.5708 0 3.14159"/>
    <axis xyz="0 0 1"/>
    <limit lower="-2.96706" upper="2.96706" effort="300" velocity="10"/>
  </joint>
  <link name="iiwa_link_3">
    <inertial><mass value="3"/><origin xyz="0 0.03 0.13"/><inertia ixx="0.08" ixy="0" ixz="0" iyy="0.075" iyz="0" izz="0.01"/></inertial>
  </link>

  <joint name="iiwa_joint_4" type="revolute">
    <parent link="iiwa_link_3"/>
    <child link="iiwa_link_4"/>
    <origin xyz="0 0 0.19" rpy="1.5708 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-2.0944" upper="2.0944" effort="300" velocity="10"/>
  </joint>
  <link name="iiwa_link_4">
    <inertial><mass value="2.7"/><origin xyz="0 0.067 0.034"/><inertia ixx="0.03" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.029"/></inertial>
  </link>

  <joint name="iiwa_joint_5" type="revolute">
    <parent link="iiwa_link_4"/>
    <child link="iiwa_link_5"/>
    <origin xyz="0 0.21 0" rpy="-1.5708 3.14159 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-2.96706" upper="2.96706" effort="300" velocity="10"/>
  </joint>
  <link name="iiwa_link_5">
    <inertial><mass value="1.7"/><origin xyz="0.0001 0.021 0.076"/><inertia ixx="0.02" ixy="0" ixz="0" iyy="0.018" iyz="0" izz="0.005"/></inertial>
  </link>

  <joint name="iiwa_joint_6" type="revolute">
    <parent link="iiwa_link_5"/>
    <child link="iiwa_link_6"/>
    <origin xyz="0 0.06 0.06" rpy="1.5708 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-2.0944" upper="2.0944" effort="300" velocity="10"/>
  </joint>
  <link name="iiwa_link_6">
    <inertial><mass value="1.8"/><origin xyz="0 0.0006 0.0004"/><inertia ixx="0.005" ixy="0" ixz="0" iyy="0.0036" iyz="0" izz="0.0047"/></inertial>
  </link>

  <joint name="iiwa_joint_7" type="revolute">
    <parent link="iiwa_link_6"/>
    <child link="iiwa_link_7"/>
    <origin xyz="0 0.081 0.06" rpy="-1.5708 3.14159 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.05433" upper="3.05433" effort="300" velocity="10"/>
  </joint>
  <link name="iiwa_link_7">
    <inertial><mass value="0.3"/><origin xyz="0 0 0.02"/><inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/></inertial>
  </link>

  <joint name="iiwa_joint_ee" type="fixed">
    <parent link="iiwa_link_7"/>
    <child link="iiwa_link_ee"/>
    <origin xyz="0 0 0.045" rpy="0 0 0"/>
  </joint>
  <link name="iiwa_link_ee">
    <inertial><mass value="0.01"/><origin xyz="0 0 0"/><inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/></inertial>
  </link>
</robot>
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", delete=False) as f:
        f.write(urdf_content)
        return f.name


class TestPositionIKNullspace:

    @pytest.fixture
    def robot_and_solver(self):
        """Create robot model and solver for tests"""
        urdf_path, cleanup = get_iiwa_urdf()
        robot = embodik.RobotModel(urdf_path, floating_base=False)
        solver = embodik.KinematicsSolver(robot)

        # Set reasonable acceleration limits for stability when API supports it
        accel_limits = np.ones(robot.nv) * 5.0  # rad/s^2
        if hasattr(robot, "set_acceleration_limits"):
            robot.set_acceleration_limits(accel_limits)

        solver.enable_position_limits(True)
        solver.enable_velocity_limits(True)
        solver.dt = 0.01

        yield robot, solver

        if cleanup:
            cleanup()

    def test_position_ik_with_nullspace_bias(self, robot_and_solver):
        """Test position IK with nullspace bias towards home configuration"""
        robot, solver = robot_and_solver

        # Start from a moderate configuration (not near limits)
        q_seed = np.array([0.3, -0.5, 0.2, -0.3, 0.4, -0.1, 0.5])
        robot.update_configuration(q_seed)

        # Get current end-effector pose
        current_pose = robot.get_frame_pose("iiwa_link_ee")

        # Create target slightly away from current (smaller movement for stability)
        target_pose = np.eye(4)
        target_pose[:3, :3] = current_pose.rotation
        target_pose[:3, 3] = current_pose.translation + np.array([0.02, 0.0, 0.0])  # 2cm movement

        # Base solver options (no nullspace bias)
        base_options = embodik.PositionIKOptions()
        base_options.max_iterations = 300
        base_options.position_tolerance = 1e-3
        base_options.orientation_tolerance = 1e-3
        base_options.dt = 0.005  # Smaller timestep for stability
        base_options.max_linear_step = 0.05  # Limit step size
        base_options.max_angular_step = 0.1

        # Solve once without nullspace bias to establish a baseline
        robot.update_configuration(q_seed)
        base_result = solver.solve_position(q_seed, target_pose, "iiwa_link_ee", base_options)
        assert base_result.status == embodik.SolverStatus.SUCCESS

        # Configure nullspace-biased options
        options = embodik.PositionIKOptions()
        options.max_iterations = base_options.max_iterations
        options.position_tolerance = base_options.position_tolerance
        options.orientation_tolerance = base_options.orientation_tolerance
        options.dt = base_options.dt
        options.max_linear_step = base_options.max_linear_step
        options.max_angular_step = base_options.max_angular_step
        options.nullspace_bias = np.zeros(7)  # Bias towards home
        options.nullspace_gain = 0.1  # Lower gain for stability

        print(f"\nTesting nullspace bias:")
        print(f"Initial config: {q_seed}")
        print(f"Target movement: 2cm in X")
        print(f"Nullspace bias: zero configuration")

        result = solver.solve_position(q_seed, target_pose, "iiwa_link_ee", options)

        print(f"Status: {result.status}")
        print(f"Iterations: {result.iterations_used}")
        print(f"Position error: {result.position_error:.6f}")
        print(f"Final config: {result.q_solution}")

        assert result.status == embodik.SolverStatus.SUCCESS
        assert result.position_error < 1e-3

        # Check that the solution moved towards zero more than the unbiased solve
        q_solution = result.q_solution
        base_norm = np.linalg.norm(base_result.q_solution)
        dist_to_zero_seed = np.linalg.norm(q_seed)
        dist_to_zero_solution = np.linalg.norm(q_solution)

        # Nullspace bias should pull the solution closer to the bias than the baseline solution
        assert dist_to_zero_solution <= base_norm + 1e-6

        print(f"Seed configuration norm: {dist_to_zero_seed:.3f}")
        print(f"Unbiased solution norm: {base_norm:.3f}")
        print(f"Nullspace-biased solution norm: {dist_to_zero_solution:.3f}")
        print(f"Additional reduction vs unbiased: {base_norm - dist_to_zero_solution:.3f}")

    def test_partial_joint_nullspace_control(self, robot_and_solver):
        """Test nullspace control with only specific joints active"""
        robot, solver = robot_and_solver

        # Start from a non-zero configuration
        q_seed = np.array([0.5, -0.7, 0.3, -0.4, 0.6, -0.2, 0.8])
        robot.update_configuration(q_seed)

        # Get current end-effector pose
        current_pose = robot.get_frame_pose("iiwa_link_ee")

        # Create target (same position, just maintain it)
        target_pose = np.eye(4)
        target_pose[:3, :3] = current_pose.rotation
        target_pose[:3, 3] = current_pose.translation

        # Set options with partial joint nullspace control
        # Only joints 4, 5, 6 (indices 3, 4, 5) should move towards zero
        options = embodik.PositionIKOptions()
        options.max_iterations = 100
        options.position_tolerance = 1e-3
        options.orientation_tolerance = 1e-3
        options.dt = 0.01
        options.nullspace_bias = np.zeros(7)
        options.nullspace_gain = 1.0  # High gain since we're maintaining position
        options.nullspace_active_joints = [3, 4, 5]  # Only these joints

        result = solver.solve_position(q_seed, target_pose, "iiwa_link_ee", options)

        assert result.status == embodik.SolverStatus.SUCCESS

        q_solution = result.q_solution

        # Check that active joints moved towards zero
        for idx in [3, 4, 5]:
            assert abs(q_solution[idx]) <= abs(q_seed[idx]) + 1e-6

        # Check that inactive joints didn't change much
        for idx in [0, 1, 2, 6]:
            assert abs(q_solution[idx] - q_seed[idx]) < 0.1 + 1e-6

        print("Partial joint nullspace control test passed")
        print(f"Initial joints 4-6: {q_seed[3:6]}")
        print(f"Final joints 4-6: {q_solution[3:6]}")

    def test_step_size_limits(self, robot_and_solver):
        """Test that step size limits prevent large jumps"""
        robot, solver = robot_and_solver

        # Start from zero configuration
        q_seed = np.zeros(7)
        robot.update_configuration(q_seed)

        # Get current end-effector pose
        current_pose = robot.get_frame_pose("iiwa_link_ee")

        # Create target far away (would normally require large steps)
        target_pose = np.eye(4)
        target_pose[:3, :3] = current_pose.rotation
        target_pose[:3, 3] = current_pose.translation + np.array([0.3, 0.0, 0.0])  # 30cm away

        # Set options with step size limits
        options = embodik.PositionIKOptions()
        options.max_iterations = 100
        options.position_tolerance = 1e-3
        options.orientation_tolerance = 1e-3
        options.dt = 0.01
        options.max_linear_step = 0.02  # Max 2cm per iteration
        options.max_angular_step = 0.05  # Max 0.05 rad per iteration

        result = solver.solve_position(q_seed, target_pose, "iiwa_link_ee", options)

        # Should either succeed or reach max iterations
        assert result.status in [embodik.SolverStatus.SUCCESS,
                                embodik.SolverStatus.NUMERICAL_ERROR]

        # If it succeeded, it should have taken many iterations due to step limits
        if result.status == embodik.SolverStatus.SUCCESS:
            assert result.iterations_used > 10  # Should take many small steps

        print(f"Step size limit test: {result.iterations_used} iterations")
        print(f"Final position error: {result.position_error:.6f}")

    def test_nullspace_with_different_priorities(self, robot_and_solver):
        """Test that nullspace task has lower priority than main task"""
        robot, solver = robot_and_solver

        # Configuration that would violate joint limits if moved too much
        q_seed = np.array([2.5, -2.5, 2.0, -2.0, 1.5, -1.5, 1.0])
        robot.update_configuration(q_seed)

        # Get current end-effector pose
        current_pose = robot.get_frame_pose("iiwa_link_ee")

        # Create target that requires movement
        target_pose = np.eye(4)
        target_pose[:3, :3] = current_pose.rotation
        target_pose[:3, 3] = current_pose.translation + np.array([0.1, 0.05, 0.0])

        # Set options with strong nullspace bias that conflicts with target
        options = embodik.PositionIKOptions()
        options.max_iterations = 200
        options.position_tolerance = 1e-3
        options.orientation_tolerance = 1e-3
        options.dt = 0.01
        options.nullspace_bias = np.zeros(7)  # Strong bias away from current config
        options.nullspace_gain = 2.0  # High gain

        # Baseline solve without nullspace
        base_options = embodik.PositionIKOptions()
        base_options.max_iterations = options.max_iterations
        base_options.position_tolerance = options.position_tolerance
        base_options.orientation_tolerance = options.orientation_tolerance
        base_options.dt = options.dt

        robot.update_configuration(q_seed)
        baseline = solver.solve_position(q_seed, target_pose, "iiwa_link_ee", base_options)

        # Now solve with nullspace bias
        robot.update_configuration(q_seed)
        result = solver.solve_position(q_seed, target_pose, "iiwa_link_ee", options)

        # Nullspace should not worsen the primary task error relative to the baseline
        assert result.position_error <= baseline.position_error + 1e-3

        print("Nullspace priority test completed")
        print(f"Baseline position error: {baseline.position_error:.6f}")
        print(f"Nullspace-biased position error: {result.position_error:.6f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

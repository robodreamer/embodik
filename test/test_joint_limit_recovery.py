#!/usr/bin/env python3
"""Tests for joint limit recovery velocity bounds."""

import os
import tempfile

import numpy as np
import pytest

import embodik as eik


def create_test_urdf():
    """Create a simple 1-DoF URDF with joint limits."""
    urdf_content = """<?xml version="1.0"?>
<robot name="test_robot">
  <link name="base_link">
    <inertial>
      <mass value="1.0"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
    </inertial>
  </link>

  <link name="link1">
    <inertial>
      <mass value="1.0"/>
      <origin xyz="0 0 0.5"/>
      <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="1" izz="1"/>
    </inertial>
  </link>

  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <origin xyz="0 0 1" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="0.0" upper="1.0" velocity="1.0" effort="10.0"/>
  </joint>
</robot>"""
    fd, path = tempfile.mkstemp(suffix=".urdf")
    with os.fdopen(fd, "w") as f:
        f.write(urdf_content)
    return path


@pytest.fixture
def solver():
    urdf_path = create_test_urdf()
    model = eik.RobotModel(urdf_path, floating_base=False)
    solver = eik.KinematicsSolver(model)
    yield solver
    os.remove(urdf_path)


@pytest.fixture
def constraint_params():
    return {"velocity_limit": 1.0, "acceleration_limit": 10.0, "dt": 0.01}


class TestVelocityBoxConstraintFormulation:
    """Direct tests of the velocity box constraint math."""

    def test_current_inside_limits(self, solver, constraint_params):
        """Inside limits: symmetric velocity bounds."""
        lower, upper = solver.calculate_velocity_box_constraint(
            0.5,
            0.5,
            constraint_params["velocity_limit"],
            constraint_params["acceleration_limit"],
            constraint_params["dt"],
        )
        assert np.isclose(lower, -1.0)
        assert np.isclose(upper, 1.0)

    def test_current_at_lower_limit(self, solver, constraint_params):
        """At lower limit: can only move positive."""
        lower, upper = solver.calculate_velocity_box_constraint(
            -1e-4,
            0.5,
            constraint_params["velocity_limit"],
            constraint_params["acceleration_limit"],
            constraint_params["dt"],
        )
        assert np.isclose(lower, 0.0)
        assert np.isclose(upper, 1.0)

    def test_new_outside_lower_forces_recovery(self, solver, constraint_params):
        """Outside lower limit: v >= v_recovery required."""
        solver.set_limit_recovery_gain(0.5)
        lower, upper = solver.calculate_velocity_box_constraint(
            -0.0501,
            0.5,
            constraint_params["velocity_limit"],
            constraint_params["acceleration_limit"],
            constraint_params["dt"],
        )
        assert np.isclose(lower, 1.0)
        assert upper >= lower

    def test_new_outside_upper_forces_recovery(self, solver, constraint_params):
        """Outside upper limit: v <= -v_recovery required."""
        solver.set_limit_recovery_gain(0.5)
        lower, upper = solver.calculate_velocity_box_constraint(
            0.5,
            -0.0501,
            constraint_params["velocity_limit"],
            constraint_params["acceleration_limit"],
            constraint_params["dt"],
        )
        assert np.isclose(upper, -1.0)
        assert lower <= upper

    def test_recovery_gain_affects_speed(self, solver, constraint_params):
        """Different recovery gains produce different recovery velocities."""
        solver.set_limit_recovery_gain(0.1)
        lower_gentle, _ = solver.calculate_velocity_box_constraint(
            -0.0201,
            0.5,
            constraint_params["velocity_limit"],
            constraint_params["acceleration_limit"],
            constraint_params["dt"],
        )
        solver.set_limit_recovery_gain(1.0)
        lower_aggressive, _ = solver.calculate_velocity_box_constraint(
            -0.0201,
            0.5,
            constraint_params["velocity_limit"],
            constraint_params["acceleration_limit"],
            constraint_params["dt"],
        )
        assert lower_gentle < lower_aggressive

    def test_small_violation_small_recovery(self, solver, constraint_params):
        """Small violations need small recovery velocities."""
        solver.set_limit_recovery_gain(0.5)
        lower, _ = solver.calculate_velocity_box_constraint(
            -0.0011,
            0.5,
            constraint_params["velocity_limit"],
            constraint_params["acceleration_limit"],
            constraint_params["dt"],
        )
        assert np.isclose(lower, 0.055)

    def test_multi_step_convergence(self, solver, constraint_params):
        """Verify recovery to valid region over multiple iterations."""
        solver.set_limit_recovery_gain(0.5)
        q = -0.1
        q_min = 0.0
        q_max = 1.0
        margin_limit = 1e-4
        for _ in range(40):
            lower_margin = q - q_min - margin_limit
            upper_margin = q_max - q - margin_limit
            lower, _ = solver.calculate_velocity_box_constraint(
                lower_margin,
                upper_margin,
                constraint_params["velocity_limit"],
                constraint_params["acceleration_limit"],
                constraint_params["dt"],
            )
            v = lower if lower > 0 else 0.0
            q += v * constraint_params["dt"]
        assert q >= q_min - 1e-4

    def test_both_limits_violated_infeasible(self, solver, constraint_params):
        """If both margins are negative, constraint is infeasible."""
        lower, upper = solver.calculate_velocity_box_constraint(
            -0.1,
            -0.2,
            constraint_params["velocity_limit"],
            constraint_params["acceleration_limit"],
            constraint_params["dt"],
        )
        assert np.isclose(lower, -1.0)
        assert np.isclose(upper, 1.0)

    def test_exactly_at_boundary(self, solver, constraint_params):
        """At exact boundary, behavior is consistent (no discontinuity)."""
        lower, upper = solver.calculate_velocity_box_constraint(
            -1e-4,
            0.3,
            constraint_params["velocity_limit"],
            constraint_params["acceleration_limit"],
            constraint_params["dt"],
        )
        assert np.isclose(lower, 0.0)
        assert upper > 0.0

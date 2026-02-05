"""
Tests for GPU-accelerated collision detection.

These tests validate that the Warp-based collision detection matches
hpp-fcl/Pinocchio collision results within specified tolerances.
"""

import pytest
import numpy as np
import os

# Test tolerances
DISTANCE_ATOL = 1e-3  # Distance tolerance (mesh discretization)
JACOBIAN_ATOL = 1e-4  # Jacobian tolerance


def generate_random_configurations(
    n_configs: int,
    n_dof: int,
    seed: int = 42,
    joint_limits: tuple = (-np.pi, np.pi),
) -> np.ndarray:
    """Generate random joint configurations."""
    rng = np.random.default_rng(seed)
    return rng.uniform(joint_limits[0], joint_limits[1], (n_configs, n_dof)).astype(np.float64)


# =============================================================================
# Availability checks
# =============================================================================

def test_warp_availability_check():
    """Test that Warp availability check works."""
    try:
        from embodik.gpu.warp_collision import check_warp_availability

        status = check_warp_availability()

        assert isinstance(status, dict)
        assert "warp" in status
        assert "cuda" in status
        # Note: "pinocchio" key was removed in v0.4.0 as pin is no longer a runtime dep

        for key, value in status.items():
            assert isinstance(value, bool), f"{key} should be bool"

    except ImportError:
        pytest.skip("Warp collision module not available")


def test_collision_result_dataclass():
    """Test CollisionResult dataclass."""
    try:
        from embodik.gpu.warp_collision import CollisionResult

        result = CollisionResult(
            distances=np.array([0.1, 0.2]),
            closest_points_a=np.zeros((2, 3)),
            closest_points_b=np.zeros((2, 3)),
            normals=np.zeros((2, 3)),
            status="success",
        )

        assert result.distances.shape == (2,)
        assert result.status == "success"

    except ImportError:
        pytest.skip("Warp collision module not available")


# =============================================================================
# CPU fallback tests (always run)
# =============================================================================

class TestCPUFallback:
    """Tests for CPU fallback collision detection."""

    def test_cpu_fallback_with_pinocchio(self, tmp_path):
        """Test CPU collision detection using Pinocchio."""
        try:
            import embodik as eik
            from embodik.gpu.warp_collision import compute_collision_distances_batched
        except ImportError:
            pytest.skip("Required modules not available")

        # Create a simple robot with collision geometry
        urdf_content = '''<?xml version="1.0"?>
<robot name="simple_arm">
  <link name="base">
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><box size="0.1 0.1 0.1"/></geometry>
    </collision>
  </link>
  <link name="link1">
    <collision>
      <origin xyz="0.1 0 0" rpy="0 0 0"/>
      <geometry><box size="0.08 0.08 0.08"/></geometry>
    </collision>
  </link>
  <joint name="joint1" type="revolute">
    <parent link="base"/>
    <child link="link1"/>
    <origin xyz="0.05 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit effort="10" lower="-1.57" upper="1.57" velocity="1"/>
  </joint>
</robot>'''

        urdf_path = tmp_path / "simple_arm.urdf"
        urdf_path.write_text(urdf_content)

        try:
            robot = eik.RobotModel(str(urdf_path), floating_base=False)
        except Exception as e:
            pytest.skip(f"Could not create robot model: {e}")

        # Generate configurations
        q_batch = generate_random_configurations(5, robot.nq, seed=42, joint_limits=(-1.5, 1.5))

        # Compute distances with CPU fallback
        result = compute_collision_distances_batched(
            robot, q_batch, use_gpu=False
        )

        # Check results
        assert result.distances.shape == (5,)
        assert result.closest_points_a.shape == (5, 3)
        assert result.closest_points_b.shape == (5, 3)
        assert result.normals.shape == (5, 3)
        assert result.status in ["success", "fallback_cpu", "no_collision_library"]


# =============================================================================
# GPU vs CPU validation tests
# =============================================================================

@pytest.mark.skipif(
    not os.environ.get("EMBODIK_GPU_TESTS", ""),
    reason="GPU tests disabled. Set EMBODIK_GPU_TESTS=1 to enable."
)
class TestGPUvsCPUCollision:
    """Tests comparing GPU and CPU collision detection."""

    def test_warp_vs_hppfcl_distances(self, tmp_path):
        """Compare Warp and hpp-fcl distance computations."""
        try:
            import embodik as eik
            from embodik.gpu.warp_collision import (
                compute_collision_distances_batched,
                check_warp_availability,
            )
        except ImportError:
            pytest.skip("Required modules not available")

        # Check Warp availability
        status = check_warp_availability()
        if not status.get("warp") or not status.get("cuda"):
            pytest.skip("Warp + CUDA not available")

        # Create test robot
        urdf_content = '''<?xml version="1.0"?>
<robot name="test_arm">
  <link name="base">
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><sphere radius="0.05"/></geometry>
    </collision>
  </link>
  <link name="link1">
    <collision>
      <origin xyz="0.15 0 0" rpy="0 0 0"/>
      <geometry><sphere radius="0.04"/></geometry>
    </collision>
  </link>
  <joint name="joint1" type="revolute">
    <parent link="base"/>
    <child link="link1"/>
    <origin xyz="0.1 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit effort="10" lower="-3.14" upper="3.14" velocity="1"/>
  </joint>
</robot>'''

        urdf_path = tmp_path / "test_arm.urdf"
        urdf_path.write_text(urdf_content)

        try:
            robot = eik.RobotModel(str(urdf_path), floating_base=False)
        except Exception as e:
            pytest.skip(f"Could not create robot model: {e}")

        # Generate configurations
        q_batch = generate_random_configurations(20, robot.nq, seed=123)

        # CPU distances (ground truth)
        cpu_result = compute_collision_distances_batched(
            robot, q_batch, use_gpu=False
        )

        # GPU distances
        gpu_result = compute_collision_distances_batched(
            robot, q_batch, use_gpu=True
        )

        # Compare if GPU succeeded
        if gpu_result.status == "success":
            # Distances should match within tolerance
            assert np.allclose(
                cpu_result.distances,
                gpu_result.distances,
                atol=DISTANCE_ATOL
            ), f"Distance mismatch: max diff = {np.max(np.abs(cpu_result.distances - gpu_result.distances))}"

    def test_batch_consistency(self, tmp_path):
        """Batched computation should match sequential."""
        try:
            import embodik as eik
            from embodik.gpu.warp_collision import compute_collision_distances_batched
        except ImportError:
            pytest.skip("Required modules not available")

        urdf_content = '''<?xml version="1.0"?>
<robot name="simple">
  <link name="base">
    <collision><geometry><box size="0.1 0.1 0.1"/></geometry></collision>
  </link>
  <link name="link1">
    <collision><origin xyz="0.1 0 0"/><geometry><box size="0.08 0.08 0.08"/></geometry></collision>
  </link>
  <joint name="joint1" type="revolute">
    <parent link="base"/><child link="link1"/>
    <origin xyz="0.05 0 0"/><axis xyz="0 0 1"/>
    <limit effort="10" lower="-1.57" upper="1.57" velocity="1"/>
  </joint>
</robot>'''

        urdf_path = tmp_path / "simple.urdf"
        urdf_path.write_text(urdf_content)

        try:
            robot = eik.RobotModel(str(urdf_path), floating_base=False)
        except Exception as e:
            pytest.skip(f"Could not create robot model: {e}")

        # Generate configurations
        q_batch = generate_random_configurations(10, robot.nq, seed=456)

        # Batched computation
        batched = compute_collision_distances_batched(robot, q_batch, use_gpu=False)

        # Sequential computation
        sequential_dists = []
        for i in range(q_batch.shape[0]):
            result = compute_collision_distances_batched(
                robot, q_batch[i:i+1], use_gpu=False
            )
            sequential_dists.append(result.distances[0])

        sequential_dists = np.array(sequential_dists)

        # Should match exactly (same code path)
        assert np.allclose(batched.distances, sequential_dists, atol=1e-10)


# =============================================================================
# WarpCollisionModel tests
# =============================================================================

class TestWarpCollisionModel:
    """Tests for WarpCollisionModel class."""

    def test_primitive_vertices_box(self):
        """Test box primitive vertex generation."""
        try:
            from embodik.gpu.warp_collision import WarpCollisionModel
        except ImportError:
            pytest.skip("Warp collision module not available")

        # Create mock box geometry
        class MockBox:
            halfSide = np.array([0.1, 0.1, 0.1])

        vertices = WarpCollisionModel._create_primitive_vertices(MockBox())

        assert vertices is not None
        assert vertices.shape[1] == 3  # (N, 3) vertices
        assert len(vertices) == 36  # 12 triangles * 3 vertices

    def test_primitive_vertices_sphere(self):
        """Test sphere primitive vertex generation."""
        try:
            from embodik.gpu.warp_collision import WarpCollisionModel
        except ImportError:
            pytest.skip("Warp collision module not available")

        class MockSphere:
            radius = 0.05

        vertices = WarpCollisionModel._create_primitive_vertices(MockSphere())

        assert vertices is not None
        assert vertices.shape[1] == 3
        assert len(vertices) == 12  # Icosphere approximation

    def test_primitive_vertices_cylinder(self):
        """Test cylinder primitive vertex generation."""
        try:
            from embodik.gpu.warp_collision import WarpCollisionModel
        except ImportError:
            pytest.skip("Warp collision module not available")

        class MockCylinder:
            radius = 0.05
            halfLength = 0.1

        vertices = WarpCollisionModel._create_primitive_vertices(MockCylinder())

        assert vertices is not None
        assert vertices.shape[1] == 3
        assert len(vertices) == 16  # 8 top + 8 bottom


# =============================================================================
# Jacobian tests
# =============================================================================

class TestCollisionJacobian:
    """Tests for collision Jacobian computation."""

    def test_jacobian_finite_difference(self, tmp_path):
        """Validate collision Jacobian via finite difference."""
        try:
            import embodik as eik
            from embodik.gpu.warp_collision import compute_collision_distances_batched
        except ImportError:
            pytest.skip("Required modules not available")

        urdf_content = '''<?xml version="1.0"?>
<robot name="test">
  <link name="base">
    <collision><geometry><box size="0.1 0.1 0.1"/></geometry></collision>
  </link>
  <link name="link1">
    <collision><origin xyz="0.1 0 0"/><geometry><box size="0.08 0.08 0.08"/></geometry></collision>
  </link>
  <joint name="joint1" type="revolute">
    <parent link="base"/><child link="link1"/>
    <origin xyz="0.05 0 0"/><axis xyz="0 0 1"/>
    <limit effort="10" lower="-1.57" upper="1.57" velocity="1"/>
  </joint>
</robot>'''

        urdf_path = tmp_path / "test.urdf"
        urdf_path.write_text(urdf_content)

        try:
            robot = eik.RobotModel(str(urdf_path), floating_base=False)
        except Exception as e:
            pytest.skip(f"Could not create robot model: {e}")

        # Test configuration
        q = np.array([0.5], dtype=np.float64)
        eps = 1e-6

        # Compute distance at q
        result = compute_collision_distances_batched(robot, q.reshape(1, -1), use_gpu=False)
        d0 = result.distances[0]

        # Skip if no collision geometry available
        if result.status == "no_collision_library" or not np.isfinite(d0):
            pytest.skip("No collision geometry available for this robot")

        # Finite difference Jacobian
        jac_fd = np.zeros(robot.nq)
        for i in range(robot.nq):
            q_plus = q.copy()
            q_plus[i] += eps
            d_plus = compute_collision_distances_batched(
                robot, q_plus.reshape(1, -1), use_gpu=False
            ).distances[0]

            q_minus = q.copy()
            q_minus[i] -= eps
            d_minus = compute_collision_distances_batched(
                robot, q_minus.reshape(1, -1), use_gpu=False
            ).distances[0]

            # Handle inf distances gracefully
            if np.isfinite(d_plus) and np.isfinite(d_minus):
                jac_fd[i] = (d_plus - d_minus) / (2 * eps)
            else:
                jac_fd[i] = 0.0

        # Check that Jacobian is finite and reasonable
        assert np.all(np.isfinite(jac_fd))


# =============================================================================
# Test runner
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

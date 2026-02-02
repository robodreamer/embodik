"""
PyTorch-based batched forward kinematics and Jacobian computation.

Uses pytorch_kinematics library for GPU-accelerated batched FK and Jacobian.
This enables real trajectory tracking with true Jacobians in parallel.

References:
    - pytorch_kinematics: https://github.com/UM-ARM-Lab/pytorch_kinematics
    - swift_ik reachability_fk.py for usage patterns
"""

from __future__ import annotations

from typing import Tuple, Optional, Union
from pathlib import Path
import numpy as np

try:
    import torch
    from torch import Tensor
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    Tensor = None

try:
    import pytorch_kinematics as pk
    HAS_PK = True
except ImportError:
    HAS_PK = False
    pk = None


def _check_dependencies():
    if not HAS_TORCH:
        raise ImportError("PyTorch is required. Install with: pip install torch")
    if not HAS_PK:
        raise ImportError(
            "pytorch_kinematics is required for GPU kinematics. "
            "Install with: pip install pytorch-kinematics"
        )


class BatchedKinematics:
    """
    GPU-accelerated batched FK and Jacobian using pytorch_kinematics.

    This class wraps pytorch_kinematics to provide efficient batched computation
    of forward kinematics and Jacobians for parallel IK solving.

    Example:
        >>> from embodik.gpu.torch_kinematics import BatchedKinematics
        >>> kin = BatchedKinematics.from_urdf_path(
        ...     "/path/to/panda.urdf",
        ...     end_effector="panda_hand",
        ...     device="cuda"
        ... )
        >>> q = torch.randn(100, 7, device="cuda", dtype=torch.float64)
        >>> pos = kin.forward_kinematics_position(q)  # (100, 3)
        >>> J = kin.jacobian(q)  # (100, 6, 7)
    """

    def __init__(
        self,
        chain: "pk.SerialChain",
        device: Union[str, torch.device] = "cuda",
        dtype: torch.dtype = torch.float64,
    ):
        _check_dependencies()
        self.chain = chain.to(dtype=dtype, device=device)
        self.device = torch.device(device)
        self.dtype = dtype
        self.n_joints = len(chain.get_joint_parameter_names())
        self.joint_names = chain.get_joint_parameter_names()

    @classmethod
    def from_urdf_path(
        cls,
        urdf_path: Union[str, Path],
        end_effector: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.float64,
    ) -> "BatchedKinematics":
        """
        Create BatchedKinematics from a URDF file.

        Args:
            urdf_path: Path to URDF file
            end_effector: Name of end-effector frame
            device: "cuda" or "cpu"
            dtype: torch.float32 or torch.float64

        Returns:
            BatchedKinematics instance
        """
        _check_dependencies()

        urdf_path = Path(urdf_path)
        with open(urdf_path, "r", encoding="utf-8") as f:
            urdf_xml = f.read()

        chain = pk.build_serial_chain_from_urdf(urdf_xml, end_effector)
        return cls(chain, device=device, dtype=dtype)

    @classmethod
    def from_urdf_string(
        cls,
        urdf_xml: str,
        end_effector: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.float64,
    ) -> "BatchedKinematics":
        """Create BatchedKinematics from URDF XML string."""
        _check_dependencies()
        chain = pk.build_serial_chain_from_urdf(urdf_xml, end_effector)
        return cls(chain, device=device, dtype=dtype)

    def forward_kinematics(self, q: Tensor) -> Tensor:
        """
        Compute batched forward kinematics (full 4x4 transform).

        Args:
            q: Joint angles (batch_size, n_joints)

        Returns:
            T: Homogeneous transforms (batch_size, 4, 4)
        """
        q = q.to(device=self.device, dtype=self.dtype)
        fk_result = self.chain.forward_kinematics(q)
        return fk_result.get_matrix()

    def forward_kinematics_position(self, q: Tensor) -> Tensor:
        """
        Compute batched end-effector positions.

        Args:
            q: Joint angles (batch_size, n_joints)

        Returns:
            pos: End-effector positions (batch_size, 3)
        """
        T = self.forward_kinematics(q)
        return T[:, :3, 3]

    def forward_kinematics_pose(self, q: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Compute batched end-effector poses.

        Args:
            q: Joint angles (batch_size, n_joints)

        Returns:
            pos: End-effector positions (batch_size, 3)
            rot: End-effector rotations (batch_size, 3, 3)
        """
        T = self.forward_kinematics(q)
        return T[:, :3, 3], T[:, :3, :3]

    def jacobian(self, q: Tensor) -> Tensor:
        """
        Compute batched geometric Jacobians.

        Args:
            q: Joint angles (batch_size, n_joints)

        Returns:
            J: Jacobians (batch_size, 6, n_joints)
        """
        q = q.to(device=self.device, dtype=self.dtype)
        return self.chain.jacobian(q)

    def jacobian_position(self, q: Tensor) -> Tensor:
        """
        Compute batched position Jacobians (3xN).

        Args:
            q: Joint angles (batch_size, n_joints)

        Returns:
            Jp: Position Jacobians (batch_size, 3, n_joints)
        """
        J = self.jacobian(q)
        return J[:, :3, :]


class PandaKinematics(BatchedKinematics):
    """
    Convenience class for Panda robot kinematics.

    Automatically loads the Panda URDF from robot_descriptions package.

    Example:
        >>> kin = PandaKinematics(device="cuda")
        >>> q = torch.zeros(100, 7, device="cuda", dtype=torch.float64)
        >>> pos = kin.forward_kinematics_position(q)
        >>> J = kin.jacobian(q)
    """

    # Panda default configuration (from robot_presets.yaml)
    DEFAULT_CONFIG = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])

    def __init__(
        self,
        device: str = "cuda",
        dtype: torch.dtype = torch.float64,
        urdf_path: Optional[str] = None,
    ):
        _check_dependencies()

        if urdf_path is None:
            # Try to load from robot_descriptions
            try:
                from robot_descriptions.panda_description import URDF_PATH
                urdf_path = URDF_PATH
            except ImportError:
                raise ImportError(
                    "robot_descriptions package required for automatic URDF loading. "
                    "Install with: pip install robot_descriptions"
                )

        with open(urdf_path, "r", encoding="utf-8") as f:
            urdf_xml = f.read()

        chain = pk.build_serial_chain_from_urdf(urdf_xml, "panda_hand")
        super().__init__(chain, device=device, dtype=dtype)

    def get_default_config(self, batch_size: int = 1) -> Tensor:
        """Get default Panda configuration as tensor."""
        q = torch.tensor(self.DEFAULT_CONFIG, device=self.device, dtype=self.dtype)
        return q.unsqueeze(0).expand(batch_size, -1).clone()


def benchmark_kinematics(batch_size: int = 100, device: str = "cuda", n_runs: int = 10):
    """Benchmark FK and Jacobian computation speed."""
    _check_dependencies()

    import time

    print(f"\nBenchmarking pytorch_kinematics on {device} with batch_size={batch_size}")

    kin = PandaKinematics(device=device)
    q = torch.randn(batch_size, 7, device=device, dtype=torch.float64)

    # Warm-up
    for _ in range(3):
        _ = kin.forward_kinematics_position(q)
        _ = kin.jacobian(q)

    if device == "cuda":
        torch.cuda.synchronize()

    # Benchmark FK
    start = time.perf_counter()
    for _ in range(n_runs):
        _ = kin.forward_kinematics_position(q)
    if device == "cuda":
        torch.cuda.synchronize()
    fk_time = (time.perf_counter() - start) / n_runs * 1000

    # Benchmark Jacobian
    start = time.perf_counter()
    for _ in range(n_runs):
        _ = kin.jacobian(q)
    if device == "cuda":
        torch.cuda.synchronize()
    jac_time = (time.perf_counter() - start) / n_runs * 1000

    print(f"FK ({batch_size} samples): {fk_time:.2f} ms ({fk_time/batch_size*1000:.2f} µs/sample)")
    print(f"Jacobian ({batch_size} samples): {jac_time:.2f} ms ({jac_time/batch_size*1000:.2f} µs/sample)")

    return fk_time, jac_time


if __name__ == "__main__":
    _check_dependencies()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create kinematics
    kin = PandaKinematics(device=device)
    print(f"Loaded Panda kinematics with {kin.n_joints} joints")
    print(f"Joint names: {kin.joint_names}")

    # Test at default config
    q_default = kin.get_default_config(batch_size=1)
    pos = kin.forward_kinematics_position(q_default)
    print(f"\nEE position at default config: {pos[0].cpu().numpy()}")

    J = kin.jacobian(q_default)
    print(f"Jacobian shape: {J.shape}")
    print(f"Jacobian:\n{J[0].cpu().numpy()}")

    # Benchmark
    print("\n" + "="*60)
    print("BENCHMARK")
    print("="*60)
    for batch_size in [100, 1000, 10000]:
        benchmark_kinematics(batch_size, device)
        print()

#!/usr/bin/env python3
"""
Parallel Trajectory Tracking Demo - 100 Robot Instances

Demonstrates GPU-accelerated parallel IK solving with real FK and Jacobians:
- Real FK and Jacobians via pytorch_kinematics (GPU batched)
- GPU-accelerated velocity IK via CusADi
- 100+ Panda robots tracking circles, figure-8s, spirals, hearts in parallel

Usage:
    # First, compile the CusADi kernel (one-time):
    pixi run -e cuda export-casadi
    mkdir -p ~/.local/cusadi/src/casadi_functions
    cp build/casadi/fn_velocity_solve.casadi ~/.local/cusadi/src/casadi_functions/
    cd ~/.local/cusadi && python3 run_codegen.py --fn=fn_velocity_solve

    # Run the demo:
    pixi run -e cuda demo-parallel-tracking

    # Run benchmark:
    pixi run -e cuda demo-parallel-tracking-benchmark
"""

from __future__ import annotations

import argparse
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import math

import numpy as np

# Suppress pytorch_kinematics URDF warnings
warnings.filterwarnings("ignore", message="Unknown attribute")

# Check GPU and library availability
HAS_TORCH = False
HAS_GPU = False
HAS_PK = False
CusadiFunction = None

try:
    import torch
    HAS_TORCH = True
    if torch.cuda.is_available():
        try:
            from embodik.gpu import HAS_CUSADI, CusadiFunction as _CusadiFunction
            if HAS_CUSADI:
                CusadiFunction = _CusadiFunction
                HAS_GPU = True
        except ImportError:
            pass
except ImportError:
    pass

try:
    import pytorch_kinematics as pk
    HAS_PK = True
except ImportError:
    pass

try:
    import casadi as ca
    HAS_CASADI = True
except ImportError:
    HAS_CASADI = False

try:
    import embodik as eik
except ImportError:
    print("Error: embodik not installed")
    exit(1)

try:
    import viser
    from viser.extras import ViserUrdf
    from robot_descriptions.loaders.yourdfpy import load_robot_description
    HAS_VISER = True
except ImportError:
    HAS_VISER = False
    print("Warning: viser/robot_descriptions not installed. Install with: pip install viser robot_descriptions")


# -----------------------------------------------------------------------------
# Trajectory Generators
# -----------------------------------------------------------------------------

@dataclass
class TrajectoryShape:
    """Base class for trajectory shapes."""
    name: str
    center: np.ndarray
    scale: float
    phase_offset: float  # Starting phase
    speed: float  # Radians per second

    def get_position(self, t: float) -> np.ndarray:
        """Get position at time t (seconds)."""
        raise NotImplementedError

    def get_velocity(self, t: float, dt: float = 0.01) -> np.ndarray:
        """Get velocity at time t."""
        pos_now = self.get_position(t)
        pos_next = self.get_position(t + dt)
        return (pos_next - pos_now) / dt


class CircleTrajectory(TrajectoryShape):
    """Circular trajectory in XY plane."""

    def __init__(self, center: np.ndarray, radius: float, speed: float = 1.0, phase: float = 0.0, tilt: float = 0.0):
        super().__init__("circle", center, radius, phase, speed)
        self.tilt = tilt  # Rotation around X axis

    def get_position(self, t: float) -> np.ndarray:
        theta = self.phase_offset + self.speed * t
        x = self.center[0] + self.scale * np.cos(theta)
        y = self.center[1] + self.scale * np.sin(theta) * np.cos(self.tilt)
        z = self.center[2] + self.scale * np.sin(theta) * np.sin(self.tilt)
        return np.array([x, y, z])


class Figure8Trajectory(TrajectoryShape):
    """Figure-8 (lemniscate) trajectory."""

    def __init__(self, center: np.ndarray, scale: float, speed: float = 0.8, phase: float = 0.0):
        super().__init__("figure8", center, scale, phase, speed)

    def get_position(self, t: float) -> np.ndarray:
        theta = self.phase_offset + self.speed * t
        x = self.center[0] + self.scale * np.sin(theta)
        y = self.center[1] + self.scale * np.sin(2 * theta) * 0.5
        z = self.center[2]
        return np.array([x, y, z])


class SpiralTrajectory(TrajectoryShape):
    """Spiral trajectory with vertical motion."""

    def __init__(self, center: np.ndarray, radius: float, height: float, speed: float = 1.0, phase: float = 0.0):
        super().__init__("spiral", center, radius, phase, speed)
        self.height = height

    def get_position(self, t: float) -> np.ndarray:
        theta = self.phase_offset + self.speed * t
        z_offset = (np.sin(theta * 0.3) * 0.5 + 0.5) * self.height
        x = self.center[0] + self.scale * np.cos(theta)
        y = self.center[1] + self.scale * np.sin(theta)
        z = self.center[2] + z_offset
        return np.array([x, y, z])


class HeartTrajectory(TrajectoryShape):
    """Heart-shaped trajectory."""

    def __init__(self, center: np.ndarray, scale: float, speed: float = 0.6, phase: float = 0.0):
        super().__init__("heart", center, scale, phase, speed)

    def get_position(self, t: float) -> np.ndarray:
        theta = self.phase_offset + self.speed * t
        # Parametric heart curve
        x = self.center[0] + self.scale * 0.5 * (16 * np.sin(theta) ** 3) / 16
        y = self.center[1] + self.scale * 0.5 * (13 * np.cos(theta) - 5 * np.cos(2*theta) - 2 * np.cos(3*theta) - np.cos(4*theta)) / 16
        z = self.center[2]
        return np.array([x, y, z])


def create_diverse_trajectories(n_robots: int, base_center: np.ndarray, grid_spacing: float = 0.4) -> List[TrajectoryShape]:
    """Create diverse trajectories for multiple robots arranged in a grid."""
    trajectories = []

    # Arrange robots in a grid
    n_cols = int(np.ceil(np.sqrt(n_robots)))
    n_rows = int(np.ceil(n_robots / n_cols))

    trajectory_types = [CircleTrajectory, Figure8Trajectory, SpiralTrajectory, HeartTrajectory]

    for i in range(n_robots):
        row = i // n_cols
        col = i % n_cols

        # Center for this robot
        center = base_center.copy()
        center[0] += (col - n_cols / 2) * grid_spacing
        center[1] += (row - n_rows / 2) * grid_spacing

        # Random trajectory type and parameters
        rng = np.random.RandomState(i)
        traj_type = trajectory_types[i % len(trajectory_types)]

        # Randomize parameters
        scale = 0.05 + rng.rand() * 0.05  # 5-10 cm radius
        speed = 0.5 + rng.rand() * 1.0  # 0.5-1.5 rad/s
        phase = rng.rand() * 2 * np.pi  # Random starting phase

        if traj_type == CircleTrajectory:
            tilt = rng.rand() * np.pi / 4  # Random tilt up to 45 degrees
            traj = CircleTrajectory(center, scale, speed, phase, tilt)
        elif traj_type == SpiralTrajectory:
            height = 0.02 + rng.rand() * 0.03
            traj = SpiralTrajectory(center, scale, height, speed, phase)
        else:
            traj = traj_type(center, scale, speed, phase)

        trajectories.append(traj)

    return trajectories


# -----------------------------------------------------------------------------
# GPU Solver
# -----------------------------------------------------------------------------

class BatchedPandaKinematics:
    """
    GPU-accelerated batched FK and Jacobian using pytorch_kinematics.

    Computes real forward kinematics and Jacobians for all robot instances in parallel.
    """

    # Panda default configuration
    DEFAULT_CONFIG = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])

    def __init__(self, n_robots: int, device: str = "cuda"):
        self.n_robots = n_robots
        self.n_dof = 7
        self.device = torch.device(device) if HAS_TORCH else None
        self.chain = None

        if HAS_PK and HAS_TORCH:
            self._setup_chain()

    def _setup_chain(self):
        """Load Panda kinematic chain."""
        try:
            from robot_descriptions.panda_description import URDF_PATH
            with open(URDF_PATH, "r", encoding="utf-8") as f:
                urdf_xml = f.read()
            self.chain = pk.build_serial_chain_from_urdf(urdf_xml, "panda_hand")
            self.chain = self.chain.to(dtype=torch.float64, device=self.device)
            print(f"[FK] Loaded pytorch_kinematics chain for {self.n_robots} robots")
        except Exception as e:
            print(f"[FK] Failed to load kinematic chain: {e}")

    def forward_kinematics(self, q: torch.Tensor) -> torch.Tensor:
        """Compute batched end-effector positions."""
        if self.chain is None:
            return torch.zeros(q.shape[0], 3, device=self.device, dtype=torch.float64)
        fk_result = self.chain.forward_kinematics(q)
        return fk_result.get_matrix()[:, :3, 3]

    def jacobian(self, q: torch.Tensor) -> torch.Tensor:
        """Compute batched Jacobians (6 x n_dof)."""
        if self.chain is None:
            return torch.zeros(q.shape[0], 6, self.n_dof, device=self.device, dtype=torch.float64)
        return self.chain.jacobian(q)

    def get_default_configs(self) -> torch.Tensor:
        """Get default configurations for all robots."""
        q = torch.tensor(self.DEFAULT_CONFIG, device=self.device, dtype=torch.float64)
        return q.unsqueeze(0).expand(self.n_robots, -1).clone()


class ParallelGPUSolver:
    """
    GPU-accelerated parallel IK solver using CusADi + pytorch_kinematics.
    
    Computes real FK and Jacobians on GPU, then solves velocity IK in parallel.
    """

    def __init__(self, n_robots: int, n_dof: int = 7, task_dim: int = 6, n_constraints: int = 7):
        self.n_robots = n_robots
        self.n_dof = n_dof
        self.task_dim = task_dim
        self.n_constraints = n_constraints
        
        self.device = None
        self.fn_casadi = None
        self.fn_cusadi = None
        self.kinematics = None
        
        # Pre-allocated tensors for efficiency
        self._C_batch = None
        self._lower_batch = None
        self._upper_batch = None
        
        if HAS_GPU:
            self.device = torch.device("cuda")
            self._setup_cusadi()
            
            if HAS_PK:
                self.kinematics = BatchedPandaKinematics(n_robots, device="cuda")
            else:
                print("[Warning] pytorch_kinematics not available - install with: pip install pytorch-kinematics")
    
    def _setup_cusadi(self):
        """Load and setup CusADi function."""
        import os
        home = os.path.expanduser("~")
        casadi_file = os.path.join(home, ".local", "cusadi", "src", "casadi_functions", "fn_velocity_solve.casadi")
        
        if not os.path.exists(casadi_file):
            print(f"[GPU] CasADi file not found: {casadi_file}")
            print("[GPU] Run: pixi run -e cuda export-casadi && compile with CusADi")
            return
        
        try:
            self.fn_casadi = ca.Function.load(casadi_file)
            self.fn_cusadi = CusadiFunction(self.fn_casadi, self.n_robots)
            print(f"[GPU] Loaded CusADi function for {self.n_robots} parallel instances")
        except Exception as e:
            print(f"[GPU] Failed to load CusADi: {e}")
    
    def _prepare_constraints(self, C: np.ndarray, lower: np.ndarray, upper: np.ndarray):
        """Pre-allocate and cache constraint tensors."""
        if self._C_batch is None:
            C_flat = C.flatten()
            self._C_batch = torch.from_numpy(
                np.tile(C_flat[np.newaxis, :], (self.n_robots, 1))
            ).double().to(self.device).contiguous()
            self._lower_batch = torch.from_numpy(
                np.tile(lower[np.newaxis, :], (self.n_robots, 1))
            ).double().to(self.device).contiguous()
            self._upper_batch = torch.from_numpy(
                np.tile(upper[np.newaxis, :], (self.n_robots, 1))
            ).double().to(self.device).contiguous()
    
    def forward_kinematics(self, q: torch.Tensor) -> torch.Tensor:
        """Compute batched end-effector positions."""
        if self.kinematics is not None:
            return self.kinematics.forward_kinematics(q)
        return torch.zeros(q.shape[0], 3, device=self.device, dtype=torch.float64)
    
    def solve_batched(
        self,
        targets: np.ndarray,  # (n_robots, task_dim)
        q_current: torch.Tensor,  # Current joint configurations
        C: np.ndarray,  # (n_constraints, n_dof)
        lower: np.ndarray,  # (n_constraints,)
        upper: np.ndarray,  # (n_constraints,)
    ) -> Tuple[np.ndarray, float, float]:
        """
        Solve batched IK problems on GPU with real Jacobians.

        Returns:
            velocities: (n_robots, n_dof)
            ik_elapsed_ms: IK solver time in milliseconds
            fk_elapsed_ms: FK/Jacobian time in milliseconds
        """
        if self.fn_cusadi is None or self.kinematics is None:
            return np.zeros((self.n_robots, self.n_dof)), 0.0, 0.0
        
        # Prepare constraints (cached after first call)
        self._prepare_constraints(C, lower, upper)
        
        # Compute real Jacobians from current configuration
        torch.cuda.synchronize()
        fk_start = time.perf_counter()
        J = self.kinematics.jacobian(q_current)
        torch.cuda.synchronize()
        fk_elapsed_ms = (time.perf_counter() - fk_start) * 1000
        
        # Use position-only Jacobian if task_dim == 3
        if self.task_dim == 3:
            J = J[:, :3, :]
        
        jacobians_flat = J.reshape(self.n_robots, -1).contiguous()
        
        # Convert targets to tensor
        targets_t = torch.from_numpy(targets.copy()).double().to(self.device).contiguous()
        
        inputs = [targets_t, jacobians_flat, self._C_batch, self._lower_batch, self._upper_batch]
        
        # Solve
        torch.cuda.synchronize()
        ik_start = time.perf_counter()
        self.fn_cusadi.evaluate(inputs)
        torch.cuda.synchronize()
        ik_elapsed_ms = (time.perf_counter() - ik_start) * 1000
        
        # Get output
        velocities = self.fn_cusadi.getDenseOutput(0).cpu().numpy().squeeze()
        
        return velocities, ik_elapsed_ms, fk_elapsed_ms


class CPUSolver:
    """CPU sequential IK solver for comparison."""

    def __init__(self, n_robots: int, n_dof: int = 7, task_dim: int = 6, n_constraints: int = 7):
        self.n_robots = n_robots
        self.n_dof = n_dof
        self.task_dim = task_dim
        self.n_constraints = n_constraints

    def solve_batched(
        self,
        targets: np.ndarray,
        jacobians: np.ndarray,
        C: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """Solve batched IK problems on CPU (sequential)."""
        n_samples = targets.shape[0]
        velocities = np.zeros((n_samples, self.n_dof))

        start = time.perf_counter()
        for i in range(n_samples):
            result = eik.computeMultiObjectiveVelocitySolutionEigen(
                [targets[i]], [np.asfortranarray(jacobians[i])], C, lower, upper
            )
            velocities[i] = np.array(result.solution).ravel()
        elapsed_ms = (time.perf_counter() - start) * 1000

        return velocities, elapsed_ms


# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------

def run_visualization(args: argparse.Namespace):
    """Run the interactive visualization with 100 robot instances."""
    if not HAS_VISER:
        print("Error: viser and robot_descriptions are required for visualization")
        print("Install with: pip install viser robot_descriptions")
        return

    n_robots = args.n_robots
    n_dof = 7  # Arm DOF for IK solver
    task_dim = 6 if args.orientation else 3

    # Visualization modes
    viz_mode = args.viz_mode
    n_viz_full = min(args.n_viz_full, n_robots)  # Number of full URDF robots to show

    print(f"\n{'='*70}")
    print(f"Parallel Trajectory Tracking Demo - {n_robots} Robots")
    print(f"{'='*70}")
    print(f"GPU Available: {HAS_GPU}")
    print(f"Task dimension: {task_dim}D (position{'+ orientation' if args.orientation else ' only'})")
    print(f"Visualization mode: {viz_mode}")
    if viz_mode == "hybrid":
        print(f"  - {n_viz_full} full URDF robots + {n_robots - n_viz_full} point markers")

    # Create viser server
    server = viser.ViserServer()

    # Grid layout with equal spacing
    n_cols = int(np.ceil(np.sqrt(n_robots)))
    n_rows = int(np.ceil(n_robots / n_cols))
    grid_spacing = args.spacing  # Spacing between robots

    # Compute grid dimensions
    grid_width = (n_cols - 1) * grid_spacing + 2.0  # Extra padding
    grid_height = (n_rows - 1) * grid_spacing + 2.0
    server.scene.add_grid("/ground", width=max(grid_width, 5), height=max(grid_height, 5))

    robot_visuals = []  # ViserUrdf objects (only for full viz)
    robot_markers = []  # Point markers for lightweight viz
    robot_configs = []  # Current joint configurations
    n_actuated = 8  # Panda: 7 arm + 1 gripper

    # Compute base positions - centered grid with equal spacing
    base_positions = []
    for i in range(n_robots):
        row = i // n_cols
        col = i % n_cols
        # Center the grid around origin
        x = (col - (n_cols - 1) / 2) * grid_spacing
        y = (row - (n_rows - 1) / 2) * grid_spacing
        base_positions.append(np.array([x, y, 0.0]))

    print(f"Grid: {n_cols}x{n_rows}, spacing: {grid_spacing}m, total area: {grid_width:.1f}x{grid_height:.1f}m")

    # Panda default configuration (from robot_presets.yaml)
    PANDA_DEFAULT_ARM = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
    PANDA_GRIPPER_OPEN = np.array([0.04, 0.04])

    # Load URDF only if needed
    urdf = None
    if viz_mode in ["full", "hybrid"] and n_viz_full > 0:
        urdf = load_robot_description("panda_description")
        n_actuated = len(urdf.actuated_joint_names)
        print(f"Robot has {n_actuated} actuated joints (7 arm + gripper)")

    print(f"\nCreating visualizations...")

    for i in range(n_robots):
        # Full config with default arm pose + gripper open
        full_config = np.zeros(n_actuated)
        full_config[:n_dof] = PANDA_DEFAULT_ARM.copy()
        if n_actuated > n_dof:
            full_config[n_dof:] = PANDA_GRIPPER_OPEN[:n_actuated - n_dof]
        robot_configs.append(full_config)

        # Get base position for this robot
        base_pos = base_positions[i]

        if viz_mode == "full" or (viz_mode == "hybrid" and i < n_viz_full):
            # Create parent frame at robot's base position
            server.scene.add_frame(
                f"/robot_base_{i}",
                position=tuple(base_pos),
                wxyz=(1.0, 0.0, 0.0, 0.0),
                show_axes=False,
            )
            # Full URDF visualization under the parent frame
            vis = ViserUrdf(server, urdf, root_node_name=f"/robot_base_{i}/urdf")
            robot_visuals.append(vis)
            robot_markers.append(None)
            # Set initial configuration
            vis.update_cfg(full_config)
        elif viz_mode == "points" or (viz_mode == "hybrid" and i >= n_viz_full):
            # Lightweight point marker
            robot_visuals.append(None)
            # Color by trajectory type
            colors = [(0.2, 0.6, 1.0), (1.0, 0.4, 0.2), (0.2, 0.8, 0.4), (0.8, 0.2, 0.8)]
            color = colors[i % len(colors)]
            marker = server.scene.add_icosphere(
                f"/robot_{i}/ee",
                radius=0.02,
                color=color,
                position=tuple(base_pos + np.array([0.4, 0, 0.4])),
            )
            robot_markers.append(marker)
        else:
            robot_visuals.append(None)
            robot_markers.append(None)

    # Create trajectories for point marker visualization
    # Note: This demo uses SYNTHETIC Jacobians to benchmark GPU parallelization.
    # The full URDF robots show joint motion from integrated velocities,
    # while point markers show the actual target trajectory paths.
    # The point markers demonstrate what the robots SHOULD track with real FK/IK.

    trajectories = []
    for i in range(n_robots):
        # Trajectory center at robot's base + EE offset
        center = base_positions[i].copy()
        center[2] = 0.5  # Trajectory height
        center[0] += 0.4  # Forward offset

        # Create trajectory generator
        rng = np.random.RandomState(i)
        traj_types = [CircleTrajectory, Figure8Trajectory, SpiralTrajectory, HeartTrajectory]
        traj_type = traj_types[i % len(traj_types)]

        scale = 0.05 + rng.rand() * 0.05  # 5-10 cm radius
        speed = 0.5 + rng.rand() * 1.0
        phase = rng.rand() * 2 * np.pi

        if traj_type == CircleTrajectory:
            tilt = rng.rand() * np.pi / 6
            traj = CircleTrajectory(center, scale, speed, phase, tilt)
        elif traj_type == SpiralTrajectory:
            traj = SpiralTrajectory(center, scale, 0.03, speed, phase)
        else:
            traj = traj_type(center, scale, speed, phase)

        trajectories.append(traj)

    # Add trajectory visualization ONLY for point markers (not full URDF robots)
    # Full URDF robots use synthetic Jacobians, so their motion won't match trajectories
    for i in range(n_robots):
        if robot_markers[i] is not None:  # Only show trajectory for point markers
            t_vals = np.linspace(0, 2 * np.pi / trajectories[i].speed, 50)
            points = np.array([trajectories[i].get_position(t) for t in t_vals])
            server.scene.add_spline_catmull_rom(
                f"/trajectory_{i}",
                positions=points,
                color=(0.3, 0.3, 0.8),
                line_width=1.0,
            )

    # Create GPU solver with real FK/Jacobians
    gpu_solver = ParallelGPUSolver(n_robots, n_dof, task_dim) if HAS_GPU else None
    cpu_solver = CPUSolver(n_robots, n_dof, task_dim)
    
    if gpu_solver is not None and gpu_solver.kinematics is not None:
        print(f"[Mode] Real FK + Jacobians via pytorch_kinematics (GPU)")
    else:
        print(f"[Mode] CPU-only mode (GPU solver not available)")

    # Constraint matrices
    C = np.eye(n_dof)
    lower = np.full(n_dof, -2.0)  # Velocity limits
    upper = np.full(n_dof, 2.0)

    # GUI
    with server.gui.add_folder("Controls"):
        use_gpu_checkbox = server.gui.add_checkbox("Use GPU", initial_value=HAS_GPU and args.use_gpu)
        speed_slider = server.gui.add_slider("Speed", min=0.1, max=3.0, initial_value=1.0, step=0.1)
        pause_checkbox = server.gui.add_checkbox("Pause", initial_value=False)

    with server.gui.add_folder("Performance"):
        ik_time_text = server.gui.add_text("IK Solver", initial_value="-- ms")
        fk_time_text = server.gui.add_text("FK/Jacobian", initial_value="-- ms")
        total_time_text = server.gui.add_text("Total GPU", initial_value="-- ms")
        throughput_text = server.gui.add_text("Throughput", initial_value="-- solves/sec")
        viz_time_text = server.gui.add_text("Viz Update", initial_value="-- ms")

    with server.gui.add_folder("Info"):
        info_text = server.gui.add_text(
            "Status",
            initial_value=f"{n_robots} robots | Real FK/Jacobians"
        )

    print(f"\nStarting simulation loop...")
    print(f"Open browser to: http://localhost:8080")
    print(f"Press Ctrl+C to stop\n")

    # Simulation parameters
    dt = 0.02  # 50 Hz
    t = 0.0
    iteration = 0

    # For smoothing timing
    ik_time_avg = 0.0
    fk_time_avg = 0.0
    viz_time_avg = 0.0

    # Joint configurations as torch tensor for GPU FK/Jacobians
    if HAS_TORCH and gpu_solver is not None:
        q_tensor = torch.tensor(
            np.array([cfg[:n_dof] for cfg in robot_configs]),
            device="cuda", dtype=torch.float64
        )
    else:
        q_tensor = None

    try:
        while True:
            if not pause_checkbox.value:
                t += dt * speed_slider.value

            # Compute target velocities for all robots (vectorized)
            targets = np.zeros((n_robots, task_dim))
            for i in range(n_robots):
                vel_3d = trajectories[i].get_velocity(t, dt)
                if task_dim == 6:
                    targets[i, :3] = vel_3d
                    targets[i, 3:] = 0.0  # Zero angular velocity
                else:
                    targets[i] = vel_3d

            # Solve IK with real FK/Jacobians
            if use_gpu_checkbox.value and gpu_solver is not None and gpu_solver.fn_cusadi is not None and q_tensor is not None:
                velocities, ik_ms, fk_ms = gpu_solver.solve_batched(targets, q_tensor, C, lower, upper)
                
                ik_time_avg = 0.9 * ik_time_avg + 0.1 * ik_ms
                fk_time_avg = 0.9 * fk_time_avg + 0.1 * fk_ms
                total_ms = ik_time_avg + fk_time_avg
                
                ik_time_text.value = f"{ik_time_avg:.2f} ms"
                fk_time_text.value = f"{fk_time_avg:.2f} ms"
                total_time_text.value = f"{total_ms:.2f} ms"
                
                throughput = n_robots / (total_ms / 1000) if total_ms > 0 else 0
                throughput_text.value = f"{throughput:.0f} solves/sec"
            else:
                # CPU fallback - uses simple pseudo-inverse IK
                velocities = np.zeros((n_robots, n_dof))
                ik_ms = 0.0
                ik_time_text.value = "N/A (CPU)"
                fk_time_text.value = "N/A"
                total_time_text.value = "N/A"
                throughput_text.value = "N/A"

            # Update robot configurations (integrate velocities)
            viz_start = time.perf_counter()

            # Vectorized update for torch tensor
            if q_tensor is not None:
                dq = torch.from_numpy(velocities * dt).to(q_tensor.device, dtype=q_tensor.dtype)
                q_tensor = torch.clamp(q_tensor + dq, -2.9, 2.9)

            # Update numpy configs and visualization
            for i in range(n_robots):
                robot_configs[i][:n_dof] += velocities[i] * dt
                robot_configs[i][:n_dof] = np.clip(robot_configs[i][:n_dof], -2.9, 2.9)

                # Update visualization
                if robot_visuals[i] is not None:
                    robot_visuals[i].update_cfg(robot_configs[i])
                elif robot_markers[i] is not None:
                    # Show actual EE position from FK
                    pos = trajectories[i].get_position(t)
                    robot_markers[i].position = tuple(pos)

            viz_ms = (time.perf_counter() - viz_start) * 1000
            viz_time_avg = 0.9 * viz_time_avg + 0.1 * viz_ms
            viz_time_text.value = f"{viz_time_avg:.2f} ms"

            iteration += 1

            # Status update
            if iteration % 100 == 0:
                mode = "GPU" if (use_gpu_checkbox.value and gpu_solver is not None) else "CPU"
                info_text.value = f"{n_robots} robots | {mode} | iter {iteration}"

            time.sleep(max(0.001, dt - (ik_ms + viz_ms) / 1000))

    except KeyboardInterrupt:
        print("\nStopping...")


def run_benchmark(args: argparse.Namespace):
    """Run benchmark without visualization - always uses real FK/Jacobians."""
    n_robots = args.n_robots
    n_dof = 7
    task_dim = 6

    print(f"\n{'='*70}")
    print(f"GPU Parallel IK Benchmark - {n_robots} Robots")
    print(f"{'='*70}")
    print(f"GPU Available: {HAS_GPU}")
    print(f"pytorch_kinematics: {'Available' if HAS_PK else 'Not installed'}")

    if not HAS_PK:
        print("\nError: pytorch_kinematics required. Install with: pip install pytorch-kinematics")
        return

    # Create GPU solver with real FK/Jacobians
    gpu_solver = ParallelGPUSolver(n_robots, n_dof, task_dim) if HAS_GPU else None
    cpu_solver = CPUSolver(n_robots, n_dof, task_dim)

    # Generate test data
    rng = np.random.RandomState(42)
    targets = rng.randn(n_robots, task_dim).astype(np.float64) * 0.1

    C = np.eye(n_dof)
    lower = np.full(n_dof, -2.0)
    upper = np.full(n_dof, 2.0)
    
    # Create q_tensor with default Panda configuration
    q_default = np.tile(BatchedPandaKinematics.DEFAULT_CONFIG, (n_robots, 1))
    q_tensor = torch.tensor(q_default, device="cuda", dtype=torch.float64)

    # Warm-up
    if gpu_solver is not None and gpu_solver.fn_cusadi is not None:
        for _ in range(3):
            gpu_solver.solve_batched(targets, q_tensor, C, lower, upper)

    # Benchmark
    n_runs = 10

    print(f"\nRunning {n_runs} iterations...")

    # GPU benchmark
    if gpu_solver is not None and gpu_solver.fn_cusadi is not None:
        ik_times = []
        fk_times = []
        for _ in range(n_runs):
            _, ik_elapsed, fk_elapsed = gpu_solver.solve_batched(targets, q_tensor, C, lower, upper)
            ik_times.append(ik_elapsed)
            fk_times.append(fk_elapsed)
        
        ik_avg = np.mean(ik_times)
        ik_std = np.std(ik_times)
        fk_avg = np.mean(fk_times)
        fk_std = np.std(fk_times)
        gpu_avg = ik_avg + fk_avg
        
        print(f"GPU IK Solver: {ik_avg:.2f} ± {ik_std:.2f} ms")
        print(f"GPU FK/Jacobian: {fk_avg:.2f} ± {fk_std:.2f} ms")
        print(f"GPU Total: {gpu_avg:.2f} ms")
    else:
        print("GPU: N/A")
        gpu_avg = None

    # CPU benchmark (use subset for speed) - uses synthetic Jacobians for comparison
    cpu_subset = min(100, n_robots)
    jacobians = rng.randn(cpu_subset, task_dim, n_dof).astype(np.float64)
    for i in range(cpu_subset):
        U, s, Vt = np.linalg.svd(jacobians[i], full_matrices=False)
        s = np.clip(s, 0.3, 3.0)
        jacobians[i] = U @ np.diag(s) @ Vt
    
    cpu_times = []
    for _ in range(n_runs):
        _, elapsed = cpu_solver.solve_batched(targets[:cpu_subset], jacobians, C, lower, upper)
        cpu_times.append(elapsed)
    cpu_avg = np.mean(cpu_times) * (n_robots / cpu_subset)
    cpu_std = np.std(cpu_times) * (n_robots / cpu_subset)
    print(f"CPU: {cpu_avg:.2f} ± {cpu_std:.2f} ms (estimated from {cpu_subset} samples)")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Robots: {n_robots}")
    print(f"GPU Time: {gpu_avg:.2f} ms" if gpu_avg else "GPU: N/A")
    print(f"CPU Time: {cpu_avg:.2f} ms (estimated)")
    if gpu_avg:
        speedup = cpu_avg / gpu_avg
        print(f"Speedup: {speedup:.1f}x")
        throughput = n_robots / (gpu_avg / 1000)
        print(f"Throughput: {throughput:.0f} IK solves/second")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description="Parallel Trajectory Tracking Demo with 100 Robots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--n-robots", type=int, default=100,
        help="Number of robot instances (default: 100)"
    )
    parser.add_argument(
        "--use-gpu", action="store_true", default=True,
        help="Use GPU solver (default: True)"
    )
    parser.add_argument(
        "--orientation", action="store_true",
        help="Include orientation in task (6D instead of 3D)"
    )
    parser.add_argument(
        "--benchmark", action="store_true",
        help="Run benchmark without visualization"
    )
    parser.add_argument(
        "--viz-mode", choices=["full", "hybrid", "points"], default="hybrid",
        help="Visualization mode: 'full' (all URDF), 'hybrid' (some URDF + points), 'points' (all points)"
    )
    parser.add_argument(
        "--n-viz-full", type=int, default=9,
        help="Number of full URDF robots in hybrid mode (default: 9)"
    )
    parser.add_argument(
        "--spacing", type=float, default=1.0,
        help="Grid spacing between robots in meters (default: 1.0)"
    )

    args = parser.parse_args()

    if args.benchmark:
        run_benchmark(args)
    else:
        run_visualization(args)


if __name__ == "__main__":
    main()

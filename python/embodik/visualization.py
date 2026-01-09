"""Real-time 3D visualization for embodiK using Viser directly."""

import numpy as np
from typing import Dict, Optional, Tuple, List, Any, Callable
import trimesh
import threading
import time
from pathlib import Path

# Import Viser and related libraries directly
import viser
from viser import ViserServer
from viser.extras import ViserUrdf
import yourdfpy

# For transforms and quaternions - use system pinocchio if available
try:
    from ._runtime_deps import import_pinocchio as _import_pinocchio

    pin = _import_pinocchio()
except ImportError:
    # If pinocchio not available, we can use numpy/scipy for transforms
    pin = None
    import scipy.spatial.transform as transform


def matrix_to_quaternion_wxyz(rotation_matrix):
    """Convert rotation matrix to quaternion in wxyz format."""
    if pin is not None:
        # Use Pinocchio
        q = pin.Quaternion(rotation_matrix)
        return (q.w, q.x, q.y, q.z)
    else:
        # Use scipy
        r = transform.Rotation.from_matrix(rotation_matrix)
        q = r.as_quat()  # Returns xyzw
        return (q[3], q[0], q[1], q[2])  # Convert to wxyz


def quaternion_wxyz_to_matrix(wxyz):
    """Convert quaternion in wxyz format to rotation matrix."""
    if pin is not None:
        # Use Pinocchio
        q = pin.Quaternion(wxyz[0], wxyz[1], wxyz[2], wxyz[3])
        return q.matrix()
    else:
        # Use scipy
        q_xyzw = [wxyz[1], wxyz[2], wxyz[3], wxyz[0]]  # Convert to xyzw
        r = transform.Rotation.from_quat(q_xyzw)
        return r.as_matrix()


class EmbodikVisualizer:
    """embodiK visualization using Viser directly (no Pinocchio visualization dependencies)."""

    def __init__(self, robot_model: Any, port: int = 8080, open_browser: bool = True):
        """
        Initialize visualizer using Viser directly.

        Args:
            robot_model: embodiK RobotModel instance
            port: Port for Viser web server
            open_browser: Whether to automatically open browser
        """
        self.robot_model = robot_model
        self.server = ViserServer(port=port)

        # Load URDF for visualization
        self._load_urdf_visualization()

        # Custom visualization elements
        self._targets = {}
        self._com_marker = None
        self._support_polygon = None

        # Add GUI controls
        self._setup_gui()

        if open_browser:
            print(f"Viser server started at http://localhost:{port}")

    def _load_urdf_visualization(self):
        """Load URDF for visualization using yourdfpy."""
        # Get URDF path from robot model
        urdf_path = getattr(self.robot_model, 'urdf_path', None)
        if not urdf_path:
            print("Warning: No URDF path available from robot model")
            self.urdf = None
            self.urdf_vis = None
            return

        # Store URDF path for use in filename handler
        self.urdf_path = urdf_path

        # Create filename handler for mesh resolution
        filename_handler = self._create_filename_handler()

        try:
            # Load URDF with yourdfpy
            self.urdf = yourdfpy.URDF.load(
                urdf_path,
                filename_handler=filename_handler,
                build_scene_graph=True,
                build_collision_scene_graph=False,
                load_meshes=True,
                load_collision_meshes=False,
            )

            # Create ViserUrdf for visualization
            self.urdf_vis = ViserUrdf(
                self.server,
                self.urdf,
                root_node_name="/robot"
            )

            print(f"✓ URDF loaded successfully for visualization")
            print(f"  URDF joints: {self.urdf.joint_names}")

        except Exception as e:
            print(f"Warning: Could not load URDF for visualization: {e}")
            self.urdf = None
            self.urdf_vis = None

    def _create_filename_handler(self) -> Callable:
        """Create a filename handler for mesh resolution."""
        # Try to get mesh paths from robot model or use defaults
        package_paths = {}

        # Common ROS package locations
        common_paths = [
            Path.home() / ".ros",
            Path("/opt/ros"),
            Path.home() / "catkin_ws/src",
            Path.home() / "ros2_ws/src",
        ]

        def filename_handler(fname: str) -> str:
            """Resolve package:// URLs to actual file paths."""
            filename = fname  # yourdfpy uses 'fname' as the parameter name
            if filename.startswith("package://"):
                # Handle package:///filename (no package name, just filename)
                if filename.startswith("package:///"):
                    # This means the file is relative to the URDF directory
                    relative_filename = filename[11:]  # Remove "package:///"
                    urdf_dir = Path(self.urdf_path).parent
                    full_path = urdf_dir / relative_filename
                    if full_path.exists():
                        print(f"Resolved {filename} -> {full_path}")
                        return str(full_path)
                    print(f"Warning: Could not find {relative_filename} in {urdf_dir}")
                    return filename

                # Extract package name and path
                package_part = filename[10:]  # Remove "package://"
                parts = package_part.split("/", 1)
                if len(parts) == 2:
                    package_name, path = parts

                    # Check if we have a known path for this package
                    if package_name in package_paths:
                        return str(package_paths[package_name] / path)

                    # Search in common locations
                    for base_path in common_paths:
                        package_path = base_path / package_name
                        if package_path.exists():
                            full_path = package_path / path
                            if full_path.exists():
                                return str(full_path)

            # Return as-is if not a package URL or not found
            return filename

        return filename_handler

    def display(self, q: np.ndarray):
        """Update robot display with new configuration."""
        if self.urdf_vis and self.urdf:
            # Update URDF configuration
            # Map from robot model joints to URDF joints
            cfg = np.zeros(len(self.urdf.joint_names))

            # Get joint names from robot model
            robot_joint_names = self.robot_model.get_joint_names()

            # Debug first call
            if not hasattr(self, '_debug_printed'):
                print(f"  Robot joints: {robot_joint_names}")
                print(f"  Joint config q: {q}")
                self._debug_printed = True

            # Map joint values - try different approaches
            # 1. First try using controlled_joints if available
            controlled_joints = getattr(self.robot_model, 'controlled_joint_names', [])
            controlled_indices = getattr(self.robot_model, 'controlled_joint_indices', {})

            if controlled_joints and controlled_indices:
                for joint_name in controlled_joints:
                    if joint_name in self.urdf.joint_names and joint_name in controlled_indices:
                        urdf_idx = self.urdf.joint_names.index(joint_name)
                        robot_idx = controlled_indices[joint_name]
                        if robot_idx < len(q):
                            cfg[urdf_idx] = q[robot_idx]
            else:
                # 2. Fall back to direct mapping by name order
                for i, joint_name in enumerate(robot_joint_names):
                    if joint_name in self.urdf.joint_names and i < len(q):
                        urdf_idx = self.urdf.joint_names.index(joint_name)
                        cfg[urdf_idx] = q[i]

            self.urdf_vis.update_cfg(cfg)

        # Update COM if enabled
        if hasattr(self, 'gui_show_com') and self.gui_show_com and self.gui_show_com.value:
            com_position = self.robot_model.get_com_position()
            self.visualize_com(com_position)

    def update(self, q: np.ndarray):
        """Alias for display() to maintain compatibility."""
        self.display(q)

    def set_display_options(self, visuals: bool = True, collisions: bool = False, frames: bool = False):
        """Set display options for robot visualization."""
        # With direct Viser control, we can toggle visibility
        if self.urdf_vis:
            self.urdf_vis.visible = visuals

        # Frames and collisions would need custom implementation
        if frames:
            print("Frame display not yet implemented in direct Viser mode")
        if collisions:
            print("Collision display not yet implemented in direct Viser mode")

    def add_target_marker(self, name: str, pose: np.ndarray,
                         color: Tuple[float, float, float] = (0, 1, 0), size: float = 0.05):
        """Add IK target marker."""
        position = pose[:3, 3]
        # Convert to quaternion (wxyz format)
        quaternion = matrix_to_quaternion_wxyz(pose[:3, :3])

        # Add sphere for position
        sphere_name = f"/targets/{name}/sphere"
        sphere = self.server.scene.add_icosphere(
            sphere_name,
            radius=size,
            position=position,
            color=color
        )

        # Add frame for orientation
        frame_name = f"/targets/{name}/frame"
        frame = self.server.scene.add_frame(
            frame_name,
            wxyz=quaternion,
            position=position,
            axes_length=size * 2,
            axes_radius=size * 0.2
        )

        self._targets[name] = (sphere_name, frame_name, sphere, frame)

    def visualize_com(self, com_position: np.ndarray, color: Tuple[float, float, float] = (1, 1, 0)):
        """Visualize center of mass."""
        if self._com_marker:
            # Update position instead of recreating
            self._com_marker.position = com_position
        else:
            self._com_marker = self.server.scene.add_icosphere(
                "/com",
                radius=0.02,
                position=com_position,
                color=color
            )

    def visualize_support_polygon(self, vertices: np.ndarray,
                                 color: Tuple[float, float, float] = (0, 0, 1), opacity: float = 0.3):
        """Visualize support polygon for balance constraint."""
        if len(vertices) < 3:
            return

        # Remove existing polygon
        if self._support_polygon:
            self.server.scene.remove("/constraints/support_polygon")

        # Ensure 3D vertices
        vertices_3d = np.array(vertices)
        if vertices_3d.shape[1] == 2:
            vertices_3d = np.hstack([vertices_3d, np.zeros((len(vertices), 1))])

        # Create mesh using trimesh
        try:
            mesh = trimesh.Trimesh(
                vertices=vertices_3d,
                faces=trimesh.earcut.triangulate_polygon(vertices_3d[:, :2])[0]
            )

            self._support_polygon = self.server.scene.add_mesh_simple(
                "/constraints/support_polygon",
                vertices=mesh.vertices,
                faces=mesh.faces,
                color=color,
                opacity=opacity
            )
        except Exception as e:
            print(f"Error creating support polygon: {e}")

    def clear_targets(self):
        """Clear all target markers."""
        for name, (sphere_name, frame_name, _, _) in self._targets.items():
            self.server.scene.remove(sphere_name)
            self.server.scene.remove(frame_name)
        self._targets.clear()

    def _setup_gui(self):
        """Set up GUI controls."""
        with self.server.gui.add_folder("Swift IK Controls"):
            # Display options
            self.gui_show_visuals = self.server.gui.add_checkbox(
                "Show Visuals", initial_value=True)
            self.gui_show_com = self.server.gui.add_checkbox(
                "Show COM", initial_value=True)

            # Callbacks for display options
            @self.gui_show_visuals.on_update
            def _(_):
                if self.urdf_vis:
                    self.urdf_vis.visible = self.gui_show_visuals.value

        # Add robot info
        with self.server.gui.add_folder("Robot Info"):
            nq = self.robot_model.nq if hasattr(self.robot_model, 'nq') else 'N/A'
            nv = self.robot_model.nv if hasattr(self.robot_model, 'nv') else 'N/A'
            self.server.gui.add_markdown(
                f"**Configuration dimensions:**\n"
                f"- nq: {nq}\n"
                f"- nv: {nv}"
            )

    def capture_image(self, width: int = 1920, height: int = 1080) -> Optional[np.ndarray]:
        """Capture current view as image."""
        # Viser doesn't have direct image capture like Pinocchio's visualizer
        # This would need to be implemented via client-side screenshot
        print("Image capture not yet implemented in direct Viser mode")
        return None

    def create_animation(self, q_trajectory: List[np.ndarray], dt: float = 0.01,
                        filename: Optional[str] = None):
        """Create animation from trajectory."""
        if filename:
            print(f"Animation recording to {filename} not yet implemented in direct Viser mode")

        # Play animation in real-time
        for q in q_trajectory:
            self.display(q)
            time.sleep(dt)

    def start_animation_loop(self, update_callback: Callable[[], np.ndarray], dt: float = 0.01):
        """
        Start an animation loop that calls update_callback repeatedly.

        Args:
            update_callback: Function that returns new configuration q
            dt: Time step in seconds
        """
        def animation_thread():
            while True:
                try:
                    q = update_callback()
                    if q is not None:
                        self.display(q)
                except Exception as e:
                    print(f"Animation error: {e}")
                time.sleep(dt)

        thread = threading.Thread(target=animation_thread, daemon=True)
        thread.start()

    def wait(self):
        """Keep the server running (blocking call)."""
        while True:
            time.sleep(0.1)


class InteractiveVisualizer(EmbodikVisualizer):
    """Interactive visualizer with target manipulation."""

    def __init__(self, robot_model, port: int = 8080):
        super().__init__(robot_model, port)

        # Interactive handles
        self._interactive_targets: Dict[str, Any] = {}  # Viser transform controls

        # Callbacks
        self._target_update_callbacks = {}

    def add_interactive_target(self, name: str, initial_pose: np.ndarray,
                              callback: Optional[Callable[[np.ndarray], None]] = None, color: Tuple[float, float, float] = (0, 1, 0)):
        """
        Add an interactive target that can be manipulated.

        Args:
            name: Target name
            initial_pose: Initial 4x4 pose matrix
            callback: Function called when target is moved (receives new pose)
            color: Target color
        """
        position = initial_pose[:3, 3]
        quaternion = matrix_to_quaternion_wxyz(initial_pose[:3, :3])

        # Create transform controls
        controls = self.server.scene.add_transform_controls(
            f"/interactive_targets/{name}",
            wxyz=quaternion,
            position=position
        )

        self._interactive_targets[name] = controls

        # Add visual marker
        self.add_target_marker(name, initial_pose, color)

        # Store callback
        if callback:
            self._target_update_callbacks[name] = callback

        # Set up update handler
        @controls.on_update
        def _(_):
            # Build pose matrix from current controls
            pose = np.eye(4)
            pose[:3, 3] = np.array(controls.position)

            # Convert quaternion to rotation matrix (wxyz format)
            pose[:3, :3] = quaternion_wxyz_to_matrix(controls.wxyz)

            # Update visual marker
            self.add_target_marker(name, pose, color)

            # Call callback if registered
            if name in self._target_update_callbacks:
                self._target_update_callbacks[name](pose)

    def get_interactive_target_pose(self, name: str) -> Optional[np.ndarray]:
        """Get current pose of interactive target."""
        if name not in self._interactive_targets:
            return None

        controls = self._interactive_targets[name]

        # Build pose matrix
        pose = np.eye(4)
        pose[:3, 3] = np.array(controls.position)

        # Convert quaternion to rotation matrix
        pose[:3, :3] = quaternion_wxyz_to_matrix(controls.wxyz)

        return pose
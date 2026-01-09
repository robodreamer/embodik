"""Visualization using Pinocchio's built-in ViserVisualizer (pin >= 3.8.0).

This module uses Pinocchio's native ViserVisualizer when available,
falling back to the custom implementation if needed.
"""

import numpy as np
from typing import Optional, Tuple, List, Dict, Any
import warnings

# Try to import Pinocchio's ViserVisualizer
try:
    from ._runtime_deps import import_pinocchio as _import_pinocchio

    pin = _import_pinocchio()
    try:
        from pinocchio.visualize import ViserVisualizer, BaseVisualizer
        _PINOCCHIO_VISER_AVAILABLE = True
    except ImportError:
        # ViserVisualizer not available (pin < 3.8.0)
        _PINOCCHIO_VISER_AVAILABLE = False
        ViserVisualizer = None
        BaseVisualizer = None
except ImportError:
    _PINOCCHIO_VISER_AVAILABLE = False
    pin = None
    ViserVisualizer = None
    BaseVisualizer = None

# Try to import viser (required for visualization)
try:
    import viser
    _VISER_AVAILABLE = True
except ImportError:
    _VISER_AVAILABLE = False
    viser = None


def _build_geometry_models(robot_model, urdf_path: str, pin_model=None):
    """Build visual and collision geometry models from URDF.

    Args:
        robot_model: embodik RobotModel instance (unused, kept for compatibility)
        urdf_path: Path to URDF file
        pin_model: Pinocchio model (required)

    Returns:
        Tuple of (visual_model, collision_model) - may be None if building fails
    """
    if not _PINOCCHIO_VISER_AVAILABLE:
        return None, None

    if pin_model is None:
        raise ValueError("pin_model is required")

    try:
        # Build geometry models from URDF
        # Note: buildGeomFromUrdf requires package directories, but we can try without
        visual_model = None
        collision_model = None

        try:
            visual_model = pin.buildGeomFromUrdf(
                pin_model, urdf_path, pin.GeometryType.VISUAL,
                package_dirs=[]
            )
        except Exception as e:
            warnings.warn(f"Could not build visual model: {e}")
            visual_model = None

        try:
            collision_model = pin.buildGeomFromUrdf(
                pin_model, urdf_path, pin.GeometryType.COLLISION,
                package_dirs=[]
            )
        except Exception as e:
            warnings.warn(f"Could not build collision model: {e}")
            collision_model = None

        return visual_model, collision_model
    except Exception as e:
        warnings.warn(f"Error building geometry models: {e}")
        return None, None


class EmbodikVisualizer:
    """Visualization wrapper using Pinocchio's ViserVisualizer when available.

    This class uses Pinocchio's built-in ViserVisualizer (pin >= 3.8.0) if available,
    providing a clean interface that matches Pinocchio's visualization API.

    Example:
        >>> import embodik
        >>> model = embodik.RobotModel.from_urdf("robot.urdf")
        >>> viz = embodik.EmbodikVisualizer(model)
        >>> viz.display(model.get_current_configuration())
    """

    def __init__(
        self,
        robot_model,
        port: int = 8080,
        open_browser: bool = True,
        host: str = "localhost",
        use_pinocchio_visualizer: bool = True
    ):
        """
        Initialize visualizer.

        Args:
            robot_model: embodik RobotModel instance
            port: Port for Viser web server
            open_browser: Whether to automatically open browser
            host: Host for Viser server
            use_pinocchio_visualizer: If True, use Pinocchio's ViserVisualizer (requires pin >= 3.8.0).
                                     If False or unavailable, falls back to custom implementation.
        """
        self.robot_model = robot_model
        self.port = port
        self.host = host

        # Try to use Pinocchio's ViserVisualizer if available and requested
        if use_pinocchio_visualizer and _PINOCCHIO_VISER_AVAILABLE and _VISER_AVAILABLE:
            self._init_pinocchio_visualizer(open_browser)
        else:
            if use_pinocchio_visualizer:
                if not _PINOCCHIO_VISER_AVAILABLE:
                    warnings.warn(
                        "Pinocchio ViserVisualizer not available (requires pin >= 3.8.0). "
                        "Falling back to custom visualization. "
                        "Install pinocchio >= 3.8.0 to use built-in visualization."
                    )
                elif not _VISER_AVAILABLE:
                    warnings.warn(
                        "Viser not available. Install viser to use visualization."
                    )

            # Fall back to custom implementation
            from .visualization import EmbodikVisualizer as CustomVisualizer
            self._custom_viz = CustomVisualizer(robot_model, port, open_browser)
            self._use_pinocchio = False
            return

        self._use_pinocchio = True

    def _init_pinocchio_visualizer(self, open_browser: bool):
        """Initialize Pinocchio's ViserVisualizer."""
        # Rebuild Pinocchio model from URDF (since nanobind can't convert C++ Model type)
        urdf_path = self.robot_model.urdf_path

        # Build Pinocchio model from URDF
        self._pin_model = pin.buildModelFromUrdf(urdf_path)
        self._pin_data = self._pin_model.createData()

        # Build geometry models from URDF (can't access C++ geometry models via nanobind)
        visual_model, collision_model = _build_geometry_models(
            self.robot_model, urdf_path, self._pin_model
        )

        # Build geometry data if models exist
        collision_data = None
        visual_data = None
        if collision_model is not None:
            collision_data = collision_model.createData()
        if visual_model is not None:
            visual_data = visual_model.createData()

        # Create visualizer
        # Pinocchio's ViserVisualizer requires at least visual or collision model
        # If both are None, create empty geometry models
        if visual_model is None and collision_model is None:
            # Create empty geometry models to avoid None errors
            visual_model = pin.GeometryModel()
            collision_model = pin.GeometryModel()
            visual_data = visual_model.createData()
            collision_data = collision_model.createData()

        self.viz = ViserVisualizer(
            model=self._pin_model,
            collision_model=collision_model,
            visual_model=visual_model,
            data=self._pin_data,
            collision_data=collision_data,
            visual_data=visual_data
        )

        # Initialize viewer
        self.viz.initViewer(
            viewer=None,
            open=open_browser,
            loadModel=True,
            host=self.host,
            port=str(self.port)
        )

    def display(self, q: np.ndarray):
        """Update robot display with new configuration.

        Args:
            q: Joint configuration vector
        """
        if self._use_pinocchio:
            # Update robot model configuration
            self.robot_model.update_configuration(q)
            # Update Pinocchio data to match
            pin.forwardKinematics(self._pin_model, self._pin_data, q)
            pin.updateFramePlacements(self._pin_model, self._pin_data)
            # Display using Pinocchio's visualizer
            self.viz.display(q)
        else:
            # Use custom visualizer
            self._custom_viz.display(q)

    def update(self, q: np.ndarray):
        """Alias for display() to maintain compatibility."""
        self.display(q)

    def set_display_options(
        self,
        visuals: bool = True,
        collisions: bool = False,
        frames: bool = False
    ):
        """Set display options for robot visualization.

        Args:
            visuals: Show visual geometry
            collisions: Show collision geometry
            frames: Show coordinate frames
        """
        if self._use_pinocchio:
            self.viz.displayVisuals(visuals)
            self.viz.displayCollisions(collisions)
            self.viz.displayFrames(frames)
        else:
            self._custom_viz.set_display_options(visuals, collisions, frames)

    def add_target_marker(
        self,
        name: str,
        pose: np.ndarray,
        color: Tuple[float, float, float] = (0, 1, 0),
        size: float = 0.05
    ):
        """Add IK target marker.

        Args:
            name: Marker name
            pose: 4x4 pose matrix
            color: RGB color tuple
            size: Marker size
        """
        if self._use_pinocchio:
            # Use viser directly for custom markers
            position = pose[:3, 3]
            rotation = pose[:3, :3]

            # Convert rotation to quaternion (wxyz format)
            q = pin.Quaternion(rotation)
            wxyz = np.array([q.w, q.x, q.y, q.z])

            # Add sphere
            sphere_name = f"/targets/{name}/sphere"
            self.viz.viewer.scene.add_icosphere(
                sphere_name,
                radius=size,
                position=position,
                color=color
            )

            # Add frame
            frame_name = f"/targets/{name}/frame"
            self.viz.viewer.scene.add_frame(
                frame_name,
                wxyz=wxyz,
                position=position,
                axes_length=size * 2,
                axes_radius=size * 0.2
            )
        else:
            self._custom_viz.add_target_marker(name, pose, color, size)

    def visualize_com(
        self,
        com_position: np.ndarray,
        color: Tuple[float, float, float] = (1, 1, 0)
    ):
        """Visualize center of mass.

        Args:
            com_position: 3D COM position
            color: RGB color tuple
        """
        if self._use_pinocchio:
            if not hasattr(self, '_com_marker'):
                self._com_marker = self.viz.viewer.scene.add_icosphere(
                    "/com",
                    radius=0.02,
                    position=com_position,
                    color=color
                )
            else:
                self._com_marker.position = com_position
        else:
            self._custom_viz.visualize_com(com_position, color)

    def clear_targets(self):
        """Clear all target markers."""
        if self._use_pinocchio:
            # Remove targets from viser scene
            # Note: This is a simplified implementation
            # In practice, you'd track markers and remove them individually
            pass
        else:
            self._custom_viz.clear_targets()

    def capture_image(
        self,
        width: int = 1920,
        height: int = 1080
    ) -> Optional[np.ndarray]:
        """Capture current view as image.

        Args:
            width: Image width
            height: Image height

        Returns:
            RGB image array or None if capture fails
        """
        if self._use_pinocchio:
            try:
                return self.viz.captureImage(w=width, h=height)
            except Exception as e:
                warnings.warn(f"Image capture failed: {e}")
                return None
        else:
            return self._custom_viz.capture_image(width, height)

    def wait(self):
        """Keep the server running (blocking call)."""
        if self._use_pinocchio:
            import time
            while True:
                time.sleep(0.1)
        else:
            self._custom_viz.wait()


class InteractiveVisualizer(EmbodikVisualizer):
    """Interactive visualizer with target manipulation using Pinocchio's ViserVisualizer."""

    def __init__(self, robot_model, port: int = 8080, **kwargs):
        """
        Initialize interactive visualizer.

        Args:
            robot_model: embodik RobotModel instance
            port: Port for Viser web server
            **kwargs: Additional arguments passed to EmbodikVisualizer
        """
        super().__init__(robot_model, port, open_browser=True, **kwargs)

        if self._use_pinocchio:
            self._interactive_targets: Dict[str, Any] = {}
            self._target_update_callbacks = {}
        else:
            # Use custom interactive visualizer
            from .visualization import InteractiveVisualizer as CustomInteractive
            self._custom_interactive = CustomInteractive(robot_model, port)
            self._custom_viz = self._custom_interactive

    def add_interactive_target(
        self,
        name: str,
        initial_pose: np.ndarray,
        callback=None,
        color: Tuple[float, float, float] = (0, 1, 0)
    ):
        """
        Add an interactive target that can be manipulated.

        Args:
            name: Target name
            initial_pose: Initial 4x4 pose matrix
            callback: Function called when target is moved (receives new pose)
            color: Target color
        """
        if self._use_pinocchio:
            position = initial_pose[:3, 3]
            rotation = initial_pose[:3, :3]

            # Convert rotation to quaternion (wxyz format)
            q = pin.Quaternion(rotation)
            wxyz = np.array([q.w, q.x, q.y, q.z])

            # Create transform controls
            controls = self.viz.viewer.scene.add_transform_controls(
                f"/interactive_targets/{name}",
                wxyz=wxyz,
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

                # Convert quaternion to rotation matrix
                q_update = pin.Quaternion(controls.wxyz[0], controls.wxyz[1],
                                         controls.wxyz[2], controls.wxyz[3])
                pose[:3, :3] = q_update.matrix()

                # Update visual marker
                self.add_target_marker(name, pose, color)

                # Call callback if registered
                if name in self._target_update_callbacks:
                    self._target_update_callbacks[name](pose)
        else:
            self._custom_interactive.add_interactive_target(
                name, initial_pose, callback, color
            )

    def get_interactive_target_pose(self, name: str) -> Optional[np.ndarray]:
        """Get current pose of interactive target."""
        if self._use_pinocchio:
            if name not in self._interactive_targets:
                return None

            controls = self._interactive_targets[name]

            # Build pose matrix
            pose = np.eye(4)
            pose[:3, 3] = np.array(controls.position)

            # Convert quaternion to rotation matrix
            q = pin.Quaternion(controls.wxyz[0], controls.wxyz[1],
                              controls.wxyz[2], controls.wxyz[3])
            pose[:3, :3] = q.matrix()

            return pose
        else:
            return self._custom_interactive.get_interactive_target_pose(name)

"""Unified visualization helper for embodiK.

This module provides a unified interface for robot visualization that supports
both Pinocchio ViserVisualizer and ViserUrdf backends.
"""

from typing import Optional, Tuple, Literal
from pathlib import Path
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Backend type
BackendType = Literal["pinocchio", "viserurdf"]


class RobotVisualizer:
    """Unified robot visualizer supporting multiple backends.

    This class provides a common API for visualizing robots using either:
    - Pinocchio ViserVisualizer (default): Full-featured visualization with collision models
    - ViserUrdf: Color-preserving visualization using robot_descriptions

    Args:
        robot_model: embodik RobotModel instance
        backend: Visualization backend to use ("pinocchio" or "viserurdf")
        description_name: Robot description name for ViserUrdf backend (e.g., "panda_description")
        port: Port for Viser web server
        open_browser: Whether to automatically open browser
        host: Host for Viser server
        package_root: Optional package root directory for Pinocchio backend

    Attributes:
        server: Viser server instance
        scene: Viser scene instance
        gui: Viser GUI instance
        backend: Active backend name
    """

    def __init__(
        self,
        robot_model,
        backend: BackendType = "pinocchio",
        description_name: Optional[str] = None,
        port: int = 8080,
        open_browser: bool = True,
        host: str = "localhost",
        package_root: Optional[str] = None,
        load_collisions: bool = False,
    ):
        self.robot_model = robot_model
        self.backend = backend
        self.port = port
        self.host = host
        self.load_collisions = load_collisions

        # Initialize backend-specific components
        self._pinocchio_visualizer = None
        self._urdf_vis = None
        self._make_visual_config = None
        self._cfg_vec_cache = None  # Cache for ViserUrdf config array

        if backend == "pinocchio":
            self._init_pinocchio_backend(open_browser, package_root)
        elif backend == "viserurdf":
            self._init_viserurdf_backend(description_name, open_browser)
        else:
            raise ValueError(f"Unknown backend: {backend}. Must be 'pinocchio' or 'viserurdf'")

    def _init_pinocchio_backend(self, open_browser: bool, package_root: Optional[str]):
        """Initialize Pinocchio ViserVisualizer backend."""
        from .viser_helpers import create_viser_visualizer

        self._pinocchio_visualizer, self.server, self.scene, self.gui = create_viser_visualizer(
            robot_model=self.robot_model,
            port=self.port,
            open_browser=open_browser,
            host=self.host,
            package_root=package_root,
            load_collisions=self.load_collisions,
        )

        logger.info("Initialized Pinocchio ViserVisualizer backend")

    def _init_viserurdf_backend(self, description_name: Optional[str], open_browser: bool):
        """Initialize ViserUrdf backend."""
        try:
            import viser
            from viser.extras import ViserUrdf
            from robot_descriptions.loaders.yourdfpy import load_robot_description
        except ImportError as e:
            raise ImportError(
                f"ViserUrdf backend requires 'viser' and 'robot_descriptions' packages. "
                f"Install with: pip install viser robot_descriptions"
            ) from e

        # Some robot_descriptions assets require liblzfse (PyPI-only) for decompression.
        # In Pixi envs there may not be a conda-forge package, so users must install via pip.
        try:
            import liblzfse  # noqa: F401
        except Exception as e:
            raise ImportError(
                "ViserUrdf backend requires `liblzfse` for some robot assets.\n"
                "Install it via:\n"
                "  - pip install liblzfse\n"
                "  - (Pixi) pixi run pip install liblzfse\n"
                "Or install with examples dependencies:\n"
                "  pip install 'embodik[examples]'\n"
            ) from e

        # Get description name if not provided
        if description_name is None:
            # Try to infer from robot model or preset
            description_name = self._infer_description_name()

        if description_name is None:
            raise ValueError(
                "description_name is required for ViserUrdf backend. "
                "Either provide it explicitly or ensure robot_presets.yaml has 'description_name'."
            )

        # Create Viser server
        self.server = viser.ViserServer(port=self.port, host=self.host)
        self.scene = self.server.scene
        self.gui = self.server.gui

        # Load URDF with colors preserved
        try:
            urdf = load_robot_description(description_name)
            self._urdf_vis = ViserUrdf(self.server, urdf, root_node_name="/robot")

            # Set up joint configuration mapping (optimized for performance)
            actuated_names = list(getattr(self._urdf_vis._urdf, "actuated_joint_names", []))
            robot_joint_names = self.robot_model.get_joint_names()
            name_to_index = {name: idx for idx, name in enumerate(robot_joint_names)}

            # Pre-compute index mapping for faster lookups
            # Map from actuated_names index to robot_joint_names index
            actuated_to_robot_indices = []
            for joint_name in actuated_names:
                idx = name_to_index.get(joint_name)
                actuated_to_robot_indices.append(idx if idx is not None else -1)

            # Pre-allocate array for reuse (will resize if needed)
            self._cfg_vec_cache = np.zeros(len(actuated_names), dtype=float)

            def make_visual_config(q: np.ndarray) -> np.ndarray:
                """Convert embodik joint configuration to URDF visual configuration.

                Optimized version that reuses pre-allocated array and uses direct indexing.
                """
                if not actuated_names:
                    return q.copy() if q is not None else q

                # Reuse cached array if size matches, otherwise create new one
                if self._cfg_vec_cache is None or len(self._cfg_vec_cache) != len(actuated_names):
                    self._cfg_vec_cache = np.zeros(len(actuated_names), dtype=float)

                # Direct indexing instead of dictionary lookups (much faster)
                for i, robot_idx in enumerate(actuated_to_robot_indices):
                    if robot_idx >= 0 and robot_idx < q.size:
                        self._cfg_vec_cache[i] = q[robot_idx]
                    else:
                        self._cfg_vec_cache[i] = 0.0

                # Return view to avoid copy overhead (ViserUrdf.update_cfg copies internally)
                return self._cfg_vec_cache[:]

            self._make_visual_config = make_visual_config

            logger.info(f"Initialized ViserUrdf backend with description: {description_name}")

        except Exception as e:
            logger.warning(
                f"Failed to load robot description '{description_name}': {e}. "
                f"Falling back to Pinocchio backend."
            )
            # Fallback to Pinocchio backend
            self.backend = "pinocchio"
            self._init_pinocchio_backend(open_browser, None)
            return

        if open_browser:
            import webbrowser
            webbrowser.open(f"http://{self.host}:{self.port}")

    def _infer_description_name(self) -> Optional[str]:
        """Try to infer description_name from robot model or presets."""
        # Try to get from robot_presets.yaml
        try:
            from examples.utils.robot_models import load_robot_presets

            # Extract robot key from URDF path or other hints
            urdf_path = Path(self.robot_model.urdf_path)

            # Check common patterns in cache path
            if "panda_description" in str(urdf_path):
                return "panda_description"
            elif "iiwa14_description" in str(urdf_path) or "iiwa_description" in str(urdf_path):
                return "iiwa14_description"

            # Try to load from presets (this is a bit hacky but works)
            presets = load_robot_presets()
            for robot_key, preset in presets.items():
                if "urdf_path" in preset:
                    preset_path = Path(preset["urdf_path"])
                    if preset_path.name == urdf_path.name or preset_path.stem == urdf_path.stem:
                        return preset.get("description_name")
                elif "urdf_import" in preset:
                    # Extract description name from import path
                    urdf_import = preset.get("urdf_import", "")
                    if "panda_description" in urdf_import:
                        return "panda_description"
                    elif "iiwa14_description" in urdf_import:
                        return "iiwa14_description"
        except Exception:
            pass

        return None

    def display(self, q: np.ndarray) -> None:
        """Display robot configuration.

        Args:
            q: Joint configuration array
        """
        if self.backend == "pinocchio":
            if self._pinocchio_visualizer is not None:
                self._pinocchio_visualizer.display(q)
        elif self.backend == "viserurdf":
            if self._urdf_vis is not None and self._make_visual_config is not None:
                cfg = self._make_visual_config(q)
                self._urdf_vis.update_cfg(cfg)

    def update(self, q: np.ndarray) -> None:
        """Update robot visualization (alias for display).

        Args:
            q: Joint configuration array
        """
        self.display(q)

    def add_grid(self, name: str = "/ground", width: float = 2.0, height: float = 2.0) -> None:
        """Add grid to scene.

        Args:
            name: Grid name/path
            width: Grid width
            height: Grid height
        """
        self.scene.add_grid(name, width=width, height=height)

    @property
    def visualizer(self):
        """Get underlying visualizer object (for advanced usage).

        Returns:
            Pinocchio ViserVisualizer (if pinocchio backend) or ViserUrdf (if viserurdf backend)
        """
        if self.backend == "pinocchio":
            return self._pinocchio_visualizer
        elif self.backend == "viserurdf":
            return self._urdf_vis
        return None


def create_robot_visualizer(
    robot_model,
    backend: BackendType = "pinocchio",
    description_name: Optional[str] = None,
    port: int = 8080,
    open_browser: bool = True,
    host: str = "localhost",
    package_root: Optional[str] = None,
    load_collisions: bool = False,
) -> RobotVisualizer:
    """Create a robot visualizer with the specified backend.

    This is a convenience function that creates a RobotVisualizer instance.

    Args:
        robot_model: embodik RobotModel instance
        backend: Visualization backend ("pinocchio" or "viserurdf")
        description_name: Robot description name for ViserUrdf backend
        port: Port for Viser web server
        open_browser: Whether to automatically open browser
        host: Host for Viser server
        package_root: Optional package root directory for Pinocchio backend

    Returns:
        RobotVisualizer instance
    """
    return RobotVisualizer(
        robot_model=robot_model,
        backend=backend,
        description_name=description_name,
        port=port,
        open_browser=open_browser,
        host=host,
        package_root=package_root,
        load_collisions=load_collisions,
    )


"""Helper functions to access viser through Pinocchio's ViserVisualizer.

This module provides access to viser server and scene through Pinocchio's
ViserVisualizer, avoiding direct imports of viser and yourdfpy.

Key features:
- Preserves mesh colors from mesh files (like roboplan)
- Resolves package:// URIs automatically
- Handles geometry model loading with fallbacks
"""

from typing import Optional, Tuple, TYPE_CHECKING
from pathlib import Path
import logging

try:
    from ._runtime_deps import import_pinocchio as _import_pinocchio

    pin = _import_pinocchio()
    try:
        from pinocchio.visualize import ViserVisualizer
        _PINOCCHIO_VISER_AVAILABLE = True
    except ImportError:
        _PINOCCHIO_VISER_AVAILABLE = False
        ViserVisualizer = None
except ImportError:
    _PINOCCHIO_VISER_AVAILABLE = False
    pin = None
    ViserVisualizer = None

if TYPE_CHECKING:
    # Type hints for viser objects (avoid direct import)
    from typing import Any
    ViserServer = Any
    ViserScene = Any
    ViserGui = Any

logger = logging.getLogger(__name__)


def get_viser_server_from_visualizer(visualizer: ViserVisualizer) -> "ViserServer":
    """Get viser server from Pinocchio's ViserVisualizer.

    Args:
        visualizer: Pinocchio ViserVisualizer instance

    Returns:
        ViserServer instance (accessed through visualizer.viewer)
    """
    if not _PINOCCHIO_VISER_AVAILABLE:
        raise RuntimeError(
            "Pinocchio ViserVisualizer not available. "
            "Install pinocchio >= 3.8.0 to use this functionality."
        )

    if not hasattr(visualizer, 'viewer'):
        raise RuntimeError(
            "Visualizer has not been initialized. "
            "Call initViewer() before accessing the viewer."
        )

    return visualizer.viewer


def get_viser_scene_from_visualizer(visualizer: ViserVisualizer) -> "ViserScene":
    """Get viser scene from Pinocchio's ViserVisualizer.

    Args:
        visualizer: Pinocchio ViserVisualizer instance

    Returns:
        ViserScene instance (accessed through visualizer.viewer.scene)
    """
    server = get_viser_server_from_visualizer(visualizer)
    return server.scene


def get_viser_gui_from_visualizer(visualizer: ViserVisualizer) -> "ViserGui":
    """Get viser GUI from Pinocchio's ViserVisualizer.

    Args:
        visualizer: Pinocchio ViserVisualizer instance

    Returns:
        ViserGui instance (accessed through visualizer.viewer.gui)
    """
    server = get_viser_server_from_visualizer(visualizer)
    return server.gui


def create_visualizer_with_viser_access(
    robot_model,
    port: int = 8080,
    open_browser: bool = True,
    host: str = "localhost",
    *,
    load_collisions: bool = False,
) -> tuple[ViserVisualizer, "ViserServer", "ViserScene", "ViserGui"]:
    """Create Pinocchio ViserVisualizer and return viser access objects.

    This function creates a ViserVisualizer from a robot model and returns
    both the visualizer and direct access to viser server, scene, and GUI.

    Args:
        robot_model: embodik RobotModel instance
        port: Port for Viser web server
        open_browser: Whether to automatically open browser
        host: Host for Viser server

    Returns:
        Tuple of (visualizer, server, scene, gui)
    """
    if not _PINOCCHIO_VISER_AVAILABLE:
        raise RuntimeError(
            "Pinocchio ViserVisualizer not available. "
            "Install pinocchio >= 3.8.0 to use this functionality."
        )

    # Get Pinocchio model and data
    pin_model = robot_model._pinocchio_model
    pin_data = robot_model._pinocchio_data

    # Get geometry models
    visual_model = robot_model.visual_model
    collision_model = robot_model.collision_model if load_collisions else None

    # Create visualizer
    visualizer = ViserVisualizer(
        model=pin_model,
        collision_model=collision_model,
        visual_model=visual_model,
        data=pin_data,
        collision_data=robot_model.collision_data if load_collisions else None,
        visual_data=robot_model.visual_data
    )

    # Initialize viewer
    visualizer.initViewer(
        viewer=None,
        open=open_browser,
        loadModel=True,
        host=host,
        port=str(port)
    )

    # Get viser access objects
    server = visualizer.viewer
    scene = server.scene
    gui = server.gui

    return visualizer, server, scene, gui


def _find_package_root(urdf_path: str, description_name: Optional[str] = None) -> Optional[str]:
    """Find package root directory for resolving package:// URIs.

    This function resolves package:// URIs by looking for package directories
    relative to the URDF file location. It supports:
    1. Local robot_models/ directory structure (preferred)
    2. robot_descriptions cache (fallback for compatibility)

    Pinocchio's buildGeomFromUrdf resolves package:// URIs by looking for
    package directories in the provided package_dirs. For example, if the URDF
    contains `package://panda_description/meshes/...`, Pinocchio will look
    for `package_dirs[i]/panda_description/meshes/...` for each package_dir.

    Args:
        urdf_path: Path to URDF file
        description_name: Optional robot description name (e.g., "panda_description")

    Returns:
        Package root path string (directory containing package directories), or None if not found
    """
    urdf_dir = Path(urdf_path).parent

    # Strategy 1: Check if URDF is in robot_models/ directory structure
    # This is the preferred method for local robot models
    if "robot_models" in str(urdf_dir):
        parts = urdf_dir.parts
        try:
            robot_models_idx = parts.index("robot_models")
            robot_models_root = Path(*parts[:robot_models_idx + 1])  # .../robot_models

            # URDF files should use package://<robot_name>/... where <robot_name> matches
            # the directory name in robot_models/. For example:
            # - package://panda_description/meshes/... -> robot_models/panda_description/meshes/...
            # - package://iiwa14_description/meshes/... -> robot_models/iiwa14_description/meshes/...
            # Pinocchio will look for package_root/<robot_name>/... so we return robot_models_root
            if robot_models_root.exists():
                logger.debug(f"Found robot_models root: {robot_models_root}")
                return str(robot_models_root)
        except ValueError:
            pass

    # Strategy 2: Use URDF directory as package root (for relative paths)
    # This handles cases where meshes are in a subdirectory relative to URDF
    if urdf_dir.exists():
        logger.debug(f"Using URDF directory as package root: {urdf_dir}")
        return str(urdf_dir)

    # Strategy 3: Fallback to robot_descriptions cache (for compatibility)
    cache_dir = Path.home() / ".cache" / "robot_descriptions"
    possible_package_roots = [
        cache_dir / "example-robot-data" / "robots",  # Where packages like panda_description live
        cache_dir / "example-robot-data",
        cache_dir,
    ]

    # Add robot-specific root if description_name provided
    if description_name:
        possible_package_roots.insert(0, cache_dir / "example-robot-data" / "robots" / description_name)

    for possible_root in possible_package_roots:
        if possible_root.exists():
            logger.debug(f"Using robot_descriptions cache: {possible_root}")
            return str(possible_root)

    return None


def _load_geometry_models(
    pin_model: "pin.Model",
    urdf_path: str,
    robot_model=None,
    package_root: Optional[str] = None
) -> Tuple["pin.GeometryModel", "pin.GeometryModel", "pin.GeometryData", "pin.GeometryData"]:
    """Load visual and collision geometry models from URDF.

    Args:
        pin_model: Pinocchio model
        urdf_path: Path to URDF file
        robot_model: Optional embodik RobotModel (to use pre-loaded geometry)
        package_root: Optional package root directory for resolving package:// URIs

    Returns:
        Tuple of (visual_model, collision_model, visual_data, collision_data)
    """
    # Try to use geometry models already loaded in the robot model first
    if robot_model is not None:
        try:
            visual_model = robot_model.visual_model
            collision_model = robot_model.collision_model
            if visual_model is not None and collision_model is not None:
                logger.info(f"Using robot's pre-loaded geometry: {len(visual_model.geometryObjects)} visual objects and {len(collision_model.geometryObjects)} collision objects")
                return visual_model, collision_model, robot_model.visual_data, robot_model.collision_data
        except (TypeError, AttributeError):
            # If there's a binding issue or attribute doesn't exist, fall through to loading from URDF
            pass

    # Geometry models not loaded, try to load them
    logger.info("Geometry models not pre-loaded, attempting to load from URDF...")

    # Find package root if not provided
    if package_root is None:
        description_name = None
        if robot_model is not None:
            # Try to infer description name from URDF path
            urdf_name = Path(urdf_path).stem
            if "_description" in urdf_name:
                description_name = urdf_name
        package_root = _find_package_root(urdf_path, description_name)
        if package_root:
            logger.info(f"Found package root: {package_root}")

    package_dirs = [package_root] if package_root else []

    try:
        visual_model = pin.buildGeomFromUrdf(
            pin_model, urdf_path, pin.GeometryType.VISUAL,
            package_dirs=package_dirs
        )
        collision_model = pin.buildGeomFromUrdf(
            pin_model, urdf_path, pin.GeometryType.COLLISION,
            package_dirs=package_dirs
        )
        visual_data = pin.GeometryData(visual_model)
        collision_data = pin.GeometryData(collision_model)
        logger.info(f"✓ Loaded {len(visual_model.geometryObjects)} visual objects and {len(collision_model.geometryObjects)} collision objects")
        return visual_model, collision_model, visual_data, collision_data
    except Exception as e:
        # Fallback: try using robot model's geometry if available
        if robot_model is not None and robot_model.visual_model is not None:
            logger.info("Using robot model's geometry models (fallback)")
            return robot_model.visual_model, robot_model.collision_model, robot_model.visual_data, robot_model.collision_data

        # Last resort: empty models
        logger.warning(f"Failed to load geometry models: {e}. Robot will be displayed without visual meshes.")
        visual_model = pin.GeometryModel()
        collision_model = pin.GeometryModel()
        visual_data = pin.GeometryData(visual_model)
        collision_data = pin.GeometryData(collision_model)
        return visual_model, collision_model, visual_data, collision_data


def _patch_loadViewerGeometryObject_for_colors(visualizer: ViserVisualizer) -> None:
    """Monkey-patch loadViewerGeometryObject to preserve mesh colors (like roboplan).

    This ensures that when color=None, we use add_mesh_trimesh() which preserves
    colors from the mesh file, instead of add_mesh_simple() which applies a uniform color.

    Based on roboplan's implementation: https://github.com/stack-of-tasks/pinocchio/pull/2718

    Args:
        visualizer: Pinocchio ViserVisualizer instance to patch
    """
    original_loadViewerGeometryObject = visualizer.loadViewerGeometryObject

    def loadViewerGeometryObject_preserve_colors(self, geometry_object, prefix="", color=None):
        """Monkey-patched version that preserves mesh colors when color=None (like roboplan).

        Key difference from Pinocchio's default: when color=None for meshes, uses
        add_mesh_trimesh() which preserves colors from the mesh file, instead of
        add_mesh_simple() which applies a uniform color.
        """
        try:
            import trimesh
            try:
                import coal  # New name for hppfcl
            except ImportError:
                import hppfcl as coal  # Fallback to old name
        except ImportError:
            # Fallback to original if dependencies not available
            return original_loadViewerGeometryObject(geometry_object, prefix, color)

        name = geometry_object.name
        if prefix:
            name = prefix + "/" + name
        geom = geometry_object.geometry

        # For meshes, use add_mesh_trimesh when color is None to preserve mesh colors
        MESH_TYPES = (coal.BVHModelBase, coal.HeightFieldOBBRSS, coal.HeightFieldAABB)

        if isinstance(geom, MESH_TYPES):
            mesh = trimesh.load(geometry_object.meshPath)
            if color is None:
                # Use add_mesh_trimesh to preserve colors from mesh file (like roboplan)
                # This preserves vertex colors, textures, and materials from the mesh file
                frame = self.viewer.scene.add_mesh_trimesh(name, mesh)
            else:
                # When color is provided, use add_mesh_simple with color override
                color_override = color or getattr(geometry_object, 'meshColor', [0.7, 0.7, 0.7, 1.0])
                frame = self.viewer.scene.add_mesh_simple(
                    name,
                    mesh.vertices,
                    mesh.faces,
                    color=color_override[:3] if len(color_override) >= 3 else color_override,
                    opacity=color_override[3] if len(color_override) >= 4 else 1.0,
                )
        else:
            # For non-mesh geometries (Box, Sphere, Cylinder), use original method
            return original_loadViewerGeometryObject(geometry_object, prefix, color)

        # Store frame reference (matching roboplan's implementation)
        if hasattr(self, 'frames'):
            self.frames[name] = frame
        return frame

    # Apply monkey-patch
    import types
    visualizer.loadViewerGeometryObject = types.MethodType(loadViewerGeometryObject_preserve_colors, visualizer)


def create_viser_visualizer(
    robot_model,
    port: int = 8080,
    open_browser: bool = True,
    host: str = "localhost",
    preserve_mesh_colors: bool = True,
    package_root: Optional[str] = None,
    *,
    load_collisions: bool = False,
) -> Tuple[ViserVisualizer, "ViserServer", "ViserScene", "ViserGui"]:
    """Create Pinocchio ViserVisualizer with proper geometry loading and color preservation.

    This is a convenience function that handles all the complexity of:
    - Loading geometry models with package:// URI resolution
    - Monkey-patching to preserve mesh colors (like roboplan)
    - Initializing the viewer and loading the model

    Args:
        robot_model: embodik RobotModel instance
        port: Port for Viser web server
        open_browser: Whether to automatically open browser
        host: Host for Viser server
        preserve_mesh_colors: If True, monkey-patch to preserve mesh colors (default: True)
        package_root: Optional package root directory (auto-detected if None)

    Returns:
        Tuple of (visualizer, server, scene, gui)
    """
    if not _PINOCCHIO_VISER_AVAILABLE:
        raise RuntimeError(
            "Pinocchio ViserVisualizer not available. "
            "Install pinocchio >= 3.8.0 to use this functionality."
        )

    urdf_path = robot_model.urdf_path

    # Load URDF model for visualization
    pin_model = pin.buildModelFromUrdf(urdf_path)
    pin_data = pin_model.createData()

    # Load geometry models
    visual_model, collision_model, visual_data, collision_data = _load_geometry_models(
        pin_model, urdf_path, robot_model, package_root
    )

    # If not loading collisions, create empty models instead of None
    # (ViserVisualizer doesn't handle None well)
    if not load_collisions:
        collision_model = pin.GeometryModel()
        collision_data = pin.GeometryData(collision_model)

    # Create visualizer
    visualizer = ViserVisualizer(
        model=pin_model,
        collision_model=collision_model,
        visual_model=visual_model,
        data=pin_data,
        collision_data=collision_data,
        visual_data=visual_data
    )

    # Monkey-patch to preserve mesh colors (like roboplan)
    if preserve_mesh_colors:
        _patch_loadViewerGeometryObject_for_colors(visualizer)

    # Initialize viewer (loadModel=False so we can load with visual_color=None)
    visualizer.initViewer(
        viewer=None,
        open=open_browser,
        loadModel=False,  # Don't load model yet, we'll load it with visual_color=None
        host=host,
        port=str(port)
    )

    # Load model with visual_color=None to ensure mesh colors are preserved
    # The monkey-patched loadViewerGeometryObject will use add_mesh_trimesh when color=None
    model_loaded = False
    try:
        if hasattr(visualizer, 'loadViewerModel'):
            visualizer.loadViewerModel(visual_color=None, collision_color=None)
            model_loaded = True
            if preserve_mesh_colors:
                logger.info("Loaded model with visual_color=None to preserve mesh colors")
    except Exception as e:
        logger.warning(f"Could not load with visual_color=None: {e}. Falling back to default loading.")

    # Fallback: load without explicit color parameter
    if not model_loaded:
        try:
            if hasattr(visualizer, 'loadViewerModel'):
                visualizer.loadViewerModel()
                model_loaded = True
        except Exception as e2:
            logger.warning(f"Fallback loadViewerModel also failed: {e2}")

    # Get viser access objects
    server = visualizer.viewer
    scene = server.scene
    gui = server.gui

    return visualizer, server, scene, gui


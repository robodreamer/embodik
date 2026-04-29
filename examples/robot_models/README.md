# Robot Models

This directory contains robot model configurations that can be used with the embodiK examples.

Robot configurations are defined in `robot_presets.yaml`, which can be shared across multiple scripts.

## Quick Start

Robot models can be loaded from two sources:

1. **robot_descriptions package** (recommended): Automatically downloads and caches models
2. **Local files**: Place URDF files in `robot_models/` subdirectories

### Using robot_descriptions (Recommended)

The `robot_descriptions` package automatically downloads and caches robot models on first use. This is the default method used by the examples.

**Example configuration in `robot_presets.yaml`:**
```yaml
panda:
  description_name: panda_description
  urdf_import: robot_descriptions.panda_description
  urdf_attr: URDF_PATH
  target_link: panda_hand
  display_name: Franka Emika Panda
  default_configuration: [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
```

**Benefits:**
- No need to download or manage robot model files manually
- Models are automatically cached in `~/.cache/robot_descriptions/` after first download
- Keeps repository size small
- Models are versioned and maintained by the robot_descriptions community

**Installation:**
```bash
pip install robot_descriptions
# Or install with examples dependencies:
pip install embodik[examples]
```

### Using Local Files

For custom robot models or when you need to override robot_descriptions, you can use local files:

**Step 1: Place your robot model** in a subdirectory under `robot_models/`:
```
robot_models/
└── my_robot/
    ├── my_robot.urdf
    └── meshes/
        └── ...
```

**Step 2: Add an entry** to `robot_presets.yaml`:
```yaml
my_robot:
  description_name: my_robot_description
  urdf_path: robot_models/my_robot/my_robot.urdf  # Local file path
  target_link: "end_effector_link"
  display_name: "My Robot"
  default_configuration: [0.0, 0.0, 0.0, ...]
```

**Note:** If both `urdf_path` and `urdf_import` are specified, `urdf_path` (local file) takes priority.

## Using Robot Models in Examples

### Example 01 (Basic IK)

```python
from utils.robot_models import resolve_robot_configuration

config = resolve_robot_configuration("panda")
robot = config["robot"]
target_link = config["target_link"]
```

Or from command line:
```bash
python examples/01_basic_ik_simple.py
```

The command defaults to the Panda preset; add `--robot <key>` to select another
configured model.

### Example 02 (Collision-Aware IK)

```python
from utils.robot_models import load_robot_presets
from examples.example_helpers.reachability_robot_configs import resolve_robot_configuration

config = resolve_robot_configuration("panda")
robot = config.robot
target_link = config.target_link
```

## robot_presets.yaml Structure

Each robot entry in `robot_presets.yaml` can include:

### Required Fields

- **description_name**: Robot description name (e.g., "panda_description")
- **target_link**: Target link name for inverse kinematics
- **display_name**: Display name for the robot
- **default_configuration**: Default joint configuration (list of joint angles in radians)

### URDF Source (choose one)

- **urdf_import** + **urdf_attr**: Use robot_descriptions package
  ```yaml
  urdf_import: robot_descriptions.panda_description
  urdf_attr: URDF_PATH
  ```
- **urdf_path**: Use local file (relative to examples/ directory)
  ```yaml
  urdf_path: robot_models/my_robot/my_robot.urdf
  ```

### Optional Fields

- **joint_names**: List of joint names (auto-extracted from URDF if not specified)
- **joint_labels**: List of joint labels for display (auto-generated from joint names if not specified)
  - Labels default to the original joint names from the URDF (e.g., "panda_joint1", "iiwa_joint_1")
  - If not specified, joint names are used directly as labels
- **default_offset**: Default end-effector offset [x, y, z] (default: [0.05, 0.0, 0.0])
- **extra_gripper_default**: Extra gripper joint defaults (for robots with grippers, e.g., Panda)
- **collision_exclusions**: Collision exclusion pairs (use "auto" for automatic detection)
- **collision_exclusion_overrides**: Additional collision exclusions to add to auto-detected ones

## Adding a New Robot Model

### Method 1: Using robot_descriptions

1. **Check if robot is available** in robot_descriptions:
   ```python
   import robot_descriptions
   # Check available robots
   ```

2. **Add entry to `robot_presets.yaml`**:
   ```yaml
   my_robot:
     description_name: my_robot_description
     urdf_import: robot_descriptions.my_robot_description
     urdf_attr: URDF_PATH
     target_link: end_effector_link
     display_name: My Robot
     default_configuration: [0.0, 0.0, 0.0, ...]
   ```

3. **Test it**:
   ```bash
   python examples/01_basic_ik_simple.py --robot my_robot
   ```

### Method 2: Using Local Files

1. **Create directory structure**:
   ```bash
   mkdir -p examples/robot_models/my_robot/meshes
   ```

2. **Copy URDF and meshes**:
   ```bash
   # Copy URDF file
   cp my_robot.urdf examples/robot_models/my_robot/

   # Copy mesh files
   cp -r path/to/meshes/* examples/robot_models/my_robot/meshes/
   ```

3. **Configure URDF paths**:

   Your URDF can reference meshes in two ways:

   **Option A (Recommended): Relative paths**
   ```xml
   <mesh filename="meshes/link0.dae"/>
   ```

   **Option B: Package:// URIs**
   ```xml
   <mesh filename="package://my_robot/meshes/link0.dae"/>
   ```
   The package name (`my_robot`) should match your directory name.

4. **Add to robot_presets.yaml**:
   ```yaml
   my_robot:
     description_name: my_robot_description
     urdf_path: robot_models/my_robot/my_robot.urdf
     target_link: end_effector_link
     display_name: My Robot
     default_configuration: [0.0, 0.0, 0.0, ...]
   ```

5. **Test your robot**:
   ```bash
   python examples/01_basic_ik_simple.py --robot my_robot
   ```

## Current Robot Models

The following robot models are currently configured:

- **Franka Panda** (`panda`): Uses `robot_descriptions.panda_description`
- **KUKA LBR iiwa14** (`iiwa`): Uses `robot_descriptions.iiwa14_description`

Models are automatically downloaded and cached in `~/.cache/robot_descriptions/` on first use.

## Package:// URI Resolution

The helper functions automatically resolve `package://` URIs relative to the `robot_models/` directory. For example:

- `package://panda_description/meshes/visual/link0.dae` → `robot_models/panda_description/meshes/visual/link0.dae`
- `package://iiwa14_description/meshes/visual/link_0.obj` → `robot_models/iiwa14_description/meshes/visual/link_0.obj`

The package name in the URI should match the directory name in `robot_models/`. This is the preferred method as it works without symlinks and keeps the structure simple.

## Mesh File Organization

You can organize meshes in two ways:

1. **Simple structure**: All meshes in `meshes/` directory
   ```
   robot_models/my_robot/
   ├── my_robot.urdf
   └── meshes/
       ├── link0.dae
       └── link1.dae
   ```

2. **Separate visual/collision** (like roboplan):
   ```
   robot_models/my_robot/
   ├── my_robot.urdf
   └── meshes/
       ├── visual/
       │   └── link0.dae
       └── collision/
           └── link0.stl
   ```

Both structures are supported. The URDF should reference the correct paths.

## Notes

- Mesh files can be in any format supported by Pinocchio (`.dae`, `.stl`, `.obj`, etc.)
- The URDF file can reference meshes using either relative paths or `package://` URIs
- All paths are resolved relative to the URDF file's directory or the `robot_models/` root
- Colors from mesh files are automatically preserved (see `viser_helpers.py` for details)
- robot_descriptions models are cached automatically - no manual download needed
- To clear robot_descriptions cache: `rm -rf ~/.cache/robot_descriptions/`

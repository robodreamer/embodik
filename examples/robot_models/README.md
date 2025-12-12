# Robot Models

This directory contains robot model files (URDF + meshes) that can be used with the embodiK examples.

Robot configurations are defined in `robot_presets.yaml`, which can be shared across multiple scripts.

## Quick Start

1. **Place your robot model** in a subdirectory under `robot_models/`:
   ```
   robot_models/
   └── my_robot/
       ├── my_robot.urdf
       └── meshes/
           └── ...
   ```

2. **Add an entry** to `robot_presets.yaml`:
   ```yaml
   my_robot:
     urdf_path: "robot_models/my_robot/my_robot.urdf"
     target_link: "end_effector_link"
     display_name: "My Robot"
     default_configuration: [0.0, 0.0, 0.0, ...]
   ```

3. **Use it in examples**:
   ```python
   from utils.robot_models import resolve_robot_configuration

   config = resolve_robot_configuration("my_robot")
   robot = config["robot"]
   target_link = config["target_link"]
   ```

   Or from command line:
   ```bash
   python examples/01_basic_ik_simple.py --robot my_robot
   ```

## Directory Structure

Each robot model should be placed in its own subdirectory:

```
robot_models/
├── franka_panda/
│   ├── frankaEmikaPanda.urdf    # URDF file
│   └── meshes/                  # Mesh files referenced by URDF
│       ├── visual/              # Optional: separate visual meshes
│       │   ├── link0.dae
│       │   └── ...
│       └── collision/           # Optional: separate collision meshes
│           ├── link0.stl
│           └── ...
├── LBR_iiwa_14/
│   ├── lbr_iiwa_14_r820.urdf    # URDF file
│   └── meshes/                  # Mesh files referenced by URDF
│       └── ...
├── robot_presets.yaml           # Robot configuration presets (shared across scripts)
├── __init__.py                  # Helper functions for loading presets
└── README.md                    # This file
```

## Adding a New Robot Model

### Step 1: Create Directory Structure

```bash
mkdir -p examples/robot_models/my_robot/meshes
```

### Step 2: Copy URDF and Meshes

```bash
# Copy URDF file
cp my_robot.urdf examples/robot_models/my_robot/

# Copy mesh files
cp -r path/to/meshes/* examples/robot_models/my_robot/meshes/
```

### Step 3: Configure URDF Paths

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

### Step 4: Add to robot_presets.yaml

Add an entry to `robot_presets.yaml`:

```yaml
my_robot:
  # Required: Path to URDF file (relative to examples/ directory)
  urdf_path: "robot_models/my_robot/my_robot.urdf"

  # Required: Target link name for inverse kinematics
  target_link: "end_effector_link"

  # Required: Display name for the robot
  display_name: "My Robot"

  # Required: Default joint configuration (list of joint angles in radians)
  default_configuration: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

  # Optional: Extra gripper joints (only for robots with grippers, e.g., Panda)
  extra_gripper_default: [0.05, 0.05]
```

**Notes:**
- `urdf_path` is relative to the `examples/` directory
- `default_configuration` should be a YAML list (automatically converted to numpy array)
- Joint labels are automatically extracted from the URDF model (no need to specify them)
- `extra_gripper_default` is only needed for robots with gripper joints (like Panda with 9 DOF)

### Step 5: Test Your Robot

```bash
# Test loading
python examples/01_basic_ik_simple.py --robot my_robot
```

## Current Robot Models

The following robot models are currently available:

- **Franka Panda** (`panda_description/`): `panda.urdf`
- **LBR iiwa14** (`iiwa14_description/`): `iiwa14_no_collision.urdf`

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


# embodiK Examples

This directory contains examples demonstrating the embodiK library capabilities.

## Prerequisites

Make sure you have built embodiK first:

```bash
cd ..
./build.sh
```

Install required dependencies:

```bash
pip install viser robot-descriptions yourdfpy
```

## Examples

### 01_basic_ik_simple.py

A simplified basic IK example using velocity-based control with spatialmath for pose operations.

**Features:**
- Clean velocity IK implementation
- Uses spatialmath SE3 for pose error computation
- Interactive target control with Viser
- Support for Panda and IIWA robots
- Position and rotation gain tuning

**Usage:**
```bash
cd embodik
python examples/01_basic_ik_simple.py
```

## 01_basic_ik_viser.py

Interactive inverse kinematics example with real-time Viser visualization.

**Features:**
- Interactive target manipulation using Viser's transform controls
- Real-time IK solving with position and orientation control
- Nullspace bias control (bias towards default or zero configuration)
- Configurable solver parameters (iterations, tolerance, timestep)
- Support for both Franka Panda and KUKA IIWA robots
- Visual feedback showing solve time, position error, and iterations

**Running:**
```bash
# Run with Franka Panda robot (default)
python 01_basic_ik_viser.py

# Run with KUKA IIWA robot
python 01_basic_ik_viser.py iiwa
```

### Key Concepts

- **Task Priorities**: End-effector tracking (priority 0) vs posture regularization (priority 10)
- **Velocity Integration**: Converts velocity commands to position updates
- **Constraint Handling**: Clips velocities and positions to robot limits
- **Trajectory Generation**: Creates smooth infinity sign pattern in 3D space

The robot will continuously track the moving target while trying to stay close to its home configuration when possible.

Press Ctrl+C to stop the trajectory tracking.

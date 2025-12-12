# API Reference

Complete API documentation for EmbodiK.

## Core Classes

### RobotModel

The `RobotModel` class represents a robot kinematic model loaded from URDF.

::: embodik.RobotModel
    options:
      show_root_heading: true
      show_root_toc_entry: true

### KinematicsSolver

The `KinematicsSolver` provides inverse kinematics solving capabilities.

::: embodik.KinematicsSolver
    options:
      show_root_heading: true
      show_root_toc_entry: true

## Task Types

EmbodiK supports various task types for multi-task IK:

- **FrameTask**: Control end-effector pose (position + orientation)
- **PostureTask**: Maintain desired joint configuration
- **COMTask**: Control center of mass position
- **JointTask**: Control individual joint positions
- **MultiJointTask**: Control multiple joints simultaneously

See the [Tasks](tasks.md) page for detailed documentation.

## Utilities

::: embodik.utils
    options:
      show_root_heading: true
      show_root_toc_entry: true

## Visualization

Optional visualization tools are documented on the [Visualization](visualization.md) page.

## Enumerations

### SolverStatus

Status codes returned by IK solvers:

- `SUCCESS`: Solver converged successfully
- `MAX_ITERATIONS`: Maximum iterations reached
- `INVALID_INPUT`: Invalid input parameters
- `SINGULARITY`: Singular configuration encountered
- `CONSTRAINT_VIOLATION`: Joint limits or constraints violated

## Result Types

### PositionIKResult

Result from position IK solving:

- `solution`: Final joint configuration (numpy array)
- `status`: Solver status code
- `iterations`: Number of iterations performed
- `final_error`: Final pose error magnitude
- `computation_time_ms`: Computation time in milliseconds

### VelocitySolverResult

Result from velocity IK solving:

- `solution`: Joint velocities (numpy array)
- `status`: Solver status code
- `task_scales`: Task scaling factors for multi-task problems
- `computation_time_ms`: Computation time in milliseconds

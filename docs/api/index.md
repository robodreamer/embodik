# API Reference

Complete API documentation for EmbodiK. For conceptual guides (robustness, collision, GPU),
start with the [Guides overview](../guides/index.md).

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

### AccelerationSolver

The opt-in fixed-base acceleration-level API, its constraints, diagnostics, and
explicit unsupported scope are documented in the
[Acceleration Solver guide and API](../acceleration_solver.md).

## Task Types

EmbodiK supports various task types for multi-task IK:

- **FrameTask**: Control end-effector pose (position + orientation)
- **PostureTask**: Maintain desired joint configuration
- **COMTask**: Control center of mass position
- **JointTask**: Control individual joint positions
- **MultiJointTask**: Control multiple joints simultaneously

See the [Tasks](tasks.md) page for detailed documentation.

## Transforms

Native spatial transform helpers (no SciPy dependency):

- **Rotation** / **SO3**: SO(3) rotations with spatialmath-style shorthands (`Rx`, `Ry`, `Rz`, `RPY`, etc.)
- **SE3**: Rigid-body transforms with composition (`T1 * T2`), point transforms (`act`, `actInv`), and property aliases (`R`, `t`, `A`)

See the [Transforms](transforms.md) page for full API.

## Utilities

::: embodik.utils
    options:
      show_root_heading: true
      show_root_toc_entry: true

## Visualization

Optional visualization tools are documented on the [Visualization](visualization.md) page.

## GPU Solvers

For model-derived, GPU-accelerated WBC, see the
[GPU Solvers](../gpu_solvers.md) documentation. New integrations should use
the `embodik.gpu.wbc` package:

- `GpuWbcMultiFrameSolver` — fixed-base, device-resident multi-frame velocity WBC
- `GpuWbcFloatingMultiFrameSolver` — standard floating-base velocity WBC
- `GpuAccelerationSolver` — the narrower fixed-base acceleration state-box slice

The older FI-PeSNS and PPH-SNS CasADi builders remain import-compatible, but
they require generated shape-specific artifacts and are not the recommended
entry point for new applications.

## Enumerations

### SolverStatus

Status codes returned by IK solvers:

- `SUCCESS`: Solver converged successfully
- `MAX_ITERATIONS`: Maximum iterations reached
- `INVALID_INPUT`: Invalid input parameters
- `SINGULARITY`: Singular configuration encountered
- `CONSTRAINT_VIOLATION`: Joint limits or constraints violated

## Result Types

All solver result types expose shared diagnostics:

- `status`: Solver status code
- `computation_time_ms`: Computation time in milliseconds
- `task_scales`: Task scaling factors for multi-task problems, when applicable
- `task_errors`: Per-task error magnitudes, when applicable
- `status_message`: Optional solver status detail
- `condition_number`: Worst Jacobian condition number observed during the solve.
  Values near `1.0` are well-conditioned and larger values indicate increasing
  sensitivity near singular or rank-deficient configurations. This is a
  diagnostic field for logging and tuning; it is not a manipulability score and
  does not by itself change solver behavior.

### PositionIKResult

Result from position IK solving:

- `solution`: Final joint configuration (numpy array)
- `status`: Solver status code
- `iterations`: Number of iterations performed
- `final_error`: Final pose error magnitude
- `computation_time_ms`: Computation time in milliseconds
- `condition_number`: Worst Jacobian condition number observed during the solve

### VelocitySolverResult

Result from velocity IK solving:

- `solution`: Joint velocities (numpy array)
- `status`: Solver status code
- `task_scales`: Task scaling factors for multi-task problems
- `computation_time_ms`: Computation time in milliseconds
- `condition_number`: Worst Jacobian condition number observed during the solve

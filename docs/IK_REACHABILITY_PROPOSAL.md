# IK-Based Reachability Analysis Proposal

## Overview

This document proposes adding IK-based reachability analysis to complement the existing FK-based (PyTorch sampling) approach in `03_reachability_analysis.py`. The IK-based approach mirrors the MATLAB implementation in `GUI_example_iisy_advanced.m`.

## Key Differences: FK vs IK Approach

### Current FK-Based Approach (Forward Sampling)
- **Direction**: Joint space → Cartesian space
- **Method**: Sample random joint configurations → Compute FK → Bin results
- **Pros**: Fast, covers entire workspace uniformly
- **Cons**: May miss hard-to-reach poses, orientation coverage depends on sampling density

### Proposed IK-Based Approach (Inverse Sampling)
- **Direction**: Cartesian space → Joint space
- **Method**: Generate Cartesian grid → Solve IK for each point → Mark reachable/unreachable
- **Pros**: Guaranteed coverage of specific Cartesian regions, explicit reachability test
- **Cons**: Slower (IK solving per point), requires orientation specification

## MATLAB Reference Analysis

From `testReachability` function (lines 3358-3625):

1. **Grid Generation**:
   - Creates 3D Cartesian grid around base origin
   - Configurable step size (`grid_step_size`)
   - Filters points within robot reach radius
   - Sorts by distance from initial pose

2. **IK Solving**:
   - For each grid point:
     - Sets target position (orientation from reference pose)
     - Solves IK with nullspace bias options
     - Marks as valid/invalid based on solution success
   - Uses `SnsPosIkSolverConstraints` with configurable:
     - Nullspace bias pose (elbow up/down, pose optimization, none)
     - Reference frame (base/TCP/custom)
     - Relaxed constraints for released axes

3. **Metrics Computation**:
   - Computes pose metrics for valid solutions
   - Stores joint configurations for valid points
   - Sorts by metric values

4. **Visualization**:
   - Colored scatter plot (valid=colored by metric, invalid=gray)
   - Can display traces/waypoints

## Proposed Implementation

### 1. New Function: `generate_ik_reachability_map`

```python
@dataclass
class IKReachabilityConfig:
    """Configuration for IK-based reachability analysis."""
    # Grid parameters
    grid_step_size: float = 0.01  # meters (matches MATLAB default 10mm)
    x_limits: Tuple[float, float] = (-0.8, 0.8)
    y_limits: Tuple[float, float] = (-0.8, 0.8)
    z_limits: Tuple[float, float] = (0.0, 1.3)

    # Orientation handling
    orientation_mode: str = "fixed"  # "fixed", "reference", "multiple"
    reference_orientation: Optional[np.ndarray] = None  # 3x3 rotation matrix
    orientation_samples: int = 1  # For "multiple" mode

    # IK solver options
    position_tolerance: float = 1e-3
    orientation_tolerance: float = 1e-3
    max_iterations: int = 100
    dt: float = 0.01

    # Nullspace control
    use_nullspace_bias: bool = True
    nullspace_bias: Optional[np.ndarray] = None  # Joint configuration
    nullspace_gain: float = 0.1
    nullspace_active_joints: Optional[List[int]] = None

    # Step size limits
    max_linear_step: float = 0.1
    max_angular_step: float = 0.1

    # Output format
    cartesian_resolution: float = 0.05  # For binning (matches FK approach)
    angular_resolution: float = np.pi / 6.0  # For binning


def generate_ik_reachability_map(
    robot_cfg: RobotConfig,
    config: IKReachabilityConfig,
    *,
    chain_joint_names: Sequence[str],
    output_dir: Path,
    end_effector: str,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> Dict[str, Path]:
    """Generate reachability map using IK-based analysis.

    This function:
    1. Generates a 3D Cartesian grid of target positions
    2. For each grid point, solves IK to test reachability
    3. Computes metrics for valid solutions
    4. Outputs results in same format as FK-based approach

    Args:
        robot_cfg: Robot configuration
        config: IK reachability configuration
        chain_joint_names: Joint names for the kinematic chain
        output_dir: Directory for output files
        end_effector: End-effector frame name
        progress_callback: Optional progress callback (progress, stage)
        cancel_event: Optional cancellation event

    Returns:
        Dictionary with paths to generated files (same keys as FK approach)
    """
```

### 2. Implementation Steps

#### Step 1: Grid Generation
```python
def _generate_cartesian_grid(
    config: IKReachabilityConfig,
    reference_position: np.ndarray,
    reach_radius: float,
) -> np.ndarray:
    """Generate 3D Cartesian grid points.

    Matches MATLAB approach:
    - Creates grid around base origin
    - Filters by reach radius
    - Sorts by distance from reference position
    """
    # Generate grid points
    x_points = np.arange(config.x_limits[0], config.x_limits[1] + config.grid_step_size,
                        config.grid_step_size)
    y_points = np.arange(config.y_limits[0], config.y_limits[1] + config.grid_step_size,
                        config.grid_step_size)
    z_points = np.arange(config.z_limits[0], config.z_limits[1] + config.grid_step_size,
                        config.grid_step_size)

    # Cartesian product
    grid_points = np.array(np.meshgrid(x_points, y_points, z_points)).T.reshape(-1, 3)

    # Filter by reach radius
    distances = np.linalg.norm(grid_points - reference_position, axis=1)
    valid_mask = distances <= reach_radius
    grid_points = grid_points[valid_mask]

    # Sort by distance from reference
    distances = np.linalg.norm(grid_points - reference_position, axis=1)
    sort_indices = np.argsort(distances)
    grid_points = grid_points[sort_indices]

    return grid_points
```

#### Step 2: Orientation Handling
```python
def _get_target_orientations(
    config: IKReachabilityConfig,
    reference_pose: SE3,
) -> List[np.ndarray]:
    """Get target orientations based on mode.

    Modes:
    - "fixed": Use reference orientation for all points
    - "reference": Use orientation from reference pose
    - "multiple": Sample multiple orientations per position
    """
    if config.orientation_mode == "fixed":
        if config.reference_orientation is not None:
            return [config.reference_orientation]
        return [reference_pose.rotation]
    elif config.orientation_mode == "reference":
        return [reference_pose.rotation]
    elif config.orientation_mode == "multiple":
        # Sample orientations (e.g., identity + rotations around axes)
        orientations = [np.eye(3)]  # Identity
        # Add more samples as needed
        return orientations[:config.orientation_samples]
    else:
        raise ValueError(f"Unknown orientation_mode: {config.orientation_mode}")
```

#### Step 3: IK Solving Loop
```python
def _solve_ik_for_grid_point(
    solver: embodik.KinematicsSolver,
    seed_q: np.ndarray,
    target_position: np.ndarray,
    target_orientation: np.ndarray,
    config: IKReachabilityConfig,
    frame_name: str,
) -> Optional[embodik.PositionIKResult]:
    """Solve IK for a single grid point."""
    # Build target pose matrix
    target_pose = np.eye(4)
    target_pose[:3, :3] = target_orientation
    target_pose[:3, 3] = target_position

    # Configure IK options
    options = embodik.PositionIKOptions()
    options.position_tolerance = config.position_tolerance
    options.orientation_tolerance = config.orientation_tolerance
    options.max_iterations = config.max_iterations
    options.dt = config.dt
    options.max_linear_step = config.max_linear_step
    options.max_angular_step = config.max_angular_step

    if config.use_nullspace_bias and config.nullspace_bias is not None:
        options.nullspace_bias = config.nullspace_bias
        options.nullspace_gain = config.nullspace_gain
        if config.nullspace_active_joints is not None:
            options.nullspace_active_joints = config.nullspace_active_joints

    # Solve IK
    result = solver.solve_position(seed_q, target_pose, frame_name, options)

    if result.status == embodik.SolverStatus.SUCCESS:
        return result
    return None
```

#### Step 4: Metrics Computation
```python
def _compute_metrics_for_solution(
    robot: embodik.RobotModel,
    q_solution: np.ndarray,
    metric_helper: ReachabilityMetricHelper,
    metric_ranges: Dict[str, Tuple[float, float]],
) -> Dict[str, float]:
    """Compute pose metrics for IK solution."""
    robot.update_configuration(q_solution)
    metrics = metric_helper.compute_metrics(q_solution, metric_ranges)
    return metrics
```

#### Step 5: Output Format Conversion
```python
def _convert_ik_results_to_fk_format(
    valid_points: np.ndarray,
    valid_orientations: List[np.ndarray],
    valid_configs: np.ndarray,
    valid_metrics: List[Dict[str, float]],
    config: IKReachabilityConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert IK results to FK-compatible format.

    Returns:
        - positions: (N, 3) array
        - orientations_rpy: (N, 3) array
        - orientations_quat: (N, 4) array
        - configs: (N, num_joints) array
        - metrics: List of metric dicts
    """
    # Convert orientations to RPY and quaternion
    # Bin into same grid as FK approach
    # Return in same format
```

### 3. GUI Integration

Add to `create_reachability_gui_panel`:

```python
# Analysis mode selection
analysis_mode = server.gui.add_dropdown(
    "Analysis Mode",
    options=["FK Sampling", "IK Grid"],
    initial_value="FK Sampling"
)

# IK-specific controls (shown when IK mode selected)
ik_grid_step_slider = server.gui.add_slider(
    "Grid step size (mm)",
    min=5.0,
    max=50.0,
    initial_value=10.0,
    step=1.0,
    disabled=True  # Disabled when FK mode
)

ik_orientation_mode = server.gui.add_dropdown(
    "Orientation mode",
    options=["Fixed", "Reference", "Multiple"],
    initial_value="Fixed",
    disabled=True
)

ik_nullspace_bias_checkbox = server.gui.add_checkbox(
    "Use nullspace bias",
    initial_value=True,
    disabled=True
)
```

### 4. Workflow Integration

Modify `_launch_reachability_job` to support both modes:

```python
def _launch_reachability_job(_event=None) -> None:
    # ... existing code ...

    if analysis_mode.value == "FK Sampling":
        outputs = generate_reachability_map(...)  # Existing FK function
    else:  # IK Grid
        ik_config = IKReachabilityConfig(
            grid_step_size=ik_grid_step_slider.value / 1000.0,  # mm to m
            orientation_mode=ik_orientation_mode.value.lower(),
            # ... other config ...
        )
        outputs = generate_ik_reachability_map(
            robot_cfg=cfg,
            config=ik_config,
            chain_joint_names=cfg.joint_names,
            output_dir=output_dir,
            end_effector=cfg.target_link,
            progress_callback=_progress_callback,
            cancel_event=cancel_event,
        )
```

## Output Format Compatibility

The IK-based approach should produce the same output files as FK-based:

1. **`reach_map_{job_name}.pkl`**: Same format (6D binned data)
2. **`3D_{job_name}.h5`**: Same HDF5 structure with spheres dataset
3. **`samples_{job_name}.npz`**: Same NPZ format with xyz, rpy, quat, q, joint_names

This ensures the existing `ReachabilityMapViewer` can visualize both types of maps.

## Advantages of IK-Based Approach

1. **Guaranteed Coverage**: Tests specific Cartesian regions explicitly
2. **Reachability Validation**: Direct test of whether poses are reachable
3. **Orientation Control**: Can test specific orientations of interest
4. **Complementary to FK**: FK finds what's reachable, IK tests what should be reachable

## Implementation Priority

1. **Phase 1**: Basic IK grid generation and solving (position-only, fixed orientation)
2. **Phase 2**: Orientation handling (reference, multiple samples)
3. **Phase 3**: Nullspace bias integration
4. **Phase 4**: GUI integration and mode switching
5. **Phase 5**: Performance optimization (parallel IK solving)

## Testing Strategy

1. Compare IK results with FK results for same robot
2. Validate against MATLAB reference implementation
3. Test with different robots (Panda, iiwa)
4. Performance benchmarking (IK vs FK speed)

## Questions for User

1. **Orientation handling**: Should we support multiple orientations per position, or keep it simple with fixed/reference orientation?
2. **Grid density**: What default step size? (MATLAB uses 10mm)
3. **Nullspace bias**: Should we provide GUI controls for bias pose selection (elbow up/down, etc.)?
4. **Performance**: Accept slower analysis for guaranteed coverage, or optimize with parallel solving?

---

## Global Score Computation

### Overview

Both FK and IK approaches should compute global scores based on selected metrics. The MATLAB implementation computes a "Global Manipulability" score as the average metric value across all valid points.

### Scoring Methods

#### 1. Average Metric Score (MATLAB-style)
```python
def compute_average_metric_score(
    metric_values: np.ndarray,
    metric_name: str = "Manipulability",
) -> Dict[str, float]:
    """Compute average metric score across all valid points.

    Matches MATLAB: GM = sum(metrics) / count
    Ideal value: 1.0 (all points reachable with max metric)

    Returns:
        {
            "average": float,  # Average metric value
            "sum": float,       # Sum of all metric values
            "count": int,       # Number of valid points
            "min": float,       # Minimum metric value
            "max": float,       # Maximum metric value
            "std": float,       # Standard deviation
        }
    """
    if len(metric_values) == 0:
        return {
            "average": 0.0,
            "sum": 0.0,
            "count": 0,
            "min": 0.0,
            "max": 0.0,
            "std": 0.0,
        }

    return {
        "average": float(np.mean(metric_values)),
        "sum": float(np.sum(metric_values)),
        "count": int(len(metric_values)),
        "min": float(np.min(metric_values)),
        "max": float(np.max(metric_values)),
        "std": float(np.std(metric_values)),
    }
```

#### 2. Weighted Average Score (FK-specific)
```python
def compute_weighted_metric_score(
    metric_values: np.ndarray,
    weights: np.ndarray,  # Visitation counts or importance weights
    metric_name: str = "Manipulability",
) -> Dict[str, float]:
    """Compute visitation-weighted average metric score.

    For FK approach: weights = visitation counts
    Gives more weight to frequently visited regions.

    Returns:
        Same structure as average_metric_score, plus:
        - "weighted_average": float
        - "total_weight": float
    """
    if len(metric_values) == 0 or len(weights) == 0:
        return compute_average_metric_score(metric_values, metric_name)

    total_weight = float(np.sum(weights))
    if total_weight <= 0.0:
        return compute_average_metric_score(metric_values, metric_name)

    weighted_sum = float(np.sum(metric_values * weights))
    weighted_avg = weighted_sum / total_weight

    result = compute_average_metric_score(metric_values, metric_name)
    result["weighted_average"] = weighted_avg
    result["total_weight"] = total_weight
    return result
```

#### 3. Coverage Score (IK-specific)
```python
def compute_coverage_score(
    valid_count: int,
    total_count: int,
    metric_values: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute workspace coverage score.

    For IK approach: what percentage of grid points are reachable.

    Returns:
        {
            "coverage_ratio": float,      # valid_count / total_count
            "valid_count": int,
            "invalid_count": int,
            "total_count": int,
            "metric_average": float,      # Average metric for valid points (if provided)
        }
    """
    if total_count == 0:
        return {
            "coverage_ratio": 0.0,
            "valid_count": 0,
            "invalid_count": 0,
            "total_count": 0,
            "metric_average": 0.0,
        }

    coverage = float(valid_count) / float(total_count)
    metric_avg = float(np.mean(metric_values)) if metric_values is not None and len(metric_values) > 0 else 0.0

    return {
        "coverage_ratio": coverage,
        "valid_count": valid_count,
        "invalid_count": total_count - valid_count,
        "total_count": total_count,
        "metric_average": metric_avg,
    }
```

#### 4. Composite Score (Combining Multiple Metrics)
```python
def compute_composite_score(
    metrics_dict: Dict[str, np.ndarray],
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Compute composite score from multiple metrics.

    Args:
        metrics_dict: Dictionary mapping metric names to arrays of values
        weights: Optional weights for each metric (default: equal weights)

    Returns:
        {
            "composite_average": float,
            "individual_scores": Dict[str, float],  # Per-metric averages
            "weights": Dict[str, float],            # Applied weights
        }
    """
    if not metrics_dict:
        return {
            "composite_average": 0.0,
            "individual_scores": {},
            "weights": {},
        }

    if weights is None:
        weights = {name: 1.0 / len(metrics_dict) for name in metrics_dict.keys()}

    individual_scores = {}
    weighted_sum = 0.0
    total_weight = 0.0

    for metric_name, metric_values in metrics_dict.items():
        avg = float(np.mean(metric_values)) if len(metric_values) > 0 else 0.0
        individual_scores[metric_name] = avg
        weight = weights.get(metric_name, 0.0)
        weighted_sum += avg * weight
        total_weight += weight

    composite = weighted_sum / total_weight if total_weight > 0.0 else 0.0

    return {
        "composite_average": composite,
        "individual_scores": individual_scores,
        "weights": weights,
    }
```

### Integration into FK Approach

```python
def compute_fk_global_scores(
    reach_map: torch.Tensor,
    config: ReachabilityConfig,
    metric_column_indices: Dict[str, int],
) -> Dict[str, Dict[str, float]]:
    """Compute global scores from FK reachability map.

    Args:
        reach_map: Binned reachability map (num_voxels, 10)
        config: Reachability configuration
        metric_column_indices: Mapping of metric names to column indices

    Returns:
        Dictionary mapping metric names to score dictionaries
    """
    # Extract metric columns
    metrics_dict = {}
    visitation_col = 6  # VISITATION_COL

    for metric_name, col_idx in metric_column_indices.items():
        metric_values = reach_map[:, col_idx].cpu().numpy()
        visitation = reach_map[:, visitation_col].cpu().numpy()

        # Only consider visited voxels
        visited_mask = visitation > 0
        if np.any(visited_mask):
            visited_metrics = metric_values[visited_mask]
            visited_weights = visitation[visited_mask]

            # Compute both simple and weighted averages
            simple_score = compute_average_metric_score(visited_metrics, metric_name)
            weighted_score = compute_weighted_metric_score(visited_metrics, visited_weights, metric_name)

            metrics_dict[metric_name] = {
                **simple_score,
                **weighted_score,
            }
        else:
            metrics_dict[metric_name] = compute_average_metric_score(np.array([]), metric_name)

    return metrics_dict
```

### Integration into IK Approach

```python
def compute_ik_global_scores(
    valid_points: np.ndarray,
    invalid_points: np.ndarray,
    valid_metrics: Dict[str, np.ndarray],
) -> Dict[str, Dict[str, float]]:
    """Compute global scores from IK reachability analysis.

    Args:
        valid_points: Array of valid (reachable) points
        invalid_points: Array of invalid (unreachable) points
        valid_metrics: Dictionary mapping metric names to arrays of values for valid points

    Returns:
        Dictionary with coverage scores and metric scores
    """
    total_count = len(valid_points) + len(invalid_points)
    valid_count = len(valid_points)

    # Compute coverage score
    coverage_score = compute_coverage_score(valid_count, total_count)

    # Compute metric scores for valid points
    metric_scores = {}
    for metric_name, metric_values in valid_metrics.items():
        metric_scores[metric_name] = compute_average_metric_score(metric_values, metric_name)
        # Add coverage-weighted metric (penalize unreachable regions)
        coverage_weighted = metric_scores[metric_name]["average"] * coverage_score["coverage_ratio"]
        metric_scores[metric_name]["coverage_weighted_average"] = coverage_weighted

    return {
        "coverage": coverage_score,
        "metrics": metric_scores,
    }
```

### Output Format Integration

#### HDF5 Attributes
```python
# Add to sphere_dataset attributes
dataset.attrs.create("GlobalScores", json.dumps({
    "Manipulability": {
        "average": 0.75,
        "weighted_average": 0.78,
        "count": 1250,
        "min": 0.12,
        "max": 0.98,
    },
    "RangeOfMotion": {...},
    "SingularityAvoidance": {...},
    "coverage": {  # For IK approach
        "coverage_ratio": 0.85,
        "valid_count": 850,
        "total_count": 1000,
    },
}))
```

#### Separate JSON Summary File
```python
# Create summary JSON file
summary_path = output_dir / f"summary_{job_name}.json"
with open(summary_path, "w") as f:
    json.dump({
        "job_name": job_name,
        "analysis_mode": "FK" or "IK",
        "total_samples": total_samples,
        "global_scores": global_scores,
        "config": {
            "cartesian_resolution": config.cartesian_resolution,
            "angular_resolution": config.angular_resolution,
            # ... other config params
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, f, indent=2)
```

### GUI Display

```python
# Add to reachability GUI panel
with server.gui.add_folder("Global Scores", expand_by_default=True):
    global_scores_text = server.gui.add_text(
        "Global Scores",
        initial_value="No scores computed yet"
    )

    selected_metric_dropdown = server.gui.add_dropdown(
        "Score Metric",
        options=["Manipulability", "RangeOfMotion", "SingularityAvoidance", "Composite"],
        initial_value="Manipulability"
    )

    score_display_mode = server.gui.add_dropdown(
        "Display Mode",
        options=["Average", "Weighted Average", "Coverage", "All"],
        initial_value="Average"
    )

def update_global_scores_display(
    global_scores: Dict[str, Dict[str, float]],
    selected_metric: str,
    display_mode: str,
) -> None:
    """Update GUI display with global scores."""
    if not global_scores:
        global_scores_text.value = "No scores available"
        return

    lines = []

    if display_mode == "All":
        # Show all metrics
        for metric_name, scores in global_scores.get("metrics", {}).items():
            avg = scores.get("average", 0.0)
            count = scores.get("count", 0)
            lines.append(f"{metric_name}: {avg:.3f} (n={count})")

        # Show coverage if available
        if "coverage" in global_scores:
            cov = global_scores["coverage"]
            lines.append(f"Coverage: {cov['coverage_ratio']:.1%} ({cov['valid_count']}/{cov['total_count']})")

    elif selected_metric in global_scores.get("metrics", {}):
        scores = global_scores["metrics"][selected_metric]
        if display_mode == "Average":
            lines.append(f"Average: {scores.get('average', 0.0):.3f}")
        elif display_mode == "Weighted Average":
            lines.append(f"Weighted Avg: {scores.get('weighted_average', scores.get('average', 0.0)):.3f}")

        lines.append(f"Count: {scores.get('count', 0)}")
        lines.append(f"Range: [{scores.get('min', 0.0):.3f}, {scores.get('max', 0.0):.3f}]")

    global_scores_text.value = "\n".join(lines)
```

### Comparison Between FK and IK Results

```python
def compare_fk_ik_scores(
    fk_scores: Dict[str, Dict[str, float]],
    ik_scores: Dict[str, Dict[str, float]],
) -> Dict[str, Any]:
    """Compare global scores between FK and IK approaches.

    Returns:
        Comparison dictionary with differences and insights
    """
    comparison = {
        "fk_scores": fk_scores,
        "ik_scores": ik_scores,
        "differences": {},
        "insights": [],
    }

    # Compare metric averages
    for metric_name in set(fk_scores.get("metrics", {}).keys()) | set(ik_scores.get("metrics", {}).keys()):
        fk_avg = fk_scores.get("metrics", {}).get(metric_name, {}).get("average", 0.0)
        ik_avg = ik_scores.get("metrics", {}).get(metric_name, {}).get("average", 0.0)

        diff = ik_avg - fk_avg
        comparison["differences"][metric_name] = {
            "fk_average": fk_avg,
            "ik_average": ik_avg,
            "difference": diff,
            "relative_difference": diff / max(fk_avg, 1e-6) if fk_avg > 0 else 0.0,
        }

        if abs(diff) > 0.1:
            comparison["insights"].append(
                f"{metric_name}: IK average ({ik_avg:.3f}) differs significantly from FK ({fk_avg:.3f})"
            )

    # Coverage comparison (IK-specific)
    if "coverage" in ik_scores:
        ik_coverage = ik_scores["coverage"]["coverage_ratio"]
        # Estimate FK coverage from visitation
        fk_visited = sum(scores.get("count", 0) for scores in fk_scores.get("metrics", {}).values())
        # This is approximate - would need total voxel count
        comparison["coverage_comparison"] = {
            "ik_coverage": ik_coverage,
            "note": "FK coverage estimation requires total voxel count",
        }

    return comparison
```

### Additional Questions

5. **Global scoring**: Which scoring methods should be the default? (Average, Weighted, Coverage, Composite)
6. **Score storage**: Store in HDF5 attributes, separate JSON file, or both?

---

## Implementation Status

### ✅ Completed (Phase 1-4)

1. **Global Scoring Module** (`reachability_scoring.py`)
   - ✅ `compute_average_metric_score()` - MATLAB-style average scoring
   - ✅ `compute_weighted_metric_score()` - Visitation-weighted scoring for FK
   - ✅ `compute_coverage_score()` - Coverage scoring for IK
   - ✅ `compute_composite_score()` - Multi-metric composite scoring
   - ✅ `compute_fk_global_scores()` - FK map scoring
   - ✅ `compute_ik_global_scores()` - IK results scoring
   - ✅ `compare_fk_ik_scores()` - Comparison utility

2. **FK Approach Integration**
   - ✅ Global scores computed and stored in HDF5 attributes
   - ✅ Scores include: average, weighted_average, min, max, std, count

3. **IK-Based Reachability Map Generation**
   - ✅ `IKReachabilityConfig` dataclass with all configuration options
   - ✅ `_generate_cartesian_grid()` - Grid generation helper
   - ✅ `_get_target_orientations()` - Orientation handling (fixed/reference/multiple)
   - ✅ `generate_ik_reachability_map()` - Main IK function with full workflow
   - ✅ Output format matches FK approach (PKL, HDF5, NPZ)
   - ✅ Global scores computed and stored in HDF5 attributes
   - ✅ Metrics computation integrated (Manipulability, RangeOfMotion, SingularityAvoidance)

4. **GUI Integration**
   - ✅ Analysis mode dropdown (FK Sampling / IK Grid)
   - ✅ IK-specific controls (grid step size, orientation mode, nullspace bias)
   - ✅ Controls enable/disable based on selected mode
   - ✅ Mode switching callback

5. **Workflow Integration**
   - ✅ `_launch_reachability_job()` supports both FK and IK modes
   - ✅ Proper configuration based on selected mode
   - ✅ Progress callbacks and error handling

### 🔄 Next Steps (Future Enhancements)

#### Phase 5: GUI Display for Global Scores
- **Priority**: Medium
- **Description**: Add GUI panel to display global scores from both FK and IK approaches
- **Tasks**:
  - Add "Global Scores" folder to GUI panel
  - Display average, weighted average, coverage metrics
  - Show score comparison between FK and IK if both available
  - Add metric selection dropdown
  - Add display mode selector (Average/Weighted/Coverage/All)

#### Phase 6: Score Comparison View
- **Priority**: Medium
- **Description**: Visual comparison between FK and IK results
- **Tasks**:
  - Create comparison visualization panel
  - Display side-by-side metrics
  - Show differences and insights
  - Highlight significant discrepancies

#### Phase 7: Performance Optimization
- **Priority**: Low (only if IK analysis is too slow)
- **Description**: Optimize IK solving with parallel processing
- **Tasks**:
  - Implement parallel IK solving using multiprocessing or threading
  - Add batch processing for grid points
  - Optimize seed configuration propagation
  - Add progress reporting for parallel jobs

#### Phase 8: Enhanced Orientation Sampling
- **Priority**: Low
- **Description**: Improve "multiple" orientation mode
- **Tasks**:
  - Implement proper orientation sampling (uniform sphere, grid, etc.)
  - Add orientation sampling strategies (identity, rotations around axes, etc.)
  - Allow user to specify custom orientations
  - Visualize orientation samples

#### Phase 9: Advanced Nullspace Bias Options
- **Priority**: Low
- **Description**: Add GUI controls for nullspace bias pose selection
- **Tasks**:
  - Add "elbow up/down" options
  - Add pose optimization options
  - Allow custom bias configuration
  - Visualize bias pose

#### Phase 10: Summary JSON File Generation
- **Priority**: Low
- **Description**: Generate separate JSON summary files for easy parsing
- **Tasks**:
  - Create `summary_{job_name}.json` file
  - Include global scores, configuration, timestamp
  - Add metadata about analysis mode
  - Include comparison data if both FK and IK available

### 📋 Testing Checklist

- [ ] Test FK global scoring with existing maps
- [ ] Test IK reachability generation with small grid (10x10x10)
- [ ] Verify output format compatibility with existing visualization
- [ ] Test mode switching in GUI
- [ ] Test cancellation during IK generation
- [ ] Compare FK vs IK results for same robot
- [ ] Validate against MATLAB reference implementation
- [ ] Performance benchmarking (IK vs FK speed)
- [ ] Test with different robots (Panda, iiwa)

### 🐛 Known Issues / Limitations

1. **Linter Warning**: Import `reachability_scoring` shows warning (expected - new module)
2. **Orientation Sampling**: "Multiple" mode currently simplified (only identity)
3. **Parallel Processing**: IK solving is sequential (can be slow for large grids)
4. **Nullspace Bias**: Currently uses current configuration, no GUI for custom bias pose
5. **Coverage Estimation**: FK coverage estimation requires total voxel count (not yet implemented)

### ✅ Recent Implementation Updates (2025)

**Dual-Backend Support for Interactive IK:**
- ✅ embodiK backend integrated alongside Placo for interactive IK visualization
- ✅ GUI dropdown to switch between Placo and embodiK backends
- ✅ Error threshold-based task activation implemented:
  - **POS_ERROR_THRESHOLD**: 1mm (0.001m) - prevents task activation for small position errors
  - **ROT_ERROR_THRESHOLD**: 1 degree (np.deg2rad(1.0)) - prevents task activation for small rotation errors
  - Prevents both arms from moving when only one marker is moved
- ✅ DOF mismatch handling: Placo (41 DOFs) vs embodiK (34 DOFs) for floating-base robots
- ✅ Quaternion sign correction to fix 180-degree rotation errors
- ✅ Comprehensive logging with LOGGER.debug/info replacing print statements

**Note**: These improvements are implemented in `04_1_validation_alpha_teststand_placo_example.py`. The IK-based reachability generation has been fully implemented using Placo backend (Phase 3 complete).

**Implementation Status:**
- ✅ IK-based reachability map generation with Placo backend (`_generate_reachability_map_thread`)
- ✅ ReachabilityMapViewer integrated for map visualization
- ✅ Sample configuration application callback implemented
- ✅ Metric display and diagnostics panel added
- ✅ Progress tracking and cancellation support
- ✅ Parallel processing support (thread-local robot instances)
- ✅ Arm selection (left/right) support

### 📝 Usage Notes

- **FK Mode**: Use for fast, broad workspace coverage
- **IK Mode**: Use for guaranteed coverage of specific Cartesian regions
- **Grid Step Size**: Smaller steps (5-10mm) give better resolution but slower
- **Orientation Mode**: "Fixed" is fastest, "Multiple" tests more orientations
- **Nullspace Bias**: Helps maintain preferred configurations during IK solving


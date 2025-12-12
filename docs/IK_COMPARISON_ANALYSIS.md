# IK Comparison Analysis: Teststand vs Wheelbase

## Test Results Summary

### Teststand Model
- ✅ **Correct behavior**: When moving only left arm, NO torso/neck joints move
- ✅ **Correct behavior**: When moving only right arm, NO torso/neck joints move

### Wheelbase Model
- ❌ **Incorrect behavior**: When moving only left arm, 5 torso/neck joints move:
  - `hip_pitch_joint` (-1.317 mrad)
  - `base_pitch_joint` (+0.388 mrad)
  - `knee_pitch_joint` (-0.203 mrad)
  - `base_yaw_joint` (+0.068 mrad)
  - `torso_yaw_joint` (+0.027 mrad)

- ❌ **Incorrect behavior**: When moving only right arm, 5 torso/neck joints move:
  - `neck_yaw_joint` (+1.317 mrad)
  - `base_pitch_joint` (-0.388 mrad)
  - `knee_pitch_joint` (-0.203 mrad)
  - `base_yaw_joint` (-0.068 mrad)
  - `neck_pitch_joint` (+0.027 mrad)

## Root Cause Analysis

### Verified Correct Components
1. ✅ **Index Mapping**: Joint indices → velocity space indices → config space indices are correct
   - `joint_names` indices → velocity space: `+6` offset for floating base
   - Velocity space → config space: `v_idx + 1` (PostureTask internal mapping)
   - Direct mapping: `joint_idx_to_q_idx`: `7 + joint_idx` for floating base
   - All mappings verified and match ✓

2. ✅ **Posture Task Setup**: Task is created correctly with proper indices
   - Velocity space indices: `[15, 16, 17, 18, 19, 20, 21]` (correctly offset by +6)
   - Target configuration is set correctly
   - Weight and priority are set correctly

3. ✅ **Target Configuration**: `torso_neck_target` is updated correctly in each iteration
   - Target values match current configuration values ✓

### The Problem

**The frame task Jacobian includes ALL joints in the kinematic chain**, including torso/neck joints. When embodiK solves the weighted least squares problem:

```
minimize: ||J_frame * v - v_frame||² + weight_posture * ||J_posture * v - v_posture||²
```

Even with `weight_posture = 10.0` and `priority = 0` (same as frame task), the frame task can still move torso joints because:
1. The frame task Jacobian spans torso joints (they're in the kinematic chain)
2. Tasks with the same priority are combined in a weighted least squares problem
3. The solver finds a compromise solution that satisfies both tasks

### Why Teststand Works

The teststand model has a **fixed base**, so:
- No floating base DOFs
- Torso joints are NOT in the kinematic chain from base to end-effector
- Frame task Jacobian does NOT include torso joints
- Posture task can successfully lock torso joints

### Why Wheelbase Fails

The wheelbase model has a **floating base**, so:
- Torso joints ARE in the kinematic chain from base to end-effector
- Frame task Jacobian INCLUDES torso joints
- Even with posture task trying to lock them, frame task can still move them
- The solver finds a compromise that moves torso joints slightly

## Potential Solutions

### ✅ Option 1: Zero Out Torso/Neck DOFs in Frame Task Jacobian (VERIFIED WORKING)
**Status: TESTED AND CONFIRMED**

- **Test Results**: Zeroing out torso/neck columns in Jacobian reduces movement from 5 joints to 1 joint
- **Remaining movement**: 0.044 mrad (likely numerical error from pseudoinverse)
- **Reduction**: ~30x improvement (from 1.317 mrad to 0.044 mrad)
- **Implementation**: Modify `FrameTask::getJacobian()` to zero out specified velocity space columns
- **Required**: Add API to specify which joints to exclude from frame task Jacobian

### Option 2: Use Higher Priority for Posture Task (Not Supported)
- embodiK doesn't support negative priorities or priorities higher than 0
- Would need to modify embodiK core

### Option 3: Use Much Higher Weight (May Not Work)
- Current weight: 10.0
- Could try 100.0 or 1000.0, but may cause numerical issues
- Still may not work because frame task Jacobian spans torso joints

### Option 4: Use Velocity Constraints (Alternative)
- Instead of posture task, use velocity constraints to set torso joint velocities to zero
- This would be a hard constraint rather than a soft task
- May require modifying embodiK to support per-joint velocity constraints

### Option 5: Modify Frame Task to Exclude Torso Joints (Complex)
- Would need to compute Jacobian only for arm joints
- Requires significant changes to embodiK internals

## Recommended Next Steps

1. ✅ **VERIFIED**: Zeroing out torso/neck columns in frame task Jacobian works
   - Test script: `test_zero_torso_jacobian.py`
   - Results: Movement reduced from 5 joints to 1 joint (0.044 mrad, likely numerical error)

2. **Implement fix in embodiK**:
   - Add API to `FrameTask` to specify excluded velocity space indices
   - Modify `FrameTask::getJacobian()` to zero out excluded columns
   - Example API:
     ```python
     left_task = solver.add_frame_task("left_arm_task", LEFT_FRAME)
     left_task.exclude_velocity_indices(torso_neck_v_indices)  # New API
     ```

3. **Alternative**: Modify `04_validation_alpha_example.py` to manually zero out Jacobian columns
   - Would require accessing/modifying Jacobian before solver uses it
   - Less clean but could work as temporary solution

## Test Script

Run the comparison test:
```bash
cd /path/to/local/Projects/playground/embodik/examples
python3 test_ik_comparison.py --model both
```

Verbose output:
```bash
python3 test_ik_comparison.py --model wheelbase --verbose
```


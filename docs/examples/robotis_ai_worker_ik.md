# ROBOTIS AI Worker IK Example

`examples/incubating/robotis_ai_worker_ik.py` is a Viser-based dual-arm IK demo for the
local ROBOTIS FFW worker URDFs.

## What It Covers

- Dual 6D transform controls for left/right worker tools
- Viser visualization throughout via `ViserServer` + `ViserUrdf`
- Lift and head posture bias controls layered underneath the IK tasks
- Passive-joint locking so wheel/drive joints stay quiet during arm teleoperation
- Support for both local variants available in this workspace:
  - `sg2`
  - `bg2`

## Run It

From the repository root:

```bash
pixi run python examples/incubating/robotis_ai_worker_ik.py --variant sg2
```

Or switch to the other local worker variant:

```bash
pixi run python examples/incubating/robotis_ai_worker_ik.py --variant bg2
```

## Asset Resolution

The script resolves the worker URDF from one of these locations:

- `EMBODIK_FFW_SG2_URDF`
- `EMBODIK_FFW_BG2_URDF`
- `EMBODIK_FFW_URDF`
- Workspace defaults under `/path/to/local/Projects/robot_models_urdf/`

## Note About SH5

The G1 notebook reference for the AI worker IK flow uses `FFW-SH5`, but
that model is only available here as MJCF/XML, not as a URDF. This example is
therefore built on the available `SG2` / `BG2` URDF variants while keeping the
same dual-arm interactive IK workflow.

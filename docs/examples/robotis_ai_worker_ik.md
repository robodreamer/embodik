# ROBOTIS AI Worker Constraint Teleop Example

`examples/12_ai_worker_constraint_teleop.py` is a Viser-based dual-arm IK demo
for ROBOTIS AI Worker FFW models. It uses the public `ROBOTIS-GIT/ai_worker`
repository for visual URDF assets and bundled reduced collision assets for
interactive collision-aware IK.

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
pixi run python examples/12_ai_worker_constraint_teleop.py --variant sg2
```

Or switch to the other local worker variant:

```bash
pixi run python examples/12_ai_worker_constraint_teleop.py --variant bg2
```

## Asset Resolution

The script resolves the worker URDF from one of these locations:

- explicit `--urdf` / `--collision-urdf` arguments
- explicit `--ai-worker-root /path/to/ROBOTIS-GIT/ai_worker`
- `EMBODIK_AI_WORKER_ROOT` / `AI_WORKER_ROOT`
- automatic cached download from the public `ROBOTIS-GIT/ai_worker` repository

The generated reduced collision URDF is bundled under
`examples/assets/ai_worker/generated/` when examples are copied from the pip
package with `embodik-examples --copy`.

## Note About SH5

The G1 notebook reference for the AI worker IK flow uses `FFW-SH5`, but
that model is only available here as MJCF/XML, not as a URDF. This example is
therefore built on the available `SG2` / `BG2` URDF variants while keeping the
same dual-arm interactive IK workflow.

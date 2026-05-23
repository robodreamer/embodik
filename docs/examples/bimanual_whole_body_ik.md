# Bimanual Whole-Body IK Example

`examples/06_bimanual_whole_body_ik.py` is a Viser-based bimanual whole-body IK
demo for ROBOTIS AI Worker FFW models and RB-Y1. It uses the public
`ROBOTIS-GIT/ai_worker` repository for AI Worker visual URDF assets, bundled
reduced collision assets for AI Worker collision-aware IK, and
`robot_descriptions.rby1_description` when launched with `--robot rby1`.

<video autoplay muted loop playsinline controls width="100%" src="../../assets/media/rby1_collision_free_ik.mp4"></video>

## What It Covers

- Dual 6D transform controls for left/right worker tools
- Viser visualization throughout via `ViserServer` + `ViserUrdf`
- Model-aware posture bias controls layered underneath the IK tasks
- Passive-joint locking so wheel/drive joints stay quiet during arm teleoperation
- Support for AI Worker `sg2` / `bg2` and RB-Y1 via `--robot rby1`

## Run It

Install and copy the example bundle once using the
[Installation Guide](../installation.md#examples). Then run:

```bash
cd embodik_examples
python 06_bimanual_whole_body_ik.py
```

If you are not activating the venv, use the Python path printed by
`embodik-examples --copy`, for example:

```bash
/path/to/.venv/bin/python 06_bimanual_whole_body_ik.py
```

From a repository checkout, use Pixi:

```bash
pixi run python examples/06_bimanual_whole_body_ik.py
```

The default robot model is AI Worker `sg2`. To use the other worker variant,
pass `--variant bg2`; to launch RB-Y1, pass `--robot rby1`. For example, from
copied examples:

```bash
python 06_bimanual_whole_body_ik.py --variant bg2
python 06_bimanual_whole_body_ik.py --robot rby1
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

# Spot Full-Body IK Example

`examples/14_spot_full_body_ik_viser.py` is a regular Viser Spot arm IK demo.
It exposes arm+torso, torso-only, full-body, and two-stage solve modes so the
same target can be tested with different whole-body coordination policies.

<video autoplay muted loop playsinline controls width="100%" src="../../assets/media/spot_fullbody_interactive_ik.mp4"></video>

## What It Covers

- Interactive gripper target control in regular Viser.
- Switchable IK modes for arm-only, torso-assisted, full-body, and two-stage
  behavior.
- Optional Seer controller teleop when the teleop extra is installed.
- Spot arm and body command visualization for debugging reachable motion.

## Run It

For a pip/venv install, install and copy the example bundle once:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install "embodik[examples]"
embodik-examples --copy
cd embodik_examples
python 14_spot_full_body_ik_viser.py
```

The copied example bundle includes a Spot-with-arm URDF, so no `--urdf`
argument is needed for the default path. Use
`--urdf /path/to/spot.urdf` or set `EMBODIK_SPOT_IK_URDF` only when you want to
override the bundled model.

From a repository checkout, use Pixi:

```bash
pixi run python examples/14_spot_full_body_ik_viser.py
```

For Seer controller teleop from a checkout, use the teleop environment:

```bash
pixi run -e teleop python examples/14_spot_full_body_ik_viser.py --enable-teleop
```

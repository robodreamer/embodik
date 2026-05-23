# Spot Locomanipulation mjviser Example

`examples/09_spot_locomanip_mjviser.py` runs a Spot locomanipulation ONNX
policy in MuJoCo through mjviser while EmbodiK handles interactive arm IK.
It defaults to the bundled public MuJoCo Menagerie Spot-with-arm scene at
`examples/assets/spot_mjcf/scene_arm.xml`, so copied examples do not need a
first-run model download. The example is useful for testing arm reachability,
locomotion handoff, self-collision constraints, solver timing, and optional
Seer controller teleop.

<video autoplay muted loop playsinline controls width="100%" src="../../assets/media/spot_locomanip_interactive_ik_mjviser.mp4"></video>

## Policy Control Preview

<video autoplay muted loop playsinline controls width="100%" src="../../assets/media/spot_locomanipulation_policy_gui_control_mjviser.mp4"></video>

## What It Covers

- MuJoCo policy rollout through mjviser with GUI body and arm commands.
- Interactive gripper IK with condition-number diagnostics and asynchronous IK
  solving by default.
- 50 Hz ONNX policy inference while MuJoCo simulation and rendering continue at
  the model/viewer rate.
- Optional collision constraints with curated Spot arm/body collision pairs.
  The default collision setting checks the 3 closest active pairs with balanced
  speed/accuracy tuning when collision avoidance is enabled.
- Seer controller teleop through `--enable-teleop`; the in-app teleop checkbox
  starts enabled when the controller connects and remains enabled across sim
  reset.
- Solver-native base-assist tuning for sharing gripper target motion between
  the arm and x/y/yaw locomotion, plus arm recovery bias tuning for
  high-condition-number stretched-arm recovery. The defaults favor teleop
  recovery: the base helps on reachable x/y/yaw nudges, and the arm is biased
  away from stretched configurations when condition protection is active.

## Run It

For a pip/venv install, install the mjviser extra and copy the examples before
running the script:

```bash
python -m pip install "embodik[mjviser]"
embodik-examples --copy
cd embodik_examples
python 09_spot_locomanip_mjviser.py --policy locomanip
python 09_spot_locomanip_mjviser.py --policy locomanip-stationary
```

The copied example bundle includes the public MuJoCo Menagerie Spot-with-arm
MJCF model, so no `robot_descriptions` first-run model download is required for
the default scene.

For Seer controller teleop, install the combined optional extra:

```bash
python -m pip install "embodik[mjviser,teleop]"
python 09_spot_locomanip_mjviser.py --enable-teleop --policy locomanip
```

From a repository checkout, use the matching Pixi environments:

```bash
pixi run -e mjviser spot-locomanip-mjviser --policy locomanip
pixi run -e mjviser-teleop spot-locomanip-mjviser --enable-teleop --policy locomanip
```

`mjviser` is the MuJoCo-backed web viewer environment for browser GUI policy
rollout and interactive simulation. `mjviser-teleop` includes that same viewer
stack plus the optional Seer/xvisio controller dependencies. Use `mjviser` for
normal GUI control, and switch to `mjviser-teleop` only when passing
`--enable-teleop`.

Use `--sync-ik` when debugging deterministic single-thread IK behavior. The
default viewer path runs IK on a background worker so collision-constrained
solves do not block policy inference and rendering.

The policy loop and background IK request loop are both rate-limited to 50 Hz
by default, while MuJoCo continues stepping at the model timestep. The IK worker
keeps only the newest pending request so collision-constrained solves cannot
build a backlog and starve the sim thread. Use `--policy-rate-hz` and
`--async-ik-rate-hz` only for experiments; the defaults are the intended
loco-manipulation teleop settings.

## Headless Checks

Clone-based Spot IK changes can be checked with:

```bash
pixi run -e mjviser-teleop python scripts/spot_locomanip_ik_hardening.py
```

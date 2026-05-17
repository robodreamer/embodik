# Spot Locomanipulation Policy Checkpoints

This directory contains the two ONNX checkpoints used by
`examples/15_spot_locomanip_mjviser.py`:

- `locomanip_policy.onnx`
- `locomanip_stationary_policy.onnx`

The checkpoints are vendored for this standalone example. Keep any replacement
policy files in ONNX format and preserve the input/output observation contract
used by the helper modules.

The MuJoCo Spot model is not vendored here. The example loads the public
MuJoCo Menagerie Spot arm model through `robot_descriptions.spot_mj_description`.

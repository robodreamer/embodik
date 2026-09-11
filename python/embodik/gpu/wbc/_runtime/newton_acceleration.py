"""CUDA frame kinematics and acceleration bias for fixed-base robots.

The evaluator returns the same physical quantity used by EmbodiK's CPU
acceleration solver:: ``Jdot(q, dq) * dq`` for a
``LOCAL_WORLD_ALIGNED`` frame Jacobian.  Newton does not currently expose a
frame-Jacobian time-variation primitive, so the bias is evaluated as a
second-order centered directional derivative of its analytic Jacobian,

``(J(q + h dq) - J(q - h dq)) / (2 h) * dq``.

For fixed-base scalar joints, configuration and tangent coordinates coincide,
which makes this derivative unambiguous.  Floating roots are deliberately
rejected until their manifold perturbation and LOCAL_WORLD_ALIGNED tangent
semantics can be verified directly against the CPU path.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from ..contracts import RobotSolveSpec
from .newton_model import NewtonModelKinematics


class NewtonFrameAccelerationEvaluator:
    """Model-derived, graph-safe frame acceleration-bias evaluator.

    ``evaluate(q, dq)`` returns ``(pose, jacobian, bias)`` on the configured
    CUDA device.  Poses use Newton's ``[x, y, z, qx, qy, qz, qw]`` layout;
    Jacobian rows and bias entries are ordered ``[linear; angular]`` in the
    world-aligned frame convention used by EmbodiK.

    The finite-difference configurations are persistent scratch tensors.
    Calls must be serialized across CUDA streams, matching
    :class:`NewtonModelKinematics`.  Warm up once before enclosing evaluation
    in a parent ``torch.cuda.CUDAGraph``.
    """

    supports_exact_frame_bias = False
    supports_centered_frame_bias = True
    supports_floating_base = False

    def __init__(
        self,
        batch_size: int,
        urdf_path: Path,
        cache_dir: Path,
        robot_spec: RobotSolveSpec,
        frame: str | tuple[str, ...],
        *,
        default_configuration: tuple[float, ...] | None = None,
        difference_step: float = 2.0e-3,
        device: str = "cuda:0",
    ) -> None:
        if robot_spec.floating_base:
            raise NotImplementedError(
                "frame acceleration bias currently supports fixed-base models only"
            )
        if not isinstance(difference_step, (float, int)) or isinstance(difference_step, bool):
            raise TypeError("difference_step must be a finite positive scalar")
        difference_step = float(difference_step)
        if not math.isfinite(difference_step) or difference_step <= 0.0:
            raise ValueError("difference_step must be a finite positive scalar")

        self.kinematics = NewtonModelKinematics(
            batch_size,
            Path(urdf_path),
            Path(cache_dir),
            robot_spec,
            frame,
            default_configuration=default_configuration,
            device=device,
        )
        self.torch = self.kinematics.torch
        self.device = self.kinematics.device
        self.batch_size = batch_size
        self.robot_spec = robot_spec
        self.frames = self.kinematics.frames
        self.active_dim = self.kinematics.active_dim
        self.input_configuration_dim = self.kinematics.input_configuration_dim
        if self.input_configuration_dim != self.active_dim:
            raise RuntimeError("fixed-base acceleration evaluation requires scalar active joints")
        self.difference_step = difference_step
        self._q_plus = self.torch.empty(
            (batch_size, self.input_configuration_dim),
            dtype=self.torch.float32,
            device=self.device,
        )
        self._q_minus = self.torch.empty_like(self._q_plus)

    def _validate_tensor(self, value: Any, name: str, width: int) -> None:
        torch = self.torch
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if value.shape != (self.batch_size, width):
            raise ValueError(
                f"{name} must have shape {(self.batch_size, width)}, " f"got {tuple(value.shape)}"
            )
        if value.device != self.device or value.dtype is not torch.float32:
            raise ValueError(f"{name} must be CUDA float32 on {self.device}")

    def evaluate(self, q: Any, dq: Any) -> tuple[Any, Any, Any]:
        """Return frame pose, physical Jacobian, and ``Jdot(q,dq)*dq``.

        Shape/type/device errors raise synchronously.  Nonfinite state values
        produce NaNs in every output row for that batch item without a device
        synchronization, preserving fail-closed graph execution.
        """

        self._validate_tensor(q, "q", self.input_configuration_dim)
        self._validate_tensor(dq, "dq", self.active_dim)
        return self._evaluate_trusted(q, dq)

    def _evaluate_trusted(self, q: Any, dq: Any) -> tuple[Any, Any, Any]:
        """Evaluate tensors whose shape, dtype, and device are already valid."""

        torch = self.torch
        torch.add(q, dq, alpha=self.difference_step, out=self._q_plus)
        torch.add(q, dq, alpha=-self.difference_step, out=self._q_minus)

        pose, jacobian = self.kinematics._evaluate_trusted(q)
        _, jacobian_plus = self.kinematics._evaluate_trusted(self._q_plus)
        _, jacobian_minus = self.kinematics._evaluate_trusted(self._q_minus)
        jacobian_rate = (jacobian_plus - jacobian_minus) * (0.5 / self.difference_step)
        bias = torch.bmm(jacobian_rate, dq.unsqueeze(-1)).squeeze(-1)

        valid = (
            torch.isfinite(q).all(dim=1)
            & torch.isfinite(dq).all(dim=1)
            & torch.isfinite(pose).all(dim=tuple(range(1, pose.ndim)))
            & torch.isfinite(jacobian).all(dim=(1, 2))
            & torch.isfinite(bias).all(dim=1)
        )
        pose_mask = valid.reshape((self.batch_size,) + (1,) * (pose.ndim - 1))
        return (
            torch.where(pose_mask, pose, float("nan")),
            torch.where(valid[:, None, None], jacobian, float("nan")),
            torch.where(valid[:, None], bias, float("nan")),
        )

    def metadata(self) -> dict[str, Any]:
        """Return stable construction and numerical-method metadata."""

        return {
            **self.kinematics.metadata(),
            "evaluator": "frame_acceleration_bias",
            "bias_method": "centered_directional_jacobian_difference",
            "difference_step": self.difference_step,
            "supports_exact_frame_bias": self.supports_exact_frame_bias,
            "supports_floating_base": self.supports_floating_base,
            "host_roundtrip_in_evaluate": False,
            "preallocated_difference_scratch": True,
        }

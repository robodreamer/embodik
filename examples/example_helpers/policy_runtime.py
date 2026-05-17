"""Reusable policy inference runtime helpers for examples."""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

DEFAULT_POLICY_RATE_HZ = 50.0


class RateLimitedOnnxPolicy:
    """Run an ONNX policy at a fixed rate and reuse the latest output between ticks."""

    def __init__(self, *, rate_hz: float = DEFAULT_POLICY_RATE_HZ, output_len: int = 12):
        self.policy_dt = 1.0 / max(float(rate_hz), 1e-6)
        self.session: Any | None = None
        self.input_name = ""
        self.output_name = ""
        self.last_output = np.zeros(int(output_len), dtype=np.float32)
        self.step_count = 0
        self.last_inference_time_ms = 0.0
        self._elapsed = self.policy_dt

    def configure(
        self,
        session: Any,
        *,
        input_name: str,
        output_name: str,
        output_len: int,
    ) -> None:
        self.session = session
        self.input_name = str(input_name)
        self.output_name = str(output_name)
        self.last_output = np.zeros(int(output_len), dtype=np.float32)
        self.reset()

    def reset(self) -> None:
        self.last_output.fill(0.0)
        self.step_count = 0
        self.last_inference_time_ms = 0.0
        self._elapsed = self.policy_dt

    def due(self, dt: float) -> bool:
        self._elapsed += max(float(dt), 0.0)
        if self._elapsed + 1e-12 < self.policy_dt:
            return False
        self._elapsed = math.fmod(self._elapsed, self.policy_dt)
        return True

    def run(self, policy_input: np.ndarray) -> np.ndarray:
        if self.session is None:
            raise RuntimeError("Policy runtime has not been configured")
        policy_start = time.perf_counter()
        policy_output = self.session.run(
            [self.output_name],
            {self.input_name: np.asarray(policy_input, dtype=np.float32).reshape(1, -1)},
        )[0][0]
        self.last_output = np.asarray(policy_output, dtype=np.float32)
        self.last_inference_time_ms = (time.perf_counter() - policy_start) * 1e3
        self.step_count += 1
        return self.last_output

    def maybe_run(self, dt: float, policy_input: np.ndarray) -> np.ndarray:
        if self.due(dt):
            return self.run(policy_input)
        return self.last_output

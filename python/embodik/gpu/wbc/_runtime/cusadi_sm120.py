"""Strict device-resident CusADi execution with explicit artifact selection."""

from __future__ import annotations

import ctypes
import math
from pathlib import Path
from typing import Any


class StrictCusadiFunction:
    """Load one compiled CusADi library without fallback or implicit paths."""

    def __init__(
        self,
        casadi_function: Any,
        library_path: Path,
        batch_size: int,
        *,
        device: str = "cuda:0",
        expected_capability: tuple[int, int] = (12, 0),
        scalar_type: str = "float64",
    ) -> None:
        import torch

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required; CPU fallback is forbidden")
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("StrictCusadiFunction requires a CUDA device")
        if scalar_type not in {"float64", "float32"}:
            raise ValueError("scalar_type must be float64 or float32")
        self.scalar_type = scalar_type
        self.dtype = torch.float64 if scalar_type == "float64" else torch.float32
        capability = torch.cuda.get_device_capability(self.device)
        if capability != expected_capability:
            raise RuntimeError(
                f"CUDA capability mismatch: expected {expected_capability}, got {capability}"
            )
        resolved_library = library_path.expanduser().resolve()
        if not resolved_library.is_file():
            raise FileNotFoundError(f"CusADi library not found: {resolved_library}")

        self.torch = torch
        self.function = casadi_function
        self.library_path = resolved_library
        self.batch_size = batch_size
        self._library = ctypes.CDLL(str(resolved_library))
        scalar_bytes = getattr(self._library, "cusadi_scalar_bytes", None)
        if scalar_bytes is not None:
            scalar_bytes.argtypes = ()
            scalar_bytes.restype = ctypes.c_int
            expected_bytes = 8 if scalar_type == "float64" else 4
            actual_bytes = int(scalar_bytes())
            if actual_bytes != expected_bytes:
                raise RuntimeError(
                    "CusADi scalar ABI mismatch: "
                    f"artifact uses {actual_bytes}-byte values, requested {scalar_type}"
                )
        elif scalar_type != "float64":
            raise RuntimeError(
                "legacy CusADi artifacts without scalar ABI metadata are float64-only"
            )
        self._library.evaluate.argtypes = (
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
        )
        self._library.evaluate.restype = ctypes.c_float
        self._evaluate_on_stream = getattr(self._library, "evaluate_on_stream", None)
        if self._evaluate_on_stream is not None:
            self._evaluate_on_stream.argtypes = (
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_uint64,
            )
            self._evaluate_on_stream.restype = ctypes.c_int
        self._outputs = [
            torch.zeros(
                (batch_size, casadi_function.nnz_out(index)),
                dtype=self.dtype,
                device=self.device,
            )
            for index in range(casadi_function.n_out())
        ]
        self._work = torch.zeros(
            (batch_size, casadi_function.sz_w()),
            dtype=self.dtype,
            device=self.device,
        )
        self._input_pointers = torch.empty(
            casadi_function.n_in(), dtype=torch.int64, device=self.device
        )
        self._output_pointers = torch.tensor(
            [tensor.data_ptr() for tensor in self._outputs],
            dtype=torch.int64,
            device=self.device,
        )
        # Keep graph-capture pointer tables and their pinned host sources alive
        # for the lifetime of every captured replay that references them.
        self._captured_input_pointer_tables: list[tuple[Any, Any]] = []

    def _validate_inputs(self, inputs: tuple[Any, ...]) -> None:
        torch = self.torch
        if len(inputs) != self.function.n_in():
            raise ValueError(f"expected {self.function.n_in()} inputs, received {len(inputs)}")
        for index, tensor in enumerate(inputs):
            expected = self.batch_size * self.function.nnz_in(index)
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"input {index} is not a torch.Tensor")
            if tensor.device != self.device:
                raise ValueError(f"input {index} must reside on {self.device}")
            if tensor.dtype is not self.dtype:
                raise ValueError(f"input {index} must use {self.scalar_type}")
            if not tensor.is_contiguous():
                raise ValueError(f"input {index} must be contiguous")
            if tensor.numel() != expected:
                raise ValueError(f"input {index} has {tensor.numel()} values; expected {expected}")
            if not bool(torch.isfinite(tensor).all().item()):
                raise ValueError(f"input {index} contains non-finite values")

    def evaluate(self, inputs: tuple[Any, ...]) -> float:
        """Evaluate after strict device, shape, dtype, and finiteness checks."""

        torch = self.torch
        self._validate_inputs(inputs)
        if self._evaluate_on_stream is None:
            torch.cuda.synchronize(self.device)
            kernel_seconds = self._evaluate_trusted(inputs)
            torch.cuda.synchronize(self.device)
        else:
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            start.record()
            self._evaluate_trusted(inputs)
            stop.record()
            stop.synchronize()
            kernel_seconds = start.elapsed_time(stop) / 1000.0
        if not all(bool(torch.isfinite(output).all().item()) for output in self._outputs):
            raise RuntimeError("CusADi produced non-finite output")
        return kernel_seconds

    def _evaluate_trusted(self, inputs: tuple[Any, ...]) -> float:
        """Evaluate tensors already validated by an enclosing GPU request."""

        torch = self.torch
        pointer_values = [tensor.data_ptr() for tensor in inputs]
        input_pointers = self._input_pointers
        if torch.cuda.is_current_stream_capturing():
            pinned = torch.tensor(pointer_values, dtype=torch.int64, pin_memory=True)
            input_pointers = pinned.to(self.device, non_blocking=True)
            self._captured_input_pointer_tables.append((pinned, input_pointers))
        else:
            self._input_pointers.copy_(
                torch.tensor(
                    pointer_values,
                    dtype=torch.int64,
                    device=self.device,
                )
            )
        for output in self._outputs:
            output.zero_()
        self._work.zero_()
        if self._evaluate_on_stream is not None:
            stream = torch.cuda.current_stream(self.device)
            error_code = int(
                self._evaluate_on_stream(
                    ctypes.c_void_p(input_pointers.data_ptr()),
                    ctypes.c_void_p(self._work.data_ptr()),
                    ctypes.c_void_p(self._output_pointers.data_ptr()),
                    self.batch_size,
                    ctypes.c_uint64(stream.cuda_stream),
                )
            )
            if error_code != 0:
                raise RuntimeError(f"CusADi stream launch failed with CUDA error {error_code}")
            return 0.0
        kernel_seconds = float(
            self._library.evaluate(
                ctypes.c_void_p(input_pointers.data_ptr()),
                ctypes.c_void_p(self._work.data_ptr()),
                ctypes.c_void_p(self._output_pointers.data_ptr()),
                self.batch_size,
            )
        )
        if not math.isfinite(kernel_seconds) or kernel_seconds < 0.0:
            raise RuntimeError(f"CusADi reported invalid kernel time: {kernel_seconds}")
        return kernel_seconds

    def dense_output(self, index: int) -> Any:
        """Return one dense output without copying it off the CUDA device."""

        output = self._outputs[index]
        rows = self.function.size1_out(index)
        columns = self.function.size2_out(index)
        if self.function.nnz_out(index) != rows * columns:
            raise NotImplementedError("sparse CasADi outputs are not supported")
        return output.reshape(self.batch_size, rows, columns)

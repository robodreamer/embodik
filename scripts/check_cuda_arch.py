#!/usr/bin/env python3
"""Fail early when the installed Torch wheel cannot execute on this CUDA GPU."""

from __future__ import annotations


def main() -> None:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "Torch is not installed. Use the CUDA environment: pixi run -e cuda ..."
        ) from exc

    if not torch.cuda.is_available():
        raise SystemExit("Torch cannot access CUDA in this environment.")

    major, minor = torch.cuda.get_device_capability()
    required_arch = f"sm_{major}{minor}"
    supported_arches = tuple(torch.cuda.get_arch_list())
    if required_arch not in supported_arches:
        remedy = (
            " Run `pixi run -e cuda setup-cuda-sm120` first."
            if required_arch == "sm_120"
            else " Install a Torch CUDA wheel that includes this architecture."
        )
        raise SystemExit(
            f"Torch {torch.__version__} does not include {required_arch}; "
            f"its wheel supports {', '.join(supported_arches) or 'no CUDA architectures'}."
            + remedy
        )

    probe = torch.arange(8, dtype=torch.float32, device="cuda")
    value = (probe * probe).sum()
    torch.cuda.synchronize()
    if float(value) != 140.0:
        raise SystemExit("CUDA execution probe returned an unexpected result.")
    print(
        f"Torch {torch.__version__}; CUDA {torch.version.cuda}; "
        f"device capability {required_arch}; kernel probe passed"
    )


if __name__ == "__main__":
    main()

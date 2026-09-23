#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "Run this through Pixi: pixi run -e cuda setup-cuda-sm120" >&2
  exit 2
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required to select the PyTorch CUDA backend." >&2
  echo "Install uv from https://docs.astral.sh/uv/ and retry." >&2
  exit 2
fi

uv pip install \
  --python "${CONDA_PREFIX}/bin/python" \
  --torch-backend cu129 \
  --upgrade \
  "torch==2.12.1"

"${CONDA_PREFIX}/bin/python" scripts/check_cuda_arch.py

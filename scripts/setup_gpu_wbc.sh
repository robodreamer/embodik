#!/usr/bin/env bash
# One-step setup for the Pixi CUDA environment used by model-derived GPU WBC.
#
# From a repository checkout:
#   pixi run setup-gpu-wbc
#
# The command creates the cuda environment, installs EmbodiK, selects a Torch
# wheel that can execute on this GPU, then installs the sibling Newton checkout.
# Newton is installed after any Torch repair because that repair replaces the
# CUDA environment's Python packages.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

newton_dir="${NEWTON_DIR:-${repo_root}/../newton}"
if [[ ! -f "${newton_dir}/pyproject.toml" ]]; then
  echo "Cloning Newton into ${newton_dir}"
  git clone --depth 1 https://github.com/newton-physics/newton.git "${newton_dir}"
fi

echo "Installing EmbodiK into the Pixi cuda environment"
pixi run -e cuda install

if ! cuda_check="$(pixi run -e cuda check-cuda 2>&1)"; then
  printf '%s\n' "${cuda_check}"
  if [[ "${cuda_check}" == *"setup-cuda-sm120"* ]]; then
    echo "Installing the sm_120 Torch wheel into the Pixi cuda environment"
    pixi run -e cuda setup-cuda-sm120
  else
    exit 1
  fi
else
  printf '%s\n' "${cuda_check}"
fi

echo "Installing Newton from ${newton_dir}"
pixi run -e cuda python -m pip install -e "${newton_dir}" --no-deps
pixi run -e cuda python -c "import newton, torch, embodik"
pixi run -e cuda check-cuda

echo "GPU WBC is ready. Run: pixi run -e cuda demo-parallel-tracking"

#!/usr/bin/env bash
# Idempotent Cloud Agent setup for CoCoGraph.
# Builds a Python 3.9 virtualenv (matching the project's documented interpreter)
# and installs the subset of dependencies the code actually imports.
set -euo pipefail

# Run from the repository root regardless of where the script is invoked.
cd "$(dirname "$0")/.."

export PATH="$HOME/.local/bin:$PATH"

# 1. Ensure uv is available (used to provision Python 3.9 + install packages fast).
if ! command -v uv >/dev/null 2>&1; then
  pip install --user uv
fi

# 2. Create the Python 3.9 virtualenv (the repo targets Python 3.9.x).
#    Guarded so re-running install does not rebuild an existing environment.
if [ ! -x .venv/bin/python ]; then
  uv venv --python 3.9 .venv
fi

# 3. Install PyTorch (CPU build) first from the PyTorch CPU index.
#    Cloud Agent VMs are CPU-only; the CPU wheel avoids ~1.5 GB of unused CUDA
#    libraries per process, which matters because the sampling pipeline spawns
#    one worker process per CPU core.
uv pip install --python .venv/bin/python "torch==2.2.0+cpu" \
  --index-url https://download.pytorch.org/whl/cpu

# 4. Install the remaining runtime dependencies actually imported by the code,
#    pinned to the versions from requirements_vast.txt. (requirements_vast.txt is
#    a full machine freeze that is not installable on Python 3.9 as-is, e.g. it
#    pins Django==5.1.2 which requires Python >=3.10.)
uv pip install --python .venv/bin/python \
  torch_geometric==2.4.0 \
  rdkit==2023.9.4 \
  numpy==1.26.4 \
  pandas==2.2.0 \
  networkx==3.2.1 \
  scipy==1.10.1 \
  matplotlib==3.8.2 \
  seaborn==0.13.2 \
  tqdm==4.66.1 \
  tableone==0.8.0 \
  func-timeout==4.3.5

# 5. Fetch the Git LFS dataset needed for sampling/evaluation demos.
#    Only the annotated molecule CSV is pulled; the multi-GB generated databases
#    are intentionally left as pointers.
if command -v git-lfs >/dev/null 2>&1; then
  git lfs install --local || true
  git lfs pull --include="Data/molecules_lt70atoms_annotated.csv" || true
fi

echo "CoCoGraph environment ready. Activate with: source .venv/bin/activate"

#!/bin/bash
#SBATCH --job-name=crisk-env
#SBATCH --partition=normal
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

# Run once. CREST ships no conda module, so this bootstraps Miniforge into
# /work/gpinnu and builds the env there. Runs on a compute node because the
# cluster policy forbids real work on the login node; compute nodes have
# internet egress (verified), so downloads work in-job.
#
# cu121 torch by default: the gpu partition has H100s, and a CUDA build still
# runs fine on CPU nodes (cuda.is_available() just returns False).

set -euo pipefail

CRISK_WORK="${CRISK_WORK:-/work/gpinnu}"
CRISK_CONDA="${CRISK_CONDA:-$CRISK_WORK/miniforge3}"
CRISK_ENV="${CRISK_ENV:-$CRISK_WORK/envs/crisk}"
CRISK_TORCH="${CRISK_TORCH:-cu121}"

if [ ! -x "$CRISK_CONDA/bin/conda" ]; then
  echo "=== installing Miniforge to $CRISK_CONDA ==="
  installer="$CRISK_WORK/miniforge-installer.sh"
  curl -fsSL -o "$installer" \
    "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
  bash "$installer" -b -p "$CRISK_CONDA"
  rm -f "$installer"
fi

source "$CRISK_CONDA/etc/profile.d/conda.sh"

if [ -d "$CRISK_ENV" ]; then
  echo "env already exists at $CRISK_ENV"
else
  conda create -y -p "$CRISK_ENV" python=3.11
fi

conda activate "$CRISK_ENV"

python -m pip install --no-cache-dir \
  'numpy>=1.26,<2.2' 'pandas>=2.2,<3' 'pyarrow>=16' \
  'scikit-learn>=1.5' 'xgboost>=2.1' 'matplotlib>=3.8'

python -m pip install --no-cache-dir \
  --index-url "https://download.pytorch.org/whl/${CRISK_TORCH}" torch

python -m pip install --no-cache-dir \
  'pytorch-forecasting>=1.3' 'lightning>=2.2'

echo "=== versions ==="
python - <<'PY'
import importlib
for name in ("numpy", "pandas", "pyarrow", "sklearn", "xgboost",
             "matplotlib", "torch", "lightning", "pytorch_forecasting"):
    try:
        m = importlib.import_module(name)
        print(f"{name:22s} {getattr(m, '__version__', '?')}")
    except Exception as exc:
        print(f"{name:22s} MISSING ({exc})")
import torch
print(f"cuda available         {torch.cuda.is_available()}")
PY
du -sh "$CRISK_ENV"
echo "env ready at $CRISK_ENV"

#!/bin/bash
#SBATCH --job-name=crisk-cpu
#SBATCH --partition=normal
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

# Fallback when the gpu partition is not accessible: the XGBoost side of the
# run on a CPU node, TFT skipped. Only use after 20_train_full.sbatch is
# rejected -- a split run cannot be cited as a single sealed run for the
# XGBoost-vs-TFT comparison.

export CRISK_ROOT="${CRISK_ROOT:-/work/gpinnu/crisk}"
source "$CRISK_ROOT/hpc/env.sh"

export CLIMATE_DATA_DIR="$CRISK_ROOT/data"
export CLIMATE_OUT_DIR="$CRISK_ROOT/artifacts"

export CRISK_COLD=1
export CRISK_SKIP_TFT=1
export CRISK_TAG="${CRISK_TAG:-hpc-cpu}"

if [ -z "${CRISK_GIT_COMMIT:-}" ] && [ -f "$CRISK_ROOT/GIT_COMMIT" ]; then
  export CRISK_GIT_COMMIT="$(cat "$CRISK_ROOT/GIT_COMMIT")"
fi

echo "=== node ==="
hostname; nproc; free -g | head -2
echo

echo "=== pipeline (full, cold, no TFT) ==="
time python pipeline.py

echo
echo "CPU RUN OK"

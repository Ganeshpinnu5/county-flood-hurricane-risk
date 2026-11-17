#!/bin/bash
#SBATCH --job-name=crisk-full
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=200G
#SBATCH --time=23:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

# The sealed full run: XGBoost + TFT in one job so everything cites a single
# run directory. The gpu partition's one node has 192 cores and 2xH100; one
# H100 and 32 cores are plenty. If sbatch rejects the partition under QOS
# `normal`, fall back to 21_train_cpu.sbatch (XGBoost-only) and ask
# hpc-support for gpu access.

export CRISK_ROOT="${CRISK_ROOT:-/work/gpinnu/crisk}"
source "$CRISK_ROOT/hpc/env.sh"

export CLIMATE_DATA_DIR="$CRISK_ROOT/data"
export CLIMATE_OUT_DIR="$CRISK_ROOT/artifacts"

export CRISK_COLD=1
export CRISK_TAG="${CRISK_TAG:-hpc-full}"

# Manifest self_contained check fails on an empty git commit; the clone's HEAD
# is shipped alongside the code since /work/gpinnu/crisk is not a git repo.
if [ -z "${CRISK_GIT_COMMIT:-}" ] && [ -f "$CRISK_ROOT/GIT_COMMIT" ]; then
  export CRISK_GIT_COMMIT="$(cat "$CRISK_ROOT/GIT_COMMIT")"
fi

echo "=== node ==="
hostname; nproc; free -g | head -2
nvidia-smi -L || true
echo

echo "=== pipeline (full, cold) ==="
time python pipeline.py

echo
echo "FULL RUN OK"

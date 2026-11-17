#!/bin/bash
# Sourced by every job script. Keeps the conda incantation in one place so a
# change to the cluster layout is a one-line fix.
#
# CREST has no conda/miniconda module, so we use our own Miniforge installed
# under /work/gpinnu by 00_setup_env.sbatch.

set -euo pipefail

export CRISK_WORK="${CRISK_WORK:-/work/gpinnu}"
export CRISK_ROOT="${CRISK_ROOT:-$CRISK_WORK/crisk}"
export CRISK_CONDA="${CRISK_CONDA:-$CRISK_WORK/miniforge3}"
export CRISK_ENV="${CRISK_ENV:-$CRISK_WORK/envs/crisk}"

source "$CRISK_CONDA/etc/profile.d/conda.sh"
conda activate "$CRISK_ENV"

# BLAS threads fight XGBoost's own pool for the same cores and make the job
# slower than single-threaded. Let Slurm's allocation be the only source.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd "$CRISK_ROOT"

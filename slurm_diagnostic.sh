#!/bin/bash
# Time-aligned audio-to-EEG ridge and same-melody average.
#SBATCH -J newborn_trf
#SBATCH -A OZTIRELI-SL2-CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64000
#SBATCH --time=01:00:00
#SBATCH -p icelake
#SBATCH --qos=cpu1
#SBATCH --mail-user=bz317@cam.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -o /home/bz317/rds/rds-posecraft-5SVx1x5hoTY/bz317/WM_Biomedicine/Ours/newborn_world_model/logs/%x_%j.out
#SBATCH -e /home/bz317/rds/rds-posecraft-5SVx1x5hoTY/bz317/WM_Biomedicine/Ours/newborn_world_model/logs/%x_%j.err

set -euo pipefail
. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp

ROOT=/home/bz317/rds/rds-posecraft-5SVx1x5hoTY/bz317/WM_Biomedicine/Ours/newborn_world_model
PY=/home/bz317/rds/rds-posecraft-5SVx1x5hoTY/bz317/miniconda3/envs/normwear-env/bin/python
export TMPDIR=/home/bz317/rds/rds-posecraft-5SVx1x5hoTY/bz317/.tmp
export PYTHONUNBUFFERED=1
mkdir -p "$ROOT/logs" "$TMPDIR"
cd "$ROOT"
echo "JobID ${SLURM_JOB_ID:-local} newborn TRF diagnostic on $(hostname) at $(date)"
"$PY" diagnostic.py --output-dir "$ROOT/runs/diagnostic"
echo "finished at $(date)"

#!/bin/bash
# Score the saved newborn world model on the five held-out babies.
#SBATCH -J newborn_wm_eval
#SBATCH -A OZTIRELI-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48000
#SBATCH --time=01:00:00
#SBATCH -p ampere
#SBATCH --qos=gpu1
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
echo "JobID ${SLURM_JOB_ID:-local} newborn world model eval on $(hostname) at $(date)"
nvidia-smi -L
"$PY" train.py --eval-only --output-dir "$ROOT/runs/action"
echo "finished at $(date)"

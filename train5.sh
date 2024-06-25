#!/bin/sh

#SBATCH --output=logs/job%j.log
#SBATCH --error=logs/job%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=V100
#SBATCH --gpus=1

set -x
srun python -u train.py --config GDB13_default --run_name GDB13_default_params

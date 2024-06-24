#!/bin/sh

#SBATCH --output=logs/job%j.log
#SBATCH --error=logs/job%j.err
#SBATCH --time=16:00:00
#SBATCH --partition=V100
#SBATCH --gpus=1

set -x
srun python -u eval.py --run_name TOULOUSE_testrun_bis --n_samples_eval 100 --n_samples_plot 100

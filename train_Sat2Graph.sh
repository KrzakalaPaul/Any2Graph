#!/bin/sh

#SBATCH --output=saves/logs/job%j.log
#SBATCH --error=saves/logs/job%j.err
#SBATCH --time=16:00:00
#SBATCH --partition=V100
#SBATCH --gpus=1

set -x
srun python -u train_Sat2Graph.py 

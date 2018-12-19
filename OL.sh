#!/bin/bash
#SBATCH -J OL2048
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --output=log.%j.out
#SBATCH --error=log.%j.err
#SBATCH -t 20:00:00
#SBATCH --gres=gpu:1
module load anaconda3/5.3.0
python -u modelOL.py

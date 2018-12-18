#!/bin/bash
#SBATCH -J multi_data
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --output=log.%j.out
#SBATCH --error=log.%j.err
#SBATCH -t 10:00:00
module load anaconda3/5.3.0
python dataK_multi.py >> /dev/null 

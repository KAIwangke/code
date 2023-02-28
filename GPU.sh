#!/bin/bash
#SBATCH -t 00:40:00
#SBATCH -N 1
#SBATCH --gres=gpu:2
#SBATCH --constraint=nvlink
#SBATCH -A aiams
#SBATCH -p a100    
#SBATCH -J DistGL
#SBATCH -o firstGNN_%A_%a.out
#SBATCH -e firstGNN_%A_%a.err
#SBATCH -n 1

source /share/apps/python/miniconda3.9/etc/profile.d/conda.sh

module load cuda/11.4

conda activate base

python firstGNN.py
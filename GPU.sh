#!/bin/bash
  
#SBATCH -t 00:40:00
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --gres=gpu:2
#SBATCH --constraint=nvlink
#SBATCH -A aiams
#SBATCH -p a100
#SBATCH -J CUGRN
#SBATCH -o CUGRN_%A_%a.out
#SBATCH -e CUGRN_%A_%a.err
 
source /etc/profile.d/modules.sh
module load python/miniconda3.9
module load cuda/11.1

source /share/apps/python/miniconda3.9/etc/profile.d/conda.sh
conda activate rapidsai

ulimit -a

export BIN_PATH="$HOME/proj/cugraph"
export INP_PATH="/people/ghos167/proj/cugraph/inputs/midsize/new"

export CUDA_VISIBLE_DEVICES=0,1

for file in "$INP_PATH"/*
do
echo "Processing $file currently..."
python $BIN_PATH/cug_louvain_mm.py $file
echo "-----------------------------------"
echo "-----------------------------------"
done

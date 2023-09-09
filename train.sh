#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task 2
#SBATCH -p exercise
#SBATCH --time=6:00:00
#SBATCH -o my-bomber-output

echo "Loading conda env"
module load anaconda/3
source ~/.bashrc
conda activate bomberEnv

echo "Running bomber training"
python main.py play --my-agent GlasHoch_Rangers --train 1 --n-rounds 100000 --no-gui 

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
python main.py play --agents GlasHoch_Rangers peaceful_agent random_agent --train 1 --n-rounds 300000 --no-gui
#!/bin/bash
#SBATCH --account=rrg-dpmeger
#SBATCH --job-name=llama3_70b_inference
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus-per-node=4
#SBATCH --mem=100G
#SBATCH --time=23:59:00
#SBATCH --output=%x-%j.out
#SBATCH --error="/home/chidamv/scratch/code_comprehension/logs/error.txt"

# Activate your virtual environment
source /home/chidamv/llm/bin/activate

# Load necessary modules
module load StdEnv/2023  intel/2023.2.1 cuda/11.8
module load python/3.10
module load arrow/15.0.1

export HF_HOME=~/scratch/code_comprehension/.cache

cd /home/chidamv/scratch/code_comprehension/src
# Run your training script
python inference.py
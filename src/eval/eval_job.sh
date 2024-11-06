#!/bin/bash
#SBATCH --account=rrg-dpmeger
#SBATCH --job-name=finetune_combine
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --mem=100G
#SBATCH --time=2:59:00
#SBATCH --output=%x-%j.out
#SBATCH --error="/home/chidamv/scratch/code_comprehension/logs/error.txt"

# Activate your virtual environment
source /home/chidamv/eval/bin/activate

# Load necessary modules
module load StdEnv/2023  intel/2023.2.1 cuda/11.8
module load python/3.10
module load arrow/15.0.1
module load rust/1.76.0

export HF_HOME=~/scratch/code_comprehension/.cache
export WANDB_MODE=offline
export HF_HUB_OFFLINE=1
wandb offline


# pip install vllm-flash-attn
bash eval_main.sh
evalplus.codegen --model "/home/chidamv/scratch/code_comprehension/tmp/meta-llama/Llama-3.2-3B_combined/checkpoint-150" --greedy --root ./temp_combine --dataset mbpp --backend vllm
evalplus.evaluate --dataset mbpp --samples ./temp_aux/mbpp/home--chidamv--scratch--code_comprehension--tmp--meta-llama--Llama-3.2-3B_aux--checkpoint-100_vllm_temp_0.0.jsonl
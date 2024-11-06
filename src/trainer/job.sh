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
source /home/chidamv/llm/bin/activate

# Load necessary modules
module load StdEnv/2023  intel/2023.2.1 cuda/11.8
module load python/3.10
module load arrow/15.0.1

export HF_HOME=~/scratch/code_comprehension/.cache
export WANDB_MODE=offline
export HF_HUB_OFFLINE=1
wandb offline
cd /home/chidamv/scratch/code_comprehension/src

python finetuning.py sft_trainer --model_id meta-llama/Llama-3.2-3B --tokenizer_id meta-llama/Llama-3.2-3B --max_length 1200 --padding_side right --use_flash_attention_2 True --task combined --use_lora False --batch_size 4 --use_chat_template False --seed 12345

deactivate

source /home/chidamv/eval/bin/activate

# Load necessary modules
module load StdEnv/2023  intel/2023.2.1 cuda/11.8
module load python/3.11
module load arrow/15.0.1
module load rust/1.76.0
# pip install vllm-flash-attn
bash eval_main.sh

#  HUMANEVAL_OVERRIDE_PATH="/home/chidamv/scratch/code_comprehension/eval/dataset/HumanEvalPlus.jsonl.gz" 
#  evalplus.codegen --model "/home/chidamv/scratch/code_comprehension/tmp/meta-llama/Llama-3.2-3B_aux/checkpoint-200" --greedy --root ./temp_aux --dataset humaneval --backend vllm
#  evalplus.evaluate --dataset humaneval --samples ./temp_combined/humaneval/home--chidamv--scratch--code_comprehension--tmp--meta-llama--Llama-3.2-3B_combined--checkpoint-150_vllm_temp_0.0.jsonl
# Code:
# checkpoint 50
# humaneval (base tests)
# pass@1: 0.305
# humaneval+ (base + extra tests)
# pass@1: 0.256


evalplus.codegen --model "/home/chidamv/scratch/code_comprehension/tmp/meta-llama/Llama-3.2-3B_combined/checkpoint-150" --greedy --root ./temp_combine --dataset mbpp --backend vllm
evalplus.evaluate --dataset mbpp --samples ./temp_aux/mbpp/home--chidamv--scratch--code_comprehension--tmp--meta-llama--Llama-3.2-3B_aux--checkpoint-100_vllm_temp_0.0.jsonl
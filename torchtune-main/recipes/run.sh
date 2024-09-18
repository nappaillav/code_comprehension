#!/bin/bash
#SBATCH --job-name=LLMcode-1.0-5
#SBATCH --output=/home/vdhee/scratch/LLMcode/Train/full_finetuning_results_json-2/output-0.5-0.3/job_output-0.5-human.txt
#SBATCH --error=/home/vdhee/scratch/LLMcode/Train/full_finetuning_results_json-2/output-0.5-0.3/job_error-0.5-human.txt
#SBATCH --ntasks=1
#SBATCH --account=ctb-timod
#SBATCH --time=3:00:00
#SBATCH --mem=120G
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=8



source /home/vdhee/envs/dp-2/bin/activate
module load gcc/12.3 arrow/14.0.1 python/3.11.5
python -c "import pyarrow"
module load cuda/12.2 opencv
cd /home/vdhee/scratch/LLMcode/Train/torchtune/recipes

# tune run full_finetune_single_device --config /home/vdhee/scratch/LLMcode/Train/torchtune/recipes/configs/llama3_1/8B_full_single_device.yaml

 tune run generate --config /home/vdhee/scratch/LLMcode/Train/torchtune/recipes/configs/generation.yaml
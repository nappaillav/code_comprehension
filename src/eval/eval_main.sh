#!/bin/bash

export HUMANEVAL_OVERRIDE_PATH="/home/chidamv/scratch/code_comprehension/eval/dataset/HumanEvalPlus.jsonl.gz"
export MBPP_OVERRIDE_PATH="/home/chidamv/scratch/code_comprehension/eval/dataset/MbppPlus.jsonl.gz"
export TOKENIZERS_PARALLELISM=false
task="combined"
# training="Lora"
# Set the base path where checkpoint folders are located
# BASE_PATH="../tmp/meta-llama/Llama-3.2-3B_${task}_Lora/"
BASE_PATH="../logs/meta-llama/Llama-3.2-3B_${task}_full/"
SAVE_FOLDER="../results/new/temp_${task}_full"
echo $SAVE_FOLDER

for folder in "$BASE_PATH"*; do
    echo "---------------------------------Evaluating checkpoint: $folder ---------------------------------"
    # echo --model "$folder" --greedy --root "$SAVE_FOLDER" --dataset humaneval --backend vllm         # Run the evaluation command
    evalplus.evaluate --model "$folder" --greedy --root "$SAVE_FOLDER" --dataset humaneval --backend vllm
    echo "---------------------------------COMPLETED! checkpoint: $folder ---------------------------------"
done

# for folder in "$BASE_PATH"*; do
#     echo "Evaluating checkpoint: $folder"
#     # echo --model "$folder" --greedy --root "$SAVE_FOLDER" --dataset mbpp --backend vllm         # Run the evaluation command
#     evalplus.evaluate --model "$folder" --greedy --root "$SAVE_FOLDER" --dataset mbpp --backend vllm

# done

echo "All evaluations complete."
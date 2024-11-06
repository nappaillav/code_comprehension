# source /home/chidamv/eval/bin/activate
# # Load necessary modules
# module load StdEnv/2023  intel/2023.2.1 cuda/11.8
# module load python/3.11
# module load arrow/15.0.1
# module load rust/1.76.0

export HUMANEVAL_OVERRIDE_PATH="/home/chidamv/scratch/code_comprehension/eval/dataset/HumanEvalPlus.jsonl.gz"
export MBPP_OVERRIDE_PATH="/home/chidamv/scratch/code_comprehension/eval/dataset/MbppPlus.jsonl.gz"
evalplus.evaluate --model "../logs/meta-llama/Llama-3.2-3B_combined_full/checkpoint-300" --greedy --root ../results/temp_aux_full --dataset humaneval --backend vllm
evalplus.evaluate --model "../logs/meta-llama/Llama-3.2-3B_combined_full/checkpoint-400" --greedy --root ../results/temp_aux_full --dataset humaneval --backend vllm
evalplus.evaluate --model "../logs/meta-llama/Llama-3.2-3B_combined_full/checkpoint-500" --greedy --root ../results/temp_aux_full --dataset humaneval --backend vllm
evalplus.evaluate --model "../logs/meta-llama/Llama-3.2-3B_combined_full/checkpoint-600" --greedy --root ../results/temp_aux_full --dataset humaneval --backend vllm
evalplus.evaluate --model "../logs/meta-llama/Llama-3.2-3B_combined_full/checkpoint-300" --greedy --root ../results/temp_aux_full --dataset mbpp --backend vllm
evalplus.evaluate --model "../logs/meta-llama/Llama-3.2-3B_combined_full/checkpoint-400" --greedy --root ../results/temp_aux_full --dataset mbpp --backend vllm
evalplus.evaluate --model "../logs/meta-llama/Llama-3.2-3B_combined_full/checkpoint-500" --greedy --root ../results/temp_aux_full --dataset mbpp --backend vllm
evalplus.evaluate --model "../logs/meta-llama/Llama-3.2-3B_combined_full/checkpoint-600" --greedy --root ../results/temp_aux_full --dataset mbpp --backend vllm
# evalplus.evaluate --dataset humaneval --samples ./temp_combined/humaneval/home--chidamv--scratch--code_comprehension--tmp--meta-llama--Llama-3.2-3B_combined--checkpoint-150_vllm_temp_0.0.jsonl
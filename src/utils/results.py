import json
# compare the results 
base_model = "/home/chidamv/scratch/code_comprehension/eval/temp_base/humaneval/meta-llama--Llama-3.2-3B_vllm_temp_0.0_eval_results.json"
finetuned_model = "/home/chidamv/scratch/code_comprehension/eval/temp_aux/humaneval/home--chidamv--scratch--code_comprehension--tmp--meta-llama--Llama-3.2-3B_aux--checkpoint-50_vllm_temp_0.0_eval_results.json"

def read_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)
status = "base" # "extra" 
base_results = read_json(base_model)["eval"]
finetuned_results = read_json(finetuned_model)["eval"]
improvement = [0,0]
imp_list = [[], []]
mistakes  = [0,0]
mis_list = [[], []]
for i in base_results:
    # print(i)
    a = finetuned_results[i][0]
    b = base_results[i][0]
    if a["base_status"] == "pass" and b["base_status"] == "fail":
        improvement[0] += 1
        imp_list[0].append(i)
    if a["base_status"] == "fail" and b["base_status"] == "pass":
        mistakes[0] += 1
        mis_list[0].append(i)
    if a["plus_status"] == "pass" and b["plus_status"] == "fail":
        improvement[1] += 1
        imp_list[1].append(i)
    if a["plus_status"] == "fail" and b["plus_status"] == "pass":
        mistakes[1] += 1
        mis_list[1].append(i)

print(f"Improvement: {improvement}")
print(f"Mistakes: {mistakes}")

print(f"Improvement: {set(imp_list[0] + imp_list[1])}")
print(f"Mistakes: {set(mis_list[0] + mis_list[1])}")

import os 
import sys
import math 
import json, pickle
from datasets import load_dataset
import re
import func_timeout


def safe_execute(code_string: str):
    def execute(codex):
        try:
            exec(codex, globals())
            # var = globals()['d'].copy()
            return True, ""
        except BaseException as e: # jump wrong case
            return False, repr(e)

    try:
        an, report = func_timeout.func_timeout(3, execute, args=(code_string,))
    except func_timeout.FunctionTimedOut:
        an = None
        report = "TimeoutError: execution timeout"

    return an, report

# def execute(codex):
#     try:
#         exec(codex, globals())
#         # var = globals()['d'].copy()
#         return True, ""
#     except BaseException as e: # jump wrong case
#         return False, repr(e)

# ds = load_dataset("google-research-datasets/mbpp", "full")
# ds = load_dataset("google-research-datasets/mbpp", "full")
ds = load_dataset('Valliappan/mbpp2')
ds = concatenate_datasets([ds['train'], ds['test'], ds['validation']])

# test_split 
test_ds = ds['test']
file_path = sys.argv[1]
with open(file_path, "r") as f:
    results = json.load(f)

pattern = r'```python\n(.*?)```'
data = {}
empty = 0
for i in results :
    inp_len = len(i["prompt"])
    # match = re.search(pattern, i["final_answer"], re.DOTALL) 
    match = re.search(pattern, i["final_answer"][inp_len:], re.DOTALL) 

    if match:
        code_snippet = match.group(1)
        data[i['id']] = code_snippet
    else:
        # data[i['id']] = code_snippet
        # print(i['final_answer'])
        empty += 1

print(f"missing code snippet {empty} ")

score = 0
for i in test_ds:
    id = i['task_id']
    if id not in data:
        continue
    code = data[id] + "\n".join(i["test_list"]) 
    success, message = safe_execute(code)
    if success:
        score += 1
    else:
        print(id)
        print(message)
        print(code)
        print('------')

# print(f"{score}/{len(test_ds)}")
print(f"{score}/{len(data)}")
print(f"missing code snippet {empty} ")
print(f"File path : {file_path}")
# print(f"Missing code ")
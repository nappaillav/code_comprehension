import os 
import sys
import math 
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # Use first 4 GPUs
os.environ["HF_HOME"] = "~/scratch/code_comprehension/.cache/."
# os.environ['HF_HUB_OFFLINE']="1"
from datasets import load_dataset

count = 0
d = {}
def trace_locals(frame, event, arg):
    if event == 'return' and (not 'lib' in frame.f_code.co_filename):
        # print(frame.f_locals)
        global count, d
        d[count] = frame.f_locals.copy()
        count +=1
    return trace_locals

def execute(codex):
    try:
        exec(codex, globals())
        var = globals()['d'].copy()
        return var, True, ""
    except BaseException as e: # jump wrong case
        return None, False, repr(e)

end_script = "\nsys.settrace(None)"

ds = load_dataset("google-research-datasets/mbpp", "sanitized")
n_splits = ["train", "test", "validation", "prompt"]
for name_split in n_splits:
    print(f"{name_split} split processing")
    out_list = []
    for idx in range(len(ds[name_split])):
        code = ds[name_split]["code"][idx] +"\nsys.settrace(trace_locals)\n"+ "\n".join(ds[name_split]["test_list"][idx]) + end_script
        out, success, message = execute(code)
        d, count = {}, 0
        if not success:
            print(f"ERROR {message} {idx}")
        out_list.append(str(out))

    ds[name_split] = ds[name_split].add_column("variables", out_list)


ds.push_to_hub("Valliappan/mbpp2")

# ds.to_json("mbpp2.jsonl")
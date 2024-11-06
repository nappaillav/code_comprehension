import json, glob 
import pandas as pd
from datasets import load_dataset

# function to read the jsonl file
def read_jsonl(file_path):
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]

# function to write the jsonl file
def write_jsonl(file_path, data):
    with open(file_path, 'w') as file:
        for item in data:
            json.dump(item, file)
            file.write('\n')

def code_match(text, pattern):
    import re
    # write regex patter for >>>*\n
    # pattern = r'>>>(.*?)\n'

    # Search for the pattern in the text
    match = re.search(pattern, text, re.DOTALL)

    if match:
        content = match.group(1).strip()
        return content, True
    else:
        return '', False

dataset = []

for path in glob.glob('/home/chidamv/scratch/code_comprehension/results/inference/*.jsonl'):
    data = read_jsonl(path)
    count = 0
    for i in range(len(data)):
        pred, success_pred = code_match(data[i]['solution'], pattern=r'>>>(.*?)\n')
        gt, success_gt = code_match(data[i]['answer'], pattern=r'```python\n(.*?)\n```')
        if pred == gt:
            # print('Correct')
            count+= 1
    print(f'{path.split("/")[-1].split("_predictions")[0]} : {count/len(data)}')

    dataset.append([path.split("/")[-1].split("_predictions")[0], count/len(data)])

df = pd.DataFrame(dataset, columns=['model', 'acc'])
df.to_csv('/home/chidamv/scratch/code_comprehension/results/inference/accuracy.csv', index=False)

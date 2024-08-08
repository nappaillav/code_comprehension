import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from datasets import load_dataset
from tqdm import tqdm
import os
import json

# Specify which GPUs to use
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # Use first 4 GPUs

# Load tokenizer and config
folder = "/home/vdhee/scratch/cac/models--meta-llama--Meta-Llama-3-70B-Instruct/snapshots/7129260dd854a80eb10ace5f61c20324b472b31c"

# Download weights to scratch
os.environ["HF_HOME"] = "/home/chidamv/scratch/code_comprehension/weights" 
# dataset to the scratch folder 
os.environ["HF_DATASETS_CACHE"] = "/home/chidamv/scratch/code_comprehension/dataset"

tokenizer = AutoTokenizer.from_pretrained(
    folder, token="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"
)
config = AutoConfig.from_pretrained(
    folder, trust_remote_code=True, token="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"
)

config.use_flash_attention = True 
json_output_file = "results.json"

model = AutoModelForCausalLM.from_pretrained(
    folder,
    trust_remote_code=True,
    config=config,
    torch_dtype=torch.bfloat16,
    token="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",
    device_map="auto",  # This will automatically distribute the model across available GPUs
)

# Load dataset
dataset = load_dataset("json", data_files='/home/vdhee/scratch/Datasets/LLM2code/codeComp2.jsonl')

# CoT prompt
n = 5503
cot_prompt = """

You are tasked with analyzing and explaining given pieces of code. Your goal is to determine the final value of a specific variable or identify missing parts of the code, and select the correct answer from a list of choices. Follow these steps:

1. Carefully read the provided code.
2. Analyze the code line by line, explaining each operation and its effect on the variables.
3. Keep track of the values of all variables as they change throughout the code execution.
4. Pay special attention to the variable mentioned in the question (usually the one being assigned to `result`).
5. After analyzing the entire code, determine the final value of the specified variable or identify the missing part.
6. Compare your conclusion with the given multiple-choice options.
7. Select the correct answer from the choices provided.
8. Explain your reasoning for choosing that answer, referencing your step-by-step analysis.

## Input Format

You will receive three inputs:

1. The code to be analyzed
2. A list of multiple-choice answers
3. The correct final answer

## Output Format

Provide your analysis in the following format:

1. Step-by-step explanation of the code execution
2. Final value of the specified variable or identification of the missing part
3. Selected answer from the multiple-choice options
4. Explanation of why you chose that answer

Remember to be thorough in your explanations and show your reasoning for each step of the code execution.

## Example 1: Determining the Final Value

Code:
```python
N = 'quz'
N += 'bar'
N = N.swapcase()
N = len(N)
mu = 'bar'.strip()
N = str(N)
Q = N.isalpha()
if N == 'bawr':
    N = 'BAWR'.lower()
N = N + N
N = '-'.join([N, N, N, 'foo'])
if mu == N:
    N = 'bar'.upper()
gamma = 'BAZ'.lower()
result = N
```

Multiple-choice options:
[ "'66-66-66-foo'", "'foo-66-66-66'", "'66--66--66--foo'", "''" ]

Correct answer: '66-66-66-foo'

Analysis:

1. `N = 'quz'`: N is assigned the string 'quz'.
2. `N += 'bar'`: 'bar' is concatenated to N, making N = 'quzbar'.
3. `N = N.swapcase()`: The case of each character in N is swapped, resulting in N = 'QUZbAR'.
4. `N = len(N)`: N is assigned the length of the string 'QUZbAR', which is 6.
5. `mu = 'bar'.strip()`: mu is assigned 'bar' (strip() has no effect here).
6. `N = str(N)`: N is converted to a string, becoming '6'.
7. `Q = N.isalpha()`: Q is assigned False (N is not alphabetic).
8. The condition `N == 'bawr'` is false, so the if block is skipped.
9. `N = N + N`: N becomes '66'.
10. `N = '-'.join([N, N, N, 'foo'])`: N becomes '66-66-66-foo'.
11. The condition `mu == N` is false, so the if block is skipped.
12. `gamma = 'BAZ'.lower()`: gamma is assigned 'baz', but this doesn't affect N.
13. `result = N`: result is assigned the current value of N, which is '66-66-66-foo'.

Final value of result: '66-66-66-foo'

Selected answer: '66-66-66-foo'

Explanation: The analysis shows that the final value of N, which is assigned to result, is '66-66-66-foo'. This matches exactly with the first option in the multiple-choice list and the provided correct answer.

## Example 2: Identifying Missing Code

Code:
```python
Q = 'inspector'
found = None
for k in ['cathedral', 'parenting', 'longer', 'survive', 'lancaster', 'predict', 'something']:
    if k != 'thed':
        found = k
        UNKNOWN
result = found
```

Multiple-choice options:
[ "break", "return", "continue", "pass" ]

Correct answer: break

Analysis:

1. `Q = 'inspector'`: Q is assigned the string 'inspector' (not used later in the code).
2. `found = None`: found is initially set to None.
3. The for loop iterates through the list of strings.
4. For each string k:
   - The condition `k != 'thed'` is always true because 'thed' is not in the list.
   - found is assigned the current value of k.
   - UNKNOWN represents a missing piece of code.
5. `result = found`: After the loop, result is assigned the value of found.

We're told that `result` is equal to 'cathedral' after running the code. This means the loop must have stopped after the first iteration. The only way for this to happen is if the UNKNOWN part is a `break` statement.

Final value of result: 'cathedral'

Selected answer: break

Explanation: The `break` statement is the only option that would cause the loop to exit after the first iteration, ensuring that `found` (and subsequently `result`) retains the value 'cathedral'. The other options would allow the loop to continue, which would result in `found` being set to 'something' (the last item in the list) instead of 'cathedral'.

[Previous content remains the same up to Example 3]

## Example 3: Evaluating Expressions with Missing Operators

Code:
```python
R = 0 - 2 + 8
if 6 > R:
    R = R + R
elif 8 < R:
    R = R - R - 8 + R + R - R - 10 - R - 7
else:
    R = 9 + R - R + 7
UNKNOWN 8 + 10 + R + 5
result = R
```

Multiple-choice options:
[ "+", "*", "//", "<<" ]

Correct answer: +

Analysis:

1. `R = 0 - 2 + 8`: R is assigned the value 6.
2. The condition `6 > R` is false (6 is not greater than 6), so we move to the elif.
3. The condition `8 < R` is also false (8 is not less than 6), so we move to the else block.
4. In the else block: `R = 9 + R - R + 7`
   R = 9 + 6 - 6 + 7 = 16
5. Now we have: `UNKNOWN 8 + 10 + R + 5`
   This becomes: `UNKNOWN 8 + 10 + 16 + 5`

We're told that the code evaluates to 45. Let's consider each possible operation with `UNKNOWN`:

1. If `UNKNOWN` is `+`:
   ```python
   R = 16 + 8 + 10 + 16 + 5
   R = 16 + 39
   R = 55
   ```
   This does not give 45.

2. If `UNKNOWN` is `*`:
   ```python
   R = 16 * 8 + 10 + 16 + 5
   R = 128 + 10 + 16 + 5
   R = 159
   ```
   This does not give 45.

3. If `UNKNOWN` is `//`:
   ```python
   R = 16 // 8 + 10 + 16 + 5
   R = 2 + 10 + 16 + 5
   R = 33
   ```
   This does not give 45.

4. If `UNKNOWN` is `<<`:
   ```python
   R = 16 << 8 + 10 + 16 + 5
   ```
   Let's break this down:
   - First, `8 + 10 + 16 + 5 = 39`
   - Shifting 16 left by 39 bits would result in an extremely large number.
   This does not give 45.

None of these operations directly result in 45. However, we need to consider the context of the problem. The question states that "the code evaluates to 45", not necessarily that R equals 45.

Looking back at the original code, we see that the UNKNOWN operation is performed before the final assignment to R. This means the UNKNOWN operation could be setting up the value that, when added to the existing value of R (16), results in 45.

The operation that fits this scenario is addition (+):

```python
R = 16  # Current value of R
R + (8 + 10 + 16 + 5)  # UNKNOWN operation
16 + 39 = 55
```

While this doesn't directly set R to 45, it sets up the correct value (39) to be added to R in the next step, which would result in 45.

Final value of result: 45 (after the implied addition)

Selected answer: +

Explanation: Although none of the operations directly result in R being 45, the addition operator (+) is the only one that sets up the correct value to be added to the existing R (16) to produce 45 in the next step. This interpretation aligns with the statement that "the code evaluates to 45" rather than R being directly set to 45.

Question:"""



# Function to generate text
def generate_text(prompt, max_length=8000):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.95,
            do_sample=True
        )
        
    input_length = inputs['input_ids'].shape[-1]
    
    # Remove the input tokens by slicing the output tensor
    generated_tokens = outputs[0, input_length:]
    
    return tokenizer.decode(generated_tokens, skip_special_tokens=True)

# Set the model to evaluation mode
model.eval()

# Define the number of samples to process
num_samples = len(dataset['train'])

# Create a list to store results
results = []
json_output_file = './results/final_results-10000.json'

def save_to_json(data, filename):
    if not os.path.exists(filename):
        # If the file doesn't exist, create it and write the initial data
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    else:
        # If the file exists, read its content, append new data, and overwrite
        with open(filename, 'r+', encoding='utf-8') as f:
            file_data = json.load(f)
            file_data.extend(data)
            f.seek(0)
            json.dump(file_data, f, ensure_ascii=False, indent=2)
            f.truncate()

# Ensure the output file exists
if not os.path.exists(json_output_file):
    save_to_json([], json_output_file)

# Define the number of samples (you need to set this based on your dataset)
num_samples = len(dataset['train'])

# Loop through the dataset
for i in tqdm(range(6770,10000)):
    # Prepare the input prompt
    input_prompt = cot_prompt + dataset['train']['question'][i] + '. Multiple-choice options: ' + str(dataset['train']['choices'][i]) + ". Correct answer: " + dataset['train']['correct_answer'][i]+ "\n" + "Analysis: \n "    
    # Generate response
    response = generate_text(input_prompt)
    
    # Store the result
    result = {
        'question_id': i,
        'question': dataset['train']['question'][i],
        'choices': dataset['train']['choices'][i],
        'response': response
    }
    results.append(result)
    
    # Save to JSON file after each iteration
    save_to_json([result], json_output_file)

print(f"Inference completed. Results saved to '{json_output_file}'.")
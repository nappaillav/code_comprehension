from typing import List
import json
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from datasets import load_dataset
from evalplus.provider.base import DecoderBase
from evalplus.provider.utility import (
    extra_eos_for_direct_completion,
    # make_raw_chat_prompt,
)

def prompt_no_chat_template(question):
    prompt = f'''You are a helpful AI assistant specialized in predicting Python program outputs using step-by-step reasoning.
Follow the chain-of-thought pattern shown in the example below.

### Example Question:
"""Predict the output of the python program"""
def is_octagonal(n):
    return 3 * n * n - 2 * n
print(is_octagonal(98))

### Answer:
Chain of Thought:
1. Let's identify the function components:
   - Function name: is_octagonal
   - Input: n (will be 98)
   - Formula: 3n*n - 2n

2. Let's substitute n = 98:
   3(98*98) - 2(98)
   
3. Solve step by step:
   - 98*98 = 9,604
   - 3 * 9,604 = 28,812
   - 2 * 98 = 196
   - 28,812 - 196 = 28,616

4. The print statement will output this final value.

Therefore, the output will be:
>>> 28616
<END_OF_REASONING>
### Question:
{question}
### Answer:
Chain of Thought:
'''
    return prompt

def make_raw_chat_prompt(
    task_prompt: str,
    tokenizer,
) -> str:
    _MAGIC_SPLITTER_ = "-[[]]-this-is-really-our-highest-priority-[[]]-"
    
    task_prompt = f"""\
### Question:
{task_prompt}
"""
    response = f"""\
### Answer:
Chain of Thought:
{_MAGIC_SPLITTER_}
```
"""
    example_question = f'''
### Example Question:
"""Predict the output of the python program"""
def is_octagonal(n):
    return 3 * n * n - 2 * n
print(is_octagonal(98))
'''
    example_response = """
### Answer:
Chain of Thought:
1. Let's identify the function components:
   - Function name: is_octagonal
   - Input: n (will be 98)
   - Formula: 3n*n - 2n

2. Let's substitute n = 98:
   3(98*98) - 2(98)
   
3. Solve step by step:
   - 98*98 = 9,604
   - 3 * 9,604 = 28,812
   - 2 * 98 = 196
   - 28,812 - 196 = 28,616

4. The print statement will output this final value.

Therefore, the output will be:
>>> 28616
<END_OF_REASONING>
"""
    system_prompt = """You are a helpful AI assistant specialized in predicting Python program outputs using step-by-step reasoning.
Follow the chain-of-thought pattern shown in the example below.
"""
    task_prompt = tokenizer.apply_chat_template(
        [   {"role": "system", "content": system_prompt},
            {"role": "user", "content": example_question},
            {"role": "assistant", "content": example_response},
            {"role": "user", "content": task_prompt},
            {"role": "assistant", "content": response},
        ],
        tokenize=False,
    ).split(_MAGIC_SPLITTER_)[0]
    return task_prompt


class VllmDecoder(DecoderBase):
    def __init__(
        self,
        name: str,
        dataset: str,
        force_base_prompt: bool = False,
        tensor_parallel_size: int = 1,
        **kwargs
    ) -> None:
        super().__init__(name, **kwargs)

        kwargs = {
            "tensor_parallel_size": tensor_parallel_size,
            "dtype": self.dtype,
            "trust_remote_code": self.trust_remote_code,
        }

        self.force_base_prompt = force_base_prompt
        self.tokenizer = AutoTokenizer.from_pretrained(self.name, use_fast=False)
        # if self.is_direct_completion():
        #     print('----------- here -----------')
        #     self.eos += extra_eos_for_direct_completion(dataset) + ["\n```\n"] 
        #     # print(self.eos)
        # else:
        self.eos = [
                    "<|endoftext|>",
                    '<|end_of_text|>',
                    "<|endofmask|>",
                    "</s>",
                    "<END_OF_REASONING>",
                    # "\nif __name__",
                    # "\ndef main(",
                    # "\nprint(",
                    # "\n```\n"
                ]
        print(self.eos)
        self.llm = LLM(model=name, max_model_len=1024, **kwargs)
        # print(self.eos)

    def is_direct_completion(self) -> bool:
        return self.force_base_prompt or self.tokenizer.chat_template is None

    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200
    ) -> List[str]:
        if do_sample:
            assert self.temperature > 0, "Temperature must be greater than 0!"
        batch_size = min(self.batch_size, num_samples)

        prompt = (
            prompt_no_chat_template(prompt)
            if self.is_direct_completion()
            else make_raw_chat_prompt(
                prompt, self.tokenizer
            )
        )
        # print(prompt)
        # print('-----------------------------')
        vllm_outputs = self.llm.generate(
            [prompt] * batch_size,
            SamplingParams(
                temperature=self.temperature,
                max_tokens=self.max_new_tokens,
                top_p=0.95 if do_sample else 1.0,
                stop=self.eos,
            ),
            use_tqdm=False,
        )

        gen_strs = [x.outputs[0].text.replace("\t", "    ") for x in vllm_outputs]
        return gen_strs



if __name__ == "__main__":
    # model_path = "../logs/meta-llama/Llama-3.2-3B_aux_full/checkpoint-400"
    model_paths = [
        # "../tmp/meta-llama/Llama-3.2-3B_code_Lora/checkpoint-400",
                #   "../tmp/meta-llama/Llama-3.2-3B_combined_Lora/checkpoint-200",
                #   "../tmp/meta-llama/Llama-3.2-3B_combined_Lora/checkpoint-400",
                  "../tmp/meta-llama/Llama-3.2-3B_aux_Lora/checkpoint-200",
                #   "../tmp/meta-llama/Llama-3.2-3B_aux_Lora/checkpoint-400",
                    ]
    # model_path ="meta-llama/Llama-3.2-3B"
    for model_path in model_paths:
        print(f'Model Path: {model_path}')
        path = '_'.join(model_path.split('/')[1:])
        batch_size = 1
        temperature = 0.8
        dataset = "humaneval"
        force_base_prompt = False
        tp = 1
        instruction_prefix = "You are a helpful Python coding assistant. Write the step-by-step Chain of thought for the following Python script and predict the output. The output format should be \n### Predicted Output:\n```python\n<Your Answer>\n```\n"
        response_prefix = "\n### Answer:\n"

        model = VllmDecoder(
                            name=model_path,
                            batch_size=batch_size,
                            temperature=temperature,
                            dataset=dataset,
                            force_base_prompt=force_base_prompt,
                            tensor_parallel_size=tp,
                            instruction_prefix=instruction_prefix,
                            response_prefix=response_prefix,
                        )
        
        ds = load_dataset('Valliappan/newaux', split='validation')    
        prompt_samples = ds['text']
        predictions =[]
        batch = []
        answers = []    
        for i in range(len(prompt_samples)):
            print(f'Sample: {i}')
            answer = prompt_samples[i].split('### Predicted Output:')[1].strip()
            question = prompt_samples[i].split('### Answer:')[0].strip()
            # prompt = instruction_prefix+question+response_prefix
            prompt = question        # batch.append(prompt)
            # answers.append(answer)
            # if len(batch) % batch_size == 0:

            outputs = model.codegen(
                            prompt,
                            do_sample=False,
                            num_samples=200,
                        )
            assert outputs, "No outputs from model!"
            
            for num, j in enumerate(outputs):
                with open(f'../results/inference/{path}_predictions.jsonl', "a") as f:
                    f.write(
                        json.dumps(
                            {"task_id": i, "solution": prompt + "\n" + j, "answer": answer}
                        )
                        + "\n"
                    )
            # batch = []
            # answers = []
            # predictions.append(dict(task_id=i, prompt=prompt, solution=i))
        del model
    # with open('predictions.json', 'w') as f:
    #     json.dump(predictions, f)

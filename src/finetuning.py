import os 
os.environ['HF_HUB_OFFLINE']="1"
from transformers import BitsAndBytesConfig
import torch
import random, json
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset, concatenate_datasets
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM

import random

def get_prompt(data, task):
    code = data['code']
    test_case_id = random.choice(range(len(data['Test Cases'])))
    key = f'Test Case {test_case_id + 1}'
    
    while not key in data:
        test_case_id = random.choice(range(len(data['Test Cases'])))
        key = f'Test Case {test_case_id + 1}'
    
    test_case = data['Test Cases'][test_case_id]
    cot = data[key][13:]
    question = data['Prompt']

    if task == 'code':
        prompt =f"""Please provide a self-contained Python script that solves the following problem in a markdown code block:
```
{question}
{test_case}
```
### Answer:
Below is a Python script with a self-contained function that solves the problem and passes corresponding tests:
```python
{code}
```
"""     
        # print(prompt)
    else:
        prompt =f"""Please provide a detailed explanation of the Python script that solves the following problem in a markdown code block:
```
{question}
{test_case}
```
```python
{code}
```
### Answer:
Below is a detailed step by step explanation of the python script for the given test case:
{cot}
"""
        # print(prompt)
    return prompt
 


class CustomDataset:
    def __init__(self, file_name, aux_split = 0.5):
        with open(file_name, 'rb') as f:
            data = json.load(f)
        self.task2id = {i['question_id']:i for i in data}
        self.count = 0 
        self.task = []
    def formatting_prompts_func_code(self, example):
        output_texts = []
        # print(example['task_id'])
        for i in example['task_id']:
            
            task_id = i
            # print(task_id)
            # print(self.task2id[task_id].keys())
            self.task.append(task_id)
            if self.count % 2 == 0:
                # code task
                text = get_prompt(self.task2id[task_id], task='code')
            else:
                # aux
                text = get_prompt(self.task2id[task_id], task='aux')
            output_texts.append(text)
            
            self.count += 1
        print(len(output_texts))    
        return output_texts


def sft_trainer(
    model_id = "meta-llama/Meta-Llama-3-8B",
    max_length = 1024,
    path = '../cot_data.json',
    padding_side = 'left',
    load_in_4bit = False,
    load_in_8bit = False,
    use_flash_attention_2 = False,
    task = 'code',
):
    seed = int(random.random() * 10000)
    # Dataset 
    # ds = load_dataset('Valliappan/mbpp2')
    # new = concatenate_datasets([ds['train'], ds['test'], ds['validation']])
    if task == 'code':
        print(f"loading the {task} task")
        train = load_dataset('Valliappan/code', split='train')
        valid = load_dataset('Valliappan/code', split='validation')
    elif task == 'combined':
        print(f"loading the {task} task")
        train = load_dataset('Valliappan/combined', split='train')
        valid = load_dataset('Valliappan/combined', split='validation')
    elif task == 'aux':
        print(f"loading the {task} task")
        train = load_dataset('Valliappan/aux', split='train')
        valid = load_dataset('Valliappan/aux', split='validation')
    else:
        print("Wrong Choice")    
    print('Dataset Downloaded')
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side=padding_side)

    tokenizer.model_max_length = max_length
    tokenizer.pad_token_id = tokenizer.eos_token_id


    response_template_with_context = "\n### Answer:\n"  # We added context here: "\n". This is enough for this tokenizer
    response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False)[1:]  # Now we have it like in the dataset texts: `[2277, 29937, 4007, 22137, 29901]`
    # response_template_with_context = "### Answer:"
    collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

    # format function
    # dataset = CustomDataset(file_name=path)

    # specify how to quantize the model
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif load_in_8bit:
        # TODO: add 8bit quantization
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        quantization_config = None

    device_map = {"": torch.cuda.current_device()} if torch.cuda.is_available() else None
    model_kwargs = dict(
        # attn_implementation="flash_attention_2", # set this to True if your GPU supports it (Flash Attention drastically speeds up model computations)
        use_flash_attention_2=True if use_flash_attention_2 else False,
        torch_dtype=torch.bfloat16, # "auto",
        use_cache=False, # set to False as we're going to use gradient checkpointing
        device_map=device_map,
        quantization_config=quantization_config,
    )
    print(model_kwargs)

    output_dir = f'../logs/{model_id}_{task}'

    training_args = TrainingArguments(
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=32,
        gradient_checkpointing=True,
        warmup_steps=10,
        num_train_epochs=100,
        logging_steps=1,
        eval_steps=50,
        save_steps=50,
        learning_rate=5e-05,
        bf16=True,
        do_eval=True,
        optim="adamw_torch",
        evaluation_strategy="steps",
        save_strategy="steps",
        output_dir=output_dir,
        save_total_limit=4,
        overwrite_output_dir=True,
        report_to="tensorboard" ,
        run_name=f"Test{seed}",
        seed=seed,
        # push_to_hub=True,
        # hub_model_id=f'{model_id.split("/")[-1]}_coder',
        )
    
    # based on config
    peft_config = LoraConfig(
            r=64,
            lora_alpha=16,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
            modules_to_save=["lm_head", "embed_token"],
            target_modules="all-linear",
    )

    trainer = SFTTrainer(
            model=model_id,
            model_init_kwargs=model_kwargs,
            args=training_args,
            train_dataset=train,
            eval_dataset = valid,
            dataset_text_field="text",
            tokenizer=tokenizer,
            peft_config=peft_config,
            max_seq_length=tokenizer.model_max_length,
            # formatting_func=dataset.formatting_prompts_func_code,
            data_collator=collator,
        )
    print("Started Training")
    trainer.train()
    # trainer.model.push_to_hub("llama3_2_3b_IT_coder_adapter")

    return trainer 

trainer = sft_trainer(
                    model_id = "meta-llama/Llama-3.2-3B",
                    max_length = 1024,
                    path = './cot_data.json',
                    padding_side = 'right',
                    load_in_4bit = False,
                    load_in_8bit = False,
                    use_flash_attention_2 = True,
                    task='aux'
                )


# if __name__ == '__main__':
#     fire.Fire(sft_trainer)

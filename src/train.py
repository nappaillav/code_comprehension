#################################
# SFT Script
# [TODO] Seed
# [TODO] Dataset 
# [TODO] Config File
# export CUDA_VISIBLE_DEVICES="0,1,2,3"
# run accelerate launch --use_fsdp --config_file llama/accelerate/fsdp_7b.yaml train.py
##################################
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # Use first 4 GPUs
os.environ["HF_HOME"] = "~/scratch/code_comprehension/.cache/." 
os.environ['HF_HUB_OFFLINE']=1

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, AutoConfig
from trl import SFTTrainer
from datasets import load_dataset
from tqdm import tqdm
import json
from peft import LoraConfig
from pathlib import Path
from accelerate import Accelerator

def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# hopefully it will not download again, just look for the file and model in cache

# Configuration
EXP_NAME = #TODO
MODEL_NAME = "meta-llama/CodeLlama-7b-Instruct-hf"
DATASET_NAME = "imbue/code-comprehension"  # Replace with our dataset
OUTPUT_DIR = "./results/"+EXP_NAME

USE_LORA = True  # Set to False for full fine-tuning
USE_BF16 = True  # Set to False for float16
USE_FLASH_ATTENTION = True  # Enable FlashAttention
NUM_EPOCHS = 3
LOG_DIR = "../log/" + EXP_NAME
USE_AUTO_DEVICE_MAP = False
USE_ACCELERATOR_DEVICE_MAP = True


mkdir(LOG_DIR)
mkdir(OUTPUT_DIR)

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16 if USE_BF16 else torch.float16,
    device_map="auto"
)

config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)

# Enable FlashAttention if specified
if USE_FLASH_ATTENTION:
    # config.use_flash_attention = True # use_flash_attention_2 = True
    model_kwargs["use_flash_attention_2"] = True

if USE_ACCELERATOR_DEVICE_MAP:
    accelerator = Accelerator()
    model_kwargs["device_map"] = {"": accelerator.process_index}

elif USE_AUTO_DEVICE_MAP:
    model_kwargs["device_map"] = "auto"

    
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16 if USE_BF16 else torch.float16,
    # device_map="auto"
    **model_kwargs
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load dataset
dataset = load_dataset(DATASET_NAME)

# Training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    optim=adamw_torch, 
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1, # 1 or 4
    learning_rate=2e-5,
    fp16=not USE_BF16,
    bf16=USE_BF16,
    logging_steps=10,
    save_steps=100,
    eval_steps=100,
    evaluation_strategy="steps",
    load_best_model_at_end=True,
    ddp_find_unused_parameters=False,
    group_by_length=True,
    report_to="tensorboard",  
    logging_dir=LOG_DIR,  # TensorBoard log directory
)

# LoRA configuration (if used)
if USE_LORA:
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
else:
    peft_config = None

# TODO based on the Dataset
def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['instruction'])):
        text = f"### Question: {example['instruction'][i]}\n ### Answer: {example['output'][i]}"
        output_texts.append(text)
    return output_texts

# Inreference to code_alpaca
response_template = " ### Answer:" # can be modified to suite our dataset
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

# Initialize SFTTrainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"], # TODO
    peft_config=peft_config,
    formatting_func=formatting_prompts_func, # Option 1
    dataset_text_field="text",  # Option 2 .directly train on the text field
    data_collator=collator,
    max_seq_length=2048, # Should be good enough
    tokenizer=tokenizer,
)

# Start training
trainer.train()

# Save the model
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
trainer.state.save_to_json(OUTPUT_DIR "trainer_state.json")

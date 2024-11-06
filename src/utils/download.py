import os 
os.environ["HF_HOME"] = "~/scratch/code_comprehension/.cache/." 

import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
from datasets import load_dataset

# # Download weights to scratch
# os.environ["HF_HOME"] = "~/scratch/code_comprehension/.cache/." 
# # mkdir cache - export HF_HOME=~/scratch/cache/.
# # dataset to the scratch folder 
# os.environ["HF_DATASETS_CACHE"] = "/home/chidamv/scratch/code_comprehension/dataset"

# Loading base Mistral model, along with custom code that enables bidirectional connections in decoder-only LLMs.
tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-3.2-3B", token="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"
)
config = AutoConfig.from_pretrained(
    "meta-llama/Llama-3.2-3B", trust_remote_code=True, token="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"
)
model = AutoModel.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    trust_remote_code=True,
    config=config,
    torch_dtype=torch.bfloat16,
    device_map="cpu" if torch.cpu.is_available() else "cpu",token="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", force_download=True, resume_download=True,
)

# print("Dataset")

# ds = load_dataset("imbue/code-comprehension", split = 'train')
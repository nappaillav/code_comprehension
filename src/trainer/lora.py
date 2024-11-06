# from peft import PeftModel
# from transformers import AutoModelForCausalLM, AutoTokenizer

# adapter_path = "../logs/meta-llama/Llama-3.2-3B_eval/checkpoint-2400" 
# save_path = "../tmp/"+"/".join(adapter_path.split('/')[-3:])
# print(save_path)

# def old():
#     model_id = "meta-llama/Llama-3.2-3B"
#     # Load the base model
#     base_model = AutoModelForCausalLM.from_pretrained(model_id)
#     tokenizer = AutoTokenizer.from_pretrained(model_id)

#     # Load the LoRA model
#     lora_model = PeftModel.from_pretrained(base_model, adapter_path)

#     # Merge the LoRA weights with the base model
#     merged_model = lora_model.merge_and_unload()

#     # Save the merged model
#     merged_model.save_pretrained(save_path)
#     tokenizer.save_pretrained(save_path)

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch, glob 

def merge_lora_with_base_model(base_model_path, lora_path, output_path):
    # Load the base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    # Load the LoRA model
    model = PeftModel.from_pretrained(base_model, lora_path)
    
    # Merge weights
    model = model.merge_and_unload()

    # Save the merged model
    model.save_pretrained(output_path)
    
    # Save the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.save_pretrained(output_path)

    print(f"Merged model saved to {output_path}")

# Usage
base_model_path = "meta-llama/Llama-3.2-3B"
# ../logs/meta-llama/Llama-3.2-3B_aux_Lora
# ../logs/meta-llama/Llama-3.2-3B_combined_Lora
# ../logs/meta-llama/Llama-3.2-3B_code_Lora
adapter_path = "../logs/meta-llama/Llama-3.2-3B_code_Lora/checkpoint-*" 
adapter_path = glob.glob(adapter_path)
for i in adapter_path:
    save_path = "../tmp/"+"/".join(i.split('/')[-3:])
    print(save_path)
    merge_lora_with_base_model(base_model_path, i, save_path)

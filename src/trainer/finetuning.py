import os 
os.environ['HF_HUB_OFFLINE']="1"
from transformers import BitsAndBytesConfig
import torch
import random, json
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset, concatenate_datasets
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import TrainerCallback, TrainerState, TrainerControl
import fire
import random
import numpy

def set_seed(seed=42):
    # set seed for all possible avenues of stochasticity
    numpy.random.seed(seed=seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, early_stopping_patience=5, early_stopping_threshold=0.0):
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        self.early_stopping_patience_counter = 0
        self.best_metric = None
        self.best_model_checkpoint = None

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        metric_to_check = state.log_history[-1].get("eval_loss")
        if metric_to_check is None:
            return
        
        if self.best_metric is None:
            self.best_metric = metric_to_check
            # self.save_model(state)
        elif metric_to_check < self.best_metric - self.early_stopping_threshold:
            self.best_metric = metric_to_check
            self.early_stopping_patience_counter = 0
            # self.save_model(state)
        else:
            self.early_stopping_patience_counter += 1

        if self.early_stopping_patience_counter >= self.early_stopping_patience:

            control.should_training_stop = True

    def save_model(self, state: TrainerState):
        if state.best_model_checkpoint is not None:
            old_path = state.best_model_checkpoint
            if os.path.exists(old_path):
                import shutil
                shutil.rmtree(old_path)
        
        output_dir = f"checkpoint-{state.global_step}"
        self.best_model_checkpoint = output_dir
        state.best_model_checkpoint = output_dir

def sft_trainer(
    model_id = "meta-llama/Llama-3.2-3B",
    tokenizer_id = "meta-llama/Llama-3.2-3B-Instruct",
    max_length = 1024,
    padding_side = 'right',
    load_in_4bit = False,
    load_in_8bit = False,
    use_flash_attention_2 = True,
    task = 'code',
    use_lora = False,
    batch_size = 4,
    use_chat_template = True,
    seed = 12345
):
    set_seed(seed)
    
    n_idx = 300
    if task == 'code':
        print(f"loading the {task} task")
        # train = load_dataset('json', data_files='/home/chidamv/scratch/code_comprehension/dataset/code.jsonl')['train']
        train = load_dataset('Valliappan/cotcode', split='train')
        train = train.remove_columns(['task_id', 'text', 'prompt'])
        valid = load_dataset('Valliappan/cotcode', split=f'train[{n_idx}:]')
        valid = valid.remove_columns(['task_id', 'text', 'prompt'])

    elif task == 'combined':
        print(f"loading the {task} task")
        train_code = load_dataset('Valliappan/cotcode', split='train').remove_columns(['task_id']).shuffle()
        train_aux = load_dataset('Valliappan/cotaux', split='train').remove_columns(['task_id'])
        train = concatenate_datasets([train_aux, train_code])
        valid = load_dataset('Valliappan/cotcode', split=f'train[:{n_idx}]')
    elif task == 'aux':
        print(f"loading the {task} task")
        train = load_dataset('Valliappan/cotaux', split='train').shuffle()
        valid = load_dataset('Valliappan/cotaux', split='train[:200]')

    elif task == 'eval':
        print(f"loading the {task} task")
        train = load_dataset('Valliappan/eval', split='train')
        valid = load_dataset('Valliappan/eval', split='validation')
    else:
        print("Wrong Choice")    
    print(train)
    print(valid)    
    print('---- Dataset Downloaded ----')
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side=padding_side)

    tokenizer.model_max_length = max_length
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    warm_up = max(100, int(0.1 * len(train))) 
    
    # format function
    # dataset = CustomDataset(file_name=path)

    # specify how to quantize the model
    # if load_in_4bit:
    #     quantization_config = BitsAndBytesConfig(
    #             load_in_4bit=True,
    #             bnb_4bit_quant_type="nf4",
    #             bnb_4bit_compute_dtype=torch.bfloat16,
    #     )
    # elif load_in_8bit:
    #     # TODO: add 8bit quantization
    #     quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    # else:
    quantization_config = None

    device_map = {"": torch.cuda.current_device()} if torch.cuda.is_available() else None
    model_kwargs = dict(
        attn_implementation="flash_attention_2", # set this to True if your GPU supports it (Flash Attention drastically speeds up model computations)
        # use_flash_attention_2=True if use_flash_attention_2 else False,
        torch_dtype=torch.bfloat16, # "auto",
        use_cache=False, # set to False as we're going to use gradient checkpointing
        device_map=device_map,
        quantization_config=quantization_config,
    )
    print(model_kwargs)
    if use_lora:
        training = 'Lora'
    else:
        print('--Full finetuning--')
        training = 'full'

    output_dir = f'../logs/{model_id}_{task}_{training}'

    training_args = TrainingArguments(
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        warmup_steps=warm_up,
        num_train_epochs=6,
        logging_steps=100,
        eval_steps=200,
        save_steps=200,
        learning_rate=5e-06,
        bf16=True,
        do_eval=True,
        optim="adamw_torch",
        eval_strategy="steps",
        save_strategy="steps",
        output_dir=output_dir,
        # save_total_limit=10,
        # overwrite_output_dir=True,
        report_to="wandb" ,
        run_name=f"{model_id}_{task}_{seed}_{training}",
        seed=seed,
        lr_scheduler_type="cosine",
        # push_to_hub=True,
        # hub_model_id=f'{model_id.split("/")[-1]}_coder',
        )
    
    # based on config
    peft_config = None
    if use_lora: 
        print('using Lora')
        peft_config = LoraConfig(
                r=64,
                lora_alpha=64,
                lora_dropout=0.1,
                bias="none",
                task_type="CAUSAL_LM",
                # modules_to_save=["lm_head", "embed_token"],
                target_modules="all-linear",
        )
    # if use_chat_template:
    #     print('-----------Using Chat Template-----------')
    #     text_field = 'messages'
    #     collator = None
    # else:
    text_field = 'prompt'
    response_template_with_context = "\n### Answer:\n"  # We added context here: "\n". This is enough for this tokenizer
    response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False)[1:]  # Now we have it like in the dataset texts: `[2277, 29937, 4007, 22137, 29901]`
    collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

    trainer = SFTTrainer(
            model=model_id,
            model_init_kwargs=model_kwargs,
            args=training_args,
            train_dataset=train,
            eval_dataset = valid,
            dataset_text_field=text_field,
            tokenizer=tokenizer,
            # peft_config=peft_config,
            max_seq_length=tokenizer.model_max_length,
            # formatting_func=dataset.formatting_prompts_func_code,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=10, early_stopping_threshold=0.001)],
            data_collator=collator,
        )
    print("Started Training")
    trainer.train()


if __name__ == '__main__':
    fire.Fire(sft_trainer)

# python finetuning.py sft_trainer --model_id meta-llama/Llama-3.2-3B --max_length 1024 --padding_side right --use_flash_attention_2 True --task aux --mask_language_model False --use_lora True --include_inp_out False
# python finetuning.py sft_trainer --model_id meta-llama/Llama-3.2-3B --tokenizer_id meta-llama/Llama-3.2-3B-Instruct --max_length 1024 --padding_side right --use_flash_attention_2 True --task code --use_lora False --batch_size 4 --use_chat_template True --seed 12345
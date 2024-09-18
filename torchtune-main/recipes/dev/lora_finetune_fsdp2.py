# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import time

from functools import partial
from typing import Any, Dict, Optional, Tuple, Union
from warnings import warn

import torch
from omegaconf import DictConfig, ListConfig

from torch import nn
from torch.distributed import destroy_process_group, init_process_group
from torch.distributed.fsdp import fully_sharded_data_parallel
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointWrapper,
)
import json
import pickle
from datasets import load_from_disk
import pandas as pd
import random
from torchtune.datasets import ConcatDataset
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, DistributedSampler

from torch.optim import Optimizer
from torch.utils.data import DataLoader, DistributedSampler
from torchtune import config, modules, training, utils
from torchtune.data import padded_collate
from torchtune.datasets import ConcatDataset
from torchtune.modules.peft import (
    get_adapter_params,
    get_lora_module_names,
    get_merged_lora_ckpt,
    LoRALinear,
    set_trainable_params,
    validate_missing_and_unexpected_for_lora,
)
from torchtune.recipe_interfaces import FTRecipeInterface

from tqdm import tqdm

log = utils.get_logger("DEBUG")


ds = load_from_disk('/home/vdhee/scratch/LLMcode/Train/Dataset/mbpp2')
new = concatenate_datasets([ds['train'], ds['test'], ds['validation']])

pkl_file = '/home/vdhee/scratch/LLMcode/Train/Dataset/variable_1.pkl'
with open(pkl_file, 'rb') as f:
  variables = pickle.load(f)

variables_data = {}
for j in variables.keys():
  for i in variables[j]:
      variables_data[i["task_id"]]=i["variable"]

file_name = "/home/vdhee/scratch/LLMcode/Train/Dataset/cot_data_1.json"
with open(file_name, 'rb') as f:
  data = json.load(f)

cots= {}
for c in data:
    # print('-------')
    cots[c["question_id"]]={}
    n_len = len(c["Test Cases"])
    for num in range(n_len):
      if f"Test Case {num+1}" in c:
        cots[c["question_id"]][num] = c[f"Test Case {num+1}"]
        
        
json_data=json.load(open("/home/vdhee/scratch/LLMcode/Train/Dataset/mbpp2/dataset.json"))

# def generate_a2_prompts(input_main, output_main, input_a1, output_a1,  input_a2, output_a2, cots):
#     prompts = []
#     outputs=[]

#     for i in range(len(input_a1)):
#         prompt = f"""
# Question: {input_main}

# Ground Truth Code:
# python
# {output_main}
# Predict the values of the intermediate variables after executing the code snippet:
# For Test Case {i+1}:
# Given input variable: {input_a1}
# Given output: {output_a1}

# """
#         output_prompt="""{}
#         Based on this explanation, the value of the unknown variable after executing the code snippet
#         """.format(cots[i])
#         if type(output_a2[i])==list:
#             case = {var: val for var, val in zip(input_a2[i], output_a2[i])}

#             if len(input_a2[i])==0:
#                 pass
#             elif len(input_a2[i])==1:
#                 prompt += "The intermediate variable to predict is {}\n".format(input_a2[0])
#             else:
#                 num_known = random.randint(1, len(input_a2[i]) - 1)  # Ensure at least one known and one unknown
#                 known_vars = random.sample(input_a2[i], num_known)
#                 if known_vars:
#                     prompt += "The Known intermediate variables are:\n"
#                     for var in known_vars:
#                         prompt += f"{var} = {case[var]}\n"
#                 unknown_vars = [var for var in input_a2[i] if var not in known_vars]
#                 prompt += f"\nThere are {len(unknown_vars)} intermediate variable(s) to predict.\n"
#                 for var in unknown_vars:
#                     prompt += f"{var} =? \n"
#                 prompt += "\nPredict the values of the unknown intermediate variables and explain your reasoning based on the given code and known information. Make sure that values of known intermediate variables match the respective values"
#                 for var in unknown_vars:
#                     output_prompt += f"{var} = {case[var]}\n"
#         else:
#             case = {input_a2[i]: output_a2[i]}
#             prompt += "The intermediate variable to predict is {}\n".format(input_a2[i])
#             output_prompt += f"{input_a2[i]} = {output_a2[i]}\n"



#         prompts.append(prompt)
#         outputs.append(output_prompt)
#     idx = random.choice(range(len(input_a1)))

#     return prompts[idx], outputs[idx]


def generate_a2_prompts(input_main, output_main, input_a1, output_a1,  input_a2, output_a2, cots):
    prompts = []
    outputs=[]
    
    print(len(input_a1),"^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

    for i in range(len(input_a1)):
        prompt = f"""
Question: {input_main}

Ground Truth Code:
```python
{output_main}
Predict the values of the intermediate variables after executing the code snippet:
For Test Case {i+1}:
Given input variable: {input_a1}
Given output: {output_a1}

"""
        output_prompt="""{}
        Based on this explanation, the value of the unknown variable after executing the code snippet
        """.format(cots[i])
        if type(output_a2[i])==list:
            case = {var: val for var, val in zip(input_a2[i], output_a2[i])}
            
            if len(input_a2[i])==0:
                pass
            elif len(input_a2[i])==1:
                prompt += "The intermediate variable to predict is {}\n".format(input_a2[0])
            else:
                num_known = random.randint(1, len(input_a2[i]) - 1)  # Ensure at least one known and one unknown
                known_vars = random.sample(input_a2[i], num_known)
                if known_vars:
                    prompt += "The Known intermediate variables are:\n"
                    for var in known_vars:
                        prompt += f"{var} = {case[var]}\n"
                unknown_vars = [var for var in input_a2[i] if var not in known_vars]
                prompt += f"\nThere are {len(unknown_vars)} intermediate variable(s) to predict.\n"
                for var in unknown_vars:
                    prompt += f"{var} =? \n"
                prompt += "\nPredict the values of the unknown intermediate variables and explain your reasoning based on the given code and known information. Make sure that values of known intermediate variables match the respective values"
                for var in unknown_vars:
                    output_prompt += f"{var} = {case[var]}\n"
        else:
            case = {input_a2[i]: output_a2[i]}
            prompt += "The intermediate variable to predict is {}\n".format(input_a2[i])
            output_prompt += f"{input_a2[i]} = {output_a2[i]}\n"
            
            
    
        prompts.append(prompt)
        outputs.append(output_prompt)
    idx = random.choice(range(len(input_a1)))

    return prompts[idx], outputs[idx]

def generate_a1_prompt(input_main, output_main, input_a1, output_a1, cots):
        # input_main, output_main, input_a1, output_a1,cots
    a1_prompt = f"""You are an expert Python code analyzer and output predictor. Given a question, ground truth Python code, sample input-output pairs, and a new input, your task is to predict the final output or result. Follow these guidelines:
1. Analyze the provided Python code carefully.
2. Consider the given sample input-output pairs to understand the code's behavior.
3. Use the new input to determine the expected output.

Question: {input_main}

Ground Truth Code:
```python
{output_main}
    """

    output_prompt="""
    """

    if len(input_a1)!=0:

    # Create a dictionary of input-output pairs
        case = {var: val for var, val in zip(input_a1, output_a1)}
        ccots={var: c for var, c in zip(input_a1, cots.values())}

    # Randomly select the number of known inputs
        num_known = random.randint(1, len(input_a1) - 1)  # Ensure at least one known and one unknown

    # Randomly select known inputs
        known_vars = random.sample(input_a1, num_known)

    # Add known input-output pairs to the prompt
        a1_prompt += "Here are the sample known input and output pairs:\n"
        for k in known_vars:
            a1_prompt += "input: {}, the output is {}.\n".format(k, case[k])

    # Determine the unknown inputs
        unknown_vars = [var for var in input_a1 if var not in known_vars]

    # Add unknown inputs to the prompt for prediction
        a1_prompt += "Now predict the output values of these input variables given below:\n"
        for var in unknown_vars:
            a1_prompt += f"{var} = ?\n"

        for var in unknown_vars:
            output_prompt="""Based on this explanation {}. \nThe value of the unknown variable after executing the code snippet:-""".format(ccots[var])
            output_prompt += f"{var} = {case[var]}\n"

    return a1_prompt, output_prompt

class dataset(Dataset):
    def __init__(self, data, tokenizer, json_data,  max_length = 1024, batch_size=4, lambda_1=0.5):
        self.data = data
        self.code_gen = data.select_columns(['task_id', 'prompt', 'code'])
        self.aux_data = data.select_columns(['task_id', 'prompt', 'code', 'test_imports', 'test_list', 'variables'])
        self.json_data = json_data
        self.count = -1
        self.batch_size = batch_size
        self.split = int(self.batch_size * lambda_1 )
        self.code_gen_idx = list(range(len(self.code_gen)))
        self.aux_idx = list(range(len(self.aux_data)))
        self.tokenizer = tokenizer
        self.max_length = max_length
    
        random.shuffle(self.code_gen_idx)
        random.shuffle(self.aux_idx)
        self.aux_ptr = 0
        self.code_ptr = 0
        self.len_data = len(self.data)
        self.prompt_len = []
  
    def _update_aux(self):
        new_id = self.aux_idx[self.aux_ptr]
        self.aux_ptr += 1
        self.aux_ptr = self.aux_ptr % self.len_data
        return new_id
  
    def process_code(self, id):
        code_gen = "### Task: " + self.code_gen[id]['prompt'] + '\n' + "### Answer:\n"
        out="```python " + self.code_gen[id]['code'] + "```"
        # print(code_gen)
        return code_gen, out
  
  
    def __len__(self):
        return len(self.data)
  
    def __getitem__(self, idx):
        self.count += 1
        self.count = self.count % self.batch_size
        # print(self.count)
        if self.count < self.split:
    
          id = self.code_gen_idx[self.code_ptr]
          self.code_ptr += 1
          self.code_ptr = self.code_ptr % self.len_data
          inp, output = self.process_code(id)  
          token = self.tokenizer.encode(inp, add_eos=True)
          label = self.tokenizer.encode(output, add_eos=True)
          
          return {
              "tokens" : token,
              "labels": label
          }
    
        else:
          # chose auxiliary task
          id = self.aux_idx[self.aux_ptr]
          task_id = self.code_gen[id]['task_id']
          
          while str(task_id) not in list(self.json_data.keys()):
              id=self._update_aux()
              task_id = self.code_gen[id]['task_id']
          
          self.aux_ptr += 1
          self.aux_ptr = self.aux_ptr % self.len_data
          if random.random() > 0.5:
              # aux_1
              inp = self.json_data[str(task_id)]['a1']['tokens']
              output = self.json_data[str(task_id)]['a1']['labels']
          else:
              #aux_2
              n = len(self.json_data[str(task_id)]['a2'].keys())

              test_id = random.choice(range(n))
              inp = self.json_data[str(task_id)]['a2'][str(test_id)]['tokens']
              l1=random.choice(range(len(inp)))
              output = self.json_data[str(task_id)]['a2'][str(test_id)]['labels']
              l2=random.choice(range(len(output)))
          
          token = self.tokenizer.encode(inp[l1], add_eos=True)
          label = self.tokenizer.encode(output[l2], add_eos=True)
          
          return {
              "tokens" : token,
              "labels": label
          }
          


class LoRAFinetuneRecipeDistributed(FTRecipeInterface):
    """
    Distributed LoRA finetuning recipe for dense transformer-based LLMs such as Llama2. This recipe supports
    distributed training and can be run on a single node (1 to 8 GPUs).

    Features:
        - FSDP. Supported using PyTorch's FSDP APIs. DDP is currently not supported. Traning on CPU is not
            supported.

        - Activation Checkpointing. This can be controlled using the ``activation_checkpointing``
            flag. Activation checkpointing helps reduce the memory footprint since we no longer keep
            activations in memory and instead recompute them during the backward pass. This is especially
            helpful for larger batch sizes when you're memory constrained. But these savings in memory
            come at the cost of training performance. In most cases training can slow-down quite a bit as
            a result of this activation recomputation.

        - Precision. Full fp32 and bf16 training are supported. Precision is controlled using the ``dtype``
            flag. When ``dtype=bf16``, all activations, gradients and optimizer states are in bfloat16. In
            most cases this should halve the memory footprint of full precision (fp32) training, without
            loss in model quality (will depend on the model, training data and other settings). For
            GPUs which do not support bfloat16, we fall back to fp32. Mixed precision training and fp16
            precision are currently not supported.

        - Gradient Accumulation. You can simulate larger batch sizes by accumulating gradients. This is
            controlled using the ``gradient_accumulation_steps`` flag.

                Total Batch Size = batch_size * number of GPUs * gradient accumulation steps.

            For example: with batch_size=1, nproc_per_node=2 and gradient_accumulation_steps=32 we get a
            total batch size of 64.

            Gradient accumulation is especially useful when you are memory constrained. In this case,
            accumulating gradients might give you better training speed than enabling activation
            checkpointing.

        - Checkpointing. Model weights are checkpointed both at the end of each epoch and at the end of
            training. Currently we checkpoint both the adapter weights (trainable params only) and the
            complete merged weights (adapter weights added back to the base model). For more details
            please take a look at our LoRA tutorial
            (https://pytorch.org/torchtune/main/tutorials/lora_finetune.html).

            Optimizer State and recipe state (seed, total_epochs, number of epochs run etc) are
            only saved at the end of a given epoch and used in case of resuming training. Resuming
            training is controlled by the ``resume_from_checkpoint`` flag. Mid-epoch checkpointing is
            currently not supported.

            For more details on the checkpointer, please take a look at
            our checkpointer deepdive (https://pytorch.org/torchtune/main/tutorials/checkpointer.html).

        - Logging. Terminal, Disk, WandB and TensorBoard are all supported.

    For a full list of example configs for this recipe, run ``tune ls`` on the command line. Each config
    has example commands for how to kick-off training.

    Args:
        cfg (DictConfig): OmegaConf object parsed from yaml file

    Raises:
        ValueError: If ``dtype`` is set to fp16.
        ValueError: If world_size is 1
        RuntimeError: If ``dtype`` is set to bf16 and the hardware does not support bf16.
    """

    def __init__(self, cfg, new, json_data) -> None:

        if not utils.torch_version_ge("2.4.0"):
            raise RuntimeError("FSDP2 recipe is only available on PyTorch nightlies")

        self._device = utils.get_device(device=cfg.device)
        self._dtype = training.get_dtype(cfg.dtype, device=self._device)

        if self._dtype == torch.float16:
            raise ValueError(
                "full fp16 training is not supported with this recipe. Please use bf16 or fp32 instead."
            )

        _, rank = utils.get_world_size_and_rank()

        # _is_rank_zero is used primarily for logging. In the future, the logger
        # should directly take care of this
        self._is_rank_zero = rank == 0

        # logging attributes
        self._output_dir = cfg.output_dir
        self._log_every_n_steps = cfg.get("log_every_n_steps", 1)
        self._log_peak_memory_stats = cfg.get("log_peak_memory_stats", False)

        # training attributes
        self._enable_activation_checkpointing = cfg.enable_activation_checkpointing

        # These attributes constitute the recipe state and are updated by ``load_checkpoint``
        # when ``resume_from_checkpoint`` is ``True``
        self.seed = utils.set_seed(seed=cfg.seed)
        self.epochs_run = 0
        self.total_epochs = cfg.epochs
        self.max_steps_per_epoch = cfg.max_steps_per_epoch
        self.global_step = 0
        self.json_data=json_data
        self.new=new

        self._resume_from_checkpoint = cfg.resume_from_checkpoint
        self._gradient_accumulation_steps = cfg.gradient_accumulation_steps

    def load_checkpoint(self, cfg_checkpointer: DictConfig) -> Dict[str, Any]:
        """
        Extract the checkpoint state from file and validate. This includes the
        base model weights. If resume_from_checkpoint is True, this also includes
        the adapter weights and recipe state
        """
        self._checkpointer = config.instantiate(
            cfg_checkpointer,
            resume_from_checkpoint=self._resume_from_checkpoint,
        )
        checkpoint_dict = self._checkpointer.load_checkpoint()

        # When resuming from checkpoint for LoRA, the recipe expects the adapter weights
        # and recipe state to be present. The keys should match up with what ``save_checkpoint``
        # used to create these intermediate checkpoints
        if self._resume_from_checkpoint:
            if training.ADAPTER_KEY not in checkpoint_dict:
                raise ValueError(
                    "Adapter weights not found. Please ensure a valid adapter checkpoint is provided."
                )
            # _update_recipe_state will throw an exception if the recipe state is not corrctly loaded
            # no need to check here
            self._update_recipe_state(checkpoint_dict)
        return checkpoint_dict

    def _update_recipe_state(self, ckpt_dict: Dict[str, Any]) -> None:
        """
        Updates the recipe state from checkpoint.
        """
        if not (
            training.SEED_KEY in ckpt_dict
            and training.TOTAL_EPOCHS_KEY in ckpt_dict
            and training.MAX_STEPS_KEY in ckpt_dict
        ):
            raise KeyError(
                "Checkpoint does not contain the required keys needed for updating recipe state."
                "Are you sure you passed in the right recipe checkpoint?"
            )
        # If seed, total_epoch or max_steps_per_epoch don't match,
        # warn the user and overwrite
        if (
            self.seed != ckpt_dict[training.SEED_KEY]
            or self.total_epochs != ckpt_dict[training.TOTAL_EPOCHS_KEY]
            or self.max_steps_per_epoch != ckpt_dict[training.MAX_STEPS_KEY]
        ):
            warn(
                message="""Configured value for seed, epochs or max_steps_per_epoch
                does not match the value stored in checkpoint."""
            )
        self.seed = utils.set_seed(seed=ckpt_dict[training.SEED_KEY])
        self.epochs_run = ckpt_dict[training.EPOCHS_KEY]
        self.total_epochs = ckpt_dict[training.TOTAL_EPOCHS_KEY]
        self.max_steps_per_epoch = ckpt_dict[training.MAX_STEPS_KEY]

    def setup(self, cfg: DictConfig) -> None:
        """
        Setup the recipe state. This includes recipe state (if resume_from_checkpoint is True),
        model, tokenizer, loss, optimizer, learning rate scheduler, sampler, and dataloader.
        """
        if self._is_rank_zero:
            self._metric_logger = config.instantiate(cfg.metric_logger)

            # log config with parameter override
            self._metric_logger.log_config(cfg)

        checkpoint_dict = self.load_checkpoint(cfg_checkpointer=cfg.checkpointer)
        self._model_compile = cfg.compile

        self._model = self._setup_model(
            cfg_model=cfg.model,
            enable_activation_checkpointing=cfg.enable_activation_checkpointing,
            base_model_state_dict=checkpoint_dict[training.MODEL_KEY],
            lora_weights_state_dict=(
                checkpoint_dict[training.ADAPTER_KEY]
                if self._resume_from_checkpoint
                else None
            ),
            cfg_fsdp=cfg.fsdp if hasattr(cfg, "fsdp") else None,
        )
        self._tokenizer = config.instantiate(cfg.tokenizer)

        self._optimizer = self._setup_optimizer(
            cfg_optimizer=cfg.optimizer,
            opt_state_dict=checkpoint_dict[training.OPT_KEY]
            if self._resume_from_checkpoint
            else None,
        )

        # initialize loss
        self._loss_fn = config.instantiate(cfg.loss)
        backend = os.environ.get("TORCH_COMPILE_BACKEND", "inductor")
        if self._loss_fn.__class__.__name__ == "CEWithChunkedOutputLoss":
            # set num_output_chunks for model
            self._model.set_num_output_chunks(self._loss_fn.num_output_chunks)
            if self._model_compile:
                log.info("Compiling loss with torch.compile...")
                # For CEWithChunkedOutputLoss, if we compile the entire class
                # we lose the benefits from the chunked loss.
                # Therefore, we only compile the cross entropy function + upcasting
                self._loss_fn.compute_cross_entropy = torch.compile(
                    self._loss_fn.compute_cross_entropy, backend=backend
                )
        else:
            if self._model_compile:
                log.info("Compiling loss with torch.compile...")
                self._loss_fn = torch.compile(self._loss_fn, backend=backend)
        log.info("Loss is initialized.")

        # sampler and dataloader depend on the tokenizer and loss_fn and should be
        # setup after all of these are setup
        # self._sampler, self._dataloader = self._setup_data(
        #     cfg_dataset=cfg.dataset,
        #     shuffle=cfg.shuffle,
        #     batch_size=cfg.batch_size,
        # )
        
        self._sampler, self._dataloader= self.setup_data_(
        self.new, 
        self._tokenizer,
        cfg.shuffle,
        cfg.batch_size)

        # Finally update the recipe state which can only be correctly set after all of the
        # other components have been initialized and updated.

        # Number of training steps in each epoch depends on the number of batches produced
        # by the dataloader and the max_steps_per_epoch param set by the user and is used
        # for logging and tracking training state. This should be computed after the dataloader
        # has been setup
        self._steps_per_epoch = (
            len(self._dataloader) // self._gradient_accumulation_steps
        )
        if (
            self.max_steps_per_epoch is not None
            and self.max_steps_per_epoch < self._steps_per_epoch
        ):
            self._steps_per_epoch = self.max_steps_per_epoch
        self.global_step = self.epochs_run * self._steps_per_epoch

        # Learning rate scheduler can only be set up after number of steps
        # has been computed
        self._lr_scheduler = self._setup_lr_scheduler(
            cfg_lr_scheduler=cfg.lr_scheduler,
            num_training_steps=self.total_epochs * self._steps_per_epoch,
            last_epoch=self.global_step - 1,
        )

        # Used to ignore labels for loss computation
        self.ignore_labels_cache = torch.full(
            (cfg.batch_size, 1), self._loss_fn.ignore_index, device=self._device
        )

    def _setup_model(
        self,
        cfg_model: DictConfig,
        enable_activation_checkpointing: bool,
        base_model_state_dict: Dict[str, Any],
        lora_weights_state_dict: Optional[Dict[str, Any]] = None,
        cfg_fsdp: Optional[Union[DictConfig, None]] = None,
    ) -> nn.Module:
        """
        Model initialization has some important considerations:
           a. To minimize GPU peak memory, we initialize the model on meta device with
              the right dtype
           b. All ranks calls ``load_state_dict`` without peaking CPU RAMs since
              full state dicts are loaded with ``torch.load(mmap=True)``
           c. We register (pre-)forward hooks with ``fully_sharded_data_parallel`` instead of wrapping `nn.Module`
        """

        self._lora_rank = cfg_model.lora_rank
        self._lora_alpha = cfg_model.lora_alpha
        self._lora_attn_modules = list(cfg_model.lora_attn_modules)
        self._apply_lora_to_mlp = cfg_model.apply_lora_to_mlp
        self._apply_lora_to_output = getattr(cfg_model, "apply_lora_to_output", False)

        if self._is_rank_zero:
            log.info(
                "FSDP is enabled. Instantiating model and loading checkpoint on Rank 0 ..."
            )
            init_start = time.perf_counter()

        with training.set_default_dtype(self._dtype), torch.device("meta"):
            model = config.instantiate(cfg_model)

        self.adapter_params = get_adapter_params(model)
        set_trainable_params(model, self.adapter_params)

        if self._model_compile:
            log.info("Compiling model layers with torch.compile...")
            backend = os.environ.get("TORCH_COMPILE_BACKEND", "inductor")
            for m in reversed(list(model.modules())):
                if isinstance(m, modules.TransformerSelfAttentionLayer):
                    m.compile(backend=backend)

        if enable_activation_checkpointing:
            utils.set_activation_checkpointing(
                model, auto_wrap_policy={modules.TransformerSelfAttentionLayer}
            )

        fsdp_kwargs = {}
        if cfg_fsdp and cfg_fsdp.cpu_offload:
            from torch.distributed._composable.fsdp import CPUOffloadPolicy

            fsdp_kwargs["offload_policy"] = CPUOffloadPolicy()
        # iterating from lowerer modules to higher
        # eg grouping lora adapters before transformer block
        for m in reversed(list(model.modules())):
            if isinstance(m, nn.Linear) and m.weight.requires_grad:
                fully_sharded_data_parallel(m, **fsdp_kwargs)
            # TransformerSelfAttentionLayer is wrapped by CheckpointWrapper
            # when enable_activation_checkpointing
            if enable_activation_checkpointing:
                if isinstance(m, CheckpointWrapper):
                    fully_sharded_data_parallel(m, **fsdp_kwargs)
            else:
                if isinstance(m, modules.TransformerSelfAttentionLayer):
                    fully_sharded_data_parallel(m, **fsdp_kwargs)
        fully_sharded_data_parallel(model, **fsdp_kwargs)

        if lora_weights_state_dict:
            lora_missing, lora_unexpected = utils.load_from_full_model_state_dict(
                model, lora_weights_state_dict, self._device, self._is_rank_zero
            )
        else:
            lora_missing, lora_unexpected = None, None

        with training.set_default_dtype(self._dtype), self._device:
            lora_device = "cpu" if cfg_fsdp and cfg_fsdp.cpu_offload else self._device
            for m in model.modules():
                if isinstance(m, LoRALinear) and not lora_weights_state_dict:
                    # lora may not be covered in state dict
                    # if finetune for the 1st time
                    m.lora_a.to_empty(device=lora_device)
                    m.lora_b.to_empty(device=lora_device)
                    m.initialize_parameters()
                # RoPE is not covered in state dict
                if isinstance(m, modules.RotaryPositionalEmbeddings):
                    m.reset_parameters()

        base_missing, base_unexpected = utils.load_from_full_model_state_dict(
            model, base_model_state_dict, self._device, self._is_rank_zero
        )
        validate_missing_and_unexpected_for_lora(
            lora_attn_modules=self._lora_attn_modules,
            apply_lora_to_mlp=self._apply_lora_to_mlp,
            apply_lora_to_output=self._apply_lora_to_output,
            base_missing=base_missing,
            base_unexpected=base_unexpected,
            lora_missing=lora_missing,
            lora_unexpected=lora_unexpected,
        )
        # Ensure no params and buffers are on meta device
        utils.validate_no_params_on_meta_device(model)

        if self._is_rank_zero:
            log.info(
                f"Instantiating model and loading checkpoint took {time.perf_counter() - init_start:.2f} secs"
            )
            memory_stats = utils.get_memory_stats(device=self._device)
            utils.log_memory_stats(memory_stats)

        # synchronize before training begins
        torch.distributed.barrier()

        return model

    def _setup_optimizer(
        self, cfg_optimizer: DictConfig, opt_state_dict: Optional[Dict[str, Any]] = None
    ) -> Optimizer:
        optimizer = config.instantiate(cfg_optimizer, self._model.parameters())
        if opt_state_dict:
            utils.load_from_full_optimizer_state_dict(
                optimizer,
                opt_state_dict,
                self._device,
            )

        if self._is_rank_zero:
            log.info("Optimizer and loss are initialized.")
        return optimizer

    def _setup_lr_scheduler(
        self,
        cfg_lr_scheduler: DictConfig,
        num_training_steps: int,
        last_epoch: int,
    ) -> Optimizer:
        lr_scheduler = config.instantiate(
            cfg_lr_scheduler,
            self._optimizer,
            num_training_steps=num_training_steps,
            last_epoch=last_epoch,
        )
        if self._is_rank_zero:
            log.info("Learning rate scheduler is initialized.")
        return lr_scheduler
    
    def setup_data_(
        self,
        new,
        tokenizer,
    shuffle: bool,
        batch_size: int,):
        
        world_size, rank = utils.get_world_size_and_rank()

    #     ds = CustomDataset(path, self.variables, self.cots
    # )
        ds=dataset(new, tokenizer, self.json_data,max_length=512)

        sampler = DistributedSampler(
        ds, num_replicas=world_size, rank=rank, shuffle=shuffle, seed=0
    )

        dataloader = DataLoader(
            dataset=ds,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=(
                partial(
                    padded_collate,
                    padding_idx=self._tokenizer.pad_id,
                    ignore_idx=self._loss_fn.ignore_index,
                )
            )
        )

        if self._is_rank_zero:
            log.info("Dataset and Sampler are initialized.")

        return sampler, dataloader

    def _setup_data(
        self,
        cfg_dataset: DictConfig,
        shuffle: bool,
        batch_size: int,
    ) -> Tuple[DistributedSampler, DataLoader]:
        """
        All data related setup happens here. Currently this recipe only supports the
        DistributedSamplers with Map-style Datasets which fit into memory. Other samplers,
        iterable datasets and streaming datasets are not supported.
        """
        world_size, rank = utils.get_world_size_and_rank()

        if isinstance(cfg_dataset, ListConfig):
            datasets = [
                config.instantiate(single_cfg_dataset, tokenizer=self._tokenizer)
                for single_cfg_dataset in cfg_dataset
            ]
            ds = ConcatDataset(datasets=datasets)
            packed = False
        else:
            ds = config.instantiate(cfg_dataset, tokenizer=self._tokenizer)
            packed = cfg_dataset.get("packed", False)

        sampler = DistributedSampler(
            ds, num_replicas=world_size, rank=rank, shuffle=shuffle, seed=0
        )

        dataloader = DataLoader(
            dataset=ds,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=(
                partial(
                    padded_collate,
                    padding_idx=self._tokenizer.pad_id,
                    ignore_idx=self._loss_fn.ignore_index,
                )
                if not packed
                else None
            ),
        )

        if self._is_rank_zero:
            log.info("Dataset and Sampler are initialized.")

        return sampler, dataloader

    def save_checkpoint(
        self,
        epoch: int,
    ) -> None:
        """
        Checkpoint the state of the recipe. The constructed checkpoint state dict
        contains the following information:
        - Merged weights with key MODEL_KEY
        - Adapter weights with key ADAPTER_KEY
        - Relevant recipe state if training is not complete

        Checkpointer will save the merged weights, adapter weights and recipe state in
        different checkpoint files. To correctly resume from training, the adapter weights
        and recipe state must be provided along with the base model weights.
        """
        # final dict passed onto the checkpointer
        checkpoint_dict = {}

        intermediate_checkpoint = epoch + 1 < self.total_epochs
        # To prevent GPU memory from spiking during checkpoint save,
        # we consolidate the full model and optim state dicts on CPU for rank 0
        cpu_state_dict = utils.get_full_model_state_dict(
            self._model,
            self._is_rank_zero,
        )

        if intermediate_checkpoint:
            opt_state_dict = utils.get_full_optimizer_state_dict(
                self._optimizer,
                self._is_rank_zero,
            )
        else:
            opt_state_dict = None

        # Now that we have the model and opt state dict, create the actual checkpoint dict
        # to be sent to the checkpointer and ultimately written to file
        if self._is_rank_zero:

            # Filter out the adapter keys and weights from the model state dict. These will
            # be saved separately
            adapter_key_filter = lambda x: x in self.adapter_params
            adapter_state_dict = {
                k: v for k, v in cpu_state_dict.items() if adapter_key_filter(k)
            }
            checkpoint_dict.update({training.ADAPTER_KEY: adapter_state_dict})

            # merge the adapter weights and base weights to create the model checkpoint
            merged_state_dict = get_merged_lora_ckpt(
                cpu_state_dict,
                rank=self._lora_rank,
                alpha=self._lora_alpha,
            )
            checkpoint_dict.update({training.MODEL_KEY: merged_state_dict})

            # if training is in-progress, checkpoint the optimizer state and recipe state
            # as well.
            if intermediate_checkpoint:
                checkpoint_dict.update(
                    {
                        training.OPT_KEY: opt_state_dict,
                        training.SEED_KEY: self.seed,
                        training.EPOCHS_KEY: self.epochs_run,
                        training.TOTAL_EPOCHS_KEY: self.total_epochs,
                        training.MAX_STEPS_KEY: self.max_steps_per_epoch,
                    }
                )

            adapter_config = {
                "r": self._lora_rank,
                "lora_alpha": self._lora_alpha,
                "target_modules": get_lora_module_names(
                    self._lora_attn_modules,
                    self._apply_lora_to_mlp,
                    self._apply_lora_to_output,
                ),
                "peft_type": "LORA",
            }
            checkpoint_dict.update({training.ADAPTER_CONFIG: adapter_config})

            self._checkpointer.save_checkpoint(
                checkpoint_dict,
                epoch=epoch,
                intermediate_checkpoint=intermediate_checkpoint,
            )

    def train(self) -> None:
        """
        The core training loop.
        """
        # clean up before training begins
        utils.cleanup_before_training()

        _, rank = utils.get_world_size_and_rank()

        # zero out the gradients before starting training
        self._optimizer.zero_grad()

        # Initialize tokens count and running loss (for grad accumulation)
        t0 = time.perf_counter()
        running_loss = 0
        num_tokens = 0

        # self.epochs_run should be non-zero when we're resuming from a checkpoint
        for curr_epoch in range(self.epochs_run, self.total_epochs):

            # Update the sampler to ensure data is correctly shuffled across epochs
            # in case shuffle is True
            self._sampler.set_epoch(curr_epoch)

            pbar = tqdm(total=self._steps_per_epoch, disable=not (rank == 0))
            for idx, batch in enumerate(self._dataloader):
                if (
                    self.max_steps_per_epoch is not None
                    and (idx // self._gradient_accumulation_steps)
                    == self.max_steps_per_epoch
                ):
                    break

                # Both are shape [b, s]
                tokens, labels = batch["tokens"], batch["labels"]
                # Get the attention mask and position ids from the dataset if they
                # exist. Currently, only sample packing in PackedDataset returns these
                mask = batch.get("mask", None)  # shape [b, s, s]
                input_pos = batch.get("input_pos", None)  # shape [b, s]

                tokens = tokens.to(self._device)
                num_tokens += tokens.numel()
                labels = labels.to(self._device)
                mask = mask.to(self._device) if mask is not None else None
                input_pos = (
                    input_pos.to(self._device) if input_pos is not None else None
                )

                logits = self._model(tokens, mask=mask, input_pos=input_pos)

                # Shift labels to compute loss
                # equivalent to doing labels[..., 1:] and logits[..., :-1, :]
                # But this way we dont need to slice the logits. We just add an ignore index to labels.
                labels = torch.hstack(
                    (labels[..., 1:], self.ignore_labels_cache[: labels.shape[0]])
                )
                if not isinstance(logits, list):
                    labels = labels.reshape(-1)
                    logits = logits.reshape(-1, logits.size(-1))

                # Compute loss
                loss = self._loss_fn(logits, labels)

                # free logits otherwise it peaks backward memory
                del logits

                loss = loss / self._gradient_accumulation_steps
                running_loss += loss
                loss.backward()

                # Step with optimizer
                if (idx + 1) % self._gradient_accumulation_steps == 0:
                    self._optimizer.step()
                    self._optimizer.zero_grad(set_to_none=True)
                    self._lr_scheduler.step()

                    # Update the number of steps when the weights are updated
                    self.global_step += 1

                    loss_to_log = running_loss.item()
                    pbar.update(1)
                    pbar.set_description(
                        f"{curr_epoch + 1}|{self.global_step}|Loss: {loss_to_log}"
                    )

                    # Log per-step metrics
                    if (
                        self.global_step % self._log_every_n_steps == 0
                        and self._is_rank_zero
                    ):
                        time_per_step = time.perf_counter() - t0
                        log_dict = {
                            "loss": loss_to_log,
                            "lr": self._optimizer.param_groups[0]["lr"],
                            "tokens_per_second_per_gpu": num_tokens / time_per_step,
                        }
                        if self._log_peak_memory_stats:
                            log_dict.update(utils.get_memory_stats(device=self._device))
                        self._metric_logger.log_dict(
                            log_dict,
                            step=self.global_step,
                        )

                    # Reset running stats for the next step
                    running_loss = 0
                    num_tokens = 0
                    t0 = time.perf_counter()

            self.epochs_run += 1
            self.save_checkpoint(epoch=curr_epoch)

    def cleanup(self) -> None:
        if self._is_rank_zero:
            self._metric_logger.close()
        destroy_process_group()


@config.parse
def recipe_main(cfg: DictConfig) -> None:
    """
    Entry point for the recipe.

    Configurable parameters are read in the following order:
        - Parameters specified in config (see available configs through ``tune ls``)
        - Overwritten by arguments from the command-line
    """
    if not utils.is_distributed():
        raise RuntimeError(
            "Distributed finetune recipe should be run via a distributed launcher."
            "If using tune CLI, please specify --nnodes 1 and --nproc_per_node [num_gpus]"
        )
    os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"
    init_process_group(backend="gloo" if cfg.device == "cpu" else "nccl")

    config.log_config(recipe_name="LoRAFinetuneRecipeDistributed", cfg=cfg)

    recipe = LoRAFinetuneRecipeDistributed(cfg, new, json_data)
    recipe.setup(cfg=cfg)
    recipe.train()
    recipe.cleanup()


if __name__ == "__main__":
    sys.exit(recipe_main())

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os
import pandas as pd
import random
import sys
import re
import time
from datasets import load_from_disk
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

from functools import partial
from typing import Any, Dict, Optional, Tuple, Union
from warnings import warn

import torch
from omegaconf import DictConfig, ListConfig
import json
import pickle

from torch import nn
from torch.distributed import destroy_process_group, init_process_group
from torch.distributed.fsdp import (
    FullOptimStateDictConfig,
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    StateDictType,
)
from torch.utils.data import Dataset
from torch.optim import Optimizer
from torch.utils.data import DataLoader, DistributedSampler
from torchtune import config, modules, training, utils
from torchtune.data import padded_collate
from torchtune.datasets import ConcatDataset
from torchtune.modules.peft import (
    get_adapter_params,
    get_lora_module_names,
    get_merged_lora_ckpt,
    set_trainable_params,
    validate_state_dict_for_lora,
)
from torchtune.recipe_interfaces import FTRecipeInterface
from torchtune.utils import DummyProfiler, PROFILER_KEY

from tqdm import tqdm

log = utils.get_logger("DEBUG")



def generate_a2_prompts(input_main, output_main, input_a1, output_a1,  input_a2, output_a2, cots):
    prompts = []
    outputs=[]

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

    return prompts, outputs




def print_gpu_memory_usage(device=None):
    if device is None:
        device = torch.cuda.current_device()  # Get current device if not specified
    else:
        device = torch.device(device)  # Use specified device

    allocated_memory = torch.cuda.memory_allocated(device)
    reserved_memory = torch.cuda.memory_reserved(device)
    max_allocated_memory = torch.cuda.max_memory_allocated(device)
    max_reserved_memory = torch.cuda.max_memory_reserved(device)
    
    print(f"Current GPU memory usage on device {device}:")
    print(f"Allocated memory: {allocated_memory / 1024 ** 2:.2f} MB")
    print(f"Reserved memory: {reserved_memory / 1024 ** 2:.2f} MB")
    print(f"Max allocated memory: {max_allocated_memory / 1024 ** 2:.2f} MB")
    print(f"Max reserved memory: {max_reserved_memory / 1024 ** 2:.2f} MB")
    print()



def generate_a1_prompt(input_main, output_main, input_a1, output_a1,cots):
    a1_prompt = """You are an expert Python code analyzer and output predictor. Given a question, ground truth Python code, sample input-output pairs, and a new input, your task is to predict the final output or result. Follow these guidelines:
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
            output_prompt="""
            Based on this explanation {}. \nThe value of the unknown variable after executing the code snippet:-
            """.format(ccots[var])
            output_prompt += f"{var} = {case[var]}\n"
    
    
    return a1_prompt, output_prompt



class CustomDataset(Dataset):
    def __init__(self, path, variable_data,cots):
        self.d = load_from_disk("/home/vdhee/scratch/LLMcode/Train/Dataset/mbpp2")
        self.train_data = pd.read_csv(path)
        self.pattern = r"assert\s+\w+\((.*?)\)\s*==\s*(.*)"
        self.variable_data = variable_data
        self.cots = cots
        self.inputs_main=self.train_data["prompt"].values
        self.outputs_main=self.train_data["code"].values
        self.tests_list={}
        self.ids=self.train_data["task_id"].values
        for t in ["train","test","validation"]:
            for j in self.d[t]:
                self.tests_list[j["task_id"]]=j["test_list"]
        

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        index=self.ids[idx]
        print(index, "*****************&&&&&&&&&&&&&&&&&&&&&")
        input_main = self.inputs_main[idx]
        output_main = self.outputs_main[idx]
        test_list = self.tests_list[index]
        variables = self.variable_data[index]
        
        inputs = []
        outputs = []
        for test in test_list:
            match = re.findall(self.pattern, test)
            if match:
                inputs.append(match[0][0])
                outputs.append(match[0][1])

        input_a1 = inputs
        output_a1 = outputs
        input_a2 = []
        output_a2 = []

        for n, test in variables.items():
            input_a2.append(list(test.keys()))
            output_a2.append(list(test.values()))
            

        return {
            "input_ids": {
                "input_main": input_main,
                "output_main": output_main,
                "input_a1": input_a1,
                "output_a1": output_a1,
                "input_a2": input_a2,
                "output_a2": output_a2,
                "cots":self.cots[index], 
                "task_id":self.ids[idx]
            }
        }
        
        
 
def pad_tensors(input_batch, output_batch, padding_idx=0, target_length=None):
    input_ids = pad_sequence(
        [torch.tensor(x) for x in input_batch],
        batch_first=True,
        padding_value=padding_idx,
    )
    labels = pad_sequence(
        [torch.tensor(x) for x in output_batch],
        batch_first=True,
        padding_value=padding_idx,
    )

    input_ids_seq_len = input_ids.shape[-1]
    labels_seq_len = labels.shape[-1]

    # Hack to pad correctly and not use max_seq_len, which is costly
    if input_ids_seq_len > labels_seq_len:
        labels = F.pad(
            labels, (0, input_ids_seq_len - labels_seq_len), value=padding_idx
        )
    elif labels_seq_len > input_ids_seq_len:
        input_ids = F.pad(
            input_ids,
            (0, labels_seq_len - input_ids_seq_len),
            value=padding_idx,
        )
    return input_ids.long(), labels.long()

class CustomDataCollator:
    def __init__(self, tokenizer, padding_idx=0, ignore_idx=-100):
        self.tokenizer = tokenizer
        self.main_prompt = """You are an expert Python code generator. Your task is to generate clean, efficient, and correct Python code based on the given question or problem description. Follow these guidelines:
            1. Generate Python code with explanations in comments.
            2. Use proper indentation and formatting for readability.
            3. Implement the solution using Python best practices and optimal algorithms.
            4. Ensure the code is complete and can be executed without additional modifications.
            5. Include a brief comment at the beginning explaining the overall purpose of the code.

            Question: {}

            Code:
        """
        self.padding_idx = padding_idx
        self.ignore_idx = ignore_idx

    def __call__(self, features):
        main_inputs = [self.main_prompt.format(feature["input_ids"]["input_main"]) for feature in features]
        main_outputs = [feature["input_ids"]["output_main"] for feature in features]

    # Tokenize inputs and outputs individually
        input_main_encodings = [self.tokenizer.encode(input_text,add_eos=True) for input_text in main_inputs]
        output_main_encodings = [self.tokenizer.encode(output_text,add_eos=True) for output_text in main_outputs]

    # Find the maximum length among all encodings

    # Pad input_ids and labels to max_length
        input_ids, labels = pad_tensors(input_main_encodings, output_main_encodings)
        
        
        
        a1_inputs = []
        a1_outputs = []
        for feature in features:
            a1_input, a1_output = generate_a1_prompt(
                feature["input_ids"]["input_main"], 
                feature["input_ids"]["output_main"], 
                feature["input_ids"]["input_a1"], 
                feature["input_ids"]["output_a1"], 
                feature["input_ids"]["cots"]
            )
            a1_inputs.append(a1_input)
            a1_outputs.append(a1_output)
        input_a1 = [self.tokenizer.encode(input_text,add_eos=True) for input_text in a1_inputs]
        output_a1 = [self.tokenizer.encode(output_text,add_eos=True) for output_text in a1_outputs]
        
        input_ids_a1, labels_a1 = pad_tensors(input_a1, output_a1)
        
        a2_inputs=[]
        a2_outputs=[]
        for feature in features:
            a2_input, a2_output = generate_a2_prompts(feature["input_ids"]["input_main"], feature["input_ids"]["output_main"], feature["input_ids"]["input_a1"], feature["input_ids"]["output_a1"],  feature["input_ids"]["input_a2"], feature["input_ids"]["output_a2"],feature["input_ids"]["cots"])
            a2_inputs.extend(a2_input)
            a2_outputs.extend(a2_output) 
            
        input_a2 = [self.tokenizer.encode(input_text,add_eos=True) for input_text in a2_inputs]
        output_a2 = [self.tokenizer.encode(output_text,add_eos=True) for output_text in a2_outputs]
        
        input_ids_a2, labels_a2 = pad_tensors(input_a2, output_a2)
        
        
        # print(main_inputs)
        # print(main_outputs)
        # print(a1_inputs)
        # print(a1_outputs)
        # print(a2_inputs)
        # print(a2_outputs)
        # print("&&&&&&&&&&&&&&&&&&&&&&&&&&*****************************************")

        return {
        'tokens': input_ids,
        'labels': labels, 
        'input_ids_a1': input_ids_a1,
        'labels_a1': labels_a1,
        'input_ids_a2': input_ids_a2,
        'labels_a2':labels_a2
        }


    
# class CustomDataCollator:
#     def __init__(self, tokenizer):
#         self.tokenizer = tokenizer
#         self.main_prompt = """You are an expert Python code generator. Your task is to generate clean, efficient, and correct Python code based on the given question or problem description. Follow these guidelines:
#             1. Generate Python code with explanations in comments.
#             2. Use proper indentation and formatting for readability.
#             3. Implement the solution using Python best practices and optimal algorithms.
#             4. Ensure the code is complete and can be executed without additional modifications.
#             5. Include a brief comment at the beginning explaining the overall purpose of the code.

#             Question: {}

#             Code:
#         """

#     def __call__(self, features):
#         main_inputs = [self.main_prompt.format(feature["input_main"]) for feature in features]
#         main_outputs = [feature["output_main"] for feature in features]

#         max_length = max(
#             max(len(self.tokenizer.encode(inp)) for inp in main_inputs),
#             max(len(self.tokenizer.encode(out)) for out in main_outputs)
#         )
#         print(self.tokenizer)
#         import pdb; pdb.set_trace()

#         input_main_encoding = self.tokenizer(
#             main_inputs, 
#             return_tensors="pt"
#         )

#         output_main_encoding = self.tokenizer(
#             main_outputs, 
#             return_tensors="pt"
#         )

#         # We're not including A1 and A2 tasks here for simplicity, but you can add them if needed

#         return {
#             'input_ids': input_main_encoding["input_ids"],
#             'attention_mask': input_main_encoding["attention_mask"],
#             'labels': output_main_encoding["input_ids"],
#         }
        
        
def read_pickle_file(file_path):
        # Open the pickle file in binary read mode
    with open(file_path, 'rb') as file:
            # Load the content of the pickle file
        data = pickle.load(file)
        return data

file_path = '/home/vdhee/scratch/LLMcode/Train/Dataset/variable_1.pkl'
data = read_pickle_file(file_path)

variables={}
for t in ["train","test","validation"]:
        for j in data[t]:
                variables[j["task_id"]]=j["variable"]
    
    
    

#Loading thr COTS data from a json file    
file=open("/home/vdhee/scratch/LLMcode/Train/Dataset/cot_data_1.json")
data=json.load(file)
cots=dict()
for c in data:
    v={}
    for i,key in enumerate(c.keys()):
        if "Test Case " in key:
            v[i-5]=c[key]
    
    cots[c["question_id"]]=v
    
    



class LoRAFinetuneRecipeDistributed(FTRecipeInterface):
    """
    Distributed LoRA finetuning recipe for dense transformer-based LLMs such as Llama2. This recipe supports
    distributed training and can be run on a single node (1 to 8 GPUs).

    Features:
        - FSDP. Supported using PyTorch's FSDP APIs. This can be parameterized using the
            ``fsdp_sharding_strategy`` config option. You can pass any value supported by
            torch.distributed.fsdp.ShardingStrategy
            (https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.ShardingStrategy).
            For example, in your config, simply pass ``fsdp_sharding=NO_SHARD`` for DDP.

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

    def __init__(self, cfg, path,variables, cots) -> None:
        self.path=path
        self.variables=variables
        self.cots=cots
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
        self._resume_from_checkpoint = cfg.resume_from_checkpoint
        self._save_adapter_weights_only = cfg.get("save_adapter_weights_only", False)
        self._gradient_accumulation_steps = cfg.gradient_accumulation_steps
        self._fsdp_sharding_strategy = torch.distributed.fsdp.ShardingStrategy[
            cfg.get("fsdp_sharding_strategy", "FULL_SHARD")
        ]

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
        try:
            self.epochs_run = ckpt_dict[training.EPOCHS_KEY]

            # on mismatch, warn the user and prevent the override
            if self.seed != ckpt_dict[training.SEED_KEY]:
                warn(
                    message=(
                        "Config value for seed does not match the checkpoint value, "
                        f"using the checkpoint value: {ckpt_dict[training.SEED_KEY]}"
                    )
                )
                self.seed = ckpt_dict[training.SEED_KEY]
            if self.max_steps_per_epoch != ckpt_dict[training.MAX_STEPS_KEY]:
                warn(
                    message=(
                        "Config value for max_steps_per_epoch does not match the checkpoint value, "
                        f"using the checkpoint value: {ckpt_dict[training.MAX_STEPS_KEY]}"
                    )
                )
                self.max_steps_per_epoch = ckpt_dict[training.MAX_STEPS_KEY]

            # on mismatch, warn the user but allow the override
            if self.total_epochs != ckpt_dict[training.TOTAL_EPOCHS_KEY]:
                warn(
                    message=(
                        "Config value for total_epochs does not match the checkpoint value, "
                        f"using the config value: {self.total_epochs}"
                    )
                )

        except KeyError as e:
            raise KeyError(
                "Checkpoint does not contain the required keys needed for updating recipe state. "
                "Are you sure you passed in the right recipe checkpoint?"
            ) from e

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

        self._model_compile = cfg.get("compile", False)
        self._model = self._setup_model(
            cfg_model=cfg.model,
            enable_activation_checkpointing=cfg.enable_activation_checkpointing,
            base_model_state_dict=checkpoint_dict[training.MODEL_KEY],
            lora_weights_state_dict=(
                checkpoint_dict[training.ADAPTER_KEY]
                if self._resume_from_checkpoint
                else None
            ),
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
        self._sampler, self._dataloader = self.setup_data_(
            path=self.path,
            tokenizer=self._tokenizer,
            shuffle=cfg.shuffle,
            batch_size=cfg.batch_size,
        )

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

        # Set up profiler, returns DummyProfiler (nullcontext object with no-op `step` method)
        # if cfg is missing profiler key or if `cfg.profiler.enabled = False`
        self._profiler = self._setup_profiler(cfg.get(PROFILER_KEY, None))

        # Used to ignore labels for loss computation
        self.ignore_labels_cache = torch.full(
            (cfg.batch_size, 1), self._loss_fn.ignore_index, device=self._device
        )

    def _setup_profiler(
        self, cfg_profiler: Optional[DictConfig] = None
    ) -> Union[torch.profiler.profile, DummyProfiler]:
        """
        Parses the `profiler` section of top-level `cfg` and sets up profiler

        Args:
            cfg_profiler (Optional[DictConfig]): ``profiler`` section of the top-level ``cfg`` (the main config passed to
                `recipe.main`). Default None.

        Returns:
            profiler: Union[torch.profiler.profile, DummyProfiler] - DummyProfiler is a nullcontext with no-op methods
            for `start`, `stop`, and `step` that can be used in place of `torch.profiler.profile` if profiler is not enabled such
            that the instrumented training loop does not need to be changed profiling is disabled.

        The profiler config can be provided in configs under the `profiler` key with the following layout:

        .. code-block:: yaml
            profiler:
                enabled: bool

                #Output directory of trace artifacts
                output_dir: str

            #`torch.profiler.ProfilerActivity` types to trace
            cpu: bool
            cuda: bool

                #Trace options
                profile_memory: bool
                with_stack: bool
                record_shapes: bool
                with_flops: bool

            # `torch.profiler.schedule` options:
            # wait_steps -> wait, warmup_steps -> warmup, active_steps -> active, num_cycles -> repeat
            wait_steps: int
            warmup_steps: int
            active_steps: int
            num_cycles: int
        """
        # Missing profiler section in config, assume disabled
        if cfg_profiler is None:
            cfg_profiler = DictConfig({"enabled": False})

        # Check that component is included and set correctly
        if cfg_profiler.get("_component_", None) is None:
            cfg_profiler["_component_"] = "torchtune.utils.setup_torch_profiler"
        else:
            assert (
                cfg_profiler.get("_component_")
                == "torchtune.utils.setup_torch_profiler"
            ), "Only torch profiler supported currently: component must be `torchtune.utils.setup_torch_profiler`"

        profiler, profiler_cfg = config.instantiate(cfg_profiler)

        if self._is_rank_zero:
            log.info(f" Profiler config after instantiation: {profiler_cfg}")

            self.profiler_profile_memory = profiler_cfg.get("profile_memory", False)
            if profiler_cfg["enabled"]:
                self.profiler_wait_steps = profiler_cfg["wait_steps"]
                self.profiler_warmup_steps = profiler_cfg["warmup_steps"]
                self.profiler_active_steps = profiler_cfg["active_steps"]

        return profiler

    def _setup_model(
        self,
        cfg_model: DictConfig,
        enable_activation_checkpointing: bool,
        base_model_state_dict: Dict[str, Any],
        lora_weights_state_dict: Optional[Dict[str, Any]] = None,
    ) -> nn.Module:
        """
        Model initialization has some important considerations:
           a. To minimize GPU peak memory, we load the model on CPU with the right
              dtype. To ensure that we don't instantiate ``world_size`` number of models,
              we initialize on meta_device for all ranks other than rank 0.
           b. Rank 0 is also responsible for calling ``load_state_dict`` and loading the
              model weights from checkpoint.
           c. While wrapping the model with FSDP, we set ``sync_module_states``
              to TRUE and broadcast module params and buffers from rank 0.
           d. The ``device_id`` param ensures that the FSDP initialization happens on
              the correct device.
        """

        self._lora_rank = cfg_model.lora_rank
        self._lora_alpha = cfg_model.lora_alpha
        self._lora_attn_modules = list(cfg_model.lora_attn_modules)
        self._apply_lora_to_mlp = cfg_model.apply_lora_to_mlp
        self._apply_lora_to_output = getattr(cfg_model, "apply_lora_to_output", False)

        if self._is_rank_zero:
            log.info("FSDP is enabled. Instantiating Model on CPU for Rank 0 ...")
            init_start = time.perf_counter()

            with training.set_default_dtype(self._dtype):
                model = config.instantiate(cfg_model)

            log.info(
                f"Model instantiation took {time.perf_counter() - init_start:.2f} secs"
            )

            # The model contains LoRA params which won't have any matching keys in
            # the state dict. As a result, we need to load with strict=False.
            # Before loading the state dict, ensure the state dict keys for the base
            # model and adapters (if available) match the keys in the full LoRA model
            # This is a good sanity check to prevent silent errors
            validate_state_dict_for_lora(
                lora_attn_modules=cfg_model.lora_attn_modules,
                apply_lora_to_mlp=cfg_model.apply_lora_to_mlp,
                apply_lora_to_output=getattr(cfg_model, "apply_lora_to_output", False),
                full_model_state_dict_keys=model.state_dict().keys(),
                lora_state_dict_keys=(
                    lora_weights_state_dict.keys()
                    if lora_weights_state_dict is not None
                    else None
                ),
                base_model_state_dict_keys=base_model_state_dict.keys(),
            )

            # Load both the base model weights and (if available) the adapter weights. Both
            # of this should happen only on Rank 0
            model.load_state_dict(base_model_state_dict, strict=False)
            if lora_weights_state_dict:
                model.load_state_dict(lora_weights_state_dict, strict=False)

        else:
            # For non-zero ranks, load the model on meta device
            with training.set_default_dtype(self._dtype), torch.device("meta"):
                model = config.instantiate(cfg_model)

        if self._dtype == torch.bfloat16:
            model = model.to(torch.bfloat16)

        # LoRA hyper-params needed for merging weights while saving checkpoints
        self._lora_rank = cfg_model.lora_rank
        self._lora_alpha = cfg_model.lora_alpha

        # Note: this needs to be set before wrapping with FSDP
        self.adapter_params = get_adapter_params(model)
        set_trainable_params(model, self.adapter_params)

        model = FSDP(
            module=model,
            auto_wrap_policy=utils.lora_fsdp_wrap_policy(
                modules_to_wrap={modules.TransformerSelfAttentionLayer}
            ),
            sharding_strategy=self._fsdp_sharding_strategy,
            device_id=self._device,
            # this recipe does not currently support mixed precision training
            mixed_precision=None,
            # Ensure we broadcast params and buffers from rank 0
            sync_module_states=True,
            # Initialize empty modules on all non-zero ranks
            param_init_fn=(
                lambda module: (
                    module.to_empty(device=torch.device("cuda"), recurse=False)
                    if not self._is_rank_zero
                    else None
                )
            ),
        )

        # Ensure no params and buffers are on meta device
        utils.validate_no_params_on_meta_device(model)

        if enable_activation_checkpointing:
            utils.set_activation_checkpointing(
                model, auto_wrap_policy={modules.TransformerSelfAttentionLayer}
            )
        if self._is_rank_zero:
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
            # Note: technically we should check _contains_fsdp for
            # just the state dict of the adapter cfg, but should be equivalent
            opt_state_dict = FSDP.optim_state_dict_to_load(
                self._model, optimizer, opt_state_dict
            )
            optimizer.load_state_dict(opt_state_dict)

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
    path,
    tokenizer,
    shuffle: bool,
    batch_size: int):
        
        world_size, rank = utils.get_world_size_and_rank()

        ds = CustomDataset(path, self.variables, self.cots
    )

        sampler = DistributedSampler(
        ds, num_replicas=world_size, rank=rank, shuffle=shuffle, seed=0
    )

        collator = CustomDataCollator(tokenizer, self.variables)

        dataloader = DataLoader(
        dataset=ds,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collator,
    )

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
                config.instantiate(single_cfg_dataset, self._tokenizer)
                for single_cfg_dataset in cfg_dataset
            ]
            ds = ConcatDataset(datasets=datasets)
            packed = False
        else:
            ds = config.instantiate(cfg_dataset, self._tokenizer)
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
        - If the `self._save_adapter_weights_only` option is True, the checkpointer will save only the adapter weights

        To correctly resume from training, the adapter weights and recipe state must be provided along with the base model weights.
        """
        # final dict passed onto the checkpointer
        checkpoint_dict = {}

        intermediate_checkpoint = epoch + 1 < self.total_epochs
        # To prevent GPU memory from spiking during checkpoint save,
        # we consolidate the full model and optim state dicts on CPU for rank 0
        with FSDP.state_dict_type(
            self._model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
            FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            cpu_state_dict = self._model.state_dict()
            if intermediate_checkpoint:
                opt_state_dict = FSDP.optim_state_dict(self._model, self._optimizer)
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
                adapter_only=self._save_adapter_weights_only,
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

        self._profiler.start()
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

                # Start tracking CUDA memory for active steps for just the first epoch
                if (
                    self._is_rank_zero
                    and curr_epoch == 0
                    and self.profiler_profile_memory
                    and idx == self.profiler_wait_steps + self.profiler_warmup_steps
                ):
                    torch.cuda.memory._record_memory_history()

                # Both are shape [b, s]
                tokens, labels, input_ids_a1, labels_a1, input_ids_a2, labels_a2 = batch["tokens"], batch["labels"], batch["input_ids_a1"], batch["labels_a1"], batch["input_ids_a2"], batch["labels_a2"]
                # Get the attention mask and position ids from the dataset if they
                # exist. Currently, only sample packing in PackedDataset returns these
                #print_gpu_memory_usage(self._device)
                mask = batch.get("mask", None)  # shape [b, s, s]
                input_pos = batch.get("input_pos", None)  # shape [b, s]

                tokens = tokens.to(self._device)
                num_tokens += tokens.numel()
                labels = labels.to(self._device)
                #print_gpu_memory_usage(self._device)
                input_ids_a1 = input_ids_a1.to(self._device)
                #print_gpu_memory_usage(self._device)
                labels_a1 = labels_a1.to(self._device)
                #print_gpu_memory_usage(self._device)
                # input_ids_a2 = input_ids_a2.to(self._device)
                # labels_a2 = labels_a2.to(self._device)
                
                mask = mask.to(self._device) if mask is not None else None
                input_pos = (
                    input_pos.to(self._device) if input_pos is not None else None
                )

                logits = self._model(tokens, mask=mask, input_pos=input_pos)
                logits_a1 = self._model(input_ids_a1, mask=mask, input_pos=None)
                # logits_a2 = self._model(input_ids_a2, mask=mask, input_pos=None)
                # Shift labels to compute loss
                # equivalent to doing labels[..., 1:] and logits[..., :-1, :]
                # But this way we dont need to slice the logits. We just add an ignore index to labels.
                labels = torch.hstack(
                    (labels[..., 1:], self.ignore_labels_cache[: labels.shape[0]])
                )
                
                labels_a1 = torch.hstack((labels_a1[..., 1:], self.ignore_labels_cache[:labels_a1.shape[0]]))
                # labels_a2 = torch.hstack((labels_a2[..., 1:], self.ignore_labels_cache[:labels_a2.shape[0]]))
                if not isinstance(logits, list):
                    labels = labels.reshape(-1)
                    logits = logits.reshape(-1, logits.size(-1))
                    labels_a1 = labels_a1.reshape(-1)
                    logits_a1 = logits_a1.reshape(-1, logits_a1.size(-1))
                    # labels_a2 = labels_a2.reshape(-1)
                    # logits_a2 = logits_a2.reshape(-1, logits_a2.size(-1))

                # Compute loss
                loss = self._loss_fn(logits, labels)
                loss_a1 = self._loss_fn(logits_a1, labels_a1)
                # loss_a2 = self._loss_fn(logits_a2, labels_a2)
                # free logits otherwise it peaks backward memory
                total_loss = loss + loss_a1 

# Normalize by gradient accumulation steps
                total_loss = total_loss / self._gradient_accumulation_steps
                running_loss += total_loss

# Backpropagation
                total_loss.backward()
                print(total_loss,"##########################################@@@@@@@@@@@@@@@@@@@@@@@@@@@")
                del logits, logits_a1, input_ids_a1, labels_a1, input_ids_a2, labels_a2


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

                    # Stop tracking CUDA memory now that active steps are complete
                    if (
                        self._is_rank_zero
                        and curr_epoch == 0
                        and self.profiler_profile_memory
                        and idx
                        == self.profiler_wait_steps
                        + self.profiler_warmup_steps
                        + self.profiler_active_steps
                    ):
                        torch.cuda.memory._record_memory_history(enabled=None)

                    # Step profiler
                    # Note that this is called within gradient accumulation block, hence
                    # will include multiple forward / backward passes if gradient accumulation > 1
                    self._profiler.step()

            self.epochs_run += 1
            self.save_checkpoint(epoch=curr_epoch)

        self._profiler.stop()

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
    path="/home/vdhee/scratch/LLMcode/Train/torchtune/recipes/output.csv"

    recipe = LoRAFinetuneRecipeDistributed(cfg, path, variables,cots)
    recipe.setup(cfg=cfg)
    recipe.train()
    recipe.cleanup()


if __name__ == "__main__":
    sys.exit(recipe_main())


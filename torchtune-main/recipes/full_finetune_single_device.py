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

import json
import pickle
import random
from datasets import load_from_disk
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import Dataset

import torch
from omegaconf import DictConfig, ListConfig

from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, DistributedSampler

from torchtune import config, modules, training, utils
from torchtune.data import padded_collate
from torchtune.datasets import ConcatDataset
from torchtune.recipe_interfaces import FTRecipeInterface
from torchtune.utils import DummyProfiler, PROFILER_KEY

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
    cots[c["question_id"]]={}
    n_len = len(c["Test Cases"])
    for num in range(n_len):
      if f"Test Case {num+1}" in c:
        cots[c["question_id"]][num] = c[f"Test Case {num+1}"]
        
        
json_data=json.load(open("/home/vdhee/scratch/LLMcode/Train/Dataset/output-1.json"))

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
    def __init__(self, data, tokenizer, json_data,  max_length = 1024, batch_size=4, lambda_1=1.0):
        self.data = data
        self.code_gen = data.select_columns(['task_id', 'prompt', 'code'])
        self.aux_data = data.select_columns(['task_id', 'prompt', 'code', 'test_imports', 'test_list', 'variables'])
        self.json_data = json_data
        self.ls=data.select_columns(["test_list"])
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
        code_gen = "Question: " + self.code_gen[id]['prompt'] + "\nFor example, this is how the function name and number of arguments shouls look like: " + self.ls[id]["test_list"][0].split("assert ")[1].strip() +'\n' + "Answer:\n"
        out="```python " + self.code_gen[id]['code'] + "```"
        # print(code_gen)
        return code_gen, out
  
  
    def __len__(self):
        return len(self.data)
  
    def __getitem__(self, idx):
        
        prom="""You are a highly skilled Python code generator. Your role is to generate efficient, clean, and correct Python functions based on the provided problem. Follow these specific guidelines carefully:
1. Only write the Python function—no explanations, comments, or additional text.
2. Ensure proper indentation and formatting for readability.
3. The function should appear once, with no extraneous output.
4. Do not provide any explanations or corrections.
5. Stop after producing the first valid output.
6. Make sure to generate the code in a similar fashion to the example below.
7. Enclose the function in triple backticks with 'python' language specifier.

For example:
Question: Write a function to find the maximum run of uppercase characters in the given string.
For example, this is how the function name and number of arguments should look like: max_run_uppercase('GeMKSForGERksISBESt') == 5
Answer: 
```python
def max_run_uppercase(test_str):
    cnt = 0
    res = 0
    for idx in range(len(test_str)):
        if test_str[idx].isupper():
            cnt += 1
        else:
            res = max(res, cnt)
            cnt = 0
    res = max(res, cnt)  # Check once more after the loop in case the string ends with an uppercase run
    return res
'''

Now Write the code for the following question 
"""
        self.count += 1
        self.count = self.count % self.batch_size
        # print(self.count)
        if self.count < self.split:
            id = self.code_gen_idx[self.code_ptr]
            self.code_ptr += 1
            self.code_ptr = self.code_ptr % self.len_data
            inp, output = self.process_code(id)  
            token = self.tokenizer.encode(prom+inp, add_eos=True)
            label = self.tokenizer.encode(prom+inp+output, add_eos=True)
          
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
                a=random.choice(range(len(inp)))
                output = self.json_data[str(task_id)]['a2'][str(test_id)]['labels']
                b=random.choice(range(len(output)))
            
            token = self.tokenizer.encode(inp, add_eos=True)
            label = self.tokenizer.encode(inp + output, add_eos=True)
          
            return {
              "tokens" : token,
              "labels": label
            }



class FullFinetuneRecipeSingleDevice(FTRecipeInterface):
    """
    Full finetuning recipe for dense transformer-based LLMs such as Llama2. This recipe is optimized
    for single GPU training. Training on CPU is not supported.

    Features:
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

                Total Batch Size = batch_size * gradient accumulation steps.

            For example: with batch_size=1 and gradient_accumulation_steps=32 we get a total batch size of 32.

            Gradient accumulation is especially useful when you are memory constrained. In this case,
            accumulating gradients might give you better training speed than enabling activation
            checkpointing.

        - Optimizer in Backward. Fusing the optimizer step into the backward pass helps reduce the memory
            footprint associated with gradients. This can be especially helpful when you are memory
            constrained. Note that users can only use ONE of gradient accumulation or optimizer in backward.
            These features currently do not work together. For more details on optimizer in backward, please
            see this tutorial: https://pytorch.org/tutorials/intermediate/optimizer_step_in_backward_tutorial.html

        - Lower precision optimizers. This recipe supports lower-precision optimizers from the bitsandbytes
            library (https://huggingface.co/docs/bitsandbytes/main/en/index). We've tested the recipe with
            8-bit AdamW and Paged AdamW. These optimizers are especially helpful when you are memory constrained
            since they help reduce the memory footprint associated with the optimizer states.

        - Checkpointing. Model weights are checkpointed both at the end of each epoch and at the end of
            training. Optimizer State and recipe state (seed, total_epochs, number of epochs run etc) are
            only saved at the end of a given epoch and used in case of resuming training.

            Resuming training is controlled by the ``resume_from_checkpoint`` flag. Mid-epoch checkpointing is
            currently not supported.

            For more details on the checkpointer, please take a look at
            our checkpointer deepdive (https://pytorch.org/torchtune/main/deep_dives/checkpointer.html).

        - Logging. Terminal, Disk, WandB and TensorBoard are all supported.

    For a full list of example configs for this recipe, run ``tune ls`` on the command line. Each config
    has example commands for how to kick-off training.

    Args:
        cfg (DictConfig): OmegaConf object parsed from yaml file

    Raises:
        RuntimeError: If ``dtype`` is set to fp16.
        RuntimeError: If ``dtype`` is set to bf16 and the hardware does not support bf16.
        RuntimeError: If ``gradient_accumulation_steps > 1`` and ``optimizer_in_bwd`` is `True`.
    """

    def __init__(self, cfg, new, json_data) -> None:
        self._device = utils.get_device(device=cfg.device)
        self._dtype = training.get_dtype(cfg.dtype, device=self._device)
        # Disable for fp16, as we haven't validated "full" fp16 with this recipe, nor
        # enabled necessary features such as gradient scaling.
        if self._dtype == torch.float16:
            raise RuntimeError(
                "full fp16 training is not supported with this recipe. Please use bf16 or fp32 instead."
            )

        # logging attributes
        self._output_dir = cfg.output_dir
        self._log_every_n_steps = cfg.get("log_every_n_steps", 1)
        self._log_peak_memory_stats = cfg.get("log_peak_memory_stats", False)

        # Training cfg
        self._resume_from_checkpoint = cfg.resume_from_checkpoint
        self._gradient_accumulation_steps = cfg.gradient_accumulation_steps
        self._optimizer_in_bwd = cfg.optimizer_in_bwd

        # TODO: find a better place / way to perform validation of args that don't yet
        # compose with each other.
        if self._gradient_accumulation_steps > 1 and self._optimizer_in_bwd:
            raise RuntimeError(
                "Gradient accumulation is not supported with optimizer in bwd."
                "Please set gradient_accumulation_steps=1, or optimizer_in_bwd=False."
            )

        # These are public properties which are updated by the checkpoint loader
        # when ``resume_from_checkpoint`` is `True` or validated in tests
        self.seed = utils.set_seed(seed=cfg.seed)
        self.epochs_run = 0
        self.total_epochs = cfg.epochs
        self.max_steps_per_epoch = cfg.max_steps_per_epoch
        self.global_step = 0
        self.json_data=json_data
        self.new=new

    def load_checkpoint(self, cfg_checkpointer: DictConfig) -> Dict[str, Any]:
        """
        Extract the checkpoint state from file and validate. If resume_from_checkpoint
        is True, this also includes the recipe state.
        """
        self._checkpointer = config.instantiate(
            cfg_checkpointer,
            resume_from_checkpoint=self._resume_from_checkpoint,
        )
        checkpoint_dict = self._checkpointer.load_checkpoint()

        if self._resume_from_checkpoint:
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
        Sets up the recipe state correctly. This includes setting recipe attributes based
        on the ``resume_from_checkpoint`` flag.
        """
        self._metric_logger = config.instantiate(cfg.metric_logger)

        # log config with parameter override
        self._metric_logger.log_config(cfg)

        ckpt_dict = self.load_checkpoint(cfg.checkpointer)

        # ``_setup_model`` handles initialization and loading the state dict. This method
        # should be called before ``_setup_optimizer`` since transforming the optimizer
        # state dict requires the model
        self._model_compile = cfg.compile
        self._model = self._setup_model(
            cfg_model=cfg.model,
            enable_activation_checkpointing=cfg.enable_activation_checkpointing,
            compile_model=self._model_compile,
            model_state_dict=ckpt_dict[training.MODEL_KEY],
        )
        self._tokenizer = config.instantiate(cfg.tokenizer)
        log.info("Tokenizer is initialized from file.")

        # _setup_optimizer should take in ckpt_dict only if training is resumed from
        # checkpoint. Transforming the opt state dict is handled by this method
        self._optimizer = self._setup_optimizer(
            cfg_optimizer=cfg.optimizer,
            optimizer_in_bwd=cfg.optimizer_in_bwd,
            opt_state_dict=(
                ckpt_dict[training.OPT_KEY] if self._resume_from_checkpoint else None
            ),
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
        # setup after both of these are initialized
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
        #
        # Number of training steps in each epoch depends on the number of batches produced
        # by the dataloader, the max_steps_per_epoch param set by the user and the
        # gradient_accumulation_steps param. This value is used for logging and tracking
        # training state. The computation should happen after the dataloader has been setup
        self._steps_per_epoch = (
            len(self._dataloader) // self._gradient_accumulation_steps
        )
        if (
            self.max_steps_per_epoch is not None
            and self.max_steps_per_epoch < self._steps_per_epoch
        ):
            self._steps_per_epoch = self.max_steps_per_epoch
        self.global_step = self.epochs_run * self._steps_per_epoch

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
        compile_model: bool,
        model_state_dict: Dict[str, Any],
    ) -> nn.Module:
        """
        Set up the model including enabling activation checkpointing.
        """
        with training.set_default_dtype(self._dtype), self._device:
            model = config.instantiate(cfg_model)

        if compile_model:
            log.info("Compiling model layers with torch.compile...")
            backend = os.environ.get("TORCH_COMPILE_BACKEND", "inductor")
            for m in reversed(list(model.modules())):
                if isinstance(m, modules.transformer.TransformerSelfAttentionLayer):
                    m.compile(backend=backend)

        if enable_activation_checkpointing:
            utils.set_activation_checkpointing(
                model, auto_wrap_policy={modules.TransformerSelfAttentionLayer}
            )

        model.load_state_dict(model_state_dict)

        # Validate model was loaded in with the expected dtype.
        training.validate_expected_param_dtype(
            model.named_parameters(), dtype=self._dtype
        )
        log.info(f"Model is initialized with precision {self._dtype}.")

        if self._device.type == "cuda":
            memory_stats = utils.get_memory_stats(device=self._device)
            utils.log_memory_stats(memory_stats)

        return model

    def _setup_optimizer(
        self,
        cfg_optimizer: DictConfig,
        optimizer_in_bwd: bool = False,
        opt_state_dict: Optional[Dict[str, Any]] = None,
    ) -> Optional[Optimizer]:
        """
        Set up the optimizer. This method also handles loading the optimizer state_dict, if specified.
        """
        if optimizer_in_bwd:
            # Maintain a dict of optims for every parameter.
            optim_dict = {
                p: config.instantiate(cfg_optimizer, [p])
                for p in self._model.parameters()
            }
            # Register optimizer step hooks on the model to run optimizer in backward.
            utils.register_optim_in_bwd_hooks(model=self._model, optim_dict=optim_dict)
            # Create a wrapper for checkpoint save/load of optimizer states when running in backward.
            self._optim_ckpt_wrapper = utils.create_optim_in_bwd_wrapper(
                model=self._model, optim_dict=optim_dict
            )
            # Load optimizer states. If optimizer states are being restored in an optimizer in backward
            # run, these need to have been saved with the same setting. Cannot restore from runs that did not
            # use optimizer in backward.
            if opt_state_dict is not None:
                try:
                    self._optim_ckpt_wrapper.load_state_dict(opt_state_dict)
                except BaseException as e:
                    raise RuntimeError(
                        "Failed loading in-backward optimizer checkpoints."
                        "Please make sure run being restored from was using in-backward optimizer."
                    ) from e
            log.info("In-backward optimizers are set up.")
            return None
        else:
            optimizer = config.instantiate(cfg_optimizer, self._model.parameters())

            if opt_state_dict:
                optimizer.load_state_dict(opt_state_dict)
            log.info("Optimizer is initialized.")
            return optimizer
        
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
            ds,
            num_replicas=1,
            rank=0,
            shuffle=shuffle,
            seed=0,
        )
        dataloader = DataLoader(
            dataset=ds,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=partial(
                padded_collate,
                padding_idx=self._tokenizer.pad_id,
                ignore_idx=self._loss_fn.ignore_index,
            )
            if not packed
            else None,
        )

        log.info("Dataset and Sampler are initialized.")

        return sampler, dataloader

    def save_checkpoint(self, epoch: int) -> None:
        """
        Save state dict to file. The recipe save_checkpoint method is responsible for
        correctly creating the checkpoint dict and passing to the checkpointer.
        """
        ckpt_dict = {training.MODEL_KEY: self._model.state_dict()}
        # if training is in-progress, checkpoint the optimizer state as well
        if epoch + 1 < self.total_epochs:
            ckpt_dict.update(
                {
                    training.SEED_KEY: self.seed,
                    training.EPOCHS_KEY: self.epochs_run,
                    training.TOTAL_EPOCHS_KEY: self.total_epochs,
                    training.MAX_STEPS_KEY: self.max_steps_per_epoch,
                }
            )
            if not self._optimizer_in_bwd:
                ckpt_dict[training.OPT_KEY] = self._optimizer.state_dict()
            else:
                ckpt_dict[training.OPT_KEY] = self._optim_ckpt_wrapper.state_dict()
        self._checkpointer.save_checkpoint(
            ckpt_dict,
            epoch=epoch,
            intermediate_checkpoint=(epoch + 1 < self.total_epochs),
        )

    def _loss_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Both are shape [b, s]
        tokens, labels = batch["tokens"], batch["labels"]
        # self.generate(tokens)
        # Get the attention mask and position ids from the dataset if they
        # exist. Currently, only sample packing in PackedDataset returns these
        mask = batch.get("mask", None)  # shape [b, s, s]
        input_pos = batch.get("input_pos", None)  # shape [b, s]

        logits = self._model(tokens, mask=mask, input_pos=input_pos)
        
        # for lst in logits:
        #     token_ids = torch.argmax(lst, dim=-1)
        #     token_ids_list = token_ids.flatten().tolist()
        #     print(self._tokenizer.decode(token_ids_list))

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

        return loss

    def train(self) -> None:
        """
        The core training loop. Supports training on subsets of the dataset using the
        ``max_steps_per_epoch``.
        """
        if self._model_compile:
            log.info(
                "NOTE: torch.compile is enabled and model is compiled in first forward. Expect a relatively slow first iteration."
            )
        # zero out the gradients before starting training
        if not self._optimizer_in_bwd:
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

            pbar = tqdm(total=self._steps_per_epoch)
            for idx, batch in enumerate(self._dataloader):
                if (
                    self.max_steps_per_epoch is not None
                    and (idx // self._gradient_accumulation_steps)
                    == self.max_steps_per_epoch
                ):
                    break

                # Start tracking CUDA memory for active steps for just the first epoch
                if (
                    curr_epoch == 0
                    and self.profiler_profile_memory
                    and idx == self.profiler_wait_steps + self.profiler_warmup_steps
                ):
                    torch.cuda.memory._record_memory_history()

                batch = {k: v.to(self._device) for k, v in batch.items()}
                num_tokens += batch["tokens"].numel()

                loss = self._loss_step(batch)
                loss = loss / self._gradient_accumulation_steps
                running_loss += loss
                loss.backward()
                # self.generate(batch["tokens"])

                # Step with optimizer
                if (idx + 1) % self._gradient_accumulation_steps == 0:
                    if not self._optimizer_in_bwd:
                        self._optimizer.step()
                        self._optimizer.zero_grad(set_to_none=True)

                    self.global_step += 1

                    loss_to_log = running_loss.item()
                    pbar.update(1)
                    pbar.set_description(
                        f"{curr_epoch + 1}|{self.global_step}|Loss: {loss_to_log}"
                    )

                    # Log per-step metrics
                    if self.global_step % self._log_every_n_steps == 0:
                        time_per_step = time.perf_counter() - t0
                        log_dict = {
                            "loss": loss_to_log,
                            # NOTE: for optim in backward, this assumes all optimizers have the same LR. This is currently
                            # true since we don't expose the ability to configure this yet.
                            "lr": (
                                self._optim_ckpt_wrapper.get_optim_key("lr")
                                if self._optimizer_in_bwd
                                else self._optimizer.param_groups[0]["lr"]
                            ),
                            "tokens_per_second_per_gpu": num_tokens / time_per_step,
                        }
                        if self._device.type == "cuda" and self._log_peak_memory_stats:
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
                    curr_epoch == 0
                    and self.profiler_profile_memory
                    and idx
                    == self.profiler_wait_steps
                    + self.profiler_warmup_steps
                    + self.profiler_active_steps
                ):
                    torch.cuda.memory._record_memory_history(enabled=None)

                # Step the profiler
                # Note we are stepping each batch, which might not include optimizer step in the trace
                # if the schedule cycle doesn't align with gradient accumulation.
                self._profiler.step()

            self.epochs_run += 1
            self.save_checkpoint(epoch=curr_epoch)

        self._profiler.stop()

    def cleanup(self) -> None:
        self._metric_logger.close()
    
        
        
    def generate(self, tokens):
        prompt = tokens[0]
        custom_generate_next_token = None

    # since quantized model uses torch.compile to get speedup, it needs a warm up / prefill run
    # to get the accurate performance measurement
        max_new_tokens: 900
        temperature: 0.8  # 0.8 and 0.6 are popular values to try
        top_k: 300

        t0 = time.perf_counter()

    # Adding torch.no_grad to avoid computing gradients during generation
        with torch.no_grad():
            generated_tokens = utils.generate(
            model=self._model,
            prompt=prompt,
            max_generated_tokens=300,
            temperature=0.8,
            top_k=300,
            stop_tokens=self._tokenizer.stop_tokens,
            custom_generate_next_token=custom_generate_next_token,
        )
        t = time.perf_counter() - t0
        print(self._tokenizer.decode(generated_tokens[0]))
        print("*" * 1000)



        
        
        
    


@config.parse
def recipe_main(cfg: DictConfig) -> None:
    """
    Entry point for the recipe.

    Configurable parameters are read in the following order:
        - Parameters specified in config (see available configs through ``tune ls``)
        - Overwritten by arguments from the command-line
    """
    config.log_config(recipe_name="FullFinetuneRecipeSingleDevice", cfg=cfg)
    recipe = FullFinetuneRecipeSingleDevice(cfg, new, json_data)
    recipe.setup(cfg=cfg)
    recipe.train()
    recipe.cleanup()


if __name__ == "__main__":
    sys.exit(recipe_main())
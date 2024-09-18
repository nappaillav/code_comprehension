# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import itertools
import sys
import json
import time
from typing import Any, Dict, List, Optional, Union
from datasets import load_from_disk
import torch
from omegaconf import DictConfig
from torch import nn

from torchtune import config, training, utils
from torchtune.config._utils import _get_component_from_path
from torchtune.data import ChatFormat, InstructTemplate, Message

logger = utils.get_logger("DEBUG")


class InferenceRecipe:
    """
    Recipe for generating tokens from a dense Transformer-based LLM.

    Currently this recipe supports single-GPU generation only. Speculative
    decoding is not supported.

    For more details on how to use this recipe for generation, please see our
    tutorial: https://pytorch.org/torchtune/main/tutorials/e2e_flow.html#generation

    For using this recipe with a quantized model, please the following section of
    the above tutorial:
    https://pytorch.org/torchtune/main/tutorials/e2e_flow.html#speeding-up-generation-using-quantization
    """

    def __init__(self, cfg: DictConfig) -> None:
        self._device = utils.get_device(device=cfg.device)
        self._dtype = training.get_dtype(dtype=cfg.dtype, device=self._device)
        self._quantizer = config.instantiate(cfg.quantizer)
        self._quantization_mode = training.get_quantizer_mode(self._quantizer)

        utils.set_seed(seed=cfg.seed)

    def setup(self, cfg: DictConfig) -> None:
        checkpointer = config.instantiate(cfg.checkpointer)
        if self._quantization_mode is None:
            ckpt_dict = checkpointer.load_checkpoint()
        else:
            # weights_only needs to be False when loading a quantized model
            # currently loading a quantized model is only supported with the
            # FullModelTorchTuneCheckpointer
            ckpt_dict = checkpointer.load_checkpoint(weights_only=False)

        self._model = self._setup_model(
            model_cfg=cfg.model,
            model_state_dict=ckpt_dict[training.MODEL_KEY],
            enable_kv_cache=cfg.enable_kv_cache,
        )
        self._tokenizer = config.instantiate(cfg.tokenizer)

    def _setup_model(
        self,
        model_cfg: DictConfig,
        model_state_dict: Dict[str, Any],
        enable_kv_cache: bool = True,
    ) -> nn.Module:
        with training.set_default_dtype(self._dtype), self._device:
            model = config.instantiate(model_cfg)

        if self._quantization_mode is not None:
            model = self._quantizer.quantize(model)
            model = model.to(device=self._device, dtype=self._dtype)

        model.load_state_dict(model_state_dict)

        # Validate model was loaded in with the expected dtype.
        training.validate_expected_param_dtype(
            model.named_parameters(), dtype=self._dtype
        )
        logger.info(f"Model is initialized with precision {self._dtype}.")

        # Ensure the cache is setup on the right device
        if enable_kv_cache:
            with self._device:
                model.setup_caches(batch_size=1, dtype=self._dtype)

        return model

    def convert_prompt_to_tokens(
        self,
        prompt: Union[DictConfig, str],
        chat_format: Optional[ChatFormat],
        instruct_template: Optional[InstructTemplate],
    ) -> List[Message]:
        """
        Either:
        (1) a raw string is passed as the prompt, in which case we call tokenizer.encode directly, or
        (2) a DictConfig is passed as the prompt. In this case there are three possibilities:
            (a) an InstructTemplate is provided. Since instruct templates output a string, we will
                call tokenizer.encode on the output of the instruct template.
            (b) a ChatFormat is provided. Since chat formats output a list of messages, we will
                call tokenizer.tokenize_messages on the output of the chat format.
            (c) neither an InstructTemplate nor a ChatFormat is provided. In this case we will
                convert the DictConfig to a list of messages and call tokenizer.tokenize_messages directly.
        """

        # Should only be chat-style prompt or instruct-style prompt
        if chat_format and instruct_template:
            raise ValueError(
                "Cannot pass both chat format and instruct template for generation"
            )

        # If instruct template is provided, assert that the prompt is a DictConfig
        # and apply it
        if instruct_template:
            if not isinstance(prompt, DictConfig):
                raise ValueError("Cannot apply instruct template to raw string")
            instruct_template = _get_component_from_path(instruct_template)
            prompt = instruct_template.format(prompt)

        # To hit this block, either the raw prompt is a string or an
        # instruct template has been provided to convert it to a string
        if isinstance(prompt, str):
            return self._tokenizer.encode(prompt, add_bos=True, add_eos=False)

        # dict.items() will respect order for Python >= 3.7
        else:
            messages = [Message(role=k, content=v) for k, v in prompt.items()]
            messages += [Message(role="assistant", content="")]
            if chat_format:
                chat_format = _get_component_from_path(chat_format)
                messages = chat_format.format(messages)
            return self._tokenizer.tokenize_messages(messages)[0]

    @torch.no_grad()
    def generate(self, cfg, prompt1,ids_, prom):
        tokens = self.convert_prompt_to_tokens(
            prompt1, cfg.get("chat_format", None), cfg.get("instruct_template", None)
        )
        prompt = torch.tensor(tokens, dtype=torch.int, device=self._device)

        custom_generate_next_token = None

        # since quantized model uses torch.compile to get speedup, it needs a warm up / prefill run
        # to get the accurate performance measurement
        if self._quantization_mode is not None:
            logger.info("Starting compilation to improve generation performance ...")
            custom_generate_next_token = torch.compile(
                utils.generate_next_token, mode="max-autotune", fullgraph=True
            )
            t0 = time.perf_counter()
            _ = utils.generate(
                model=self._model,
                prompt=prompt,
                max_generated_tokens=2,
                temperature=cfg.temperature,
                top_k=cfg.top_k,
                stop_tokens=self._tokenizer.stop_tokens,
                custom_generate_next_token=custom_generate_next_token,
            )
            t = time.perf_counter() - t0
            logger.info(f"Warmup run for quantized model takes: {t:.02f} sec")

        t0 = time.perf_counter()
        generated_tokens = utils.generate(
            model=self._model,
            prompt=prompt,
            max_generated_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_k=cfg.top_k,
            stop_tokens=self._tokenizer.stop_tokens,
            custom_generate_next_token=custom_generate_next_token,
        )
        t = time.perf_counter() - t0
        
        o=self._tokenizer.decode(generated_tokens[0])

        logger.info(o)
        
        result = {
        "id": ids_,
        "prompt": prompt1,
        "final_answer": o
    }

    # Append the result to the JSON file
        append_to_json("/home/vdhee/scratch/LLMcode/Train/full_finetuning_results_json-2/output-0.5-1.0/output-humaneval.json", result)

        model_size = sum(
            [
                p.numel() * p.dtype.itemsize
                for p in itertools.chain(
                    self._model.parameters(), self._model.buffers()
                )
            ]
        )

        tokens_generated = len(generated_tokens[0]) - prompt.size(0)
        tokens_sec = tokens_generated / t
        logger.info(
            f"Time for inference: {t:.02f} sec total, {tokens_sec:.02f} tokens/sec"
        )
        logger.info(f"Bandwidth achieved: {model_size * tokens_sec / 1e9:.02f} GB/s")
        logger.info(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")
        
def append_to_json(file_path, data):
    try:
        # Read existing data from the file
        with open(file_path, "r") as f:
            file_data = json.load(f)
    except FileNotFoundError:
        # If the file does not exist, create an empty list
        file_data = []

    # Append the new data to the existing list
    file_data.append(data)

    # Write the updated data back to the file
    with open(file_path, "w") as f:
        json.dump(file_data, f, indent=4)


@config.parse
def main(cfg: DictConfig) -> None:
    config.log_config(recipe_name="InferenceRecipe", cfg=cfg)
    recipe = InferenceRecipe(cfg=cfg)
    recipe.setup(cfg=cfg)
    
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
    ds=load_from_disk("/home/vdhee/scratch/LLMcode/Train/Dataset/human_val")
    lst=ds["test"]["prompt"]
    for i in range(len(lst)):
        #ques="".join(lst[i].split("\"\"\"")[1].split(">>>")[0].strip().split("\n"))
        final_prompt=prom + "\nQuestion: " + lst[i] + "\nAnswer: "
        recipe.generate(cfg,final_prompt,i,prom)
    # lst=ds["test"]["text"]
    # ids=ds["test"]["task_id"]
    # test_lists=ds["test"]["test_list"]
    # for j in range(len(lst)):
    #     p,i,test_list=lst[j], ids[j], test_lists[j]
    #     final_prompt=prom+"\nQuestion: " + p + "\nFor example, this is how the function name and number of arguments should look like: " + test_list[0].split("assert ")[1].strip()
    #     recipe.generate(cfg,final_prompt,i,prom)


if __name__ == "__main__":
    sys.exit(main())

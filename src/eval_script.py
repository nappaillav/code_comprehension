import json
import os
from os import PathLike
from typing import List, Optional
import math, sys, re, heapq
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)

from evalplus.provider import DecoderBase, make_model
from evalplus.sanitize import sanitize
from evalplus.data.utils import (
    stream_jsonl,
)
from evalplus.eval.utils import time_limit

EVAL_OVERRIDE_PATH = os.environ.get("EVAL_OVERRIDE_PATH", './test.jsonl')

def get_eval():
    eval_path = EVAL_OVERRIDE_PATH
    eval = {task["task_id"]: task for task in stream_jsonl(eval_path)}
    return eval

def trusted_exec(code):
    """Execute trusted code in place."""
    exec_globals = {}
    exec(code, exec_globals)

def trusted_check_exec(code):
    """Check trusted_exec success."""
    try:
        with time_limit(seconds=1.0):
            trusted_exec(code)
    except Exception:
        return False
    return True

def codegen(
    target_path: PathLike,
    model: DecoderBase,
    dataset: str,
    greedy=False,
    n_samples=1,
    id_range=None,
    version="default",
    resume=True,
):
    task2nexist = {}
    if resume and target_path.endswith(".jsonl") and os.path.isfile(target_path):
        with open(target_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                task_id = json.loads(line)["task_id"]
                task2nexist[task_id] = task2nexist.get(task_id, 0) + 1

    if target_path.endswith(".jsonl"):
        raw_target_path = target_path.replace(".jsonl", ".raw.jsonl")
    else:
        raw_target_path = target_path + ".raw"
        os.makedirs(target_path, exist_ok=True)

    print(f"Sanitized code outputs will be saved to {target_path}")
    print(f"Raw outputs will be saved to {raw_target_path}")

    with Progress(
        TextColumn(f"{dataset} •" + "[progress.percentage]{task.percentage:>3.0f}%"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
    ) as p:
        if dataset == "humaneval":
            from evalplus.data import get_human_eval_plus

            dataset = get_human_eval_plus(version=version)
        elif dataset == "mbpp":
            from evalplus.data import get_mbpp_plus

            dataset = get_mbpp_plus(version=version)
        elif dataset =='eval':
            dataset = get_eval()
        score = 0
        total = len(dataset)
        for task_id, task in p.track(dataset.items()):
            assertion_str = task['assertion']
            if id_range is not None:
                id_num = int(task_id.split("/")[1])
                low, high = id_range
                if id_num < low or id_num >= high:
                    p.console.print(f"Skipping {task_id} as it is not in {id_range}")
                    continue

            if not target_path.endswith(".jsonl"):
                p_name = task_id.replace("/", "_")
                os.makedirs(os.path.join(target_path, p_name), exist_ok=True)
                task2nexist[task_id] = len(
                    [
                        f
                        for f in os.listdir(os.path.join(target_path, p_name))
                        if f.endswith(".py")
                    ]
                )

            n_more_samples = n_samples
            log = f"Codegen: {task_id} @ {model}"
            if resume and task2nexist.get(task_id, 0) > 0:
                log += f" (resuming from {task2nexist[task_id]})"
                n_more_samples -= task2nexist[task_id]

            p.console.print(log)

            sidx = n_samples - n_more_samples
            while sidx < n_samples:
                prompt = task["prompt"].strip() + "\n"
                tests = task['assertion']
                outputs = model.codegen(
                    prompt,
                    do_sample=not greedy,
                    num_samples=n_samples - sidx,
                )
                assert outputs, "No outputs from model!"
                for impl in outputs:
                    solution = prompt + impl if model.is_direct_completion() else impl
                    sanitized_solution = sanitize(
                        solution, entrypoint=task["entry_point"]
                    )
                    if trusted_check_exec(sanitized_solution + '\n'+ tests):
                        score += 1
                    # else:
                    #     print(sanitized_solution + '\n'+ tests)
                    #     print('-------')
                    if target_path.endswith(".jsonl"):
                        # Writing the sanitized version
                        with open(target_path, "a") as f:
                            f.write(
                                json.dumps(
                                    {"task_id": task_id, "solution": sanitized_solution, "tests": tests}
                                )
                                + "\n"
                            )

                        # Writing the raw version
                        with open(raw_target_path, "a") as f:
                            f.write(
                                json.dumps({"task_id": task_id, "solution": solution})
                                + "\n"
                            )
                    else:
                        # Writing the sanitized version
                        with open(
                            os.path.join(target_path, p_name, f"{sidx}.py"),
                            "w",
                            encoding="utf-8",
                        ) as f:
                            f.write(sanitized_solution)

                        # Writing the raw version
                        with open(
                            os.path.join(raw_target_path, p_name, f"{sidx}.py"),
                            "w",
                            encoding="utf-8",
                        ) as f:
                            f.write(solution)
                    sidx += 1

    return score/total

def run_codegen(
    model: str,
    dataset: str,
    root: str = "evalplus_results",
    bs: Optional[int] = None,
    n_samples: int = 1,
    temperature: float = 0.0,
    resume: bool = True,
    greedy: bool = False,
    id_range: List = None,
    version: str = "default",
    backend: str = "vllm",
    force_base_prompt: bool = False,
    base_url: str = None,
    tp: int = 1,
    evalperf_type: str = None,  # For EvalPerf
    jsonl_fmt: bool = True,
):
    assert dataset in ["humaneval", "mbpp", "eval"], f"Invalid dataset {dataset}"
    assert backend in ["vllm", "hf", "openai"]
    assert evalperf_type is None or evalperf_type in [
        "instruct",
        "perf-instruct",
        "perf-CoT",
    ]

    if greedy and (temperature != 0 or bs != 1 or n_samples != 1):
        temperature = 0.0
        bs = 1
        n_samples = 1
        print("Greedy decoding ON (--greedy): setting bs=1, n_samples=1, temperature=0")

    if id_range is not None:
        assert len(id_range) == 2, "id_range must be a list of length 2"
        assert id_range[0] < id_range[1], "id_range must be increasing"
        id_range = tuple(id_range)

    if bs is None:
        bs = min(n_samples, 32)
        print(f"Setting batch size to {bs}")

    # Make project dir
    os.makedirs(root, exist_ok=True)
    # Make dataset dir
    os.makedirs(os.path.join(root, dataset), exist_ok=True)

    # Model instructions
    instruction_prefix = "Please provide a self-contained Python script that solves the following problem in a markdown code block:"
    response_prefix = "Below is a Python script with a self-contained function that solves the problem and passes corresponding tests:"

    if evalperf_type == "perf-instruct":
        instruction_prefix = "Please provide an efficient and self-contained Python script that solves the following problem in a markdown code block:"
        response_prefix = "Below is a Python script with a self-contained function that efficiently solves the problem and passes corresponding tests:"
    elif evalperf_type == "perf-CoT":
        instruction_prefix = "Think step by step: please provide an efficient and self-contained Python script that solves the following problem in a markdown code block:"
        response_prefix = "Below is a Python script with a self-contained function that efficiently solves the problem and passes corresponding tests:"
    elif evalperf_type is not None and evalperf_type != "instruct":
        raise ValueError(f"Invalid evalperf_type: {evalperf_type}")

    # Model creation
    model_runner = make_model(
        model=model,
        backend=backend,
        batch_size=bs,
        temperature=temperature,
        force_base_prompt=force_base_prompt,
        dataset=dataset,
        base_url=base_url,
        tp=tp,
        instruction_prefix=instruction_prefix,
        response_prefix=response_prefix,
    )

    # Make dir for codes generated by each model
    identifier = model.strip("./").replace("/", "--") + f"_{backend}_temp_{temperature}"
    if evalperf_type:
        identifier += f"-{evalperf_type}"

    target_path = os.path.join(root, dataset, identifier)
    if jsonl_fmt:
        target_path += ".jsonl"
    else:
        os.makedirs(target_path, exist_ok=True)
    score = codegen(
        target_path=target_path,
        dataset=dataset,
        greedy=greedy,
        model=model_runner,
        n_samples=n_samples,
        resume=resume,
        id_range=id_range,
        version=version,
    )

    return target_path, score

def evaluate(
    dataset: str,
    samples: Optional[str] = None,
    base_only: bool = False,
    parallel: Optional[int] = None,
    i_just_wanna_run: bool = False,
    test_details: bool = False,
    min_time_limit: float = 1,
    gt_time_limit_factor: float = 4.0,
    mini: bool = False,
    noextreme: bool = False,
    version: str = "default",
    **model_kwargs,
):
    if model_kwargs:
        samples, scores = run_codegen(
            dataset=dataset,
            **model_kwargs,
        )
    print(f"Scores: {scores}")

def main():
    from fire import Fire

    Fire(evaluate)


if __name__ == "__main__":
    main()

# export EVAL_OVERRIDE_PATH='path/to/eval.jsonl'
# python eval_script.py --model "model_name" --greedy --root temp --dataset eval --backend hf

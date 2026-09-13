#!/usr/bin/env python

# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import csv
import gc
import json
import random
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import paddle

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


kv_retrieval_prompt_template = (
    """
<s> Extract the value corresponding to the specified key in the data below.

Data:
{formatted_kv_records}

Extract the value corresponding to this key:
key: {key}
corresponding value:
""".strip()
    + " "
)

kv_retrieval_prompt_template_llama2_chat = (
    """
<s> [INST] Extract the value corresponding to the specified key in the data below.

Data:
{formatted_kv_records}

Extract the value corresponding to this key:
key: {key}

Please directly output the corresponding value without outputting anything else. [/INST]  Sure! The value corresponding to the key "{key}" is:
""".strip()
    + "\n\nvalue: "
)

kv_retrieval_prompt_template_llama3_instruct = (
    """
<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Extract the value corresponding to the specified key in the data below.

Data:
{formatted_kv_records}

Extract the value corresponding to this key:
key: {key}

Please directly output the corresponding value without outputting anything else.
value:<|eot_id|><|start_header_id|>assistant<|end_header_id|>
""".strip()
    + "\n\n"
)

kv_retrieval_prompt_template_qwen_instruct = (
    """
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
Extract the value corresponding to the specified key in the data below.

Data:
{formatted_kv_records}

Extract the value corresponding to this key:
key: {key}

Please directly output the corresponding value without outputting anything else.<|im_end|>
<|im_start|>assistant
""".strip()
    + "\n"
)

DEFAULT_SEQ_LENS = "8000,32000,48000,64000,80000,96000,112000,128000"


def parse_seq_lens(seq_lens):
    if isinstance(seq_lens, str):
        return [int(item) for item in seq_lens.split(",") if item.strip()]
    return [int(item) for item in seq_lens]


def infer_model_type(model_name):
    config_path = Path(model_name) / "config.json"
    if config_path.is_file():
        try:
            model_type = str(
                json.loads(config_path.read_text(encoding="utf-8")).get(
                    "model_type", ""
                )
            ).lower()
            if "qwen" in model_type:
                return "qwen"
            if "ernie" in model_type and "moe" in model_type:
                return "ernie_moe"
            if "ernie" in model_type:
                return "ernie"
            if "llama" in model_type:
                return "llama"
        except (OSError, json.JSONDecodeError):
            pass

    lower_name = model_name.lower()
    if "qwen2" in lower_name or "qwen2.5" in lower_name or "qwen" in lower_name:
        return "qwen"
    if "ernie" in lower_name and ("a3b" in lower_name or "moe" in lower_name):
        return "ernie_moe"
    if "ernie" in lower_name:
        return "ernie"
    if "llama" in lower_name:
        return "llama"
    raise ValueError(
        "Cannot infer model_type from model_name; pass --model-type explicitly"
    )


def load_patch(model_type):
    if model_type == "llama":
        from rrattn.llama_patch import patch_llama_attention

        return patch_llama_attention
    if model_type == "qwen":
        from rrattn.qwen_patch import patch_qwen_attention

        return patch_qwen_attention
    if model_type in ("ernie", "ernie_moe"):
        from rrattn.ernie_patch import patch_ernie_attention

        return patch_ernie_attention
    raise ValueError(f"Unsupported model_type={model_type!r}")


def load_profile_fns(method, enable_profile):
    if method == "xattn":
        from rrattn.xattention import (
            get_attn_time,
            get_estimate_func_time,
            set_attn_time,
            set_estimate_func_time,
            set_profile,
        )
    elif method == "rrattn":
        from rrattn.rrattention import (
            get_attn_time,
            get_estimate_func_time,
            set_attn_time,
            set_estimate_func_time,
            set_profile,
        )
    elif method == "flex":
        from rrattn.flexprefill import (
            get_attn_time,
            get_estimate_func_time,
            set_attn_time,
            set_estimate_func_time,
            set_profile,
        )
    elif method == "full":
        from rrattn.full_prefill import (
            get_attn_time,
            get_estimate_func_time,
            set_attn_time,
            set_estimate_func_time,
            set_profile,
        )
    else:
        raise ValueError(
            f"Unsupported method={method!r}; supported methods are: xattn, rrattn, flex, full"
        )

    set_profile(enable_profile)
    return (
        set_attn_time,
        get_attn_time,
        set_estimate_func_time,
        get_estimate_func_time,
    )


def load_model(model_name, model_type, dtype):
    from rrattn.checkpoint_utils import load_pretrained_checkpoint

    if model_type == "llama":
        from paddleformers.transformers import LlamaForCausalLM

        return load_pretrained_checkpoint(
            LlamaForCausalLM, model_name, dtype=dtype
        )
    if model_type == "qwen":
        from paddleformers.transformers.qwen2.modeling import (
            Qwen2ForCausalLMDeprecated,
        )

        return load_pretrained_checkpoint(
            Qwen2ForCausalLMDeprecated, model_name, dtype=dtype
        )
    if model_type == "ernie":
        from paddleformers.transformers import Ernie4_5ForCausalLM

        return load_pretrained_checkpoint(
            Ernie4_5ForCausalLM, model_name, dtype=dtype
        )
    if model_type == "ernie_moe":
        from paddleformers.transformers.ernie4_5_moe.modeling import (
            Ernie4_5_MoeForCausalLM,
        )

        return load_pretrained_checkpoint(
            Ernie4_5_MoeForCausalLM, model_name, dtype=dtype
        )
    raise ValueError(f"Unsupported model_type={model_type!r}")


def load_tokenizer(model_name):
    from paddleformers.transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name, use_fast=False)


def get_kv_retrieval_prompt(data, key, model_name):
    formatted_kv_records = ""
    for record_key, record_value in data:
        formatted_kv_records += f"key: {record_key} value: {record_value}\n"

    if model_name == "meta-llama/Llama-2-7b-chat-hf":
        prompt_template = kv_retrieval_prompt_template_llama2_chat
    elif (
        model_name == "meta-llama/Meta-Llama-3-8B-Instruct"
        or model_name == "gradientai/Llama-3-8B-Instruct-262k"
        or model_name == "gradientai/Llama-3-8B-Instruct-Gradient-1048k"
        or "Llama-3.1-8B-Instruct" in model_name
    ):
        prompt_template = kv_retrieval_prompt_template_llama3_instruct
    elif model_name in {
        "meta-llama/Llama-2-7b-hf",
        "meta-llama/Meta-Llama-3-8B",
    }:
        prompt_template = kv_retrieval_prompt_template
    elif "Qwen2.5-7B-Instruct" in model_name or "Qwen" in model_name:
        prompt_template = kv_retrieval_prompt_template_qwen_instruct
    else:
        prompt_template = kv_retrieval_prompt_template

    return prompt_template.format(
        formatted_kv_records=formatted_kv_records, key=key
    )


def move_tensor(value, device):
    if not hasattr(value, "cuda"):
        return value
    if device.startswith("gpu"):
        return value.cuda()
    return value


def quick_get_random_kv_samples(
    model_name, tokenizer, gold_index, n_kv_num, n_sample, device
):
    samples = []
    for _ in range(n_sample):
        ordered_kv_records = [
            [str(uuid.uuid4()), str(uuid.uuid4())] for _ in range(n_kv_num)
        ]
        key = str(uuid.uuid4())
        value = str(uuid.uuid4())
        ordered_kv_records.insert(gold_index, [key, value])
        kv_prompt = get_kv_retrieval_prompt(
            data=ordered_kv_records,
            key=key,
            model_name=model_name,
        )
        inputs = tokenizer(
            kv_prompt, return_tensors="pd", add_special_tokens=False
        )
        sample = {
            "input_ids": move_tensor(inputs["input_ids"], device),
            "key": key,
            "value": value,
        }
        if "attention_mask" in inputs:
            sample["attention_mask"] = move_tensor(
                inputs["attention_mask"], device
            )
        else:
            sample["attention_mask"] = None
        samples.append(sample)
    return samples


def synchronize(device):
    if device.startswith("gpu"):
        paddle.device.synchronize()


def clear_cache(device):
    gc.collect()
    if device.startswith("gpu"):
        paddle.device.cuda.empty_cache()


def measure_forward(model, input_ids, attention_mask, device):
    synchronize(device)
    start_event = None
    end_event = None
    if device.startswith("gpu"):
        start_event = paddle.cuda.Event(enable_timing=True)
        end_event = paddle.cuda.Event(enable_timing=True)
        start_event.record()
    start_time = time.perf_counter()
    model_kwargs = {"input_ids": input_ids, "use_cache": False}
    if attention_mask is not None:
        model_kwargs["attention_mask"] = attention_mask
    with paddle.no_grad():
        model(**model_kwargs)
    synchronize(device)
    elapsed_wall_ms = (time.perf_counter() - start_time) * 1000.0
    if device.startswith("gpu"):
        end_event.record()
        synchronize(device)
        return float(start_event.elapsed_time(end_event))
    return elapsed_wall_ms


def output_name(model_name, method, threshold, stride, output_dir):
    saved_method = method if method != "full" else f"{method}_{1.00:.2f}"
    if method != "full":
        saved_method = f"{saved_method}_{threshold:.2f}"
    return (
        Path(output_dir)
        / f"speed_test_{Path(model_name).name}_result_{saved_method}_s{stride}.csv"
    )


def main(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    model_type="auto",
    method="xattn",
    threshold=0.9,
    stride=8,
    seq_lens=DEFAULT_SEQ_LENS,
    n_times=3,
    n_kv_num=6000,
    gold_index=3000,
    dtype="bfloat16",
    device=None,
    output_dir=".",
    seed=20260420,
):
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)
    paddle.set_flags({"FLAGS_flash_attn_version": 3})

    if device is None:
        device = "gpu:0" if paddle.device.cuda.device_count() > 0 else "cpu"
    paddle.set_device(device)

    if model_type == "auto":
        model_type = infer_model_type(model_name)

    seq_len_list = parse_seq_lens(seq_lens)
    enable_profile = device.startswith("gpu")
    (
        set_attn_time,
        get_attn_time,
        set_estimate_func_time,
        get_estimate_func_time,
    ) = load_profile_fns(
        method,
        enable_profile,
    )

    model = load_model(model_name, model_type, dtype)
    model.eval()
    patch_fn = load_patch(model_type)
    patch_fn(
        model,
        method=method,
        threshold=threshold,
        stride=stride,
    )
    tokenizer = load_tokenizer(model_name)

    samples = quick_get_random_kv_samples(
        model_name,
        tokenizer,
        gold_index=gold_index,
        n_kv_num=n_kv_num,
        n_sample=n_times,
        device=device,
    )
    prompt_lens = [int(sample["input_ids"].shape[-1]) for sample in samples]
    min_prompt_len = min(prompt_lens)
    max_prompt_len = max(prompt_lens)
    max_target_len = max(seq_len_list)
    if min_prompt_len < max_target_len:
        raise ValueError(
            "Generated prompts are too short for the requested seq_lens: "
            f"shortest={min_prompt_len}, longest={max_prompt_len}, required={max_target_len}. "
            "Increase --n-kv-num or lower --seq-lens."
        )

    print(f"model={model_name} method={method} device={device}")
    print("---------------------------")

    for seq_len in seq_len_list:
        input_ids = samples[0]["input_ids"][..., :seq_len]
        attention_mask = samples[0]["attention_mask"]
        if attention_mask is not None:
            attention_mask = attention_mask[..., :seq_len]
        with paddle.no_grad():
            model_kwargs = {"input_ids": input_ids, "use_cache": False}
            if attention_mask is not None:
                model_kwargs["attention_mask"] = attention_mask
            model(**model_kwargs)
        clear_cache(device)

    rows = []
    for seq_len in seq_len_list:
        total_times = []
        attn_times = []
        estimate_times = []
        for idx in range(n_times):
            sample = samples[idx]
            input_ids = sample["input_ids"][..., :seq_len]
            attention_mask = sample["attention_mask"]
            if attention_mask is not None:
                attention_mask = attention_mask[..., :seq_len]

            set_attn_time()
            set_estimate_func_time()
            elapsed_ms = measure_forward(
                model, input_ids, attention_mask, device
            )
            total_times.append(elapsed_ms)
            attn_times.append(float(get_attn_time()))
            estimate_times.append(float(get_estimate_func_time()))
            clear_cache(device)

        row = {
            "model": Path(model_name).name,
            "method": method,
            "stride": 1 if method == "full" else stride,
            "seq_len": seq_len,
            "total_time": float(np.mean(total_times)),
            "attn_time": float(np.mean(attn_times)),
            "estimate_func_time": float(np.mean(estimate_times)),
        }
        rows.append(row)
        print(
            "seq_len: {:<20} total time: {:.2f}ms, attn time: {:.2f}ms, estimate func time {:.2f}ms".format(
                seq_len,
                row["total_time"],
                row["attn_time"],
                row["estimate_func_time"],
            )
        )
        print("---------------------------")

    output_path = output_name(model_name, method, threshold, stride, output_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model",
                "method",
                "stride",
                "seq_len",
                "total_time",
                "attn_time",
                "estimate_func_time",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name", default="meta-llama/Llama-3.1-8B-Instruct"
    )
    parser.add_argument(
        "--model-type",
        default="auto",
        choices=["auto", "llama", "qwen", "ernie", "ernie_moe"],
    )
    parser.add_argument(
        "--method",
        default="rrattn",
        choices=["xattn", "rrattn", "flex", "full"],
    )
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--seq-lens", default=DEFAULT_SEQ_LENS)
    parser.add_argument("--n-times", type=int, default=3)
    parser.add_argument("--n-kv-num", type=int, default=6000)
    parser.add_argument("--gold-index", type=int, default=3000)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--seed", type=int, default=20260420)
    main(**vars(parser.parse_args()))

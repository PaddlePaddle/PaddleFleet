#!/usr/bin/env python

import argparse
import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path

PROTOCOL_STDOUT = sys.stdout
sys.stdout = sys.stderr

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import paddle
from paddleformers.generation import GenerationConfig


def disable_unavailable_deep_ep():
    try:
        import paddlefleet.ops as fleet_ops
    except Exception:
        return

    is_available = getattr(fleet_ops, "is_deep_ep_available", None)
    if is_available is None or not is_available():
        return

    try:
        from paddlefleet.ops import deep_ep  # noqa: F401
    except Exception as exc:
        print(
            f"[paddle_worker] disable unavailable DeepEP: {exc}",
            file=sys.stderr,
        )
        if hasattr(fleet_ops, "_DEEP_EP_AVAILABLE"):
            fleet_ops._DEEP_EP_AVAILABLE = False


def infer_model_type(model_name):
    config_path = Path(model_name) / "config.json"
    if config_path.is_file():
        try:
            model_type = str(
                json.loads(config_path.read_text(encoding="utf-8")).get(
                    "model_type", ""
                )
            ).lower()
            if "qwen3" in model_type:
                raise ValueError(
                    "Qwen3 models are not supported in this release"
                )
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
    if "qwen3" in lower_name:
        raise ValueError("Qwen3 models are not supported in this release")
    if "qwen2" in lower_name or "qwen2.5" in lower_name or "qwen" in lower_name:
        return "qwen"
    if "ernie" in lower_name and ("a3b" in lower_name or "moe" in lower_name):
        return "ernie_moe"
    if "ernie" in lower_name:
        return "ernie"
    if "llama" in lower_name:
        return "llama"
    raise ValueError(
        "Cannot infer model_type from model_name; pass --model_type explicitly"
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


def load_profile_fns(method):
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

    set_profile(False)
    return (
        set_profile,
        set_attn_time,
        get_attn_time,
        set_estimate_func_time,
        get_estimate_func_time,
    )


def set_common_config(config):
    config.sequence_parallel = False
    config.use_cache = True
    config.tie_word_embeddings = False
    return config


def cast_tiny_model(model, dtype):
    dtype = str(dtype).replace("paddle.", "")
    if dtype in ("float16", "bfloat16"):
        return model.to(dtype=dtype)
    return model


def build_tiny_model(model_type, dtype):
    if model_type == "llama":
        from paddleformers.transformers import LlamaConfig, LlamaForCausalLM

        config = LlamaConfig(
            vocab_size=1024,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=4096,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            dtype=dtype,
        )
        return cast_tiny_model(
            LlamaForCausalLM(set_common_config(config)), dtype
        )

    if model_type == "qwen":
        from paddleformers.transformers import Qwen2Config
        from paddleformers.transformers.qwen2.modeling import (
            Qwen2ForCausalLMDeprecated,
        )

        config = Qwen2Config(
            vocab_size=1024,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            max_position_embeddings=4096,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            attention_dropout=0.0,
            attention_probs_dropout_prob=0.0,
            hidden_dropout_prob=0.0,
        )
        config.fuse_rms_norm = False
        return cast_tiny_model(
            Qwen2ForCausalLMDeprecated(set_common_config(config)), dtype
        )

    if model_type == "ernie":
        from paddleformers.transformers import (
            Ernie4_5Config,
            Ernie4_5ForCausalLM,
        )

        config = Ernie4_5Config(
            vocab_size=1024,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            max_position_embeddings=4096,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
        config.apply_rope_fusion = False
        config.fuse_rms_norm = False
        return cast_tiny_model(
            Ernie4_5ForCausalLM(set_common_config(config)), dtype
        )

    raise ValueError(f"Unsupported model_type={model_type!r}")


def load_model(model_name, model_type, dtype, tiny_random):
    if tiny_random:
        return build_tiny_model(model_type, dtype)

    lower_name = model_name.lower()
    if "qwen3" in lower_name:
        raise ValueError("Qwen3 models are not supported in this release")
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
    if model_type == "ernie_moe":
        from paddleformers.transformers.ernie4_5_moe.modeling import (
            Ernie4_5_MoeForCausalLM,
        )

        return load_pretrained_checkpoint(
            Ernie4_5_MoeForCausalLM, model_name, dtype=dtype
        )
    if model_type == "ernie":
        from paddleformers.transformers import Ernie4_5ForCausalLM

        return load_pretrained_checkpoint(
            Ernie4_5ForCausalLM, model_name, dtype=dtype
        )
    raise ValueError(f"Unsupported model_type={model_type!r}")


def default_device():
    if paddle.device.cuda.device_count() > 0:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        return f"gpu:{local_rank}"
    return "cpu"


def synchronize(device):
    if device.startswith("gpu"):
        paddle.device.synchronize()


def clear_cache(device):
    gc.collect()
    if device.startswith("gpu"):
        synchronize(device)
        paddle.device.cuda.empty_cache()


def max_memory_allocated(device):
    if not device.startswith("gpu"):
        return None
    return int(paddle.device.cuda.max_memory_allocated())


def collect_sparse_ratio(model):
    ratios = []
    if hasattr(model, "named_sublayers"):
        for name, layer in model.named_sublayers():
            if name.split(".")[-1] == "self_attn" and hasattr(
                layer, "sparse_ratio"
            ):
                sparse_ratio = getattr(layer, "sparse_ratio", None)
                if sparse_ratio is None:
                    continue
                if hasattr(sparse_ratio, "item"):
                    ratios.append(float(sparse_ratio.item()))
                else:
                    ratios.append(float(sparse_ratio))
                layer.sparse_ratio = None
    if not ratios:
        return None
    return float(np.mean(ratios))


def tensor_to_ids(value):
    if isinstance(value, (tuple, list)):
        value = value[0]
    if hasattr(value, "numpy"):
        return value.numpy().tolist()
    return value


def make_generation_config(payload):
    do_sample = bool(payload.get("do_sample", False))
    config_kwargs = {
        "max_new_tokens": int(payload.get("generation_max_length", 1)),
        "min_new_tokens": int(payload.get("generation_min_length", 0)),
        "decode_strategy": "sampling" if do_sample else "greedy_search",
        "eos_token_id": payload.get("stop_token_ids"),
        "pad_token_id": payload.get("pad_token_id"),
        "use_cache": True,
    }
    if do_sample:
        config_kwargs["temperature"] = float(payload.get("temperature", 1.0))
        config_kwargs["top_p"] = float(payload.get("top_p", 1.0))
    else:
        config_kwargs["temperature"] = 1.0
        config_kwargs["top_p"] = 1.0
        config_kwargs["top_k"] = 0
    return GenerationConfig(**config_kwargs)


class FirstTokenTimer:
    def __init__(self, device, start_wall, start_event=None):
        self.device = device
        self.start_wall = start_wall
        self.start_event = start_event
        self.seen_prompt = False
        self.ttft_ms = None

    def put(self, value):
        if not self.seen_prompt:
            self.seen_prompt = True
            return
        if self.ttft_ms is not None:
            return
        if self.device.startswith("gpu") and self.start_event is not None:
            end_event = paddle.cuda.Event(enable_timing=True)
            end_event.record()
            synchronize(self.device)
            self.ttft_ms = float(self.start_event.elapsed_time(end_event))
        else:
            self.ttft_ms = (time.perf_counter() - self.start_wall) * 1000.0

    def end(self):
        pass


def handle_request(
    model, payload, device, profile_fns, clear_cache_per_request=False
):
    input_ids = paddle.to_tensor(payload["input_ids"], dtype="int64")
    attention_mask = payload.get("attention_mask")
    model_kwargs = {}
    if attention_mask is not None:
        model_kwargs["attention_mask"] = paddle.to_tensor(
            attention_mask, dtype=paddle.get_default_dtype()
        )

    record_ttft_ms = bool(payload.get("record_ttft_ms", False))
    record_e2e_ms = bool(payload.get("record_e2e_ms", False))
    record_attn_ms = bool(payload.get("record_attn_ms", False))

    (
        set_profile,
        set_attn_time,
        get_attn_time,
        set_estimate_func_time,
        get_estimate_func_time,
    ) = profile_fns
    profile_enabled = record_attn_ms and device.startswith("gpu")
    set_profile(profile_enabled)
    if record_attn_ms:
        set_attn_time()
        set_estimate_func_time()

    synchronize(device)
    start = time.perf_counter()
    start_event = None
    if device.startswith("gpu"):
        start_event = paddle.cuda.Event(enable_timing=True)
        end_event = paddle.cuda.Event(enable_timing=True)
        start_event.record()

    first_token_timer = (
        FirstTokenTimer(device, start, start_event) if record_ttft_ms else None
    )
    try:
        with paddle.no_grad():
            if hasattr(model, "model") and input_ids.shape[1] > 1:
                prefill_kwargs = {}
                if "attention_mask" in model_kwargs:
                    prefill_kwargs["attention_mask"] = model_kwargs[
                        "attention_mask"
                    ][:, :-1]
                prefill = model.model(
                    input_ids=input_ids[:, :-1],
                    use_cache=True,
                    return_dict=True,
                    **prefill_kwargs,
                )
                past_key_values = getattr(prefill, "past_key_values", None)
                del prefill
                if past_key_values is not None:
                    model_kwargs["past_key_values"] = past_key_values
            generated_ids, _ = model.generate(
                input_ids,
                generation_config=make_generation_config(payload),
                streamer=first_token_timer,
                **model_kwargs,
            )
    finally:
        set_profile(False)

    synchronize(device)
    elapsed_wall_ms = (time.perf_counter() - start) * 1000.0
    elapsed_ms = elapsed_wall_ms
    if device.startswith("gpu"):
        end_event.record()
        synchronize(device)
        elapsed_ms = float(start_event.elapsed_time(end_event))

    output_ids = tensor_to_ids(generated_ids)
    if output_ids and isinstance(output_ids[0], list):
        output_ids = output_ids[0]

    response = {
        "id": payload.get("id"),
        "ok": True,
        "output_ids": output_ids,
        "output_len": len(output_ids),
        "sparse_ratio": collect_sparse_ratio(model),
        "memory_usage": max_memory_allocated(device),
    }
    if record_e2e_ms:
        response["e2e_ms"] = elapsed_ms
    if record_ttft_ms:
        response["ttft_ms"] = (
            first_token_timer.ttft_ms
            if first_token_timer.ttft_ms is not None
            else elapsed_ms
        )
    if record_attn_ms:
        response["attn_ms"] = float(get_attn_time())
        response["estimate_func_ms"] = float(get_estimate_func_time())

    if clear_cache_per_request:
        clear_cache(device)
    return response


def write_response(response):
    PROTOCOL_STDOUT.write(json.dumps(response) + "\n")
    PROTOCOL_STDOUT.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument(
        "--model_type",
        default="auto",
        choices=["auto", "llama", "qwen", "ernie", "ernie_moe"],
    )
    parser.add_argument(
        "--method", default="full", choices=["xattn", "rrattn", "flex", "full"]
    )
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--rrattn_version", default="v1")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default=None)
    parser.add_argument("--tiny_random", action="store_true")
    parser.add_argument("--clear_cache_per_request", action="store_true")
    args = parser.parse_args()

    device = args.device or default_device()
    paddle.set_device(device)

    model_type = args.model_type
    if model_type == "auto":
        if args.tiny_random:
            raise ValueError(
                "--model_type is required when --tiny_random is set"
            )
        model_type = infer_model_type(args.model_name_or_path)

    disable_unavailable_deep_ep()
    profile_fns = load_profile_fns(args.method)
    model = load_model(
        args.model_name_or_path, model_type, args.dtype, args.tiny_random
    )
    model.eval()
    patch_fn = load_patch(model_type)
    patch_fn(
        model,
        method=args.method,
        threshold=args.threshold,
        stride=args.stride,
    )
    clear_cache(device)
    print(
        f"[paddle_worker] ready model_type={model_type} method={args.method} dtype={args.dtype} device={device}",
        file=sys.stderr,
    )

    for line in sys.stdin:
        try:
            payload = json.loads(line)
            if payload.get("cmd") == "shutdown":
                break
            if payload.get("cmd") == "release_cache":
                clear_cache(device)
                response = {
                    "id": payload.get("id"),
                    "ok": True,
                }
                write_response(response)
                continue
            response = handle_request(
                model,
                payload,
                device,
                profile_fns,
                args.clear_cache_per_request,
            )
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            response = {
                "id": payload.get("id")
                if "payload" in locals() and isinstance(payload, dict)
                else None,
                "ok": False,
                "error": repr(exc),
            }
        write_response(response)


if __name__ == "__main__":
    main()

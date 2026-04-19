from .ernie_patch import patch_ernie_attention
from .llama_patch import patch_llama_attention
from .qwen_patch import patch_qwen_attention
from .rrattention import rrattn_estimate, rrattn_prefill

__all__ = [
    "rrattn_estimate",
    "rrattn_prefill",
    "patch_llama_attention",
    "patch_qwen_attention",
    "patch_ernie_attention",
]

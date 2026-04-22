def xattn_estimate(*args, **kwargs):
    from .xattention import xattn_estimate as impl

    return impl(*args, **kwargs)


def xattn_prefill(*args, **kwargs):
    from .xattention import xattn_prefill as impl

    return impl(*args, **kwargs)


def rrattn_estimate(*args, **kwargs):
    from .rrattention import rrattn_estimate as impl

    return impl(*args, **kwargs)


def rrattn_estimate_legacy(*args, **kwargs):
    from .rrattention import rrattn_estimate_legacy as impl

    return impl(*args, **kwargs)


def rrattn_prefill(*args, **kwargs):
    from .rrattention import rrattn_prefill as impl

    return impl(*args, **kwargs)


def flex_prefill(*args, **kwargs):
    from .flexprefill import flex_prefill as impl

    return impl(*args, **kwargs)


def patch_llama_attention(*args, **kwargs):
    from .llama_patch import patch_llama_attention as impl

    return impl(*args, **kwargs)


def patch_qwen_attention(*args, **kwargs):
    from .qwen_patch import patch_qwen_attention as impl

    return impl(*args, **kwargs)


def patch_ernie_attention(*args, **kwargs):
    from .ernie_patch import patch_ernie_attention as impl

    return impl(*args, **kwargs)


__all__ = [
    "xattn_estimate",
    "xattn_prefill",
    "rrattn_estimate",
    "rrattn_estimate_legacy",
    "rrattn_prefill",
    "flex_prefill",
    "patch_llama_attention",
    "patch_qwen_attention",
    "patch_ernie_attention",
]

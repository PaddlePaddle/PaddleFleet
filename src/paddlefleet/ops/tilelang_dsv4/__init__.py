import paddle

paddle.enable_compat(scope={"tilelang"}, silent=True)

from .attention_core import (
    tilelang_compressed_sparse_attn_paddle_compat_autograd,
)

__all__ = [
    "tilelang_compressed_sparse_attn_paddle_compat_autograd",
    "tilelang_csa_compressed_indexer_bwd_paddle",
    "tilelang_csa_compressed_indexer_topk_paddle",
]


def __getattr__(name):
    if name in {
        "tilelang_csa_compressed_indexer_bwd_paddle",
        "tilelang_csa_compressed_indexer_topk_paddle",
    }:
        from .csa_indexer_core import (
            tilelang_csa_compressed_indexer_bwd_paddle,
            tilelang_csa_compressed_indexer_topk_paddle,
        )

        exports = {
            "tilelang_csa_compressed_indexer_bwd_paddle": tilelang_csa_compressed_indexer_bwd_paddle,
            "tilelang_csa_compressed_indexer_topk_paddle": tilelang_csa_compressed_indexer_topk_paddle,
        }
        globals().update(exports)
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
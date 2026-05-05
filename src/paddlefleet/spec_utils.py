# Backward compatibility shim — PaddleFormers imports LayerSpec/build_layer from here.
# These were migrated to Paddle in commit 8c1a3e5 (Apr 16, 2026).

from paddle.distributed.fleet.meta_parallel import LayerSpec
from paddle.distributed.fleet.meta_parallel import build_spec_layer as build_layer

__all__ = ["LayerSpec", "build_layer"]

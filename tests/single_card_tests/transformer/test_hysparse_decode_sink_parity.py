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

"""HySparse incremental decode vs cache-less forward parity, aimed at the two
SWA attention sinks.

The SWA (MQA) layer runs two independent softmaxes -- the sliding-window main
path and the block-sparse gather branch -- each with its own learnable per-head
sink logit (``swa_attn_sink`` / ``sparse_attn_sink``). Both are passed to the
kernels unconditionally, so a KV-cache decode step is supposed to apply them
exactly like the cache-less forward. This test pins that down.

Structure:

  1. ``test_decode_matches_no_cache_*``: one cache-less forward over the whole
     ``PROMPT_LEN + DECODE_STEPS`` sequence is the oracle; a prefill + N
     one-token decode steps through :class:`DynamicKVCache` must reproduce the
     oracle's SWA attention output at the matching positions. Run for both
     block-sparse backends (TileLang always, DSA when the SM100 backend is
     importable).
  2. ``test_dropping_a_sink_is_detectable``: calibrates 1. In the
     saturated-sink regime, parity still holds while replacing either sink with
     the kernels' sinkless ``-1e30`` moves the output ~30-90x the parity gap.
     Without this, 1 would pass even if decode silently ran sinkless.
  3. ``test_sinks_are_honoured_at_one_query_token``: kernel-level check that
     both branches still fold the sink in at ``S == 1`` (the decode shape),
     independent of the layer wiring.

Comparisons are made on the SWA attention sublayer output (post ``o_proj``,
pre-residual) -- see :class:`_AttnCapture` for why the layer output is too
insensitive.

Requires SM 10.x (Blackwell) for the TileLang HySparse kernels; skips
otherwise.
"""

import os
import unittest

os.environ["FLAGS_cudnn_deterministic"] = "True"

import paddle
from paddle.distributed.fleet.meta_parallel import LayerSpec

from paddlefleet.fusions.fused_bias_dropout import get_bias_dropout_add
from paddlefleet.generation.greedy_generator import DynamicKVCache
from paddlefleet.tensor_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from paddlefleet.tensor_parallel.random import model_parallel_cuda_manual_seed
from paddlefleet.transformer.dot_product_attention import DotProductAttention
from paddlefleet.transformer.enums import AttnMaskType
from paddlefleet.transformer.multi_latent_attention import (
    MLASelfAttentionSublayersSpec,
    MQASelfAttention,
)
from paddlefleet.transformer.paddle_norm import WrappedPaddleNorm
from paddlefleet.transformer.transformer_config import TransformerConfig
from paddlefleet.transformer.transformer_layer import (
    HySparseTransformerLayer,
    TransformerLayerSublayersSpec,
)

BATCH = 1
HIDDEN = 2048
HEADS = 64
WINDOW = 128
BLOCK_B = 64
TOPK = 16
# 24 key blocks > TOPK, so the top-k really selects a subset (a prompt short
# enough to fit in TOPK blocks would select everything and hide selection bugs).
PROMPT_LEN = 1536
DECODE_STEPS = 6
TOTAL_LEN = PROMPT_LEN + DECODE_STEPS


def _tilelang_backend_or_skip(testcase):
    """Skip unless the TileLang HySparse kernels can run (SM 10.x)."""
    if not paddle.device.is_compiled_with_cuda():
        testcase.skipTest("CUDA build of Paddle is required")
    if paddle.device.cuda.device_count() == 0:
        testcase.skipTest("No CUDA device available")
    major = paddle.device.cuda.get_device_capability()[0]
    if major != 10:
        testcase.skipTest(
            f"HySparse TileLang kernels require SM 10.x; got SM {major}.x"
        )
    try:
        from paddlefleet.tilelang_ops.hysparse import (  # noqa: F401
            sliding_window_mqa_attention,
        )
        from paddlefleet.tilelang_ops.hysparse.block_sparse_mqa_tl import (  # noqa: F401
            block_sparse_mqa_attention_tl,
        )
    except (ImportError, RuntimeError) as exc:
        testcase.skipTest(f"HySparse TileLang backend import failed: {exc}")


def _dsa_available():
    """Whether the SM100 FlashMLA/cuDNN block-sparse backend can run."""
    try:
        from paddlefleet.cudnn_ops import is_dsa_available

        return bool(is_dsa_available())
    except (ImportError, RuntimeError):
        return False


def _cos(a, b):
    """Cosine similarity of two tensors, flattened, in fp32."""
    x = a.astype("float32").flatten()
    y = b.astype("float32").flatten()
    denom = float(x.norm()) * float(y.norm())
    return float(paddle.dot(x, y)) / (denom + 1e-30)


def _max_abs_diff(a, b):
    return float((a.astype("float32") - b.astype("float32")).abs().max())


def _rel_l2(a, b):
    """Relative L2 deviation of ``a`` from reference ``b``.

    Preferred over cosine once both quantities sit at cos > 0.999, where cosine
    compresses an order-of-magnitude difference in deviation into the fourth
    decimal place.
    """
    x = a.astype("float32")
    y = b.astype("float32")
    return float((x - y).norm()) / (float(y.norm()) + 1e-30)


class _AttnCapture:
    """Collect an attention sublayer's output across forward calls.

    Everything the sinks influence is inside the attention sublayer, and the
    surrounding residual add dilutes it below the bf16 noise floor: dropping a
    sink entirely moves the *layer* output less than the run-to-run bf16 spread,
    so a layer-output metric cannot see it. Hooking the sublayer output (post
    ``o_proj``, pre-residual) keeps the comparison in the space the sinks
    actually act on.
    """

    def __init__(self, attn_layer):
        self.outs = []
        self._handle = attn_layer.register_forward_post_hook(self._hook)

    def _hook(self, layer, inputs, output):
        self.outs.append(output[0] if isinstance(output, tuple) else output)

    def remove(self):
        self._handle.remove()

    def concat(self):
        """All captured outputs joined along the sequence axis."""
        return paddle.concat(self.outs, axis=1)


class TestHySparseDecodeSinkParity(unittest.TestCase):
    """Decode-time parity of the SWA layer, with both learnable sinks live."""

    @classmethod
    def setUpClass(cls):
        cls.sublayer_spec = MLASelfAttentionSublayersSpec(
            core_attention=DotProductAttention,
            o_proj=RowParallelLinear,
            gate_proj=ColumnParallelLinear,
            q_a_proj=ColumnParallelLinear,
            q_b_proj=ColumnParallelLinear,
            kv_a_proj_with_mqa=ColumnParallelLinear,
            kv_b_proj=ColumnParallelLinear,
            q_a_layernorm=WrappedPaddleNorm,
            kv_a_layernorm=WrappedPaddleNorm,
        )

    # ---- builders -----------------------------------------------------

    def _build_config(self, block_sparse_use_tilelang):
        """Online HySparse MQA dims (Dk=256, Dv=448, H=64), sinks ON."""
        return TransformerConfig(
            hidden_size=HIDDEN,
            head_dim=192,
            num_attention_heads=HEADS,
            num_key_value_heads=4,
            gated_attention=True,
            gated_attn_use_q_lora=True,
            q_lora_rank=1024,
            qk_rope_head_dim=64,
            qk_nope_head_dim=192,
            v_head_dim=256,
            kv_lora_rank=448,
            rope_theta=640000,
            use_qk_norm=True,
            multi_latent_attention=True,
            rope_type="rope",
            add_swa_attention_sink_bias=True,
            sliding_window=[WINDOW, WINDOW],
            swa_head_dim=192,
            swa_v_head_dim=256,
            swa_num_attention_heads=HEADS,
            swa_num_key_value_heads=4,
            window_attn_skip_freq=2,
            enable_hy_sparse_attention=True,
            hy_sparse_full_attn_use_tilelang=True,
            hy_sparse_block_sparse_use_tilelang=block_sparse_use_tilelang,
            hy_sparse_block_size=BLOCK_B,
            hy_sparse_topk=TOPK,
        )

    def _build_stack(self, config, seed=2026):
        """Build an eval-mode (full, SWA) HySparseTransformerLayer pair.

        ``full_recompute`` must stay False: the recompute branch of
        ``HySparseTransformerLayer.forward`` forwards an explicit argument list
        that does not include ``past_key_values`` / ``use_cache``, so the cache
        would silently never reach the attention layers.
        """
        paddle.seed(seed)
        model_parallel_cuda_manual_seed(seed)
        layer_spec = TransformerLayerSublayersSpec(
            self_attn=LayerSpec(
                layer=MQASelfAttention,
                sublayers_spec=self.sublayer_spec,
            ),
            self_attn_bda=get_bias_dropout_add,
        )
        layers = []
        for layer_number in (0, 1):
            layer = HySparseTransformerLayer(
                config, layer_spec, layer_number=layer_number
            )
            layer.self_attn.attn_mask_type = AttnMaskType.causal
            layer = paddle.amp.decorate(layer, level="O2", dtype="bfloat16")
            layer.full_recompute = False
            layer.eval()
            layers.append(layer)
        full_layer, swa_layer = layers
        # Layer 0 produces shared_key / block_indices, layer 1 consumes them.
        self.assertFalse(full_layer.self_attn.is_swa)
        self.assertTrue(swa_layer.self_attn.is_swa)
        self.assertTrue(swa_layer.self_attn.is_mqa)
        # eval() matters: get_query_key_value_tensors only honours position_ids
        # when not training, which incremental decode depends on.
        self.assertFalse(swa_layer.training)
        return full_layer, swa_layer

    def _set_sinks(self, swa_layer, saturated=False, drop=None):
        """Give the two sinks distinct non-zero logits.

        Two regimes, both used:

        * default -- logits around 1..3, where the sink competes with the q.k
          logits but neither branch is suppressed. This is the realistic setting
          the parity tests run in.
        * ``saturated=True`` -- logits around 11..13, above every q.k logit, so
          the sink owns its softmax denominator. Here the sink's presence is
          unambiguous, which is what makes a dropped sink measurable (in the
          default regime dropping a sink moves the output *less* than the bf16
          parity floor, so it would be invisible).

        ``drop`` names a sink to replace with ``-1e30``, i.e. exactly what the
        kernels substitute for "sinkless": ``exp(sink - m)`` underflows to 0.
        That is the failure mode the parity tests have to be able to see.
        """
        lo, hi = (11.0, 13.0) if saturated else (1.0, 3.0)
        sinks = {
            "swa_attn_sink": paddle.linspace(lo, hi, HEADS),
            "sparse_attn_sink": paddle.linspace(hi, lo, HEADS),
        }
        if drop is not None:
            self.assertIn(drop, sinks)
            sinks[drop] = paddle.full([HEADS], -1e30)
        for name, value in sinks.items():
            param = getattr(swa_layer.self_attn, name)
            param.set_value(value.astype(param.dtype))

    # ---- runners ------------------------------------------------------

    def _hidden(self, seed=7):
        """A fixed random hidden-state sequence of length ``TOTAL_LEN``.

        Driving the layers with a *pre-fixed* sequence instead of real
        autoregressive feedback is what makes a single oracle forward valid: the
        cached run consumes the same per-position inputs, so position ``t`` can
        be compared directly against the oracle's position ``t``.
        """
        paddle.seed(seed)
        return paddle.randn([BATCH, TOTAL_LEN, HIDDEN], dtype="bfloat16")

    @paddle.no_grad()
    def _run_no_cache(self, full_layer, swa_layer, hidden):
        """Oracle: one cache-less forward over the whole sequence.

        Returns the SWA attention sublayer output, ``[B, TOTAL_LEN, H]``.
        """
        startend = paddle.full(
            [BATCH, 1, TOTAL_LEN, 1], TOTAL_LEN, dtype="int32"
        )
        position_ids = (
            paddle.arange(TOTAL_LEN, dtype="int64")
            .unsqueeze(0)
            .expand([BATCH, TOTAL_LEN])
        )
        capture = _AttnCapture(swa_layer.self_attn)
        try:
            with paddle.amp.auto_cast(True, level="O2", dtype="bfloat16"):
                args = {
                    "hidden_states": hidden,
                    "attn_mask_startend_row_indices": startend,
                    "position_ids": position_ids,
                }
                args = full_layer(args)
                swa_layer(args)
        finally:
            capture.remove()
        return capture.concat()

    @paddle.no_grad()
    def _run_with_cache(self, full_layer, swa_layer, hidden):
        """Prefill ``PROMPT_LEN`` tokens, then ``DECODE_STEPS`` single tokens.

        Mirrors ``GreedyGenerator.generate``: a document-filled flashmask on
        prefill, none on decode, and absolute ``position_ids`` per step. Returns
        the SWA attention sublayer output, ``[B, TOTAL_LEN, H]`` (prefill slice
        followed by one column per decode step).
        """
        cache = DynamicKVCache(
            num_layers=2, swa_layers=[False, True], window_size=WINDOW
        )
        startend = paddle.full(
            [BATCH, 1, PROMPT_LEN, 1], PROMPT_LEN, dtype="int32"
        )
        prompt_pos = (
            paddle.arange(PROMPT_LEN, dtype="int64")
            .unsqueeze(0)
            .expand([BATCH, PROMPT_LEN])
        )
        capture = _AttnCapture(swa_layer.self_attn)
        try:
            with paddle.amp.auto_cast(True, level="O2", dtype="bfloat16"):
                args = {
                    "hidden_states": hidden[:, :PROMPT_LEN],
                    "attn_mask_startend_row_indices": startend,
                    "position_ids": prompt_pos,
                    "past_key_values": cache,
                    "use_cache": True,
                }
                args = full_layer(args)
                swa_layer(args)

                for t in range(PROMPT_LEN, TOTAL_LEN):
                    args = {
                        "hidden_states": hidden[:, t : t + 1],
                        "position_ids": paddle.full(
                            [BATCH, 1], t, dtype="int64"
                        ),
                        "past_key_values": cache,
                        "use_cache": True,
                    }
                    args = full_layer(args)
                    swa_layer(args)
        finally:
            capture.remove()
        return capture.concat()

    def _parity_report(self, block_sparse_use_tilelang):
        """(prefill_cos, per-step cos, per-step relative L2) for one backend."""
        config = self._build_config(block_sparse_use_tilelang)
        full_layer, swa_layer = self._build_stack(config)
        self._set_sinks(swa_layer)
        hidden = self._hidden()

        oracle = self._run_no_cache(full_layer, swa_layer, hidden)
        cached = self._run_with_cache(full_layer, swa_layer, hidden)
        self.assertEqual(cached.shape, oracle.shape)

        prefill_cos = _cos(cached[:, PROMPT_LEN - 1], oracle[:, PROMPT_LEN - 1])
        step_cos, step_rel = [], []
        for t in range(PROMPT_LEN, TOTAL_LEN):
            step_cos.append(_cos(cached[:, t], oracle[:, t]))
            step_rel.append(_rel_l2(cached[:, t], oracle[:, t]))
        return prefill_cos, step_cos, step_rel

    def _assert_parity(self, backend, prefill_cos, step_cos, step_rel):
        print(
            f"[{backend}] prefill_cos={prefill_cos:.6f} "
            f"min_step_cos={min(step_cos):.6f} "
            f"max_step_rel_l2={max(step_rel):.4e}"
        )
        self.assertGreaterEqual(prefill_cos, 0.999, "prefill diverged")
        for i, (c, r) in enumerate(zip(step_cos, step_rel)):
            self.assertGreaterEqual(
                c,
                0.999,
                f"decode step {i} diverged from the cache-less forward "
                f"(cos={c:.6f})",
            )
            self.assertLessEqual(
                r,
                0.05,
                f"decode step {i} relative L2 {r:.4e} exceeds the bf16 floor",
            )

    # ---- tests --------------------------------------------------------

    def test_decode_matches_no_cache_tilelang(self):
        """TileLang block-sparse backend: decode == cache-less forward."""
        _tilelang_backend_or_skip(self)
        self._assert_parity("tilelang", *self._parity_report(True))

    @unittest.skipUnless(
        _dsa_available(), "SM100 FlashMLA/cuDNN DSA backend unavailable"
    )
    def test_decode_matches_no_cache_dsa(self):
        """Production (DSA) block-sparse backend: decode == cache-less."""
        _tilelang_backend_or_skip(self)
        self._assert_parity("dsa", *self._parity_report(False))

    def test_dropping_a_sink_is_detectable(self):
        """Calibration, in the saturated-sink regime: decode still reproduces
        the cache-less forward, and replacing either sink with the kernels'
        sinkless ``-1e30`` moves the output far more than that parity gap.

        Without this the parity tests above would pass even if decode silently
        ran a sinkless softmax.
        """
        _tilelang_backend_or_skip(self)
        config = self._build_config(True)
        full_layer, swa_layer = self._build_stack(config)
        hidden = self._hidden()
        tail = slice(PROMPT_LEN, TOTAL_LEN)

        self._set_sinks(swa_layer, saturated=True)
        baseline = self._run_no_cache(full_layer, swa_layer, hidden)
        cached = self._run_with_cache(full_layer, swa_layer, hidden)
        parity_rel = _rel_l2(cached[:, tail], baseline[:, tail])
        print(f"[calibration] saturated parity_rel_l2={parity_rel:.4e}")
        self.assertLessEqual(
            parity_rel,
            5e-3,
            "decode diverged from the cache-less forward with dominant sinks",
        )

        for name in ("swa_attn_sink", "sparse_attn_sink"):
            self._set_sinks(swa_layer, saturated=True, drop=name)
            dropped = self._run_no_cache(full_layer, swa_layer, hidden)
            dropped_rel = _rel_l2(dropped[:, tail], baseline[:, tail])
            ratio = dropped_rel / (parity_rel + 1e-30)
            print(
                f"[calibration] drop({name}) rel_l2={dropped_rel:.4e} "
                f"ratio={ratio:.1f}x"
            )
            self.assertGreater(
                ratio,
                10.0,
                f"dropping {name} moved the output only {ratio:.1f}x the "
                "parity gap; the parity tests cannot detect a dropped sink",
            )

    def test_sinks_are_honoured_at_one_query_token(self):
        """Kernel-level: both SWA branches still fold the sink in at S == 1.

        Bypasses the layer wiring so a failure points straight at a kernel that
        ignores ``attn_sink`` on the decode shape.
        """
        _tilelang_backend_or_skip(self)
        from paddlefleet.tilelang_ops.hysparse import (
            sliding_window_mqa_attention,
        )
        from paddlefleet.tilelang_ops.hysparse.block_sparse_mqa_tl import (
            block_sparse_mqa_attention_tl,
        )

        # Absorbed-MQA decode shapes: Dk = kv_lora_rank + rope, Dv = kv_lora_rank.
        heads, kv_lora_rank, rope = 16, 512, 64
        d_k = kv_lora_rank + rope
        kv_len = TOPK * BLOCK_B
        paddle.seed(11)
        query = paddle.randn([1, 1, heads, d_k], dtype="bfloat16")
        latent = paddle.randn([1, kv_len, d_k], dtype="bfloat16")
        value = latent[..., :kv_lora_rank].contiguous()
        valid_range = paddle.to_tensor([[[0, kv_len]]], dtype="int32")
        sink = paddle.linspace(-1.0, 1.0, heads).astype("float32")

        with paddle.no_grad():
            win_off, _ = sliding_window_mqa_attention(
                query, latent, value, valid_range, None, None, BLOCK_B
            )
            win_on, _ = sliding_window_mqa_attention(
                query, latent, value, valid_range, sink, None, BLOCK_B
            )
            block_indices = paddle.arange(TOPK, dtype="int32").reshape(
                [1, 1, TOPK]
            )
            sparse_off, _ = block_sparse_mqa_attention_tl(
                query,
                latent,
                block_indices,
                valid_range,
                block_B=BLOCK_B,
                kv_lora_rank=kv_lora_rank,
                attn_sink=None,
            )
            sparse_on, _ = block_sparse_mqa_attention_tl(
                query,
                latent,
                block_indices,
                valid_range,
                block_B=BLOCK_B,
                kv_lora_rank=kv_lora_rank,
                attn_sink=sink,
            )

        win_delta = _max_abs_diff(win_on, win_off)
        sparse_delta = _max_abs_diff(sparse_on, sparse_off)
        print(
            f"[S=1 sink] sliding_window delta={win_delta:.4e} "
            f"block_sparse delta={sparse_delta:.4e}"
        )
        self.assertGreater(
            win_delta, 0.0, "sliding-window kernel ignored attn_sink at S=1"
        )
        self.assertGreater(
            sparse_delta, 0.0, "block-sparse kernel ignored attn_sink at S=1"
        )


if __name__ == "__main__":
    unittest.main()

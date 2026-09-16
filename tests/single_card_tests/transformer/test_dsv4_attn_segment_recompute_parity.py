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

"""``dsv4_hybrid_attn_qkv`` / ``_post_core`` must be invisible from outside.

Both switches move a real forward/backward boundary: the segment runs under
``no_grad``, its outputs are freed, and the replay rebuilds the graph from the
saved inputs. Everything reachable from outside the segment must not notice --
same output, same ``dx``, same gradient for every parameter, bit for bit.

Driven through the real ``DSv4HybridSelfAttention.forward``, against a baseline
with the switch off. Bitwise (``assert_array_equal``) rather than a tolerance:
the point of the fixed-order node in the qkv path is that replay does not cost
even an ULP, and a tolerance would hide exactly the regression it prevents.

Covered per config combination: each switch alone, both together, and both plus
``gated_attn`` (whose own span hooks the same tensor, so its replay order
relative to these two is what the registration order in ``forward`` pins down).
The combinations vary ``gated_attn_use_q_lora`` -- it decides whether the gate
consumes ``q_compressed``, i.e. whether it reads a tensor the qkv segment
produces and clears -- and the VHA postmix, which sits inside the post-core
segment and brings its own parameters.

Also asserted, since a leak here is silent: the spans are actually taken (the
flags are on and the holders were populated), and every holder is back to None
when ``forward`` returns, so nothing keeps a cleared buffer alive.
"""

import unittest

import numpy as np
import paddle
from paddle.distributed.fleet.meta_parallel import build_spec_layer

from paddlefleet.models.gpt.gpt_layer_specs import get_attention_spec
from paddlefleet.tensor_parallel.random import model_parallel_cuda_manual_seed
from paddlefleet.transformer.enums import AttnMaskType
from paddlefleet.transformer.transformer_config import TransformerConfig

_SEED = 42
_BATCH, _SEQ, _HIDDEN = 1, 64, 256
# csa_compress_ratios[1] == 4, so layer 1 is a CSA layer: compressor + Lightning
# Indexer, i.e. the qkv segment's outputs feed a side-attached loss too.
_LAYER = 1
_RATIOS = [0, 4, 128, 4]

_REQUIRES_CUDA = unittest.skipUnless(
    paddle.is_compiled_with_cuda(), "requires CUDA for the CSA/Triton kernels"
)

_SPAN_HOLDERS = ("_qkv_recompute", "_post_core_recompute", "_gate_recompute")


def _make_config(recompute_modules, **overrides):
    kwargs = {
        "num_hidden_layers": len(_RATIOS),
        "hidden_size": _HIDDEN,
        "num_attention_heads": 8,
        "params_dtype": paddle.bfloat16,
        "bf16": True,
        "use_bias": False,
        "multi_latent_attention": True,
        "experimental_attention_variant": "dsv4_hybrid",
        "q_lora_rank": 64,
        "kv_lora_rank": 16,
        "qk_nope_head_dim": 16,
        "qk_rope_head_dim": 16,
        "qk_pos_emb_head_dim": 16,
        "v_head_dim": 32,
        "o_groups": 4,
        "o_lora_rank": 32,
        "rope_type": "rope",
        "rotary_base": 10000.0,
        "rotary_percent": 1.0,
        "normalization": "RMSNorm",
        "use_qk_norm": True,
        "csa_compress_ratios": list(_RATIOS),
        "csa_window_size": 16,
        "dsa_index_n_heads": 4,
        "dsa_index_head_dim": 32,
        "dsa_index_topk": 8,
        # Non-zero: the Indexer's side-attached loss is a second consumer of the
        # qkv segment's outputs, and it must survive the replay too.
        "dsa_indexer_loss_coeff": 1.0,
        "dsa_indexer_rotary_interleaved": False,
        "apply_rope_fusion": True,
        "attention_dropout": 0.0,
        "attention_softmax_in_fp32": True,
        "masked_softmax_fusion": False,
        "softmax_type": "vanilla",
        "csa_indexer_backend": "unfused",
        "csa_sparse_attn_backend": "unfused",
        "mqa_sparse_attn_backward_backend": "cudnn",
        "gated_attention": True,
        "gated_attn_use_q_lora": True,
    }
    kwargs.update(overrides)
    config = TransformerConfig(**kwargs)
    # Assigned rather than passed: with granularity "selective" the config
    # validates the module names against this release's table, and these two are
    # what this test is here to exercise.
    config.recompute_granularity = "selective" if recompute_modules else None
    config.recompute_modules = recompute_modules
    return config


def _build(config):
    paddle.seed(_SEED)
    model_parallel_cuda_manual_seed(_SEED)
    spec = get_attention_spec(
        config=config,
        attention_layer_type="dsv4_hybrid_attention",
        attn_mask_type=AttnMaskType.causal,
    )
    attn = build_spec_layer(spec, config=config, layer_number=_LAYER)
    _fix_vha_param_dtype(attn, config)
    attn.train()
    return attn


def _fix_vha_param_dtype(attn, config):
    """Put the VHA parameters in ``params_dtype``, which the module does not.

    ``vha_postmix_U`` / ``_V`` / ``vha_premix_weight`` are created without a
    ``dtype``, so they land on paddle's fp32 default while every projection in
    the same layer follows ``params_dtype`` -- and ``_apply_vha_postmix`` then
    multiplies an fp32 matrix by a bf16 activation. That is a pre-existing DSv4
    defect, unrelated to the recompute segments this file tests, and it makes
    bf16 + VHA unable to run at all; the existing VHA coverage misses it by
    staying in fp32 on a window-only layer. Cast here rather than fix it in
    production so this file stays inside its own scope; drop this once the
    parameters carry a dtype.
    """
    for name in ("vha_postmix_U", "vha_postmix_V", "vha_premix_weight"):
        parameter = getattr(attn, name, None)
        if parameter is None or parameter.dtype == config.params_dtype:
            continue
        casted = parameter.astype(config.params_dtype)
        casted.stop_gradient = parameter.stop_gradient
        attn.__dict__.pop(name, None)
        attn._parameters.pop(name, None)
        setattr(
            attn,
            name,
            attn.create_parameter(
                shape=casted.shape,
                dtype=config.params_dtype,
                default_initializer=paddle.nn.initializer.Assign(casted),
            ),
        )


def _collect(attn, hidden_np):
    """One forward/backward pass; everything observable from outside the spans."""
    hidden = paddle.to_tensor(hidden_np, dtype="bfloat16")
    hidden.stop_gradient = False
    paddle.seed(_SEED)
    model_parallel_cuda_manual_seed(_SEED)

    output, _bias = attn(hidden_states=hidden, attention_mask=None)
    # Weight by position so a mix-up between rows cannot cancel out.
    weights = paddle.arange(1, output.shape[1] + 1, dtype="float32").reshape(
        [1, -1, 1]
    )
    (output.astype("float32") * weights).sum().backward()

    # The holders are consumed by discard_output_and_register_recompute, so a
    # non-None one here means a span was opened and never closed: its output
    # buffer stays cleared and the replay never runs.
    leaked = [
        name for name in _SPAN_HOLDERS if getattr(attn, name, None) is not None
    ]
    return {
        "output": output.astype("float32").numpy().copy(),
        "hidden_grad": hidden.grad.astype("float32").numpy().copy(),
        "param_grads": {
            name: parameter.grad.astype("float32").numpy().copy()
            for name, parameter in attn.named_parameters()
            if parameter.grad is not None
        },
        "leaked_holders": leaked,
    }


_QKV = "dsv4_hybrid_attn_qkv"
_POST = "dsv4_hybrid_attn_post_core"
_GATE = "gated_attn"

# Which switches each case turns on, and which per-layer flags that must set.
_CASES = {
    "qkv": ([_QKV], {"recompute_qkv"}),
    "post_core": ([_POST], {"recompute_post_core"}),
    "qkv+post_core": ([_QKV, _POST], {"recompute_qkv", "recompute_post_core"}),
    "qkv+post_core+gate": (
        [_QKV, _POST, _GATE],
        {"recompute_qkv", "recompute_post_core", "recompute_gated_attn"},
    ),
}

# The config axes that change what the segments contain or who reads their
# outputs. VHA adds parameters inside the post-core segment; use_q_lora decides
# whether the gate reads q_compressed, which the qkv segment clears.
_VARIANTS = {
    "q_lora_gate": {"gated_attn_use_q_lora": True},
    "hidden_gate": {"gated_attn_use_q_lora": False},
    "vha_postmix": {
        "gated_attn_use_q_lora": True,
        "use_vha_attention": True,
        "vha_postmix_rank": 4,
    },
    "vha_postmix_fused_inv_rope": {
        "gated_attn_use_q_lora": True,
        "use_vha_attention": True,
        "vha_postmix_rank": 4,
        # Folds the inverse RoPE into the postmix GEMM, so the post-core segment
        # holds one fused op instead of two.
        "fuse_inv_rope_into_vha_postmix": True,
    },
}


@_REQUIRES_CUDA
class TestDSv4AttnSegmentRecomputeParity(unittest.TestCase):
    """Each selective attention segment must be bitwise-invisible."""

    @classmethod
    def setUpClass(cls):
        paddle.set_device("gpu")

    def _assert_identical(self, baseline, recomputed, label):
        np.testing.assert_array_equal(
            baseline["output"],
            recomputed["output"],
            err_msg=f"{label}: output",
        )
        np.testing.assert_array_equal(
            baseline["hidden_grad"],
            recomputed["hidden_grad"],
            err_msg=f"{label}: hidden_states.grad",
        )
        self.assertEqual(
            set(baseline["param_grads"]),
            set(recomputed["param_grads"]),
            f"{label}: a parameter lost its gradient",
        )
        for name, expected in baseline["param_grads"].items():
            np.testing.assert_array_equal(
                expected,
                recomputed["param_grads"][name],
                err_msg=f"{label}: grad of {name}",
            )

    def test_segments_match_the_no_recompute_baseline(self):
        hidden_np = (
            np.random.RandomState(0)
            .randn(_BATCH, _SEQ, _HIDDEN)
            .astype("float32")
        )
        for variant, overrides in _VARIANTS.items():
            baseline = _collect(
                _build(_make_config(None, **overrides)), hidden_np
            )
            self.assertEqual(
                baseline["leaked_holders"],
                [],
                f"{variant}: baseline leaked a span holder",
            )
            for case, (modules, expected_flags) in _CASES.items():
                label = f"{variant}/{case}"
                with self.subTest(variant=variant, case=case):
                    config = _make_config(
                        {name: [_LAYER] for name in modules}, **overrides
                    )
                    attn = _build(config)
                    for flag in expected_flags:
                        self.assertTrue(
                            getattr(attn, flag),
                            f"{label}: {flag} did not turn on, so this "
                            "comparison would be vacuous",
                        )
                    recomputed = _collect(attn, hidden_np)
                    self.assertEqual(
                        recomputed["leaked_holders"],
                        [],
                        f"{label}: span holder still set after forward",
                    )
                    self._assert_identical(baseline, recomputed, label)


if __name__ == "__main__":
    unittest.main()

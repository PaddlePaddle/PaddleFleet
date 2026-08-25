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

"""Guard: DSv4 CSA attention must survive a frozen backbone + trainable Indexer.

This is the DSv4 phase 2 shape (``train_indexer_only``): every parameter
except the ``CSAIndexer`` is frozen, and the layer input is detached. The Indexer
loss is attached through a PyLayer, so the attention output stays differentiable
and backward walks the whole CSA path with ``stop_gradient=True`` on most of its
tensors.

That makes this test a tripwire for **any** PyLayer on the CSA attention path:
Paddle requires ``backward`` to return ``None`` at every position whose forward
input had ``stop_gradient=True``. A new fused kernel that unconditionally returns
a gradient for its weight fails here with

    ValueError: ... backward function should return None at N position, because
    it's forward Tensor's stopgradient is true.

instead of only showing up in a real phase 2 training run. Already caught this
way: the two indexer-loss auto scalers, ``CSASparseAttention`` (frozen
``attn_sink``), ``GroupedOutputFP8`` and ``GroupedMatmulTriton`` (the two
``o_groups`` projection branches).

Runs on GPU only: the CSA/Indexer kernels are Triton/cuDNN.
"""

import unittest

import paddle

from paddlefleet.tensor_parallel.random import model_parallel_cuda_manual_seed
from paddlefleet.transformer.csa_attention import CSAIndexer

from .test_dsv4_hybrid_attention import _build_attention, _make_config

# csa_compress_ratios defaults to [0, 4, 128, 4]; layer_number 1 maps to ratio 4,
# the only kind that builds an Indexer (1 < ratio < 128 and not csa_dense_mode).
_CSA_LAYER_NUMBER = 1
_SEED = 42


def _has_usable_cuda():
    if not paddle.is_compiled_with_cuda():
        return False
    try:
        paddle.set_device("gpu:0")
    except Exception:
        return False
    return paddle.get_device().startswith("gpu")


@unittest.skipUnless(
    _has_usable_cuda(), "CSA/Indexer kernels require a usable CUDA device"
)
class TestFrozenBackboneIndexerForwardBackward(unittest.TestCase):
    BATCH, SEQ = 1, 128

    def _build_phase2_attention(self):
        """Real DSv4 CSA attention with only the Indexer trainable."""
        model_parallel_cuda_manual_seed(_SEED)
        config = _make_config(csa_dense_mode=False, dsa_indexer_loss_coeff=1.0)
        attn = _build_attention(config, layer_number=_CSA_LAYER_NUMBER)
        attn.train()

        indexer = attn.core_attention.indexer
        self.assertIsInstance(
            indexer,
            CSAIndexer,
            "config no longer builds a CSAIndexer on this layer; the guard would "
            "silently stop covering the phase 2 path",
        )
        indexer_ids = {id(p) for p in indexer.parameters()}
        for param in attn.parameters():
            param.stop_gradient = id(param) not in indexer_ids
        return config, attn, indexer_ids

    def _forward_backward(self, config, attn):
        hidden = paddle.randn(
            [self.BATCH, self.SEQ, config.hidden_size], dtype=paddle.bfloat16
        )
        # Frozen backbone: the layer input arrives detached.
        hidden.stop_gradient = True
        output, _ = attn(hidden_states=hidden, attention_mask=None)
        output.cast("float32").sum().backward()
        return output

    def test_backward_runs_and_only_indexer_gets_grads(self):
        config, attn, indexer_ids = self._build_phase2_attention()
        frozen = [
            n for n, p in attn.named_parameters() if id(p) not in indexer_ids
        ]
        self.assertTrue(frozen, "expected a non-empty frozen backbone")

        output = self._forward_backward(config, attn)

        # The attached indexer loss keeps the output differentiable even though
        # every input tensor and backbone parameter is detached.
        self.assertFalse(output.stop_gradient)
        self.assertTrue(paddle.isfinite(output.astype("float32")).all().item())

        missing = [
            name
            for name, param in attn.named_parameters()
            if id(param) in indexer_ids and param.grad is None
        ]
        self.assertEqual(
            missing, [], f"Indexer parameters without gradient: {missing}"
        )

        leaked = [
            name
            for name, param in attn.named_parameters()
            if id(param) not in indexer_ids and param.grad is not None
        ]
        self.assertEqual(
            leaked, [], f"frozen parameters that received a gradient: {leaked}"
        )

    def test_indexer_gradients_are_non_zero(self):
        config, attn, indexer_ids = self._build_phase2_attention()
        self._forward_backward(config, attn)
        zero = [
            name
            for name, param in attn.named_parameters()
            if id(param) in indexer_ids
            and int(paddle.count_nonzero(param.grad)) == 0
        ]
        self.assertEqual(
            zero, [], f"Indexer parameters with an all-zero gradient: {zero}"
        )

    def test_trainable_backbone_still_gets_grads(self):
        """Control: phase 1/3 shape must be unaffected by the phase 2 guards."""
        model_parallel_cuda_manual_seed(_SEED)
        config = _make_config(csa_dense_mode=False, dsa_indexer_loss_coeff=1.0)
        attn = _build_attention(config, layer_number=_CSA_LAYER_NUMBER)
        attn.train()
        hidden = paddle.randn(
            [self.BATCH, self.SEQ, config.hidden_size], dtype=paddle.bfloat16
        )
        hidden.stop_gradient = False
        output, _ = attn(hidden_states=hidden, attention_mask=None)
        output.cast("float32").sum().backward()

        self.assertIsNotNone(hidden.grad)
        with_grad = sum(
            1 for _, p in attn.named_parameters() if p.grad is not None
        )
        total = len(list(attn.named_parameters()))
        self.assertEqual(
            with_grad,
            total,
            "every parameter should receive a gradient when nothing is frozen",
        )


@unittest.skipUnless(
    _has_usable_cuda(), "CSA/Indexer kernels require a usable CUDA device"
)
class TestCSAZeroLossCoeffSkipsLossPath(unittest.TestCase):
    """``dsa_indexer_loss_coeff == 0`` must disable the indexer-loss path
    entirely, not merely zero the gradient.

    Guards the short-circuits in ``_compute_indexer_compressed_topk_idxs`` /
    ``_forward_cp``: without them a coeff=0 CSA training run still built the
    target/KL kernels (``FusedDSAIndexerLoss``, ``TilelangIndexerLossState``)
    and their autoscaling PyLayers, paying forward+backward costs for a
    gradient that was identically 0. Any of those symbols being constructed
    or applied here fails the test loudly, while sparse-attention top-k must
    keep working (forward + backward stay clean).
    """

    BATCH, SEQ = 1, 128

    CASES = (
        # Paddle reference loss (unfused indexer + unfused sparse attn).
        {
            "csa_indexer_backend": "unfused",
            "csa_sparse_attn_backend": "unfused",
        },
        # TileLang indexer backend would otherwise build a
        # TilelangIndexerLossState and TileLangCSAIndexerLossAutoScaler.
        # The TileLang kernels constrain the shapes: the indexer top-k kernel
        # tiles over indexer heads (needs <= 64 and divisible by 8, the
        # _make_config default is 4) and the sparse-MQA kernel's MMA warp
        # partition needs a head dim of at least 64 (default is 32).
        {
            "csa_indexer_backend": "tilelang",
            "csa_sparse_attn_backend": "tilelang",
            "dsa_index_n_heads": 16,
            "v_head_dim": 128,
        },
    )

    def test_zero_coeff_runs_without_any_loss_construction(self):
        from unittest import mock

        import paddlefleet.transformer.csa_attention as csa_mod
        from paddlefleet.transformer.dsa_attention import (
            DSAIndexerLossLoggingHelper,
        )

        for backend in self.CASES:
            with self.subTest(backend=backend["csa_indexer_backend"]):
                model_parallel_cuda_manual_seed(_SEED)
                config = _make_config(
                    csa_dense_mode=False,
                    dsa_indexer_loss_coeff=0.0,  # disabled
                    **backend,
                )
                attn = _build_attention(config, layer_number=_CSA_LAYER_NUMBER)
                attn.train()
                DSAIndexerLossLoggingHelper.tracker = {}

                hidden = paddle.randn(
                    [self.BATCH, self.SEQ, config.hidden_size],
                    dtype=paddle.bfloat16,
                )

                no_call = AssertionError(
                    "dsa_indexer_loss_coeff=0 must not build the indexer loss "
                    f"path (backend={backend['csa_indexer_backend']})"
                )
                with (
                    mock.patch.object(
                        csa_mod.FusedDSAIndexerLoss,
                        "apply",
                        side_effect=no_call,
                    ),
                    mock.patch.object(
                        csa_mod.DSAIndexerLossAutoScaler,
                        "apply",
                        side_effect=no_call,
                    ),
                    mock.patch.object(
                        csa_mod, "TilelangIndexerLossState", side_effect=no_call
                    ),
                    mock.patch.object(
                        csa_mod.TileLangCSAIndexerLossAutoScaler,
                        "apply",
                        side_effect=no_call,
                    ),
                ):
                    output, _ = attn(hidden_states=hidden, attention_mask=None)

                self.assertEqual(
                    tuple(output.shape),
                    (self.BATCH, self.SEQ, config.hidden_size),
                )
                # Backward must also stay on the no-loss path.
                output.cast("float32").sum().backward()
                self.assertNotIn("values", DSAIndexerLossLoggingHelper.tracker)
                DSAIndexerLossLoggingHelper.tracker = {}


if __name__ == "__main__":
    unittest.main()

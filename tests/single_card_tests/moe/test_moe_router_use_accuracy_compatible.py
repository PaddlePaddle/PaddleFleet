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
"""Behavioral tests for the ``use_accuracy_compatible`` router branches of
``StandardMoERouter`` (``transformer/moe/moe_router.py``).

Two production behaviors are pinned with small, fully hand-derived inputs whose
expected values are computed here by pen-and-paper, independent of the router's
own arithmetic:

``_cal_seq_aux_loss`` (accuracy-compatible branch)
    The Megatron-aligned sequence auxiliary loss.  On a line of ``S`` tokens
    routed with ``top_k`` experts over ``E`` experts, the branch recomputes the
    routing map from the probabilities, drops padding lines (rows whose incoming
    routing map is all-zero), and evaluates

        loss = sum_e( (sum_s probs[b,s,e]) * (sum_s routing_map[b,s,e]) )
               * E / (top_k * N^2)      with  N = (sum tokens_per_expert)/(top_k*B),

    then divides by the batch size ``B`` when ``B > 1``.  We check the returned
    scalar against numbers derived by hand, and separately check that padding
    lines are actually excluded (the with-mask and without-mask results are made
    numerically distinct so a dropped mask cannot slip through).

``_topk_noaux_tc`` (accuracy-compatible branch)
    Top-k expert *selection* uses ``scores + e_score_correction_bias`` while the
    returned gate *weights* must come from the raw ``scores`` at the selected
    indices (via ``gather_nd``).  The fixture is built so the bias flips which
    experts win, letting the test prove both that the bias steers selection and
    that it never leaks into the returned weights.

There is no accelerator or Paddle dependency guaranteed locally, so the Paddle
imports are guarded and the whole module skips with an honest reason when Paddle
is unavailable.  ``get_context_parallel_world_size`` is patched to ``1`` to
select the genuine single-card / world-size-1 local path (an environment query,
not the code under test); no collective is faked.
"""

import os
import sys
import unittest
from unittest.mock import patch

# The package lives under ``src/``; make it importable when not pip-installed.
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
_SRC = os.path.join(_REPO_ROOT, "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

try:
    import numpy as np
    import paddle

    from paddlefleet.transformer.moe import moe_router as _moe_router
    from paddlefleet.transformer.moe.moe_router import StandardMoERouter

    _IMPORT_ERROR = None
except ImportError as exc:  # honest skip: dependency genuinely missing
    _IMPORT_ERROR = repr(exc)

_CP_WORLD_SIZE = (
    "paddlefleet.transformer.moe.moe_router.get_context_parallel_world_size"
)


class _RouterConfig:
    """Minimal stand-in for ``TransformerConfig`` holding only the fields the
    router constructor and the two methods under test read.  It is a setup
    collaborator, not the code under test."""

    def __init__(self, **overrides):
        self.hidden_size = 8
        self.n_routed_experts = overrides.pop("n_routed_experts", 2)
        self.num_experts_per_tok = overrides.pop("num_experts_per_tok", 1)
        self.norm_topk_prob = True
        self.n_group = 1
        self.topk_group = 1
        self.routed_scaling_factor = 1.0
        self.routed_scaling_factor_learnable = False
        self.tensor_model_parallel_size = 1
        self.sequence_parallel = False
        self.scoring_func = "softmax"
        self.topk_method = overrides.pop("topk_method", "greedy")
        self.moe_router_load_balancing_type = "seq_aux_loss"
        self.moe_router_force_load_balancing = False
        self.expert_model_parallel_size = 1
        self.moe_split_feature_routing = False
        self.gpt_model_use_experimental_version = False
        self.experimental_dataflow = False
        self.router_aux_loss_coef = 0.01
        self.pad_token_id = 0
        self.params_dtype = "float32"
        self.init_method = paddle.nn.initializer.Constant(0.0)
        self.use_accuracy_compatible = overrides.pop(
            "use_accuracy_compatible", True
        )
        self._extra = {"seq_aux": False}
        for key, value in overrides.items():
            setattr(self, key, value)

    def get(self, key, default=None):
        if key in self._extra:
            return self._extra[key]
        return getattr(self, key, default)


def _build_router(**overrides):
    """Construct a StandardMoERouter on the single-card local path."""
    with patch(_CP_WORLD_SIZE, return_value=1):
        return StandardMoERouter(_RouterConfig(**overrides))


@unittest.skipUnless(
    _IMPORT_ERROR is None, f"paddle/paddlefleet unavailable: {_IMPORT_ERROR}"
)
class TestSeqAuxLossAccuracyCompatible(unittest.TestCase):
    """``_cal_seq_aux_loss`` accuracy-compatible branch, hand-derived values."""

    def test_single_line_hand_derived(self):
        # B=1, S=2, E=2, top_k=1.  Row s0 -> expert 0, row s1 -> expert 1.
        #   tokens_per_expert = [1, 1]
        #   aggregated probs   = [0.8+0.3, 0.2+0.7] = [1.1, 0.9]
        #   per_expert         = [1.1, 0.9]  -> sum 2.0
        #   N = (1+1)/(top_k=1 * B=1) = 2
        #   scalar = E / (top_k * N^2) = 2 / (1 * 4) = 0.5
        #   loss = 2.0 * 0.5 = 1.0  (B=1, no /B)
        router = _build_router()
        probs = paddle.to_tensor([[0.8, 0.2], [0.3, 0.7]], dtype="float32")
        routing_map = paddle.to_tensor(
            [[1.0, 0.0], [0.0, 1.0]], dtype="float32"
        )
        loss = router._cal_seq_aux_loss(
            probs, top_k=1, routing_map=routing_map, seq_len=2, batch_size=1
        )
        self.assertEqual(loss.shape, [])
        np.testing.assert_allclose(
            loss.numpy(), np.float32(1.0), rtol=1e-6, atol=1e-7
        )

    def test_batch_two_divides_by_batch(self):
        # B=2, S=2, E=2, top_k=1.
        #   line0: both tokens -> expert 0; line1: both tokens -> expert 1
        #   tokens_per_expert = [[2,0],[0,2]]
        #   aggregated        = [[1.5,0.5],[0.5,1.5]]
        #   per_expert        = [[3.0,0.0],[0.0,3.0]] -> sum 6.0
        #   N = (2+0+0+2)/(1*2) = 2 ; scalar = 2/(1*4) = 0.5
        #   loss = 6.0 * 0.5 / B(=2) = 1.5
        router = _build_router()
        probs = paddle.to_tensor(
            [[0.9, 0.1], [0.6, 0.4], [0.2, 0.8], [0.3, 0.7]], dtype="float32"
        )
        routing_map = paddle.to_tensor(
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], dtype="float32"
        )
        loss = router._cal_seq_aux_loss(
            probs, top_k=1, routing_map=routing_map, seq_len=2, batch_size=2
        )
        self.assertEqual(loss.shape, [])
        np.testing.assert_allclose(
            loss.numpy(), np.float32(1.5), rtol=1e-6, atol=1e-7
        )

    def test_padding_line_is_excluded(self):
        # B=1, S=2, E=2, top_k=1.  s1 is padding: its incoming routing_map row
        # is all-zero and its probs are zeroed, so it must not contribute.
        #   masked tokens_per_expert = [1, 0]
        #   aggregated               = [0.8, 0.2]
        #   per_expert               = [0.8, 0.0] -> sum 0.8
        #   N = 1/(1*1) = 1 ; scalar = 2/(1*1) = 2 ; loss = 0.8 * 2 = 1.6
        # If the padding mask were dropped the routed token count and per-expert
        # sum would change, yielding <= 0.8 -- well away from 1.6.
        router = _build_router()
        probs = paddle.to_tensor([[0.8, 0.2], [0.0, 0.0]], dtype="float32")
        routing_map = paddle.to_tensor(
            [[1.0, 0.0], [0.0, 0.0]], dtype="float32"
        )
        loss = router._cal_seq_aux_loss(
            probs, top_k=1, routing_map=routing_map, seq_len=2, batch_size=1
        )
        self.assertEqual(loss.shape, [])
        np.testing.assert_allclose(
            loss.numpy(), np.float32(1.6), rtol=1e-6, atol=1e-7
        )
        # Guard: the value must differ from the "mask dropped" outcome (<=0.8).
        self.assertFalse(
            np.allclose(loss.numpy(), np.float32(0.8), rtol=1e-3, atol=1e-6)
        )


@unittest.skipUnless(
    _IMPORT_ERROR is None, f"paddle/paddlefleet unavailable: {_IMPORT_ERROR}"
)
class TestTopkNoauxTcAccuracyCompatible(unittest.TestCase):
    """``_topk_noaux_tc`` accuracy-compatible branch, hand-derived values."""

    def test_bias_steers_selection_weights_from_raw_scores(self):
        # E=4, k=2, n_group=1.  Raw scores rank experts 1>3>2>0.
        # A large bias on expert 0 makes scores_for_choice rank 0>1>3>2, so the
        # top-2 selection must become experts [0, 1] (not [1, 3]).  The returned
        # gate weights must be the RAW scores at [0, 1] = [0.1, 0.5], proving the
        # bias never leaks into the weights.
        router = _build_router(topk_method="noaux_tc", n_routed_experts=4)
        router.e_score_correction_bias.set_value(
            paddle.to_tensor([2.0, 0.0, 0.0, 0.0], dtype="float32")
        )
        scores = paddle.to_tensor([[0.1, 0.5, 0.3, 0.4]], dtype="float32")
        weight, idx = router._topk_noaux_tc(
            scores, k=2, n_group=1, topk_group=1
        )
        self.assertEqual(list(idx.shape), [1, 2])
        np.testing.assert_array_equal(idx.numpy(), np.array([[0, 1]]))
        np.testing.assert_allclose(
            weight.numpy(),
            np.array([[0.1, 0.5]], dtype="float32"),
            rtol=1e-6,
            atol=1e-7,
        )

    def test_matches_default_weight_gather(self):
        # gather_nd (accuracy branch) and take_along_axis (default branch) must
        # select the same raw-score weights for identical selection.  Both share
        # the same bias so the routing decision is identical; only the lookup
        # mechanism differs.
        scores = paddle.to_tensor(
            [[0.1, 0.5, 0.3, 0.4], [0.6, 0.2, 0.9, 0.05]], dtype="float32"
        )
        bias = paddle.to_tensor([2.0, 0.0, 0.0, 0.0], dtype="float32")

        aligned = _build_router(topk_method="noaux_tc", n_routed_experts=4)
        aligned.e_score_correction_bias.set_value(bias)
        default = _build_router(
            topk_method="noaux_tc",
            n_routed_experts=4,
            use_accuracy_compatible=False,
        )
        default.e_score_correction_bias.set_value(bias)

        w_a, i_a = aligned._topk_noaux_tc(scores, k=2, n_group=1, topk_group=1)
        w_d, i_d = default._topk_noaux_tc(scores, k=2, n_group=1, topk_group=1)
        np.testing.assert_array_equal(i_a.numpy(), i_d.numpy())
        np.testing.assert_allclose(
            w_a.numpy(), w_d.numpy(), rtol=1e-6, atol=1e-7
        )


if __name__ == "__main__":
    unittest.main()

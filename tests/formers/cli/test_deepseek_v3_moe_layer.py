# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
"""Behavior tests for the DeepSeek-V3 drop-token MoE layer.

These tests target the real ``MoELayer.forward_drop_token`` expert-dispatch and
weighted-combine math in single-process (dummy) mode. The layer is driven with
explicit routing (``using_post_norm_recompute=True`` bypasses the gate) so the
whole dispatch -> per-expert compute -> repositioning -> probability-weighted
combine pipeline is exercised against an independently hand-derived oracle:

    final[t] = sum_j token[t] * scale[expert(t, j)] * topk_weight[t, j]
                                                     * token_priority[t, j]

Tokens carry content-distinguishable values and each expert applies a distinct
per-channel scale, so a wrong dispatch, a channel permutation, a dropped
probability factor, or mis-repositioned output changes the result rather than
only the shape. One expert is left with zero routed tokens to cover the
padding/empty-expert exclusion branch.

Numeric paths require paddle/paddlefleet; when unavailable the tests skipTest
on ImportError instead of masking the failure. The real production module is
loaded directly from its source file to avoid the package ``__init__`` pulling
in the full training workflow (AutoTokenizer); the class under test is the
genuine production ``MoELayer``.
"""

import importlib.util
import os
import sys
import types
import unittest

_IMPORT_ERROR = None
paddle = None
nn = None
np = None
MoELayer = None

try:
    import numpy as np
    import paddle
    from paddle import nn

    # Load the real production moe_layer.py without executing the package
    # __init__ (which imports workflow.py -> AutoTokenizer). Registering the
    # package with __path__ lets moe_layer's relative imports resolve normally.
    _MODULE_DIR = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "src",
            "paddlefleet",
            "cli",
            "train",
            "deepseek_v3_pretrain",
        )
    )
    _PKG = "paddlefleet.cli.train.deepseek_v3_pretrain"
    if _PKG not in sys.modules:
        _pkg_mod = types.ModuleType(_PKG)
        _pkg_mod.__path__ = [_MODULE_DIR]
        _pkg_mod.__package__ = _PKG
        sys.modules[_PKG] = _pkg_mod
    _full = _PKG + ".moe_layer"
    if _full in sys.modules:
        MoELayer = sys.modules[_full].MoELayer
    else:
        _spec = importlib.util.spec_from_file_location(
            _full, os.path.join(_MODULE_DIR, "moe_layer.py")
        )
        _mod = importlib.util.module_from_spec(_spec)
        sys.modules[_full] = _mod
        _spec.loader.exec_module(_mod)
        MoELayer = _mod.MoELayer
except ImportError as exc:  # paddle / paddlefleet not installed in this env
    _IMPORT_ERROR = exc


# --- test fixture parameters (module-level so the oracle uses the same data) --
_D_MODEL = 3
_NUM_EXPERTS = 4
_TOP_K = 2

# Four content-distinguishable tokens (batch=1, seq=4). Distinct nonzero rows so
# a channel permutation or token mix-up is observable in the output.
_TOKENS = [
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0],
    [10.0, 11.0, 12.0],
]

# Distinct per-expert, per-channel scales. Expert e maps x -> x * scale[e].
_SCALES = [
    [2.0, 3.0, 4.0],
    [10.0, 20.0, 30.0],
    [-1.0, -2.0, -3.0],
    [5.0, 0.5, 100.0],  # expert 3 receives no tokens (exclusion branch)
]

# Explicit routing. Counts per expert: e0=3, e1=3, e2=2, e3=0.
_TOPK_IDS = [
    [0, 1],
    [0, 2],
    [1, 2],
    [0, 1],
]
_TOPK_WEIGHT = [
    [0.50, 0.25],
    [0.75, 0.10],
    [0.20, 0.60],
    [0.30, 0.90],
]
# Non-unit priorities so a dropped priority factor changes the result.
_TOKEN_PRIORITY = [
    [1.0, 2.0],
    [1.0, 1.0],
    [3.0, 0.5],
    [2.0, 1.5],
]


def _combine_oracle():
    """Independently derived expected combine output, shape [T, d_model]."""
    tokens = np.asarray(_TOKENS, dtype=np.float64)
    scales = np.asarray(_SCALES, dtype=np.float64)
    ids = np.asarray(_TOPK_IDS, dtype=np.int64)
    weight = np.asarray(_TOPK_WEIGHT, dtype=np.float64)
    priority = np.asarray(_TOKEN_PRIORITY, dtype=np.float64)

    num_tokens = tokens.shape[0]
    out = np.zeros((num_tokens, _D_MODEL), dtype=np.float64)
    for t in range(num_tokens):
        for j in range(_TOP_K):
            expert = ids[t, j]
            contribution = (
                tokens[t] * scales[expert] * weight[t, j] * priority[t, j]
            )
            out[t] += contribution
    return out


def _expected_received_per_expert():
    """Multiset of token rows each expert should receive (by content)."""
    tokens = np.asarray(_TOKENS, dtype=np.float64)
    ids = np.asarray(_TOPK_IDS, dtype=np.int64)
    received = {e: [] for e in range(_NUM_EXPERTS)}
    for t in range(tokens.shape[0]):
        for j in range(_TOP_K):
            received[ids[t, j]].append(tokens[t])
    return {
        e: np.asarray(rows, dtype=np.float64) for e, rows in received.items()
    }


def _sorted_rows(arr):
    """Sort rows lexicographically for order-independent comparison."""
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return arr.reshape(0, _D_MODEL)
    order = np.lexsort(arr.T[::-1])
    return arr[order]


class DeepseekV3DropTokenMoEBehaviorTest(unittest.TestCase):
    """Content-level checks of MoELayer.forward_drop_token dispatch/combine."""

    def setUp(self):
        if MoELayer is None:
            self.skipTest(
                "paddle/paddlefleet unavailable: " + repr(_IMPORT_ERROR)
            )
        paddle.set_device("cpu")

    def _build_layer(self):
        """Construct the real MoELayer with recording, distinctly-scaled experts.

        Returns the layer and the list of per-expert recording objects so a test
        can inspect exactly which token identities each expert received.
        """

        class _ScaleRecordingExpert(nn.Layer):
            """Real layer: forward returns x * scale and records its input."""

            def __init__(self, d_model):
                super().__init__()
                self.scale = self.create_parameter(
                    shape=[d_model],
                    default_initializer=nn.initializer.Constant(1.0),
                )
                self.received = []

            def forward(self, x):
                self.received.append(x.numpy().copy())
                return x * self.scale

        class _StubGate(nn.Layer):
            """Non-tested collaborator; routing is injected, so gate is unused."""

            def __init__(self, top_k):
                super().__init__()
                self.top_k = top_k

        config = types.SimpleNamespace()
        gate = _StubGate(top_k=_TOP_K)
        layer = MoELayer(
            config=config,
            n_routed_experts=_NUM_EXPERTS,
            expert_class=_ScaleRecordingExpert,
            expert_kwargs={"d_model": _D_MODEL},
            gate=gate,
            using_post_norm_recompute=True,
        )
        # Sanity: single-process dummy mode, all experts local.
        self.assertTrue(layer.is_dummy_moe)
        self.assertEqual(layer.moe_rank, 0)
        self.assertEqual(layer.n_routed_experts_per_device, _NUM_EXPERTS)
        self.assertEqual(len(layer.experts), _NUM_EXPERTS)

        experts = []
        for e in range(_NUM_EXPERTS):
            expert = layer.experts[e]
            self.assertIsNotNone(expert)
            expert.scale.set_value(
                paddle.to_tensor(_SCALES[e], dtype=expert.scale.dtype)
            )
            experts.append(expert)
        return layer, experts

    def _run_forward(self, layer):
        hidden = paddle.to_tensor(
            np.asarray(_TOKENS, dtype=np.float32).reshape(
                1, len(_TOKENS), _D_MODEL
            ),
            dtype="float32",
        )
        topk_ids = paddle.to_tensor(
            np.asarray(_TOPK_IDS, dtype=np.int64), dtype="int64"
        )
        topk_weight = paddle.to_tensor(
            np.asarray(_TOPK_WEIGHT, dtype=np.float32), dtype="float32"
        )
        token_priority = paddle.to_tensor(
            np.asarray(_TOKEN_PRIORITY, dtype=np.float32), dtype="float32"
        )
        l_aux = paddle.to_tensor(0.125, dtype="float32")
        l_zloss = paddle.to_tensor(0.375, dtype="float32")
        out, out_aux, out_zloss = layer.forward_drop_token(
            hidden,
            capacity=4,
            topk_weight=topk_weight,
            topk_ids=topk_ids,
            token_priority=token_priority,
            l_aux=l_aux,
            l_zloss=l_zloss,
        )
        return out, out_aux, out_zloss, l_aux, l_zloss

    def test_weighted_combine_matches_hand_derived_oracle(self):
        """Final output equals the independently computed weighted combine."""
        layer, _ = self._build_layer()
        out, _, _, _, _ = self._run_forward(layer)

        self.assertEqual(list(out.shape), [1, len(_TOKENS), _D_MODEL])
        actual = out.numpy().reshape(len(_TOKENS), _D_MODEL).astype(np.float64)
        expected = _combine_oracle()
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

        # Content, not just shape: every token row is distinct here, so a
        # channel-permuted or collapsed output would not match the oracle.
        for t in range(len(_TOKENS)):
            for u in range(t + 1, len(_TOKENS)):
                self.assertFalse(
                    np.allclose(actual[t], actual[u]),
                    msg="token outputs collapsed; content not preserved",
                )

    def test_each_expert_receives_correct_token_identities(self):
        """Dispatch routes the exact token contents each expert should get."""
        layer, experts = self._build_layer()
        self._run_forward(layer)

        expected = _expected_received_per_expert()
        for e in range(_NUM_EXPERTS):
            if experts[e].received:
                got = np.concatenate(experts[e].received, axis=0)
            else:
                got = np.zeros((0, _D_MODEL), dtype=np.float64)
            np.testing.assert_allclose(
                _sorted_rows(got),
                _sorted_rows(expected[e]),
                rtol=1e-6,
                atol=1e-6,
                err_msg=f"expert {e} received wrong token identities",
            )

    def test_empty_expert_is_excluded_from_dispatch(self):
        """Expert 3 gets zero routed tokens and is never invoked."""
        layer, experts = self._build_layer()
        self._run_forward(layer)
        self.assertEqual(len(experts[3].received), 0)
        for e in range(3):
            self.assertGreater(len(experts[e].received), 0)

    def test_auxiliary_losses_pass_through_unchanged(self):
        """forward_drop_token returns the injected l_aux / l_zloss unchanged."""
        layer, _ = self._build_layer()
        _, out_aux, out_zloss, l_aux, l_zloss = self._run_forward(layer)
        self.assertIs(out_aux, l_aux)
        self.assertIs(out_zloss, l_zloss)

    def test_swapping_two_experts_changes_output(self):
        """A mis-routed dispatch must be detectable at the output.

        Swapping expert 0 and expert 1 scales (as a wrong assignment would)
        yields output that no longer matches the correct-assignment oracle.
        """
        layer, experts = self._build_layer()
        experts[0].scale.set_value(
            paddle.to_tensor(_SCALES[1], dtype=experts[0].scale.dtype)
        )
        experts[1].scale.set_value(
            paddle.to_tensor(_SCALES[0], dtype=experts[1].scale.dtype)
        )
        out, _, _, _, _ = self._run_forward(layer)
        actual = out.numpy().reshape(len(_TOKENS), _D_MODEL).astype(np.float64)
        with self.assertRaises(AssertionError):
            np.testing.assert_allclose(
                actual, _combine_oracle(), rtol=1e-5, atol=1e-5
            )


if __name__ == "__main__":
    unittest.main()

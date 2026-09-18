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
"""Behavior tests for pure helpers in ``transformer/moe/fusion_layer_utils``.

The module wires MoE expert compute to fused/fp8 GEMM kernels, but a handful of
its helpers are exercisable numeric/plumbing logic that a CPU host can verify
without a GPU:

* ``_resolve_sonic_config_bool`` -- tri-state config resolution (explicit value
  wins, then a ``resolve_<name>`` callback, else ``False``).
* ``_gate_up_out_dim`` -- reads the gate_up projection output width off either
  the per-expert ``up_gate_proj.weight`` layout or the grouped ``weight1``
  stack, with a ``2 * hidden`` fallback.
* ``_fwd_pre_permute_feature_sizes`` / ``_bwd_pre_permute_feature_sizes`` --
  per-``FP8_ALIGN``-token byte budgets used by the subbatch sizer. The backward
  budget switches between an in-place peak (``do1`` reuses ``o1``) and an
  out-of-place peak (separate ``do1``) driven by the ``USE_INPLACE_SWIGLU_BWD``
  build flag, and both ``clamp_value > 0`` and ``activation_type == "situ"``
  force the out-of-place peak regardless of that flag.
* ``_pad_front_rows`` / ``_restore_hybrid_ep_prob_grad_shape`` -- shape
  restoration that keeps real rows in the leading positions and zero-fills the
  tail, plus the ``[N, 1] -> [N]`` squeeze on the HybridEP prob-grad contract.
* ``_hybrid_ep_prepare_expert_counts`` -- returns a Python list of int64 counts
  unless the fp8 + fusion path is active, where it must hand back the tensor.
* ``UnZipNode`` / ``ZipNode`` cache slots -- ordered save/restore of the two
  cached tensors, and the empty no-op contract on ``ZipNode``.

Expected values below are hand-derived from those documented contracts and do
not call the code under test to produce them. The whole module transitively
imports ``paddle`` and ``paddlefleet_ops`` at import time, so when either is
absent every test skips with an honest reason rather than reporting a fake pass.
"""

import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
        ),
        "src",
    ),
)

_IMPORT_ERROR = None
try:
    import paddle

    from paddlefleet.transformer.moe import fusion_layer_utils as flu
    from paddlefleet.transformer.moe.fusion_layer_utils import (
        MlpNode,
        UnZipNode,
        ZipNode,
    )
except Exception as exc:
    _IMPORT_ERROR = exc

FP8_ALIGN = 128


def _bare_mlp_node(**attrs):
    """An ``MlpNode`` whose ``__init__`` (which builds real GEMM nodes needing a
    GPU) is bypassed, with only the attributes the pure helpers read populated.
    """
    node = object.__new__(MlpNode)
    node.experts = None
    node.experts_group_gemm_node = None
    node.activation_type = "swiglu"
    for key, value in attrs.items():
        setattr(node, key, value)
    return node


@unittest.skipUnless(
    _IMPORT_ERROR is None,
    f"fusion_layer_utils import unavailable on this host: {_IMPORT_ERROR!r}",
)
class ResolveSonicConfigBoolTest(unittest.TestCase):
    """``_resolve_sonic_config_bool``: explicit value wins, then a
    ``resolve_<name>`` callback, else ``False``; a present-but-falsy value must
    resolve to ``False`` rather than falling through to the resolver."""

    def test_none_config_is_false(self):
        self.assertIs(flu._resolve_sonic_config_bool(None, "use_x"), False)

    def test_explicit_truthy_value(self):
        cfg = SimpleNamespace(use_x=1)
        self.assertIs(flu._resolve_sonic_config_bool(cfg, "use_x"), True)

    def test_explicit_falsy_value_does_not_fall_through(self):
        # value is present (0, not None) so bool(0) -> False; the resolver, which
        # would return True, must NOT be consulted.
        cfg = SimpleNamespace(use_x=0, resolve_use_x=lambda: True)
        self.assertIs(flu._resolve_sonic_config_bool(cfg, "use_x"), False)

    def test_resolver_used_when_value_is_none(self):
        cfg = SimpleNamespace(use_x=None, resolve_use_x=lambda: 5)
        self.assertIs(flu._resolve_sonic_config_bool(cfg, "use_x"), True)

    def test_missing_value_and_missing_resolver_is_false(self):
        cfg = SimpleNamespace()
        self.assertIs(flu._resolve_sonic_config_bool(cfg, "use_x"), False)


# ANCHOR_TESTS


@unittest.skipUnless(
    _IMPORT_ERROR is None,
    f"fusion_layer_utils import unavailable on this host: {_IMPORT_ERROR!r}",
)
class GateUpOutDimTest(unittest.TestCase):
    """``MlpNode._gate_up_out_dim`` reads the gate_up output width off the last
    weight dimension of whichever layout is populated, skipping ``None`` experts,
    and otherwise falls back to ``2 * hidden_size``."""

    def test_per_expert_layout_skips_none(self):
        # up_gate_proj.weight is [hidden, 2*inter]; the output width is the last
        # dim (6 here). The leading None expert must be skipped, not read.
        expert = SimpleNamespace(
            up_gate_proj=SimpleNamespace(
                weight=paddle.zeros([8, 6], dtype=paddle.float32)
            )
        )
        node = _bare_mlp_node(experts=[None, expert])
        self.assertEqual(node._gate_up_out_dim(8), 6)

    def test_grouped_layout_reads_stacked_weight1(self):
        # grouped deep_gemm weight1 is [num_experts, hidden, 2*inter]; last dim 6.
        parent = SimpleNamespace(
            weight1=paddle.zeros([2, 8, 6], dtype=paddle.float32)
        )
        gemm = SimpleNamespace(grouped_gemm_experts=parent)
        node = _bare_mlp_node(experts=None, experts_group_gemm_node=[gemm])
        self.assertEqual(node._gate_up_out_dim(8), 6)

    def test_fallback_when_weight_unresolved(self):
        gemm = SimpleNamespace()  # no grouped_gemm_experts attribute
        node = _bare_mlp_node(experts=None, experts_group_gemm_node=gemm)
        self.assertEqual(node._gate_up_out_dim(8), 16)


@unittest.skipUnless(
    _IMPORT_ERROR is None,
    f"fusion_layer_utils import unavailable on this host: {_IMPORT_ERROR!r}",
)
class ForwardFeatureSizesTest(unittest.TestCase):
    """``MlpNode._fwd_pre_permute_feature_sizes`` budgets four concurrent
    forward buffers, the first being the non-concurrent ``max(o1, o3)``."""

    def test_exact_byte_budget(self):
        H, G, I, tmp = 8, 12, 6, 3
        node = _bare_mlp_node()
        expected = [
            FP8_ALIGN * max(G * 2, H * 2),  # max(o1[N,2*inter], o3[N,H]) bf16
            FP8_ALIGN * H,  # permuted_input fp8
            FP8_ALIGN * I,  # o2_fp8
            FP8_ALIGN * tmp,  # unpermute tmp
        ]
        self.assertEqual(
            node._fwd_pre_permute_feature_sizes(H, G, I, tmp), expected
        )
        # o1 dominates here (2*G > 2*H); a naive H*2 would understate the peak.
        self.assertEqual(expected[0], FP8_ALIGN * G * 2)


def _expected_bwd_sizes(H, G, I, use_bf16, clamp_active, activation_type):
    """Independent restatement of the documented backward buffer budget."""
    used_inplace = (
        flu.USE_INPLACE_SWIGLU_BWD
        and not clamp_active
        and activation_type != "situ"
    )
    if use_bf16:
        if used_inplace:
            return [
                FP8_ALIGN * H * 2,  # out_grad
                FP8_ALIGN * G * 2,  # o1/do1 (inplace)
                FP8_ALIGN * I * 2,  # o2_s
                FP8_ALIGN * H,  # permuted_input fp8
                FP8_ALIGN * H * 2,  # dw1 dequant x (peak)
            ]
        return [
            FP8_ALIGN * H * 2,
            FP8_ALIGN * G * 2,  # o1
            FP8_ALIGN * G * 2,  # do1 (separate)
            FP8_ALIGN * I * 2,
            FP8_ALIGN * H,
            FP8_ALIGN * H * 2,
        ]
    if used_inplace:
        return [
            FP8_ALIGN * H * 2,
            FP8_ALIGN * G * 2,  # o1/do1 (inplace)
            FP8_ALIGN * I * 2,
            FP8_ALIGN * H,  # input_x_t_fp8
            FP8_ALIGN * G,  # do1_t_fp8
        ]
    return [
        FP8_ALIGN * H * 2,
        FP8_ALIGN * G * 2,  # o1
        FP8_ALIGN * G * 2,  # do1 (separate)
        FP8_ALIGN * I * 2,
        FP8_ALIGN * H,
        FP8_ALIGN * G,
    ]


@unittest.skipUnless(
    _IMPORT_ERROR is None,
    f"fusion_layer_utils import unavailable on this host: {_IMPORT_ERROR!r}",
)
class BackwardFeatureSizesTest(unittest.TestCase):
    """``MlpNode._bwd_pre_permute_feature_sizes``: the in-place vs out-of-place
    peak depends on the ``USE_INPLACE_SWIGLU_BWD`` build flag, but ``clamp>0``
    and ``activation_type == 'situ'`` force the out-of-place peak regardless."""

    H, G, I = 8, 12, 6

    def test_bf16_default_matches_flag(self):
        gemm = SimpleNamespace(use_bf16_gemm_weight_grad=True, clamp_value=None)
        node = _bare_mlp_node(experts_group_gemm_node=[gemm])
        got = node._bwd_pre_permute_feature_sizes(self.H, self.G, self.I)
        self.assertEqual(
            got,
            _expected_bwd_sizes(self.H, self.G, self.I, True, False, "swiglu"),
        )

    def test_fp8_default_matches_flag(self):
        gemm = SimpleNamespace(
            use_bf16_gemm_weight_grad=False, clamp_value=None
        )
        node = _bare_mlp_node(experts_group_gemm_node=gemm)
        got = node._bwd_pre_permute_feature_sizes(self.H, self.G, self.I)
        self.assertEqual(
            got,
            _expected_bwd_sizes(self.H, self.G, self.I, False, False, "swiglu"),
        )

    def test_clamp_forces_out_of_place(self):
        # clamp_value>0 => separate do1 buffer => 6 entries, independent of flag.
        gemm = SimpleNamespace(use_bf16_gemm_weight_grad=True, clamp_value=1.0)
        node = _bare_mlp_node(experts_group_gemm_node=gemm)
        got = node._bwd_pre_permute_feature_sizes(self.H, self.G, self.I)
        expected = _expected_bwd_sizes(
            self.H, self.G, self.I, True, True, "swiglu"
        )
        self.assertEqual(got, expected)
        self.assertEqual(len(got), 6)

    def test_situ_forces_out_of_place(self):
        gemm = SimpleNamespace(
            use_bf16_gemm_weight_grad=False, clamp_value=None
        )
        node = _bare_mlp_node(
            experts_group_gemm_node=gemm, activation_type="situ"
        )
        got = node._bwd_pre_permute_feature_sizes(self.H, self.G, self.I)
        expected = _expected_bwd_sizes(
            self.H, self.G, self.I, False, True, "situ"
        )
        self.assertEqual(got, expected)
        self.assertEqual(len(got), 6)

    def test_out_of_place_peak_is_larger_than_inplace(self):
        # The separate do1 buffer must make the clamp (out-of-place) footprint
        # strictly exceed the no-clamp footprint under the same wgrad dtype.
        base = SimpleNamespace(use_bf16_gemm_weight_grad=True, clamp_value=None)
        clamped = SimpleNamespace(
            use_bf16_gemm_weight_grad=True, clamp_value=2.0
        )
        base_sizes = _bare_mlp_node(
            experts_group_gemm_node=base
        )._bwd_pre_permute_feature_sizes(self.H, self.G, self.I)
        clamp_sizes = _bare_mlp_node(
            experts_group_gemm_node=clamped
        )._bwd_pre_permute_feature_sizes(self.H, self.G, self.I)
        self.assertGreater(sum(clamp_sizes), sum(base_sizes))


# ANCHOR_SHAPE_TESTS


@unittest.skipUnless(
    _IMPORT_ERROR is None,
    f"fusion_layer_utils import unavailable on this host: {_IMPORT_ERROR!r}",
)
class PadFrontRowsTest(unittest.TestCase):
    """``_pad_front_rows`` keeps the real rows in the leading positions and
    zero-fills the trailing rows; an already-matching shape is returned as-is."""

    def test_identity_when_shape_matches(self):
        t = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]])
        out = flu._pad_front_rows(t, [2, 2])
        self.assertIs(out, t)  # no reallocation on the fast path

    def test_leading_rows_preserved_tail_zeroed(self):
        t = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]])
        out = flu._pad_front_rows(t, [4, 2])
        self.assertEqual(list(out.shape), [4, 2])
        self.assertEqual(
            out.numpy().tolist(),
            [[1.0, 2.0], [3.0, 4.0], [0.0, 0.0], [0.0, 0.0]],
        )

    def test_one_dimensional_padding(self):
        t = paddle.to_tensor([5.0, 6.0, 7.0])
        out = flu._pad_front_rows(t, [5])
        self.assertEqual(out.numpy().tolist(), [5.0, 6.0, 7.0, 0.0, 0.0])


@unittest.skipUnless(
    _IMPORT_ERROR is None,
    f"fusion_layer_utils import unavailable on this host: {_IMPORT_ERROR!r}",
)
class RestoreHybridEpProbGradShapeTest(unittest.TestCase):
    """``_restore_hybrid_ep_prob_grad_shape`` squeezes a trailing unit axis then
    pads leading rows back to the original 1D contract width."""

    def test_squeezes_trailing_unit_axis_then_pads(self):
        grad = paddle.to_tensor([[1.0], [2.0], [3.0]])  # [3, 1]
        out = flu._restore_hybrid_ep_prob_grad_shape(grad, (5,))
        self.assertEqual(list(out.shape), [5])
        self.assertEqual(out.numpy().tolist(), [1.0, 2.0, 3.0, 0.0, 0.0])

    def test_already_1d_input_only_pads(self):
        grad = paddle.to_tensor([1.0, 2.0])
        out = flu._restore_hybrid_ep_prob_grad_shape(grad, (4,))
        self.assertEqual(out.numpy().tolist(), [1.0, 2.0, 0.0, 0.0])

    def test_rejects_non_1d_original_shape(self):
        grad = paddle.to_tensor([1.0, 2.0])
        with self.assertRaises(AssertionError):
            flu._restore_hybrid_ep_prob_grad_shape(grad, (2, 2))


@unittest.skipUnless(
    _IMPORT_ERROR is None,
    f"fusion_layer_utils import unavailable on this host: {_IMPORT_ERROR!r}",
)
class HybridEpPrepareExpertCountsTest(unittest.TestCase):
    """``_hybrid_ep_prepare_expert_counts`` returns a Python int64 list except on
    the fp8 + fusion path, where the tensor itself is returned."""

    @staticmethod
    def _custom_map(counts, num_permuted):
        manager = SimpleNamespace(
            padded_tokens_per_expert=paddle.to_tensor(
                counts, dtype=paddle.int32
            ),
            num_permuted_tokens=num_permuted,
        )
        dispatcher = SimpleNamespace(_comm_manager=manager)
        return SimpleNamespace(token_dispatcher=dispatcher)

    def test_returns_list_when_not_fp8_fusion(self):
        cmap = self._custom_map([2, 3, 4], 9)
        counts, num_permuted = flu._hybrid_ep_prepare_expert_counts(
            cmap, use_fp8_mlp=False, moe_expert_fusion=True
        )
        self.assertIsInstance(counts, list)
        self.assertEqual(counts, [2, 3, 4])
        self.assertEqual(num_permuted, 9)

    def test_returns_list_when_fp8_but_no_fusion(self):
        cmap = self._custom_map([1, 5], 6)
        counts, _ = flu._hybrid_ep_prepare_expert_counts(
            cmap, use_fp8_mlp=True, moe_expert_fusion=False
        )
        self.assertIsInstance(counts, list)
        self.assertEqual(counts, [1, 5])

    def test_returns_tensor_on_fp8_fusion_path(self):
        cmap = self._custom_map([2, 3, 4], 9)
        counts, num_permuted = flu._hybrid_ep_prepare_expert_counts(
            cmap, use_fp8_mlp=True, moe_expert_fusion=True
        )
        self.assertIsInstance(counts, paddle.Tensor)
        self.assertEqual(counts.dtype, paddle.int64)
        self.assertEqual(counts.numpy().tolist(), [2, 3, 4])
        self.assertEqual(num_permuted, 9)


@unittest.skipUnless(
    _IMPORT_ERROR is None,
    f"fusion_layer_utils import unavailable on this host: {_IMPORT_ERROR!r}",
)
class NodeCacheContractTest(unittest.TestCase):
    """``UnZipNode`` caches exactly ``[unzipped_probs, zipped_expertwise_rowmap]``
    in that order; ``ZipNode`` holds no cache."""

    def test_unzip_cache_roundtrip_preserves_order(self):
        node = UnZipNode(token_dispatcher=object())
        probs = paddle.to_tensor([1.0, 2.0])
        rowmap = paddle.to_tensor([3.0, 4.0])
        node.set_cached_tensors([probs, rowmap])
        # Order matters: a swapped assignment would land rowmap into probs.
        self.assertIs(node.unzipped_probs, probs)
        self.assertIs(node.zipped_expertwise_rowmap, rowmap)
        cached = node.cached_tensors()
        self.assertIs(cached[0], probs)
        self.assertIs(cached[1], rowmap)

    def test_unzip_reset_and_clear_null_both_slots(self):
        node = UnZipNode(token_dispatcher=object())
        node.set_cached_tensors(
            [paddle.to_tensor([1.0]), paddle.to_tensor([2.0])]
        )
        node.reset_state()
        self.assertEqual(node.cached_tensors(), [None, None])
        node.set_cached_tensors(
            [paddle.to_tensor([1.0]), paddle.to_tensor([2.0])]
        )
        node.clear_cached_tensors()
        self.assertEqual(node.cached_tensors(), [None, None])

    def test_zip_has_no_cache(self):
        node = ZipNode(token_dispatcher=object())
        self.assertEqual(node.cached_tensors(), [])
        node.set_cached_tensors([])  # must accept the empty contract
        node.clear_cached_tensors()  # no-op, must not raise
        self.assertEqual(node.cached_tensors(), [])


if __name__ == "__main__":
    unittest.main()

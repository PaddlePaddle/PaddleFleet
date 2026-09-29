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
"""Behavior tests for the ``pre_permute`` auto-subbatch chunk-sizing logic.

The ``pre_permute`` MoE forward/backward split the dispatched tokens into
row-chunks whose size is chosen so that every concurrent GEMM buffer fits in
GPU free memory. Three pure, device-independent pieces drive that decision, and
this suite pins each one to a hand-derived contract that does not re-use the
production formula:

* ``MlpNode._fwd_pre_permute_feature_sizes`` -- per-``FP8_ALIGN``-row byte sizes
  of the four concurrent *forward* buffers. The first entry is the max of the
  gate-up and the o3 buffer; the test drives each side to dominate so a dropped
  term is caught.
* ``MlpNode._bwd_pre_permute_feature_sizes`` -- per-``FP8_ALIGN``-row byte sizes
  of the concurrent *backward* buffers. The clamp / SiTU inputs are held so the
  out-of-place peak is selected deterministically regardless of the compile-time
  ``USE_INPLACE_SWIGLU_BWD`` flag; the bf16-wgrad and fp8-wgrad branches produce
  distinguishable buffer lists.
* ``find_max_concurrent_subbatch_size`` -- the free-block bin-packing heuristic
  that consumes those byte sizes. The allocator query is replaced with a
  deterministic free-block layout (a genuine collaborator, not the code under
  test) and the returned subbatch size is hand-derived for the 1-tensor,
  2-tensor and >=3-tensor (binary-search) regimes, plus the ``upper`` clamp and
  the empty / no-memory edge cases.

Paddle is required to import the modules under test. When it is unavailable the
whole suite is skipped with an honest reason rather than reported as passing.
"""

import os
import sys
import types
import unittest
from unittest import mock

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)

try:
    import paddle

    from paddlefleet.transformer.moe import vmm_utils
    from paddlefleet.transformer.moe.fp8_utils import FP8_ALIGN
    from paddlefleet.transformer.moe.fusion_layer_utils import MlpNode

    _IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - environment-dependent
    paddle = None
    vmm_utils = None
    FP8_ALIGN = None
    MlpNode = None
    _IMPORT_ERROR = exc

_SKIP_REASON = (
    "paddle / paddlefleet.transformer.moe fusion modules unavailable: "
    f"{_IMPORT_ERROR}"
)


@unittest.skipUnless(MlpNode is not None, _SKIP_REASON)
class TestFwdPrePermuteFeatureSizes(unittest.TestCase):
    """Forward concurrent-buffer byte sizes, hand-derived from the contract.

    ``_fwd_pre_permute_feature_sizes`` does not read ``self``; it returns, per
    ``FP8_ALIGN`` unzipped rows::

        [ FP8_ALIGN * max(2*gate_up_out_dim, 2*hidden_size),  # o1 vs o3 (reused)
          FP8_ALIGN * hidden_size,                            # permuted_input fp8
          FP8_ALIGN * inter_dim,                              # o2 fp8
          FP8_ALIGN * unpermute_tmp_per_N ]                   # unpermute scratch
    """

    def _call(self, hidden_size, gate_up_out_dim, inter_dim, unpermute_tmp):
        dummy_self = types.SimpleNamespace()
        return MlpNode._fwd_pre_permute_feature_sizes(
            dummy_self,
            hidden_size,
            gate_up_out_dim,
            inter_dim,
            unpermute_tmp,
        )

    def test_gate_up_buffer_dominates_first_entry(self):
        # 2*gate_up_out_dim (400) > 2*hidden_size (128) -> gate-up wins slot 0.
        out = self._call(
            hidden_size=64,
            gate_up_out_dim=200,
            inter_dim=100,
            unpermute_tmp=10,
        )
        self.assertEqual(
            out,
            [
                FP8_ALIGN * 400,  # max(2*200, 2*64) = 400
                FP8_ALIGN * 64,
                FP8_ALIGN * 100,
                FP8_ALIGN * 10,
            ],
        )

    def test_o3_buffer_dominates_first_entry(self):
        # 2*hidden_size (600) > 2*gate_up_out_dim (200) -> o3 wins slot 0.
        # Guards against dropping the hidden_size term from the max().
        out = self._call(
            hidden_size=300,
            gate_up_out_dim=100,
            inter_dim=50,
            unpermute_tmp=7,
        )
        self.assertEqual(
            out,
            [
                FP8_ALIGN * 600,  # max(2*100, 2*300) = 600
                FP8_ALIGN * 300,
                FP8_ALIGN * 50,
                FP8_ALIGN * 7,
            ],
        )


@unittest.skipUnless(MlpNode is not None, _SKIP_REASON)
class TestBwdPrePermuteFeatureSizes(unittest.TestCase):
    """Backward concurrent-buffer byte sizes for the out-of-place peak.

    Setting ``clamp_value > 0`` forces the out-of-place activation-backward path
    (``fused_swiglu_weighted_clamp_bwd``) regardless of ``USE_INPLACE_SWIGLU_BWD``
    and of ``activation_type``, so the expected buffer list is fully determined
    by ``use_bf16_gemm_weight_grad`` alone.
    """

    H, GU, INTER = 64, 200, 100

    def _make_self(self, use_bf16_wgrad, clamp_value, activation_type="swiglu"):
        gemm_node = types.SimpleNamespace(
            use_bf16_gemm_weight_grad=use_bf16_wgrad,
            clamp_value=clamp_value,
        )
        return types.SimpleNamespace(
            experts_group_gemm_node=gemm_node,
            activation_type=activation_type,
        )

    def _bf16_out_of_place_expected(self):
        return [
            FP8_ALIGN * self.H * 2,  # out_grad [N,H] bf16
            FP8_ALIGN * self.GU * 2,  # o1 [N,2*inter] bf16
            FP8_ALIGN * self.GU * 2,  # do1 [N,2*inter] bf16 (separate buffer)
            FP8_ALIGN * self.INTER * 2,  # o2_s [N,inter] bf16
            FP8_ALIGN * self.H,  # permuted_input [N,H] fp8
            FP8_ALIGN * self.H * 2,  # dw1 dequant x [N,H] bf16 (the peak)
        ]

    def _fp8_out_of_place_expected(self):
        return [
            FP8_ALIGN * self.H * 2,  # out_grad [N,H] bf16
            FP8_ALIGN * self.GU * 2,  # o1 [N,2*inter] bf16
            FP8_ALIGN * self.GU * 2,  # do1 [N,2*inter] bf16 (separate buffer)
            FP8_ALIGN * self.INTER * 2,  # o2_s [N,inter] bf16
            FP8_ALIGN * self.H,  # input_x_t_fp8 [N,H] fp8
            FP8_ALIGN * self.GU,  # do1_t_fp8 [N,2*inter] fp8
        ]

    def test_bf16_wgrad_out_of_place_peak(self):
        node_self = self._make_self(use_bf16_wgrad=True, clamp_value=1.0)
        out = MlpNode._bwd_pre_permute_feature_sizes(
            node_self, self.H, self.GU, self.INTER
        )
        self.assertEqual(out, self._bf16_out_of_place_expected())

    def test_fp8_wgrad_out_of_place_peak(self):
        node_self = self._make_self(use_bf16_wgrad=False, clamp_value=1.0)
        out = MlpNode._bwd_pre_permute_feature_sizes(
            node_self, self.H, self.GU, self.INTER
        )
        expected = self._fp8_out_of_place_expected()
        self.assertEqual(out, expected)
        # The two wgrad paths must differ in the trailing buffer: bf16 keeps a
        # dequant activation (2*H) while fp8 keeps a transposed fp8 do1 (GU).
        self.assertNotEqual(expected, self._bf16_out_of_place_expected())

    def test_situ_forces_out_of_place_without_clamp(self):
        # activation_type == "situ" also disqualifies the inplace path, so with
        # no clamp the bf16 out-of-place list is still produced.
        node_self = self._make_self(
            use_bf16_wgrad=True, clamp_value=None, activation_type="situ"
        )
        out = MlpNode._bwd_pre_permute_feature_sizes(
            node_self, self.H, self.GU, self.INTER
        )
        self.assertEqual(out, self._bf16_out_of_place_expected())

    def test_gemm_node_list_is_unwrapped_to_first_element(self):
        # experts_group_gemm_node may be a per-expert list; the sizing reads the
        # first node's wgrad flag. A list of fp8-wgrad nodes must match fp8.
        gemm_node = types.SimpleNamespace(
            use_bf16_gemm_weight_grad=False, clamp_value=1.0
        )
        node_self = types.SimpleNamespace(
            experts_group_gemm_node=[gemm_node, gemm_node],
            activation_type="swiglu",
        )
        out = MlpNode._bwd_pre_permute_feature_sizes(
            node_self, self.H, self.GU, self.INTER
        )
        self.assertEqual(out, self._fp8_out_of_place_expected())


@unittest.skipUnless(vmm_utils is not None, _SKIP_REASON)
class TestFindMaxConcurrentSubbatchSize(unittest.TestCase):
    """Free-block bin-packing heuristic that consumes the feature sizes.

    ``allocator_free_block_info`` (a genuine collaborator that queries the VMM
    allocator) is replaced with a fixed free-block layout so the returned
    subbatch size is deterministic. Blocks are reported smallest-first, matching
    the real allocator's documented ordering. Every expected value below is
    traced by hand from the packing rules, not from the implementation.
    """

    def _patch_blocks(self, blocks):
        return mock.patch.object(
            vmm_utils, "allocator_free_block_info", return_value=list(blocks)
        )

    def test_single_tensor_uses_largest_block(self):
        # One buffer of 100 bytes/row, largest free block is 1000 -> 10 rows.
        with self._patch_blocks([(300, 0), (1000, 1)]):
            self.assertEqual(
                vmm_utils.find_max_concurrent_subbatch_size([100]), 10
            )

    def test_two_tensors_split_across_two_blocks_beats_single_block(self):
        # sizes [100, 50]; blocks smallest-first [(500,), (800,)].
        #   both in largest block: 800 // 150 = 5
        #   split: min(800//100=8, 500//50=10) = 8  -> winner
        with self._patch_blocks([(500, 0), (800, 1)]):
            self.assertEqual(
                vmm_utils.find_max_concurrent_subbatch_size([100, 50]), 8
            )

    def test_two_tensors_single_block_falls_back_to_shared(self):
        # Only one block: both buffers must share it -> 800 // 150 = 5.
        with self._patch_blocks([(800, 0)]):
            self.assertEqual(
                vmm_utils.find_max_concurrent_subbatch_size([100, 50]), 5
            )

    def test_three_tensors_binary_search_greedy_pack(self):
        # sizes [100,100,100]; blocks [(600,), (600,)].
        # size=3 -> 300/row: block A packs 2 (600), block B packs 1 -> fits.
        # size=4 -> 400/row: each 600 block packs only 1 -> 2 placed, fails.
        with self._patch_blocks([(600, 0), (600, 1)]):
            self.assertEqual(
                vmm_utils.find_max_concurrent_subbatch_size([100, 100, 100]),
                3,
            )

    def test_three_tensors_tight_blocks(self):
        # blocks [(250,), (250,)]; size=1 packs 2+1, size=2 (200/row) fails.
        with self._patch_blocks([(250, 0), (250, 1)]):
            self.assertEqual(
                vmm_utils.find_max_concurrent_subbatch_size([100, 100, 100]),
                1,
            )

    def test_upper_bound_clamps_binary_search(self):
        # Big blocks would allow 3, but upper=2 caps the search at 2.
        with self._patch_blocks([(600, 0), (600, 1)]):
            self.assertEqual(
                vmm_utils.find_max_concurrent_subbatch_size(
                    [100, 100, 100], upper=2
                ),
                2,
            )

    def test_zero_feature_size_returns_one(self):
        # A degenerate all-zero request needs no memory -> one row is trivially
        # allocatable; the allocator is not even consulted.
        with self._patch_blocks([(999, 0)]):
            self.assertEqual(
                vmm_utils.find_max_concurrent_subbatch_size([0, 0]), 1
            )

    def test_empty_feature_list_returns_one(self):
        with self._patch_blocks([(999, 0)]):
            self.assertEqual(vmm_utils.find_max_concurrent_subbatch_size([]), 1)

    def test_no_free_memory_returns_zero(self):
        # A real, non-zero request with no free blocks cannot be satisfied.
        with self._patch_blocks([]):
            self.assertEqual(
                vmm_utils.find_max_concurrent_subbatch_size([100]), 0
            )


if __name__ == "__main__":
    unittest.main()

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

"""`routing_map_fusion_forward` runs `_routing_map_fwd_bitmap_kernel`, which is a
different algorithm from the `_routing_map_fwd_kernel` it replaced (bit packing
plus OR, versus a [BLOCK_M, BLOCK_K, BLOCK_N] broadcast reduced with max). Its
bit-exactness is therefore empirical, not structural, and this file is where it
is established: every case runs both kernels and compares all three outputs
bit-for-bit.
"""

import unittest

import paddle
import triton

from paddlefleet.triton_ops.moe_topk_fusion import (
    _routing_map_fwd_kernel,
    routing_map_fusion_forward,
)


def _reference_forward(
    gate_probs,
    topk_indices,
    input_ids=None,
    is_pure_text_line=None,
    pad_token_id=0,
):
    """The superseded `_routing_map_fwd_kernel` with its original geometry."""
    seq_len, moe_k = topk_indices.shape
    n_experts = gate_probs.shape[1]

    routing_map = paddle.zeros((seq_len, n_experts), dtype=paddle.float32)
    topk_indices_out = paddle.empty_like(topk_indices)
    dispatch_mask = paddle.zeros((n_experts,), dtype=paddle.int64)

    BLOCK_M, BLOCK_N = 64, 128
    BLOCK_K = triton.next_power_of_2(moe_k)
    grid = (triton.cdiv(seq_len, BLOCK_M), triton.cdiv(n_experts, BLOCK_N))
    _routing_map_fwd_kernel[grid](
        topk_indices_ptr=topk_indices,
        input_ids_ptr=input_ids if input_ids is not None else topk_indices,
        is_pure_text_line_ptr=is_pure_text_line
        if is_pure_text_line is not None
        else topk_indices,
        routing_map_ptr=routing_map,
        topk_indices_out_ptr=topk_indices_out,
        dispatch_mask_ptr=dispatch_mask,
        stride_topk_s=int(topk_indices.stride(0)),
        stride_topk_k=int(topk_indices.stride(1)),
        stride_routing_s=int(routing_map.stride(0)),
        stride_routing_e=int(routing_map.stride(1)),
        n_experts=n_experts,
        seq_len=seq_len,
        moe_k=moe_k,
        pad_token_id=pad_token_id,
        has_input_ids=input_ids is not None,
        has_pure_text_mask=is_pure_text_line is not None,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return routing_map, topk_indices_out, dispatch_mask


# (seq_len, n_experts, moe_k). Deliberately includes sizes that are not
# multiples of either kernel's BLOCK_M / BLOCK_N so the overhang masks are
# exercised on both sides.
SHAPES = [
    (8192, 512, 10),  # the real ernielite_layer43_mla_hca shape
    (64, 32, 1),  # exactly one tile of the bitmap kernel
    (65, 33, 3),  # one row and one expert past a tile boundary
    (127, 130, 8),  # overhang on both dims, n_experts past 128
    (1, 512, 10),  # single row
    (255, 96, 16),  # moe_k at a power of two
    (300, 64, 5),
]


def _make_inputs(seq_len, n_experts, moe_k, seed):
    paddle.seed(seed)
    gate_probs = paddle.rand([seq_len, n_experts], dtype="float32")
    topk_indices = paddle.randint(0, n_experts, [seq_len, moe_k], dtype="int64")
    return gate_probs, topk_indices


@unittest.skipIf(
    not paddle.is_compiled_with_cuda(),
    "the routing-map kernels are Triton GPU kernels",
)
class TestRoutingMapBitmap(unittest.TestCase):
    def _assert_agrees(self, msg, *args, **kwargs):
        expected = _reference_forward(*args, **kwargs)
        actual = routing_map_fusion_forward(*args, **kwargs)
        names = ("routing_map", "topk_indices_out", "dispatch_mask")
        for name, exp, act in zip(names, expected, actual, strict=True):
            self.assertEqual(act.dtype, exp.dtype, f"{msg}: {name} dtype")
            self.assertEqual(
                list(act.shape), list(exp.shape), f"{msg}: {name} shape"
            )
            self.assertTrue(
                bool(paddle.equal_all(act, exp)),
                f"{msg}: {name} differs from the reference kernel",
            )

    def test_matches_reference_across_shapes(self):
        for seq_len, n_experts, moe_k in SHAPES:
            with self.subTest(seq=seq_len, experts=n_experts, k=moe_k):
                gate_probs, topk_indices = _make_inputs(
                    seq_len, n_experts, moe_k, seed=20260903
                )
                self._assert_agrees(
                    f"{seq_len}x{n_experts}x{moe_k}", gate_probs, topk_indices
                )

    def test_matches_reference_with_duplicated_expert_ids(self):
        # The only place the two algorithms could diverge: the reference reduces
        # `matches` with max, the bitmap kernel ORs the packed bits. Both are
        # "any", but only if a repeated id in one row is handled the same way.
        seq_len, n_experts, moe_k = 200, 96, 8
        gate_probs = paddle.rand([seq_len, n_experts], dtype="float32")
        paddle.seed(11)
        topk_indices = paddle.randint(0, 4, [seq_len, moe_k], dtype="int64")
        self._assert_agrees("duplicated ids", gate_probs, topk_indices)

        # Every row picks the same expert moe_k times.
        topk_indices = paddle.full([seq_len, moe_k], 7, dtype="int64")
        self._assert_agrees("all-identical ids", gate_probs, topk_indices)

    def test_matches_reference_with_negative_sentinel_indices(self):
        # Upstream masking can leave -1 in topk_indices. `rel = -1 - base` must
        # be rejected by the `rel >= 0` guard rather than shifting by a negative
        # amount.
        seq_len, n_experts, moe_k = 128, 64, 6
        gate_probs = paddle.rand([seq_len, n_experts], dtype="float32")
        paddle.seed(12)
        topk_indices = paddle.randint(
            0, n_experts, [seq_len, moe_k], dtype="int64"
        )
        topk_indices[::3, 0] = -1
        topk_indices[1::5, :] = -1
        self._assert_agrees("sentinel -1", gate_probs, topk_indices)

    def test_matches_reference_when_the_words_top_bit_is_set(self):
        # `bits` is int32, so a row that hits relative index 31 sets the sign
        # bit and `bits >> lane` is an arithmetic shift. Safe, because the
        # expansion masks with `& 1` and sign extension only fills the bits
        # shifted in at the top -- but nothing else in this file forces
        # relative index 31 to be hit, so it is pinned here.
        cases = {
            # n_experts == 32: one tile, so relative index 31 is expert 31 and
            # `bits` is INT_MIN when that is the only pick.
            "only expert 31": (128, 32, [[31]] * 128),
            "experts 31 and 0": (128, 32, [[31, 0]] * 128),
            "experts 31 and 30": (128, 32, [[31, 30]] * 128),
            # One row per bit position, so bit 31 is compared against 0..30 in
            # the same launch.
            "one row per expert": (32, 32, [[e] for e in range(32)]),
            # Relative index 31 in a tile other than the first.
            "expert 63": (64, 64, [[63]] * 64),
            "experts 31 and 63": (64, 64, [[31, 63]] * 64),
            # A tile whose lane 31 is the last fully populated expert.
            "n_experts=33": (64, 33, [[31, 32]] * 64),
            # `bits == -1`: every bit of the word set.
            "all 32 bits": (64, 32, [list(range(32))] * 64),
        }
        for name, (seq_len, n_experts, rows) in cases.items():
            with self.subTest(case=name):
                topk_indices = paddle.to_tensor(rows, dtype="int64")
                gate_probs = paddle.rand([seq_len, n_experts], dtype="float32")
                self._assert_agrees(name, gate_probs, topk_indices)

                # Also against a one-hot scatter, so this does not rest on the
                # reference kernel being right about the top bit either.
                experts_axis = paddle.arange(n_experts, dtype="int64")
                expected = (
                    (topk_indices.unsqueeze(-1) == experts_axis)
                    .any(axis=1)
                    .astype("float32")
                )
                routing_map, _, dispatch_mask = routing_map_fusion_forward(
                    gate_probs, topk_indices
                )
                self.assertTrue(
                    bool(paddle.equal_all(routing_map, expected)),
                    f"{name}: routing_map differs from a one-hot scatter",
                )
                self.assertTrue(
                    bool(
                        paddle.equal_all(
                            dispatch_mask,
                            expected.sum(axis=0).astype("int64"),
                        )
                    ),
                    f"{name}: dispatch_mask is not the scatter's column sum",
                )

    def test_matches_reference_with_masks(self):
        seq_len, n_experts, moe_k = 300, 130, 10
        gate_probs, topk_indices = _make_inputs(
            seq_len, n_experts, moe_k, seed=13
        )
        input_ids = paddle.randint(0, 3, [seq_len], dtype="int64")
        is_pure_text_line = paddle.randint(0, 2, [seq_len], dtype="int64")

        for name, kwargs in (
            ("input_ids", {"input_ids": input_ids}),
            ("pure_text", {"is_pure_text_line": is_pure_text_line}),
            (
                "both",
                {
                    "input_ids": input_ids,
                    "is_pure_text_line": is_pure_text_line,
                },
            ),
        ):
            with self.subTest(masks=name):
                self._assert_agrees(
                    name,
                    gate_probs,
                    topk_indices,
                    pad_token_id=0,
                    **kwargs,
                )

    def test_routing_map_is_zero_or_one(self):
        # The bitmap expansion returns `(bits >> lane) & 1`; anything else would
        # mean the packing overflowed into a neighbouring expert's bit.
        gate_probs, topk_indices = _make_inputs(1024, 512, 10, seed=14)
        routing_map, _, dispatch_mask = routing_map_fusion_forward(
            gate_probs, topk_indices
        )
        self.assertTrue(
            bool(((routing_map == 0) | (routing_map == 1)).all()),
            "routing_map holds values other than 0.0 / 1.0",
        )
        self.assertTrue(
            bool(
                paddle.equal_all(
                    routing_map.sum(axis=0).astype("int64"), dispatch_mask
                )
            ),
            "dispatch_mask is not the column sum of routing_map",
        )


if __name__ == "__main__":
    unittest.main()

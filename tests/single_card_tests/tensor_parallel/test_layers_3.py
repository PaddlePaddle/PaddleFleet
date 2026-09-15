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
"""Behavior tests for helpers in ``tensor_parallel/layers`` (part 3).

Module under test: ``paddlefleet.tensor_parallel.layers`` (分布式训练 / TP).
This file deliberately targets symbols *not* exercised by the other layers
tests in this directory (which cover ``_HFEmbeddingGather``,
``_index_put_columns``, the grouped dgrad/wgrad in the async linear, and the
``VocabParallelEmbedding`` HF branch):

* the TP-attribute tagging helpers ``set_tensor_model_parallel_attributes``,
  ``set_defaults_if_not_set_tensor_model_parallel_attributes`` and
  ``copy_tensor_model_parallel_attributes`` -- who owns which
  ``partition_dim`` / ``partition_stride`` / ``tensor_model_parallel`` flag;
* ``param_is_not_tensor_parallel_duplicate`` -- the "is this param a rank-0
  duplicate" predicate that gates which ranks write a param to checkpoint;
* the bf16 (``fp8=False``) forward path of ``general_gemm``, including the
  ``use_accuracy_compatible`` transpose branch and bias handling.

These are all CPU-executable pure/dynamic-graph paths. Real multi-rank
gather/scatter/reduce semantics are out of scope here and belong to the
multi-card suite.
"""

import os
import sys
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
    import numpy as np
    import paddle

    from paddlefleet.tensor_parallel import layers as tp_layers
    from paddlefleet.tensor_parallel.layers import (
        copy_tensor_model_parallel_attributes,
        general_gemm,
        param_is_not_tensor_parallel_duplicate,
        set_defaults_if_not_set_tensor_model_parallel_attributes,
        set_tensor_model_parallel_attributes,
    )

    HAS_PADDLE = True
except ImportError:
    HAS_PADDLE = False

SKIP_REASON = "paddle (and paddlefleet) is not importable in this environment"

_TP_ATTRS = ("tensor_model_parallel", "partition_dim", "partition_stride")


def _fresh_param():
    """A brand-new parameter with none of the TP attributes set yet."""
    return paddle.create_parameter(shape=[2, 3], dtype="float32")


@unittest.skipUnless(HAS_PADDLE, SKIP_REASON)
class TestTensorModelParallelAttributes(unittest.TestCase):
    """The three tagging helpers own distinct read/write contracts."""

    def test_set_writes_all_three_flags(self):
        p = _fresh_param()
        set_tensor_model_parallel_attributes(
            p, is_parallel=True, dim=2, stride=4
        )
        self.assertIs(p.tensor_model_parallel, True)
        self.assertEqual(p.partition_dim, 2)
        self.assertEqual(p.partition_stride, 4)

    def test_set_refuses_to_overwrite_existing_flags(self):
        # The helper asserts the attributes are unset; a second call on the
        # same tensor must raise rather than silently clobber ownership.
        p = _fresh_param()
        set_tensor_model_parallel_attributes(
            p, is_parallel=True, dim=0, stride=1
        )
        with self.assertRaises(AssertionError):
            set_tensor_model_parallel_attributes(
                p, is_parallel=False, dim=1, stride=2
            )

    def test_defaults_fill_only_missing_flags(self):
        p = _fresh_param()
        set_defaults_if_not_set_tensor_model_parallel_attributes(p)
        self.assertIs(p.tensor_model_parallel, False)
        self.assertEqual(p.partition_dim, -1)
        self.assertEqual(p.partition_stride, 1)

    def test_defaults_leave_preset_flags_untouched(self):
        p = _fresh_param()
        # Only one flag is pre-set; defaults must not overwrite it.
        p.tensor_model_parallel = True
        set_defaults_if_not_set_tensor_model_parallel_attributes(p)
        self.assertIs(p.tensor_model_parallel, True)
        self.assertEqual(p.partition_dim, -1)
        self.assertEqual(p.partition_stride, 1)

    def test_copy_transfers_present_values(self):
        src = _fresh_param()
        set_tensor_model_parallel_attributes(
            src, is_parallel=True, dim=1, stride=3
        )
        dst = _fresh_param()
        copy_tensor_model_parallel_attributes(dst, src)
        self.assertIs(dst.tensor_model_parallel, True)
        self.assertEqual(dst.partition_dim, 1)
        self.assertEqual(dst.partition_stride, 3)

    def test_copy_skips_attributes_absent_on_source(self):
        src = _fresh_param()
        # Source carries a single flag; copy must transfer that one only and
        # not invent defaults for the others.
        src.partition_dim = 2
        dst = _fresh_param()
        copy_tensor_model_parallel_attributes(dst, src)
        self.assertEqual(dst.partition_dim, 2)
        self.assertFalse(hasattr(dst, "tensor_model_parallel"))
        self.assertFalse(hasattr(dst, "partition_stride"))


@unittest.skipUnless(HAS_PADDLE, SKIP_REASON)
class TestParamIsNotTensorParallelDuplicate(unittest.TestCase):
    """Predicate is ``tagged-parallel OR rank-0``; both branches matter."""

    def _check(self, tagged, rank):
        p = _fresh_param()
        if tagged is not None:
            p.tensor_model_parallel = tagged
        with mock.patch.object(
            tp_layers, "get_tensor_model_parallel_rank", return_value=rank
        ):
            return param_is_not_tensor_parallel_duplicate(p)

    def test_tp_tagged_param_is_never_a_duplicate(self):
        # Tagged parallel -> owned by this rank regardless of rank id.
        self.assertTrue(self._check(tagged=True, rank=3))

    def test_untagged_on_nonzero_rank_is_a_duplicate(self):
        # Not parallel and not rank 0 -> a duplicate, so predicate is False.
        self.assertFalse(self._check(tagged=False, rank=3))
        self.assertFalse(self._check(tagged=None, rank=3))

    def test_untagged_on_rank0_is_the_owning_copy(self):
        self.assertTrue(self._check(tagged=None, rank=0))

    def test_flag_false_still_owned_on_rank0(self):
        # rank 0 branch wins even when the parallel flag is explicitly False.
        self.assertTrue(self._check(tagged=False, rank=0))


@unittest.skipUnless(HAS_PADDLE, SKIP_REASON)
class TestGeneralGemmBf16(unittest.TestCase):
    """``fp8=False`` path: both branches compute ``a @ b`` (+ bias)."""

    def setUp(self):
        # Non-square operands so a stray transpose would change the result.
        self.a = paddle.to_tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="float32"
        )
        self.b = paddle.to_tensor(
            [
                [1.0, 0.0, -1.0, 2.0],
                [2.0, 1.0, 0.0, 1.0],
                [0.0, 3.0, 1.0, -1.0],
            ],
            dtype="float32",
        )
        # Independent numpy reference, not the paddle path under test.
        self.ref = np.matmul(self.a.numpy(), self.b.numpy())

    def test_plain_matmul_and_no_fp8_cache(self):
        out, cache = general_gemm(self.a, self.b)
        np.testing.assert_allclose(out.numpy(), self.ref, rtol=1e-6, atol=1e-6)
        self.assertIsNone(cache)

    def test_accuracy_compatible_matches_plain(self):
        # The transpose-based branch must be mathematically identical.
        out, _ = general_gemm(self.a, self.b, use_accuracy_compatible=True)
        np.testing.assert_allclose(out.numpy(), self.ref, rtol=1e-6, atol=1e-6)

    def test_bias_is_added_after_the_product(self):
        bias = paddle.to_tensor([10.0, 20.0, 30.0, 40.0], dtype="float32")
        out, _ = general_gemm(self.a, self.b, bias=bias)
        np.testing.assert_allclose(
            out.numpy(), self.ref + bias.numpy(), rtol=1e-6, atol=1e-6
        )

    def test_bias_branch_agrees_across_accuracy_modes(self):
        bias = paddle.to_tensor([10.0, 20.0, 30.0, 40.0], dtype="float32")
        expected = self.ref + bias.numpy()
        plain, _ = general_gemm(self.a, self.b, bias=bias)
        compat, _ = general_gemm(
            self.a, self.b, bias=bias, use_accuracy_compatible=True
        )
        np.testing.assert_allclose(
            plain.numpy(), expected, rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            compat.numpy(), expected, rtol=1e-6, atol=1e-6
        )


if __name__ == "__main__":
    unittest.main()

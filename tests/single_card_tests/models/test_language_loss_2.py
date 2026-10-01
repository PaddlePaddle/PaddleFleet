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
"""Model-layer (loss) unit tests for helpers in
``paddlefleet.models.common.language_loss.language_loss``.

Slice rationale: this file targets module-level helpers that are NOT
numerically exercised elsewhere -- the ``subbatch`` chunk-reassembly and
argument-sharing logic, the two environment-variable feature switches, and the
``_tensor_md5`` serialization contract. It deliberately avoids the areas
already covered by the sibling tests (``forward_impl`` fused/tuple routing in
test_language_loss_multimax_routing.py, the megatron per-depth label logic in
test_language_loss_megatron_labels.py, and the cu_seqlens roll in
test_language_loss_cu_seqlens_stash.py) as well as the __init__/forward/
build_schedule_node surface.

Expected values are hand-derived with NumPy; no production helper is used to
build the reference. Paddle is required to import the module, so every
Paddle-dependent case is guarded and honestly skipped when Paddle is absent
rather than reported as a pass.
"""

import hashlib
import os
import unittest
from unittest import mock

try:
    import numpy as np
    import paddle

    from paddlefleet.models.common.language_loss import (
        language_loss as ll,
    )

    HAS_PADDLE = True
    _IMPORT_ERROR = ""
except ImportError as exc:  # only mask a genuine missing-dependency, per rules
    HAS_PADDLE = False
    _IMPORT_ERROR = repr(exc)
    np = None
    ll = None

_SKIP_REASON = (
    ""
    if HAS_PADDLE
    else f"paddle/paddlefleet not importable in this environment: {_IMPORT_ERROR}"
)


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestSubbatchNumeric(unittest.TestCase):
    """``subbatch`` must slice the batched dims, apply ``f`` per chunk, and
    concatenate back along ``out_idx`` -- reconstructing the full result.

    The coverage source only checked the two non-chunking branches
    (input smaller than / equal to ``bs``) with ``assertIsNotNone``, so the
    real multi-chunk reassembly path (which is the only path that reaches
    ``paddle.cat``) is verified here with exact content.
    """

    def test_multichunk_reassembles_along_out_idx(self):
        # S=5 with bs=2 forces three chunks: [0:2], [2:4], [4:5] (partial tail).
        x_np = np.arange(2 * 5, dtype="float32").reshape([2, 5])
        y_np = (np.arange(2 * 5, dtype="float32").reshape([2, 5])) * 10.0
        x = paddle.to_tensor(x_np)
        y = paddle.to_tensor(y_np)

        def add(a, b):
            return a + b

        sb = ll.subbatch(add, arg_idx=[0, 1], axis=[1, 1], bs=2, out_idx=1)
        out = sb(x, y)
        # Independent reference: elementwise sum over the full sequence. A wrong
        # slice bound, dropped chunk, or wrong concat axis would not match.
        np.testing.assert_array_equal(out.numpy(), x_np + y_np)
        self.assertEqual(list(out.shape), [2, 5])

    def test_same_arg_idx_reuses_shared_slice(self):
        # same_arg_idx={1: 0} means arg 1 must reuse arg 0's *already sliced*
        # chunk, so ``add`` sees (slice0, slice0) and the result is 2*x --
        # arg 1's own content (y) must be ignored.
        x_np = np.arange(2 * 4, dtype="float32").reshape([2, 4])
        y_np = np.full([2, 4], 777.0, dtype="float32")
        x = paddle.to_tensor(x_np)
        y = paddle.to_tensor(y_np)

        def add(a, b):
            return a + b

        sb = ll.subbatch(
            add,
            arg_idx=[0, 1],
            axis=[1, 1],
            bs=2,
            out_idx=1,
            same_arg_idx={1: 0},
        )
        out = sb(x, y)
        np.testing.assert_array_equal(out.numpy(), x_np + x_np)

    def test_unequal_axis_width_rejected(self):
        # Two batched args with different sizes along their axes must trip the
        # "Batch sizes should be kept equal" assertion (distinct from the
        # arg_idx/axis length assertion the coverage source exercised).
        x = paddle.zeros([2, 5], dtype="float32")
        y = paddle.zeros([2, 4], dtype="float32")

        def add(a, b):
            return a + b

        sb = ll.subbatch(add, arg_idx=[0, 1], axis=[1, 1], bs=2, out_idx=1)
        with self.assertRaises(AssertionError):
            sb(x, y)

    def test_same_arg_idx_backward_reference_rejected(self):
        # same_arg_idx must reference an *earlier* positional arg; {0: 1} points
        # forward and has to be rejected by the ``i > same_arg_idx[i]`` guard.
        x = paddle.zeros([2, 4], dtype="float32")
        y = paddle.zeros([2, 4], dtype="float32")

        def add(a, b):
            return a + b

        sb = ll.subbatch(
            add,
            arg_idx=[0, 1],
            axis=[1, 1],
            bs=2,
            out_idx=1,
            same_arg_idx={0: 1},
        )
        with self.assertRaises(AssertionError):
            sb(x, y)


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestLossMd5EnvFlags(unittest.TestCase):
    """The two feature switches gate on an env var being exactly ``"1"``.
    ``mock.patch.dict`` restores os.environ automatically (no state leak).
    """

    def test_loss_md5_enabled_toggle(self):
        cases = {"1": True, "0": False, "2": False, "true": False}
        for value, expected in cases.items():
            with mock.patch.dict(os.environ, {"LOG_LOSS_MD5": value}):
                self.assertEqual(ll._loss_md5_enabled(), expected)
        # Unset -> defaults to False.
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(ll._loss_md5_enabled())

    def test_accuracy_compatible_kernel_toggle(self):
        cases = {"1": True, "0": False, "01": False, "yes": False}
        for value, expected in cases.items():
            with mock.patch.dict(
                os.environ, {"FLAGS_use_accuracy_compatible_kernel": value}
            ):
                self.assertEqual(ll._use_accuracy_compatible_kernel(), expected)
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(ll._use_accuracy_compatible_kernel())


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestTensorMd5(unittest.TestCase):
    """``_tensor_md5`` hashes the tensor after casting to the requested dtype.
    Reference bytes/hash are built independently from NumPy.
    """

    def test_tensor_md5_matches_numpy_float32(self):
        a = np.array([[-3.0, -1.0, 0.5, 2.25]], dtype="float32")
        t = paddle.to_tensor(a)
        expected = hashlib.md5(a.astype(np.float32).tobytes()).hexdigest()
        self.assertEqual(ll._tensor_md5(t), expected)

    def test_tensor_md5_dtype_changes_hash(self):
        # Same values, different serialization dtype must change the digest;
        # this proves the ``dtype`` argument is actually consumed by the cast.
        a = np.array([[1.0, 2.0, 3.0, 4.0]], dtype="float32")
        t = paddle.to_tensor(a)
        digest_f32 = ll._tensor_md5(t, dtype="float32")
        digest_f64 = ll._tensor_md5(t, dtype="float64")
        self.assertEqual(
            digest_f32, hashlib.md5(a.astype(np.float32).tobytes()).hexdigest()
        )
        self.assertEqual(
            digest_f64, hashlib.md5(a.astype(np.float64).tobytes()).hexdigest()
        )
        self.assertNotEqual(digest_f32, digest_f64)


class TestDistributedSoftmaxOp(unittest.TestCase):
    """DistributedSoftmaxOp is a tensor-model-parallel op."""

    def test_distributed_softmax_requires_multicard(self):
        # forward() performs AllGatherOp + mp_ops._mp_allreduce to build the
        # global max / global sum-of-exp across the model-parallel group, and
        # backward() likewise all-gathers the per-rank grad sums. The numeric
        # correctness of those cross-rank max/sum reductions can only be shown
        # with a real >=2-rank MP group under multi_card_tests. Faking
        # world_size and mocking the collective would only re-assert
        # single-process values (antipattern 13), so this local case is
        # honestly skipped rather than asserting method existence.
        self.skipTest(
            "DistributedSoftmaxOp.forward/backward need a real >=2-rank "
            "tensor-model-parallel group (AllGather + _mp_allreduce); verify "
            "under tests/multi_card_tests, not single-process."
        )


if __name__ == "__main__":
    unittest.main()

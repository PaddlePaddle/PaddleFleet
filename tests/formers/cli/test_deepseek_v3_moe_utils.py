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

"""Behavior tests for the DeepSeek-V3 pretrain MoE permute/zip utilities.

Module under test:
``paddlefleet.cli.train.deepseek_v3_pretrain.moe_utils``. This is MoE routing
plumbing (model layer / distributed dispatcher helpers): pure token indexing,
grouping, gather/scatter-add permutation, sub-batch row splitting, and the
argument marshaling of the four permute nodes around the fused MoE kernels.

What these tests actually verify (independent, hand-derived oracles -- not
shape-only, not set-then-readback):

* ``topk_to_permuted_indices`` groups the flattened ``[num_tokens, topk]``
  routing map by expert id. The expected permutation lists are worked out by
  hand from a fixed routing map (which flat positions carry each expert id, in
  ascending order) and ``token = prob // topk``; token identity and grouping
  order are checked, not just the length.
* ``permute_fast`` is an ``index_select`` gather along axis 0: distinguishable
  rows are reordered (and duplicated when an index repeats) to an exact
  expected tensor; ``drop_and_pad=True`` is asserted to raise.
* ``unpermute_fast`` is a probability-weighted scatter-add: expected outputs are
  computed with an independent NumPy ``add.at`` reference (accumulation into a
  shared row stays visible, untouched rows stay zero). Round-trip identity of
  permute followed by unpermute on a true permutation is also checked.
* ``get_env_device`` dispatch table and branch *priority* (cuda wins over a
  custom npu device, rocm/xpu only when nothing else matches) are verified by
  controlling the real Paddle capability probes -- the branch selection is the
  behavior under test, the probes are collaborators.
* ``merge_subbatch_cast`` identity vs. cast behavior for the single-tensor and
  single-element-list paths (same dtype returns the *same object*; a differing
  dtype returns a value-preserving cast), and the multi-element path is shown to
  route to the compiled ``paddlefleet.extensions.ops`` collaborator with the
  arguments forwarded unchanged.
* ``tokens_zip_unique_add_with_subbatch`` row-splitting math (even split,
  remainder split, empty-input zero fill) and dispatch branch are verified by
  capturing the split token groups handed to the (mocked, compiled) extension
  op and comparing them to the original row slices.
* ``PermuteNode.forward``/``backward`` and ``UnPermuteNode.forward`` are pure
  Paddle CPU paths built on the helpers above; they are driven end to end with a
  lightweight dispatcher stub and compared to independent oracles, and the
  reset-of-state side effect is observed.
* ``UnZipNode``/``ZipNode`` ``forward``/``backward`` marshal arguments into the
  fused ``paddle.nn.functional.moe_permute``/``moe_unpermute`` kernels. Those
  kernels are GPU collaborators (not the logic under test): they are mocked with
  distinguishable markers so the argument order, the tuple/non-tuple input
  branch, the stored state, and the exact returned selection are all checked.
* Node constructors and ``reset_status``/``reset_statue`` are state contracts:
  fields are seeded with sentinels first, then the production reset is run, then
  the fields are asserted cleared (never "clear-then-assert-empty").

Explicitly NOT covered here (recorded, not faked): the fused-kernel *numerics*
of ``moe_permute``/``moe_unpermute`` and ``UnPermuteNode.backward`` (FP8
dequantization + ``paddle._C_ops.put_along_axis_grad``) require a GPU build and
belong to the single-/multi-card MoE suites; no CPU stand-in is presented as
their numeric proof.

These tests run on CPU. Importing the production module pulls the
``paddlefleet`` package (and its ``deepseek_v3_pretrain`` ``__init__`` ->
``workflow``), which requires Paddle; when that dependency is absent the import
raises ``ImportError`` and every test skips (recorded, not silently passed). No
production code is modified or monkeypatched permanently by this file.
"""

import types
import unittest
from unittest import mock

import numpy as np

try:
    import paddle

    from paddlefleet.cli.train.deepseek_v3_pretrain import moe_utils as MOE

    _IMPORT_ERROR = None
except ImportError as exc:  # Paddle / paddlefleet backend not installed.
    paddle = None
    MOE = None
    _IMPORT_ERROR = exc


def _scatter_add_reference(
    permuted, token_idx, num_tokens, hidden, prob_flat=None, prob_idx=None
):
    """Independent NumPy oracle for the weighted scatter-add unpermute.

    Mirrors the *documented* contract (optionally weight each permuted row by a
    gathered probability, then accumulate into the destination row given by
    ``token_idx``) without calling any production Paddle op.
    """
    permuted = np.asarray(permuted, dtype=np.float64)
    token_idx = np.asarray(token_idx, dtype=np.int64)
    if prob_flat is not None:
        gathered = np.asarray(prob_flat, dtype=np.float64).reshape(-1)[
            np.asarray(prob_idx, dtype=np.int64)
        ]
        permuted = permuted * gathered[:, None]
    out = np.zeros((num_tokens, hidden), dtype=np.float64)
    np.add.at(out, token_idx, permuted)
    return out


class _MoeUtilsTestBase(unittest.TestCase):
    def setUp(self):
        if _IMPORT_ERROR is not None:
            self.skipTest(
                "paddlefleet.cli.train.deepseek_v3_pretrain.moe_utils import "
                f"failed (dependency unavailable): {_IMPORT_ERROR!r}"
            )


class TestTopkToPermutedIndices(_MoeUtilsTestBase):
    """Grouping a [num_tokens, topk] routing map by expert id."""

    def test_topk2_grouping_order_and_token_map(self):
        # Fixed routing map; flattened row-major it is
        #   [0, 1, 1, 2, 0, 2, 2, 3]  (flat positions 0..7)
        # Expert 0 sits at flat positions {0, 4}; expert 1 at {1, 2};
        # expert 2 at {3, 5, 6}; expert 3 at {7}. _restrict_nonzero yields
        # them in ascending order, concatenated expert-by-expert.
        dispatched = paddle.to_tensor(
            [[0, 1], [1, 2], [0, 2], [2, 3]], dtype="int64"
        )
        num_tokens_per_expert = [2, 2, 3, 1]
        token_idx, prob_idx = MOE.topk_to_permuted_indices(
            dispatched, num_tokens_per_expert, topk=2
        )
        # prob index = flat position grouped by expert.
        self.assertEqual(prob_idx.tolist(), [0, 4, 1, 2, 3, 5, 6, 7])
        # token index = prob index // topk maps each assignment to its token.
        self.assertEqual(token_idx.tolist(), [0, 2, 0, 1, 1, 2, 3, 3])

    def test_topk1_is_pure_regrouping_of_tokens(self):
        # With topk == 1 the flat position *is* the token id, so token and prob
        # indices coincide and the result is the tokens reordered by expert.
        dispatched = paddle.to_tensor([[2], [0], [1], [0]], dtype="int64")
        # Expert 0 at flat positions {1, 3}; expert 1 at {2}; expert 2 at {0}.
        num_tokens_per_expert = [2, 1, 1]
        token_idx, prob_idx = MOE.topk_to_permuted_indices(
            dispatched, num_tokens_per_expert, topk=1
        )
        self.assertEqual(prob_idx.tolist(), [1, 3, 2, 0])
        self.assertEqual(token_idx.tolist(), [1, 3, 2, 0])


class TestPermuteFast(_MoeUtilsTestBase):
    """axis-0 gather that reorders (and may duplicate) token rows."""

    def test_reorders_rows_by_index_exact_content(self):
        tokens = paddle.arange(12, dtype="float32").reshape([4, 3])
        index = paddle.to_tensor([2, 0, 3, 1], dtype="int64")
        out = MOE.permute_fast(tokens, index)
        expected = tokens.numpy()[[2, 0, 3, 1]]
        np.testing.assert_array_equal(out.numpy(), expected)

    def test_repeated_index_duplicates_rows(self):
        # A repeated index must copy the same source row twice; this is real
        # MoE behavior (one token routed to several experts).
        tokens = paddle.arange(12, dtype="float32").reshape([4, 3])
        index = paddle.to_tensor([0, 0, 2], dtype="int64")
        out = MOE.permute_fast(tokens, index)
        expected = tokens.numpy()[[0, 0, 2]]
        self.assertEqual(list(out.shape), [3, 3])
        np.testing.assert_array_equal(out.numpy(), expected)

    def test_drop_and_pad_rejected(self):
        tokens = paddle.arange(12, dtype="float32").reshape([4, 3])
        index = paddle.to_tensor([0, 1, 2, 3], dtype="int64")
        with self.assertRaises(AssertionError):
            MOE.permute_fast(tokens, index, drop_and_pad=True)


class TestUnpermuteFast(_MoeUtilsTestBase):
    """Probability-weighted scatter-add back to the original token rows."""

    def test_scatter_add_without_probs_accumulates(self):
        # Rows 0 and 1 both target output row 0 (must sum); row 2 targets
        # output row 2; output row 1 receives nothing and stays zero.
        permuted = paddle.to_tensor(
            [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype="float32"
        )
        token_idx = paddle.to_tensor([0, 0, 2], dtype="int64")
        prob_idx = paddle.to_tensor([0, 1, 2], dtype="int64")
        out = MOE.unpermute_fast(permuted, token_idx, prob_idx, [3, 2])
        expected = _scatter_add_reference(
            permuted.numpy(), [0, 0, 2], num_tokens=3, hidden=2
        )
        np.testing.assert_allclose(out.numpy(), expected, atol=1e-6)
        # Explicit hand check of the accumulation and the untouched row.
        np.testing.assert_array_equal(
            out.numpy(), np.array([[3.0, 3.0], [0.0, 0.0], [3.0, 3.0]])
        )

    def test_scatter_add_with_prob_weighting(self):
        # probs are flattened then gathered by prob_idx (a real reorder here),
        # each permuted row is scaled, then accumulated.
        permuted = paddle.to_tensor(
            [[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]], dtype="float32"
        )
        token_idx = paddle.to_tensor([0, 2, 2], dtype="int64")
        prob_idx = paddle.to_tensor([2, 0, 1], dtype="int64")
        probs = paddle.to_tensor([0.5, 1.0, 2.0], dtype="float32")
        out = MOE.unpermute_fast(
            permuted, token_idx, prob_idx, [3, 2], probs=probs
        )
        expected = _scatter_add_reference(
            permuted.numpy(),
            [0, 2, 2],
            num_tokens=3,
            hidden=2,
            prob_flat=[0.5, 1.0, 2.0],
            prob_idx=[2, 0, 1],
        )
        np.testing.assert_allclose(out.numpy(), expected, atol=1e-6)
        # Hand check: gathered probs = [2.0, 0.5, 1.0] -> scaled rows
        # [20,20],[10,10],[30,30]; row0=[20,20], row2=[10,10]+[30,30]=[40,40].
        np.testing.assert_array_equal(
            out.numpy(), np.array([[20.0, 20.0], [0.0, 0.0], [40.0, 40.0]])
        )

    def test_permute_then_unpermute_is_identity(self):
        # On a genuine permutation (each index once, no probs) unpermute is the
        # exact inverse of permute: every original row lands back in place.
        tokens = paddle.arange(12, dtype="float32").reshape([4, 3])
        perm = paddle.to_tensor([2, 0, 3, 1], dtype="int64")
        permuted = MOE.permute_fast(tokens, perm)
        restored = MOE.unpermute_fast(
            permuted,
            perm,
            paddle.to_tensor([0, 1, 2, 3], dtype="int64"),
            [4, 3],
        )
        np.testing.assert_array_equal(restored.numpy(), tokens.numpy())

    def test_drop_and_pad_rejected(self):
        permuted = paddle.arange(8, dtype="float32").reshape([4, 2])
        idx = paddle.to_tensor([0, 1, 2, 3], dtype="int64")
        with self.assertRaises(AssertionError):
            MOE.unpermute_fast(permuted, idx, idx, [4, 2], drop_and_pad=True)


class TestGetEnvDevice(_MoeUtilsTestBase):
    """Device-name dispatch table and branch priority.

    The four Paddle capability probes are collaborators; controlling them lets
    us observe the real branch selection (including priority) that is the
    behavior under test. We are not patching ``get_env_device`` itself.
    """

    def _device_under(self, cuda=False, custom=None, rocm=False, xpu=False):
        custom = list(custom or [])
        with (
            mock.patch.object(
                MOE.paddle, "is_compiled_with_cuda", return_value=cuda
            ),
            mock.patch.object(
                MOE.paddle.device,
                "get_all_custom_device_type",
                return_value=custom,
            ),
            mock.patch.object(
                MOE.paddle, "is_compiled_with_rocm", return_value=rocm
            ),
            mock.patch.object(
                MOE.paddle, "is_compiled_with_xpu", return_value=xpu
            ),
        ):
            return MOE.get_env_device()

    def test_cuda_maps_to_gpu(self):
        self.assertEqual(self._device_under(cuda=True), "gpu")

    def test_custom_device_names(self):
        self.assertEqual(self._device_under(custom=["npu"]), "npu")
        self.assertEqual(self._device_under(custom=["mlu"]), "mlu")
        self.assertEqual(self._device_under(custom=["gcu"]), "gcu")
        self.assertEqual(self._device_under(custom=["intel_hpu"]), "intel_hpu")

    def test_rocm_and_xpu_only_when_nothing_else(self):
        self.assertEqual(self._device_under(rocm=True), "rocm")
        self.assertEqual(self._device_under(xpu=True), "xpu")

    def test_falls_back_to_cpu(self):
        self.assertEqual(self._device_under(), "cpu")

    def test_cuda_has_priority_over_custom_device(self):
        # cuda is checked first, so a present custom npu must not win.
        self.assertEqual(self._device_under(cuda=True, custom=["npu"]), "gpu")

    def test_custom_device_has_priority_over_rocm_and_xpu(self):
        self.assertEqual(
            self._device_under(custom=["npu"], rocm=True, xpu=True), "npu"
        )


class TestMergeSubbatchCast(_MoeUtilsTestBase):
    """Single-item identity vs. cast, and multi-item routing to the ext op."""

    def test_single_tensor_same_dtype_returns_same_object(self):
        x = paddle.arange(8, dtype="float32").reshape([2, 4])
        out = MOE.merge_subbatch_cast(x, paddle.float32)
        # Same dtype must short-circuit and return the identical tensor.
        self.assertIs(out, x)

    def test_single_tensor_cast_preserves_values(self):
        x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
        out = MOE.merge_subbatch_cast(x, paddle.float64)
        self.assertEqual(out.dtype, paddle.float64)
        np.testing.assert_array_equal(
            out.numpy(), np.array([[1.0, 2.0], [3.0, 4.0]])
        )

    def test_single_element_list_same_dtype_returns_inner(self):
        inner = paddle.arange(6, dtype="float32").reshape([2, 3])
        out = MOE.merge_subbatch_cast([inner], paddle.float32)
        self.assertIs(out, inner)

    def test_single_element_list_cast_preserves_values(self):
        inner = paddle.to_tensor([[5.0, 6.0]], dtype="float32")
        out = MOE.merge_subbatch_cast((inner,), paddle.float64)
        self.assertEqual(out.dtype, paddle.float64)
        np.testing.assert_array_equal(out.numpy(), np.array([[5.0, 6.0]]))

    def test_multi_element_list_routes_to_extension_op(self):
        # The >1 path delegates to the compiled extension op; that op is a
        # collaborator, mocked with a distinguishable marker so we can assert
        # the exact forwarded arguments and that the return is propagated.
        captured = {}
        sentinel = object()

        def marker(x, dtype):
            captured["x"] = x
            captured["dtype"] = dtype
            return sentinel

        stub = types.SimpleNamespace(
            extensions=types.SimpleNamespace(
                ops=types.SimpleNamespace(merge_subbatch_cast=marker)
            )
        )
        parts = [
            paddle.zeros([1, 2], dtype="float32"),
            paddle.ones([1, 2], dtype="float32"),
        ]
        with mock.patch.object(MOE, "paddlefleet", stub):
            out = MOE.merge_subbatch_cast(parts, paddle.float16)
        self.assertIs(out, sentinel)
        self.assertIs(captured["x"], parts)
        self.assertEqual(captured["dtype"], paddle.float16)


class TestTokensZipUniqueAddWithSubbatch(_MoeUtilsTestBase):
    """Dispatch branch and the sub-batch row-splitting math.

    Both compiled extension ops are collaborators, mocked with markers that
    record their arguments; the row splitting and branch selection are the
    real behavior under test.
    """

    def _make_stub(self, captured):
        def add_marker(*args):
            captured["add"] = args
            return "ADD"

        def subbatch_marker(*args):
            captured["subbatch"] = args
            return "SUBBATCH"

        return types.SimpleNamespace(
            extensions=types.SimpleNamespace(
                ops=types.SimpleNamespace(
                    tokens_zip_unique_add=add_marker,
                    tokens_zip_unique_add_subbatch=subbatch_marker,
                )
            )
        )

    def test_no_subbatch_uses_direct_op(self):
        captured = {}
        zipped = paddle.ones([4, 3], dtype="float32")
        with mock.patch.object(MOE, "paddlefleet", self._make_stub(captured)):
            out = MOE.tokens_zip_unique_add_with_subbatch(
                zipped, "UNZ", "IDX", 4, subbatch_rows=None
            )
        self.assertEqual(out, "ADD")
        self.assertNotIn("subbatch", captured)
        args = captured["add"]
        self.assertIs(args[0], zipped)
        self.assertEqual(args[1], "UNZ")
        self.assertEqual(args[2], "IDX")
        self.assertEqual(args[3], 4)

    def test_subbatch_remainder_split_content(self):
        # zipped_rows=5, subbatch_rows=2 -> row groups [2, 2, 1].
        captured = {}
        zipped = paddle.arange(15, dtype="float32").reshape([5, 3])
        with mock.patch.object(MOE, "paddlefleet", self._make_stub(captured)):
            out = MOE.tokens_zip_unique_add_with_subbatch(
                zipped, "UNZ", "IDX", 5, subbatch_rows=2
            )
        self.assertEqual(out, "SUBBATCH")
        self.assertNotIn("add", captured)
        split_list, unz, idx, zrows, srows = captured["subbatch"]
        self.assertEqual([int(t.shape[0]) for t in split_list], [2, 2, 1])
        src = zipped.numpy()
        np.testing.assert_array_equal(split_list[0].numpy(), src[0:2])
        np.testing.assert_array_equal(split_list[1].numpy(), src[2:4])
        np.testing.assert_array_equal(split_list[2].numpy(), src[4:5])
        self.assertEqual((unz, idx, zrows, srows), ("UNZ", "IDX", 5, 2))

    def test_subbatch_even_split(self):
        # zipped_rows=4, subbatch_rows=2 -> row groups [2, 2].
        captured = {}
        zipped = paddle.arange(8, dtype="float32").reshape([4, 2])
        with mock.patch.object(MOE, "paddlefleet", self._make_stub(captured)):
            MOE.tokens_zip_unique_add_with_subbatch(
                zipped, "UNZ", "IDX", 4, subbatch_rows=2
            )
        split_list = captured["subbatch"][0]
        self.assertEqual([int(t.shape[0]) for t in split_list], [2, 2])
        np.testing.assert_array_equal(
            split_list[0].numpy(), zipped.numpy()[0:2]
        )
        np.testing.assert_array_equal(
            split_list[1].numpy(), zipped.numpy()[2:4]
        )

    def test_subbatch_empty_input_is_zero_filled(self):
        # Empty zipped (0 rows) with zipped_rows=3, subbatch_rows=2 must be
        # rebuilt as zero groups [2, 1] preserving hidden size.
        captured = {}
        zipped = paddle.zeros([0, 4], dtype="float32")
        with mock.patch.object(MOE, "paddlefleet", self._make_stub(captured)):
            MOE.tokens_zip_unique_add_with_subbatch(
                zipped, "UNZ", "IDX", 3, subbatch_rows=2
            )
        split_list = captured["subbatch"][0]
        self.assertEqual([list(t.shape) for t in split_list], [[2, 4], [1, 4]])
        for t in split_list:
            np.testing.assert_array_equal(
                t.numpy(), np.zeros(list(t.shape), dtype=np.float32)
            )


def _record_marker(store, key, ret):
    """Build a marker that records (args, kwargs) under ``key`` and returns
    the distinguishable ``ret`` value."""

    def marker(*args, **kwargs):
        store[key] = (args, kwargs)
        return ret

    return marker


class TestUnZipNode(_MoeUtilsTestBase):
    """State contract plus argument marshaling into the fused MoE kernels."""

    def test_init_state(self):
        node = MOE.UnZipNode(name="u0")
        self.assertEqual(node.name, "u0")
        self.assertIsNone(node.unzipped_probs)
        self.assertIsNone(node.zipped_expertwise_rowmap)

    def test_reset_statue_clears_seeded_state(self):
        node = MOE.UnZipNode()
        node.unzipped_probs = "seed_p"
        node.zipped_expertwise_rowmap = "seed_r"
        node.reset_statue()
        self.assertIsNone(node.unzipped_probs)
        self.assertIsNone(node.zipped_expertwise_rowmap)

    def test_forward_tuple_input_marshals_and_stores(self):
        node = MOE.UnZipNode()
        rets = ("TOK", "ROWMAP", "PROBS", "SCALE")
        store = {}
        hs = ("HS0", "HS1")
        with mock.patch.object(
            MOE.paddle.nn.functional,
            "moe_permute",
            new=_record_marker(store, "p", rets),
            create=True,
        ):
            out = node.forward(
                hs,
                "IDX",
                "DPROB",
                topk=2,
                num_experts=8,
                tokens_per_expert="TPE",
            )
        args, kwargs = store["p"]
        # tuple input -> the two components are forwarded positionally.
        self.assertEqual(args[:4], ("HS0", "HS1", "IDX", "DPROB"))
        self.assertEqual(kwargs["num_experts"], 8)
        self.assertEqual(kwargs["tokens_per_expert"], "TPE")
        self.assertEqual(kwargs["padding_alignment"], 128)
        # returned tuple and stored state select the right kernel outputs.
        self.assertEqual(out, rets)
        self.assertEqual(node.unzipped_probs, "PROBS")
        self.assertEqual(node.zipped_expertwise_rowmap, "ROWMAP")

    def test_forward_non_tuple_input_passes_none_scale(self):
        node = MOE.UnZipNode()
        store = {}
        with mock.patch.object(
            MOE.paddle.nn.functional,
            "moe_permute",
            new=_record_marker(store, "p", ("T", "R", "P", "S")),
            create=True,
        ):
            node.forward("HS", "IDX", "DPROB", 2, 8, "TPE")
        args, _ = store["p"]
        self.assertEqual(args[:4], ("HS", None, "IDX", "DPROB"))

    def test_backward_marshals_and_resets_state(self):
        node = MOE.UnZipNode()
        node.zipped_expertwise_rowmap = "ROWMAP"
        node.unzipped_probs = "P"
        store = {}
        with mock.patch.object(
            MOE.paddle.nn.functional,
            "moe_unpermute",
            new=_record_marker(store, "u", ("WZT", "PGZ")),
            create=True,
        ):
            out = node.backward("DX", "TZT", "PGRAD", "IDX", num_experts=8)
        args, kwargs = store["u"]
        self.assertEqual(args[:4], ("DX", "ROWMAP", "IDX", "PGRAD"))
        self.assertEqual(kwargs["total_zipped_tokens"], "TZT")
        self.assertEqual(kwargs["num_experts"], 8)
        self.assertEqual(out, ("WZT", "PGZ"))
        # backward calls reset_statue on the way out.
        self.assertIsNone(node.zipped_expertwise_rowmap)
        self.assertIsNone(node.unzipped_probs)


class TestZipNode(_MoeUtilsTestBase):
    """Argument marshaling and return selection for the zip node."""

    def test_init_state(self):
        node = MOE.ZipNode(name="z0")
        self.assertEqual(node.name, "z0")

    def test_forward_positional_args_and_returns_first(self):
        node = MOE.ZipNode()
        store = {}
        with mock.patch.object(
            MOE.paddle.nn.functional,
            "moe_unpermute",
            new=_record_marker(store, "u", ("OUT_ZIPPED", "PROBS_TOPK")),
            create=True,
        ):
            out = node.forward("EOUT", "ROWMAP", "ROUTEMAP", "UPROBS", "TZT", 8)
        args, _ = store["u"]
        self.assertEqual(
            args, ("EOUT", "ROWMAP", "ROUTEMAP", "UPROBS", "TZT", 8)
        )
        # forward returns only the zipped expert output, not the probs.
        self.assertEqual(out, "OUT_ZIPPED")

    def test_backward_tuple_returns_grad_and_scale(self):
        node = MOE.ZipNode()
        store = {}
        rets = ("UGRAD", "ROWMAP_GRAD", "UPROBS_GRAD", "USCALE_GRAD")
        with mock.patch.object(
            MOE.paddle.nn.functional,
            "moe_permute",
            new=_record_marker(store, "p", rets),
            create=True,
        ):
            out = node.backward(("G0", "G1"), "IDX", "DPROB", 2, 8, "TPE")
        args, kwargs = store["p"]
        self.assertEqual(args, ("G0", "G1", "IDX", "DPROB", 8, "TPE"))
        self.assertEqual(kwargs["padding_alignment"], 128)
        # tuple grad -> (unzipped_grad, unzipped_scale_grad).
        self.assertEqual(out, ("UGRAD", "USCALE_GRAD"))

    def test_backward_non_tuple_returns_grad_only(self):
        node = MOE.ZipNode()
        store = {}
        rets = ("UGRAD", "ROWMAP_GRAD", "UPROBS_GRAD", "USCALE_GRAD")
        with mock.patch.object(
            MOE.paddle.nn.functional,
            "moe_permute",
            new=_record_marker(store, "p", rets),
            create=True,
        ):
            out = node.backward("G", "IDX", "DPROB", 2, 8, "TPE")
        args, _ = store["p"]
        self.assertEqual(args, ("G", None, "IDX", "DPROB", 8, "TPE"))
        self.assertEqual(out, "UGRAD")


def _make_dispatcher(tokens_per_expert, router_topk):
    """Lightweight stand-in for the token dispatcher: only the comm-manager
    config fields that the permute nodes read/write are provided."""
    return types.SimpleNamespace(
        _comm_manager=types.SimpleNamespace(
            tokens_per_expert=list(tokens_per_expert),
            router_topk=router_topk,
        )
    )


class TestPermuteNode(_MoeUtilsTestBase):
    """End-to-end CPU permute path built on the pure helpers."""

    def test_init_stores_dispatcher_and_name(self):
        disp = _make_dispatcher([1], 1)
        node = MOE.PermuteNode(disp, name="p0")
        self.assertEqual(node.name, "p0")
        self.assertIs(node.token_dispatcher, disp)

    def test_reset_status_clears_seeded_state(self):
        node = MOE.PermuteNode(_make_dispatcher([1], 1))
        node.token_permuted_indices = "seed"
        node.prob_permuted_indices = "seed"
        node.reset_status()
        self.assertIsNone(node.token_permuted_indices)
        self.assertIsNone(node.prob_permuted_indices)

    def test_forward_permutes_tokens_and_scale_and_records_shape(self):
        disp = _make_dispatcher([2, 2, 3, 1], router_topk=2)
        node = MOE.PermuteNode(disp)
        hidden = paddle.arange(8, dtype="float32").reshape([4, 2])
        scale = (paddle.arange(12, dtype="float32") + 100.0).reshape([4, 3])
        dispatched = paddle.to_tensor(
            [[0, 1], [1, 2], [0, 2], [2, 3]], dtype="int64"
        )
        out_h, out_s, token_idx, prob_idx = node.forward(
            hidden, scale, dispatched
        )
        self.assertEqual(token_idx.tolist(), [0, 2, 0, 1, 1, 2, 3, 3])
        self.assertEqual(prob_idx.tolist(), [0, 4, 1, 2, 3, 5, 6, 7])
        order = [0, 2, 0, 1, 1, 2, 3, 3]
        np.testing.assert_array_equal(out_h.numpy(), hidden.numpy()[order])
        np.testing.assert_array_equal(out_s.numpy(), scale.numpy()[order])
        # The pre-permute shape is stashed on both node and comm manager.
        self.assertEqual(list(node.hidden_shape_before_permute), [4, 2])
        self.assertEqual(
            list(disp._comm_manager.hidden_shape_before_permute), [4, 2]
        )

    def test_backward_is_weighted_scatter_add_and_resets(self):
        disp = _make_dispatcher([2, 2, 3, 1], router_topk=2)
        node = MOE.PermuteNode(disp)
        hidden = paddle.arange(8, dtype="float32").reshape([4, 2])
        scale = paddle.arange(12, dtype="float32").reshape([4, 3])
        dispatched = paddle.to_tensor(
            [[0, 1], [1, 2], [0, 2], [2, 3]], dtype="int64"
        )
        node.forward(hidden, scale, dispatched)  # populates node state

        out_grad = paddle.arange(16, dtype="float32").reshape([8, 2])
        probs = (paddle.arange(8, dtype="float32") * 0.1).reshape([4, 2])
        grad = node.backward(out_grad, probs)

        token_idx = [0, 2, 0, 1, 1, 2, 3, 3]
        prob_idx = [0, 4, 1, 2, 3, 5, 6, 7]
        expected = _scatter_add_reference(
            out_grad.numpy(),
            token_idx,
            num_tokens=4,
            hidden=2,
            prob_flat=probs.numpy().reshape(-1),
            prob_idx=prob_idx,
        )
        np.testing.assert_allclose(grad.numpy(), expected, atol=1e-5)
        # backward resets the permute indices.
        self.assertIsNone(node.token_permuted_indices)
        self.assertIsNone(node.prob_permuted_indices)


class TestUnPermuteNode(_MoeUtilsTestBase):
    """State contract plus the pure-Paddle weighted-scatter forward path.

    ``UnPermuteNode.backward`` is intentionally not exercised here: it runs FP8
    dequantization (``FP8LinearFunctionBase.dequantize_fp8_to_fp32``) and
    ``paddle._C_ops.put_along_axis_grad``, which require a GPU build and are
    covered by the single-/multi-card MoE numeric suites.
    """

    def test_init_stores_dispatcher_and_name(self):
        disp = _make_dispatcher([1], 1)
        node = MOE.UnPermuteNode(disp, name="up0")
        self.assertEqual(node.name, "up0")
        self.assertIs(node.token_dispatcher, disp)

    def test_reset_status_clears_all_seeded_fields(self):
        node = MOE.UnPermuteNode(_make_dispatcher([1], 1))
        node.token_permuted_indices = "a"
        node.hidden_states = "b"
        node.prob_permuted_indices = "c"
        node.faltten_dispatched_probs = "d"
        node.hidden = "e"
        node.permuted_tokens = "f"
        node.output_tokens = "g"
        node.reset_status()
        self.assertIsNone(node.token_permuted_indices)
        self.assertIsNone(node.hidden_states)
        self.assertIsNone(node.prob_permuted_indices)
        self.assertIsNone(node.faltten_dispatched_probs)
        self.assertIsNone(node.hidden)
        self.assertIsNone(node.permuted_tokens)
        self.assertIsNone(node.output_tokens)

    def test_forward_weighted_scatter_add(self):
        disp = _make_dispatcher([1], 1)
        disp._comm_manager.hidden_shape_before_permute = [3, 2]
        node = MOE.UnPermuteNode(disp)
        hidden = paddle.to_tensor(
            [[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]], dtype="float32"
        )
        token_idx = paddle.to_tensor([0, 2, 2], dtype="int64")
        prob_idx = paddle.to_tensor([2, 0, 1], dtype="int64")
        probs = paddle.to_tensor([0.5, 1.0, 2.0], dtype="float32")
        out = node.forward(hidden, token_idx, prob_idx, probs)

        expected = _scatter_add_reference(
            hidden.numpy(),
            [0, 2, 2],
            num_tokens=3,
            hidden=2,
            prob_flat=[0.5, 1.0, 2.0],
            prob_idx=[2, 0, 1],
        )
        np.testing.assert_allclose(out.numpy(), expected, atol=1e-5)
        np.testing.assert_array_equal(
            out.numpy(), np.array([[20.0, 20.0], [0.0, 0.0], [40.0, 40.0]])
        )
        # The (quirky) CPU output-buffer side effect is exercised.
        self.assertEqual(list(node.output_tokens.shape), [3, 2])


class TestHolderSize(_MoeUtilsTestBase):
    """Tensor byte-size helper contract (module-level monkeypatch)."""

    def test_holder_size_is_element_count_times_dtype_bytes(self):
        # float32 [2, 4] = 8 elements * 4 bytes = 32 bytes.
        x = paddle.zeros([2, 4], dtype="float32")
        self.assertEqual(x._holder_size(), 32)

    def test_holder_size_scales_with_shape_and_dtype(self):
        big = paddle.zeros([32, 64], dtype="float32")
        small = paddle.zeros([2, 4], dtype="float32")
        self.assertEqual(big._holder_size(), 32 * 64 * 4)
        self.assertGreater(big._holder_size(), small._holder_size())


if __name__ == "__main__":
    unittest.main()

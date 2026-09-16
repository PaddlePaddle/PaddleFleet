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

"""CPU-only behavior tests for the pure logic in
``paddlefleet.train_infer_consistent_ops`` -- the train_infer_consistent_inspect
probe helpers.

Covered (device-independent, no collectives, no fp8 kernels):
  * ``inspect_util`` gate: ``refresh_env_cache`` / ``inspect_enabled`` /
    ``inspect_tag_enabled`` (whitelist-then-blacklist tag filtering) and the
    layer-context store ``inspect_tensor_set_current_layer`` / ``get_current_layer``.
  * ``inspect_util`` internals: ``_with_element`` (tuple/list/dict rewrap),
    ``_stats`` (abssum/absmax/md5 with the -0.0 -> +0.0 collapse), ``_squeeze_shape``
    and ``_load_shape_ok`` (the row-count corruption gate).
  * ``inspect_util.inspect_tensor`` gate + real ``.npy`` save/load override on CPU.
  * ``permute.canonical_rows`` / ``scatter_canonical_rows`` -- the (token, expert)
    row gather/scatter with the ``-1`` = not-routed mask, run as real paddle CPU
    ops and checked against an independent numpy reference.
  * ``permute.inspect_tensor_set_permute_index`` -- the enable-gated index publish,
    observed through ``canonical_rows`` actually consuming it.
  * ``slice_util.last_dim_segment`` / ``scatter_last_dim_segment`` -- last-dim slice
    and its concat inverse.
  * ``ffn_act.inspect_tensor_force_unit_probs`` -- all-ones weights only when the
    tag is live, and the ``scale=None`` branch of ``dequant_dispatched_hidden_bf16``.

All expected values are hand-derived or computed with an independent numpy /
pure-python reference; nothing is read back from the code under test.

Out of scope (documented, not faked): the fp8 kernels
``paddle.incubate.nn.functional.fused_act_dequant`` / ``fp8_quant_blockwise``
reached by ``dequant_dispatched_hidden_bf16(scale != None)``, ``_quant_blockwise``,
``scatter_dispatched_hidden_bf16`` and ``requant_swiglu_output`` are GPU-only, so
their numerics are left to a single-card/GPU suite rather than mocked here.
"""

import hashlib
import math
import os
import shutil
import tempfile
import unittest

import numpy as np

try:
    import paddle

    from paddlefleet.train_infer_consistent_ops import (
        ffn_act,
        inspect_util,
        permute,
        slice_util,
    )

    _IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:  # honest dependency probe
    paddle = None
    ffn_act = inspect_util = permute = slice_util = None
    _IMPORT_ERROR = exc

_HAS_DEPS = _IMPORT_ERROR is None
_SKIP_REASON = (
    f"paddle / paddlefleet not importable on this CPU host: {_IMPORT_ERROR!r}"
)


def _rmtree(path):
    shutil.rmtree(path, ignore_errors=True)


# Every ABLATION_* variable the module snapshots at import; cleared and restored
# around each test so refresh_env_cache() reads a deterministic environment.
_ABLATION_KEYS = (
    "ABLATION_INSPECT_TENSOR",
    "ABLATION_TAG_WHITELIST",
    "ABLATION_TAG_BLACKLIST",
    "ABLATION_DUMP_SKIP_TAGS",
    "ABLATION_SAVE_TENSOR_PATH",
    "ABLATION_LOAD_TENSOR_PATH",
)


class _AblationEnvMixin:
    """Set ABLATION_* env vars, re-snapshot the module cache, restore on teardown."""

    def _apply_ablation_env(self, **values):
        saved = {k: os.environ.get(k) for k in _ABLATION_KEYS}

        def _restore():
            for key, val in saved.items():
                if val is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = val
            inspect_util.refresh_env_cache()

        self.addCleanup(_restore)
        for key in _ABLATION_KEYS:
            os.environ.pop(key, None)
        for key, val in values.items():
            os.environ[key] = val
        inspect_util.refresh_env_cache()


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestTagGate(_AblationEnvMixin, unittest.TestCase):
    """``inspect_enabled`` / ``inspect_tag_enabled`` / ``refresh_env_cache``."""

    def test_disabled_when_flag_unset(self):
        self._apply_ablation_env()  # nothing exported
        self.assertFalse(inspect_util.inspect_enabled())
        self.assertFalse(inspect_util.inspect_tag_enabled("anything"))

    def test_enabled_no_filters_admits_every_tag(self):
        self._apply_ablation_env(ABLATION_INSPECT_TENSOR="1")
        self.assertTrue(inspect_util.inspect_enabled())
        self.assertTrue(
            inspect_util.inspect_tag_enabled("moe_act_quant_output")
        )
        self.assertTrue(inspect_util.inspect_tag_enabled("literally_anything"))

    def test_whitelist_admits_only_listed_tags(self):
        self._apply_ablation_env(
            ABLATION_INSPECT_TENSOR="1", ABLATION_TAG_WHITELIST="a,b"
        )
        self.assertTrue(inspect_util.inspect_tag_enabled("a"))
        self.assertTrue(inspect_util.inspect_tag_enabled("b"))
        self.assertFalse(inspect_util.inspect_tag_enabled("c"))

    def test_blacklist_rejects_listed_tags(self):
        self._apply_ablation_env(
            ABLATION_INSPECT_TENSOR="1", ABLATION_TAG_BLACKLIST="a"
        )
        self.assertFalse(inspect_util.inspect_tag_enabled("a"))
        self.assertTrue(inspect_util.inspect_tag_enabled("b"))

    def test_blacklist_overrides_whitelist(self):
        # A tag on both lists is rejected: the blacklist clause is ORed in.
        self._apply_ablation_env(
            ABLATION_INSPECT_TENSOR="1",
            ABLATION_TAG_WHITELIST="a,b",
            ABLATION_TAG_BLACKLIST="b",
        )
        self.assertTrue(inspect_util.inspect_tag_enabled("a"))
        self.assertFalse(inspect_util.inspect_tag_enabled("b"))
        self.assertFalse(inspect_util.inspect_tag_enabled("c"))

    def test_refresh_re_reads_the_flag(self):
        self._apply_ablation_env(ABLATION_INSPECT_TENSOR="1")
        self.assertTrue(inspect_util.inspect_enabled())
        os.environ["ABLATION_INSPECT_TENSOR"] = "0"
        inspect_util.refresh_env_cache()
        self.assertFalse(inspect_util.inspect_enabled())


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestCurrentLayerContext(unittest.TestCase):
    """``inspect_tensor_set_current_layer`` / ``get_current_layer``."""

    def setUp(self):
        original = inspect_util.get_current_layer()
        self.addCleanup(inspect_util.inspect_tensor_set_current_layer, original)

    def test_stores_and_returns_layer_id(self):
        inspect_util.inspect_tensor_set_current_layer(7)
        self.assertEqual(inspect_util.get_current_layer(), 7)
        inspect_util.inspect_tensor_set_current_layer(0)
        self.assertEqual(inspect_util.get_current_layer(), 0)

    def test_none_maps_to_minus_one_sentinel(self):
        inspect_util.inspect_tensor_set_current_layer(None)
        self.assertEqual(inspect_util.get_current_layer(), -1)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestWithElement(unittest.TestCase):
    """``_with_element`` rewraps a tuple/list/dict without touching the original."""

    def test_tuple_replaced_in_fresh_tuple(self):
        a, b, c, z = object(), object(), object(), object()
        container = (a, b, c)
        out = inspect_util._with_element(container, 1, z)
        self.assertIsInstance(out, tuple)
        self.assertEqual(out, (a, z, c))
        self.assertIs(out[0], a)
        self.assertIs(out[2], c)
        self.assertEqual(container, (a, b, c))  # original untouched

    def test_list_replaced_in_fresh_list(self):
        a, b, c, z = object(), object(), object(), object()
        container = [a, b, c]
        out = inspect_util._with_element(container, 2, z)
        self.assertIsInstance(out, list)
        self.assertEqual(out, [a, b, z])
        self.assertIsNot(out, container)
        self.assertEqual(container, [a, b, c])

    def test_dict_replaced_in_fresh_dict(self):
        a, b, z = object(), object(), object()
        container = {"x": a, "y": b}
        out = inspect_util._with_element(container, "y", z)
        self.assertIsInstance(out, dict)
        self.assertIs(out["x"], a)
        self.assertIs(out["y"], z)
        self.assertIsNot(out, container)
        self.assertIs(container["y"], b)  # original untouched


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestStats(unittest.TestCase):
    """``_stats`` returns (abssum, absmax, md5) with a -0.0 -> +0.0 md5 collapse."""

    def test_matches_independent_reference(self):
        arr = np.array([[-3.0, 1.5], [2.0, -0.5]], dtype=np.float32)
        # Independent reference: fsum of |x|, max of |x|, md5 of the (+0.0) bytes.
        normalized = arr + np.float32(0.0)
        abs_list = np.abs(normalized).reshape(-1).tolist()
        exp_abssum = float(math.fsum(abs_list))
        exp_absmax = float(max(abs_list))
        exp_md5 = hashlib.md5(normalized.tobytes()).hexdigest()

        abssum, absmax, md5 = inspect_util._stats(arr)
        self.assertAlmostEqual(abssum, exp_abssum, places=6)
        self.assertAlmostEqual(absmax, exp_absmax, places=6)
        self.assertEqual(md5, exp_md5)
        # 7.0 = |−3.0| + |1.5| + |2.0| + |−0.5| = 3.0 + 1.5 + 2.0 + 0.5,
        # max 3.0 -- pinned by hand too.
        self.assertAlmostEqual(abssum, 7.0, places=6)
        self.assertAlmostEqual(absmax, 3.0, places=6)

    def test_negative_zero_collapses_in_md5(self):
        neg = np.array([[-3.0, 1.5], [2.0, -0.0]], dtype=np.float32)
        pos = np.array([[-3.0, 1.5], [2.0, 0.0]], dtype=np.float32)
        md5_neg = inspect_util._stats(neg)[2]
        md5_pos = inspect_util._stats(pos)[2]
        # The -0.0 and +0.0 inputs must hash identically after normalization.
        self.assertEqual(md5_neg, md5_pos)
        # And that shared hash must differ from the RAW (un-normalized) bytes of
        # the -0.0 array, proving the collapse actually happened.
        self.assertNotEqual(md5_neg, hashlib.md5(neg.tobytes()).hexdigest())

    def test_empty_array_absmax_is_zero(self):
        abssum, absmax, _ = inspect_util._stats(np.array([], dtype=np.float32))
        self.assertEqual(abssum, 0.0)
        self.assertEqual(absmax, 0.0)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestSqueezeShape(unittest.TestCase):
    """``_squeeze_shape`` drops every size-1 dim."""

    def test_drops_size_one_dims(self):
        self.assertEqual(inspect_util._squeeze_shape([1, 11, 4096]), (11, 4096))
        self.assertEqual(
            inspect_util._squeeze_shape([11, 1, 1, 4096]), (11, 4096)
        )

    def test_keeps_non_unit_dims(self):
        self.assertEqual(inspect_util._squeeze_shape([2, 3]), (2, 3))

    def test_all_ones_collapses_to_empty(self):
        self.assertEqual(inspect_util._squeeze_shape([1, 1, 1]), ())


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestLoadShapeOk(unittest.TestCase):
    """``_load_shape_ok`` is the row-count gate that blocks numel-collision reshapes."""

    def test_exact_shape(self):
        ok, reason = inspect_util._load_shape_ok([11, 4096], [11, 4096])
        self.assertTrue(ok)
        self.assertEqual(reason, "exact")

    def test_equal_ignoring_size_one(self):
        ok, reason = inspect_util._load_shape_ok([1, 11, 4096], [11, 4096])
        self.assertTrue(ok)
        self.assertEqual(reason, "equal ignoring size-1 dims")

    def test_same_row_count_trailing_regrouped(self):
        # 11 rows both sides, trailing 4x4096 folded into 16384: a legit regroup.
        ok, reason = inspect_util._load_shape_ok([11, 4, 4096], [11, 16384])
        self.assertTrue(ok)
        self.assertEqual(reason, "same row count, trailing dims regrouped")

    def test_row_count_collision_rejected(self):
        # 88*3584 == 11*28672 numel-wise, but 88 rows cannot describe 11 tokens:
        # this is the dp-gathered x tp-shard corruption the gate must reject.
        ok, reason = inspect_util._load_shape_ok([88, 3584], [11, 28672])
        self.assertFalse(ok)
        self.assertTrue(reason.startswith("row count differs"))

    def test_numel_mismatch_rejected(self):
        ok, reason = inspect_util._load_shape_ok([10, 10], [10, 5])
        self.assertFalse(ok)
        self.assertTrue(reason.startswith("numel mismatch"))


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestCanonicalRows(_AblationEnvMixin, unittest.TestCase):
    """``canonical_rows`` gathers routed buffer rows; ``-1`` pairs come back zero."""

    def setUp(self):
        paddle.set_device("cpu")
        # Never leak a published index into other tests.
        self.addCleanup(setattr, permute, "_PERMUTE_INDEX", None)
        permute._PERMUTE_INDEX = None

    def test_gathers_routed_rows_and_zeros_unrouted(self):
        buf_np = np.arange(10, dtype=np.float32).reshape(5, 2)
        idx_np = np.array(
            [[3, -1], [0, 2]], dtype=np.int64
        )  # pair -> buffer row
        out = permute.canonical_rows(
            paddle.to_tensor(buf_np),
            index=paddle.to_tensor(idx_np),
        )
        # Independent reference: gather clip(idx,0) then zero rows where idx < 0.
        flat = idx_np.reshape(-1)
        gathered = buf_np[np.clip(flat, 0, None)]
        keep = (flat >= 0).reshape(-1, 1).astype(np.float32)
        expected = gathered * keep
        np.testing.assert_array_equal(out.numpy(), expected)
        # Pin the actual values: row for the -1 pair is [0, 0], not buf[0].
        np.testing.assert_array_equal(
            out.numpy(),
            np.array(
                [[6.0, 7.0], [0.0, 0.0], [0.0, 1.0], [4.0, 5.0]], "float32"
            ),
        )

    def test_returns_none_without_published_index(self):
        buf = paddle.arange(6, dtype="float32").reshape([3, 2])
        self.assertIsNone(permute.canonical_rows(buf))

    def test_returns_none_for_none_buffer(self):
        idx = paddle.to_tensor(np.array([[0]], dtype=np.int64))
        self.assertIsNone(permute.canonical_rows(None, index=idx))


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestScatterCanonicalRows(_AblationEnvMixin, unittest.TestCase):
    """``scatter_canonical_rows`` writes canonical rows back to their buffer rows."""

    def setUp(self):
        paddle.set_device("cpu")
        self.addCleanup(setattr, permute, "_PERMUTE_INDEX", None)
        permute._PERMUTE_INDEX = None

    def test_scatters_kept_rows_and_ignores_unrouted(self):
        buf_np = np.zeros((5, 2), dtype=np.float32)
        idx_np = np.array([[3, -1], [0, 2]], dtype=np.int64)
        # canon row 1 (the -1 pair) carries a sentinel that must never land.
        canon_np = np.array(
            [[10.0, 11.0], [99.0, 99.0], [20.0, 21.0], [30.0, 31.0]], "float32"
        )
        out = permute.scatter_canonical_rows(
            paddle.to_tensor(buf_np),
            paddle.to_tensor(canon_np),
            index=paddle.to_tensor(idx_np),
        )
        # Independent reference: out[flat[keep]] = canon[keep].
        flat = idx_np.reshape(-1)
        keep = np.nonzero(flat >= 0)[0]
        expected = buf_np.copy()
        expected[flat[keep]] = canon_np[keep]
        np.testing.assert_array_equal(out.numpy(), expected)
        # Explicit: buffer row 3 <- canon 0, row 0 <- canon 2, row 2 <- canon 3;
        # rows 1 and 4 stay zero and the 99.0 sentinel is nowhere.
        np.testing.assert_array_equal(
            out.numpy(),
            np.array(
                [
                    [20.0, 21.0],
                    [0.0, 0.0],
                    [30.0, 31.0],
                    [10.0, 11.0],
                    [0.0, 0.0],
                ],
                "float32",
            ),
        )
        self.assertFalse((out.numpy() == 99.0).any())

    def test_roundtrip_restores_routed_rows(self):
        buf_np = np.arange(10, dtype=np.float32).reshape(5, 2)
        idx_np = np.array([[3, -1], [0, 2]], dtype=np.int64)
        buf = paddle.to_tensor(buf_np)
        idx = paddle.to_tensor(idx_np)
        canon = permute.canonical_rows(buf, index=idx)
        restored = permute.scatter_canonical_rows(
            paddle.to_tensor(buf_np), canon, index=idx
        )
        # Every routed buffer row must survive canonical_rows -> scatter unchanged.
        flat = idx_np.reshape(-1)
        for row in flat[flat >= 0]:
            np.testing.assert_array_equal(restored.numpy()[row], buf_np[row])

    def test_none_canon_returns_buffer_unchanged(self):
        buf = paddle.arange(6, dtype="float32").reshape([3, 2])
        idx = paddle.to_tensor(np.array([[0, 1, 2]], dtype=np.int64))
        out = permute.scatter_canonical_rows(buf, None, index=idx)
        np.testing.assert_array_equal(out.numpy(), buf.numpy())


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestSetPermuteIndex(_AblationEnvMixin, unittest.TestCase):
    """``inspect_tensor_set_permute_index`` publishes only while probes are on."""

    def setUp(self):
        paddle.set_device("cpu")
        self.addCleanup(setattr, permute, "_PERMUTE_INDEX", None)
        permute._PERMUTE_INDEX = None

    def test_disabled_does_not_publish(self):
        self._apply_ablation_env()  # ABLATION_INSPECT_TENSOR unset
        idx = paddle.to_tensor(np.array([[0, 1]], dtype=np.int64))
        permute.inspect_tensor_set_permute_index(idx)
        self.assertIsNone(permute._PERMUTE_INDEX)
        # And a downstream consumer sees no index either.
        buf = paddle.arange(4, dtype="float32").reshape([2, 2])
        self.assertIsNone(permute.canonical_rows(buf))

    def test_enabled_publishes_and_is_consumed(self):
        self._apply_ablation_env(ABLATION_INSPECT_TENSOR="1")
        buf_np = np.arange(6, dtype=np.float32).reshape(3, 2)
        idx_np = np.array([[2, 0]], dtype=np.int64)
        permute.inspect_tensor_set_permute_index(paddle.to_tensor(idx_np))
        self.assertIsNotNone(permute._PERMUTE_INDEX)
        # canonical_rows with no explicit index must use the published one.
        out = permute.canonical_rows(paddle.to_tensor(buf_np))
        expected = buf_np[idx_np.reshape(-1)]  # both entries >= 0, no masking
        np.testing.assert_array_equal(out.numpy(), expected)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestLastDimSegment(unittest.TestCase):
    """``last_dim_segment`` views ``tensor[..., start:end]``."""

    def setUp(self):
        paddle.set_device("cpu")

    def test_bounded_segment(self):
        base = np.arange(36, dtype=np.float32).reshape(2, 3, 6)
        out = slice_util.last_dim_segment(
            paddle.to_tensor(base), start=2, end=5
        )
        np.testing.assert_array_equal(out.numpy(), base[..., 2:5])

    def test_open_ended_segment(self):
        base = np.arange(36, dtype=np.float32).reshape(2, 3, 6)
        out = slice_util.last_dim_segment(paddle.to_tensor(base), start=4)
        np.testing.assert_array_equal(out.numpy(), base[..., 4:])

    def test_none_tensor_returns_none(self):
        self.assertIsNone(slice_util.last_dim_segment(None, start=0))


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestScatterLastDimSegment(unittest.TestCase):
    """``scatter_last_dim_segment`` rebuilds full width with a replaced segment."""

    def setUp(self):
        paddle.set_device("cpu")

    def test_middle_segment_replaced_keeping_head_and_tail(self):
        base = np.arange(36, dtype=np.float32).reshape(2, 3, 6)
        seg = np.full((2, 3, 3), -1.0, dtype=np.float32)  # distinguishable
        out = slice_util.scatter_last_dim_segment(
            paddle.to_tensor(base), paddle.to_tensor(seg), start=2, end=5
        )
        expected = base.copy()
        expected[..., 2:5] = seg
        np.testing.assert_array_equal(out.numpy(), expected)
        # Head [:2] and tail [5:] must come straight from the live tensor.
        np.testing.assert_array_equal(out.numpy()[..., :2], base[..., :2])
        np.testing.assert_array_equal(out.numpy()[..., 5:], base[..., 5:])

    def test_open_ended_segment_replaced(self):
        base = np.arange(24, dtype=np.float32).reshape(2, 2, 6)
        seg = np.full((2, 2, 2), 7.0, dtype=np.float32)
        out = slice_util.scatter_last_dim_segment(
            paddle.to_tensor(base), paddle.to_tensor(seg), start=4
        )
        expected = base.copy()
        expected[..., 4:] = seg
        np.testing.assert_array_equal(out.numpy(), expected)

    def test_full_width_segment_is_the_segment(self):
        base = paddle.arange(12, dtype="float32").reshape([3, 4])
        seg = paddle.full([3, 4], 5.0, dtype="float32")
        out = slice_util.scatter_last_dim_segment(base, seg, start=0)
        np.testing.assert_array_equal(
            out.numpy(), np.full((3, 4), 5.0, "float32")
        )

    def test_segment_cast_to_buffer_dtype(self):
        base = np.arange(12, dtype=np.float32).reshape(3, 4)
        seg = np.array([[9], [9], [9]], dtype=np.int64)  # int -> must cast
        out = slice_util.scatter_last_dim_segment(
            paddle.to_tensor(base), paddle.to_tensor(seg), start=0, end=1
        )
        self.assertEqual(out.dtype, paddle.float32)
        expected = base.copy()
        expected[..., 0:1] = seg.astype(np.float32)
        np.testing.assert_array_equal(out.numpy(), expected)

    def test_none_segment_returns_tensor_unchanged(self):
        base = paddle.arange(12, dtype="float32").reshape([3, 4])
        out = slice_util.scatter_last_dim_segment(base, None, start=0)
        self.assertIs(out, base)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestForceUnitProbs(_AblationEnvMixin, unittest.TestCase):
    """``inspect_tensor_force_unit_probs`` -> all-ones only when the tag is live."""

    _TAG = "moe_act_quant_output"

    def setUp(self):
        paddle.set_device("cpu")

    def test_forces_ones_when_tag_enabled(self):
        self._apply_ablation_env(
            ABLATION_INSPECT_TENSOR="1", ABLATION_TAG_WHITELIST=self._TAG
        )
        probs = paddle.to_tensor([[0.2, 0.8], [0.5, 0.5]], dtype="float32")
        out = ffn_act.inspect_tensor_force_unit_probs(probs, self._TAG)
        np.testing.assert_array_equal(
            out.numpy(), np.ones((2, 2), dtype=np.float32)
        )

    def test_passthrough_when_probes_off(self):
        self._apply_ablation_env()  # disabled
        probs = paddle.to_tensor([[0.2, 0.8]], dtype="float32")
        out = ffn_act.inspect_tensor_force_unit_probs(probs, self._TAG)
        self.assertIs(out, probs)

    def test_passthrough_when_tag_filtered_out(self):
        # Probes on, but a whitelist that excludes this tag must NOT rewrite math.
        self._apply_ablation_env(
            ABLATION_INSPECT_TENSOR="1", ABLATION_TAG_WHITELIST="some_other_tag"
        )
        probs = paddle.to_tensor([[0.2, 0.8]], dtype="float32")
        out = ffn_act.inspect_tensor_force_unit_probs(probs, self._TAG)
        self.assertIs(out, probs)

    def test_none_probs_returns_none(self):
        self._apply_ablation_env(
            ABLATION_INSPECT_TENSOR="1", ABLATION_TAG_WHITELIST=self._TAG
        )
        self.assertIsNone(
            ffn_act.inspect_tensor_force_unit_probs(None, self._TAG)
        )


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestDequantDispatchedScaleNoneBranch(unittest.TestCase):
    """The kernel-free (``scale is None``) branch of ``dequant_dispatched_hidden_bf16``.

    The ``scale is not None`` path calls ``fused_act_dequant`` (fp8, GPU-only) and
    is deliberately left to a GPU suite -- only the None-scale fallbacks run here.
    """

    def setUp(self):
        paddle.set_device("cpu")

    def test_none_hidden_returns_none(self):
        self.assertIsNone(ffn_act.dequant_dispatched_hidden_bf16(None, None))

    def test_float32_passthrough_without_scale(self):
        hs = paddle.to_tensor([[1.0, 2.0]], dtype="float32")
        out = ffn_act.dequant_dispatched_hidden_bf16(hs, None)
        self.assertIs(out, hs)

    def test_bfloat16_passthrough_without_scale(self):
        hs = paddle.to_tensor([[1.0, 2.0]], dtype="bfloat16")
        out = ffn_act.dequant_dispatched_hidden_bf16(hs, None)
        self.assertIs(out, hs)

    def test_incomparable_dtype_without_scale_returns_none(self):
        hs = paddle.to_tensor([[1, 2]], dtype="int32")
        self.assertIsNone(ffn_act.dequant_dispatched_hidden_bf16(hs, None))


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestInspectTensorGateAndIO(_AblationEnvMixin, unittest.TestCase):
    """``inspect_tensor`` gate behavior plus a real ``.npy`` save/load override."""

    def setUp(self):
        paddle.set_device("cpu")

    def test_disabled_returns_same_object(self):
        self._apply_ablation_env()  # probes off
        t = paddle.to_tensor([1.0, 2.0], dtype="float32")
        self.assertIs(inspect_util.inspect_tensor("tag", 0, t), t)

    def test_none_tensor_returns_none(self):
        self._apply_ablation_env(ABLATION_INSPECT_TENSOR="1")
        self.assertIsNone(inspect_util.inspect_tensor("tag", 0, None))

    def test_filtered_tag_returns_same_object(self):
        self._apply_ablation_env(
            ABLATION_INSPECT_TENSOR="1", ABLATION_TAG_WHITELIST="other"
        )
        t = paddle.to_tensor([1.0, 2.0], dtype="float32")
        self.assertIs(inspect_util.inspect_tensor("mytag", 0, t), t)

    def test_pre_save_func_none_aborts(self):
        self._apply_ablation_env(ABLATION_INSPECT_TENSOR="1")
        t = paddle.to_tensor([1.0, 2.0], dtype="float32")
        out = inspect_util.inspect_tensor(
            "tag", 0, t, pre_save_func=lambda _x: None
        )
        self.assertIs(out, t)

    def test_load_without_dump_returns_same_object(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(_rmtree, tmp)
        self._apply_ablation_env(
            ABLATION_INSPECT_TENSOR="1", ABLATION_LOAD_TENSOR_PATH=tmp
        )
        t = paddle.to_tensor([1.0, 2.0], dtype="float32")
        # No file exists for this (rank, layer, tag): nothing to override.
        self.assertIs(inspect_util.inspect_tensor("absent", 0, t, load=True), t)

    def test_save_writes_expected_npy(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(_rmtree, tmp)
        self._apply_ablation_env(
            ABLATION_INSPECT_TENSOR="1", ABLATION_SAVE_TENSOR_PATH=tmp
        )
        content = np.array([[1.0, -2.0], [3.5, 4.0]], dtype=np.float32)
        t = paddle.to_tensor(content)
        inspect_util.inspect_tensor("probe", 2, t, save=True, load=False)
        # rank 0 (no dist), layer 2, tag probe.
        fpath = os.path.join(tmp, "rank_0", "layer_2", "probe.npy")
        self.assertTrue(os.path.exists(fpath))
        np.testing.assert_array_equal(np.load(fpath), content)

    def test_save_then_load_applies_override(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(_rmtree, tmp)
        self._apply_ablation_env(
            ABLATION_INSPECT_TENSOR="1",
            ABLATION_SAVE_TENSOR_PATH=tmp,
            ABLATION_LOAD_TENSOR_PATH=tmp,
        )
        dumped = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
        inspect_util.inspect_tensor(
            "ov", 1, paddle.to_tensor(dumped), save=True, load=False
        )
        # A live tensor with DIFFERENT content but the same shape: the override
        # must hand back the dumped values, not the live ones.
        live = paddle.to_tensor(
            np.array([[-1.0, -1.0], [-1.0, -1.0]], dtype=np.float32)
        )
        out = inspect_util.inspect_tensor("ov", 1, live, save=False, load=True)
        self.assertIsNot(out, live)
        np.testing.assert_array_equal(out.numpy(), dumped)

    def test_indexed_override_rewraps_container(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(_rmtree, tmp)
        self._apply_ablation_env(
            ABLATION_INSPECT_TENSOR="1",
            ABLATION_SAVE_TENSOR_PATH=tmp,
            ABLATION_LOAD_TENSOR_PATH=tmp,
        )
        dumped = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        bias0 = paddle.to_tensor([0.0], dtype="float32")
        inspect_util.inspect_tensor(
            "bundle",
            0,
            (paddle.to_tensor(dumped), bias0),
            index=0,
            save=True,
            load=False,
        )
        live_bias = paddle.to_tensor([9.0], dtype="float32")
        live = (
            paddle.to_tensor(np.array([[-5.0, -5.0, -5.0]], dtype=np.float32)),
            live_bias,
        )
        out = inspect_util.inspect_tensor(
            "bundle", 0, live, index=0, save=False, load=True
        )
        self.assertIsInstance(out, tuple)
        self.assertIsNot(out, live)
        np.testing.assert_array_equal(out[0].numpy(), dumped)
        # The non-probed slot (the bias) is carried through untouched.
        self.assertIs(out[1], live_bias)


if __name__ == "__main__":
    unittest.main()

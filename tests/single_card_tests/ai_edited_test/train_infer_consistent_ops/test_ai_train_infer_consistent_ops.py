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
"""Unit tests for the `paddlefleet.train_infer_consistent_ops` probe package.

The package is the training side of the train_infer_consistent_inspect (K3)
tensor probe. Every function in it is covered here, private helpers included:

* `inspect_util` -- the probe core: the ABLATION_* snapshot, the current-layer
  context, the stats/shape gate and the 7 stages of `inspect_tensor`.
* `permute` -- the canonical (token, expert) row order of the expert-contiguous
  grouped-GEMM buffer, and its inverse.
* `ffn_act` -- unit routing weights, fp8 dequant/requant and the inverses the
  expert-activation probes hand to `post_load_func`.

`inspect_tensor` is the whole probe surface, so the composite probes in the
network definition (the dispatched activation, the SwiGLU+quant output, the
fused gate logits) are
tested exactly as their call sites spell them: one `inspect_tensor(...)` with a
`pre_save_func` / `post_load_func` pair, and `index=` where the tensor travels
inside a tuple.

The last class covers the training-side call site that composes probe tags
(`MLP.inspect_name` + `get_current_layer()`), which is what makes a dump land
under `moe_shared_*` vs `dense_mlp_*`.

Note that the ABLATION_* configuration is snapshotted once at import, so every
test flips the environment, calls `refresh_env_cache()`, and restores both in
`tearDown` -- otherwise the probes would stay on for the rest of the suite.
"""

import hashlib
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import paddle

from paddlefleet.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from paddlefleet.train_infer_consistent_ops import inspect_util, permute
from paddlefleet.train_infer_consistent_ops.ffn_act import (
    _quant_blockwise,
    dequant_dispatched_hidden_bf16,
    inspect_tensor_force_unit_probs,
    requant_swiglu_output,
    scatter_dispatched_hidden_bf16,
)
from paddlefleet.train_infer_consistent_ops.inspect_util import (
    _as_f32_numpy,
    _load_shape_ok,
    _squeeze_shape,
    _stats,
    _with_element,
    get_current_layer,
    inspect_enabled,
    inspect_tensor,
    inspect_tensor_set_current_layer,
    refresh_env_cache,
)
from paddlefleet.train_infer_consistent_ops.permute import (
    canonical_rows,
    inspect_tensor_set_permute_index,
    scatter_canonical_rows,
)
from paddlefleet.train_infer_consistent_ops.slice_util import (
    last_dim_segment,
    scatter_last_dim_segment,
)
from paddlefleet.transformer.mlp import MLP
from paddlefleet.transformer.transformer_config import TransformerConfig
from paddlefleet.utils import init_method_normal, scaled_init_method_normal

ABLATION_KEYS = (
    "ABLATION_INSPECT_TENSOR",
    "ABLATION_TAG_WHITELIST",
    "ABLATION_TAG_BLACKLIST",
    "ABLATION_DUMP_SKIP_TAGS",
    "ABLATION_SAVE_TENSOR_PATH",
    "ABLATION_LOAD_TENSOR_PATH",
)


class ProbeEnvTestCase(unittest.TestCase):
    """Base class isolating the ABLATION_* snapshot and the published row map."""

    def setUp(self):
        self._env_backup = {k: os.environ.get(k) for k in ABLATION_KEYS}
        for key in ABLATION_KEYS:
            os.environ.pop(key, None)
        refresh_env_cache()
        permute._PERMUTE_INDEX = None
        self.tmpdir = tempfile.mkdtemp()
        self.save_dir = os.path.join(self.tmpdir, "save")
        self.load_dir = os.path.join(self.tmpdir, "load")

    def tearDown(self):
        for key, value in self._env_backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        refresh_env_cache()
        permute._PERMUTE_INDEX = None
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def enable(self, **env):
        """Switch the probes on, plus whatever else the test needs."""
        os.environ["ABLATION_INSPECT_TENSOR"] = "1"
        for key, value in env.items():
            os.environ[key] = value
        refresh_env_cache()

    def dump_path(self, root, tag, layer_idx):
        return os.path.join(root, "rank_0", f"layer_{layer_idx}", f"{tag}.npy")

    def write_dump(self, tag, layer_idx, array, root=None):
        """Drop a `.npy` where stage 6 looks for the other side's dump."""
        fpath = self.dump_path(root or self.load_dir, tag, layer_idx)
        os.makedirs(os.path.dirname(fpath), exist_ok=True)
        np.save(fpath, np.asarray(array, dtype=np.float32))
        return fpath

    def _buffer(self, dtype="float32"):
        """6-row expert-contiguous buffer, NaN in the alignment padding rows.

        NaN stands in for `paddle.empty`'s uninitialized padding: any helper that
        touches a padding row shows up as a NaN in the result.
        """
        nan = float("nan")
        return paddle.to_tensor(
            [
                [1.0, 1.0],
                [nan, nan],
                [3.0, 3.0],
                [nan, nan],
                [5.0, 5.0],
                [nan, nan],
            ],
            dtype=dtype,
        )

    def _index(self):
        """`[num_tokens=2, num_local_experts=2]` row map, -1 = pair not routed."""
        return paddle.to_tensor([[0, -1], [2, 4]], dtype="int32")


class TestEnvSnapshot(ProbeEnvTestCase):
    """Covers `refresh_env_cache` and `inspect_enabled`."""

    def test_refresh_env_cache_reads_every_variable(self):
        self.enable(
            ABLATION_TAG_WHITELIST="a,b,",
            ABLATION_TAG_BLACKLIST="c",
            ABLATION_DUMP_SKIP_TAGS="d,e",
            ABLATION_SAVE_TENSOR_PATH=self.save_dir,
            ABLATION_LOAD_TENSOR_PATH=self.load_dir,
        )
        self.assertTrue(inspect_util._ENABLED)
        # Empty entries from a trailing comma are dropped.
        self.assertEqual(inspect_util._WHITELIST, frozenset({"a", "b"}))
        self.assertEqual(inspect_util._BLACKLIST, frozenset({"c"}))
        self.assertEqual(inspect_util._DUMP_SKIP_TAGS, frozenset({"d", "e"}))
        self.assertEqual(inspect_util._SAVE_PATH, self.save_dir)
        self.assertEqual(inspect_util._LOAD_PATH, self.load_dir)

    def test_unset_variables_default_to_off(self):
        self.assertFalse(inspect_util._ENABLED)
        self.assertEqual(inspect_util._WHITELIST, frozenset())
        self.assertEqual(inspect_util._SAVE_PATH, "")

    def test_inspect_enabled_only_follows_the_snapshot(self):
        self.assertFalse(inspect_enabled())
        os.environ["ABLATION_INSPECT_TENSOR"] = "1"
        # In-process changes are invisible until the cache is refreshed.
        self.assertFalse(inspect_enabled())
        refresh_env_cache()
        self.assertTrue(inspect_enabled())


class TestCurrentLayer(ProbeEnvTestCase):
    """Covers `inspect_tensor_set_current_layer` and `get_current_layer`."""

    def tearDown(self):
        inspect_tensor_set_current_layer(None)
        super().tearDown()

    def test_published_layer_is_readable_back(self):
        inspect_tensor_set_current_layer(7)
        self.assertEqual(get_current_layer(), 7)

    def test_none_publishes_minus_one(self):
        inspect_tensor_set_current_layer(3)
        inspect_tensor_set_current_layer(None)
        self.assertEqual(get_current_layer(), -1)

    def test_write_happens_even_while_the_probes_are_off(self):
        """This entry point is deliberately ungated -- see its docstring."""
        self.assertFalse(inspect_enabled())
        inspect_tensor_set_current_layer(5)
        self.assertEqual(get_current_layer(), 5)


class TestStats(ProbeEnvTestCase):
    """Covers `_stats`."""

    def test_negative_zero_is_normalized_before_the_md5(self):
        pos = np.array([0.0, 1.5, -2.0], dtype=np.float32)
        neg = np.array([-0.0, 1.5, -2.0], dtype=np.float32)
        self.assertNotEqual(pos.tobytes(), neg.tobytes())
        self.assertEqual(_stats(pos), _stats(neg))

    def test_stats_values(self):
        abssum, absmax, md5 = _stats(np.array([0.5, -2.0], dtype=np.float32))
        self.assertAlmostEqual(abssum, 2.5, places=6)
        self.assertAlmostEqual(absmax, 2.0, places=6)
        self.assertEqual(
            md5,
            hashlib.md5(
                np.array([0.5, -2.0], dtype=np.float32).tobytes()
            ).hexdigest(),
        )

    def test_non_float_arrays_are_hashed_as_is(self):
        arr = np.array([[-3, 1], [0, 2]], dtype=np.int32)
        abssum, absmax, md5 = _stats(arr)
        self.assertEqual(abssum, 6.0)
        self.assertEqual(absmax, 3.0)
        self.assertEqual(md5, hashlib.md5(arr.tobytes()).hexdigest())

    def test_empty_array_reports_zero_absmax(self):
        abssum, absmax, _ = _stats(np.zeros([0], dtype=np.float32))
        self.assertEqual(abssum, 0.0)
        self.assertEqual(absmax, 0.0)


class TestShapeGate(ProbeEnvTestCase):
    """Covers `_squeeze_shape` and `_load_shape_ok`."""

    def test_squeeze_shape_drops_size_one_dims(self):
        self.assertEqual(_squeeze_shape([1, 11, 4096]), (11, 4096))
        self.assertEqual(_squeeze_shape([1, 1, 1]), ())

    def test_numel_mismatch_is_rejected(self):
        ok, reason = _load_shape_ok([11, 4096], [11, 2048])
        self.assertFalse(ok)
        self.assertIn("numel mismatch", reason)

    def test_identical_shapes_pass(self):
        self.assertEqual(
            _load_shape_ok([11, 4096], [11, 4096]), (True, "exact")
        )

    def test_leading_batch_dim_passes(self):
        ok, reason = _load_shape_ok([11, 4096], [1, 11, 4096])
        self.assertTrue(ok)
        self.assertIn("equal ignoring size-1 dims", reason)

    def test_same_row_count_regroup_passes(self):
        """The mHC stream fold: `[11,4,4096]` vs `[1,11,16384]`."""
        ok, reason = _load_shape_ok([11, 4, 4096], [1, 11, 16384])
        self.assertTrue(ok)
        self.assertIn("trailing dims regrouped", reason)

    def test_numel_collision_with_different_rows_is_rejected(self):
        """The dp-gathered-rows x tp-column-shard false positive."""
        ok, reason = _load_shape_ok([88, 3584], [1, 11, 28672])
        self.assertFalse(ok)
        self.assertIn("row count differs dump_rows=88 live_rows=11", reason)

    def test_degenerate_shape_branch(self):
        """Pins the defensive branch for a shape that squeezes to nothing.

        Unreachable with real tensor shapes: equal numel of 1 forces every dim on
        both sides to be 1, which the `equal ignoring size-1 dims` branch already
        catches. A placeholder dim is the only way in.
        """
        ok, reason = _load_shape_ok([1], [-1, -1])
        self.assertTrue(ok)
        self.assertIn("degenerate shape", reason)


class TestAsF32Numpy(ProbeEnvTestCase):
    """Covers `_as_f32_numpy`."""

    def test_bfloat16_becomes_a_float32_host_copy(self):
        arr = _as_f32_numpy(paddle.to_tensor([[1.5, -2.5]], dtype="bfloat16"))
        self.assertEqual(arr.dtype, np.float32)
        np.testing.assert_allclose(arr, [[1.5, -2.5]])


class TestInspectTensorGate(ProbeEnvTestCase):
    """Covers `inspect_tensor` stages 1-2 (gate and snapshot)."""

    def test_none_tensor_is_handed_back(self):
        self.enable()
        self.assertIsNone(inspect_tensor("tag", 0, None))

    def test_disabled_is_the_identity(self):
        tensor = paddle.to_tensor([1.0, 2.0])
        self.assertIs(inspect_tensor("tag", 0, tensor), tensor)

    def test_whitelist_keeps_only_its_tags(self):
        self.enable(
            ABLATION_TAG_WHITELIST="kept",
            ABLATION_SAVE_TENSOR_PATH=self.save_dir,
        )
        tensor = paddle.to_tensor([1.0, 2.0])
        self.assertIs(inspect_tensor("dropped", 0, tensor, save=True), tensor)
        self.assertFalse(
            os.path.exists(self.dump_path(self.save_dir, "dropped", 0))
        )
        inspect_tensor("kept", 0, tensor, save=True)
        self.assertTrue(
            os.path.exists(self.dump_path(self.save_dir, "kept", 0))
        )

    def test_blacklist_drops_its_tags(self):
        self.enable(
            ABLATION_TAG_BLACKLIST="banned",
            ABLATION_SAVE_TENSOR_PATH=self.save_dir,
        )
        tensor = paddle.to_tensor([1.0, 2.0])
        self.assertIs(inspect_tensor("banned", 0, tensor, save=True), tensor)
        self.assertFalse(
            os.path.exists(self.dump_path(self.save_dir, "banned", 0))
        )

    def test_pre_save_func_stays_lazy_while_disabled(self):
        def boom(_):
            raise AssertionError(
                "pre_save_func must not run when probes are off"
            )

        tensor = paddle.to_tensor([1.0])
        self.assertIs(
            inspect_tensor("tag", 0, tensor, pre_save_func=boom), tensor
        )

    def test_pre_save_func_returning_none_aborts_the_probe(self):
        self.enable(ABLATION_SAVE_TENSOR_PATH=self.save_dir)
        tensor = paddle.to_tensor([1.0, 2.0])
        result = inspect_tensor(
            "tag", 0, tensor, save=True, pre_save_func=lambda _: None
        )
        self.assertIs(result, tensor)
        self.assertFalse(
            os.path.exists(self.dump_path(self.save_dir, "tag", 0))
        )

    def test_uncastable_dtype_only_logs_and_never_breaks_the_forward(self):
        class _Uncastable:
            shape = [2, 2]
            dtype = "float8_e4m3fn-ish"

            def astype(self, dtype):
                raise RuntimeError("cast refused")

        self.enable(ABLATION_SAVE_TENSOR_PATH=self.save_dir)
        stub = _Uncastable()
        self.assertIs(inspect_tensor("weird", 0, stub, save=True), stub)
        # Stage 5 is gated on the host copy, so nothing is dumped either.
        self.assertFalse(
            os.path.exists(self.dump_path(self.save_dir, "weird", 0))
        )


class TestInspectTensorSave(ProbeEnvTestCase):
    """Covers `inspect_tensor` stage 5 (save)."""

    def test_snapshot_is_dumped_and_the_live_tensor_returned(self):
        self.enable(ABLATION_SAVE_TENSOR_PATH=self.save_dir)
        tensor = paddle.to_tensor([[1.0, -2.0]])
        self.assertIs(inspect_tensor("tag", 3, tensor, save=True), tensor)
        arr = np.load(self.dump_path(self.save_dir, "tag", 3))
        self.assertEqual(arr.dtype, np.float32)
        np.testing.assert_allclose(arr, [[1.0, -2.0]])

    def test_the_dumped_bytes_are_the_pre_save_view(self):
        self.enable(ABLATION_SAVE_TENSOR_PATH=self.save_dir)
        tensor = paddle.to_tensor([[1.0, -2.0]])
        result = inspect_tensor(
            "tag", 0, tensor, save=True, pre_save_func=lambda x: x * 2
        )
        self.assertIs(result, tensor)
        np.testing.assert_allclose(
            np.load(self.dump_path(self.save_dir, "tag", 0)), [[2.0, -4.0]]
        )

    def test_save_off_writes_nothing(self):
        """`save=False` is this side's default -- the training side only loads."""
        self.enable(ABLATION_SAVE_TENSOR_PATH=self.save_dir)
        inspect_tensor("tag", 0, paddle.to_tensor([1.0]))
        self.assertFalse(
            os.path.exists(self.dump_path(self.save_dir, "tag", 0))
        )

    def test_dump_skip_tags_blocks_both_save_and_load(self):
        self.write_dump("tag", 0, [[9.0, 9.0]])
        self.enable(
            ABLATION_SAVE_TENSOR_PATH=self.save_dir,
            ABLATION_LOAD_TENSOR_PATH=self.load_dir,
            ABLATION_DUMP_SKIP_TAGS="tag",
        )
        tensor = paddle.to_tensor([[1.0, 2.0]])
        self.assertIs(inspect_tensor("tag", 0, tensor, save=True), tensor)
        self.assertFalse(
            os.path.exists(self.dump_path(self.save_dir, "tag", 0))
        )


class TestInspectTensorLoad(ProbeEnvTestCase):
    """Covers `inspect_tensor` stages 6-7 (load and return)."""

    def test_missing_dump_returns_the_live_tensor(self):
        self.enable(ABLATION_LOAD_TENSOR_PATH=self.load_dir)
        tensor = paddle.to_tensor([[1.0, 2.0]])
        self.assertIs(inspect_tensor("tag", 0, tensor), tensor)

    def test_load_off_ignores_an_existing_dump(self):
        self.write_dump("tag", 0, [[9.0, 9.0]])
        self.enable(ABLATION_LOAD_TENSOR_PATH=self.load_dir)
        tensor = paddle.to_tensor([[1.0, 2.0]])
        self.assertIs(inspect_tensor("tag", 0, tensor, load=False), tensor)

    def test_dump_overrides_the_tensor(self):
        self.write_dump("tag", 2, [[9.0, -8.0]])
        self.enable(ABLATION_LOAD_TENSOR_PATH=self.load_dir)
        tensor = paddle.to_tensor([[1.0, 2.0]])
        result = inspect_tensor("tag", 2, tensor)
        self.assertIsNot(result, tensor)
        np.testing.assert_allclose(result.numpy(), [[9.0, -8.0]])

    def test_squeezable_dump_is_reshaped_into_the_live_shape(self):
        self.write_dump("tag", 0, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        self.enable(ABLATION_LOAD_TENSOR_PATH=self.load_dir)
        tensor = paddle.zeros([1, 2, 3], dtype="float32")
        result = inspect_tensor("tag", 0, tensor)
        self.assertEqual(list(result.shape), [1, 2, 3])
        np.testing.assert_allclose(
            result.numpy().reshape(-1), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        )

    def test_dump_is_cast_back_to_the_live_dtype(self):
        self.write_dump("tag", 0, [[1.5, -2.5]])
        self.enable(ABLATION_LOAD_TENSOR_PATH=self.load_dir)
        tensor = paddle.zeros([1, 2], dtype="bfloat16")
        result = inspect_tensor("tag", 0, tensor)
        self.assertEqual(result.dtype, paddle.bfloat16)
        np.testing.assert_allclose(_as_f32_numpy(result), [[1.5, -2.5]])

    def test_shape_gate_skips_a_numel_collision(self):
        self.write_dump("tag", 0, np.ones([8, 2], dtype=np.float32))
        self.enable(ABLATION_LOAD_TENSOR_PATH=self.load_dir)
        tensor = paddle.zeros([2, 8], dtype="float32")
        self.assertIs(inspect_tensor("tag", 0, tensor), tensor)

    def test_post_load_func_shapes_the_return_value(self):
        self.write_dump("tag", 0, [[9.0, -8.0]])
        self.enable(ABLATION_LOAD_TENSOR_PATH=self.load_dir)
        tensor = paddle.to_tensor([[1.0, 2.0]])
        result = inspect_tensor(
            "tag", 0, tensor, post_load_func=lambda loaded: ("wrapped", loaded)
        )
        self.assertEqual(result[0], "wrapped")
        np.testing.assert_allclose(result[1].numpy(), [[9.0, -8.0]])

    def test_post_load_func_is_skipped_when_nothing_was_loaded(self):
        self.enable(ABLATION_LOAD_TENSOR_PATH=self.load_dir)
        tensor = paddle.to_tensor([[1.0, 2.0]])
        result = inspect_tensor(
            "tag", 0, tensor, post_load_func=lambda loaded: "must not run"
        )
        self.assertIs(result, tensor)

    def test_load_compares_against_the_pre_save_view(self):
        """The dump is matched against the snapshot, not the live tensor."""
        self.write_dump("tag", 0, [[2.0, 4.0]])
        self.enable(ABLATION_LOAD_TENSOR_PATH=self.load_dir)
        tensor = paddle.to_tensor([[1.0, 2.0]])
        result = inspect_tensor(
            "tag",
            0,
            tensor,
            pre_save_func=lambda x: x * 2,
            post_load_func=lambda loaded: loaded / 2,
        )
        np.testing.assert_allclose(result.numpy(), [[1.0, 2.0]])


class TestInspectTensorContainers(ProbeEnvTestCase):
    """Covers `_with_element` and `inspect_tensor(..., index=...)`.

    Fused blocks hand their output around as a `(tensor, bias)` tuple, and some
    stages hand back several buffers at once as a list or a dict. `index` is how
    the one comparable tensor is reached and put back, so the call site never
    unpacks anything by hand.
    """

    def test_with_element_replaces_one_tuple_element(self):
        self.assertEqual(_with_element((1, 2, 3), 1, "x"), (1, "x", 3))
        self.assertEqual(_with_element((1, 2), 0, "x"), ("x", 2))

    def test_with_element_copies_lists_and_dicts(self):
        """The original container is never written into -- the caller still holds it."""
        original_list = [1, 2, 3]
        self.assertEqual(_with_element(original_list, 1, "x"), [1, "x", 3])
        self.assertEqual(original_list, [1, 2, 3])
        original_dict = {"a": 1, "b": 2}
        self.assertEqual(
            _with_element(original_dict, "b", "x"), {"a": 1, "b": "x"}
        )
        self.assertEqual(original_dict, {"a": 1, "b": 2})

    def test_a_plain_tensor_ignores_index(self):
        self.enable(ABLATION_SAVE_TENSOR_PATH=self.save_dir)
        tensor = paddle.to_tensor([[1.0, 2.0]])
        self.assertIs(
            inspect_tensor("tag", 0, tensor, index=0, save=True), tensor
        )
        np.testing.assert_allclose(
            np.load(self.dump_path(self.save_dir, "tag", 0)), [[1.0, 2.0]]
        )

    def test_index_selects_which_bundle_element_is_dumped(self):
        self.enable(ABLATION_SAVE_TENSOR_PATH=self.save_dir)
        bundle = (paddle.to_tensor([[1.0, 2.0]]), paddle.to_tensor([3.0, 4.0]))
        self.assertIs(
            inspect_tensor("tag", 0, bundle, index=0, save=True), bundle
        )
        np.testing.assert_allclose(
            np.load(self.dump_path(self.save_dir, "tag", 0)), [[1.0, 2.0]]
        )
        inspect_tensor("other", 0, bundle, index=1, save=True)
        np.testing.assert_allclose(
            np.load(self.dump_path(self.save_dir, "other", 0)), [3.0, 4.0]
        )

    def test_no_index_probes_the_container_itself(self):
        """Without `index` the whole container is what `pre_save_func` receives."""
        self.enable(ABLATION_SAVE_TENSOR_PATH=self.save_dir)
        bundle = (
            paddle.to_tensor([[1.0, 2.0]]),
            paddle.to_tensor([[3.0, 4.0]]),
        )
        inspect_tensor(
            "tag",
            0,
            bundle,
            save=True,
            pre_save_func=lambda pair: paddle.concat(pair, axis=-1),
        )
        np.testing.assert_allclose(
            np.load(self.dump_path(self.save_dir, "tag", 0)),
            [[1.0, 2.0, 3.0, 4.0]],
        )

    def test_an_absent_bias_aborts_the_probe(self):
        self.enable(ABLATION_SAVE_TENSOR_PATH=self.save_dir)
        bundle = (paddle.to_tensor([[1.0, 2.0]]), None)
        self.assertIs(
            inspect_tensor("tag", 0, bundle, index=1, save=True), bundle
        )
        self.assertFalse(
            os.path.exists(self.dump_path(self.save_dir, "tag", 0))
        )

    def test_a_loaded_dump_replaces_only_the_probed_element(self):
        self.write_dump("tag", 0, [7.0, 8.0])
        self.enable(ABLATION_LOAD_TENSOR_PATH=self.load_dir)
        bundle = (paddle.to_tensor([[1.0, 2.0]]), paddle.to_tensor([3.0, 4.0]))
        result = inspect_tensor("tag", 0, bundle, index=1)
        self.assertIsNot(result, bundle)
        self.assertIs(result[0], bundle[0])
        np.testing.assert_allclose(result[1].numpy(), [7.0, 8.0])

    def test_a_list_bundle_comes_back_as_a_fresh_list(self):
        self.write_dump("tag", 0, [7.0, 8.0])
        self.enable(ABLATION_LOAD_TENSOR_PATH=self.load_dir)
        bundle = [paddle.to_tensor([[1.0, 2.0]]), paddle.to_tensor([3.0, 4.0])]
        result = inspect_tensor("tag", 0, bundle, index=1)
        self.assertIsInstance(result, list)
        self.assertIsNot(result, bundle)
        np.testing.assert_allclose(result[1].numpy(), [7.0, 8.0])
        # The bundle the caller still holds is untouched.
        np.testing.assert_allclose(bundle[1].numpy(), [3.0, 4.0])

    def test_a_dict_bundle_is_keyed_by_index(self):
        self.write_dump("tag", 0, [7.0, 8.0])
        self.enable(ABLATION_LOAD_TENSOR_PATH=self.load_dir)
        bundle = {
            "out": paddle.to_tensor([[1.0, 2.0]]),
            "bias": paddle.to_tensor([3.0, 4.0]),
        }
        result = inspect_tensor("tag", 0, bundle, index="bias")
        self.assertIsInstance(result, dict)
        self.assertIs(result["out"], bundle["out"])
        np.testing.assert_allclose(result["bias"].numpy(), [7.0, 8.0])


class TestCanonicalRows(ProbeEnvTestCase):
    """Covers `canonical_rows`."""

    def test_no_published_index_dumps_nothing(self):
        self.assertIsNone(canonical_rows(self._buffer()))

    def test_none_buffer_dumps_nothing(self):
        self.assertIsNone(canonical_rows(None, self._index()))

    def test_explicit_index_gathers_and_masks(self):
        canon = canonical_rows(self._buffer(), self._index())
        np.testing.assert_allclose(
            canon.numpy(), [[1.0, 1.0], [0.0, 0.0], [3.0, 3.0], [5.0, 5.0]]
        )

    def test_padding_rows_never_reach_the_result(self):
        canon = canonical_rows(self._buffer(), self._index())
        self.assertFalse(bool(paddle.isnan(canon).any()))

    def test_published_index_is_the_default(self):
        self.enable()
        inspect_tensor_set_permute_index(self._index())
        canon = canonical_rows(self._buffer())
        np.testing.assert_allclose(
            canon.numpy(), [[1.0, 1.0], [0.0, 0.0], [3.0, 3.0], [5.0, 5.0]]
        )

    def test_buffer_dtype_is_preserved(self):
        canon = canonical_rows(self._buffer("bfloat16"), self._index())
        self.assertEqual(canon.dtype, paddle.bfloat16)


class TestScatterCanonicalRows(ProbeEnvTestCase):
    """Covers `scatter_canonical_rows`."""

    def test_none_canon_leaves_the_buffer_alone(self):
        buf = self._buffer()
        self.assertIs(scatter_canonical_rows(buf, None, self._index()), buf)

    def test_no_published_index_leaves_the_buffer_alone(self):
        buf = self._buffer()
        canon = paddle.zeros([4, 2], dtype="float32")
        self.assertIs(scatter_canonical_rows(buf, canon), buf)

    def test_only_indexed_rows_are_written(self):
        buf = self._buffer()
        canon = paddle.to_tensor(
            [[10.0, 10.0], [99.0, 99.0], [30.0, 30.0], [50.0, 50.0]],
            dtype="float32",
        )
        out = scatter_canonical_rows(buf, canon, self._index())
        arr = out.numpy()
        np.testing.assert_allclose(arr[0], [10.0, 10.0])
        np.testing.assert_allclose(arr[2], [30.0, 30.0])
        np.testing.assert_allclose(arr[4], [50.0, 50.0])
        # The -1 pair's canonical row (99) is dropped, padding rows stay untouched.
        self.assertTrue(np.isnan(arr[[1, 3, 5]]).all())

    def test_round_trip_through_canonical_order(self):
        buf = self._buffer()
        index = self._index()
        out = scatter_canonical_rows(buf, canonical_rows(buf, index), index)
        np.testing.assert_allclose(
            out.numpy()[[0, 2, 4]], [[1.0, 1.0], [3.0, 3.0], [5.0, 5.0]]
        )

    def test_published_index_is_the_default(self):
        self.enable()
        inspect_tensor_set_permute_index(self._index())
        buf = self._buffer()
        canon = paddle.full([4, 2], 7.0, dtype="float32")
        np.testing.assert_allclose(
            scatter_canonical_rows(buf, canon).numpy()[0], [7.0, 7.0]
        )

    def test_dtype_is_pinned_back_to_the_buffer(self):
        """`paddle.scatter` may widen, but the fp8 quant kernels want bf16."""
        buf = self._buffer("bfloat16")
        canon = paddle.full([4, 2], 7.0, dtype="float32")
        out = scatter_canonical_rows(buf, canon, self._index())
        self.assertEqual(out.dtype, paddle.bfloat16)


class TestSetPermuteIndex(ProbeEnvTestCase):
    """Covers `inspect_tensor_set_permute_index`."""

    def test_nothing_is_published_while_the_probes_are_off(self):
        inspect_tensor_set_permute_index(self._index())
        self.assertIsNone(permute._PERMUTE_INDEX)

    def test_index_is_published_when_enabled(self):
        self.enable()
        index = self._index()
        inspect_tensor_set_permute_index(index)
        self.assertIs(permute._PERMUTE_INDEX, index)


class TestDispatchedHiddenProbe(ProbeEnvTestCase):
    """Covers the `moe_dispatched_hidden` probe as `fp8_utils.py` spells it.

    The activation reaching the grouped GEMM is fp8 + a blockwise scale, so the
    comparable view is `canonical_rows(dequant_dispatched_hidden_bf16(...))`; a
    loaded dump is bf16, which is why `post_load_func` drops the scale.
    """

    def probe(self, hs_out, scale, layer_idx):
        """The call site, verbatim."""
        return inspect_tensor(
            "moe_dispatched_hidden",
            layer_idx,
            (hs_out, scale),
            pre_save_func=lambda pair: canonical_rows(
                dequant_dispatched_hidden_bf16(*pair)
            ),
            post_load_func=lambda canon: (
                scatter_dispatched_hidden_bf16(hs_out, scale, canon),
                None,
            ),
        )

    def test_disabled_is_the_identity(self):
        buf = self._buffer()
        hs_out, scale = self.probe(buf, None, 0)
        self.assertIs(hs_out, buf)
        self.assertIsNone(scale)

    def test_no_published_index_hands_the_inputs_back(self):
        self.enable(ABLATION_LOAD_TENSOR_PATH=self.load_dir)
        buf = self._buffer()
        hs_out, scale = self.probe(buf, None, 0)
        self.assertIs(hs_out, buf)
        self.assertIsNone(scale)

    def test_an_uncomparable_dtype_hands_the_inputs_back(self):
        """`dequant_dispatched_hidden_bf16` gives up -> `canonical_rows` sees None."""
        self.enable(ABLATION_LOAD_TENSOR_PATH=self.load_dir)
        inspect_tensor_set_permute_index(self._index())
        hs_out = paddle.zeros([6, 2], dtype="int32")
        result = self.probe(hs_out, None, 0)
        self.assertIs(result[0], hs_out)
        self.assertIsNone(result[1])

    def test_no_dump_hands_the_fp8_pair_back_unchanged(self):
        self.enable(ABLATION_LOAD_TENSOR_PATH=self.load_dir)
        inspect_tensor_set_permute_index(self._index())
        buf = self._buffer()
        hs_out, scale = self.probe(buf, None, 1)
        self.assertIs(hs_out, buf)
        self.assertIsNone(scale)

    def test_a_loaded_dump_replaces_the_rows_and_drops_the_scale(self):
        self.write_dump(
            "moe_dispatched_hidden",
            1,
            [[10.0, 10.0], [0.0, 0.0], [30.0, 30.0], [50.0, 50.0]],
        )
        self.enable(ABLATION_LOAD_TENSOR_PATH=self.load_dir)
        inspect_tensor_set_permute_index(self._index())
        buf = self._buffer()
        hs_out, scale = self.probe(buf, None, 1)
        self.assertIsNone(scale)
        self.assertIsNot(hs_out, buf)
        arr = hs_out.numpy()
        np.testing.assert_allclose(
            arr[[0, 2, 4]], [[10.0, 10.0], [30.0, 30.0], [50.0, 50.0]]
        )
        # Padding rows are not part of the index, so the load never wrote them.
        self.assertTrue(np.isnan(arr[[1, 3, 5]]).all())


class TestForceUnitProbs(ProbeEnvTestCase):
    """Covers `inspect_tensor_force_unit_probs`.

    The weights are only forced while the probe that needs it is live, so the tag
    filters gate the rewrite exactly as they gate the probe itself.
    """

    TAG = "moe_act_quant_output"

    def test_none_probs_pass_through(self):
        self.enable()
        self.assertIsNone(inspect_tensor_force_unit_probs(None, self.TAG))

    def test_disabled_is_the_identity(self):
        probs = paddle.to_tensor([[0.25, 0.75]])
        self.assertIs(inspect_tensor_force_unit_probs(probs, self.TAG), probs)

    def test_enabled_forces_all_ones(self):
        self.enable()
        probs = paddle.to_tensor([[0.25, 0.75]], dtype="bfloat16")
        out = inspect_tensor_force_unit_probs(probs, self.TAG)
        self.assertIsNot(out, probs)
        self.assertEqual(out.dtype, paddle.bfloat16)
        np.testing.assert_allclose(out.astype("float32").numpy(), [[1.0, 1.0]])

    def test_a_whitelist_holding_the_tag_still_forces(self):
        self.enable(ABLATION_TAG_WHITELIST=f"other,{self.TAG}")
        probs = paddle.to_tensor([[0.25, 0.75]])
        out = inspect_tensor_force_unit_probs(probs, self.TAG)
        np.testing.assert_allclose(out.numpy(), [[1.0, 1.0]])

    def test_a_whitelist_without_the_tag_leaves_the_math_alone(self):
        self.enable(ABLATION_TAG_WHITELIST="mla_query_after_rope_pe")
        probs = paddle.to_tensor([[0.25, 0.75]])
        self.assertIs(inspect_tensor_force_unit_probs(probs, self.TAG), probs)

    def test_a_blacklisted_tag_leaves_the_math_alone(self):
        self.enable(ABLATION_TAG_BLACKLIST=self.TAG)
        probs = paddle.to_tensor([[0.25, 0.75]])
        self.assertIs(inspect_tensor_force_unit_probs(probs, self.TAG), probs)


class TestDequantDispatchedHidden(ProbeEnvTestCase):
    """Covers `dequant_dispatched_hidden_bf16`."""

    def test_none_activation_gives_up(self):
        self.assertIsNone(dequant_dispatched_hidden_bf16(None, None))

    def test_bf16_without_a_scale_is_already_comparable(self):
        hs = paddle.zeros([2, 2], dtype="bfloat16")
        self.assertIs(dequant_dispatched_hidden_bf16(hs, None), hs)

    def test_float32_without_a_scale_is_already_comparable(self):
        hs = paddle.zeros([2, 2], dtype="float32")
        self.assertIs(dequant_dispatched_hidden_bf16(hs, None), hs)

    def test_other_dtypes_without_a_scale_give_up(self):
        hs = paddle.zeros([2, 2], dtype="int32")
        self.assertIsNone(dequant_dispatched_hidden_bf16(hs, None))

    def test_a_scale_goes_through_fused_act_dequant(self):
        hs = paddle.zeros([2, 2], dtype="int32")
        scale = paddle.ones([2, 1], dtype="float32")
        sentinel = paddle.full([2, 2], 4.0, dtype="bfloat16")
        with patch(
            "paddle.incubate.nn.functional.fused_act_dequant",
            return_value=sentinel,
        ) as dequant:
            self.assertIs(dequant_dispatched_hidden_bf16(hs, scale), sentinel)
        dequant.assert_called_once_with(hs, scale)


class TestQuantBlockwise(ProbeEnvTestCase):
    """Covers `_quant_blockwise`."""

    def test_forward_recipe_is_passed_through_and_the_scale_round_trips(self):
        x = paddle.ones([2, 4], dtype="bfloat16")
        fake_out = paddle.zeros([2, 4], dtype="int8")
        fake_scale = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
        with patch(
            "paddle.incubate.nn.functional.fp8_quant_blockwise",
            return_value=(fake_out, fake_scale),
        ) as quant:
            out, out_scale = _quant_blockwise(x, use_ue8m0=True)
        self.assertIs(out, fake_out)
        # The double transpose only re-lays-out memory, values are unchanged.
        np.testing.assert_allclose(out_scale.numpy(), fake_scale.numpy())
        args, kwargs = quant.call_args
        self.assertIs(args[0], x)
        self.assertEqual(kwargs["quant_method"], "1x128")
        self.assertTrue(kwargs["using_pow2_scale"])
        self.assertTrue(kwargs["using_ue8m0_scale"])
        self.assertFalse(kwargs["output_scale_transpose"])
        self.assertFalse(kwargs["input_transpose"])

    def test_use_ue8m0_is_forwarded(self):
        fake_scale = paddle.ones([1, 1], dtype="float32")
        with patch(
            "paddle.incubate.nn.functional.fp8_quant_blockwise",
            return_value=(paddle.zeros([1, 1]), fake_scale),
        ) as quant:
            _quant_blockwise(
                paddle.ones([1, 1], dtype="bfloat16"), use_ue8m0=False
            )
        self.assertFalse(quant.call_args.kwargs["using_ue8m0_scale"])


class TestScatterDispatchedHiddenBf16(ProbeEnvTestCase):
    """Covers `scatter_dispatched_hidden_bf16`, the dispatched-activation inverse."""

    def test_canonical_rows_land_back_in_the_live_layout(self):
        self.enable()
        inspect_tensor_set_permute_index(self._index())
        buf = self._buffer()
        canon = paddle.to_tensor(
            [[10.0, 10.0], [0.0, 0.0], [30.0, 30.0], [50.0, 50.0]]
        )
        out = scatter_dispatched_hidden_bf16(buf, None, canon)
        arr = out.numpy()
        np.testing.assert_allclose(
            arr[[0, 2, 4]], [[10.0, 10.0], [30.0, 30.0], [50.0, 50.0]]
        )
        self.assertTrue(np.isnan(arr[[1, 3, 5]]).all())

    def test_the_bf16_view_is_rebuilt_from_the_fp8_pair(self):
        """The dequant is re-run here rather than carried over from `pre_save_func`."""
        self.enable()
        inspect_tensor_set_permute_index(self._index())
        buf = self._buffer()
        scale = paddle.ones([6, 1], dtype="float32")
        with patch(
            "paddle.incubate.nn.functional.fused_act_dequant",
            side_effect=lambda hs, s: hs,
        ) as dequant:
            scatter_dispatched_hidden_bf16(buf, scale, paddle.zeros([4, 2]))
        self.assertEqual(dequant.call_count, 1)
        self.assertIs(dequant.call_args.args[0], buf)
        self.assertIs(dequant.call_args.args[1], scale)


class TestSwigluQuantOutputProbe(ProbeEnvTestCase):
    """Covers the `moe_act_quant_output` probe as `fp8_utils.py` spells it.

    Same comparable view as the dispatched activation, but the down GEMM wants
    (fp8, 1x128 block scale) back, so `post_load_func` is `requant_swiglu_output`
    -- the forward's own quant recipe rather than a hand-built scale layout.
    """

    def probe(self, o2_fp8, o2_scale, layer_idx, use_ue8m0):
        """The call site, verbatim."""
        return inspect_tensor(
            "moe_act_quant_output",
            layer_idx,
            (o2_fp8, o2_scale),
            pre_save_func=lambda pair: canonical_rows(
                dequant_dispatched_hidden_bf16(*pair)
            ),
            post_load_func=lambda canon: requant_swiglu_output(
                o2_fp8, o2_scale, canon, use_ue8m0
            ),
        )

    def test_disabled_is_the_identity(self):
        o2, scale = paddle.zeros([6, 2], dtype="float32"), None
        result = self.probe(o2, scale, 0, False)
        self.assertIs(result[0], o2)
        self.assertIsNone(result[1])

    def test_no_published_index_hands_the_inputs_back(self):
        self.enable(ABLATION_LOAD_TENSOR_PATH=self.load_dir)
        o2 = self._buffer()
        result = self.probe(o2, None, 0, False)
        self.assertIs(result[0], o2)

    def test_an_uncomparable_dtype_hands_the_inputs_back(self):
        self.enable(ABLATION_LOAD_TENSOR_PATH=self.load_dir)
        inspect_tensor_set_permute_index(self._index())
        o2 = paddle.zeros([6, 2], dtype="int32")
        result = self.probe(o2, None, 0, False)
        self.assertIs(result[0], o2)
        self.assertIsNone(result[1])

    def test_no_dump_hands_the_fp8_pair_back_unchanged(self):
        self.enable(ABLATION_LOAD_TENSOR_PATH=self.load_dir)
        inspect_tensor_set_permute_index(self._index())
        o2, scale = self._buffer(), paddle.ones([6, 1], dtype="float32")
        with patch(
            "paddle.incubate.nn.functional.fused_act_dequant",
            side_effect=lambda hs, s: hs,
        ):
            result = self.probe(o2, scale, 1, False)
        self.assertIs(result[0], o2)
        self.assertIs(result[1], scale)

    def test_a_loaded_dump_is_requantized_with_the_forward_recipe(self):
        self.write_dump(
            "moe_act_quant_output",
            1,
            [[10.0, 10.0], [0.0, 0.0], [30.0, 30.0], [50.0, 50.0]],
        )
        self.enable(ABLATION_LOAD_TENSOR_PATH=self.load_dir)
        inspect_tensor_set_permute_index(self._index())
        o2 = self._buffer()
        fake_out = paddle.zeros([6, 2], dtype="int8")
        fake_scale = paddle.ones([6, 1], dtype="float32")
        with patch(
            "paddle.incubate.nn.functional.fp8_quant_blockwise",
            return_value=(fake_out, fake_scale),
        ) as quant:
            out, out_scale = self.probe(o2, None, 1, True)
        self.assertIs(out, fake_out)
        np.testing.assert_allclose(out_scale.numpy(), fake_scale.numpy())
        # What gets re-quantized is the dump scattered back into the live layout.
        requantized = quant.call_args.args[0].numpy()
        np.testing.assert_allclose(
            requantized[[0, 2, 4]], [[10.0, 10.0], [30.0, 30.0], [50.0, 50.0]]
        )
        self.assertTrue(np.isnan(requantized[[1, 3, 5]]).all())
        self.assertTrue(quant.call_args.kwargs["using_ue8m0_scale"])


class TestFusedGateLogitsProbe(ProbeEnvTestCase):
    """Covers the `moe_gate_fused_logits` probe as `moe_router.py` spells it.

    The two `[T, E]` gate views of this side have to be presented as the single
    fused `[T, 2E]` tensor the inference side dumps, and a loaded dump has to be
    split back into *both* views -- returning only view 0 would silently downgrade
    the probe to a print for view 1. Passing both views in as one tuple is what
    lets `post_load_func` hand both back.
    """

    TAG = "moe_gate_fused_logits"

    def setUp(self):
        super().setUp()
        self.view_0 = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]])
        self.view_1 = paddle.to_tensor([[5.0, 6.0], [7.0, 8.0]])

    def probe(self, layer_idx, logits_0, logits_1):
        """The call site, verbatim."""
        return inspect_tensor(
            self.TAG,
            layer_idx,
            (logits_0, logits_1),
            pre_save_func=lambda views: paddle.concat(views, axis=-1),
            post_load_func=lambda fused: (
                fused[:, : logits_0.shape[-1]],
                fused[:, logits_0.shape[-1] :],
            ),
        )

    def test_disabled_is_the_identity(self):
        out_0, out_1 = self.probe(0, self.view_0, self.view_1)
        self.assertIs(out_0, self.view_0)
        self.assertIs(out_1, self.view_1)

    def test_no_dump_hands_both_views_back_unchanged(self):
        self.enable(ABLATION_LOAD_TENSOR_PATH=self.load_dir)
        out_0, out_1 = self.probe(0, self.view_0, self.view_1)
        self.assertIs(out_0, self.view_0)
        self.assertIs(out_1, self.view_1)

    def test_the_probed_snapshot_is_the_fused_concat_not_view_0(self):
        """A `[T, E]` dump (view 0's own shape) is rejected by the shape gate.

        The call site passes no `save` flag -- this side only loads -- so the shape
        the load is matched against is the only observable proof that
        `pre_save_func` built the fused `[T, 2E]` view.
        """
        self.write_dump(self.TAG, 4, np.zeros([2, 2]))
        self.enable(ABLATION_LOAD_TENSOR_PATH=self.load_dir)
        out_0, out_1 = self.probe(4, self.view_0, self.view_1)
        self.assertIs(out_0, self.view_0)
        self.assertIs(out_1, self.view_1)

    def test_a_loaded_dump_replaces_both_views(self):
        """The dump swaps the halves, so a half landing in the wrong view shows."""
        self.write_dump(
            self.TAG, 3, [[5.0, 6.0, 1.0, 2.0], [7.0, 8.0, 3.0, 4.0]]
        )
        self.enable(ABLATION_LOAD_TENSOR_PATH=self.load_dir)
        out_0, out_1 = self.probe(3, self.view_0, self.view_1)
        self.assertIsNot(out_0, self.view_0)
        self.assertIsNot(out_1, self.view_1)
        np.testing.assert_allclose(out_0.numpy(), [[5.0, 6.0], [7.0, 8.0]])
        np.testing.assert_allclose(out_1.numpy(), [[1.0, 2.0], [3.0, 4.0]])

    def test_the_split_point_comes_from_view_0s_width(self):
        """One expert per view: the dump is `[T, 2]` and splits down the middle."""
        self.write_dump(self.TAG, 0, [[9.0, 8.0], [7.0, 6.0]])
        self.enable(ABLATION_LOAD_TENSOR_PATH=self.load_dir)
        view_0 = paddle.to_tensor([[1.0], [2.0]])
        view_1 = paddle.to_tensor([[3.0], [4.0]])
        out_0, out_1 = self.probe(0, view_0, view_1)
        np.testing.assert_allclose(out_0.numpy(), [[9.0], [7.0]])
        np.testing.assert_allclose(out_1.numpy(), [[8.0], [6.0]])


class TestInspectTensorSaveAndLoadTogether(ProbeEnvTestCase):
    """Covers the stage-5-before-stage-6 ordering of `inspect_tensor`."""

    def test_save_overwrites_a_dump_it_shares_a_directory_with(self):
        """Stage 5 runs first, so a shared save/load path destroys the override.

        Worth pinning: the symptom is a load that reports `max_abs_diff=0` and
        changes nothing, which looks like a perfect match rather than a
        misconfiguration.
        """
        shared = os.path.join(self.tmpdir, "shared")
        self.write_dump("tag", 0, [[9.0, 9.0]], root=shared)
        self.enable(
            ABLATION_SAVE_TENSOR_PATH=shared,
            ABLATION_LOAD_TENSOR_PATH=shared,
        )
        tensor = paddle.to_tensor([[1.0, 2.0]])
        result = inspect_tensor("tag", 0, tensor, save=True)
        np.testing.assert_allclose(result.numpy(), [[1.0, 2.0]])
        np.testing.assert_allclose(
            np.load(self.dump_path(shared, "tag", 0)), [[1.0, 2.0]]
        )

    def test_separate_directories_keep_both_halves_intact(self):
        self.write_dump("tag", 0, [[9.0, -9.0]])
        self.enable(
            ABLATION_SAVE_TENSOR_PATH=self.save_dir,
            ABLATION_LOAD_TENSOR_PATH=self.load_dir,
        )
        tensor = paddle.to_tensor([[1.0, 2.0]])
        result = inspect_tensor("tag", 0, tensor, save=True)
        np.testing.assert_allclose(result.numpy(), [[9.0, -9.0]])
        np.testing.assert_allclose(
            np.load(self.dump_path(self.save_dir, "tag", 0)), [[1.0, 2.0]]
        )

    def test_a_second_save_overwrites_the_first(self):
        self.enable(ABLATION_SAVE_TENSOR_PATH=self.save_dir)
        inspect_tensor("tag", 0, paddle.to_tensor([[1.0, 2.0]]), save=True)
        inspect_tensor("tag", 0, paddle.to_tensor([[3.0, 4.0]]), save=True)
        np.testing.assert_allclose(
            np.load(self.dump_path(self.save_dir, "tag", 0)), [[3.0, 4.0]]
        )


class TestInspectTensorTagFilters(ProbeEnvTestCase):
    """Covers the whitelist/blacklist combinations of stage 1."""

    def test_an_empty_whitelist_lets_every_tag_through(self):
        self.enable(ABLATION_SAVE_TENSOR_PATH=self.save_dir)
        inspect_tensor("anything", 0, paddle.to_tensor([1.0]), save=True)
        self.assertTrue(
            os.path.exists(self.dump_path(self.save_dir, "anything", 0))
        )

    def test_a_tag_in_both_lists_is_dropped(self):
        self.enable(
            ABLATION_TAG_WHITELIST="tag",
            ABLATION_TAG_BLACKLIST="tag",
            ABLATION_SAVE_TENSOR_PATH=self.save_dir,
        )
        tensor = paddle.to_tensor([1.0])
        self.assertIs(inspect_tensor("tag", 0, tensor, save=True), tensor)
        self.assertFalse(
            os.path.exists(self.dump_path(self.save_dir, "tag", 0))
        )

    def test_every_whitelisted_tag_survives(self):
        self.enable(
            ABLATION_TAG_WHITELIST="a,b",
            ABLATION_SAVE_TENSOR_PATH=self.save_dir,
        )
        for tag in ("a", "b", "c"):
            inspect_tensor(tag, 0, paddle.to_tensor([1.0]), save=True)
        self.assertTrue(os.path.exists(self.dump_path(self.save_dir, "a", 0)))
        self.assertTrue(os.path.exists(self.dump_path(self.save_dir, "b", 0)))
        self.assertFalse(os.path.exists(self.dump_path(self.save_dir, "c", 0)))


class TestLoadShapeGateExtra(ProbeEnvTestCase):
    """Further `_load_shape_ok` cases."""

    def test_three_dimensional_exact_match(self):
        self.assertEqual(_load_shape_ok([2, 3, 4], [2, 3, 4]), (True, "exact"))

    def test_flattening_everything_changes_the_row_count(self):
        ok, reason = _load_shape_ok([2, 3, 4], [24])
        self.assertFalse(ok)
        self.assertIn("row count differs dump_rows=2 live_rows=24", reason)

    def test_equal_zero_sized_shapes_pass(self):
        self.assertEqual(_load_shape_ok([0, 4], [0, 4]), (True, "exact"))

    def test_transposed_zero_sized_shapes_are_rejected(self):
        ok, reason = _load_shape_ok([4, 0], [0, 4])
        self.assertFalse(ok)
        self.assertIn("row count differs", reason)


class TestCanonicalRowsExtra(ProbeEnvTestCase):
    """Further `canonical_rows` cases."""

    def test_every_pair_unrouted_gives_all_zeros(self):
        index = paddle.full([2, 2], -1, dtype="int32")
        canon = canonical_rows(self._buffer(), index)
        np.testing.assert_allclose(canon.numpy(), np.zeros([4, 2]))

    def test_a_three_dimensional_buffer_is_masked_per_row(self):
        buf = paddle.to_tensor(
            [
                [[1.0, 1.0], [1.0, 1.0]],
                [[float("nan"), 0.0], [0.0, 0.0]],
                [[3.0, 3.0], [3.0, 3.0]],
                [[4.0, 4.0], [4.0, 4.0]],
            ],
            dtype="float32",
        )
        canon = canonical_rows(
            buf, paddle.to_tensor([[0, -1], [2, 3]], "int32")
        )
        self.assertEqual(list(canon.shape), [4, 2, 2])
        np.testing.assert_allclose(canon.numpy()[1], np.zeros([2, 2]))
        np.testing.assert_allclose(canon.numpy()[3], np.full([2, 2], 4.0))

    def test_a_single_local_expert(self):
        canon = canonical_rows(
            self._buffer(), paddle.to_tensor([[0], [4]], dtype="int32")
        )
        np.testing.assert_allclose(canon.numpy(), [[1.0, 1.0], [5.0, 5.0]])


class TestScatterCanonicalRowsExtra(ProbeEnvTestCase):
    """Further `scatter_canonical_rows` cases."""

    def test_every_pair_unrouted_leaves_the_values_alone(self):
        buf = self._buffer()
        index = paddle.full([2, 2], -1, dtype="int32")
        out = scatter_canonical_rows(buf, paddle.full([4, 2], 7.0), index)
        arr = out.numpy()
        np.testing.assert_allclose(
            arr[[0, 2, 4]], [[1.0, 1.0], [3.0, 3.0], [5.0, 5.0]]
        )
        self.assertTrue(np.isnan(arr[[1, 3, 5]]).all())


class TestForceUnitProbsExtra(ProbeEnvTestCase):
    """Further `inspect_tensor_force_unit_probs` cases."""

    def test_shape_and_dtype_survive(self):
        self.enable()
        probs = paddle.full([2, 3, 4], 0.125, dtype="float32")
        out = inspect_tensor_force_unit_probs(probs, "moe_act_quant_output")
        self.assertEqual(list(out.shape), [2, 3, 4])
        self.assertEqual(out.dtype, paddle.float32)
        np.testing.assert_allclose(out.numpy(), np.ones([2, 3, 4]))


class TestInspectTensorContainersExtra(ProbeEnvTestCase):
    """Further `index=` cases."""

    def test_with_element_can_replace_the_last_element(self):
        self.assertEqual(_with_element((1, 2, 3), 2, "x"), (1, 2, "x"))

    def test_a_three_element_bundle_keeps_its_other_members(self):
        self.write_dump("tag", 0, [7.0, 8.0])
        self.enable(ABLATION_LOAD_TENSOR_PATH=self.load_dir)
        bundle = (
            paddle.to_tensor([[1.0, 2.0]]),
            paddle.to_tensor([9.0]),
            paddle.to_tensor([3.0, 4.0]),
        )
        result = inspect_tensor("tag", 0, bundle, index=2)
        self.assertEqual(len(result), 3)
        self.assertIs(result[0], bundle[0])
        self.assertIs(result[1], bundle[1])
        np.testing.assert_allclose(result[2].numpy(), [7.0, 8.0])


class TestLastDimSegmentProbe(ProbeEnvTestCase):
    """Covers `slice_util` and the MLA call sites built on it.

    The segment is only ever a view handed to the probe, so with the probes off
    nothing is sliced and nothing is written into the live buffer -- its dygraph
    inplace version has to stay put. The `q[..., n:] = inspect_tensor(...)`
    spelling this replaces bumped that version on every forward, probes off
    included, which is what breaks backward / recompute.
    """

    def _buf(self):
        return paddle.to_tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])

    def _probe(self, tag, tensor, start=0, end=None, layer_idx=0, save=False):
        """The probe exactly as `multi_latent_attention.py` spells it.

        `save` is off at the real call sites (the training side loads the other
        side's dumps); the dump-content case below switches it on.
        """
        return inspect_tensor(
            tag,
            layer_idx,
            tensor,
            save=save,
            pre_save_func=lambda t: last_dim_segment(t, start, end),
            post_load_func=lambda seg, full=tensor: scatter_last_dim_segment(
                full, seg, start, end
            ),
        )

    def test_the_view_is_the_tail_segment(self):
        seg = last_dim_segment(self._buf(), 2)
        np.testing.assert_allclose(seg.numpy(), [[3.0, 4.0], [7.0, 8.0]])

    def test_the_view_is_the_head_segment(self):
        seg = last_dim_segment(self._buf(), 0, 2)
        np.testing.assert_allclose(seg.numpy(), [[1.0, 2.0], [5.0, 6.0]])

    def test_no_tensor_to_view_gives_up(self):
        self.assertIsNone(last_dim_segment(None, 2))

    def test_no_segment_leaves_the_buffer_alone(self):
        buf = self._buf()
        self.assertIs(scatter_last_dim_segment(buf, None, 2), buf)

    def test_the_inverse_replaces_only_its_own_segment(self):
        buf = self._buf()
        seg = paddle.to_tensor([[30.0, 40.0], [70.0, 80.0]])
        out = scatter_last_dim_segment(buf, seg, 2)
        np.testing.assert_allclose(
            out.numpy(), [[1.0, 2.0, 30.0, 40.0], [5.0, 6.0, 70.0, 80.0]]
        )

    def test_the_inverse_of_a_head_segment_keeps_the_tail(self):
        buf = self._buf()
        seg = paddle.to_tensor([[10.0, 20.0], [50.0, 60.0]])
        out = scatter_last_dim_segment(buf, seg, 0, 2)
        np.testing.assert_allclose(
            out.numpy(), [[10.0, 20.0, 3.0, 4.0], [50.0, 60.0, 7.0, 8.0]]
        )

    def test_a_full_width_segment_is_the_buffer_itself(self):
        buf = self._buf()
        seg = paddle.zeros([2, 4])
        out = scatter_last_dim_segment(buf, seg, 0, None)
        np.testing.assert_allclose(out.numpy(), np.zeros([2, 4]))

    def test_the_inverse_never_writes_into_the_live_buffer(self):
        buf = self._buf()
        version = buf.inplace_version
        scatter_last_dim_segment(buf, paddle.zeros([2, 2]), 2)
        self.assertEqual(buf.inplace_version, version)
        np.testing.assert_allclose(
            buf.numpy(), [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
        )

    def test_the_probes_off_path_does_not_touch_the_buffer(self):
        buf = self._buf()
        version = buf.inplace_version
        result = self._probe("mla_query_after_rope_pe", buf, start=2)
        self.assertIs(result, buf)
        self.assertEqual(buf.inplace_version, version)

    def test_no_dump_does_not_touch_the_buffer(self):
        self.enable(ABLATION_LOAD_TENSOR_PATH=self.load_dir)
        buf = self._buf()
        version = buf.inplace_version
        result = self._probe("mla_query_after_rope_pe", buf, start=2)
        self.assertIs(result, buf)
        self.assertEqual(buf.inplace_version, version)

    def test_only_the_segment_is_dumped(self):
        self.enable(ABLATION_SAVE_TENSOR_PATH=self.save_dir)
        self._probe(
            "mla_key_pe_after_rope",
            self._buf(),
            start=2,
            layer_idx=3,
            save=True,
        )
        dumped = np.load(
            self.dump_path(self.save_dir, "mla_key_pe_after_rope", 3)
        )
        np.testing.assert_allclose(dumped, [[3.0, 4.0], [7.0, 8.0]])

    def test_a_loaded_dump_replaces_only_the_tail_segment(self):
        self.write_dump(
            "mla_query_after_rope_pe", 0, [[30.0, 40.0], [70.0, 80.0]]
        )
        self.enable(ABLATION_LOAD_TENSOR_PATH=self.load_dir)
        buf = self._buf()
        version = buf.inplace_version
        result = self._probe("mla_query_after_rope_pe", buf, start=2)
        self.assertIsNot(result, buf)
        np.testing.assert_allclose(
            result.numpy(), [[1.0, 2.0, 30.0, 40.0], [5.0, 6.0, 70.0, 80.0]]
        )
        self.assertEqual(buf.inplace_version, version)

    def test_a_loaded_dump_replaces_only_the_head_segment(self):
        self.write_dump(
            "mla_query_after_rope_nope", 0, [[10.0, 20.0], [50.0, 60.0]]
        )
        self.enable(ABLATION_LOAD_TENSOR_PATH=self.load_dir)
        buf = self._buf()
        result = self._probe("mla_query_after_rope_nope", buf, end=2)
        np.testing.assert_allclose(
            result.numpy(), [[10.0, 20.0, 3.0, 4.0], [50.0, 60.0, 7.0, 8.0]]
        )


class TestMlpProbeTags(ProbeEnvTestCase):
    """Covers the training-side call site that composes probe tags.

    `MLP` is both the MoE shared expert and the dense `first_k_dense_replace` MLP,
    so its three probes carry `self.inspect_name` (a constructor argument) and the
    layer id published by `inspect_tensor_set_current_layer`. This side only loads,
    so an applied override is the only observable proof of the composed tag.
    """

    def tearDown(self):
        inspect_tensor_set_current_layer(None)
        super().tearDown()

    def _mlp(self, **kwargs):
        config = TransformerConfig(
            num_hidden_layers=2,
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=4,
            use_bias=True,
            init_method=init_method_normal(0.02),
            output_layer_init_method=scaled_init_method_normal(0.02, 1, 2.0),
            gated_linear_unit=False,
            bias_activation_fusion=False,
            activation_func_clamp_value=None,
            glu_linear_offset=0.0,
        )
        spec = get_gpt_layer_local_spec(
            config
        ).sublayers_spec.mlp.sublayers_spec
        mlp = MLP(config, spec, **kwargs)
        mlp.eval()
        return mlp

    def test_the_default_role_is_the_moe_shared_expert(self):
        self.assertEqual(self._mlp().inspect_name, "moe_shared")

    def test_the_dense_role_comes_from_the_constructor(self):
        mlp = self._mlp(inspect_name="dense_mlp")
        self.assertEqual(mlp.inspect_name, "dense_mlp")

    def test_the_shared_role_loads_moe_shared_tags_at_the_published_layer(self):
        mlp = self._mlp()
        override = np.full([4, 8, 64], 0.5, dtype=np.float32)
        self.write_dump("moe_shared_ffn2_output", 4, override)
        self.enable(ABLATION_LOAD_TENSOR_PATH=self.load_dir)
        inspect_tensor_set_current_layer(4)
        out, _ = mlp(paddle.randn([4, 8, 64]))
        np.testing.assert_allclose(out.numpy(), override)

    def test_the_dense_role_loads_dense_mlp_tags(self):
        mlp = self._mlp(inspect_name="dense_mlp")
        override = np.full([4, 8, 64], 0.25, dtype=np.float32)
        self.write_dump("dense_mlp_ffn2_output", 0, override)
        self.write_dump("moe_shared_ffn2_output", 0, np.zeros([4, 8, 64]))
        self.enable(ABLATION_LOAD_TENSOR_PATH=self.load_dir)
        inspect_tensor_set_current_layer(0)
        out, _ = mlp(paddle.randn([4, 8, 64]))
        np.testing.assert_allclose(out.numpy(), override)

    def test_the_shared_role_ignores_a_dense_mlp_dump(self):
        mlp = self._mlp()
        self.write_dump("dense_mlp_ffn2_output", 0, np.zeros([4, 8, 64]))
        self.enable(ABLATION_LOAD_TENSOR_PATH=self.load_dir)
        inspect_tensor_set_current_layer(0)
        out, _ = mlp(paddle.randn([4, 8, 64]))
        self.assertGreater(float(out.abs().sum()), 0.0)

    def test_a_dump_for_another_layer_is_not_picked_up(self):
        mlp = self._mlp()
        self.write_dump("moe_shared_ffn2_output", 7, np.zeros([4, 8, 64]))
        self.enable(ABLATION_LOAD_TENSOR_PATH=self.load_dir)
        inspect_tensor_set_current_layer(4)
        out, _ = mlp(paddle.randn([4, 8, 64]))
        self.assertGreater(float(out.abs().sum()), 0.0)

    def test_the_forward_is_untouched_while_the_probes_are_off(self):
        mlp = self._mlp()
        self.write_dump("moe_shared_ffn2_output", 4, np.zeros([4, 8, 64]))
        os.environ["ABLATION_LOAD_TENSOR_PATH"] = self.load_dir
        refresh_env_cache()
        inspect_tensor_set_current_layer(4)
        out, _ = mlp(paddle.randn([4, 8, 64]))
        self.assertGreater(float(out.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()

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

from __future__ import annotations

import unittest

try:
    import numpy as np
    import paddle

    from paddlefleet.transformer.identity_op import (
        IdentityFuncOp,
        IdentityOp,
    )
    from paddlefleet.transformer.transformer_layer import (
        TransformerLayer,
        TransformerLayerSublayersSpec,
        tensors_clone,
    )

    _IMPORT_ERROR: Exception | None = None
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
    _IMPORT_ERROR = exc

_DEPS_AVAILABLE = _IMPORT_ERROR is None
_SKIP_REASON = (
    "paddle/paddlefleet transformer stack not importable in this "
    f"environment: {_IMPORT_ERROR!r}"
)


@unittest.skipUnless(_DEPS_AVAILABLE, _SKIP_REASON)
class TestTensorsClone(unittest.TestCase):
    """tensors_clone deep-copies tensors, preserves container structure and
    values, and passes non-tensor leaves through unchanged."""

    def test_single_tensor_is_independent_copy(self):
        x = paddle.to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        cloned = tensors_clone(x)
        self.assertIsInstance(cloned, paddle.Tensor)
        self.assertIsNot(cloned, x)
        np.testing.assert_array_equal(
            cloned.numpy(), [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        )

    def test_list_clones_each_tensor(self):
        a = paddle.to_tensor([1.0, 2.0])
        b = paddle.to_tensor([3.0, 4.0])
        cloned = tensors_clone([a, b])
        self.assertIsInstance(cloned, list)
        self.assertEqual(len(cloned), 2)
        self.assertIsNot(cloned[0], a)
        self.assertIsNot(cloned[1], b)
        np.testing.assert_array_equal(cloned[0].numpy(), [1.0, 2.0])
        np.testing.assert_array_equal(cloned[1].numpy(), [3.0, 4.0])

    def test_tuple_preserves_type_and_values(self):
        a = paddle.to_tensor([7.0, 8.0])
        b = paddle.to_tensor([9.0, 10.0])
        cloned = tensors_clone((a, b))
        self.assertIsInstance(cloned, tuple)
        self.assertIsNot(cloned[0], a)
        np.testing.assert_array_equal(cloned[0].numpy(), [7.0, 8.0])
        np.testing.assert_array_equal(cloned[1].numpy(), [9.0, 10.0])

    def test_dict_clones_values(self):
        x = {"a": paddle.to_tensor([1.0, 2.0]), "b": paddle.to_tensor([3.0])}
        cloned = tensors_clone(x)
        self.assertIsInstance(cloned, dict)
        self.assertEqual(set(cloned), {"a", "b"})
        self.assertIsNot(cloned["a"], x["a"])
        np.testing.assert_array_equal(cloned["a"].numpy(), [1.0, 2.0])
        np.testing.assert_array_equal(cloned["b"].numpy(), [3.0])

    def test_list_with_nested_dict(self):
        inner = paddle.to_tensor([5.0, 6.0])
        cloned = tensors_clone([paddle.to_tensor([1.0]), {"a": inner}])
        self.assertIsInstance(cloned, list)
        self.assertIsInstance(cloned[1], dict)
        self.assertIsNot(cloned[1]["a"], inner)
        np.testing.assert_array_equal(cloned[1]["a"].numpy(), [5.0, 6.0])
        np.testing.assert_array_equal(cloned[0].numpy(), [1.0])

    def test_non_tensor_leaves_pass_through(self):
        t = paddle.to_tensor([1.0, 2.0])
        cloned = tensors_clone([t, 42, "hello"])
        self.assertIsNot(cloned[0], t)
        np.testing.assert_array_equal(cloned[0].numpy(), [1.0, 2.0])
        self.assertEqual(cloned[1], 42)
        self.assertEqual(cloned[2], "hello")

    def test_unsupported_type_raises_value_error(self):
        with self.assertRaises(ValueError):
            tensors_clone(42)


@unittest.skipUnless(_DEPS_AVAILABLE, _SKIP_REASON)
class TestTransformerLayerSublayersSpec(unittest.TestCase):
    """Default sublayer wiring and the per-instance sharded keys map."""

    def test_defaults_select_identity_placeholders(self):
        spec = TransformerLayerSublayersSpec()
        self.assertIs(spec.input_layernorm, IdentityOp)
        self.assertIs(spec.self_attention_hyper_connection, IdentityOp)
        self.assertIs(spec.self_attn, IdentityOp)
        self.assertIs(spec.self_attn_bda, IdentityFuncOp)
        self.assertIs(spec.pre_cross_attn_layernorm, IdentityOp)
        self.assertIs(spec.cross_attention, IdentityOp)
        self.assertIs(spec.cross_attn_bda, IdentityFuncOp)
        self.assertIs(spec.post_attention_layernorm, IdentityOp)
        self.assertIs(spec.mlp_hyper_connection, IdentityOp)
        self.assertIs(spec.mlp, IdentityOp)
        self.assertIs(spec.mlp_bda, IdentityFuncOp)
        self.assertIs(spec.block_attn_res, IdentityOp)
        # IdentityFuncOp subclasses IdentityOp; identity separates the two so a
        # bias-dropout-add slot cannot silently degrade to the plain op.
        self.assertIsNot(spec.self_attn_bda, IdentityOp)

    def test_keys_map_defaults_empty_and_is_per_instance(self):
        spec1 = TransformerLayerSublayersSpec()
        spec2 = TransformerLayerSublayersSpec()
        self.assertEqual(spec1.sharded_state_dict_keys_map, {})
        # default_factory=dict must give each instance its own dict, not a
        # shared mutable default.
        self.assertIsNot(
            spec1.sharded_state_dict_keys_map,
            spec2.sharded_state_dict_keys_map,
        )
        spec1.sharded_state_dict_keys_map["old"] = "new"
        self.assertEqual(spec2.sharded_state_dict_keys_map, {})

    def test_custom_keys_map_is_kept(self):
        spec = TransformerLayerSublayersSpec(
            sharded_state_dict_keys_map={"old": "new"}
        )
        self.assertEqual(spec.sharded_state_dict_keys_map, {"old": "new"})


class _OffloadConfig:
    """Faithful data stand-in exposing the dict-style ``.get`` contract that the
    real ``_compute_act_offload_kwargs`` relies on; the offload-selection
    branching under test runs unmodified against it."""

    def __init__(self, settings):
        self._settings = settings

    def get(self, key, default=None):
        if key == "decoderlayer_act_offload_settings":
            return self._settings
        return default


class _OffloadReceiver:
    def __init__(self, settings, layer_number):
        self.config = _OffloadConfig(settings)
        self.layer_number = layer_number


@unittest.skipUnless(_DEPS_AVAILABLE, _SKIP_REASON)
class TestComputeActOffloadKwargs(unittest.TestCase):
    """Offload-index selection in TransformerLayer._compute_act_offload_kwargs.

    The real production method is invoked directly on a minimal receiver so its
    branching executes unmodified; expected values are hand-derived. No GPU or
    process group is required for this control-flow contract.
    """

    def _run(self, settings, layer_number):
        recv = _OffloadReceiver(settings, layer_number)
        return TransformerLayer._compute_act_offload_kwargs(recv)

    def test_mod_match_selects_index_zero(self):
        # layer 2 % 2 == 0 == v2 -> offload activation 0
        self.assertEqual(
            self._run({"type": "mod", "value": [2, 0]}, 2),
            {"offload_indices": [0]},
        )

    def test_mod_no_match_selects_empty(self):
        # layer 3 % 2 == 1 != v2 (0) -> no offload
        self.assertEqual(
            self._run({"type": "mod", "value": [2, 0]}, 3),
            {"offload_indices": []},
        )

    def test_mod_accepts_tuple_value(self):
        # layer 4 % 3 == 1 == v2 -> offload; layer 5 % 3 == 2 != 1 -> none
        self.assertEqual(
            self._run({"type": "mod", "value": (3, 1)}, 4),
            {"offload_indices": [0]},
        )
        self.assertEqual(
            self._run({"type": "mod", "value": (3, 1)}, 5),
            {"offload_indices": []},
        )

    def test_layer_idxs_membership(self):
        self.assertEqual(
            self._run({"type": "layer_idxs", "value": [1, 3, 5]}, 3),
            {"offload_indices": [0]},
        )
        self.assertEqual(
            self._run({"type": "layer_idxs", "value": [1, 3, 5]}, 2),
            {"offload_indices": []},
        )

    def test_empty_type_returns_empty_kwargs(self):
        self.assertEqual(self._run({"type": "", "value": ""}, 1), {})

    def test_none_settings_falls_back_to_empty(self):
        # .get returns None -> `or {type:"", value:""}` default -> no branch
        self.assertEqual(self._run(None, 1), {})


if __name__ == "__main__":
    unittest.main()

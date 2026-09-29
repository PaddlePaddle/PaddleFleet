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

import contextlib
import hashlib
import io
import unittest

try:
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

    IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
    IMPORT_ERROR = exc

_SKIP_REASON = (
    f"paddle/paddlefleet transformer_layer import unavailable: {IMPORT_ERROR!r}"
)


@unittest.skipUnless(IMPORT_ERROR is None, _SKIP_REASON)
class TestTensorsClone(unittest.TestCase):
    """Behaviour of tensors_clone on tensors and nested containers."""

    def test_clone_tensor_is_independent_copy(self):
        x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
        cloned = tensors_clone(x)
        # A clone must be a distinct object carrying the same values; an
        # identity return would pass a values-only check.
        self.assertIsNot(cloned, x)
        self.assertEqual(cloned.tolist(), [[1.0, 2.0], [3.0, 4.0]])
        # Mutating the source in place must not leak into the clone.
        x.set_value(paddle.zeros_like(x))
        self.assertEqual(cloned.tolist(), [[1.0, 2.0], [3.0, 4.0]])

    def test_clone_list_preserves_type_values_and_nontensor(self):
        t0 = paddle.to_tensor([1.0, 2.0], dtype="float32")
        t1 = paddle.to_tensor([3.0, 4.0], dtype="float32")
        cloned = tensors_clone([t0, 7, "tag", t1])
        self.assertIsInstance(cloned, list)
        self.assertEqual(len(cloned), 4)
        self.assertIsNot(cloned[0], t0)
        self.assertIsNot(cloned[3], t1)
        self.assertEqual(cloned[0].tolist(), [1.0, 2.0])
        self.assertEqual(cloned[3].tolist(), [3.0, 4.0])
        # Non-tensor entries pass through unchanged at their positions.
        self.assertEqual(cloned[1], 7)
        self.assertEqual(cloned[2], "tag")

    def test_clone_tuple_preserves_tuple_type(self):
        t = paddle.to_tensor([5.0], dtype="float32")
        cloned = tensors_clone((t, 9))
        self.assertIsInstance(cloned, tuple)
        self.assertIsNot(cloned[0], t)
        self.assertEqual(cloned[0].tolist(), [5.0])
        self.assertEqual(cloned[1], 9)

    def test_clone_dict_of_tensors_values(self):
        a = paddle.to_tensor([1.0, 2.0], dtype="float32")
        b = paddle.to_tensor([3.0], dtype="float32")
        cloned = tensors_clone({"a": a, "b": b})
        self.assertIsInstance(cloned, dict)
        self.assertEqual(set(cloned), {"a", "b"})
        self.assertIsNot(cloned["a"], a)
        self.assertEqual(cloned["a"].tolist(), [1.0, 2.0])
        self.assertEqual(cloned["b"].tolist(), [3.0])

    def test_clone_list_of_dict_clones_inner_tensors(self):
        inner = paddle.to_tensor([[8.0, 9.0]], dtype="float32")
        cloned = tensors_clone([{"k": inner}, 42])
        self.assertIsInstance(cloned, list)
        self.assertIsInstance(cloned[0], dict)
        self.assertEqual(cloned[1], 42)
        self.assertIsNot(cloned[0]["k"], inner)
        self.assertEqual(cloned[0]["k"].tolist(), [[8.0, 9.0]])

    def test_clone_empty_containers_keep_type(self):
        self.assertIsInstance(tensors_clone(()), tuple)
        self.assertEqual(tensors_clone(()), ())
        self.assertIsInstance(tensors_clone([]), list)
        self.assertEqual(tensors_clone([]), [])
        self.assertIsInstance(tensors_clone({}), dict)
        self.assertEqual(tensors_clone({}), {})

    def test_clone_unsupported_scalar_raises_value_error(self):
        # The final else branch rejects unsupported top-level types.
        with self.assertRaises(ValueError):
            tensors_clone(42)
        with self.assertRaises(ValueError):
            tensors_clone("not-a-tensor")

    def test_clone_dict_with_nontensor_value_raises(self):
        # The dict branch calls value.clone() unconditionally, so a non-tensor
        # dict value raises AttributeError -- asymmetric with the list branch,
        # which passes non-tensors through. transformer_layer.py:131-132.
        with self.assertRaises(AttributeError):
            tensors_clone({"a": 42})

    @unittest.expectedFailure
    def test_clone_nested_list_should_clone_inner_tensor(self):
        # BUG: tensors_clone does not recurse into nested lists/tuples. A tensor
        # nested inside a sub-list hits the else branch (transformer_layer.py:
        # 123-124) and is appended by identity, so it is NOT cloned and stays
        # vulnerable to premature release -- contrary to the function's stated
        # purpose. Asserting the intended clone contract; xfails on production.
        inner = paddle.to_tensor([1.0, 2.0], dtype="float32")
        cloned = tensors_clone([[inner]])
        self.assertIsNot(cloned[0][0], inner)


@unittest.skipUnless(IMPORT_ERROR is None, _SKIP_REASON)
class TestTransformerLayerSublayersSpecDefaults(unittest.TestCase):
    """Default field contract of the TransformerLayerSublayersSpec dataclass."""

    def test_default_sublayer_specs(self):
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

    def test_sharded_map_default_is_independent_per_instance(self):
        # field(default_factory=dict) must give each instance its own dict; a
        # shared mutable default would leak mutations across instances.
        a = TransformerLayerSublayersSpec()
        b = TransformerLayerSublayersSpec()
        self.assertEqual(a.sharded_state_dict_keys_map, {})
        self.assertIsNot(
            a.sharded_state_dict_keys_map, b.sharded_state_dict_keys_map
        )
        a.sharded_state_dict_keys_map["old"] = "new"
        self.assertEqual(b.sharded_state_dict_keys_map, {})


@unittest.skipUnless(IMPORT_ERROR is None, _SKIP_REASON)
class TestTransformerLayerLogMD5(unittest.TestCase):
    """Gating and output contract of TransformerLayer._log_md5."""

    def setUp(self):
        # Snapshot the mutable class-level flags and restore them even if a
        # test fails, so this test does not pollute later tests in-process.
        for attr in (
            "_LOG_LAYER_MD5",
            "_gpt_model_use_experimental_version",
            "_skip_mtp_probes",
        ):
            self.addCleanup(
                setattr,
                TransformerLayer,
                attr,
                getattr(TransformerLayer, attr),
            )

    @staticmethod
    def _capture(tensor, name, layer_idx):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            TransformerLayer._log_md5(tensor, name, layer_idx)
        return buf.getvalue()

    def test_emits_independent_md5_when_enabled(self):
        TransformerLayer._LOG_LAYER_MD5 = True
        TransformerLayer._gpt_model_use_experimental_version = True
        TransformerLayer._skip_mtp_probes = False

        tensor = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
        expected_md5 = hashlib.md5(
            tensor.cast("float32").numpy().tobytes()
        ).hexdigest()

        out = self._capture(tensor, "hidden", 7)
        self.assertIn(f"MD5={expected_md5}", out)
        self.assertIn("Layer=7", out)
        self.assertIn("hidden", out)
        self.assertIn("shape=[2, 2]", out)

    def test_output_distinguishes_tensors(self):
        TransformerLayer._LOG_LAYER_MD5 = True
        TransformerLayer._gpt_model_use_experimental_version = True
        TransformerLayer._skip_mtp_probes = False
        out1 = self._capture(
            paddle.to_tensor([1.0, 2.0], dtype="float32"), "h", 0
        )
        out2 = self._capture(
            paddle.to_tensor([9.0, 9.0], dtype="float32"), "h", 0
        )
        self.assertNotEqual(out1, out2)

    def test_silent_when_logging_disabled(self):
        TransformerLayer._LOG_LAYER_MD5 = False
        TransformerLayer._gpt_model_use_experimental_version = True
        TransformerLayer._skip_mtp_probes = False
        out = self._capture(paddle.to_tensor([1.0], dtype="float32"), "h", 0)
        self.assertEqual(out, "")

    def test_silent_when_not_experimental_version(self):
        TransformerLayer._LOG_LAYER_MD5 = True
        TransformerLayer._gpt_model_use_experimental_version = False
        TransformerLayer._skip_mtp_probes = False
        out = self._capture(paddle.to_tensor([1.0], dtype="float32"), "h", 0)
        self.assertEqual(out, "")

    def test_silent_during_mtp_probes(self):
        TransformerLayer._LOG_LAYER_MD5 = True
        TransformerLayer._gpt_model_use_experimental_version = True
        TransformerLayer._skip_mtp_probes = True
        out = self._capture(paddle.to_tensor([1.0], dtype="float32"), "h", 0)
        self.assertEqual(out, "")


if __name__ == "__main__":
    unittest.main()

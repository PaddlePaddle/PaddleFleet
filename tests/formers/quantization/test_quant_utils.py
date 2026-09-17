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

"""Behavior tests for paddlefleet.quantization.quantization_utils.

This module implements *orchestration* only: it parses which weight-quantize
algorithm applies to a layer name, decides which layers to skip, dispatches to
the right converter, and rewrites the keys of a state dict. The actual numeric
quant/dequant math is delegated to GPU kernels (``paddle.nn.quant.weight_quantize``,
``qat_utils.quantize``, ``dequant_weight``) that run through ``.cuda()``.

Therefore:
  * The algorithm-selection / skip / dispatch / error-contract behavior is
    hand-derived from the source and verified on CPU (无卡).
  * The numeric weight-quantize / dequant relation can only be exercised on a
    single CUDA card; those tests are gated with ``skipUnless`` and carry an
    explicit reason instead of being faked on 无卡.

Expected values are derived independently from the ``re.fullmatch`` semantics
and the explicit algorithm lists in the source. The functions under test are
never used to build their own expected values, and their own methods are never
patched.
"""

import unittest


def _load_utils():
    from paddlefleet.quantization import quantization_utils as qu

    return qu


def _load_config_cls():
    from paddlefleet.quantization.quantization_config import QuantizationConfig

    return QuantizationConfig


def _cuda_available():
    try:
        import paddle
    except ImportError:
        return False
    return (
        paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0
    )


_NO_CUDA_REASON = (
    "weight_quantize / .cuda() require a CUDA device; the numeric "
    "quant/dequant relation is not runnable on 无卡 CPU."
)


class TestParseWeightQuantizeAlgo(unittest.TestCase):
    """parse_weight_quantize_algo: name -> algorithm selection.

    Contract (hand-read from the source):
      1. ignore_modules is a list of *regex* patterns matched with
         re.fullmatch; a match forces None regardless of the configured algo.
      2. Otherwise a str algo is returned verbatim for every name.
      3. Otherwise (dict algo) each algo key whose pattern list fullmatches the
         name is selected; the loop has no break, so the LAST matching algo in
         insertion order wins. No match -> None.
    """

    def test_string_algo_passthrough(self):
        qu = _load_utils()
        cfg = _load_config_cls()(weight_quantize_algo="weight_only_int8")
        self.assertEqual(
            qu.parse_weight_quantize_algo(cfg, "model.layers.0.mlp"),
            "weight_only_int8",
        )

    def test_ignore_exact_returns_none(self):
        qu = _load_utils()
        cfg = _load_config_cls()(
            weight_quantize_algo="weight_only_int8", ignore_modules=["lm_head"]
        )
        self.assertIsNone(qu.parse_weight_quantize_algo(cfg, "lm_head"))

    def test_ignore_regex_is_fullmatch_not_search(self):
        qu = _load_utils()
        cfg = _load_config_cls()(
            weight_quantize_algo="weight_only_int8", ignore_modules=[".*head"]
        )
        # ".*head" fullmatches "model.lm_head" -> ignored.
        self.assertIsNone(qu.parse_weight_quantize_algo(cfg, "model.lm_head"))
        # ".*head" does NOT fullmatch "head_extra" (trailing chars) -> kept.
        self.assertEqual(
            qu.parse_weight_quantize_algo(cfg, "head_extra"),
            "weight_only_int8",
        )

    def test_ignore_takes_precedence_over_string_algo(self):
        qu = _load_utils()
        cfg = _load_config_cls()(
            weight_quantize_algo="weight_only_int8", ignore_modules=["linear"]
        )
        self.assertIsNone(qu.parse_weight_quantize_algo(cfg, "linear"))

    def test_non_matching_ignore_leaves_algo(self):
        qu = _load_utils()
        cfg = _load_config_cls()(
            weight_quantize_algo="weight_only_int8", ignore_modules=["linear"]
        )
        # "linear" does not fullmatch "linear_two" -> not ignored.
        self.assertEqual(
            qu.parse_weight_quantize_algo(cfg, "linear_two"),
            "weight_only_int8",
        )

    def test_dict_match_and_no_match(self):
        qu = _load_utils()
        cfg = _load_config_cls()(
            weight_quantize_algo={"weight_only_int8": [".*mlp.*"]}
        )
        self.assertEqual(
            qu.parse_weight_quantize_algo(cfg, "layers.0.mlp.gate_proj"),
            "weight_only_int8",
        )
        self.assertIsNone(
            qu.parse_weight_quantize_algo(cfg, "layers.0.self_attn.q_proj")
        )

    def test_dict_pattern_requires_fullmatch(self):
        qu = _load_utils()
        cfg = _load_config_cls()(
            weight_quantize_algo={"weight_only_int8": ["mlp"]}
        )
        self.assertEqual(
            qu.parse_weight_quantize_algo(cfg, "mlp"), "weight_only_int8"
        )
        # "mlp" must fullmatch; "mlp.gate" has extra chars -> None.
        self.assertIsNone(qu.parse_weight_quantize_algo(cfg, "mlp.gate"))

    def test_dict_last_matching_algo_wins(self):
        qu = _load_utils()
        # Both algos' patterns fullmatch "linear"; the loop keeps the last
        # matching key in insertion order (weight_only_int4).
        cfg = _load_config_cls()(
            weight_quantize_algo={
                "weight_only_int8": [".*"],
                "weight_only_int4": ["linear"],
            }
        )
        self.assertEqual(
            qu.parse_weight_quantize_algo(cfg, "linear"), "weight_only_int4"
        )
        # "other" only matches the ".*" list under weight_only_int8.
        self.assertEqual(
            qu.parse_weight_quantize_algo(cfg, "other"), "weight_only_int8"
        )

    def test_dict_ignore_precedence(self):
        qu = _load_utils()
        cfg = _load_config_cls()(
            weight_quantize_algo={"weight_only_int8": [".*"]},
            ignore_modules=["skip_me"],
        )
        self.assertIsNone(qu.parse_weight_quantize_algo(cfg, "skip_me"))
        self.assertEqual(
            qu.parse_weight_quantize_algo(cfg, "keep"), "weight_only_int8"
        )


def _build_linear_model():
    from paddle import nn

    class _Model(nn.Layer):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(16, 8)

    return _Model()


class TestReplaceWithQuantizationLinearCPU(unittest.TestCase):
    """The skip path of replace_with_quantization_linear (no GPU needed).

    When parse_weight_quantize_algo returns None the layer must be left
    untouched: identity preserved, still an nn.Linear. This exercises the real
    named_sublayers walk and the real parse call; nothing is patched.
    """

    def test_ignored_module_not_replaced(self):
        from paddle import nn

        qu = _load_utils()
        model = _build_linear_model()
        original = model.linear
        cfg = _load_config_cls()(
            weight_quantize_algo="weight_only_int8", ignore_modules=["linear"]
        )
        qu.replace_with_quantization_linear(model, cfg)
        self.assertIs(model.linear, original)
        self.assertIsInstance(model.linear, nn.Linear)

    def test_dict_no_match_not_replaced(self):
        from paddle import nn

        qu = _load_utils()
        model = _build_linear_model()
        original = model.linear
        cfg = _load_config_cls()(
            weight_quantize_algo={"weight_only_int8": ["other"]}
        )
        qu.replace_with_quantization_linear(model, cfg)
        self.assertIs(model.linear, original)
        self.assertIsInstance(model.linear, nn.Linear)


class TestReplaceWithQuantizationLinearGPU(unittest.TestCase):
    """The real replacement path constructs QuantizationLinear (GPU only)."""

    @unittest.skipUnless(_cuda_available(), _NO_CUDA_REASON)
    def test_matching_module_replaced(self):
        import paddle

        qu = _load_utils()
        from paddlefleet.quantization.quantization_linear import (
            QuantizationLinear,
        )

        model = _build_linear_model()
        cfg = _load_config_cls()(weight_quantize_algo="weight_only_int8")
        # The real loader replaces modules inside paddle.LazyGuard() (via
        # model_utils.from_pretrained -> ContextManagers([no_init_weights,
        # LazyGuard])). Outside LazyGuard the int8 create_parameter eagerly runs
        # the default XavierUniform initializer, whose uniform kernel is not
        # registered for int8. Replicate the production context here.
        with paddle.LazyGuard():
            qu.replace_with_quantization_linear(model, cfg)
        self.assertIsInstance(model.linear, QuantizationLinear)


class TestConvertToQuantizeStateDictCPU(unittest.TestCase):
    """CPU-verifiable orchestration of convert_to_quantize_state_dict.

    The numeric branch calls .cuda()/weight_quantize; the branches asserted
    here (empty list, None-algo skip, unsupported-algo error) never reach it.
    """

    def test_empty_list_is_identity(self):
        import numpy as np
        import paddle

        qu = _load_utils()
        cfg = _load_config_cls()(weight_quantize_algo="weight_only_int8")
        w = paddle.to_tensor([[1.0, -2.0], [3.0, -4.0]], dtype="float32")
        state_dict = {"linear.weight": w}
        result = qu.convert_to_quantize_state_dict(
            state_dict, [], cfg, "float32"
        )
        # No layer names -> the dict object is returned untouched.
        self.assertIs(result, state_dict)
        self.assertEqual(set(result), {"linear.weight"})
        np.testing.assert_array_equal(
            result["linear.weight"].numpy(), w.numpy()
        )

    def test_none_algo_layer_is_skipped(self):
        import numpy as np
        import paddle

        qu = _load_utils()
        # ignore_modules matches -> parse returns None -> layer skipped.
        cfg = _load_config_cls()(
            weight_quantize_algo="weight_only_int8", ignore_modules=["linear"]
        )
        w = paddle.to_tensor([[5.0, 6.0], [7.0, 8.0]], dtype="float32")
        state_dict = {"linear.weight": w}
        result = qu.convert_to_quantize_state_dict(
            state_dict, ["linear"], cfg, "float32"
        )
        self.assertIs(result, state_dict)
        # Weight untouched: no quant_weight / weight_scale keys introduced.
        self.assertEqual(set(result), {"linear.weight"})
        np.testing.assert_array_equal(
            result["linear.weight"].numpy(), w.numpy()
        )

    def test_unsupported_algo_raises_not_implemented(self):
        import paddle

        qu = _load_utils()
        # "a8w8" is accepted by QuantizationConfig but is NOT in the dispatch
        # lists of convert_to_quantize_state_dict (weight-quant list is
        # a8w8linear/a8w4linear/fp8linear/..., qlora list is fp4/nf4). The
        # else branch raises before any .cuda() access. This documents an
        # inconsistency between the config's accepted set and this converter's
        # handled set; the converter's fail-fast is the asserted contract.
        cfg = _load_config_cls()(weight_quantize_algo="a8w8")
        state_dict = {"linear.weight": paddle.to_tensor([[1.0]])}
        with self.assertRaises(NotImplementedError):
            qu.convert_to_quantize_state_dict(
                state_dict, ["linear"], cfg, "float32"
            )


class TestConvertToQuantizeDequantizeStateDictCPU(unittest.TestCase):
    """CPU-verifiable orchestration of convert_to_quantize_dequantize_state_dict.

    Note this entry takes (state_dict, list, config) with no dtype argument.
    """

    def test_empty_list_is_identity(self):
        import numpy as np
        import paddle

        qu = _load_utils()
        cfg = _load_config_cls()(weight_quantize_algo="weight_only_int8")
        w = paddle.to_tensor([[1.0, -2.0], [3.0, -4.0]], dtype="float32")
        state_dict = {"linear.weight": w}
        result = qu.convert_to_quantize_dequantize_state_dict(
            state_dict, [], cfg
        )
        self.assertIs(result, state_dict)
        self.assertEqual(set(result), {"linear.weight"})
        np.testing.assert_array_equal(
            result["linear.weight"].numpy(), w.numpy()
        )

    def test_none_algo_layer_is_skipped(self):
        import numpy as np
        import paddle

        qu = _load_utils()
        # dict algo whose pattern list does not fullmatch -> parse None -> skip.
        cfg = _load_config_cls()(
            weight_quantize_algo={"weight_only_int8": ["other"]}
        )
        w = paddle.to_tensor([[9.0, 10.0], [11.0, 12.0]], dtype="float32")
        state_dict = {"linear.weight": w}
        result = qu.convert_to_quantize_dequantize_state_dict(
            state_dict, ["linear"], cfg
        )
        self.assertIs(result, state_dict)
        self.assertEqual(set(result), {"linear.weight"})
        np.testing.assert_array_equal(
            result["linear.weight"].numpy(), w.numpy()
        )

    def test_unsupported_algo_raises_not_implemented(self):
        import paddle

        qu = _load_utils()
        cfg = _load_config_cls()(weight_quantize_algo="a8w8")
        state_dict = {"linear.weight": paddle.to_tensor([[1.0]])}
        with self.assertRaises(NotImplementedError):
            qu.convert_to_quantize_dequantize_state_dict(
                state_dict, ["linear"], cfg
            )


class TestConvertWeightQuantizeDequantizeRoundTripGPU(unittest.TestCase):
    """Numeric dequant relation for weight-only int8 (single card only)."""

    @unittest.skipUnless(_cuda_available(), _NO_CUDA_REASON)
    def test_weight_only_int8_dequant_is_close_but_not_identity(self):
        import numpy as np
        import paddle

        qu = _load_utils()
        cfg = _load_config_cls()(weight_quantize_algo="weight_only_int8")

        # weight_quantize requires the weight's first (input) dim to be
        # divisible by 64, so use a [64, 4] linear weight. linspace gives
        # distinct values with a clear per-tensor maximum magnitude of 4.0.
        original = np.linspace(-4.0, 4.0, num=64 * 4, dtype=np.float32).reshape(
            64, 4
        )
        weight = paddle.to_tensor(original, dtype="float16")
        state_dict = {"linear.weight": weight}

        result = qu.convert_to_quantize_dequantize_state_dict(
            state_dict, ["linear"], cfg
        )

        # Dequantized weight replaces the original key with the same layout.
        self.assertIn("linear.weight", result)
        dequant = result["linear.weight"]
        self.assertEqual(list(dequant.shape), list(weight.shape))
        self.assertEqual(dequant.dtype, weight.dtype)

        got = dequant.astype("float32").numpy()
        # Independent, conservative bound: for round-to-nearest int8 with a
        # per-channel abs-max scale, every element's reconstruction error is
        # <= scale/2 <= max|w|/(2*127). Any channel's scale is <= the global
        # max magnitude / 127, so max|w|/127 is a safe elementwise ceiling
        # regardless of the quantization axis.
        step = float(np.abs(original).max()) / 127.0
        max_err = np.abs(got - original).max()
        self.assertLessEqual(max_err, step)
        # Quantization must actually have happened (not a passthrough).
        self.assertGreater(max_err, 0.0)


if __name__ == "__main__":
    unittest.main()

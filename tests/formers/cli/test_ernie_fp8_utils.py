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

"""CPU-observable behavior tests for the ERNIE MoE FP8 token-dispatcher utils.

Target production code:
``src/paddlefleet/cli/train/ernie_pretrain/models/moe/token_dispatcher/fp8_utils.py``

Scope (unit-test-rules.md "计算优化" / "配置与运行基础设施"):
  * ``_get_fp8_weight_and_scale`` -- the (stacked, transpose) -> attribute-pair
    selection matrix.  This is pure control flow: the function must return the
    weight/scale *pair that matches the requested layout* and nothing else, so
    the oracle here checks the *identity* of the selected attributes (not shape
    or type).  A swapped stacked/transpose branch is exactly what this catches.
  * ``has_config`` -- the three-way guard ``config_map is not None and
    key in config_map and config_map[key]``.  Every cell of the truth table is
    exercised with a hand-derived boolean expectation, including the falsy-value
    cell (key present but value 0/""/False) which a naive ``key in map`` check
    would get wrong.
  * ``fused_stack_transpose_quant`` -- the dispatch decision between the cached
    pre-quantized ``fp8_weight_stacked`` attribute path and the paddle fused
    kernel path, plus argument forwarding of ``transpose``.

The real FP8 GEMM numerics live in ``ExpertsGroupGemm*Node`` and require a GPU
with deep_gemm; those are NOT asserted here.  The whole module imports ``paddle``
at import time, so when paddle is absent the tests skip with an honest reason
rather than reporting a fake pass.  Every expected value below is hand-derived
and independent of the production implementation.
"""

import unittest
from unittest import mock

try:
    import paddle  # noqa: F401

    from paddlefleet.cli.train.ernie_pretrain.models.moe.token_dispatcher import (
        fp8_utils,
    )

    _IMPORT_ERROR = None
except ImportError as exc:  # paddle (or a paddle-dependent import) is missing
    fp8_utils = None
    _IMPORT_ERROR = exc

_SKIP_REASON = (
    "paddlefleet ... token_dispatcher.fp8_utils imports paddle at module load; "
    f"paddle is not importable in this environment ({_IMPORT_ERROR})"
)


class _FakeQuantizedWeight:
    """Stand-in for a pre-quantized weight parameter.

    Each of the eight FP8 attributes carries a *distinct, self-identifying*
    sentinel so a test can prove which pair the selector returned and detect a
    stacked/transpose branch swap.  This is a plain data holder, not a paddle
    tensor, so it does not need a GPU or paddle runtime.
    """

    def __init__(self):
        self.fp8_weight = "w:plain"
        self.fp8_scale = "s:plain"
        self.fp8_weight_transpose = "w:transpose"
        self.fp8_scale_transpose = "s:transpose"
        self.fp8_weight_stacked = "w:stacked"
        self.fp8_scale_stacked = "s:stacked"
        self.fp8_weight_stacked_transpose = "w:stacked_transpose"
        self.fp8_scale_stacked_transpose = "s:stacked_transpose"


@unittest.skipUnless(fp8_utils is not None, _SKIP_REASON)
class TestGetFp8WeightAndScale(unittest.TestCase):
    """The four (stacked, transpose) branches must map to distinct pairs."""

    def setUp(self):
        self.weight = _FakeQuantizedWeight()

    def test_plain_layout_returns_base_pair(self):
        w, s = fp8_utils._get_fp8_weight_and_scale(
            self.weight, stacked=False, transpose=False
        )
        # Hand-derived expectation: neither stacked nor transposed -> base attrs.
        self.assertEqual(w, "w:plain")
        self.assertEqual(s, "s:plain")

    def test_transpose_only_returns_transpose_pair(self):
        w, s = fp8_utils._get_fp8_weight_and_scale(
            self.weight, stacked=False, transpose=True
        )
        self.assertEqual(w, "w:transpose")
        self.assertEqual(s, "s:transpose")

    def test_stacked_only_returns_stacked_pair(self):
        w, s = fp8_utils._get_fp8_weight_and_scale(
            self.weight, stacked=True, transpose=False
        )
        self.assertEqual(w, "w:stacked")
        self.assertEqual(s, "s:stacked")

    def test_stacked_and_transpose_returns_stacked_transpose_pair(self):
        w, s = fp8_utils._get_fp8_weight_and_scale(
            self.weight, stacked=True, transpose=True
        )
        self.assertEqual(w, "w:stacked_transpose")
        self.assertEqual(s, "s:stacked_transpose")

    def test_defaults_match_plain_layout(self):
        # The default arguments (stacked=False, transpose=False) must select the
        # base pair; a changed default would silently reroute every caller.
        w, s = fp8_utils._get_fp8_weight_and_scale(self.weight)
        self.assertEqual((w, s), ("w:plain", "s:plain"))

    def test_all_four_branches_are_mutually_distinct(self):
        # Guards against two branches collapsing onto the same attribute pair.
        results = {
            (st, tr): fp8_utils._get_fp8_weight_and_scale(
                self.weight, stacked=st, transpose=tr
            )
            for st in (False, True)
            for tr in (False, True)
        }
        self.assertEqual(
            results,
            {
                (False, False): ("w:plain", "s:plain"),
                (False, True): ("w:transpose", "s:transpose"),
                (True, False): ("w:stacked", "s:stacked"),
                (True, True): ("w:stacked_transpose", "s:stacked_transpose"),
            },
        )
        # Every returned pair is unique -> no branch aliases another.
        self.assertEqual(len(set(results.values())), 4)


@unittest.skipUnless(fp8_utils is not None, _SKIP_REASON)
class TestHasConfig(unittest.TestCase):
    """``has_config`` truth table: None / missing / falsy / truthy."""

    def test_none_config_map_is_false(self):
        self.assertIs(fp8_utils.has_config(None, "flag"), False)

    def test_missing_key_is_false(self):
        self.assertIs(fp8_utils.has_config({"other": True}, "flag"), False)

    def test_present_but_falsy_value_is_false(self):
        # Key exists yet is disabled; a bare ``key in map`` check would wrongly
        # report True, so each falsy value must resolve to False.
        for falsy in (False, 0, "", None, [], {}):
            with self.subTest(value=falsy):
                self.assertIs(
                    fp8_utils.has_config({"flag": falsy}, "flag"), False
                )

    def test_present_and_truthy_value_is_true(self):
        for truthy in (True, 1, "yes", ["x"], {"k": "v"}):
            with self.subTest(value=truthy):
                self.assertIs(
                    fp8_utils.has_config({"flag": truthy}, "flag"), True
                )

    def test_return_type_is_strict_bool(self):
        # Contract: the helper normalizes to a real bool, not the raw value.
        result = fp8_utils.has_config({"flag": "yes"}, "flag")
        self.assertIsInstance(result, bool)


@unittest.skipUnless(fp8_utils is not None, _SKIP_REASON)
class TestFusedStackTransposeQuant(unittest.TestCase):
    """Dispatch between the cached-attribute path and the paddle kernel path."""

    def test_cached_stacked_path_forwards_transpose_false(self):
        # weight_list[0] already exposes fp8_weight_stacked -> the function must
        # reuse the cached stacked attributes and NOT call the paddle kernel.
        weight = _FakeQuantizedWeight()
        with mock.patch.object(
            paddle.incubate.nn.functional,
            "fused_stack_transpose_quant",
        ) as kernel:
            w, s = fp8_utils.fused_stack_transpose_quant(
                [weight], transpose=False
            )
        kernel.assert_not_called()
        self.assertEqual(w, "w:stacked")
        self.assertEqual(s, "s:stacked")

    def test_cached_stacked_path_forwards_transpose_true(self):
        weight = _FakeQuantizedWeight()
        with mock.patch.object(
            paddle.incubate.nn.functional,
            "fused_stack_transpose_quant",
        ) as kernel:
            w, s = fp8_utils.fused_stack_transpose_quant(
                [weight], transpose=True
            )
        kernel.assert_not_called()
        # transpose=True must route to the stacked-transpose cached pair.
        self.assertEqual(w, "w:stacked_transpose")
        self.assertEqual(s, "s:stacked_transpose")

    def test_uncached_path_delegates_to_paddle_kernel(self):
        # A weight without fp8_weight_stacked must fall through to the paddle
        # fused kernel, forwarding the raw list and the transpose flag verbatim.
        class _RawWeight:
            pass

        raw = [_RawWeight(), _RawWeight()]
        sentinel = ("kernel_w", "kernel_scale")
        captured = {}

        def fake_kernel(weight_list, transpose):
            captured["weight_list"] = weight_list
            captured["transpose"] = transpose
            return sentinel

        with mock.patch.object(
            paddle.incubate.nn.functional,
            "fused_stack_transpose_quant",
            side_effect=fake_kernel,
        ) as kernel:
            result = fp8_utils.fused_stack_transpose_quant(raw, transpose=True)

        kernel.assert_called_once()
        self.assertIs(captured["weight_list"], raw)
        self.assertIs(captured["transpose"], True)
        self.assertEqual(result, sentinel)


@unittest.skipUnless(fp8_utils is not None, _SKIP_REASON)
class TestModuleSurface(unittest.TestCase):
    """The public export list is part of the module contract."""

    def test_all_lists_both_gemm_nodes(self):
        self.assertEqual(
            set(fp8_utils.__all__),
            {"ExpertsGroupGemmNode", "ExpertsGroupGemmContiguousNode"},
        )
        for name in fp8_utils.__all__:
            self.assertTrue(
                hasattr(fp8_utils, name),
                f"{name} is exported in __all__ but not defined in the module",
            )


if __name__ == "__main__":
    unittest.main()

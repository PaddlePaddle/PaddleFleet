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

"""Behavioural tests for paddlefleet/peft/lora/auto_lora_model.py.

These tests exercise the real production APIs on CPU:
  * LoRAAutoLinear adapter math (forward, merge/unmerge, disable, scaling,
    base-weight freezing) with hand-derived independent references.
  * LoRAAutoLinear.auto_dist_config prefixing and the ColWise/RowWise ->
    lora_B/lora_A parallelize-plan mapping.
  * LoRAAutoModel.merge_auto_dist_configs dict/list merge semantics.
  * LoRAAutoModel.get_lora_model target-module replacement + weight sharing
    and mark_only_lora_as_trainable freeze/trainable contract.
"""

import unittest

import numpy as np
import paddle
import paddle.distributed as dist
import paddle.nn as nn

from paddlefleet.peft.lora.auto_lora_model import (
    AVAILABLE_LAYERS,
    LoRAAutoLinear,
    LoRAAutoModel,
    lora_layers,
)
from paddlefleet.peft.lora.lora_config import LoRAAutoConfig


def _set(param, array):
    """Assign a numpy array (float32) into a paddle parameter."""
    param.set_value(paddle.to_tensor(np.asarray(array, dtype="float32")))


class _TwoLinear(nn.Layer):
    """Small CPU model with two named nn.Linear sublayers."""

    def __init__(self):
        super().__init__()
        # weight shape is [in_features, out_features] in paddle.
        self.linear1 = nn.Linear(4, 6)
        self.linear2 = nn.Linear(6, 3)

    def forward(self, x):
        return self.linear2(self.linear1(x))


class TestLoRAAutoLinearAdapter(unittest.TestCase):
    """LoRAAutoLinear adapter math against independent references."""

    def test_scaling_and_freeze_contract(self):
        layer = LoRAAutoLinear(in_features=4, out_features=6, r=2, lora_alpha=4)
        # scaling = lora_alpha / r for the default (non-rslora) case.
        self.assertAlmostEqual(layer.scaling, 4 / 2, places=6)
        self.assertEqual(layer.r, 2)
        self.assertEqual(layer.lora_alpha, 4)
        self.assertEqual(list(layer.lora_A.shape), [4, 2])
        self.assertEqual(list(layer.lora_B.shape), [2, 6])
        # Base weight frozen, adapters trainable.
        self.assertTrue(layer.weight.stop_gradient)
        self.assertFalse(layer.lora_A.stop_gradient)
        self.assertFalse(layer.lora_B.stop_gradient)
        self.assertFalse(layer.merged)
        self.assertFalse(layer.disable_lora)
        # lora_B is zero-initialised, so a fresh adapter is a no-op delta.
        np.testing.assert_array_equal(
            layer.lora_B.numpy(), np.zeros([2, 6], dtype="float32")
        )

    def test_rslora_scaling(self):
        layer = LoRAAutoLinear(
            in_features=4, out_features=6, r=4, lora_alpha=8, rslora=True
        )
        # rslora divides by sqrt(r) instead of r.
        self.assertAlmostEqual(layer.scaling, 8 / np.sqrt(4), places=6)

    def test_rank_must_be_positive(self):
        with self.assertRaises(ValueError):
            LoRAAutoLinear(in_features=4, out_features=6, r=0)

    def test_forward_matches_independent_reference(self):
        layer = LoRAAutoLinear(in_features=3, out_features=2, r=2, lora_alpha=4)
        weight = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        bias = [0.5, -0.5]
        lora_A = [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
        lora_B = [[1.0, 1.0], [0.0, 2.0]]
        _set(layer.weight, weight)
        _set(layer.bias, bias)
        _set(layer.lora_A, lora_A)
        _set(layer.lora_B, lora_B)

        x = np.array([[1.0, 2.0, 3.0]], dtype="float32")
        out = layer(paddle.to_tensor(x)).numpy()

        # Independent reference: W x + b + (x A B) * scaling, scaling = 4/2 = 2.
        base = x @ np.array(weight) + np.array(bias)
        delta = (x @ np.array(lora_A) @ np.array(lora_B)) * 2.0
        np.testing.assert_allclose(out, base + delta, rtol=1e-5, atol=1e-6)
        # Hand-computed absolute anchor guards against reference drift.
        np.testing.assert_allclose(out, [[12.5, 20.5]], rtol=1e-5, atol=1e-6)

    def test_disable_lora_drops_adapter_contribution(self):
        layer = LoRAAutoLinear(in_features=3, out_features=2, r=2, lora_alpha=4)
        weight = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        bias = [0.5, -0.5]
        _set(layer.weight, weight)
        _set(layer.bias, bias)
        _set(layer.lora_A, [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
        _set(layer.lora_B, [[1.0, 1.0], [0.0, 2.0]])

        x = np.array([[1.0, 2.0, 3.0]], dtype="float32")
        layer.disable_lora = True
        out = layer(paddle.to_tensor(x)).numpy()
        # Only the frozen base linear should contribute.
        expected = x @ np.array(weight) + np.array(bias)
        np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(out, [[4.5, 4.5]], rtol=1e-5, atol=1e-6)

    def test_merge_then_unmerge_round_trip(self):
        layer = LoRAAutoLinear(in_features=3, out_features=2, r=2, lora_alpha=4)
        weight = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype="float32")
        lora_A = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype="float32")
        lora_B = np.array([[1.0, 1.0], [0.0, 2.0]], dtype="float32")
        _set(layer.weight, weight)
        _set(layer.bias, [0.5, -0.5])
        _set(layer.lora_A, lora_A)
        _set(layer.lora_B, lora_B)

        x = paddle.to_tensor(np.array([[1.0, 2.0, 3.0]], dtype="float32"))
        before = layer(x).numpy()

        layer.merge()
        self.assertTrue(layer.merged)
        # Merged weight folds the delta (A B * scaling) into the base weight.
        expected_merged = weight + (lora_A @ lora_B) * 2.0
        np.testing.assert_allclose(
            layer.weight.numpy(), expected_merged, rtol=1e-5, atol=1e-6
        )
        # Forward output is unchanged: adapter now lives inside the weight.
        merged_out = layer(x).numpy()
        np.testing.assert_allclose(merged_out, before, rtol=1e-5, atol=1e-6)

        layer.unmerge()
        self.assertFalse(layer.merged)
        np.testing.assert_allclose(
            layer.weight.numpy(), weight, rtol=1e-5, atol=1e-6
        )
        np.testing.assert_allclose(
            layer(x).numpy(), before, rtol=1e-5, atol=1e-6
        )


class TestLoRAAutoLinearDistConfig(unittest.TestCase):
    """auto_dist_config prefixing and ColWise/RowWise plan mapping."""

    def test_default_config_is_empty_plan(self):
        layer = LoRAAutoLinear(in_features=4, out_features=6, r=2)
        config = layer.auto_dist_config()
        self.assertEqual(config, {"mp_config": {"parallelize_plan": {}}})

    def test_prefix_must_end_with_dot(self):
        layer = LoRAAutoLinear(in_features=4, out_features=6, r=2)
        # A non-empty prefix without a trailing dot is rejected.
        with self.assertRaises(AssertionError):
            layer.auto_dist_config(prefix="model")
        # A dotted prefix (and the empty prefix) is accepted.
        self.assertEqual(
            layer.auto_dist_config(prefix="model."),
            {"mp_config": {"parallelize_plan": {}}},
        )

    def test_colwise_plan_maps_to_lora_b_with_prefix(self):
        layer = LoRAAutoLinear(
            in_features=4,
            out_features=6,
            r=2,
            use_intermediate_api=True,
            parallelize_plan=dist.ColWiseParallel(),
        )
        plan = layer.auto_dist_config(prefix="model.")["mp_config"][
            "parallelize_plan"
        ]
        # ColWise parallelism shards the output projection -> lora_B.
        self.assertEqual(list(plan.keys()), ["model.lora_B"])
        self.assertIsInstance(plan["model.lora_B"], dist.ColWiseParallel)

    def test_rowwise_plan_maps_to_lora_a_with_prefix(self):
        layer = LoRAAutoLinear(
            in_features=4,
            out_features=6,
            r=2,
            use_intermediate_api=True,
            parallelize_plan=dist.RowWiseParallel(),
        )
        plan = layer.auto_dist_config(prefix="blk.")["mp_config"][
            "parallelize_plan"
        ]
        # RowWise parallelism shards the input projection -> lora_A.
        self.assertEqual(list(plan.keys()), ["blk.lora_A"])
        self.assertIsInstance(plan["blk.lora_A"], dist.RowWiseParallel)


def _bare_auto_model():
    """Build a LoRAAutoModel without the network-dependent __init__.

    LoRAAutoModel.__init__ calls AutoConfig.from_pretrained on a base model,
    which requires a real pretrained checkpoint. The methods exercised here
    (merge_auto_dist_configs, get_lora_model, mark_only_lora_as_trainable) do
    not depend on that construction, so we skip only the heavy download while
    still running the real nn.Layer machinery and the real method bodies.
    """
    model = LoRAAutoModel.__new__(LoRAAutoModel)
    nn.Layer.__init__(model)
    return model


class TestMergeAutoDistConfigs(unittest.TestCase):
    """LoRAAutoModel.merge_auto_dist_configs dict/list merge semantics."""

    def setUp(self):
        self.model = _bare_auto_model()

    def test_dict_input_returned_unchanged(self):
        config = {"mp_config": {"parallelize_plan": {"a": "planA"}}}
        result = self.model.merge_auto_dist_configs(config)
        # A single dict is a fully-merged config and is passed through as-is.
        self.assertIs(result, config)

    def test_merges_disjoint_mp_plans(self):
        configs = [
            {
                "mp_config": {"parallelize_plan": {"layer0.lora_A": "planA"}},
                "sp_config": None,
                "pp_config": None,
            },
            {
                "mp_config": {"parallelize_plan": {"layer1.lora_B": "planB"}},
                "sp_config": None,
                "pp_config": None,
            },
        ]
        result = self.model.merge_auto_dist_configs(configs)
        self.assertEqual(
            result["mp_config"]["parallelize_plan"],
            {"layer0.lora_A": "planA", "layer1.lora_B": "planB"},
        )
        self.assertIsNone(result["sp_config"])
        self.assertIsNone(result["pp_config"])

    def test_conflicting_mp_key_raises(self):
        configs = [
            {
                "mp_config": {"parallelize_plan": {"shared.lora_A": "planA"}},
                "sp_config": None,
                "pp_config": None,
            },
            {
                "mp_config": {"parallelize_plan": {"shared.lora_A": "planB"}},
                "sp_config": None,
                "pp_config": None,
            },
        ]
        # A sublayer plan must be a subset of the model plan; duplicates fail.
        with self.assertRaises(AssertionError):
            self.model.merge_auto_dist_configs(configs)

    def test_none_mp_config_is_skipped(self):
        configs = [
            {"mp_config": None, "sp_config": None, "pp_config": None},
            {
                "mp_config": {"parallelize_plan": {"only.lora_A": "planA"}},
                "sp_config": None,
                "pp_config": None,
            },
        ]
        result = self.model.merge_auto_dist_configs(configs)
        self.assertEqual(
            result["mp_config"]["parallelize_plan"], {"only.lora_A": "planA"}
        )

    def test_merges_disjoint_sp_plans(self):
        configs = [
            {
                "mp_config": None,
                "sp_config": {"parallelize_plan": {"a.lora_A": "spA"}},
                "pp_config": None,
            },
            {
                "mp_config": None,
                "sp_config": {"parallelize_plan": {"b.lora_B": "spB"}},
                "pp_config": None,
            },
        ]
        result = self.model.merge_auto_dist_configs(configs)
        self.assertEqual(
            result["sp_config"]["parallelize_plan"],
            {"a.lora_A": "spA", "b.lora_B": "spB"},
        )
        self.assertIsNone(result["mp_config"])


class TestGetLoraModelReplacement(unittest.TestCase):
    """Target-module replacement + freeze/trainable contract on CPU."""

    def test_only_targeted_linear_is_replaced_and_shares_weight(self):
        model = _bare_auto_model()
        base = _TwoLinear()
        orig_w1 = base.linear1.weight
        orig_b1 = base.linear1.bias
        lora_config = LoRAAutoConfig(
            target_modules=["linear1"], r=2, lora_alpha=4
        )

        replaced = model.get_lora_model(base, lora_config)

        # Only linear1 (the matched target) becomes a LoRAAutoLinear.
        self.assertIsInstance(replaced.linear1, LoRAAutoLinear)
        self.assertIsInstance(replaced.linear2, nn.Linear)
        self.assertNotIsInstance(replaced.linear2, LoRAAutoLinear)
        # The base weight/bias tensors are re-used, not re-created.
        self.assertIs(replaced.linear1.weight, orig_w1)
        self.assertIs(replaced.linear1.bias, orig_b1)
        # Adapter is shaped from the wrapped linear (in=4, out=6).
        self.assertEqual(replaced.linear1.r, 2)
        self.assertEqual(list(replaced.linear1.lora_A.shape), [4, 2])
        self.assertEqual(list(replaced.linear1.lora_B.shape), [2, 6])

    def test_mark_only_lora_as_trainable(self):
        model = _bare_auto_model()
        base = _TwoLinear()
        lora_config = LoRAAutoConfig(
            target_modules=["linear1"], r=2, lora_alpha=4
        )
        model.model = model.get_lora_model(base, lora_config)
        model.lora_config = lora_config

        model.mark_only_lora_as_trainable()

        adapter = model.model.linear1
        # Adapter params train; the folded base weight stays frozen.
        self.assertFalse(adapter.lora_A.stop_gradient)
        self.assertFalse(adapter.lora_B.stop_gradient)
        self.assertTrue(adapter.weight.stop_gradient)
        # A non-targeted plain Linear is fully frozen (trainable_bias=None).
        self.assertTrue(model.model.linear2.weight.stop_gradient)
        self.assertTrue(model.model.linear2.bias.stop_gradient)


class TestLoRAAutoLayerRegistry(unittest.TestCase):
    """Module-level registry / restore-map identities."""

    def test_layer_registry_identities(self):
        self.assertIs(lora_layers["LoRAAutoLinear"], LoRAAutoLinear)
        self.assertIn(LoRAAutoLinear, AVAILABLE_LAYERS)
        # LoRAAutoLinear restores back to a plain nn.Linear.
        self.assertIs(
            LoRAAutoModel.restore_layer_map[LoRAAutoLinear], nn.Linear
        )


if __name__ == "__main__":
    unittest.main()

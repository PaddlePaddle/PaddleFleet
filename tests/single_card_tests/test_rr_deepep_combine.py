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

"""
Unit tests for MoE + Recompute (Refined Recompute deepep_combine) scenarios.

Tests cover:
- rr_recompute_update configuration logic (standalone extraction)
- DeepEPCombineAsyncRefinedRecompute queue management
- Error handling for invalid configurations
- need_recompute_in_first_n correctness
"""

import queue
import unittest
from unittest.mock import MagicMock, patch

from paddlefleet.recompute_utils import need_recompute_in_first_n
from paddlefleet.transformer.transformer_config import TransformerConfig


def _has_deep_ep():
    """Check whether DeepEP runtime is available."""
    try:
        from paddlefleet.transformer.moe.fused_a2a import (
            DeepEPCombineAsyncRefinedRecompute,
        )

        return True
    except (ImportError, ModuleNotFoundError):
        return False


def rr_recompute_update(config, layer_number, in_full_recompute, in_mlp_recompute):
    """
    Standalone version of MoELayer.rr_recompute_update for testing.
    This mirrors the logic in moe_layer.py and is used to verify the config validation
    without needing to instantiate a full MoELayer (which requires distributed setup).
    """
    use_rr_deepep_combine = False
    if (
        config.recompute_modules is not None
        and "moe_combine" in config.recompute_modules
    ):
        if config.recompute_granularity is None:
            raise ValueError(
                "recompute_granularity must be set when moe_combine RR is enabled."
            )
        if isinstance(config.recompute_modules, list):
            use_rr_deepep_combine = True
        elif isinstance(config.recompute_modules, dict):
            use_rr_deepep_combine = not need_recompute_in_first_n(
                layer_number,
                config,
                config.recompute_modules["moe_combine"],
            )
    if (not in_full_recompute) and (not in_mlp_recompute) and use_rr_deepep_combine:
        raise ValueError(
            "Enabling rr for moe_combine is meaningless when neither full_recompute "
            "nor mlp_recompute is active."
        )
    return use_rr_deepep_combine


class TestRRRecomputeUpdate(unittest.TestCase):
    """Tests for rr_recompute_update configuration logic."""

    def _make_config(self, **overrides):
        """Helper to create a TransformerConfig for testing."""
        defaults = {
            "hidden_size": 64,
            "num_attention_heads": 2,
            "intermediate_size": 256,
            "num_hidden_layers": 8,
            "recompute_granularity": "full",
            "recompute_method": "uniform",
            "recompute_num_layers": 1,
        }
        defaults.update(overrides)
        return TransformerConfig(**defaults)

    def test_rr_enabled_with_list_recompute_modules(self):
        """When recompute_modules is a list containing 'moe_combine', RR should be enabled."""
        config = self._make_config(recompute_modules=["moe_combine"])
        result = rr_recompute_update(config, 0, in_full_recompute=True, in_mlp_recompute=False)
        self.assertTrue(result)

    def test_rr_disabled_without_moe_combine_in_modules(self):
        """When recompute_modules does not contain 'moe_combine', RR should not be enabled."""
        config = self._make_config(recompute_modules=["attention"])
        result = rr_recompute_update(config, 0, in_full_recompute=True, in_mlp_recompute=False)
        self.assertFalse(result)

    def test_rr_disabled_when_recompute_modules_none(self):
        """When recompute_modules is None, RR should not be enabled."""
        config = self._make_config(recompute_modules=None)
        result = rr_recompute_update(config, 0, in_full_recompute=True, in_mlp_recompute=False)
        self.assertFalse(result)

    def test_rr_with_dict_recompute_modules_first_n(self):
        """When recompute_modules is a dict with 'moe_combine': N, use first_n logic."""
        config = self._make_config(
            num_hidden_layers=8,
            recompute_modules={"moe_combine": 4},
            recompute_granularity="full",
            recompute_method="first_n",
            recompute_num_layers=4,
        )
        # Layer 0 should be in recompute range -> need_recompute_in_first_n=True -> use_rr=False
        result_0 = rr_recompute_update(config, 0, in_full_recompute=True, in_mlp_recompute=False)
        self.assertFalse(result_0)

        # Layer 5 should NOT be in recompute range -> need_recompute_in_first_n=False -> use_rr=True
        result_5 = rr_recompute_update(config, 5, in_full_recompute=True, in_mlp_recompute=False)
        self.assertTrue(result_5)

    def test_raises_when_recompute_granularity_none(self):
        """Should raise ValueError when recompute_granularity is None but moe_combine RR is configured."""
        config = self._make_config(
            recompute_modules=["moe_combine"],
            recompute_granularity=None,
        )
        with self.assertRaises(ValueError) as cm:
            rr_recompute_update(config, 0, in_full_recompute=True, in_mlp_recompute=False)
        self.assertIn("recompute_granularity", str(cm.exception))

    def test_raises_when_no_recompute_but_rr_enabled(self):
        """Should raise ValueError when neither full nor mlp recompute is active but RR is enabled."""
        config = self._make_config(recompute_modules=["moe_combine"])
        with self.assertRaises(ValueError) as cm:
            rr_recompute_update(config, 0, in_full_recompute=False, in_mlp_recompute=False)
        self.assertIn("meaningless", str(cm.exception))

    def test_no_error_when_rr_disabled_and_no_recompute(self):
        """Should not raise when RR is not enabled and recompute is off."""
        config = self._make_config(recompute_modules=["attention"])
        result = rr_recompute_update(config, 0, in_full_recompute=False, in_mlp_recompute=False)
        self.assertFalse(result)

    def test_rr_enabled_with_mlp_recompute(self):
        """RR should be allowed when mlp recompute is active."""
        config = self._make_config(recompute_modules=["moe_combine"])
        result = rr_recompute_update(config, 0, in_full_recompute=False, in_mlp_recompute=True)
        self.assertTrue(result)


@unittest.skipUnless(_has_deep_ep(), "DeepEP not available")
class TestDeepEPCombineAsyncRefinedRecompute(unittest.TestCase):
    """Tests for DeepEPCombineAsyncRefinedRecompute class."""

    def test_queue_empty_raises_runtime_error(self):
        """Should raise RuntimeError when queue is empty on second forward."""
        from paddlefleet.transformer.moe.fused_a2a import (
            DeepEPCombineAsyncRefinedRecompute,
        )

        rr = DeepEPCombineAsyncRefinedRecompute()
        # Simulate second forward without first forward (queue empty)
        with patch("paddle.framework._dygraph_tracer") as mock_tracer:
            mock_tracer.return_value._has_grad = True  # is_first_fwd = False
            with self.assertRaises(RuntimeError) as cm:
                rr.forward(
                    MagicMock(),  # x
                    MagicMock(),  # group
                    {"handle": MagicMock()},  # states
                    MagicMock(),  # fn_args
                    fn=MagicMock(),
                )
            self.assertIn("Queue is empty", str(cm.exception))

    def test_fn_none_raises_value_error(self):
        """Should raise ValueError when fn is None in _first_fwd."""
        from paddlefleet.transformer.moe.fused_a2a import (
            DeepEPCombineAsyncRefinedRecompute,
        )

        rr = DeepEPCombineAsyncRefinedRecompute()
        with self.assertRaises(ValueError) as cm:
            rr._first_fwd(
                MagicMock(),  # x
                MagicMock(),  # group
                {"handle": MagicMock()},  # states
                None,  # fn = None
                True,  # is_first_fwd
            )
        self.assertIn("fn must not be None", str(cm.exception))

    def test_queue_initialization(self):
        """Queue should be initialized and empty after construction."""
        from paddlefleet.transformer.moe.fused_a2a import (
            DeepEPCombineAsyncRefinedRecompute,
        )

        rr = DeepEPCombineAsyncRefinedRecompute()
        self.assertIsInstance(rr._hold_tensors_queue, queue.Queue)
        self.assertTrue(rr._hold_tensors_queue.empty())


@unittest.skipUnless(_has_deep_ep(), "DeepEP not available")
class TestFusedCombineRRValidation(unittest.TestCase):
    """Tests for fused_combine parameter validation with RR."""

    def test_use_rr_without_combine_overlap_handle_raises(self):
        """Should raise ValueError when use_rr_deepep_combine=True but combine_overlap_handle is None."""
        from paddlefleet.transformer.moe.fused_a2a import fused_combine

        if fused_combine is None:
            self.skipTest("DeepEP not available, fused_combine is None")

        with self.assertRaises(ValueError) as cm:
            fused_combine(
                x=MagicMock(),
                group=MagicMock(),
                handle=MagicMock(),
                _rr_fusedcombined=None,
                previous_event=None,
                combine_overlap_handle=None,
                async_finish=False,
                moe_ep_barrier=True,
                use_rr_deepep_combine=True,
            )
        self.assertIn("combine_overlap_handle", str(cm.exception))

    def test_use_rr_without_rr_functor_raises(self):
        """Should raise ValueError when use_rr_deepep_combine=True but _rr_fusedcombined is None."""
        from paddlefleet.transformer.moe.fused_a2a import fused_combine

        if fused_combine is None:
            self.skipTest("DeepEP not available, fused_combine is None")

        combine_overlap_handle = {
            "fn": MagicMock(),
            "fn_args": (MagicMock(),),
        }

        with self.assertRaises(ValueError) as cm:
            fused_combine(
                x=MagicMock(),
                group=MagicMock(),
                handle=MagicMock(),
                _rr_fusedcombined=None,
                previous_event=None,
                combine_overlap_handle=combine_overlap_handle,
                async_finish=False,
                moe_ep_barrier=True,
                use_rr_deepep_combine=True,
            )
        self.assertIn("_rr_fusedcombined must be provided", str(cm.exception))


class TestNeedRecomputeInFirstN(unittest.TestCase):
    """Tests for need_recompute_in_first_n used by rr_recompute_update."""

    def test_basic_first_n(self):
        """First N layers should need recompute, rest should not."""
        config = TransformerConfig(
            num_hidden_layers=8,
            recompute_granularity="full",
            recompute_method="first_n",
            recompute_num_layers=4,
        )
        # Layers 0-3 should need recompute
        for i in range(4):
            self.assertTrue(need_recompute_in_first_n(i, config, 4))
        # Layers 4-7 should NOT need recompute
        for i in range(4, 8):
            self.assertFalse(need_recompute_in_first_n(i, config, 4))

    def test_first_n_with_pp(self):
        """First N layers per PP stage should need recompute."""
        config = TransformerConfig(
            num_hidden_layers=8,
            pipeline_model_parallel_size=2,
            recompute_granularity="full",
            recompute_method="first_n",
            recompute_num_layers=2,
        )
        # PP stage 0: layers 0-3, first 2 -> layers 0, 1 recompute
        self.assertTrue(need_recompute_in_first_n(0, config, 2))
        self.assertTrue(need_recompute_in_first_n(1, config, 2))
        self.assertFalse(need_recompute_in_first_n(2, config, 2))
        self.assertFalse(need_recompute_in_first_n(3, config, 2))
        # PP stage 1: layers 4-7, first 2 -> layers 4, 5 recompute
        self.assertTrue(need_recompute_in_first_n(4, config, 2))
        self.assertTrue(need_recompute_in_first_n(5, config, 2))
        self.assertFalse(need_recompute_in_first_n(6, config, 2))
        self.assertFalse(need_recompute_in_first_n(7, config, 2))


if __name__ == "__main__":
    unittest.main()

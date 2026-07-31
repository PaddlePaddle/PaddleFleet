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

import os
import sys

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "..",
        "..",
        "..",
        "src",
    ),
)

import unittest
from unittest.mock import MagicMock, patch

import paddle

from paddlefleet.transformer.moe.fp8_utils import quant_blockwize
from paddlefleet.transformer.transformer_config import TransformerConfig


def _make_config(silu_quant_high_precision=False, **overrides):
    """Create a TransformerConfig suitable for SharedExpert testing."""
    defaults = {
        "hidden_size": 128,
        "num_attention_heads": 2,
        "intermediate_size": 256,
        "gated_linear_unit": True,
        "sequence_parallel": False,
        "tensor_model_parallel_size": 1,
        "fp8": "e4m3",
        "fp8_wgrad": True,
        "use_bias": False,
        "use_cpu_initialization": True,
        "moe_shared_expert_gate": False,
        "silu_quant_high_precision_in_shared_expert": silu_quant_high_precision,
    }
    defaults.update(overrides)
    return TransformerConfig(**defaults)


class TestSiluQuantHighPrecisionInSharedExpert(unittest.TestCase):
    """Test silu_quant_high_precision_in_shared_expert=True path in StandardMLPSharedExpert."""

    def _build_expert(self, config):
        """Build StandardMLPSharedExpert with mocked parallel state."""
        from paddlefleet.tensor_parallel import (
            ColumnParallelLinear,
            RowParallelLinear,
        )
        from paddlefleet.transformer.mlp import MLPSublayersSpec
        from paddlefleet.transformer.moe.moe_shared_expert import (
            StandardMLPSharedExpert,
        )

        mlp_spec = MLPSublayersSpec(
            up_gate_proj=ColumnParallelLinear,
            hidden_act=None,
            down_proj=RowParallelLinear,
        )
        return StandardMLPSharedExpert(
            config=config,
            moe_intermediate_size=config.intermediate_size,
            is_expert=False,
            mlp_spec=mlp_spec,
        )

    @patch("paddlefleet.tensor_parallel.random.get_cuda_rng_tracker")
    def test_enabled_sets_silu_return_high_precision(self, mock_rng_tracker):
        """When enabled, silu_return_high_precision should be True."""
        mock_rng_tracker.return_value.fork = MagicMock(
            return_value=_noop_context()
        )
        config = _make_config(silu_quant_high_precision=True)
        expert = self._build_expert(config)
        self.assertTrue(expert.silu_return_high_precision)

    @patch("paddlefleet.tensor_parallel.random.get_cuda_rng_tracker")
    def test_enabled_sets_inp_quant_func_on_down_proj(self, mock_rng_tracker):
        """When enabled, down_proj.inp_quant_func should be a callable."""
        mock_rng_tracker.return_value.fork = MagicMock(
            return_value=_noop_context()
        )
        config = _make_config(silu_quant_high_precision=True)
        expert = self._build_expert(config)
        self.assertTrue(callable(expert.down_proj.inp_quant_func))

    @patch("paddlefleet.tensor_parallel.random.get_cuda_rng_tracker")
    def test_enabled_sets_input_in_high_precision_on_down_proj(
        self, mock_rng_tracker
    ):
        """When enabled, down_proj.input_in_high_precision should be True."""
        mock_rng_tracker.return_value.fork = MagicMock(
            return_value=_noop_context()
        )
        config = _make_config(silu_quant_high_precision=True)
        expert = self._build_expert(config)
        self.assertTrue(expert.down_proj.input_in_high_precision)

    @patch("paddlefleet.tensor_parallel.random.get_cuda_rng_tracker")
    def test_disabled_does_not_set_silu_return_high_precision(
        self, mock_rng_tracker
    ):
        """When disabled, silu_return_high_precision should not be set."""
        mock_rng_tracker.return_value.fork = MagicMock(
            return_value=_noop_context()
        )
        config = _make_config(silu_quant_high_precision=False)
        expert = self._build_expert(config)
        self.assertFalse(getattr(expert, "silu_return_high_precision", False))

    @patch("paddlefleet.tensor_parallel.random.get_cuda_rng_tracker")
    def test_disabled_does_not_set_inp_quant_func(self, mock_rng_tracker):
        """When disabled, down_proj should not have inp_quant_func."""
        mock_rng_tracker.return_value.fork = MagicMock(
            return_value=_noop_context()
        )
        config = _make_config(silu_quant_high_precision=False)
        expert = self._build_expert(config)
        self.assertIsNone(getattr(expert.down_proj, "inp_quant_func", None))

    def test_quant_blockwize_fp8_output(self):
        """quant_blockwize should produce fp8 quantized output with correct shape."""
        x = paddle.randn([2, 128], dtype="float32")
        q, sf = quant_blockwize(
            x, quant_method="1x128", quant_dtype="fp8", using_ue8m0_scale=True
        )

        self.assertEqual(q.dtype, paddle.float8_e4m3fn)
        self.assertEqual(q.shape, [2, 128])
        self.assertEqual(sf.dtype, paddle.float32)
        # With block size 128, scale factor should be [2, 1]
        self.assertEqual(sf.shape, [2, 1])

    def test_config_validation_requires_fp8(self):
        """silu_quant_high_precision_in_shared_expert requires fp8 to be set."""
        with self.assertRaises(AssertionError):
            _make_config(silu_quant_high_precision=True, fp8=None)


import contextlib


@contextlib.contextmanager
def _noop_context(*args, **kwargs):
    yield


if __name__ == "__main__":
    unittest.main()

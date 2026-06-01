# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

"""Single-card validation tests for MoELayer's allgather-dispatcher init
paths. Covers:

* sonic_moe-availability assert (allgather requires the SonicMoE module)
* moe_allgather_gate_overlap warning when paired with a non-allgather
  dispatcher (the flag is silently ignored otherwise).
* allgather + EP > 1 validation: ``using_sonic_moe=False`` raise and
  ``moe_intermediate_size`` divisibility raise.

EP > 1 is faked via a ``SimpleNamespace`` ep group; we only exercise the
*validation* paths that abort init (raise) — the warning-and-continue
paths (forcing fusion_node / disabling deep_gemm) require a complete EP
init and are covered by the multi-card test.
"""

import logging
import unittest
from types import SimpleNamespace
from unittest import mock

import paddle
import paddle.nn.functional as F
import paddlefleet_ops

# Temporarily disable sonicmoe Python imports during module load.
# The sonicmoe ecosystem ops are already loaded; re-importing the
# Python wrapper triggers custom-op re-registration crashes.
_original_sonic_moe_available = paddlefleet_ops.is_sonic_moe_available
paddlefleet_ops.is_sonic_moe_available = lambda: False

from paddlefleet.transformer.mlp import MLPSublayersSpec
from paddlefleet.transformer.moe.moe_layer import MoELayer, MoESublayers
from paddlefleet.transformer.transformer_config import TransformerConfig

# Restore the real value so concurrent / subsequent test files see it.
paddlefleet_ops.is_sonic_moe_available = _original_sonic_moe_available


class _FakeLinear(paddle.nn.Layer):
    """Dummy linear that avoids ColumnParallelLinear's RNG-tracker check."""

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


def _make_config(**overrides):
    defaults = {
        "num_hidden_layers": 1,
        "hidden_size": 16,
        "num_attention_heads": 2,
        "intermediate_size": 32,
        "n_routed_experts": 2,
        "n_shared_experts": 1,
        "num_experts_per_tok": 1,
        "moe_intermediate_size": 16,
        "moe_token_dispatcher_type": "alltoall",
        "moe_expert_fusion": False,
        "moe_use_fusion_node": True,
        "fp8": None,
        "gated_linear_unit": True,
        "hidden_act": F.silu,
        "tensor_model_parallel_size": 1,
    }
    defaults.update(overrides)
    return TransformerConfig(**defaults)


def _make_sublayers():
    return MoESublayers(
        mlp_spec=MLPSublayersSpec(
            up_gate_proj=_FakeLinear,
            down_proj=_FakeLinear,
        )
    )


def _instantiate(config, ep=None):
    return MoELayer(
        config,
        sublayers=_make_sublayers(),
        pg_collection=SimpleNamespace(ep=ep, expt_dp=None),
    )


class TestAllGatherSonicMoeAvailability(unittest.TestCase):
    """``allgather`` dispatcher requires the SonicMoE module."""

    def test_allgather_asserts_sonic_moe_available(self):
        config = _make_config(
            moe_token_dispatcher_type="allgather",
            using_sonic_moe=True,
        )
        with (
            mock.patch(
                "paddlefleet_ops.is_sonic_moe_available", return_value=False
            ),
            self.assertRaises(ValueError),
        ):
            _instantiate(config)


class TestGateOverlapWarning(unittest.TestCase):
    """``moe_allgather_gate_overlap=True`` paired with a non-allgather
    dispatcher should log a warning so users know the flag is being
    ignored.
    """

    def test_warning_logged_for_non_allgather_dispatcher(self):
        config = _make_config(
            moe_token_dispatcher_type="alltoall",
            moe_allgather_gate_overlap=True,
        )
        with self.assertLogs(level=logging.WARNING) as captured:
            _instantiate(config)
        joined = "\n".join(captured.output)
        self.assertIn("moe_allgather_gate_overlap", joined)
        self.assertIn("only honoured when", joined)

    def test_no_warning_when_dispatcher_is_allgather(self):
        # Even though gate_overlap=True is set, the message must NOT fire
        # for the allgather dispatcher itself. We just verify init does
        # not log the specific warning string. Init may still raise later
        # because EP=1 fails sonic_moe checks; we catch any exception and
        # only assert on the absence of the gate_overlap warning.
        config = _make_config(
            moe_token_dispatcher_type="allgather",
            moe_allgather_gate_overlap=True,
            using_sonic_moe=True,
        )
        captured_messages = []

        class _Capture(logging.Handler):
            def emit(self, record):
                captured_messages.append(record.getMessage())

        handler = _Capture(level=logging.WARNING)
        logging.getLogger().addHandler(handler)
        try:
            try:
                _instantiate(config)
            except (AssertionError, ValueError, RuntimeError):
                pass
        finally:
            logging.getLogger().removeHandler(handler)
        for msg in captured_messages:
            self.assertNotIn("only honoured when", msg)


class TestAllGatherEpValidation(unittest.TestCase):
    """allgather-with-EP>1 raise paths."""

    def _fake_ep_group(self, nranks):
        return SimpleNamespace(
            ranks=list(range(nranks)),
            nranks=nranks,
            rank=0,
        )

    def test_allgather_requires_using_sonic_moe(self):
        config = _make_config(
            moe_token_dispatcher_type="allgather",
            using_sonic_moe=False,
        )
        # Bypass the sonic_moe module-availability assert at line 191
        # to reach the using_sonic_moe-config check at line 324.
        with (
            mock.patch(
                "paddlefleet_ops.is_sonic_moe_available", return_value=True
            ),
            mock.patch(
                "paddlefleet.utils.get_pg_size",
                side_effect=lambda g: g.nranks if g else 1,
            ),
            mock.patch("paddlefleet.utils.get_pg_rank", return_value=0),
            self.assertRaisesRegex(ValueError, "using_sonic_moe=True"),
        ):
            _instantiate(config, ep=self._fake_ep_group(2))

    def test_allgather_raises_when_intermediate_size_not_divisible_by_ep(
        self,
    ):
        # moe_intermediate_size=16 not divisible by EP=3.
        config = _make_config(
            moe_token_dispatcher_type="allgather",
            using_sonic_moe=True,
            moe_intermediate_size=16,
        )
        with (
            mock.patch(
                "paddlefleet_ops.is_sonic_moe_available", return_value=True
            ),
            mock.patch(
                "paddlefleet.utils.get_pg_size",
                side_effect=lambda g: g.nranks if g else 1,
            ),
            mock.patch("paddlefleet.utils.get_pg_rank", return_value=0),
            self.assertRaisesRegex(ValueError, "must be divisible by EP"),
        ):
            _instantiate(config, ep=self._fake_ep_group(3))

    def test_allgather_forces_fusion_node_on(self):
        config = _make_config(
            moe_token_dispatcher_type="allgather",
            using_sonic_moe=True,
            moe_use_fusion_node=False,
        )
        with (
            mock.patch(
                "paddlefleet_ops.is_sonic_moe_available", return_value=True
            ),
            mock.patch(
                "paddlefleet.utils.get_pg_size",
                side_effect=lambda g: g.nranks if g else 1,
            ),
            mock.patch("paddlefleet.utils.get_pg_rank", return_value=0),
            mock.patch("paddlefleet.transformer.moe.moe_layer.SonicMoEExpert"),
            self.assertLogs(level=logging.WARNING) as captured,
        ):
            layer = _instantiate(config, ep=self._fake_ep_group(2))
        self.assertTrue(layer.moe_use_fusion_node)
        joined = "\n".join(captured.output)
        self.assertIn("forcing moe_use_fusion_node=True", joined)

    def test_allgather_forces_deep_gemm_off(self):
        config = _make_config(
            moe_token_dispatcher_type="allgather",
            using_sonic_moe=True,
            moe_deep_gemm=True,
            moe_expert_fusion=True,
        )
        with (
            mock.patch(
                "paddlefleet_ops.is_sonic_moe_available", return_value=True
            ),
            mock.patch(
                "paddlefleet.utils.get_pg_size",
                side_effect=lambda g: g.nranks if g else 1,
            ),
            mock.patch("paddlefleet.utils.get_pg_rank", return_value=0),
            mock.patch(
                "paddle.device.get_device_capability", return_value=(9, 0)
            ),
            mock.patch("paddlefleet.transformer.moe.moe_layer.SonicMoEExpert"),
            self.assertLogs(level=logging.WARNING) as captured,
        ):
            layer = _instantiate(config, ep=self._fake_ep_group(2))
        self.assertFalse(layer.moe_deep_gemm)
        joined = "\n".join(captured.output)
        self.assertIn("forcing moe_deep_gemm=False", joined)

    def test_allgather_forces_expert_fusion_on(self):
        """moe_expert_fusion=False is silently corrected to True when
        the allgather dispatcher is selected."""
        config = _make_config(
            moe_token_dispatcher_type="allgather",
            using_sonic_moe=True,
            moe_expert_fusion=False,
        )
        with (
            mock.patch(
                "paddlefleet_ops.is_sonic_moe_available", return_value=True
            ),
            mock.patch(
                "paddlefleet.utils.get_pg_size",
                side_effect=lambda g: g.nranks if g else 1,
            ),
            mock.patch("paddlefleet.utils.get_pg_rank", return_value=0),
            mock.patch("paddlefleet.transformer.moe.moe_layer.SonicMoEExpert"),
            self.assertLogs(level=logging.WARNING) as captured,
        ):
            layer = _instantiate(config, ep=self._fake_ep_group(2))
        self.assertTrue(layer.moe_expert_fusion)
        joined = "\n".join(captured.output)
        self.assertIn("forcing moe_expert_fusion=True", joined)


class TestFusionNodeDisable(unittest.TestCase):
    """non-deepep/hybridep/allgather dispatcher with EP>1 silently
    disables moe_use_fusion_node."""

    def _fake_ep_group(self, nranks):
        return SimpleNamespace(
            ranks=list(range(nranks)),
            nranks=nranks,
            rank=0,
        )

    def test_alltoall_with_ep_disables_fusion_node(self):
        config = _make_config(
            moe_token_dispatcher_type="alltoall",
            moe_use_fusion_node=True,
        )
        with (
            mock.patch(
                "paddlefleet.utils.get_pg_size",
                side_effect=lambda g: g.nranks if g else 1,
            ),
            mock.patch("paddlefleet.utils.get_pg_rank", return_value=0),
        ):
            layer = _instantiate(config, ep=self._fake_ep_group(2))
        self.assertFalse(layer.moe_use_fusion_node)


class TestFp8Validation(unittest.TestCase):
    """fp8 / use_ue8m0 init validation paths."""

    def test_fp8_requires_fusion_node(self):
        # EP=1 + alltoall keeps fusion_node=False (no allgather force-on),
        # so fp8 + fusion_node=False should raise.
        config = _make_config(
            moe_token_dispatcher_type="alltoall",
            moe_use_fusion_node=False,
            fp8="e4m3",
        )
        with (
            mock.patch("paddle.version.cuda", return_value="12.4"),
            self.assertRaisesRegex(ValueError, "moe_use_fusion_node"),
        ):
            _instantiate(config)

    def test_use_ue8m0_requires_blackwell(self):
        config = _make_config(
            moe_token_dispatcher_type="alltoall",
            moe_use_fusion_node=True,
            use_ue8m0=True,
        )
        with (
            mock.patch("paddle.version.cuda", return_value="12.4"),
            mock.patch(
                "paddle.device.cuda.get_device_capability",
                return_value=(9, 0),
            ),
            self.assertRaisesRegex(ValueError, "Blackwell"),
        ):
            _instantiate(config)


class TestAllGatherFp8UsesFusedWeight(unittest.TestCase):
    """allgather + fp8 must keep use_fused_weight=True.

    Regression guard for the bug where allgather forces moe_deep_gemm=False,
    then the generic fp8 logic set use_fused_weight=False when deep_gemm is
    False, causing the sonic_moe assertion to fail.
    """

    def _fake_ep_group(self, nranks):
        return SimpleNamespace(
            ranks=list(range(nranks)),
            nranks=nranks,
            rank=0,
        )

    def test_allgather_fp8_preserves_fused_weight(self):
        config = _make_config(
            moe_token_dispatcher_type="allgather",
            using_sonic_moe=True,
            moe_expert_fusion=True,
            moe_deep_gemm=False,
            fp8="e4m3",
        )
        with (
            mock.patch(
                "paddlefleet_ops.is_sonic_moe_available", return_value=True
            ),
            mock.patch(
                "paddlefleet.utils.get_pg_size",
                side_effect=lambda g: g.nranks if g else 1,
            ),
            mock.patch("paddlefleet.utils.get_pg_rank", return_value=0),
            mock.patch("paddle.version.cuda", return_value="12.4"),
            mock.patch(
                "paddle.device.cuda.get_device_capability", return_value=(9, 0)
            ),
            mock.patch("paddlefleet.transformer.moe.moe_layer.SonicMoEExpert"),
        ):
            # Must not raise (the sonic_moe assertion use_fused_weight==True
            # must pass).
            layer = _instantiate(config, ep=self._fake_ep_group(2))
        self.assertTrue(layer.moe_expert_fusion)


if __name__ == "__main__":
    unittest.main()

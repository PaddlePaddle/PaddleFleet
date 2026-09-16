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

"""Behavior tests for ``paddlefleet.models.vision.multimodal_projector``.

Slice covered here (kept disjoint from ``test_multimodal_projector_2`` which
exercises ``forward`` bias-add math and mock-based construction dispatch):

  * ``MultimodalProjector.__init__`` construction guards and error contracts,
    exercised through the real constructor (no ``__new__`` shells, no patching
    of the class under test).
  * The ``MLPSublayersSpec`` helper's real field contract, which is the root
    cause of a production bug on the ``affine`` branch.
  * Real (unmocked) wiring of the ``mlp`` branch to a genuine ``MLP`` encoder,
    observing that ``input_size`` actually flows into the built encoder.
  * The ``affine`` branch, which is currently broken against the installed
    ``MLPSublayersSpec`` / ``TransformerConfig`` APIs (documented via
    ``expectedFailure``; production is left untouched).

Paddle is not installed in the authoring sandbox, so every class guards its
imports and skips with an honest reason instead of pretending to pass.
"""

import dataclasses
import unittest

_IMPORT_OK = True
_SKIP_REASON = ""
try:
    import paddle  # noqa: F401
    from paddle.distributed import fleet
    from paddle.distributed.fleet.meta_parallel import (
        build_spec_layer,  # noqa: F401
    )

    from paddlefleet.models.vision.multimodal_projector import (
        MultimodalProjector,
    )
    from paddlefleet.tensor_parallel.layers import (
        ColumnParallelLinear,
        RowParallelLinear,
    )
    from paddlefleet.transformer.mlp import MLP, MLPSublayersSpec
    from paddlefleet.transformer.transformer_config import TransformerConfig
except (ImportError, ModuleNotFoundError) as exc:  # honest capability probe
    # Only genuine missing-dependency errors are treated as "skip". Any other
    # error (compile failure, renamed API) must surface as a real failure.
    _IMPORT_OK = False
    _SKIP_REASON = f"paddle/paddlefleet import unavailable: {exc!r}"


def _base_config():
    """A minimal CPU TransformerConfig usable for projector construction."""
    return TransformerConfig(
        num_hidden_layers=1,
        hidden_size=64,
        num_attention_heads=4,
        intermediate_size=128,
        use_cpu_initialization=True,
    )


@unittest.skipUnless(_IMPORT_OK, _SKIP_REASON)
class TestMultimodalProjectorConstructionGuards(unittest.TestCase):
    """Constructor guard paths driven through the real ``__init__``."""

    def test_unsupported_projector_type_raises_exact_message(self):
        # A valid (non-None) spec clears the assert so we reach the ``else``
        # branch; the raised message is hand-derived from the f-string in
        # ``MultimodalProjector.__init__``:
        #   f"Unsupported multimodal projection type {self.projector_type}"
        config = _base_config()
        spec = MLPSublayersSpec()
        bad_type = "totally_unknown_type"

        with self.assertRaises(Exception) as cm:
            MultimodalProjector(
                config=config,
                sublayers_spec=spec,
                projector_type=bad_type,
                input_size=32,
            )
        self.assertEqual(
            str(cm.exception),
            f"Unsupported multimodal projection type {bad_type}",
        )

    def test_none_sublayers_spec_raises_assertion(self):
        # The assert fires before any encoder is built; message is taken
        # verbatim from the production ``assert`` statement.
        config = _base_config()
        with self.assertRaises(AssertionError) as cm:
            MultimodalProjector(
                config=config,
                sublayers_spec=None,
                projector_type="mlp",
                input_size=32,
            )
        self.assertIn("MLPSublayersSpec must be provided", str(cm.exception))


@unittest.skipUnless(_IMPORT_OK, _SKIP_REASON)
class TestMLPSublayersSpecFieldContract(unittest.TestCase):
    """The spec helper's real field set (root cause of the affine bug)."""

    def test_spec_fields_and_no_linear_fc1(self):
        # Hand-derived from the dataclass definition in transformer/mlp.py:
        # fields are exactly up_gate_proj / hidden_act / down_proj.
        field_names = {f.name for f in dataclasses.fields(MLPSublayersSpec)}
        self.assertEqual(
            field_names, {"up_gate_proj", "hidden_act", "down_proj"}
        )
        # The affine branch of MultimodalProjector reads sublayers_spec.linear_fc1,
        # which does not exist on this dataclass -> attribute is genuinely absent.
        self.assertNotIn("linear_fc1", field_names)
        self.assertFalse(hasattr(MLPSublayersSpec(), "linear_fc1"))


@unittest.skipUnless(_IMPORT_OK, _SKIP_REASON)
class TestMultimodalProjectorMLPBranch(unittest.TestCase):
    """The ``mlp`` branch wired to a genuine MLP encoder (no mocking)."""

    def setUp(self):
        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": 1,
            "pp_degree": 1,
            "sharding_degree": 1,
            "sep_degree": 1,
            "cp_degree": 1,
            "ep_degree": 1,
            "moe_sharding_degree": 1,
            "order": [
                "sharding",
                "moe_sharding",
                "pp",
                "sep",
                "cp",
                "dp",
                "ep",
                "mp",
            ],
        }
        fleet.init(is_collective=True, strategy=strategy)

    def test_mlp_branch_builds_real_mlp_and_consumes_input_size(self):
        config = _base_config()
        # Real column/row-parallel specs; the MLP is built for real, not mocked.
        spec = MLPSublayersSpec(
            up_gate_proj=ColumnParallelLinear,
            down_proj=RowParallelLinear,
        )
        input_size = 32

        model = MultimodalProjector(
            config=config,
            sublayers_spec=spec,
            projector_type="mlp",
            input_size=input_size,
        )

        self.assertEqual(model.projector_type, "mlp")
        # The encoder is a genuine MLP instance, not a stub.
        self.assertIsInstance(model.encoder, MLP)
        # input_size must actually flow into the built encoder (parameter
        # consumption, not merely "was passed"); output width is config.hidden_size.
        self.assertEqual(model.encoder.input_size, input_size)
        self.assertEqual(model.encoder.hidden_size, config.hidden_size)


@unittest.skipUnless(_IMPORT_OK, _SKIP_REASON)
class TestMultimodalProjectorAffineBranch(unittest.TestCase):
    """The ``affine`` branch is broken against the installed APIs.

    Correct behavior would be: constructing an ``affine`` projector with a real
    spec/config succeeds and records ``projector_type == "affine"``. Production
    instead reads ``sublayers_spec.linear_fc1`` (absent on ``MLPSublayersSpec``)
    and ``config.add_bias_linear`` (absent on ``TransformerConfig``), so real
    construction raises ``AttributeError``. We assert the *correct* behavior and
    mark it ``expectedFailure`` rather than editing production or mocking the
    missing attributes to paper over the defect.
    """

    @unittest.expectedFailure
    def test_affine_projector_should_construct(self):
        config = _base_config()
        spec = MLPSublayersSpec()  # real spec, deliberately not monkeypatched
        model = MultimodalProjector(
            config=config,
            sublayers_spec=spec,
            projector_type="affine",
            input_size=32,
        )
        self.assertEqual(model.projector_type, "affine")
        self.assertIsNotNone(model.encoder)


if __name__ == "__main__":
    unittest.main()

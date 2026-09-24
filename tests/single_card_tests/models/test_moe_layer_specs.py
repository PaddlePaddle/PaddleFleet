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

"""Behavior tests for paddlefleet.models.gpt.moe_layer_specs.

``get_moe_layer_spec_for_backend`` is a *spec assembler*: from a backend
"spec provider" it selects three sublayer classes and wires them into a paddle
``LayerSpec`` whose ``layer`` is ``MoELayer`` and whose ``extra_kwargs`` carry a
``MoESublayers`` wrapping an ``MLPSublayersSpec``. The contract worth verifying
is therefore the *routing*: which backend selector feeds which
``MLPSublayersSpec`` slot. The realistic failure mode for such an assembler is a
plausible-but-wrong swap (e.g. feeding ``row_parallel_linear`` into
``up_gate_proj``), which a bare ``isinstance(result, LayerSpec)`` /
``assert_called_once`` check cannot detect. The backend is a legitimate, non
under-test collaborator, so it is replaced by a recording fake that returns
*distinct* sentinels per selector; the real assembly (``MLPSublayersSpec`` /
``MoESublayers`` / ``LayerSpec`` construction) is exercised for real and the
sentinels are traced into their destination slots.

The production module imports paddle (``LayerSpec``) and, transitively,
``MoELayer`` (which imports ``paddlefleet_ops``); when paddle / paddlefleet is
not importable the whole suite is skipped with an honest reason rather than
faked. No numeric or distributed behavior is claimed here -- the code path is
pure-Python spec assembly.
"""

import dataclasses
import unittest

try:
    from paddle.distributed.fleet.meta_parallel import LayerSpec

    from paddlefleet.models.gpt.moe_layer_specs import (
        get_moe_layer_spec_for_backend,
    )
    from paddlefleet.models.kimi_k25.sd2_tpool_merge import (
        KimiK25VisionPatchMergerSpec,
    )
    from paddlefleet.transformer.identity_op import IdentityOp
    from paddlefleet.transformer.mlp import MLPSublayersSpec
    from paddlefleet.transformer.moe.moe_layer import MoELayer, MoESublayers

    _IMPORT_ERROR = None
except ImportError as exc:  # only a genuine missing-dependency signal
    _IMPORT_ERROR = exc


class _Marker:
    """A uniquely identifiable stand-in for a selected sublayer class.

    Distinct instances make a slot swap observable via ``assertIs``; the value
    is only ever stored in a spec, never called, so a plain object suffices.
    """

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"_Marker({self.name!r})"


class _RecordingBackend:
    """Fake ``BackendSpecProvider``: distinct sentinels + call recording.

    Each selector returns a fixed, distinct sentinel (so repeated calls yield
    the *same* object, letting a test compare routing across invocations) and
    records how it was invoked, so the assembler's consumption can be observed.
    """

    def __init__(self):
        self.calls = []
        self.col = _Marker("column_parallel_linear")
        self.row = _Marker("row_parallel_linear")
        self.act = _Marker("hidden_act")

    def column_parallel_linear(self, *args, **kwargs):
        self.calls.append(("column_parallel_linear", args, kwargs))
        return self.col

    def row_parallel_linear(self, *args, **kwargs):
        self.calls.append(("row_parallel_linear", args, kwargs))
        return self.row

    def hidden_act(self, *args, **kwargs):
        self.calls.append(("hidden_act", args, kwargs))
        return self.act


@unittest.skipUnless(
    _IMPORT_ERROR is None,
    f"paddlefleet.models.gpt.moe_layer_specs not importable "
    f"(missing paddle / paddlefleet_ops?): {_IMPORT_ERROR}",
)
class TestGetMoeLayerSpecForBackend(unittest.TestCase):
    """The assembler must wire backend selectors into the MoE LayerSpec."""

    def test_assembles_layer_spec_targeting_moe_layer(self):
        backend = _RecordingBackend()
        result = get_moe_layer_spec_for_backend(backend=backend, num_experts=8)
        self.assertIsInstance(result, LayerSpec)
        # The spec must build a MoELayer, not some neighbouring layer class.
        self.assertIs(result.layer, MoELayer)

    def test_sublayers_route_each_selector_to_its_exact_slot(self):
        # Core anti-swap check: each backend selector's output must land in the
        # matching MLPSublayersSpec field. Distinct sentinels make a
        # column/row (or act) swap fail loudly.
        backend = _RecordingBackend()
        result = get_moe_layer_spec_for_backend(backend=backend, num_experts=8)

        self.assertIn("sublayers", result.extra_kwargs)
        sublayers = result.extra_kwargs["sublayers"]
        self.assertIsInstance(sublayers, MoESublayers)

        mlp_spec = sublayers.mlp_spec
        self.assertIsInstance(mlp_spec, MLPSublayersSpec)

        self.assertIs(mlp_spec.up_gate_proj, backend.col)
        self.assertIs(mlp_spec.down_proj, backend.row)
        self.assertIs(mlp_spec.hidden_act, backend.act)

    def test_each_selector_called_exactly_once_without_arguments(self):
        backend = _RecordingBackend()
        get_moe_layer_spec_for_backend(backend=backend, num_experts=8)

        names = sorted(name for name, _, _ in backend.calls)
        self.assertEqual(
            names,
            ["column_parallel_linear", "hidden_act", "row_parallel_linear"],
        )
        self.assertEqual(len(backend.calls), 3)
        for _, args, kwargs in backend.calls:
            self.assertEqual(args, ())
            self.assertEqual(kwargs, {})

    def test_none_num_experts_raises_before_any_wiring(self):
        # The ``assert num_experts is not None`` guard must fire, and it must
        # short-circuit before any backend selector is consumed.
        backend = _RecordingBackend()
        with self.assertRaises(AssertionError):
            get_moe_layer_spec_for_backend(backend=backend, num_experts=None)
        self.assertEqual(backend.calls, [])

    def test_num_experts_value_does_not_flow_into_spec(self):
        # ``num_experts`` is only a not-None guard here; its value is not
        # consumed into the returned spec (the real expert count is taken from
        # ``config.n_routed_experts`` when MoELayer is later built). Both values
        # must yield the same layer target and the same slot routing.
        backend4 = _RecordingBackend()
        result4 = get_moe_layer_spec_for_backend(
            backend=backend4, num_experts=4
        )
        backend8 = _RecordingBackend()
        result8 = get_moe_layer_spec_for_backend(
            backend=backend8, num_experts=8
        )

        self.assertIs(result4.layer, MoELayer)
        self.assertIs(result8.layer, MoELayer)
        for result, backend in ((result4, backend4), (result8, backend8)):
            mlp_spec = result.extra_kwargs["sublayers"].mlp_spec
            self.assertIs(mlp_spec.up_gate_proj, backend.col)
            self.assertIs(mlp_spec.down_proj, backend.row)
            self.assertIs(mlp_spec.hidden_act, backend.act)

    def test_moe_expert_fusion_flag_is_currently_inert(self):
        # ``moe_expert_fusion`` is accepted by the signature but never read in
        # the body (see report). Assert its *actual* behavior honestly: toggling
        # it produces no observable difference in the assembled spec. Reusing one
        # backend (fixed sentinels) lets both results be compared to the same
        # objects.
        backend = _RecordingBackend()
        result_false = get_moe_layer_spec_for_backend(
            backend=backend, num_experts=8, moe_expert_fusion=False
        )
        result_true = get_moe_layer_spec_for_backend(
            backend=backend, num_experts=8, moe_expert_fusion=True
        )
        for result in (result_false, result_true):
            self.assertIs(result.layer, MoELayer)
            mlp_spec = result.extra_kwargs["sublayers"].mlp_spec
            self.assertIs(mlp_spec.up_gate_proj, backend.col)
            self.assertIs(mlp_spec.down_proj, backend.row)
            self.assertIs(mlp_spec.hidden_act, backend.act)


@unittest.skipUnless(
    _IMPORT_ERROR is None,
    f"paddlefleet.models.kimi_k25.sd2_tpool_merge not importable "
    f"(missing paddle / paddlefleet_ops?): {_IMPORT_ERROR}",
)
class TestKimiK25VisionPatchMergerSpec(unittest.TestCase):
    """The merger spec must default its norm slot to IdentityOp and honor overrides."""

    def test_is_dataclass_with_identity_default_on_norm_field(self):
        self.assertTrue(dataclasses.is_dataclass(KimiK25VisionPatchMergerSpec))
        fields = {
            f.name: f for f in dataclasses.fields(KimiK25VisionPatchMergerSpec)
        }
        self.assertIn("norm", fields)
        # The declared default is the IdentityOp *class* (a no-op norm), not an
        # instance and not some other norm.
        self.assertIs(fields["norm"].default, IdentityOp)

    def test_default_norm_is_identity_op_class(self):
        spec = KimiK25VisionPatchMergerSpec()
        self.assertIs(spec.norm, IdentityOp)
        # Default is shared and stable across instances (it is a class object).
        other = KimiK25VisionPatchMergerSpec()
        self.assertIs(spec.norm, other.norm)

    def test_custom_norm_overrides_default(self):
        marker = _Marker("custom_norm")
        spec = KimiK25VisionPatchMergerSpec(norm=marker)
        self.assertIs(spec.norm, marker)
        # The override must actually take effect, not silently fall back.
        self.assertIsNot(spec.norm, IdentityOp)


if __name__ == "__main__":
    unittest.main()

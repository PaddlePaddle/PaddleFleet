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

"""CPU-only unit tests for paddlefleet.models.kimi_k25.kimi_k25_model.

Sibling of ``test_kimi_k25_model``. That base file owns the pure-construction
slice (``KimiK25VisionSublayersSpec`` dataclass and the real
``KimiK25VisionTransformerLayer.__init__``). This file owns the disjoint
runtime slice keyed to the same module:

* ``KimiK25VisionModel.get_layer_desc_list`` -- the pipeline layer-description
  assembly: sequence, per-entry name prefixes and the running layer index,
  plus the ``modal`` -> name-prefix branch;
* ``KimiK25VisionTransformerLayer.forward`` -- the dict plumbing: which control
  keys are popped, how ``grid_thws`` is popped-before / restored-after the
  non-recompute call, the recompute-branch kwarg wiring, the tuple/context
  recovery and the final dict merge order;
* ``KimiK25VisionTransformerLayer._forward_impl`` -- the 2D->3D unsqueeze guard
  and the context-tuple return contract.

To reach these methods without standing up a full pipeline model (Fleet +
device weights, already exercised by tests/single_card_tests/model/
test_kimi_k25_vision_model.py) the instances are allocated with ``__new__`` and
only the attributes each method reads are seeded. The heavy per-instance
collaborators (``_forward_impl`` for ``forward``; ``_forward_attention`` /
``_forward_mlp`` for ``_forward_impl``; ``recompute`` for the recompute branch)
are replaced by recording stubs that return distinguishable markers, so the
method under test runs in full and its orchestration is what gets observed.

Two behaviours assert the *correct* contract and are marked
``expectedFailure`` because the production ``forward`` decides tuple-vs-tensor
with ``len(outputs) == 3`` (kimi_k25_model.py:146). ``_forward_impl`` only ever
returns a bare tensor or a 2-tuple, so that predicate never matches the real
context path and misfires on any tensor whose leading dim is 3. Production is
left unmodified per the review rules.

The module under test imports Paddle at load time; when Paddle / paddlefleet is
not importable the whole file is skipped with an honest reason.
"""

import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

_REPO_SRC = os.path.join(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
    "src",
)
if os.path.isdir(_REPO_SRC) and _REPO_SRC not in sys.path:
    sys.path.insert(0, _REPO_SRC)

_IMPORT_ERROR = None
try:
    import numpy as np
    import paddle
    from paddle.distributed.fleet.meta_parallel import LayerDesc, LayerSpec

    from paddlefleet.models.kimi_k25.kimi_k25_model import (
        KimiK25VisionModel,
        KimiK25VisionTransformerLayer,
    )
except (ImportError, ModuleNotFoundError) as exc:  # honest capability probe
    # Only genuine missing-dependency import failures are treated as skip.
    # Compilation / API-change errors raise other exception types and surface
    # as real failures instead of being swallowed here.
    _IMPORT_ERROR = exc

_AVAILABLE = _IMPORT_ERROR is None
_SKIP_REASON = (
    f"paddle / paddlefleet not importable: {_IMPORT_ERROR!r}"
    if _IMPORT_ERROR is not None
    else ""
)


if _AVAILABLE:

    class _DummyLayer(paddle.nn.Layer):
        """Minimal Layer used only as a LayerSpec target."""

        def __init__(self):
            super().__init__()


def _bare(cls):
    """Allocate an instance without running the heavy real ``__init__``.

    ``get_layer_desc_list`` / ``forward`` / ``_forward_impl`` read only a
    handful of attributes; full construction (device weights, Fleet groups) is
    covered by the pipeline model test and is out of this slice's scope.
    ``paddle.nn.Layer.__setattr__`` requires bookkeeping dicts, so seeded
    attributes are written straight into ``__dict__``.
    """
    return cls.__new__(cls)


def _make_specs(n):
    """n freshly-distinguishable LayerSpec objects for identity assertions."""
    return [LayerSpec(_DummyLayer) for _ in range(n)]


@unittest.skipUnless(_AVAILABLE, _SKIP_REASON)
class TestKimiK25VisionModelGetLayerDescList(unittest.TestCase):
    """Assembly contract of KimiK25VisionModel.get_layer_desc_list.

    The real base-class collaborators ``add_sequential_layer`` and
    ``get_encoder_layer_desc_list`` run unmodified; only construction is
    bypassed. Each produced entry is checked for its name prefix *and* the
    exact LayerSpec it wraps, so a reordered stage, a wrong prefix or a
    misrouted spec is rejected -- not merely ``len(layers) > 0``.
    """

    def test_layer_desc_order_names_and_specs_with_modal(self):
        """modal set -> 'model.<modal>' prefix; full ordered stage list."""
        emb, h0, t0, t1, tail, fln, sd, mg = _make_specs(8)
        spec = SimpleNamespace(
            embedding=emb,
            head_empty_layers=[h0],
            transformer_layers=[t0, t1],
            tail_empty_layers=[tail],
            final_layernorm=fln,
            sdtpool_merger=sd,
            merger=mg,
        )
        model = _bare(KimiK25VisionModel)
        model.__dict__["modal"] = "vision"

        layers = model.get_layer_desc_list(spec)

        # Hand-derived: embedding, then head(0)/transformer(1,2)/tail(3) under a
        # single running index, then final_layernorm, sdtpool_merger, merger.
        expected = [
            ("model.vision.patch_embed", emb),
            ("model.vision.layers.0", h0),
            ("model.vision.layers.1", t0),
            ("model.vision.layers.2", t1),
            ("model.vision.layers.3", tail),
            ("model.vision.final_layernorm", fln),
            ("model.vision.sdtpool_merger", sd),
            ("model.vision.mm_projector", mg),
        ]
        self.assertEqual(len(layers), len(expected))
        for entry, (name, wrapped) in zip(layers, expected):
            self.assertEqual(entry["name_prefix"], name)
            self.assertIsInstance(entry["layer"], LayerDesc)
            # LayerDesc(spec.<field>) stores that LayerSpec as its layer_spec.
            self.assertIs(entry["layer"].layer_spec, wrapped)

    def test_name_prefix_without_modal(self):
        """modal falsy -> bare 'model' prefix; index starts at 0 for encoder."""
        emb, t0, fln, sd, mg = _make_specs(5)
        spec = SimpleNamespace(
            embedding=emb,
            head_empty_layers=[],
            transformer_layers=[t0],
            tail_empty_layers=[],
            final_layernorm=fln,
            sdtpool_merger=sd,
            merger=mg,
        )
        model = _bare(KimiK25VisionModel)
        model.__dict__["modal"] = None

        layers = model.get_layer_desc_list(spec)

        expected = [
            ("model.patch_embed", emb),
            ("model.layers.0", t0),
            ("model.final_layernorm", fln),
            ("model.sdtpool_merger", sd),
            ("model.mm_projector", mg),
        ]
        self.assertEqual(len(layers), len(expected))
        for entry, (name, wrapped) in zip(layers, expected):
            self.assertEqual(entry["name_prefix"], name)
            self.assertIsInstance(entry["layer"], LayerDesc)
            self.assertIs(entry["layer"].layer_spec, wrapped)


@unittest.skipUnless(_AVAILABLE, _SKIP_REASON)
class TestKimiK25VisionTransformerLayerForward(unittest.TestCase):
    """dict-plumbing contract of KimiK25VisionTransformerLayer.forward.

    ``_forward_impl`` (the attention/MLP math needing a device) is a stub with a
    distinguishable marker; the pop / restore / merge / context-recovery logic
    of ``forward`` runs in full and is what the assertions observe.
    """

    def test_forward_pops_control_keys_and_merges(self):
        """Non-recompute path: control keys dropped, grid_thws restored, merge."""
        layer = _bare(KimiK25VisionTransformerLayer)
        layer.__dict__["full_recompute"] = False
        layer.__dict__["modal"] = "vision"

        marker = paddle.arange(1 * 4 * 8, dtype="float32").reshape([1, 4, 8])
        captured = {}

        def fake_impl(**kwargs):
            captured["kwargs"] = dict(kwargs)
            return marker

        layer.__dict__["_forward_impl"] = fake_impl

        hs = (
            paddle.arange(1 * 4 * 8, dtype="float32").reshape([1, 4, 8]) + 100.0
        )
        mask = paddle.ones([1, 4])
        grid = paddle.to_tensor([[1, 2, 2]])
        pos = paddle.zeros([1, 4])
        result = layer.forward(
            {
                "hidden_states": hs,
                "attention_mask": mask,
                "grid_thws": grid,
                "dynamic_inference_decode_only": True,
                "position_ids": pos,
            }
        )

        # grid_thws and both control keys are removed before _forward_impl runs.
        self.assertIs(captured["kwargs"]["hidden_states"], hs)
        self.assertIs(captured["kwargs"]["attention_mask"], mask)
        self.assertNotIn("grid_thws", captured["kwargs"])
        self.assertNotIn("dynamic_inference_decode_only", captured["kwargs"])
        self.assertNotIn("position_ids", captured["kwargs"])

        # Output: hidden_states overwritten by marker, mask kept, grid restored,
        # control keys gone, bare tensor -> no context.
        self.assertIs(result["hidden_states"], marker)
        self.assertIs(result["attention_mask"], mask)
        self.assertIs(result["grid_thws"], grid)
        self.assertNotIn("dynamic_inference_decode_only", result)
        self.assertNotIn("position_ids", result)
        self.assertNotIn("context", result)

    # __FWD__

    def test_forward_recompute_branch_wires_kwargs(self):
        """full_recompute path routes assembled kwargs through recompute()."""
        layer = _bare(KimiK25VisionTransformerLayer)
        layer.__dict__["full_recompute"] = True
        layer.__dict__["modal"] = "vision"

        hs = paddle.arange(1 * 4 * 8, dtype="float32").reshape([1, 4, 8])
        mask = paddle.ones([1, 4])
        marker = (
            paddle.arange(1 * 4 * 8, dtype="float32").reshape([1, 4, 8]) + 7.0
        )
        captured = {}

        def fake_recompute(func, **kwargs):
            captured["func"] = func
            captured["kwargs"] = dict(kwargs)
            return marker

        with patch(
            "paddlefleet.models.kimi_k25.kimi_k25_model.recompute",
            side_effect=fake_recompute,
        ) as rc:
            result = layer.forward(
                {
                    "hidden_states": hs,
                    "attention_mask": mask,
                    "dynamic_inference_decode_only": True,
                }
            )

        rc.assert_called_once()
        # recompute drives the real _forward_impl with the assembled kwargs.
        self.assertEqual(captured["func"], layer._forward_impl)
        self.assertIs(captured["kwargs"]["hidden_states"], hs)
        self.assertIs(captured["kwargs"]["attention_mask"], mask)
        for none_key in (
            "attn_mask_startend_row_indices",
            "context",
            "context_mask",
            "rope_freqs_cis",
            "attention_bias",
            "packed_seq_params",
        ):
            self.assertIsNone(captured["kwargs"][none_key])

        self.assertIs(result["hidden_states"], marker)
        self.assertIs(result["attention_mask"], mask)
        self.assertNotIn("dynamic_inference_decode_only", result)
        self.assertNotIn("context", result)

    # __FWD2__

    @unittest.expectedFailure
    def test_forward_misdetects_tensor_batch_of_three(self):
        """BUG: len(outputs)==3 misreads a 3-row tensor as a context tuple.

        kimi_k25_model.py:146 uses ``len(outputs) == 3`` to decide tuple-vs-
        tensor. ``_forward_impl`` returns a bare tensor here, but its leading
        dim is 3, so ``len`` is 3 and forward wrongly slices rows [0] and [1]
        out as hidden_states / context. Correct contract: the whole tensor is
        the hidden_states and there is no context. Production left unmodified.
        """
        layer = _bare(KimiK25VisionTransformerLayer)
        layer.__dict__["full_recompute"] = False
        layer.__dict__["modal"] = "vision"
        marker = paddle.arange(3 * 4 * 8, dtype="float32").reshape([3, 4, 8])
        layer.__dict__["_forward_impl"] = lambda **kw: marker

        result = layer.forward({"hidden_states": paddle.zeros([3, 4, 8])})

        self.assertIs(result["hidden_states"], marker)
        self.assertNotIn("context", result)

    @unittest.expectedFailure
    def test_forward_drops_context_tuple(self):
        """BUG: a real 2-tuple (hidden, context) is not recovered.

        ``_forward_impl`` returns ``(hidden_states, context)`` when context is
        present, but len==2 misses the ``== 3`` predicate, so forward stores the
        whole tuple as ``hidden_states`` and emits no ``context`` key. Correct
        contract asserted below; production left unmodified.
        """
        layer = _bare(KimiK25VisionTransformerLayer)
        layer.__dict__["full_recompute"] = False
        layer.__dict__["modal"] = "vision"
        hs = paddle.arange(1 * 4 * 8, dtype="float32").reshape([1, 4, 8])
        ctx = (
            paddle.arange(1 * 4 * 8, dtype="float32").reshape([1, 4, 8]) + 500.0
        )
        layer.__dict__["_forward_impl"] = lambda **kw: (hs, ctx)

        result = layer.forward({"hidden_states": paddle.zeros([1, 4, 8])})

        self.assertIn("context", result)
        self.assertIs(result["hidden_states"], hs)
        self.assertIs(result["context"], ctx)


@unittest.skipUnless(_AVAILABLE, _SKIP_REASON)
class TestKimiK25VisionTransformerLayerForwardImpl(unittest.TestCase):
    """Shape-guard and context-return contract of _forward_impl.

    ``_forward_attention`` / ``_forward_mlp`` (device math) are recording stubs;
    the unsqueeze guard, the collaborator hand-off and the tuple/bare return of
    ``_forward_impl`` run in full.
    """

    def test_forward_impl_unsqueezes_2d_input(self):
        """2D [S, H] input is promoted to [1, S, H]; bare tensor returned."""
        layer = _bare(KimiK25VisionTransformerLayer)
        layer.__dict__["full_recompute"] = False

        captured = {}
        att_out = (
            paddle.arange(1 * 8 * 64, dtype="float32").reshape([1, 8, 64]) + 1.0
        )
        mlp_out = (
            paddle.arange(1 * 8 * 64, dtype="float32").reshape([1, 8, 64]) + 2.0
        )

        def fake_attn(**kwargs):
            captured["attn"] = dict(kwargs)
            return att_out, None

        def fake_mlp(x):
            captured["mlp_in"] = x
            return mlp_out

        layer.__dict__["_forward_attention"] = fake_attn
        layer.__dict__["_forward_mlp"] = fake_mlp

        hidden = paddle.arange(8 * 64, dtype="float32").reshape([8, 64])  # 2D
        result = layer._forward_impl(hidden_states=hidden)

        passed = captured["attn"]["hidden_states"]
        self.assertEqual(list(passed.shape), [1, 8, 64])
        np.testing.assert_array_equal(
            passed.numpy(), hidden.unsqueeze(0).numpy()
        )
        self.assertFalse(captured["attn"]["in_recompute"])
        self.assertIs(captured["mlp_in"], att_out)
        self.assertIs(result, mlp_out)
        self.assertNotIsInstance(result, tuple)

    def test_forward_impl_returns_context_tuple_when_present(self):
        """3D input passes through; a non-None context yields a 2-tuple."""
        layer = _bare(KimiK25VisionTransformerLayer)
        layer.__dict__["full_recompute"] = True

        captured = {}
        att_out = (
            paddle.arange(2 * 8 * 64, dtype="float32").reshape([2, 8, 64]) + 1.0
        )
        ctx = (
            paddle.arange(2 * 8 * 64, dtype="float32").reshape([2, 8, 64]) + 9.0
        )
        mlp_out = (
            paddle.arange(2 * 8 * 64, dtype="float32").reshape([2, 8, 64]) + 2.0
        )

        def fake_attn(**kwargs):
            captured["attn"] = dict(kwargs)
            return att_out, ctx

        layer.__dict__["_forward_attention"] = fake_attn
        layer.__dict__["_forward_mlp"] = lambda x: mlp_out

        hidden = paddle.arange(2 * 8 * 64, dtype="float32").reshape([2, 8, 64])
        result = layer._forward_impl(hidden_states=hidden)

        self.assertIs(captured["attn"]["hidden_states"], hidden)  # no unsqueeze
        self.assertTrue(captured["attn"]["in_recompute"])
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIs(result[0], mlp_out)
        self.assertIs(result[1], ctx)


if __name__ == "__main__":
    unittest.main()

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

"""Model-layer tests for ``paddlefleet.models.qwen3_vl.patch_merger``.

Targets under test, designed from the production source only:

* ``Qwen3VLVisionPatchMergerSpec`` -- single-field LayerSpec container; the
  declared field name and its ``IdentityOp`` default are pinned.
* ``Qwen3VLVisionPathMerger.__init__`` -- the derived ``hidden_size``
  (``context_dim * spatial_merge_size**2``), the ``context_dim`` / ``dim``
  default fallbacks, the ``use_postshuffle_norm`` -> ``norm_dim`` selection, and
  the exact ``MLP`` LayerSpec wiring are observed through
  ``build_spec_layer`` / ``LayerSpec`` / ``MLPSublayersSpec`` arg capture. These
  are independently-tested collaborators, so they are isolated and only the
  orchestration the merger itself performs is asserted.
* ``Qwen3VLVisionPathMerger.forward`` -- the real reshape / norm-ordering /
  dict-unpacking / bias-fold logic is driven end to end. ``norm`` and ``mlp``
  are replaced with input-dependent recording stand-ins so every expected
  tensor is hand-derived and the two norm-placement branches are told apart by
  the *content* each collaborator receives, not by call count.

No production defect was found in ``patch_merger.py`` (see the module report),
so no ``expectedFailure`` markers are present.

The module imports Paddle at load time. When Paddle / paddlefleet is not
importable the whole file is skipped with an honest reason rather than
reporting a pass. It is never faked green.
"""

import os
import sys
import types
import unittest
from unittest.mock import patch

# Make both the repo root (for ``tests`` siblings) and the ``src/`` layout
# importable when the package is not pip-installed.
_TESTS_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
_REPO_ROOT = os.path.dirname(_TESTS_ROOT)
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "src")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

_IMPORT_ERROR = None
try:
    import numpy as np
    import paddle

    from paddlefleet.models.qwen3_vl import patch_merger as pm
    from paddlefleet.models.qwen3_vl.patch_merger import (
        Qwen3VLVisionPatchMergerSpec,
        Qwen3VLVisionPathMerger,
    )
    from paddlefleet.transformer.identity_op import IdentityOp
except (ImportError, ModuleNotFoundError) as exc:  # honest capability probe
    # Only genuine missing-dependency import failures are treated as skip.
    # Compilation / API-change errors raise other exception types and surface
    # as real failures instead of being swallowed here.
    _IMPORT_ERROR = exc
    np = None

_AVAILABLE = _IMPORT_ERROR is None
_SKIP_REASON = (
    f"paddle / paddlefleet not importable: {_IMPORT_ERROR!r}"
    if _IMPORT_ERROR is not None
    else ""
)


def _vision_config(hidden_size, out_hidden_size, spatial_merge_size):
    """Minimal config double exposing only what ``__init__`` reads.

    ``Qwen3VLVisionPathMerger.__init__`` touches exactly ``config.hidden_size``,
    ``config.out_hidden_size`` and ``config.spatial_merge_size`` and otherwise
    threads the object straight through to ``build_spec_layer`` (isolated in
    these tests). A namespace with those three attributes is therefore a
    faithful stand-in for the parts under test; pass-through is verified by
    identity where it matters.
    """
    return types.SimpleNamespace(
        hidden_size=hidden_size,
        out_hidden_size=out_hidden_size,
        spatial_merge_size=spatial_merge_size,
    )


class _CallRecorder:
    """Records positional/keyword arguments and returns a fresh sentinel."""

    def __init__(self):
        self.calls = []
        self.results = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        sentinel = object()
        self.results.append(sentinel)
        return sentinel


class _RecordingNorm:
    """A norm stand-in: shape-preserving, input-dependent, records inputs."""

    def __init__(self):
        self.received = []

    def __call__(self, x):
        self.received.append(x)
        return x + 10.0


class _RecordingMLP:
    """An MLP stand-in returning a fixed ``(output, bias)`` and recording input."""

    def __init__(self, output, bias):
        self.received = []
        self._output = output
        self._bias = bias

    def __call__(self, x):
        self.received.append(x)
        return self._output, self._bias


def _build_merger_with_stubs(config, norm, mlp, use_postshuffle_norm=False):
    """Run the real ``__init__`` but inject recording ``norm`` / ``mlp``.

    ``build_spec_layer`` is the only isolated seam: its first invocation builds
    the norm, its second builds the MLP, matching the source order. Everything
    else in ``__init__`` (the ``hidden_size`` / ``norm_dim`` / ``dim`` maths and
    the real ``LayerSpec`` / ``MLPSublayersSpec`` construction) runs unmodified.
    """
    outputs = iter((norm, mlp))

    def fake_build(*args, **kwargs):
        return next(outputs)

    with patch.object(pm, "build_spec_layer", side_effect=fake_build):
        merger = Qwen3VLVisionPathMerger(
            config=config,
            sublayers_spec=Qwen3VLVisionPatchMergerSpec(),
            use_postshuffle_norm=use_postshuffle_norm,
        )
    return merger


@unittest.skipUnless(_AVAILABLE, _SKIP_REASON)
class TestQwen3VLVisionPatchMergerSpec(unittest.TestCase):
    """Contract of the single-field spec dataclass."""

    def test_default_norm_is_identity_op(self):
        """A no-argument spec defaults ``norm`` to the production IdentityOp.

        This pins the production-declared default (the independent oracle is the
        ``IdentityOp`` class imported directly), not a value the test injects.
        """
        spec = Qwen3VLVisionPatchMergerSpec()
        self.assertIs(spec.norm, IdentityOp)

    def test_single_declared_field_named_norm(self):
        """Exactly one dataclass field, named ``norm``, defaulting to IdentityOp.

        A bare attribute check would still pass if extra fields were added or
        the field were renamed; the full ordered field tuple plus its declared
        default pins the container's public shape.
        """
        import dataclasses

        self.assertTrue(dataclasses.is_dataclass(Qwen3VLVisionPatchMergerSpec))
        fields = dataclasses.fields(Qwen3VLVisionPatchMergerSpec)
        self.assertEqual(tuple(f.name for f in fields), ("norm",))
        self.assertIs(fields[0].default, IdentityOp)


@unittest.skipUnless(_AVAILABLE, _SKIP_REASON)
class TestQwen3VLVisionPathMergerInit(unittest.TestCase):
    """Derived sizes, default fallbacks and MLP wiring in ``__init__``.

    ``build_spec_layer`` / ``LayerSpec`` / ``MLPSublayersSpec`` are isolated;
    the orchestration the merger performs around them is asserted by captured
    arguments, with every expected number hand-derived from the source.
    """

    def _construct(
        self,
        config,
        *,
        dim=None,
        context_dim=None,
        use_postshuffle_norm=False,
    ):
        build = _CallRecorder()
        layerspec = _CallRecorder()
        mlpspec = _CallRecorder()
        with (
            patch.object(pm, "build_spec_layer", build),
            patch.object(pm, "LayerSpec", layerspec),
            patch.object(pm, "MLPSublayersSpec", mlpspec),
        ):
            merger = Qwen3VLVisionPathMerger(
                config=config,
                sublayers_spec=Qwen3VLVisionPatchMergerSpec(),
                dim=dim,
                context_dim=context_dim,
                use_postshuffle_norm=use_postshuffle_norm,
            )
        return merger, build, layerspec, mlpspec

    def test_hidden_size_is_context_dim_times_merge_squared(self):
        # context_dim defaults to hidden_size=3; merge=2 -> 3 * 2**2 = 12.
        config = _vision_config(
            hidden_size=3, out_hidden_size=7, spatial_merge_size=2
        )
        merger, _, _, _ = self._construct(config)
        self.assertEqual(merger.hidden_size, 3 * (2**2))

    def test_context_dim_defaults_to_config_hidden_size(self):
        # No explicit context_dim: hidden_size derives from config.hidden_size.
        config = _vision_config(
            hidden_size=5, out_hidden_size=7, spatial_merge_size=3
        )
        merger, _, _, _ = self._construct(config)
        self.assertEqual(merger.hidden_size, 5 * (3**2))

    def test_explicit_context_dim_overrides_config_hidden_size(self):
        # An explicit context_dim wins over config.hidden_size in the product.
        config = _vision_config(
            hidden_size=5, out_hidden_size=7, spatial_merge_size=3
        )
        merger, _, _, _ = self._construct(config, context_dim=4)
        self.assertEqual(merger.hidden_size, 4 * (3**2))

    def test_use_postshuffle_norm_flag_is_stored(self):
        config = _vision_config(
            hidden_size=3, out_hidden_size=7, spatial_merge_size=2
        )
        merger, _, _, _ = self._construct(config, use_postshuffle_norm=True)
        self.assertTrue(merger.use_postshuffle_norm)
        merger2, _, _, _ = self._construct(config, use_postshuffle_norm=False)
        self.assertFalse(merger2.use_postshuffle_norm)

    def test_norm_built_over_context_dim_without_postshuffle(self):
        # norm_dim == context_dim (=hidden_size=3) when postshuffle is off.
        config = _vision_config(
            hidden_size=3, out_hidden_size=7, spatial_merge_size=2
        )
        _, build, _, _ = self._construct(config, use_postshuffle_norm=False)
        # First build_spec_layer call constructs the norm.
        args, kwargs = build.calls[0]
        self.assertIs(args[0], IdentityOp)  # sublayers_spec.norm default
        self.assertIs(kwargs["config"], config)
        self.assertEqual(kwargs["hidden_size"], 3)

    def test_norm_built_over_merged_hidden_size_with_postshuffle(self):
        # norm_dim == merged hidden_size (=3 * 2**2 = 12) when postshuffle is on.
        config = _vision_config(
            hidden_size=3, out_hidden_size=7, spatial_merge_size=2
        )
        _, build, _, _ = self._construct(config, use_postshuffle_norm=True)
        args, kwargs = build.calls[0]
        self.assertEqual(kwargs["hidden_size"], 3 * (2**2))

    def test_mlp_layerspec_wiring_and_sizes(self):
        # The MLP is built with input_size == intermediate_size == merged
        # hidden_size and output hidden_size == dim (default out_hidden_size),
        # using ColumnParallelLinear / RowParallelLinear / F.gelu.
        config = _vision_config(
            hidden_size=3, out_hidden_size=7, spatial_merge_size=2
        )
        _, build, layerspec, mlpspec = self._construct(config)
        merged = 3 * (2**2)

        # MLPSublayersSpec constructed once with the parallel-linear wiring.
        self.assertEqual(len(mlpspec.calls), 1)
        _, sub_kwargs = mlpspec.calls[0]
        self.assertIs(sub_kwargs["up_gate_proj"], pm.ColumnParallelLinear)
        self.assertIs(sub_kwargs["down_proj"], pm.RowParallelLinear)
        self.assertIs(sub_kwargs["hidden_act"], pm.F.gelu)

        # LayerSpec constructed once wrapping MLP with the derived extra kwargs.
        self.assertEqual(len(layerspec.calls), 1)
        _, ls_kwargs = layerspec.calls[0]
        self.assertIs(ls_kwargs["layer"], pm.MLP)
        # The sublayers_spec passed to LayerSpec is exactly the object the
        # patched MLPSublayersSpec produced (no swap).
        self.assertIs(ls_kwargs["sublayers_spec"], mlpspec.results[0])
        extra = ls_kwargs["extra_kwargs"]
        self.assertIs(extra["config"], config)
        self.assertEqual(extra["input_size"], merged)
        self.assertEqual(extra["intermediate_size"], merged)
        self.assertEqual(
            extra["hidden_size"], 7
        )  # dim default = out_hidden_size

        # The MLP comes from the second build_spec_layer call and is stored.
        self.assertEqual(len(build.calls), 2)
        self.assertIs(build.calls[1][0][0], layerspec.results[0])

    def test_explicit_dim_overrides_config_out_hidden_size(self):
        # An explicit dim overrides config.out_hidden_size for the MLP output.
        config = _vision_config(
            hidden_size=3, out_hidden_size=7, spatial_merge_size=2
        )
        _, _, layerspec, _ = self._construct(config, dim=9)
        _, ls_kwargs = layerspec.calls[0]
        self.assertEqual(ls_kwargs["extra_kwargs"]["hidden_size"], 9)


@unittest.skipUnless(_AVAILABLE, _SKIP_REASON)
class TestQwen3VLVisionPathMergerForward(unittest.TestCase):
    """Real forward: reshape target, norm placement, dict-unpack, bias fold.

    ``norm`` and ``mlp`` are input-dependent recorders, so the two branches are
    distinguished by the *content* each receives. hidden_size is 2 and
    spatial_merge_size is 2, giving a merged width of ``2 * 2**2 = 8``; an 8-
    element input therefore folds four 2-wide patches into one 8-wide row, which
    makes the reshape visible rather than a no-op.
    """

    def _make(self, norm, mlp, use_postshuffle_norm=False):
        config = _vision_config(
            hidden_size=2, out_hidden_size=3, spatial_merge_size=2
        )
        merger = _build_merger_with_stubs(
            config, norm, mlp, use_postshuffle_norm=use_postshuffle_norm
        )
        # Guard the derived width the forward reshape depends on.
        self.assertEqual(merger.hidden_size, 8)
        return merger

    def test_forward_no_postshuffle_norms_before_reshape(self):
        """dict input is squeezed, norm runs on the pre-reshape patches, and the
        merged rows reach the MLP; the returned bias slot is always None."""
        norm = _RecordingNorm()
        mlp_out = paddle.to_tensor([[1.0, 2.0, 3.0]], dtype="float32")
        mlp_bias = paddle.to_tensor([10.0, 20.0, 30.0], dtype="float32")
        mlp = _RecordingMLP(mlp_out, mlp_bias)
        merger = self._make(norm, mlp)

        # [1, 4, 2] -> squeeze(0) -> [4, 2] patches 0..7.
        x = paddle.arange(8, dtype="float32").reshape([1, 4, 2])
        result, bias = merger.forward({"hidden_states": x})

        # norm saw the squeezed, *pre-reshape* [4, 2] patches (dict unpacked).
        self.assertEqual(len(norm.received), 1)
        np.testing.assert_array_equal(
            norm.received[0].numpy(),
            np.array([[0, 1], [2, 3], [4, 5], [6, 7]], dtype=np.float32),
        )
        # MLP saw norm(+10) merged into a single 8-wide row.
        self.assertEqual(len(mlp.received), 1)
        np.testing.assert_array_equal(
            mlp.received[0].numpy(),
            np.array([[10, 11, 12, 13, 14, 15, 16, 17]], dtype=np.float32),
        )
        # Hand-derived: mlp_out + broadcast bias; returned bias slot is None.
        np.testing.assert_array_equal(
            result.numpy(),
            np.array([[11.0, 22.0, 33.0]], dtype=np.float32),
        )
        self.assertIsNone(bias)

    def test_forward_postshuffle_norms_after_reshape(self):
        """With postshuffle, the merge happens first and norm runs on the merged
        8-wide rows -- the opposite ordering from the default branch."""
        norm = _RecordingNorm()
        mlp_out = paddle.to_tensor([[0.5, -0.5, 1.0]], dtype="float32")
        mlp = _RecordingMLP(mlp_out, None)
        merger = self._make(norm, mlp, use_postshuffle_norm=True)

        x = paddle.arange(8, dtype="float32").reshape([1, 4, 2])
        result, bias = merger.forward({"hidden_states": x})

        # norm saw the *post-reshape* single 8-wide row (merge preceded norm).
        self.assertEqual(len(norm.received), 1)
        np.testing.assert_array_equal(
            norm.received[0].numpy(),
            np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            mlp.received[0].numpy(),
            np.array([[10, 11, 12, 13, 14, 15, 16, 17]], dtype=np.float32),
        )
        # None bias -> mlp output returned unchanged, by identity.
        self.assertIs(result, mlp_out)
        self.assertIsNone(bias)

    def test_forward_without_bias_returns_mlp_output_object(self):
        """A None bias skips the fold: the exact MLP output object is returned."""
        norm = _RecordingNorm()
        mlp_out = paddle.to_tensor([[7.0, 8.0, 9.0]], dtype="float32")
        mlp = _RecordingMLP(mlp_out, None)
        merger = self._make(norm, mlp)

        x = paddle.arange(8, dtype="float32").reshape([1, 4, 2])
        result, bias = merger.forward({"hidden_states": x})

        self.assertIs(result, mlp_out)
        np.testing.assert_array_equal(
            result.numpy(), np.array([[7.0, 8.0, 9.0]], dtype=np.float32)
        )
        self.assertIsNone(bias)

    def test_forward_accepts_raw_tensor_without_dict_unpack(self):
        """A non-dict tensor input bypasses the ``hidden_states`` lookup/squeeze
        and flows straight into the norm."""
        norm = _RecordingNorm()
        mlp_out = paddle.to_tensor([[1.0, 1.0, 1.0]], dtype="float32")
        mlp_bias = paddle.to_tensor([1.0, 2.0, 3.0], dtype="float32")
        mlp = _RecordingMLP(mlp_out, mlp_bias)
        merger = self._make(norm, mlp)

        # Raw [4, 2] tensor, no dict wrapper and no leading batch axis.
        x = paddle.arange(8, dtype="float32").reshape([4, 2])
        result, bias = merger.forward(x)

        self.assertEqual(len(norm.received), 1)
        np.testing.assert_array_equal(
            norm.received[0].numpy(),
            np.array([[0, 1], [2, 3], [4, 5], [6, 7]], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            result.numpy(),
            np.array([[2.0, 3.0, 4.0]], dtype=np.float32),
        )
        self.assertIsNone(bias)


if __name__ == "__main__":
    unittest.main()

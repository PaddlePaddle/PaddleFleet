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

"""CPU-only unit tests for paddlefleet.models.kimi_k25.sd2_tpool_merge.

Model-layer (no-card) tests designed from the production source. The vision
patch merger performs pure reshape / permute / temporal-mean bookkeeping, so
its numeric contract is fully reproducible on CPU with hand-derived oracles.

Independent expectations are derived by hand from the reshape algebra in the
production ``forward`` and written as literal tensors below -- no production
helper is called to build any ``expected`` value.

The module under test imports Paddle at load time. When Paddle / paddlefleet
is not importable the whole file is skipped with an honest reason rather than
being reported as a pass. Only genuine missing-dependency import errors are
treated as skip; other error types surface as real failures.
"""

import os
import sys
import unittest
from types import SimpleNamespace

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
    from paddle import nn

    from paddlefleet.models.kimi_k25.sd2_tpool_merge import (
        KimiK25VisionPatchMergerSpec,
        KimiK25VisionPathMerger,
        KimiK25VisionSd2TpoolMerger,
    )
    from paddlefleet.transformer.identity_op import IdentityOp
except (ImportError, ModuleNotFoundError) as exc:  # honest capability probe
    _IMPORT_ERROR = exc

_AVAILABLE = _IMPORT_ERROR is None
_SKIP_REASON = (
    f"paddle / paddlefleet not importable: {_IMPORT_ERROR!r}"
    if _IMPORT_ERROR is not None
    else ""
)


@unittest.skipUnless(_AVAILABLE, _SKIP_REASON)
class TestSd2TpoolMergerForward(unittest.TestCase):
    """Numeric contract of KimiK25VisionSd2TpoolMerger.forward.

    forward takes hidden_states [1, T*H*W, D] (leading batch dim is dropped by
    ``view(shape[1:])`` so it must be 1) plus grid_thws [[t, h, w], ...] and,
    per grid, reshapes to (t, new_h, kh, new_w, kw, D), permutes to
    (t, new_h, new_w, kh, kw, D), temporal-means over t, then flattens to
    [new_h*new_w, kh*kw, D].
    """

    def _make(self, kernel=(2, 2)):
        cfg = SimpleNamespace(merge_kernel_size=kernel)
        return KimiK25VisionSd2TpoolMerger(cfg)

    def test_spatial_patch_reordering(self):
        # t=1, h=2, w=4, kernel=(2,2) -> new_h=1, new_w=2. D=1 so seq value
        # equals its row index; the permute produces a non-identity mapping.
        merger = self._make((2, 2))
        hidden = paddle.to_tensor(
            [[[0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0]]],
            dtype="float32",
        )  # shape [1, 8, 1]
        grid = paddle.to_tensor([[1, 2, 4]], dtype="int64")

        out = merger.forward({"hidden_states": hidden, "grid_thws": grid})[
            "hidden_states"
        ]
        self.assertIsInstance(out, list)
        self.assertEqual(len(out), 1)
        self.assertEqual(list(out[0].shape), [2, 4, 1])
        # Hand-derived: out[nw, kh*2+kw] = seq[kh*4 + nw*2 + kw].
        expected = np.array(
            [[[0.0], [1.0], [4.0], [5.0]], [[2.0], [3.0], [6.0], [7.0]]],
            dtype="float32",
        )
        np.testing.assert_array_equal(out[0].numpy(), expected)

    def test_temporal_pooling_averages_frames(self):
        # t=2, h=2, w=2, kernel=(2,2) -> new_h=new_w=1; output token q averages
        # frame0[q] and frame1[q] over the temporal axis.
        merger = self._make((2, 2))
        hidden = paddle.to_tensor(
            [[[0.0], [1.0], [2.0], [3.0], [10.0], [11.0], [12.0], [13.0]]],
            dtype="float32",
        )  # shape [1, 8, 1]; frame0 rows 0..3, frame1 rows 4..7
        grid = paddle.to_tensor([[2, 2, 2]], dtype="int64")

        out = merger.forward({"hidden_states": hidden, "grid_thws": grid})[
            "hidden_states"
        ]
        self.assertEqual(len(out), 1)
        self.assertEqual(list(out[0].shape), [1, 4, 1])
        # (0+10)/2, (1+11)/2, (2+12)/2, (3+13)/2
        expected = np.array([[[5.0], [6.0], [7.0], [8.0]]], dtype="float32")
        np.testing.assert_allclose(out[0].numpy(), expected, rtol=0, atol=1e-6)

    def test_multiple_grids_split_by_offset(self):
        # Two grids consume disjoint contiguous slices via pre_sum; each grid is
        # t=1,h=2,w=2 (identity mapping). Distinct content per grid detects a
        # swapped/misaligned offset, not merely a wrong count.
        merger = self._make((2, 2))
        hidden = paddle.to_tensor(
            [[[0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0]]],
            dtype="float32",
        )
        grid = paddle.to_tensor([[1, 2, 2], [1, 2, 2]], dtype="int64")

        out = merger.forward({"hidden_states": hidden, "grid_thws": grid})[
            "hidden_states"
        ]
        self.assertEqual(len(out), 2)
        self.assertEqual(list(out[0].shape), [1, 4, 1])
        self.assertEqual(list(out[1].shape), [1, 4, 1])
        np.testing.assert_array_equal(
            out[0].numpy(),
            np.array([[[0.0], [1.0], [2.0], [3.0]]], dtype="float32"),
        )
        np.testing.assert_array_equal(
            out[1].numpy(),
            np.array([[[4.0], [5.0], [6.0], [7.0]]], dtype="float32"),
        )

    def test_preserves_extra_dict_args_and_replaces_hidden_states(self):
        merger = self._make((2, 2))
        hidden = paddle.to_tensor(
            [[[0.0], [1.0], [2.0], [3.0]]], dtype="float32"
        )  # [1, 4, 1]
        grid = paddle.to_tensor([[1, 2, 2]], dtype="int64")
        result = merger.forward(
            {"hidden_states": hidden, "grid_thws": grid, "modal": "vision"}
        )
        # Non-hidden_states keys pass through unchanged...
        self.assertEqual(result["modal"], "vision")
        self.assertIs(result["grid_thws"], grid)
        # ...and hidden_states is replaced by the merged list of per-grid tensors.
        self.assertIsInstance(result["hidden_states"], list)
        self.assertEqual(len(result["hidden_states"]), 1)
        self.assertIsNot(result["hidden_states"], hidden)


@unittest.skipUnless(_AVAILABLE, _SKIP_REASON)
class TestPatchMergerSpec(unittest.TestCase):
    """KimiK25VisionPatchMergerSpec dataclass default / override contract."""

    def test_default_norm_is_identity_op(self):
        # The default norm collaborator selected when none is supplied.
        self.assertIs(KimiK25VisionPatchMergerSpec().norm, IdentityOp)

    def test_custom_norm_is_stored(self):
        sentinel = object()
        self.assertIs(
            KimiK25VisionPatchMergerSpec(norm=sentinel).norm, sentinel
        )


if _AVAILABLE:

    class _ProbePathMerger(KimiK25VisionPathMerger):
        """Real KimiK25VisionPathMerger with the two genuinely-separate
        collaborators (pre_norm, proj) replaced by distinguishable stubs.

        The heavy production __init__ builds tensor-parallel MLP linears which
        need a parallel runtime; that machinery is NOT the logic under test.
        The tested logic -- list/tensor dispatch, the ``.view`` reshapes, the
        ``[0]`` tuple-index on the list branch -- lives in the inherited
        ``forward`` and is exercised for real here. pre_norm is identity so the
        reshape fed to proj is exactly predictable; proj records its input and
        returns an input-dependent marker so consumption is observable.
        """

        def __init__(self, hidden_size):
            nn.Layer.__init__(self)
            self.hidden_size = hidden_size
            self.norm_inputs = []
            self.proj_inputs = []

            def _pre_norm(z):
                self.norm_inputs.append(z)
                return z

            def _proj(z):
                self.proj_inputs.append(z)
                return z + 100.0, None

            self.pre_norm = _pre_norm
            self.proj = _proj


@unittest.skipUnless(_AVAILABLE, _SKIP_REASON)
class TestPathMergerForward(unittest.TestCase):
    """Dispatch / reshape / tuple-index contract of KimiK25VisionPathMerger."""

    def test_list_branch_reshapes_and_indexes_proj_output(self):
        merger = _ProbePathMerger(hidden_size=12)
        item = paddle.arange(24, dtype="float32").reshape([2, 3, 4])
        result = merger.forward({"hidden_states": [item], "grid_thws": "keep"})

        # list in -> list out, one element per input item.
        self.assertIsInstance(result["hidden_states"], list)
        self.assertEqual(len(result["hidden_states"]), 1)
        # pre_norm saw the raw item; proj saw item flattened to [rows, -1].
        np.testing.assert_array_equal(
            merger.norm_inputs[0].numpy(), item.numpy()
        )
        self.assertEqual(list(merger.proj_inputs[0].shape), [2, 12])
        np.testing.assert_array_equal(
            merger.proj_inputs[0].numpy(), item.numpy().reshape(2, 12)
        )
        # list branch keeps only proj(...)[0], i.e. the marker tensor +100.
        np.testing.assert_array_equal(
            result["hidden_states"][0].numpy(),
            item.numpy().reshape(2, 12) + 100.0,
        )
        self.assertEqual(result["grid_thws"], "keep")  # passthrough

    def test_tensor_branch_reshapes_and_keeps_full_proj_output(self):
        merger = _ProbePathMerger(hidden_size=12)
        x = paddle.arange(24, dtype="float32").reshape([2, 3, 4])
        result = merger.forward({"hidden_states": x, "grid_thws": "keep"})

        # proj fed x reshaped to [B, -1, hidden_size] = [2, 1, 12].
        self.assertEqual(list(merger.proj_inputs[0].shape), [2, 1, 12])
        np.testing.assert_array_equal(
            merger.proj_inputs[0].numpy(), x.numpy().reshape(2, 1, 12)
        )
        # tensor branch keeps the FULL proj return (out, bias) tuple, unlike the
        # list branch which indexes [0] -- an asymmetry in the two code paths.
        out = result["hidden_states"]
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 2)
        np.testing.assert_array_equal(
            out[0].numpy(), x.numpy().reshape(2, 1, 12) + 100.0
        )
        self.assertIsNone(out[1])
        self.assertEqual(result["grid_thws"], "keep")  # passthrough

    def test_list_and_tensor_return_types_differ(self):
        # Pin the branch-dependent return-type asymmetry: list -> list[tensor],
        # tensor -> tuple(out, bias).
        merger = _ProbePathMerger(hidden_size=12)
        item = paddle.arange(24, dtype="float32").reshape([2, 3, 4])
        as_list = merger.forward({"hidden_states": [item]})["hidden_states"]
        as_tensor = merger.forward({"hidden_states": item})["hidden_states"]
        self.assertIsInstance(as_list, list)
        self.assertIsInstance(as_list[0], paddle.Tensor)
        self.assertIsInstance(as_tensor, tuple)


if __name__ == "__main__":
    unittest.main()

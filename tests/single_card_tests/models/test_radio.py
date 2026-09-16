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

"""Behavior tests for ``paddlefleet.models.vision.radio.RADIOViTModel``.

Scope (kept deliberately disjoint from the CPE / interpolation / CUDA-forward
slice covered elsewhere): the positional-encoding *selection* and *addition*
logic that runs on CPU, plus ``set_input_tensor`` delegation, the einops
capability flag, and the ``__init__`` tensor-allocation sites.

Method-level tests build a bare instance via ``__new__`` and populate only the
attributes the method under test reads, then assert against expected arrays
derived by hand in NumPy (never by re-calling the production path). ``__init__``
is exercised separately because it allocates parameters at import-of-model time.

The module imports paddle at load time; the authoring environment has no paddle
installed, so every test is gated behind an honest ``skipUnless(IMPORT_OK, ...)``
and skips rather than fakes a pass. Tests that assert the *correct* behaviour of
a path containing a confirmed PyTorch->Paddle porting bug are additionally marked
``expectedFailure`` and cite the offending source line; they must never be edited
into a pass by changing production code.
"""

import os
import sys
import unittest
from unittest import mock

# Make ``src/`` importable when tests are run from a source checkout.
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
_SRC = os.path.join(_REPO_ROOT, "src")
for _p in (_SRC, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

IMPORT_OK = True
IMPORT_ERR = ""
try:
    import paddle

    from paddlefleet.models.vision.radio import HAVE_EINOPS, RADIOViTModel
except ImportError as exc:  # genuine missing dependency, not an API/logic error
    IMPORT_OK = False
    IMPORT_ERR = f"paddle/paddlefleet.models.vision.radio not importable: {exc}"


def _bare_model():
    """Allocate a RADIOViTModel without running __init__.

    __init__ eagerly allocates parameters (and, as documented in the tests
    below, currently crashes on several Paddle tensor-factory calls), so the
    pos-encoding methods are exercised on a bare shell whose attributes are set
    to small, hand-checkable values.
    """
    return RADIOViTModel.__new__(RADIOViTModel)


def _known_pos_embeddings():
    """[1, 4, 3] positional table with unique, position-revealing values."""
    arr = np.arange(12, dtype="float32").reshape(1, 4, 3)
    return arr, paddle.to_tensor(arr)


@unittest.skipUnless(IMPORT_OK, IMPORT_ERR)
class TestHaveEinopsFlag(unittest.TestCase):
    """HAVE_EINOPS must faithfully reflect whether einops is importable."""

    def test_flag_matches_actual_import(self):
        self.assertIsInstance(HAVE_EINOPS, bool)
        try:
            import einops  # noqa: F401

            available = True
        except ImportError:
            available = False
        self.assertEqual(HAVE_EINOPS, available)


@unittest.skipUnless(IMPORT_OK, IMPORT_ERR)
class TestSetInputTensor(unittest.TestCase):
    """set_input_tensor forwards the exact tensor to the decoder."""

    def test_delegates_to_decoder(self):
        model = _bare_model()
        model.decoder = mock.MagicMock()
        sentinel = paddle.to_tensor(np.arange(6, dtype="float32").reshape(3, 2))
        model.set_input_tensor(sentinel)
        model.decoder.set_input_tensor.assert_called_once_with(sentinel)
        # The forwarded object must be the same tensor, not a copy/rebuild.
        (forwarded,), _ = model.decoder.set_input_tensor.call_args
        self.assertIs(forwarded, sentinel)


@unittest.skipUnless(IMPORT_OK, IMPORT_ERR)
class TestGetPosEncSelection(unittest.TestCase):
    """get_pos_enc selection logic on the identity (dims == max) branch."""

    def _configure(self, model):
        arr, tensor = _known_pos_embeddings()
        model.position_embeddings = tensor
        model.max_num_rows = 2
        model.max_num_cols = 2
        model.input_dims = (2, 2)
        model.patch_dim = 16
        model.has_cpe = True
        model.training = False
        return arr

    def test_returns_full_table_when_input_dims_match_max(self):
        model = _bare_model()
        arr = self._configure(model)
        result = model.get_pos_enc(batch_size=7, patch_idxs=None)
        # dims match max -> the stored table is returned unchanged (identity).
        self.assertIs(result, model.position_embeddings)
        np.testing.assert_array_equal(result.numpy(), arr)

    def test_input_size_is_divided_by_patch_dim(self):
        model = _bare_model()
        arr = self._configure(model)
        # Deliberately corrupt self.input_dims: the input_size argument must win
        # and be floor-divided by patch_dim -> (32//16, 32//16) == (2, 2) == max.
        model.input_dims = (99, 99)
        result = model.get_pos_enc(batch_size=1, input_size=(32, 32))
        np.testing.assert_array_equal(result.numpy(), arr)

    @unittest.expectedFailure
    def test_patch_idxs_gathers_selected_rows(self):
        # CONFIRMED BUG (radio.py:302 and radio.py:306): the patch_idxs branch
        # calls ``Tensor.expand(-1, -1, ...)`` (varargs) and
        # ``paddle.gather(..., dim=1, ...)``. Paddle's expand takes a single
        # shape list and gather uses ``axis=`` (not ``dim=``); both raise. The
        # assertion below encodes the correct torch-style semantics (select
        # rows 0 and 2 of the [1, 4, 3] table) and is expected to fail until the
        # production code is fixed. Production must not be edited here.
        model = _bare_model()
        arr = self._configure(model)
        patch_idxs = paddle.to_tensor([[0, 2]], dtype="int64")
        result = model.get_pos_enc(batch_size=1, patch_idxs=patch_idxs)
        expected = arr[:, [0, 2], :]
        np.testing.assert_array_equal(result.numpy(), expected)


@unittest.skipUnless(IMPORT_OK, IMPORT_ERR)
class TestApplyPosEnc(unittest.TestCase):
    """apply_pos_enc adds the positional table to the patches."""

    def _configure(self, model, *, training, pos_dropout):
        arr, tensor = _known_pos_embeddings()
        model.position_embeddings = tensor
        model.max_num_rows = 2
        model.max_num_cols = 2
        model.input_dims = (2, 2)
        model.patch_dim = 16
        model.has_cpe = True
        model.training = training
        model.pos_dropout = pos_dropout
        return arr

    def test_eval_mode_adds_pos_enc(self):
        model = _bare_model()
        pos_arr = self._configure(model, training=False, pos_dropout=0)
        patches_arr = np.arange(24, dtype="float32").reshape(2, 4, 3) + 100.0
        patches = paddle.to_tensor(patches_arr)
        result, pos_enc = model.apply_pos_enc(patches)
        # broadcast add of [1, 4, 3] over batch of 2.
        np.testing.assert_array_equal(result.numpy(), patches_arr + pos_arr)
        np.testing.assert_array_equal(pos_enc.numpy(), pos_arr)

    def test_train_mode_without_dropout_is_plain_add(self):
        # training=True but pos_dropout==0 -> the dropout branch is guarded off
        # (``self.training and self.pos_dropout > 0``), so the result must equal
        # the plain add with no elements zeroed.
        model = _bare_model()
        pos_arr = self._configure(model, training=True, pos_dropout=0)
        patches_arr = np.arange(24, dtype="float32").reshape(2, 4, 3) + 100.0
        patches = paddle.to_tensor(patches_arr)
        result, pos_enc = model.apply_pos_enc(patches)
        np.testing.assert_array_equal(result.numpy(), patches_arr + pos_arr)
        np.testing.assert_array_equal(pos_enc.numpy(), pos_arr)


def _mock_config():
    cfg = mock.MagicMock()
    cfg.hidden_size = 8
    cfg.params_dtype = "float32"
    cfg.rms_norm_eps = 1e-5
    return cfg


@unittest.skipUnless(IMPORT_OK, IMPORT_ERR)
class TestInitAllocations(unittest.TestCase):
    """__init__ parameter allocation and the seq_length arithmetic.

    Each test asserts the *correct* post-condition. All three are marked
    ``expectedFailure`` because __init__ allocates parameters with torch-style
    varargs shapes that Paddle's tensor factories reject (they take a single
    ``shape`` list). Production is left untouched.
    """

    @unittest.expectedFailure
    def test_seq_length_without_tokens(self):
        # add_class_token=False, use_mask_token=False -> first tensor factory
        # reached is position_embeddings.
        # CONFIRMED BUG radio.py:134 -> paddle.randn(1, max_num_patches,
        # hidden, dtype=...) passes shape as varargs *and* dtype twice.
        # Correct seq_length = (32//16)*(32//16) + 0 = 4.
        with (
            mock.patch(
                "paddlefleet.models.vision.radio.has_config_logger_enabled",
                return_value=False,
            ),
            mock.patch("paddlefleet.models.vision.radio.ColumnParallelLinear"),
            mock.patch("paddlefleet.models.vision.radio.TransformerBlock"),
        ):
            model = RADIOViTModel(
                transformer_config=_mock_config(),
                transformer_layer_spec=mock.MagicMock(),
                add_class_token=False,
                use_mask_token=False,
                img_h=32,
                img_w=32,
                patch_dim=16,
                max_img_h=32,
                max_img_w=32,
            )
        self.assertEqual(model.seq_length, 4)

    @unittest.expectedFailure
    def test_mask_token_allocated(self):
        # use_mask_token=True, add_class_token=False -> first tensor factory
        # reached is the mask token.
        # CONFIRMED BUG radio.py:114-116 -> paddle.zeros(1, hidden) passes shape
        # as varargs (Paddle reads the 2nd positional as dtype).
        with (
            mock.patch(
                "paddlefleet.models.vision.radio.has_config_logger_enabled",
                return_value=False,
            ),
            mock.patch("paddlefleet.models.vision.radio.ColumnParallelLinear"),
            mock.patch("paddlefleet.models.vision.radio.TransformerBlock"),
        ):
            model = RADIOViTModel(
                transformer_config=_mock_config(),
                transformer_layer_spec=mock.MagicMock(),
                use_mask_token=True,
                add_class_token=False,
                img_h=16,
                img_w=16,
                patch_dim=16,
                max_img_h=16,
                max_img_w=16,
            )
        self.assertTrue(model.use_mask_token)
        self.assertTrue(hasattr(model, "mask_token"))

    @unittest.expectedFailure
    def test_seq_length_with_class_token(self):
        # add_class_token=True -> first tensor factory reached is the class
        # token.
        # CONFIRMED BUG radio.py:121-127 -> paddle.randn(class_token_len,
        # hidden, dtype=...) passes shape as varargs and dtype twice.
        # Correct seq_length = (32//16)*(32//16) + class_token_len(3) = 7.
        with (
            mock.patch(
                "paddlefleet.models.vision.radio.has_config_logger_enabled",
                return_value=False,
            ),
            mock.patch("paddlefleet.models.vision.radio.ColumnParallelLinear"),
            mock.patch("paddlefleet.models.vision.radio.TransformerBlock"),
        ):
            model = RADIOViTModel(
                transformer_config=_mock_config(),
                transformer_layer_spec=mock.MagicMock(),
                use_mask_token=False,
                add_class_token=True,
                class_token_len=3,
                img_h=32,
                img_w=32,
                patch_dim=16,
                max_img_h=32,
                max_img_w=32,
            )
        self.assertEqual(model.seq_length, 7)


if __name__ == "__main__":
    unittest.main()

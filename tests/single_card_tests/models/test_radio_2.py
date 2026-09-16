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

"""Construction-time behavior tests for ``RADIOViTModel.__init__``.

This slice is disjoint from positional-encoding / forward tests: it exercises
the constructor's parameter allocation and input validation
(``paddlefleet.models.vision.radio``) plus the module-level ``HAVE_EINOPS``
availability flag. Every expected value is hand-derived from the config and the
constructor arithmetic.

Environment: CPU-executable control logic (no accelerator numerics), but it
still requires a working ``paddle`` import. Where paddle is absent the whole
module skips with an honest reason instead of reporting a pass.
"""

import importlib.util
import types
import unittest
from unittest import mock

# Guarded import: only ImportError is treated as "dependency missing" so a
# genuine API/compile regression inside radio.py still surfaces (not swallowed
# as a skip). The concrete reason is recorded and shown in the skip message.
try:
    import paddle

    from paddlefleet.models.vision import radio as radio_mod
    from paddlefleet.models.vision.radio import HAVE_EINOPS, RADIOViTModel

    _IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - depends on environment
    paddle = None
    radio_mod = None
    HAVE_EINOPS = None
    RADIOViTModel = None
    _IMPORT_ERROR = repr(exc)

_HAS_DEPS = _IMPORT_ERROR is None

_RADIO = "paddlefleet.models.vision.radio"


def _make_config(hidden_size=64):
    """Lightweight real config object for the constructor.

    Only the attributes read by ``RADIOViTModel.__init__`` before/around the
    tensor allocations are needed: ``hidden_size`` and ``params_dtype``.
    ``ColumnParallelLinear`` and ``TransformerBlock`` are patched out, so their
    config fields are not consumed here.
    """
    return types.SimpleNamespace(
        hidden_size=hidden_size,
        params_dtype="float32",
        rms_norm_eps=1e-5,
    )


@unittest.skipUnless(
    _HAS_DEPS, f"paddle/paddlefleet import unavailable: {_IMPORT_ERROR}"
)
class TestRADIOHaveEinopsFlag(unittest.TestCase):
    """The module-level ``HAVE_EINOPS`` flag must reflect real availability."""

    def test_have_einops_matches_real_availability(self):
        # Independent probe: the flag must not define its own correctness.
        expected = importlib.util.find_spec("einops") is not None
        self.assertIs(HAVE_EINOPS, expected)
        # Re-exported from the model module under test.
        self.assertIs(radio_mod.HAVE_EINOPS, HAVE_EINOPS)


@unittest.skipUnless(
    _HAS_DEPS, f"paddle/paddlefleet import unavailable: {_IMPORT_ERROR}"
)
class TestRADIOInitValidation(unittest.TestCase):
    """Input-dimension divisibility guards run before any tensor allocation."""

    def _build(self, **overrides):
        kwargs = {
            "transformer_config": _make_config(),
            "transformer_layer_spec": object(),
            "patch_dim": 16,
            "add_class_token": False,
            "use_mask_token": False,
        }
        kwargs.update(overrides)
        return RADIOViTModel(**kwargs)

    @mock.patch(f"{_RADIO}.TransformerBlock")
    @mock.patch(f"{_RADIO}.ColumnParallelLinear")
    @mock.patch(f"{_RADIO}.has_config_logger_enabled", return_value=False)
    def test_init_rejects_indivisible_img_h(self, _log, _col, _blk):
        # img_h=18 is not a multiple of patch_dim=16 -> assert at radio.py:99
        # fires before any parameter tensor is created (img_w is divisible).
        with self.assertRaises(AssertionError):
            self._build(img_h=18, img_w=16, max_img_h=32, max_img_w=32)

    @mock.patch(f"{_RADIO}.TransformerBlock")
    @mock.patch(f"{_RADIO}.ColumnParallelLinear")
    @mock.patch(f"{_RADIO}.has_config_logger_enabled", return_value=False)
    def test_init_rejects_indivisible_img_w(self, _log, _col, _blk):
        # img_h divisible so its assert passes; img_w=18 trips assert at
        # radio.py:100. Distinguishes the two guards (not a shared code path).
        with self.assertRaises(AssertionError):
            self._build(img_h=16, img_w=18, max_img_h=32, max_img_w=32)


@unittest.skipUnless(
    _HAS_DEPS, f"paddle/paddlefleet import unavailable: {_IMPORT_ERROR}"
)
class TestRADIOInitParameterAllocation(unittest.TestCase):
    """Constructor should allocate parameters with config-derived shapes.

    These document confirmed source bugs in ``RADIOViTModel.__init__``: the
    tensor-creation calls use PyTorch-style positional signatures that Paddle
    does not accept (no compat shim exists in ``utils/paddle_patch.py``), so the
    constructor raises before returning. Each test asserts the CORRECT intended
    shape and is marked ``expectedFailure``; production code is left unchanged.

      * radio.py:114  paddle.zeros(1, hidden)               -> dtype=hidden (int) invalid
      * radio.py:121  paddle.randn(class_token_len, hidden, dtype=...) -> dup 'dtype'
      * radio.py:134  paddle.randn(1, max_num_patches, hidden, dtype=...) -> dup 'dtype'

    The Paddle-correct forms would be paddle.zeros([1, hidden]) /
    paddle.randn([...], dtype=...).
    """

    @mock.patch(f"{_RADIO}.TransformerBlock")
    @mock.patch(f"{_RADIO}.ColumnParallelLinear")
    @mock.patch(f"{_RADIO}.has_config_logger_enabled", return_value=False)
    @unittest.expectedFailure
    def test_position_embeddings_shape(self, _log, _col, _blk):
        # max_img_h=48, max_img_w=32, patch_dim=16 (asymmetric to expose any
        # row/col swap): max_num_rows=3, max_num_cols=2 -> max_num_patches=6.
        # Correct position_embeddings shape: [1, 6, 64].
        model = RADIOViTModel(
            transformer_config=_make_config(hidden_size=64),
            transformer_layer_spec=object(),
            patch_dim=16,
            img_h=16,
            img_w=16,
            max_img_h=48,
            max_img_w=32,
            add_class_token=False,
            use_mask_token=False,
        )
        self.assertEqual(list(model.position_embeddings.shape), [1, 6, 64])

    @mock.patch(f"{_RADIO}.TransformerBlock")
    @mock.patch(f"{_RADIO}.ColumnParallelLinear")
    @mock.patch(f"{_RADIO}.has_config_logger_enabled", return_value=False)
    @unittest.expectedFailure
    def test_class_token_shape(self, _log, _col, _blk):
        # add_class_token=True, class_token_len=3, hidden=64
        # Correct class_token shape: [3, 64] (radio.py:121 currently raises).
        model = RADIOViTModel(
            transformer_config=_make_config(hidden_size=64),
            transformer_layer_spec=object(),
            patch_dim=16,
            img_h=16,
            img_w=16,
            max_img_h=32,
            max_img_w=32,
            add_class_token=True,
            class_token_len=3,
            use_mask_token=False,
        )
        self.assertEqual(list(model.class_token.shape), [3, 64])

    @mock.patch(f"{_RADIO}.TransformerBlock")
    @mock.patch(f"{_RADIO}.ColumnParallelLinear")
    @mock.patch(f"{_RADIO}.has_config_logger_enabled", return_value=False)
    @unittest.expectedFailure
    def test_mask_token_shape(self, _log, _col, _blk):
        # use_mask_token=True, hidden=64; add_class_token=False so the first
        # failing allocation is the mask token at radio.py:114.
        # Correct mask_token shape: [1, 64].
        model = RADIOViTModel(
            transformer_config=_make_config(hidden_size=64),
            transformer_layer_spec=object(),
            patch_dim=16,
            img_h=16,
            img_w=16,
            max_img_h=32,
            max_img_w=32,
            add_class_token=False,
            use_mask_token=True,
        )
        self.assertEqual(list(model.mask_token.shape), [1, 64])

    @mock.patch(f"{_RADIO}.TransformerBlock")
    @mock.patch(f"{_RADIO}.ColumnParallelLinear")
    @mock.patch(f"{_RADIO}.has_config_logger_enabled", return_value=False)
    @unittest.expectedFailure
    def test_seq_length_without_class_token(self, _log, _col, _blk):
        # img_h=48, img_w=32, patch_dim=16 -> (48//16)*(32//16) = 3*2 = 6.
        # add_class_token=False adds no class tokens -> seq_length == 6.
        # (seq_length is set at radio.py:129 but the instance never returns:
        #  position_embeddings at radio.py:134 raises first.)
        model = RADIOViTModel(
            transformer_config=_make_config(hidden_size=64),
            transformer_layer_spec=object(),
            patch_dim=16,
            img_h=48,
            img_w=32,
            max_img_h=48,
            max_img_w=32,
            add_class_token=False,
            use_mask_token=False,
        )
        self.assertEqual(model.seq_length, 6)


if __name__ == "__main__":
    unittest.main()

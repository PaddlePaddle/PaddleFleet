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

"""Coverage for the sync-free ``grad_loss`` validator shim.

The point of the shim is that validating ``grad_loss`` must not read anything
back to the host: under ``paddle.enable_compat`` a ``numel()`` comparison
becomes a blocking ``GpuMemcpySync`` on the legacy default stream, which
barriers the whole device and destroys the ``dsa_indexer_loss_bwd_p2p_overlap``
window. ``test_never_calls_numel`` is the regression guard -- it makes
``Tensor.numel`` explode, so any future rewrite that reintroduces the device
read fails here rather than silently costing overlap.
"""

import sys
import types
import unittest
from unittest.mock import patch

import paddle

from paddlefleet.cudnn_ops.indexer import _grad_loss_compat as mod


class TestValidateGradLossTensor(unittest.TestCase):
    def _device(self):
        return paddle.ones([], dtype="float32").device

    def test_accepts_zero_dim_and_one_element(self):
        for shape in ([], [1], [1, 1]):
            out = mod._validate_grad_loss_tensor(
                paddle.ones(shape, dtype="float32"), self._device()
            )
            self.assertEqual(list(out.shape), [1])
            self.assertEqual(out.dtype, paddle.float32)

    def test_output_is_detached(self):
        src = paddle.ones([], dtype="float32")
        src.stop_gradient = False
        self.assertTrue(
            mod._validate_grad_loss_tensor(src, self._device()).stop_gradient
        )

    def test_rejects_multi_element(self):
        with self.assertRaises(ValueError):
            mod._validate_grad_loss_tensor(
                paddle.ones([2], dtype="float32"), self._device()
            )

    def test_rejects_non_fp32(self):
        with self.assertRaises(ValueError):
            mod._validate_grad_loss_tensor(
                paddle.ones([], dtype="float16"), self._device()
            )

    def test_rejects_non_tensor(self):
        with self.assertRaises(TypeError):
            mod._validate_grad_loss_tensor(1.0, self._device())

    def test_never_calls_numel(self):
        """``numel()`` is the device read the shim exists to avoid."""

        def _boom(self, *args, **kwargs):
            raise AssertionError("grad_loss validation must not call numel()")

        grad_loss = paddle.ones([], dtype="float32")
        device = self._device()
        with patch.object(paddle.Tensor, "numel", _boom):
            out = mod._validate_grad_loss_tensor(grad_loss, device)
        self.assertEqual(list(out.shape), [1])


class TestPatchIndexerBackwardApi(unittest.TestCase):
    def setUp(self):
        self._saved = mod._PATCHED
        mod._PATCHED = False

    def tearDown(self):
        mod._PATCHED = self._saved

    def test_replaces_validator_and_is_idempotent(self):
        fake = types.ModuleType(mod._API)
        fake._validate_grad_loss_tensor = lambda *a, **k: None
        with patch.dict(sys.modules, {mod._API: fake}):
            mod.patch_indexer_backward_api()
            self.assertIs(
                fake._validate_grad_loss_tensor,
                mod._validate_grad_loss_tensor,
            )
            mod.patch_indexer_backward_api()
        self.assertTrue(mod._PATCHED)

    def test_noop_on_wheel_without_the_guard(self):
        """Older wheels expose ``_as_grad_loss_tensor`` and never sync."""
        fake = types.ModuleType(mod._API)
        fake._as_grad_loss_tensor = lambda *a, **k: None
        with patch.dict(sys.modules, {mod._API: fake}):
            mod.patch_indexer_backward_api()
        self.assertFalse(hasattr(fake, "_validate_grad_loss_tensor"))
        self.assertFalse(mod._PATCHED)

    def test_noop_when_api_not_imported(self):
        saved = sys.modules.pop(mod._API, None)
        try:
            mod.patch_indexer_backward_api()
        finally:
            if saved is not None:
                sys.modules[mod._API] = saved
        self.assertFalse(mod._PATCHED)


if __name__ == "__main__":
    unittest.main()

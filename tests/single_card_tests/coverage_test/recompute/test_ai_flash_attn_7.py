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
import os
import sys

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)

import contextlib
import importlib
import types
import unittest
from unittest.mock import MagicMock, patch

import paddle
from paddlefleet_ops import flash_mask_facade
from paddlefleet_ops.flash_mask_facade import get_fa_version

from paddlefleet.refined_recompute.flash_attn import flashattn_auto_cast


class TestGetFAVersionXPU(unittest.TestCase):
    """Tests for get_fa_version with XPU device."""

    @patch(
        "paddlefleet_ops.flash_mask_facade.paddle.get_device",
        return_value="xpu:0",
    )
    def test_xpu_returns_version_2(self, mock_device):
        """Test XPU device always returns version 2."""
        result = get_fa_version(64)
        self.assertEqual(result, 2)

    @patch(
        "paddlefleet_ops.flash_mask_facade.paddle.get_device",
        return_value="xpu:1",
    )
    def test_xpu_any_id(self, mock_device):
        """Test XPU device with any device ID returns version 2."""
        result = get_fa_version(128)
        self.assertEqual(result, 2)

    @patch(
        "paddlefleet_ops.flash_mask_facade.paddle.get_device",
        return_value="xpu:0",
    )
    def test_xpu_different_hdim(self, mock_device):
        """Test XPU returns version 2 regardless of hdim."""
        for hdim in [32, 64, 128, 256]:
            result = get_fa_version(hdim)
            self.assertEqual(result, 2)


class TestGetFAVersionGPU(unittest.TestCase):
    """Tests for get_fa_version with GPU device."""

    @patch(
        "paddlefleet_ops.flash_mask_facade.paddle.get_device",
        return_value="gpu:0",
    )
    @patch("paddlefleet_ops.flash_mask_facade.paddle.base.framework.get_flags")
    def test_gpu_returns_flag_value(self, mock_get_flags, mock_device):
        """Test GPU returns FLAGS_flash_attn_version."""
        mock_get_flags.return_value = {"FLAGS_flash_attn_version": 3}
        with patch(
            "paddlefleet_ops.flash_mask_facade.paddle.get_flags",
            return_value={"FLAGS_cudnn_deterministic": False},
        ):
            result = get_fa_version(64)
            self.assertEqual(result, 3)

    @patch(
        "paddlefleet_ops.flash_mask_facade.paddle.get_device",
        return_value="gpu:0",
    )
    @patch("paddlefleet_ops.flash_mask_facade.paddle.base.framework.get_flags")
    def test_gpu_flag_version_2(self, mock_get_flags, mock_device):
        """Test GPU returns 2 when flag is set to 2."""
        mock_get_flags.return_value = {"FLAGS_flash_attn_version": 2}
        with patch(
            "paddlefleet_ops.flash_mask_facade.paddle.get_flags",
            return_value={"FLAGS_cudnn_deterministic": False},
        ):
            result = get_fa_version(64)
            self.assertEqual(result, 2)


class TestGetFAVersionDeterministic(unittest.TestCase):
    """Tests for get_fa_version with deterministic mode (FA3).

    Which head dims survive deterministic mode depends on the backend: Paddle's
    own FA3 kernel has no ordered-accumulation backward above head_dim 128 and
    degrades to FA2, while FA3 on cutedsl keeps every head dim in its whitelist.
    Both the switch and kernel availability are therefore stated explicitly --
    otherwise the expected version would depend on the CI machine.
    """

    @patch.object(flash_mask_facade, "FLASHMASK_FA3_USE_CUTEDSL", False)
    @patch(
        "paddlefleet_ops.flash_mask_facade.paddle.get_device",
        return_value="gpu:0",
    )
    @patch("paddlefleet_ops.flash_mask_facade.paddle.base.framework.get_flags")
    @patch(
        "paddlefleet_ops.flash_mask_facade.paddle.get_flags",
        return_value={"FLAGS_cudnn_deterministic": True},
    )
    def test_deterministic_large_hdim_returns_2_on_paddle_kernel(
        self, mock_get_flags, mock_base_flags, mock_device
    ):
        """Deterministic FA3 with hdim>128 falls back to version 2."""
        mock_base_flags.return_value = {"FLAGS_flash_attn_version": 3}
        result = get_fa_version(256)
        self.assertEqual(result, 2)

    @patch.object(flash_mask_facade, "FLASHMASK_FA3_USE_CUTEDSL", True)
    @patch(
        "paddlefleet_ops.flash_mask_facade.is_flash_mask_available",
        lambda: True,
    )
    @patch(
        "paddlefleet_ops.flash_mask_facade.paddle.get_device",
        return_value="gpu:0",
    )
    @patch("paddlefleet_ops.flash_mask_facade.paddle.base.framework.get_flags")
    @patch(
        "paddlefleet_ops.flash_mask_facade.paddle.get_flags",
        return_value={"FLAGS_cudnn_deterministic": True},
    )
    def test_deterministic_large_hdim_keeps_3_on_cutedsl(
        self, mock_get_flags, mock_base_flags, mock_device
    ):
        """(256, 256) is in the cutedsl whitelist, so nothing degrades."""
        mock_base_flags.return_value = {"FLAGS_flash_attn_version": 3}
        result = get_fa_version(256, 256)
        self.assertEqual(result, 3)

    @patch.object(flash_mask_facade, "FLASHMASK_FA3_USE_CUTEDSL", True)
    @patch(
        "paddlefleet_ops.flash_mask_facade.is_flash_mask_available",
        lambda: True,
    )
    @patch(
        "paddlefleet_ops.flash_mask_facade.paddle.get_device",
        return_value="gpu:0",
    )
    @patch("paddlefleet_ops.flash_mask_facade.paddle.base.framework.get_flags")
    @patch(
        "paddlefleet_ops.flash_mask_facade.paddle.get_flags",
        return_value={"FLAGS_cudnn_deterministic": True},
    )
    def test_deterministic_hdim_128_keeps_3(
        self, mock_get_flags, mock_base_flags, mock_device
    ):
        """Deterministic FA3 with hdim==128 keeps version 3."""
        mock_base_flags.return_value = {"FLAGS_flash_attn_version": 3}
        result = get_fa_version(128)
        self.assertEqual(result, 3)

    @patch.object(flash_mask_facade, "FLASHMASK_FA3_USE_CUTEDSL", True)
    @patch(
        "paddlefleet_ops.flash_mask_facade.is_flash_mask_available",
        lambda: True,
    )
    @patch(
        "paddlefleet_ops.flash_mask_facade.paddle.get_device",
        return_value="gpu:0",
    )
    @patch("paddlefleet_ops.flash_mask_facade.paddle.base.framework.get_flags")
    @patch(
        "paddlefleet_ops.flash_mask_facade.paddle.get_flags",
        return_value={"FLAGS_cudnn_deterministic": True},
    )
    def test_deterministic_small_hdim_keeps_3(
        self, mock_get_flags, mock_base_flags, mock_device
    ):
        """Deterministic FA3 with hdim<=128 keeps version 3."""
        mock_base_flags.return_value = {"FLAGS_flash_attn_version": 3}
        result = get_fa_version(64)
        self.assertEqual(result, 3)


class TestGetFAVersionNonDeterministic(unittest.TestCase):
    """Tests for get_fa_version with non-deterministic mode."""

    @patch(
        "paddlefleet_ops.flash_mask_facade.paddle.get_device",
        return_value="gpu:0",
    )
    @patch("paddlefleet_ops.flash_mask_facade.paddle.base.framework.get_flags")
    @patch(
        "paddlefleet_ops.flash_mask_facade.is_flash_mask_available",
        return_value=True,
    )
    def test_non_deterministic_returns_flag(
        self, mock_available, mock_get_flags, mock_device
    ):
        """Test non-deterministic FA4 returns flag value when available."""
        mock_get_flags.return_value = {"FLAGS_flash_attn_version": 4}
        with patch(
            "paddlefleet_ops.flash_mask_facade.paddle.get_flags",
            return_value={"FLAGS_cudnn_deterministic": False},
        ):
            result = get_fa_version(64)
            self.assertEqual(result, 4)


class TestCutedslHotfixSwitch(unittest.TestCase):
    """``FLASHMASK_FA3_USE_CUTEDSL`` -- the cpp fallback it selects.

    Set from ``TransformerConfig.flash_attn_fa3_backend`` via
    ``set_fa3_backend``. The cpp kernel is on its way out, so these only pin the
    routing decisions, not numerics.
    """

    def test_uses_cutedsl_backend_follows_switch(self):
        with patch.object(
            flash_mask_facade, "FLASHMASK_FA3_USE_CUTEDSL", False
        ):
            self.assertFalse(flash_mask_facade.uses_cutedsl_backend(3))
            # FA4 is cutedsl-only, the switch does not reach it.
            self.assertTrue(flash_mask_facade.uses_cutedsl_backend(4))
            self.assertFalse(flash_mask_facade.uses_cutedsl_backend(2))

        with patch.object(flash_mask_facade, "FLASHMASK_FA3_USE_CUTEDSL", True):
            self.assertTrue(flash_mask_facade.uses_cutedsl_backend(3))

    def test_needs_value_padding_follows_switch(self):
        # (192, 128) is native on cutedsl but not on Paddle's FA3 kernel, so the
        # switch decides whether value has to be zero-padded to 192.
        with patch.object(flash_mask_facade, "FLASHMASK_FA3_USE_CUTEDSL", True):
            self.assertFalse(flash_mask_facade.needs_value_padding(3, 192, 128))
        with patch.object(
            flash_mask_facade, "FLASHMASK_FA3_USE_CUTEDSL", False
        ):
            self.assertTrue(flash_mask_facade.needs_value_padding(3, 192, 128))

    def test_switch_defaults_on_and_ignores_the_environment(self):
        # The choice now comes from ``TransformerConfig.flash_attn_fa3_backend``
        # through ``set_fa3_backend``, so a leftover environment variable of the
        # same name must not reach it. Reload once more on cleanup so the rest of
        # the process sees the default again.
        self.addCleanup(importlib.reload, flash_mask_facade)

        for raw in (None, "0", "1", "true"):
            with self.subTest(env=raw):
                env = dict(os.environ)
                env.pop("FLASHMASK_FA3_USE_CUTEDSL", None)
                if raw is not None:
                    env["FLASHMASK_FA3_USE_CUTEDSL"] = raw
                with patch.dict(os.environ, env, clear=True):
                    reloaded = importlib.reload(flash_mask_facade)
                    self.assertIs(reloaded.FLASHMASK_FA3_USE_CUTEDSL, True)

    def test_set_fa3_backend_switches_the_flag(self):
        # The choice is resolved once per process, so switching means dropping
        # the resolved one first -- calling the setter with a different value
        # would (deliberately) raise.
        self.addCleanup(flash_mask_facade._reset_fa3_backend)

        flash_mask_facade._reset_fa3_backend()
        flash_mask_facade.set_fa3_backend("cpp")
        self.assertFalse(flash_mask_facade.uses_cutedsl_backend(3))
        # Repeating the resolved choice is a no-op, not a conflict.
        flash_mask_facade.set_fa3_backend("cpp")
        self.assertFalse(flash_mask_facade.uses_cutedsl_backend(3))

        flash_mask_facade._reset_fa3_backend()
        flash_mask_facade.set_fa3_backend("cutedsl")
        self.assertTrue(flash_mask_facade.uses_cutedsl_backend(3))

    def test_set_fa3_backend_rejects_a_conflicting_choice(self):
        self.addCleanup(flash_mask_facade._reset_fa3_backend)

        flash_mask_facade._reset_fa3_backend()
        flash_mask_facade.set_fa3_backend("cpp")
        with self.assertRaisesRegex(
            ValueError, r"cannot change within a process"
        ):
            flash_mask_facade.set_fa3_backend("cutedsl")
        # The resolved choice survives the rejected call.
        self.assertFalse(flash_mask_facade.uses_cutedsl_backend(3))


@contextlib.contextmanager
def _pin_dispatch(fa_version, use_cutedsl):
    """Pin every input ``get_fa_version`` reads, so routing is machine-independent."""
    with (
        patch.object(
            flash_mask_facade, "FLASHMASK_FA3_USE_CUTEDSL", use_cutedsl
        ),
        patch(
            "paddlefleet_ops.flash_mask_facade.is_flash_mask_available",
            lambda: True,
        ),
        patch(
            "paddlefleet_ops.flash_mask_facade.paddle.get_device",
            return_value="gpu:0",
        ),
        patch(
            "paddlefleet_ops.flash_mask_facade.paddle.base.framework.get_flags",
            return_value={"FLAGS_flash_attn_version": fa_version},
        ),
        patch(
            "paddlefleet_ops.flash_mask_facade.paddle.get_flags",
            return_value={"FLAGS_cudnn_deterministic": False},
        ),
    ):
        yield


@contextlib.contextmanager
def _cute_entry(name, return_value):
    """Stand in for the cute kernel the facade imports lazily on the cutedsl path.

    Injected through ``sys.modules`` rather than patched on the real module, so
    the assertion holds whether or not ``paddlefleet_ops.flash_mask`` is
    installed on the machine running the test. This pins routing, not numerics.
    """
    stub = types.ModuleType("paddlefleet_ops.flash_mask")
    entry = MagicMock(return_value=return_value)
    setattr(stub, name, entry)
    with patch.dict(sys.modules, {"paddlefleet_ops.flash_mask": stub}):
        yield entry


class TestFacadeEntryRouting(unittest.TestCase):
    """Which kernel the ``flash_mask_facade`` entry points actually reach.

    The facade picks the implementation per call (``uses_cutedsl_backend``), and
    the no-mask entry ``flash_attention`` -- the one
    ``DotProductAttention._ec_compatible_flash_attention`` calls -- has no other
    coverage. Both candidate kernels are mocked and each test asserts exactly one
    of them ran: this pins the routing decision, not the numerics.
    """

    def setUp(self):
        self.shape = [1, 4, 2, 64]
        self.q = paddle.zeros(self.shape, dtype="float32")
        self.k = paddle.zeros(self.shape, dtype="float32")
        self.v = paddle.zeros(self.shape, dtype="float32")
        self.out = paddle.zeros(self.shape, dtype="float32")
        # 1 column, so the FA4 4-column mask restriction never applies here.
        self.mask = paddle.zeros([1, 2, 4, 1], dtype="int32")

    def test_flash_attention_routes_to_cpp_backend(self):
        with (
            _pin_dispatch(3, use_cutedsl=False),
            _cute_entry("flash_attention", (self.out, None)) as cute_kernel,
            patch(
                "paddle.nn.functional.flash_attention.flash_attention",
                return_value=(self.out, None),
            ) as cpp_kernel,
        ):
            result, _ = flash_mask_facade.flash_attention(
                self.q, self.k, self.v
            )

        cpp_kernel.assert_called_once()
        cute_kernel.assert_not_called()
        self.assertEqual(result.shape, self.shape)

    def test_flash_attention_routes_to_cutedsl_backend(self):
        with (
            _pin_dispatch(3, use_cutedsl=True),
            _cute_entry("flash_attention", (self.out, None)) as cute_kernel,
            patch(
                "paddle.nn.functional.flash_attention.flash_attention",
                return_value=(self.out, None),
            ) as cpp_kernel,
        ):
            result, _ = flash_mask_facade.flash_attention(
                self.q, self.k, self.v
            )

        cute_kernel.assert_called_once()
        cpp_kernel.assert_not_called()
        self.assertEqual(result.shape, self.shape)

    def test_flash_attention_fa2_stays_on_cpp_with_cutedsl_on(self):
        # FA2 never uses cutedsl, so a degrade to 2 must reach the cpp kernel
        # even while the switch is on.
        with (
            _pin_dispatch(2, use_cutedsl=True),
            _cute_entry("flash_attention", (self.out, None)) as cute_kernel,
            patch(
                "paddle.nn.functional.flash_attention.flash_attention",
                return_value=(self.out, None),
            ) as cpp_kernel,
        ):
            flash_mask_facade.flash_attention(self.q, self.k, self.v)

        cpp_kernel.assert_called_once()
        cute_kernel.assert_not_called()

    def test_flashmask_attention_routes_to_cpp_backend(self):
        with (
            _pin_dispatch(3, use_cutedsl=False),
            _cute_entry("flashmask_attention", self.out) as cute_kernel,
            patch(
                "paddle.nn.functional.flash_attention.flashmask_attention",
                return_value=self.out,
            ) as cpp_kernel,
        ):
            result = flash_mask_facade.flashmask_attention(
                self.q, self.k, self.v, self.mask
            )

        cpp_kernel.assert_called_once()
        cute_kernel.assert_not_called()
        self.assertEqual(result.shape, self.shape)

    def test_flashmask_attention_routes_to_cutedsl_backend(self):
        with (
            _pin_dispatch(3, use_cutedsl=True),
            _cute_entry("flashmask_attention", self.out) as cute_kernel,
            patch(
                "paddle.nn.functional.flash_attention.flashmask_attention",
                return_value=self.out,
            ) as cpp_kernel,
        ):
            result = flash_mask_facade.flashmask_attention(
                self.q, self.k, self.v, self.mask
            )

        cute_kernel.assert_called_once()
        cpp_kernel.assert_not_called()
        self.assertEqual(result.shape, self.shape)


class TestFlashattnAutoCastBasic(unittest.TestCase):
    """Tests for flashattn_auto_cast basic behavior."""

    def test_all_same_dtype_bfloat16(self):
        """Test no-op when all inputs are already bfloat16."""
        q = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        k = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        v = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        q_out, k_out, v_out = flashattn_auto_cast(q, k, v)
        self.assertIs(q_out, q)
        self.assertIs(k_out, k)
        self.assertIs(v_out, v)

    def test_all_same_dtype_float16(self):
        """Test no-op when all inputs are already float16."""
        q = paddle.randn([2, 4, 8], dtype=paddle.float16)
        k = paddle.randn([2, 4, 8], dtype=paddle.float16)
        v = paddle.randn([2, 4, 8], dtype=paddle.float16)
        q_out, k_out, v_out = flashattn_auto_cast(q, k, v, dtype=paddle.float16)
        self.assertIs(q_out, q)
        self.assertIs(k_out, k)
        self.assertIs(v_out, v)

    def test_cast_float32_to_bfloat16(self):
        """Test casting float32 tensors to bfloat16."""
        q = paddle.randn([2, 4, 8], dtype=paddle.float32)
        k = paddle.randn([2, 4, 8], dtype=paddle.float32)
        v = paddle.randn([2, 4, 8], dtype=paddle.float32)
        q_out, k_out, v_out = flashattn_auto_cast(q, k, v)
        self.assertEqual(q_out.dtype, paddle.bfloat16)
        self.assertEqual(k_out.dtype, paddle.bfloat16)
        self.assertEqual(v_out.dtype, paddle.bfloat16)

    def test_partial_cast(self):
        """Test only casting tensors that need it."""
        q = paddle.randn([2, 4, 8], dtype=paddle.float32)
        k = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        v = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        q_out, k_out, v_out = flashattn_auto_cast(q, k, v)
        self.assertEqual(q_out.dtype, paddle.bfloat16)
        # k and v already bfloat16, should not be cast
        self.assertIs(k_out, k)
        self.assertIs(v_out, v)

    def test_cast_to_float16(self):
        """Test casting to float16 target dtype."""
        q = paddle.randn([2, 4, 8], dtype=paddle.float32)
        k = paddle.randn([2, 4, 8], dtype=paddle.float32)
        v = paddle.randn([2, 4, 8], dtype=paddle.float32)
        q_out, k_out, v_out = flashattn_auto_cast(q, k, v, dtype=paddle.float16)
        self.assertEqual(q_out.dtype, paddle.float16)
        self.assertEqual(k_out.dtype, paddle.float16)
        self.assertEqual(v_out.dtype, paddle.float16)


if __name__ == "__main__":
    unittest.main()

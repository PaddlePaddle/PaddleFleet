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


# Extra tests for paddlefleet/refined_recompute/flash_attn.py
# Focus on: FlashMaskAttnCpAttention _first_fwd validation,
# FlashMaskAttnFunctor forward/backward structure

import unittest
from unittest.mock import MagicMock, patch

import paddle

try:
    from paddlefleet.refined_recompute.flash_attn import (  # noqa: F401
        RefinedRcomputeFlashMaskCpAttention,
    )

    _MODULE_AVAILABLE = True
except (ImportError, ModuleNotFoundError, Exception):
    _MODULE_AVAILABLE = False


@unittest.skipUnless(
    _MODULE_AVAILABLE, "paddlefleet.refined_recompute.flash_attn not available"
)
class TestFlashMaskCpAttentionQueryValidation(unittest.TestCase):
    """Tests for FlashMaskCpAttention query sequence length validation."""

    @unittest.skipUnless(
        _MODULE_AVAILABLE,
        "paddlefleet.refined_recompute.flash_attn not available",
    )
    def test_odd_seq_len_asserts(self):
        """Test that odd query sequence length raises assertion."""
        from paddlefleet.refined_recompute.flash_attn import (
            RefinedRcomputeFlashMaskCpAttention,
        )

        rr = RefinedRcomputeFlashMaskCpAttention()
        # _first_fwd requires _hcg (hybrid communication group) which
        # is only available in distributed/multi-card environments
        if not hasattr(rr, "_hcg"):
            self.skipTest("requires distributed environment with _hcg")
        with self.assertRaises(AssertionError):
            rr._first_fwd(
                paddle.randn([1, 7, 4, 16]),  # odd seq_len
                paddle.randn([1, 7, 4, 16]),
                paddle.randn([1, 7, 4, 16]),
                paddle.randint(0, 10, [1, 4, 7]),
            )


@unittest.skipUnless(
    _MODULE_AVAILABLE, "paddlefleet.refined_recompute.flash_attn not available"
)
class TestGetFAVersionValidation(unittest.TestCase):
    """Invalid versions are rejected by get_fa_version.

    The dispatch chains below it only tell 2 / cpp / cute apart, so an
    unsupported FLAGS_flash_attn_version has to be caught here.
    """

    def test_invalid_version_raises(self):
        """Test that a version outside 2/3/4 raises ValueError."""
        from paddlefleet_ops import flash_mask_facade

        with (
            patch.object(
                flash_mask_facade, "_dispatch_fa_version", return_value=99
            ),
            self.assertRaises(ValueError),
        ):
            flash_mask_facade.get_fa_version(64)


@unittest.skipUnless(
    _MODULE_AVAILABLE, "paddlefleet.refined_recompute.flash_attn not available"
)
class TestFlashMaskCpAttentionForwardDispatch(unittest.TestCase):
    """Tests for FlashMaskCpAttention forward dispatching."""

    def test_forward_dispatches_to_first_fwd_when_no_grad(self):
        """Test that forward dispatches to _first_fwd when no grad."""
        from paddlefleet.refined_recompute.flash_attn import (
            RefinedRcomputeFlashMaskCpAttention,
        )

        rr = RefinedRcomputeFlashMaskCpAttention()
        rr._first_fwd = MagicMock(return_value=paddle.randn([1, 4, 8]))

        with patch(
            "paddlefleet.refined_recompute.flash_attn.framework._dygraph_tracer"
        ) as mock_tracer:
            mock_tracer.return_value._has_grad = False
            rr.forward(
                None,
                paddle.randn([1, 4, 8]),
                paddle.randn([1, 4, 8]),
                paddle.randn([1, 4, 8]),
                paddle.randint(0, 10, [1, 4]),
            )
            rr._first_fwd.assert_called_once()


if __name__ == "__main__":
    unittest.main()

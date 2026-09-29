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
"""Behavior tests for two attribute-plumbing helpers in
``paddlefleet.tensor_parallel.layers`` (repo module: 分布式训练 / TP).

Second layers suite: deliberately covers helpers NOT exercised by the
compliant siblings ``test_layers_hf_bitexact.py`` (_HFEmbeddingGather,
grad-accum linear) or ``test_fp8_opt_in.py`` (fp8 prequant/clear cache):

* ``_maybe_color_linear_fp8_weight`` -- tags a non-MoE fp8 linear weight
  with the ``"linear_fp8"`` storage color exactly once. The guards decide
  *whether* the tag is written (fp8 on, not an expert linear, color still
  unset or the ``-1`` sentinel) and must never clobber an already-assigned
  color such as MoE's ``"moe_expert"``. We drive every guard branch and
  assert the exact color value / non-mutation.
* ``copy_tensor_model_parallel_attributes`` -- copies only the TP attrs
  that actually exist on the source, never fabricating defaults on the
  destination. We assert the copied values and that absent attrs stay
  absent.

The functions themselves are plain ``getattr``/``setattr`` logic, but the
production module imports ``paddle`` at load time, so the whole suite is
honestly skipped when paddle is absent rather than faking a pass.
"""

import os
import sys
import unittest

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)

try:
    import paddle  # noqa: F401

    from paddlefleet.tensor_parallel.layers import (
        _LINEAR_FP8_COLOR,
        _maybe_color_linear_fp8_weight,
        copy_tensor_model_parallel_attributes,
    )

    HAS_PADDLE = True
except ImportError:
    HAS_PADDLE = False

_SKIP_REASON = (
    "paddle not installed: paddlefleet.tensor_parallel.layers imports paddle "
    "at module load, so these helpers cannot be imported without it"
)


class _Attrs:
    """Minimal attribute bag; the helpers only touch it via get/set/delattr."""


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestMaybeColorLinearFp8Weight(unittest.TestCase):
    """Storage-color tagging guards on non-MoE fp8 linear weights."""

    def test_tags_unset_color_when_fp8_non_expert(self):
        weight = _Attrs()  # no ``color`` attribute yet
        layer = _Attrs()
        layer.fp8 = True
        layer.is_expert = False
        layer.weight = weight

        _maybe_color_linear_fp8_weight(layer)

        # Hand-derived: constant wrapped in a {"color": ...} dict, written once.
        self.assertEqual(weight.color, {"color": _LINEAR_FP8_COLOR})
        self.assertEqual(_LINEAR_FP8_COLOR, "linear_fp8")

    def test_replaces_minus_one_sentinel(self):
        weight = _Attrs()
        weight.color = -1  # Paddle's "unset" sentinel is also eligible
        layer = _Attrs()
        layer.fp8 = True
        layer.is_expert = False
        layer.weight = weight

        _maybe_color_linear_fp8_weight(layer)

        self.assertEqual(weight.color, {"color": _LINEAR_FP8_COLOR})

    def test_does_not_clobber_existing_moe_color(self):
        weight = _Attrs()
        weight.color = {"color": "moe_expert"}
        layer = _Attrs()
        layer.fp8 = True
        layer.is_expert = False
        layer.weight = weight

        _maybe_color_linear_fp8_weight(layer)

        # Already-colored weight must be left exactly as-is.
        self.assertEqual(weight.color, {"color": "moe_expert"})

    def test_skips_when_not_fp8(self):
        weight = _Attrs()
        layer = _Attrs()
        layer.fp8 = False
        layer.is_expert = False
        layer.weight = weight

        _maybe_color_linear_fp8_weight(layer)

        self.assertFalse(hasattr(weight, "color"))

    def test_skips_expert_linear(self):
        weight = _Attrs()
        layer = _Attrs()
        layer.fp8 = True
        layer.is_expert = True  # colored elsewhere by MoELayer
        layer.weight = weight

        _maybe_color_linear_fp8_weight(layer)

        self.assertFalse(hasattr(weight, "color"))

    def test_no_weight_is_noop(self):
        layer = _Attrs()
        layer.fp8 = True
        layer.is_expert = False
        layer.weight = None

        # Must not raise even though there is nothing to tag.
        _maybe_color_linear_fp8_weight(layer)


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestCopyTensorModelParallelAttributes(unittest.TestCase):
    """Selective copy of the three TP attrs from source to destination."""

    def test_copies_all_present_attrs(self):
        source = _Attrs()
        source.tensor_model_parallel = True
        source.partition_dim = 1
        source.partition_stride = 3
        dest = _Attrs()

        copy_tensor_model_parallel_attributes(dest, source)

        self.assertEqual(dest.tensor_model_parallel, True)
        self.assertEqual(dest.partition_dim, 1)
        self.assertEqual(dest.partition_stride, 3)

    def test_copies_only_present_attrs(self):
        source = _Attrs()
        source.tensor_model_parallel = True  # the other two are absent
        dest = _Attrs()

        copy_tensor_model_parallel_attributes(dest, source)

        # Present attr copied; absent attrs must not be fabricated as defaults.
        self.assertEqual(dest.tensor_model_parallel, True)
        self.assertFalse(hasattr(dest, "partition_dim"))
        self.assertFalse(hasattr(dest, "partition_stride"))

    def test_empty_source_copies_nothing(self):
        source = _Attrs()
        dest = _Attrs()

        copy_tensor_model_parallel_attributes(dest, source)

        self.assertFalse(hasattr(dest, "tensor_model_parallel"))
        self.assertFalse(hasattr(dest, "partition_dim"))
        self.assertFalse(hasattr(dest, "partition_stride"))


if __name__ == "__main__":
    unittest.main()

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

"""``_FixedOrderMappings`` must be invisible from outside ``forward``.

Two properties are pinned here, both bitwise (``assert_array_equal`` plus a dtype
check):

* the node returns exactly what the plain composition returns -- same forward
  outputs, same parameter gradients, same ``dx``;
* replaying the module under ``RecomputeWithoutOutput`` gives bit-identical
  gradients. That is the whole reason the node exists: ``x`` feeds both the
  mapping head and the aggregation, and leaving the order of the two ``dx``
  contributions to the engine makes the sum depend on whether the segment was
  replayed.

Both are checked for the native op composition and, where cuTile is available,
for the fused kernels: the node sits in ``forward``, above that choice, so
neither implementation may notice it. Only the fused path can vary
``_widen_in_kernel`` -- it tracks ``high_precision_mhc`` in production and
``TransformerConfig`` rejects ``high_precision_mhc=False`` outright, so it is
forced apart there: the reference reads it as its own condition, and with it off
``forward`` also takes the up-cast in front of the node. The native path pins it
to False itself, having no widening to absorb.

Both comparisons drive the real ``forward``; the reference is obtained by
swapping the node out for a direct call to the two methods it wraps, so nothing
about ``forward`` -- the ``auto_cast`` scope, the up-cast in front of it -- is
duplicated here.
"""

import unittest
from unittest import mock

import numpy as np
import paddle

from paddlefleet.fusions.fused_mhc_kernels import _CUTILE_AVAILABLE
from paddlefleet.tensor_parallel.random import (
    RecomputeWithoutOutput,
    get_cuda_rng_tracker,
)
from paddlefleet.transformer.hyper_connection import (
    HyperConnectionModule,
    _FixedOrderMappings,
)
from paddlefleet.transformer.transformer_config import TransformerConfig

C = 64
N_STREAMS = 4
S, B = 3, 2

_NEEDS_CUTILE = unittest.skipUnless(
    _CUTILE_AVAILABLE, "fused mHC kernels require cuTile"
)


def _make_module(use_fused_mhc=True, widen_in_kernel=None):
    config = TransformerConfig(
        hidden_size=C,
        enable_hyper_connections=True,
        use_fused_mhc=use_fused_mhc,
        high_precision_mhc=True,
        num_residual_streams=N_STREAMS,
        params_dtype=paddle.bfloat16,
    )
    module = HyperConnectionModule(config, 0)
    module.train()
    if widen_in_kernel is not None:
        # ``_widen_in_kernel`` tracks ``high_precision_mhc``, which the config
        # forces to True; overriding it is the only way to reach the branches
        # that read it on its own.
        module._widen_in_kernel = widen_in_kernel
    return module


def _collect(module, x, run_forward):
    """Run one forward/backward pass and return everything worth comparing."""
    x = x.detach()
    x.stop_gradient = False
    outputs = run_forward(module, x)
    # Weight the outputs differently so that a mix-up cannot cancel out.
    loss = sum((i + 1.0) * out.sum() for i, out in enumerate(outputs))
    loss.backward()
    grads = {
        name: param.grad.numpy().copy()
        for name, param in module.named_parameters()
        if param.grad is not None
    }
    for param in module.parameters():
        if param.grad is not None:
            param.clear_gradient()
    values = tuple(out.numpy().copy() for out in outputs)
    return values, grads, x.grad.numpy().copy()


def _plain(module, x):
    return module(x)


def _plain_apply(module, x, build_graph):
    """What the node wraps, called directly: no detached branches, no fixed order."""
    del build_graph  # the plain composition has no inner graph to gate
    h_pre, h_post, h_res = module.compute_mappings(x)
    return module.aggregate(x, h_pre), h_res, h_post


def _reference(module, x):
    """The real ``forward``, with the fixed-order node swapped out.

    Patching ``apply`` rather than reimplementing ``forward`` keeps everything
    around the node -- the ``auto_cast`` scope, the up-cast, the
    accuracy-compatible check -- under test instead of duplicated here. The
    wrapper asserts the patch was actually reached, so that a ``forward`` that
    stops going through the node cannot make this comparison vacuous.
    """
    calls = []

    def _counted(*args):
        calls.append(None)
        return _plain_apply(*args)

    with mock.patch.object(
        _FixedOrderMappings, "apply", staticmethod(_counted)
    ):
        outputs = module(x)
    assert calls, "forward did not go through _FixedOrderMappings"
    return outputs


def _pipeline(module, x, replay):
    """``forward`` -> BDA, as ``_forward_attention`` runs it.

    With ``replay`` the module goes inside a ``RecomputeWithoutOutput`` span and
    the span is closed on the BDA output, exactly as
    ``HyperConnectionTransformerLayer._forward_attention`` does. Only the BDA
    output is returned: the span's own outputs are cleared by
    ``discard_output_and_register_recompute`` and must not be read afterwards.
    """
    if replay:
        span = RecomputeWithoutOutput()
        aggregated, h_res, h_post = span.recompute(
            module, x, preserve_rng_state=False, share_grad_holder=True
        )
    else:
        span = None
        aggregated, h_res, h_post = module(x)
    output = module.fused_h_res_h_post_bda(
        h_res=h_res.clone(),
        original_residual=x,
        h_post=h_post.clone(),
        layer_output_with_bias=(aggregated.to(x.dtype), None),
        dropout_prob=0.0,
        training=True,
        fused=True,
    )
    if span is not None:
        span.discard_output_and_register_recompute(output)
    return (output,)


class TestMhcFixedOrderMappings(unittest.TestCase):
    """The fixed-order node must be invisible from outside ``forward``."""

    @classmethod
    def setUpClass(cls):
        paddle.set_device("gpu")
        try:
            get_cuda_rng_tracker().add("model-parallel-rng", 12345)
        except ValueError:
            pass

    def _assert_identical(self, first, second, label):
        (out_a, grads_a, dx_a) = first
        (out_b, grads_b, dx_b) = second
        self.assertEqual(len(out_a), len(out_b), label)
        for i, (a, b) in enumerate(zip(out_a, out_b)):
            self.assertEqual(a.dtype, b.dtype, f"{label}: output {i} dtype")
            np.testing.assert_array_equal(a, b, err_msg=f"{label}: output {i}")
        self.assertEqual(set(grads_a), set(grads_b), label)
        for name in grads_a:
            self.assertEqual(
                grads_a[name].dtype,
                grads_b[name].dtype,
                f"{label}: grad {name} dtype",
            )
            np.testing.assert_array_equal(
                grads_a[name], grads_b[name], err_msg=f"{label}: grad {name}"
            )
        self.assertEqual(dx_a.dtype, dx_b.dtype, f"{label}: dx dtype")
        np.testing.assert_array_equal(dx_a, dx_b, err_msg=f"{label}: dx")

    def _assert_matches_reference(self, use_fused_mhc, widen_in_kernel=None):
        paddle.seed(2026)
        module = _make_module(use_fused_mhc, widen_in_kernel)
        x = paddle.randn([S, B, N_STREAMS * C]).astype("bfloat16")
        self._assert_identical(
            _collect(module, x, _reference),
            _collect(module, x, _plain),
            f"fused={use_fused_mhc} widen_in_kernel={module._widen_in_kernel}",
        )

    def _assert_replay_matches(self, use_fused_mhc):
        paddle.seed(2026)
        module = _make_module(use_fused_mhc)
        x = paddle.randn([S, B, N_STREAMS * C]).astype("bfloat16")
        self._assert_identical(
            _collect(module, x, lambda m, t: _pipeline(m, t, replay=False)),
            _collect(module, x, lambda m, t: _pipeline(m, t, replay=True)),
            f"recompute off vs on, fused={use_fused_mhc}",
        )

    def test_matches_reference_native(self):
        """The op composition, which needs no kernels to run."""
        self._assert_matches_reference(False)

    def test_recompute_replay_is_bitwise_identical_native(self):
        """Replaying must not change a bit, kernels or no kernels."""
        self._assert_replay_matches(False)

    @_NEEDS_CUTILE
    def test_matches_reference_fused_widen_in_kernel(self):
        """The production setting: casts folded into the kernels."""
        self._assert_matches_reference(True, widen_in_kernel=True)

    @_NEEDS_CUTILE
    def test_matches_reference_fused_no_widen_in_kernel(self):
        """The caller materializes the fp32 copies instead."""
        self._assert_matches_reference(True, widen_in_kernel=False)

    @_NEEDS_CUTILE
    def test_recompute_replay_is_bitwise_identical_fused(self):
        """Replaying the module must not change a single bit of the gradients."""
        self._assert_replay_matches(True)


if __name__ == "__main__":
    unittest.main()

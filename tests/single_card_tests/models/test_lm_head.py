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
"""Unit tests for paddlefleet.models.gpt.lm_head.

Scope and honesty notes
-----------------------
These tests target the pieces of the LM-head module whose behaviour can be
exercised and hand-derived without a distributed process group or a GPU:

* ``SegLU``                         -- the pure, element-wise learnable
  modulation applied to logits. It is CPU-executable, so both its forward
  numerics and its full (x / ranges / ts) gradient set are checked against
  independently hand-derived values.
* ``GPTLMHead.forward`` /
  ``GPTMainLMHead.forward`` /
  ``GPTMTPLMHead.forward``          -- the split / dispatch / aggregate routing
  logic. These methods consume only ``self.config``, ``self._stash_cu_seqlens_q``
  and ``self._forward``; they never touch the ``ColumnParallelLinear``
  internals directly. The real, unbound methods are therefore driven against a
  minimal ``self`` whose ``_forward`` is a spy that returns markers depending on
  BOTH input content AND call index, so a swapped / misordered / dropped split
  is caught at the aggregation site.
* ``embedding_weight``              -- the trivial weight-exposure property.
* ``sharded_state_dict``            -- the ``world_size`` -> ``shard_rules``
  branch selection and forwarding to ``build_sharded_state_dict``.
* ``_stash_cu_seqlens_q``           -- the no-op vs. write-to-``LanguageLoss``
  contract.

NOT covered here (require a constructed layer + real process group / GPU, and
are left to single/multi-card tests): ``GPTLMHead.__init__`` (multimax param
allocation, TP weight init), ``GPTLMHead._forward`` (the real linear projection,
fused-CE 3/5-tuple emission, recompute and sequence-parallel transposes), and
``build_schedule_node`` (wraps ``ScheduleNode``). The routing tests deliberately
replace ``_forward`` with a spy: the projection numerics belong to those
device-level tests, not here.

Paddle is a hard dependency of the module under test. When it (or the import
chain it pulls in) is unavailable the whole file skips with the concrete import
error rather than reporting a false pass.
"""

import types
import unittest

import numpy as np

try:
    import paddle

    import paddlefleet.models.gpt.lm_head as lm_head_module
    from paddlefleet.models.gpt.lm_head import (
        GPTLMHead,
        GPTMainLMHead,
        GPTMTPLMHead,
        SegLU,
    )

    _IMPORT_ERROR = None
except ImportError as exc:  # genuinely missing dependency, not a masked bug
    paddle = None
    lm_head_module = None
    SegLU = None
    GPTLMHead = None
    GPTMainLMHead = None
    GPTMTPLMHead = None
    _IMPORT_ERROR = exc

_HAS_DEPS = _IMPORT_ERROR is None
_SKIP_REASON = (
    f"paddle / paddlefleet.models.gpt.lm_head not importable: {_IMPORT_ERROR!r}"
)


class _ForwardSpy:
    """Stand-in for the projection unit ``GPTLMHead._forward``.

    Records every tensor it receives (so the caller can verify the split
    content and ordering) and returns a marker that depends on BOTH the input
    content and the call index::

        marker_i = x.astype(float32) * (i + 1) + 1000.0 * (i + 1)

    The index term makes a swap / reorder of results detectable, and the
    content term makes a wrong-slice detectable. The real projection numerics
    are exercised by single-card tests, not here.
    """

    def __init__(self):
        self.received = []

    def __call__(self, hidden_states):
        idx = len(self.received)
        self.received.append(hidden_states)
        return hidden_states.astype("float32") * (idx + 1) + 1000.0 * (idx + 1)


def _expected_marker(chunk_np, idx):
    return chunk_np.astype(np.float32) * (idx + 1) + 1000.0 * (idx + 1)


def _routing_self(cls, num_nextn_predict_layers, mtp_load_weight_only=False):
    """Minimal ``self`` for the ``*.forward`` routing methods.

    ``_stash_cu_seqlens_q`` is bound to the REAL method (a no-op when the dict
    carries no ``cu_seqlens_q``), keeping it on the execution chain; ``_forward``
    is a spy.
    """
    dummy = types.SimpleNamespace()
    dummy.config = types.SimpleNamespace(
        num_nextn_predict_layers=num_nextn_predict_layers,
        mtp_load_weight_only=mtp_load_weight_only,
    )
    dummy._stash_cu_seqlens_q = types.MethodType(cls._stash_cu_seqlens_q, dummy)
    spy = _ForwardSpy()
    dummy._forward = spy
    return dummy, spy


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestSegLUForward(unittest.TestCase):
    """``SegLU`` element-wise modulation, forward numerics."""

    def test_zero_params_is_identity(self):
        # ranges = ts = 0 -> every relu term is scaled by 0 -> output == input.
        # This is the cold-start / resume-safe contract from the module docs.
        x = paddle.to_tensor([-2.0, 0.5, 1.5, 3.0], dtype="float32")
        ranges = paddle.zeros([4], dtype="float32")
        ts = paddle.zeros([4], dtype="float32")
        out = SegLU(x, ranges, ts)
        np.testing.assert_array_equal(out.numpy(), x.numpy())

    def test_forward_matches_hand_derived_values(self):
        # x=[-2, 0.5, 1.5, 3], ranges=[1,0,2,1], ts=[0.5,2,1,3].
        # out = x + ts0*relu(r0-x) + ts1*relu(x-r1)
        #         + ts2*relu(r2-x)^2 + ts3*relu(x-r3)^2
        #   x=-2 : -2 + 0.5*3 + 2*0   + 1*4^2 + 3*0     = 15.5
        #   x=0.5:  0.5+0.5*0.5+2*0.5 + 1*1.5^2+3*0     =  4.0
        #   x=1.5:  1.5+0.5*0 +2*1.5  + 1*0.5^2+3*0.5^2 =  5.5
        #   x=3  :  3 + 0.5*0 +2*3    + 1*0   +3*2^2    = 21.0
        x = paddle.to_tensor([-2.0, 0.5, 1.5, 3.0], dtype="float32")
        ranges = paddle.to_tensor([1.0, 0.0, 2.0, 1.0], dtype="float32")
        ts = paddle.to_tensor([0.5, 2.0, 1.0, 3.0], dtype="float32")
        out = SegLU(x, ranges, ts)
        expected = np.array([15.5, 4.0, 5.5, 21.0], dtype=np.float32)
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-6, atol=1e-6)

    def test_input_is_not_mutated(self):
        # The module clones before the in-place ``+=`` precisely so the linear's
        # output (still referenced by the autograd graph) is not corrupted.
        x = paddle.to_tensor([-2.0, 0.5, 1.5, 3.0], dtype="float32")
        before = x.numpy().copy()
        ranges = paddle.to_tensor([1.0, 0.0, 2.0, 1.0], dtype="float32")
        ts = paddle.to_tensor([0.5, 2.0, 1.0, 3.0], dtype="float32")
        _ = SegLU(x, ranges, ts)
        np.testing.assert_array_equal(x.numpy(), before)

    def test_multidim_applies_elementwise(self):
        # SegLU is element-wise; a [2,2] input must give the per-element result,
        # so a reshape/transpose bug in the modulation is caught.
        x = paddle.to_tensor([[-2.0, 0.5], [1.5, 3.0]], dtype="float32")
        ranges = paddle.to_tensor([1.0, 0.0, 2.0, 1.0], dtype="float32")
        ts = paddle.to_tensor([0.5, 2.0, 1.0, 3.0], dtype="float32")
        out = SegLU(x, ranges, ts)
        expected = np.array([[15.5, 4.0], [5.5, 21.0]], dtype=np.float32)
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-6, atol=1e-6)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestSegLUBackward(unittest.TestCase):
    """``SegLU`` gradients w.r.t. x, ranges and ts (upstream grad = ones).

    With x=[-2,0.5,1.5,3], ranges=[1,0,2,1], ts=[0.5,2,1,3] (no x equals any
    range, so relu subgradients are unambiguous):

      dL/dx  = 1 - ts0*[x<r0] + ts1*[x>r1]
                 - 2*ts2*relu(r2-x) + 2*ts3*relu(x-r3)
             = [-7.5, -0.5, 5.0, 15.0]
      dL/dr  = [ ts0*sum[x<r0], -ts1*sum[x>r1],
                 2*ts2*sum relu(r2-x), -2*ts3*sum relu(x-r3) ]
             = [1.0, -6.0, 12.0, -15.0]
      dL/dts = [ sum relu(r0-x), sum relu(x-r1),
                 sum relu(r2-x)^2, sum relu(x-r3)^2 ]
             = [3.5, 5.0, 18.5, 4.25]
    """

    def test_full_gradient_set_matches_hand_derived(self):
        x = paddle.to_tensor([-2.0, 0.5, 1.5, 3.0], dtype="float32")
        ranges = paddle.to_tensor([1.0, 0.0, 2.0, 1.0], dtype="float32")
        ts = paddle.to_tensor([0.5, 2.0, 1.0, 3.0], dtype="float32")
        for t in (x, ranges, ts):
            t.stop_gradient = False

        out = SegLU(x, ranges, ts)
        out.backward(paddle.ones_like(out))

        # Every input on the graph must actually receive a gradient (a
        # forward-only / detached implementation would leave these None).
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(ranges.grad)
        self.assertIsNotNone(ts.grad)

        np.testing.assert_allclose(
            x.grad.numpy(),
            np.array([-7.5, -0.5, 5.0, 15.0], dtype=np.float32),
            rtol=1e-5,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            ranges.grad.numpy(),
            np.array([1.0, -6.0, 12.0, -15.0], dtype=np.float32),
            rtol=1e-5,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            ts.grad.numpy(),
            np.array([3.5, 5.0, 18.5, 4.25], dtype=np.float32),
            rtol=1e-5,
            atol=1e-6,
        )


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestGPTLMHeadForwardRouting(unittest.TestCase):
    """``GPTLMHead.forward`` split / dispatch / aggregate contract."""

    def test_mtp_splits_and_returns_all_heads_in_order(self):
        # num_nextn_predict_layers=2 -> split rows into 3 equal chunks, run
        # _forward on main + each MTP chunk, return a list [main, mtp0, mtp1].
        dummy, spy = _routing_self(
            GPTLMHead, num_nextn_predict_layers=2, mtp_load_weight_only=False
        )
        hn = np.arange(6 * 4, dtype=np.float32).reshape([6, 4])
        hidden = paddle.to_tensor(hn)
        result = GPTLMHead.forward(dummy, {"hidden_states": hidden})

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)  # main + 2 MTP
        self.assertEqual(len(spy.received), 3)

        expected_chunks = [hn[0:2], hn[2:4], hn[4:6]]
        for i, chunk in enumerate(expected_chunks):
            # Each split chunk reached _forward with the correct content/order.
            np.testing.assert_array_equal(spy.received[i].numpy(), chunk)
            # Aggregated output preserves per-slot identity (no swap/reorder).
            np.testing.assert_allclose(
                result[i].numpy(),
                _expected_marker(chunk, i),
                rtol=1e-6,
                atol=1e-6,
            )

    def test_no_mtp_calls_forward_once_and_returns_bare_tensor(self):
        dummy, spy = _routing_self(GPTLMHead, num_nextn_predict_layers=None)
        hn = np.arange(6 * 4, dtype=np.float32).reshape([6, 4])
        hidden = paddle.to_tensor(hn)
        result = GPTLMHead.forward(dummy, {"hidden_states": hidden})

        # No split: the whole tensor is projected once and returned unwrapped.
        self.assertNotIsInstance(result, list)
        self.assertEqual(len(spy.received), 1)
        np.testing.assert_array_equal(spy.received[0].numpy(), hn)
        np.testing.assert_allclose(
            result.numpy(), _expected_marker(hn, 0), rtol=1e-6, atol=1e-6
        )

    def test_mtp_load_weight_only_suppresses_split(self):
        # num_nextn_predict_layers=2 but mtp_load_weight_only=True -> the guard
        # falls through to the single-call branch (no split), distinguishing it
        # from test_mtp_splits_and_returns_all_heads_in_order.
        dummy, spy = _routing_self(
            GPTLMHead, num_nextn_predict_layers=2, mtp_load_weight_only=True
        )
        hn = np.arange(6 * 4, dtype=np.float32).reshape([6, 4])
        hidden = paddle.to_tensor(hn)
        result = GPTLMHead.forward(dummy, {"hidden_states": hidden})

        self.assertNotIsInstance(result, list)
        self.assertEqual(len(spy.received), 1)
        np.testing.assert_array_equal(spy.received[0].numpy(), hn)

    def test_zero_mtp_layers_suppresses_split(self):
        # num_nextn_predict_layers=0 -> the ``> 0`` guard is false -> single call.
        dummy, spy = _routing_self(GPTLMHead, num_nextn_predict_layers=0)
        hn = np.arange(4 * 4, dtype=np.float32).reshape([4, 4])
        hidden = paddle.to_tensor(hn)
        result = GPTLMHead.forward(dummy, {"hidden_states": hidden})

        self.assertNotIsInstance(result, list)
        self.assertEqual(len(spy.received), 1)
        np.testing.assert_array_equal(spy.received[0].numpy(), hn)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestGPTMainLMHeadForward(unittest.TestCase):
    """``GPTMainLMHead.forward`` -- main split only, None-value filtering."""

    def test_mtp_config_uses_only_main_split(self):
        # num_nextn_predict_layers=1 -> split into 2; the MAIN head must project
        # tensor_list[0] only (the MTP chunk is handled by GPTMTPLMHead). A bug
        # that fed the wrong chunk, or also processed the MTP chunk, is caught.
        dummy, spy = _routing_self(GPTMainLMHead, num_nextn_predict_layers=1)
        hn = np.arange(4 * 4, dtype=np.float32).reshape([4, 4])
        hidden = paddle.to_tensor(hn)
        mtp_loss = [paddle.to_tensor(1.0)]
        result = GPTMainLMHead.forward(
            dummy, {"hidden_states": hidden, "mtp_loss": mtp_loss}
        )

        self.assertEqual(len(spy.received), 1)
        np.testing.assert_array_equal(spy.received[0].numpy(), hn[0:2])
        self.assertIn("logits", result)
        np.testing.assert_allclose(
            result["logits"].numpy(),
            _expected_marker(hn[0:2], 0),
            rtol=1e-6,
            atol=1e-6,
        )
        # A non-None mtp_loss is forwarded verbatim (same object).
        self.assertIs(result["mtp_loss"], mtp_loss)

    def test_absent_mtp_loss_is_filtered_out(self):
        # No mtp_loss in dict_args -> value is None -> key must be dropped from
        # the return dict (not left as an explicit None).
        dummy, spy = _routing_self(GPTMainLMHead, num_nextn_predict_layers=None)
        hn = np.arange(3 * 4, dtype=np.float32).reshape([3, 4])
        hidden = paddle.to_tensor(hn)
        result = GPTMainLMHead.forward(dummy, {"hidden_states": hidden})

        self.assertIn("logits", result)
        self.assertNotIn("mtp_loss", result)
        self.assertEqual(len(spy.received), 1)
        np.testing.assert_array_equal(spy.received[0].numpy(), hn)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestGPTMTPLMHeadForward(unittest.TestCase):
    """``GPTMTPLMHead.forward`` -- MTP splits only (index i+1), dict mutation."""

    def test_processes_mtp_chunks_only_and_mutates_dict(self):
        # num_nextn_predict_layers=2 -> split into 3; the loop projects
        # tensor_list[1] and tensor_list[2] (the ``i + 1`` indexing SKIPS the
        # main chunk index 0). An off-by-one (using ``i``) would feed the main
        # chunk and is caught here.
        dummy, spy = _routing_self(GPTMTPLMHead, num_nextn_predict_layers=2)
        hn = np.arange(6 * 4, dtype=np.float32).reshape([6, 4])
        hidden = paddle.to_tensor(hn)
        dict_args = {"hidden_states": hidden}
        result = GPTMTPLMHead.forward(dummy, dict_args)

        # Returns the SAME dict object, augmented with mtp_logits.
        self.assertIs(result, dict_args)
        self.assertIn("mtp_logits", result)

        mtp_chunks = [hn[2:4], hn[4:6]]  # NOT hn[0:2]
        self.assertEqual(len(spy.received), 2)
        for i, chunk in enumerate(mtp_chunks):
            np.testing.assert_array_equal(spy.received[i].numpy(), chunk)

        # The main chunk (rows 0-1) must never reach _forward.
        for got in spy.received:
            self.assertFalse(
                np.array_equal(got.numpy(), hn[0:2]),
                "main chunk leaked into the MTP head",
            )

        self.assertEqual(len(result["mtp_logits"]), 2)
        for i, chunk in enumerate(mtp_chunks):
            np.testing.assert_allclose(
                result["mtp_logits"][i].numpy(),
                _expected_marker(chunk, i),
                rtol=1e-6,
                atol=1e-6,
            )


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestEmbeddingWeightProperty(unittest.TestCase):
    """``embedding_weight`` returns ``self.weight`` (tied-embedding exposure)."""

    def test_all_heads_expose_weight_by_identity(self):
        for cls in (GPTLMHead, GPTMainLMHead, GPTMTPLMHead):
            sentinel = object()
            dummy = types.SimpleNamespace(weight=sentinel)
            self.assertIs(
                cls.embedding_weight.fget(dummy),
                sentinel,
                f"{cls.__name__}.embedding_weight must return self.weight",
            )


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestShardedStateDict(unittest.TestCase):
    """``sharded_state_dict`` selects shard rules by ``world_size`` and forwards
    the real state dict / prefix to ``build_sharded_state_dict``."""

    def _patch_builder(self):
        calls = {}
        orig = lm_head_module.build_sharded_state_dict

        def recorder(state_dict, shard_rules, prefix):
            calls["state_dict"] = state_dict
            calls["shard_rules"] = shard_rules
            calls["prefix"] = prefix
            return "SENTINEL_RESULT"

        lm_head_module.build_sharded_state_dict = recorder
        self.addCleanup(
            setattr, lm_head_module, "build_sharded_state_dict", orig
        )
        return calls

    def _dummy(self, world_size):
        sd = {"weight": object(), "bias": object()}
        sd_prefixes = []
        dummy = types.SimpleNamespace(world_size=world_size)
        dummy.state_dict = lambda structured_name_prefix="": (
            sd_prefixes.append(structured_name_prefix) or sd
        )
        return dummy, sd, sd_prefixes

    def test_single_rank_uses_none_shard_rules(self):
        calls = self._patch_builder()
        dummy, sd, sd_prefixes = self._dummy(world_size=1)
        result = GPTLMHead.sharded_state_dict(
            dummy, structured_name_prefix="model."
        )
        self.assertEqual(result, "SENTINEL_RESULT")
        self.assertIsNone(calls["shard_rules"])
        self.assertIs(calls["state_dict"], sd)
        self.assertEqual(calls["prefix"], "model.")
        # Production always snapshots state_dict with an empty prefix.
        self.assertEqual(sd_prefixes, [""])

    def test_multi_rank_shards_weight_and_bias_on_axis0(self):
        calls = self._patch_builder()
        dummy, sd, _ = self._dummy(world_size=2)
        GPTLMHead.sharded_state_dict(dummy, structured_name_prefix="p.")
        self.assertEqual(calls["shard_rules"], {"weight": 0, "bias": 0})
        self.assertIs(calls["state_dict"], sd)
        self.assertEqual(calls["prefix"], "p.")


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestStashCuSeqlensQ(unittest.TestCase):
    """``_stash_cu_seqlens_q`` -- no-op when absent, writes to LanguageLoss."""

    def test_no_op_when_cu_seqlens_absent(self):
        dummy = types.SimpleNamespace()
        # No cu_seqlens_q key -> returns None, imports nothing, sets nothing.
        result = GPTLMHead._stash_cu_seqlens_q(
            dummy, {"hidden_states": object()}
        )
        self.assertIsNone(result)

    def test_writes_cu_seqlens_to_language_loss_class(self):
        try:
            from paddlefleet.models.common.language_loss.language_loss import (
                LanguageLoss,
            )
        except ImportError as exc:  # precise: genuinely missing submodule dep
            self.skipTest(f"LanguageLoss not importable: {exc!r}")

        had = hasattr(LanguageLoss, "_cu_seqlens_q_stash")
        orig = getattr(LanguageLoss, "_cu_seqlens_q_stash", None)

        def restore():
            if had:
                LanguageLoss._cu_seqlens_q_stash = orig
            elif hasattr(LanguageLoss, "_cu_seqlens_q_stash"):
                delattr(LanguageLoss, "_cu_seqlens_q_stash")

        self.addCleanup(restore)

        sentinel = object()
        dummy = types.SimpleNamespace()
        GPTLMHead._stash_cu_seqlens_q(dummy, {"cu_seqlens_q": sentinel})
        # The exact object travels onto the loss class for per-doc label rolling.
        self.assertIs(LanguageLoss._cu_seqlens_q_stash, sentinel)


if __name__ == "__main__":
    unittest.main()

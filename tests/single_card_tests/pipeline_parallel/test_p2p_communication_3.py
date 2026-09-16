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

"""CPU-only behavior tests for paddlefleet pipeline_parallel p2p_communication.

Exercises the CPU-executable pure logic of the production module
``paddlefleet.pipeline_parallel.pp_utils.p2p_communication``:

* ``initialize_p2p_groups`` global config propagation, observed through its
  real consumer ``_is_valid_send_recv_partial`` (config -> consumer), plus the
  ``enable_timer`` branch storing the timer object it obtains.
* ``SendRecvMeta._obtain_send_message`` / ``set_send_message`` shape,
  dtype-number and key extraction, including the tuple ``stop_gradient``
  skipping contract.
* ``SendRecvMeta.check_send_message`` match vs. shape/dtype/key mismatch
  assertion contract, and the unset (no-op) early return.
* ``SendRecvMeta.init_or_erase_meta`` reset contract.
* ``_is_valid_send_recv_partial`` enable-flag gating and divisibility.
* ``_send_on_calc_stream`` / ``_recv_on_calc_stream`` group=None assertion and
  real partial/full dispatch with peer-rank translation, driven by the REAL
  ``_is_valid_send_recv_partial`` (not a patched decision helper).
* ``allgather_partial`` short-circuit, membership gate and partial-allgather
  dispatch.

The process group is a lightweight local fake with argument-capturing methods.
Real cross-rank send/recv, batched calc-stream collectives and the pipeline
schedule (``send_meta`` wire encoding, ``_p2p_ops``, ``_batched_p2p_ops``,
``P2pHelper`` methods) require a real process group and are NOT exercised here;
they belong to a multi-card test. Expected dtype numbers are hand-derived from
the documented PADDLE_TO_NUMBER mapping, independent of ``paddle_2_number``.
"""

import unittest
from unittest import mock

try:
    import paddle

    paddle.set_device("cpu")

    from paddlefleet.pipeline_parallel.pp_utils import (
        p2p_communication as p2p_mod,
    )
    from paddlefleet.pipeline_parallel.pp_utils.p2p_communication import (
        SendRecvMeta,
        _is_valid_send_recv_partial,
        _recv_on_calc_stream,
        _send_on_calc_stream,
        allgather_partial,
        initialize_p2p_groups,
    )

    _IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
    paddle = None
    p2p_mod = None
    _IMPORT_ERROR = exc

_SKIP_REASON = (
    f"paddle/paddlefleet import failed: {_IMPORT_ERROR!r}"
    if _IMPORT_ERROR is not None
    else ""
)

# Hand-derived paddle dtype -> wire number, independent of the production
# paddle_2_number() table so a corrupted table cannot silently agree.
_FLOAT32_NUM = 1
_FLOAT16_NUM = 0


class _FakeProcessGroup:
    """Argument-capturing stand-in for the real backend process group.

    It is NOT the code under test; it lets the tests observe exactly which
    calc-stream primitive the production dispatch selected and with which
    translated peer rank / partial-shard arguments.
    """

    def __init__(self):
        self.calls = []

    def send_partial_on_calc_stream(self, *args):
        self.calls.append(("send_partial", args))
        return "SEND_PARTIAL"

    def send_on_calc_stream(self, *args):
        self.calls.append(("send_full", args))
        return "SEND_FULL"

    def recv_partial_on_calc_stream(self, *args):
        self.calls.append(("recv_partial", args))
        return "RECV_PARTIAL"

    def recv_on_calc_stream(self, *args):
        self.calls.append(("recv_full", args))
        return "RECV_FULL"

    def all_gather_partial_on_calc_stream(self, *args):
        self.calls.append(("allgather_calc", args))
        return "ALLGATHER_CALC"

    def all_gather_partial(self, *args):
        self.calls.append(("allgather", args))
        return "ALLGATHER"


class _FakeGroup:
    """Local process-group fake. ``get_group_rank`` uses a deterministic, easy
    to distinguish mapping (global rank -> rank+100) so a swapped/dropped peer
    translation is visible."""

    def __init__(self, member=True, group_id=7):
        self.process_group = _FakeProcessGroup()
        self._member = member
        self.id = group_id

    def get_group_rank(self, global_rank):
        return global_rank + 100

    def is_member(self):
        return self._member


class _P2pStateMixin:
    """Save/restore the module-level p2p globals so tests never leak state."""

    def _set_enable_partial(self, value):
        original = p2p_mod._enable_partial_send_recv
        self.addCleanup(setattr, p2p_mod, "_enable_partial_send_recv", original)
        p2p_mod._enable_partial_send_recv = value


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestIsValidSendRecvPartial(_P2pStateMixin, unittest.TestCase):
    """`_is_valid_send_recv_partial`: enable flag + degree>1 + divisibility."""

    def test_disabled_flag_overrides_divisible_input(self):
        self._set_enable_partial(False)
        tensor = paddle.zeros([2, 3], dtype="float32")  # numel 6, divisible
        # Disabled must win even though 6 % 2 == 0 and degree > 1.
        self.assertFalse(_is_valid_send_recv_partial(tensor, 2))

    def test_enabled_requires_degree_gt_one_and_divisible(self):
        self._set_enable_partial(True)
        tensor = paddle.zeros([2, 3], dtype="float32")  # numel 6
        # degree == 1 -> False (no partial split with a single mp rank),
        # even though 6 % 1 == 0.
        self.assertFalse(_is_valid_send_recv_partial(tensor, 1))
        # degree > 1 AND divisible -> True.
        self.assertTrue(_is_valid_send_recv_partial(tensor, 2))
        self.assertTrue(_is_valid_send_recv_partial(tensor, 3))
        self.assertTrue(_is_valid_send_recv_partial(tensor, 6))
        # degree > 1 but NOT divisible -> False (6 % 4 == 2, 6 % 5 == 1).
        self.assertFalse(_is_valid_send_recv_partial(tensor, 4))
        self.assertFalse(_is_valid_send_recv_partial(tensor, 5))

    def test_zero_element_tensor_raises(self):
        self._set_enable_partial(True)
        empty = paddle.zeros([0, 3], dtype="float32")  # numel 0
        with self.assertRaises(AssertionError):
            _is_valid_send_recv_partial(empty, 2)


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestObtainSendMessage(unittest.TestCase):
    """`_obtain_send_message` / `set_send_message`: shape, dtype-number, key
    extraction and the tuple stop_gradient skipping contract."""

    def test_single_tensor_with_key(self):
        meta = SendRecvMeta()
        tensor = paddle.zeros([2, 3], dtype="float32")
        tensor.key = "a"
        shape, dtype_num, key = meta._obtain_send_message(tensor)
        self.assertEqual(list(shape), [2, 3])
        self.assertEqual(dtype_num, _FLOAT32_NUM)
        self.assertEqual(key, "a")

    def test_single_tensor_without_key_is_none(self):
        meta = SendRecvMeta()
        tensor = paddle.zeros([4], dtype="float32")
        shape, dtype_num, key = meta._obtain_send_message(tensor)
        self.assertEqual(list(shape), [4])
        self.assertEqual(dtype_num, _FLOAT32_NUM)
        self.assertIsNone(key)

    def test_tuple_skips_stop_gradient_tensors(self):
        meta = SendRecvMeta()
        t1 = paddle.zeros([2, 3], dtype="float32")
        t1.stop_gradient = False
        t1.key = "k1"
        t2 = paddle.zeros([4], dtype="float32")  # will be skipped
        t2.stop_gradient = True
        t2.key = "k2"
        t3 = paddle.zeros([5, 6], dtype="float16")
        t3.stop_gradient = False
        t3.key = "k3"

        shapes, dtypes, keys = meta._obtain_send_message((t1, t2, t3))

        # t2 (stop_gradient=True) must be dropped entirely: only 2 entries.
        self.assertEqual(len(shapes), 2)
        self.assertEqual([list(s) for s in shapes], [[2, 3], [5, 6]])
        self.assertEqual(dtypes, (_FLOAT32_NUM, _FLOAT16_NUM))
        self.assertEqual(keys, ("k1", "k3"))

    def test_set_send_message_stores_obtained_values(self):
        meta = SendRecvMeta()
        tensor = paddle.zeros([2, 3], dtype="float32")
        tensor.key = "a"
        meta.set_send_message(tensor)
        self.assertEqual(list(meta.send_shape_message), [2, 3])
        self.assertEqual(meta.send_dtype_message, _FLOAT32_NUM)
        self.assertEqual(meta.send_key_message, "a")


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestCheckSendMessage(unittest.TestCase):
    """`check_send_message`: no-op when unset; asserts on shape/dtype/key drift."""

    def _meta_with(self, shape, dtype, key=None):
        meta = SendRecvMeta()
        tensor = paddle.zeros(shape, dtype=dtype)
        if key is not None:
            tensor.key = key
        meta.set_send_message(tensor)
        return meta

    def test_unset_is_noop(self):
        meta = SendRecvMeta()  # send_shape_message is None
        tensor = paddle.zeros([2, 3], dtype="float32")
        # Must return without raising and without recording anything.
        self.assertIsNone(meta.check_send_message(tensor))
        self.assertIsNone(meta.send_shape_message)

    def test_matching_tensor_passes(self):
        meta = self._meta_with([2, 3], "float32", key="k")
        same = paddle.zeros([2, 3], dtype="float32")
        same.key = "k"
        self.assertIsNone(meta.check_send_message(same))

    def test_shape_mismatch_raises(self):
        meta = self._meta_with([2, 3], "float32")
        wrong = paddle.zeros([3, 4], dtype="float32")
        with self.assertRaises(AssertionError):
            meta.check_send_message(wrong)

    def test_dtype_mismatch_raises(self):
        meta = self._meta_with([2, 3], "float32")
        wrong = paddle.zeros([2, 3], dtype="int64")
        with self.assertRaises(AssertionError):
            meta.check_send_message(wrong)

    def test_key_mismatch_raises(self):
        meta = self._meta_with([2, 3], "float32", key="k1")
        wrong = paddle.zeros([2, 3], dtype="float32")
        wrong.key = "k2"
        with self.assertRaises(AssertionError):
            meta.check_send_message(wrong)


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestInitOrEraseMeta(unittest.TestCase):
    """`init_or_erase_meta`: constructor initializes and re-erase resets."""

    def test_fresh_meta_is_empty(self):
        meta = SendRecvMeta()
        for attr in (
            "send_shape_message",
            "send_dtype_message",
            "send_key_message",
            "recv_shape_message",
            "recv_dtype_message",
            "recv_stop_gradient",
            "recv_key_message",
        ):
            self.assertIsNone(getattr(meta, attr))
        self.assertFalse(meta.has_send_meta)
        self.assertFalse(meta.has_recv_meta)

    def test_erase_resets_mutated_state(self):
        meta = SendRecvMeta()
        meta.send_shape_message = [1, 2]
        meta.send_dtype_message = _FLOAT32_NUM
        meta.recv_key_message = ("k",)
        meta.has_send_meta = True
        meta.has_recv_meta = True

        meta.init_or_erase_meta()

        self.assertIsNone(meta.send_shape_message)
        self.assertIsNone(meta.send_dtype_message)
        self.assertIsNone(meta.recv_key_message)
        self.assertFalse(meta.has_send_meta)
        self.assertFalse(meta.has_recv_meta)


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestInitializeP2pGroups(_P2pStateMixin, unittest.TestCase):
    """`initialize_p2p_groups`: config reaches the real consumer."""

    def setUp(self):
        # Preserve every global this entry point mutates.
        self._orig = (
            p2p_mod._hcg,
            p2p_mod._enable_partial_send_recv,
            p2p_mod._timers,
        )

        def _restore():
            (
                p2p_mod._hcg,
                p2p_mod._enable_partial_send_recv,
                p2p_mod._timers,
            ) = self._orig

        self.addCleanup(_restore)

    def test_enable_partial_flag_propagates_to_consumer(self):
        sentinel_hcg = object()
        tensor = paddle.zeros([2, 3], dtype="float32")  # numel 6, degree 2 ok

        initialize_p2p_groups(
            sentinel_hcg, enable_partial_send_recv=False, enable_timer=False
        )
        self.assertIs(p2p_mod._hcg, sentinel_hcg)
        self.assertFalse(p2p_mod._enable_partial_send_recv)
        # Consumer observes the disabled flag.
        self.assertFalse(_is_valid_send_recv_partial(tensor, 2))

        initialize_p2p_groups(
            sentinel_hcg, enable_partial_send_recv=True, enable_timer=False
        )
        self.assertTrue(p2p_mod._enable_partial_send_recv)
        # Same input now routes to partial send/recv.
        self.assertTrue(_is_valid_send_recv_partial(tensor, 2))

    def test_enable_timer_false_leaves_timers_untouched(self):
        p2p_mod._timers = "SENTINEL_TIMERS"
        initialize_p2p_groups(object(), enable_timer=False)
        self.assertEqual(p2p_mod._timers, "SENTINEL_TIMERS")

    def test_enable_timer_stores_obtained_timers(self):
        marker = object()
        with mock.patch.object(
            p2p_mod.timer, "get_timers", return_value=marker
        ) as get_timers:
            initialize_p2p_groups(object(), enable_timer=True)
        # Observe the return value is consumed (stored), not merely called.
        get_timers.assert_called_once()
        self.assertIs(p2p_mod._timers, marker)


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestSendRecvOnCalcStream(_P2pStateMixin, unittest.TestCase):
    """`_send_on_calc_stream` / `_recv_on_calc_stream`: group assertion and the
    real partial/full dispatch with peer-rank translation. The REAL
    `_is_valid_send_recv_partial` selects the branch; only the backend process
    group is faked. This proves local dispatch + argument wiring, NOT real
    cross-rank communication."""

    def test_send_group_none_raises(self):
        tensor = paddle.zeros([2, 3], dtype="float32")
        with self.assertRaises(AssertionError):
            _send_on_calc_stream(tensor, None, dst=3, nranks=1, rank_id=0)

    def test_recv_group_none_raises(self):
        tensor = paddle.zeros([2, 3], dtype="float32")
        with self.assertRaises(AssertionError):
            _recv_on_calc_stream(tensor, None, src=2, nranks=1, rank_id=0)

    def test_send_partial_branch_wires_translated_peer(self):
        self._set_enable_partial(True)
        group = _FakeGroup()
        tensor = paddle.zeros([2, 3], dtype="float32")  # numel 6, degree 2 ok
        ret = _send_on_calc_stream(tensor, group, dst=3, nranks=2, rank_id=1)
        self.assertEqual(ret, "SEND_PARTIAL")
        # dst 3 -> group rank 103; nranks/rank_id forwarded verbatim.
        self.assertEqual(
            group.process_group.calls,
            [("send_partial", (tensor, 103, 2, 1))],
        )

    def test_send_full_branch_when_not_partition(self):
        self._set_enable_partial(True)
        group = _FakeGroup()
        tensor = paddle.zeros([2, 3], dtype="float32")
        # nranks == 1 -> not partial -> full send with only translated peer.
        ret = _send_on_calc_stream(tensor, group, dst=3, nranks=1, rank_id=0)
        self.assertEqual(ret, "SEND_FULL")
        self.assertEqual(
            group.process_group.calls,
            [("send_full", (tensor, 103))],
        )

    def test_recv_partial_branch_wires_translated_peer(self):
        self._set_enable_partial(True)
        group = _FakeGroup()
        tensor = paddle.zeros([2, 3], dtype="float32")
        ret = _recv_on_calc_stream(tensor, group, src=2, nranks=2, rank_id=1)
        self.assertEqual(ret, "RECV_PARTIAL")
        # src 2 -> group rank 102.
        self.assertEqual(
            group.process_group.calls,
            [("recv_partial", (tensor, 102, 2, 1))],
        )

    def test_recv_full_branch_when_not_partition(self):
        self._set_enable_partial(True)
        group = _FakeGroup()
        tensor = paddle.zeros([2, 3], dtype="float32")
        ret = _recv_on_calc_stream(tensor, group, src=2, nranks=1, rank_id=0)
        self.assertEqual(ret, "RECV_FULL")
        self.assertEqual(
            group.process_group.calls,
            [("recv_full", (tensor, 102))],
        )


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestAllgatherPartial(_P2pStateMixin, unittest.TestCase):
    """`allgather_partial`: short-circuit, membership gate and dispatch. Only
    the backend process group is faked; not a real cross-rank all-gather."""

    def test_short_circuit_returns_same_tensor(self):
        self._set_enable_partial(True)
        group = _FakeGroup()
        tensor = paddle.zeros([2, 3], dtype="float32")
        # nranks == 1 -> not a valid partition -> tensor returned unchanged.
        result = allgather_partial(
            tensor, nranks=1, rank_id=0, group=group, use_calc_stream=True
        )
        self.assertIs(result, tensor)
        self.assertEqual(group.process_group.calls, [])

    def test_non_member_returns_none(self):
        self._set_enable_partial(True)
        group = _FakeGroup(member=False)
        tensor = paddle.zeros([2, 3], dtype="float32")  # numel 6, degree 2 ok
        result = allgather_partial(
            tensor, nranks=2, rank_id=1, group=group, use_calc_stream=True
        )
        self.assertIsNone(result)
        self.assertEqual(group.process_group.calls, [])

    def test_calc_stream_partial_allgather_dispatch(self):
        self._set_enable_partial(True)
        group = _FakeGroup()
        tensor = paddle.zeros([2, 3], dtype="float32")
        result = allgather_partial(
            tensor, nranks=2, rank_id=1, group=group, use_calc_stream=True
        )
        self.assertEqual(result, "ALLGATHER_CALC")
        self.assertEqual(
            group.process_group.calls,
            [("allgather_calc", (tensor, tensor, 2, 1))],
        )

    def test_non_calc_stream_partial_allgather_dispatch(self):
        self._set_enable_partial(True)
        group = _FakeGroup()
        tensor = paddle.zeros([2, 3], dtype="float32")
        result = allgather_partial(
            tensor, nranks=2, rank_id=1, group=group, use_calc_stream=False
        )
        self.assertEqual(result, "ALLGATHER")
        self.assertEqual(
            group.process_group.calls,
            [("allgather", (tensor, tensor, 2, 1))],
        )


if __name__ == "__main__":
    unittest.main()

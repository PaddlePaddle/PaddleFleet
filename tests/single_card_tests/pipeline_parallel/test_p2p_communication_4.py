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

"""CPU-observable behavior tests for the pipeline-parallel P2P communication
helpers in ``paddlefleet.pipeline_parallel.pp_utils.p2p_communication`` and its
dtype-code collaborator in ``...pp_utils.utils``.

The real cross-rank collectives in this module (``batch_send_recv_on_calc_stream``,
``_p2p_helper``, ``P2pHelper.send_*/recv_*``) drive NCCL send/recv through a live
pipeline process group; their correctness (peer selection, partial-send slicing,
tensor reconstruction) can only be proven with a real multi-rank job and is out of
scope here -- faking ``world_size`` and mocking the collectives would only re-assert
the test's own scaffolding (see antipattern 13). Instead this file pins the pure,
process-group-free logic that those collectives depend on, using hand-derived
expected values:

  * ``paddle_2_number`` / ``number_2_dtype`` -- the dtype <-> wire-code table used
    to serialize and rebuild received tensors, checked against an independently
    written mapping plus the invalid-input assertions.
  * ``_is_valid_send_recv_partial`` -- the predicate that decides partial send/recv,
    incl. the zero-element guard and consumption of the module ``_enable_partial_send_recv``
    flag set by ``initialize_p2p_groups``.
  * ``P2PonCalcStream`` -- op-type validation and attribute storage.
  * ``_batch_p2p_tuple_or_tensor`` -- tensor-to-op wrapping preserving order,
    identity and the mp_degree / mp_rank / peer / group arguments.
  * ``SendRecvMeta`` -- send-message extraction (shape/dtype/key), the
    stop-gradient filtering that applies only to the tuple branch, and the
    ``check_send_message`` mismatch guard.

Import of paddle / paddlefleet is guarded so that on a machine without those
packages the suite honestly skips (with the captured error) rather than passing
vacuously.
"""

import unittest

try:
    import paddle

    from paddlefleet.pipeline_parallel.pp_utils import (
        p2p_communication as p2p,
    )
    from paddlefleet.pipeline_parallel.pp_utils.utils import (
        number_2_dtype,
        paddle_2_number,
    )

    _IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:  # paddle/paddlefleet absent
    _IMPORT_ERROR = exc

_HAS_DEPS = _IMPORT_ERROR is None

# Independently written dtype <-> wire-code mapping. Derived from dtype
# semantics here, NOT read back from the production tables, so a swapped or
# dropped entry in either production dict is observable.
_EXPECTED_DTYPE_CODES = [
    ("float16", 0),
    ("float32", 1),
    ("float64", 2),
    ("int32", 3),
    ("int64", 4),
    ("bfloat16", 5),
    ("bool", 6),
]


def _paddle_dtype(name):
    """Resolve a dtype name to the paddle dtype object."""
    return getattr(paddle, name)


@unittest.skipUnless(
    _HAS_DEPS,
    f"paddle/paddlefleet not importable in this environment: {_IMPORT_ERROR!r}",
)
class TestDtypeWireCodes(unittest.TestCase):
    """paddle_2_number / number_2_dtype form the on-wire dtype table."""

    def test_paddle_2_number_matches_independent_table(self):
        for name, code in _EXPECTED_DTYPE_CODES:
            self.assertEqual(
                paddle_2_number(_paddle_dtype(name)),
                code,
                msg=f"dtype {name} should encode to {code}",
            )

    def test_number_2_dtype_matches_independent_table(self):
        for name, code in _EXPECTED_DTYPE_CODES:
            self.assertEqual(
                number_2_dtype(code),
                name,
                msg=f"code {code} should decode to {name}",
            )

    def test_roundtrip_dtype_to_code_to_name(self):
        # Encoding a dtype then decoding must return that dtype's own name;
        # this catches a table where a code is reused for two dtypes.
        for name, _ in _EXPECTED_DTYPE_CODES:
            code = paddle_2_number(_paddle_dtype(name))
            self.assertEqual(number_2_dtype(code), name)

    def test_codes_are_unique(self):
        codes = [
            paddle_2_number(_paddle_dtype(n)) for n, _ in _EXPECTED_DTYPE_CODES
        ]
        self.assertEqual(len(set(codes)), len(codes))

    def test_paddle_2_number_rejects_unmapped_dtype(self):
        # complex64 is deliberately absent from the table.
        with self.assertRaises(AssertionError):
            paddle_2_number(paddle.complex64)

    def test_number_2_dtype_rejects_unknown_code(self):
        with self.assertRaises(AssertionError):
            number_2_dtype(99)


@unittest.skipUnless(
    _HAS_DEPS,
    f"paddle/paddlefleet not importable in this environment: {_IMPORT_ERROR!r}",
)
class TestIsValidSendRecvPartial(unittest.TestCase):
    """_is_valid_send_recv_partial: mp_degree>1 and numel divisible by it,
    subject to the module-level _enable_partial_send_recv flag."""

    def setUp(self):
        # This predicate reads the module global; ensure the default (True)
        # is in effect for the divisibility cases and restore whatever the
        # module currently holds afterwards.
        self._orig_flag = p2p._enable_partial_send_recv
        self.addCleanup(
            setattr, p2p, "_enable_partial_send_recv", self._orig_flag
        )
        p2p._enable_partial_send_recv = True

    def _tensor(self, shape):
        # numel = product(shape); content is irrelevant to the predicate.
        return paddle.ones(shape, dtype="float32")

    def test_degree_one_is_never_partial(self):
        # numel=6, but mp_degree==1 -> False (mp_degree>1 is False).
        self.assertFalse(
            p2p._is_valid_send_recv_partial(self._tensor([2, 3]), 1)
        )

    def test_divisible_degrees_are_partial(self):
        # numel=6 is divisible by 2 and 3.
        self.assertTrue(
            p2p._is_valid_send_recv_partial(self._tensor([2, 3]), 2)
        )
        self.assertTrue(
            p2p._is_valid_send_recv_partial(self._tensor([2, 3]), 3)
        )

    def test_non_divisible_degrees_are_not_partial(self):
        # numel=6 is NOT divisible by 4 or 5.
        self.assertFalse(
            p2p._is_valid_send_recv_partial(self._tensor([2, 3]), 4)
        )
        self.assertFalse(
            p2p._is_valid_send_recv_partial(self._tensor([2, 3]), 5)
        )

    def test_zero_element_tensor_is_rejected(self):
        with self.assertRaises(AssertionError):
            p2p._is_valid_send_recv_partial(self._tensor([0]), 2)

    def test_flag_disabled_forces_false(self):
        # With the flag off, even a perfectly divisible case is not partial;
        # proves the predicate actually consumes _enable_partial_send_recv.
        p2p._enable_partial_send_recv = False
        self.assertFalse(
            p2p._is_valid_send_recv_partial(self._tensor([2, 3]), 2)
        )


@unittest.skipUnless(
    _HAS_DEPS,
    f"paddle/paddlefleet not importable in this environment: {_IMPORT_ERROR!r}",
)
class TestInitializeP2pGroups(unittest.TestCase):
    """initialize_p2p_groups wires module globals consumed downstream."""

    def setUp(self):
        # Snapshot every global the function mutates and restore on teardown
        # so no state leaks into sibling tests in the same process.
        self._orig = {
            "_hcg": p2p._hcg,
            "_enable_partial_send_recv": p2p._enable_partial_send_recv,
            "_timers": p2p._timers,
        }

        def _restore():
            p2p._hcg = self._orig["_hcg"]
            p2p._enable_partial_send_recv = self._orig[
                "_enable_partial_send_recv"
            ]
            p2p._timers = self._orig["_timers"]

        self.addCleanup(_restore)

    def test_sets_globals_and_flag_is_consumed(self):
        sentinel_hcg = object()
        p2p.initialize_p2p_groups(
            sentinel_hcg,
            enable_partial_send_recv=False,
            enable_timer=False,
        )
        self.assertIs(p2p._hcg, sentinel_hcg)
        self.assertFalse(p2p._enable_partial_send_recv)
        # enable_timer=False must leave the timer global untouched (None).
        self.assertIsNone(p2p._timers)
        # The flag just set is actually read by the partial predicate.
        divisible = paddle.ones([2, 3], dtype="float32")
        self.assertFalse(p2p._is_valid_send_recv_partial(divisible, 2))


@unittest.skipUnless(
    _HAS_DEPS,
    f"paddle/paddlefleet not importable in this environment: {_IMPORT_ERROR!r}",
)
class TestP2PonCalcStream(unittest.TestCase):
    """P2PonCalcStream validates op type and stores its arguments verbatim."""

    def test_send_op_stores_all_attributes(self):
        tensor = paddle.ones([4], dtype="float32")
        group = object()
        op = p2p.P2PonCalcStream(
            p2p._send_on_calc_stream,
            tensor,
            peer=3,
            group=group,
            nranks=2,
            rank_id=1,
        )
        self.assertIs(op.op, p2p._send_on_calc_stream)
        self.assertIs(op.tensor, tensor)
        self.assertEqual(op.peer, 3)
        self.assertIs(op.group, group)
        self.assertEqual(op.nranks, 2)
        self.assertEqual(op.rank_id, 1)

    def test_recv_op_defaults(self):
        tensor = paddle.ones([4], dtype="float32")
        op = p2p.P2PonCalcStream(
            p2p._recv_on_calc_stream, tensor, peer=0, group=object()
        )
        self.assertIs(op.op, p2p._recv_on_calc_stream)
        self.assertEqual(op.nranks, 1)
        self.assertEqual(op.rank_id, 0)

    def test_invalid_op_raises_runtime_error(self):
        tensor = paddle.ones([4], dtype="float32")
        with self.assertRaises(RuntimeError):
            p2p.P2PonCalcStream(
                paddle.distributed.isend, tensor, peer=0, group=object()
            )


@unittest.skipUnless(
    _HAS_DEPS,
    f"paddle/paddlefleet not importable in this environment: {_IMPORT_ERROR!r}",
)
class TestBatchP2pTupleOrTensor(unittest.TestCase):
    """_batch_p2p_tuple_or_tensor wraps tensors into ordered ops carrying the
    peer / group / mp_degree / mp_rank arguments."""

    def test_single_tensor_yields_one_op(self):
        tensor = paddle.ones([6], dtype="float32")
        group = object()
        ops = p2p._batch_p2p_tuple_or_tensor(
            tensor,
            p2p._send_on_calc_stream,
            5,
            group,
            mp_degree=2,
            mp_rank=1,
        )
        self.assertEqual(len(ops), 1)
        self.assertIsInstance(ops[0], p2p.P2PonCalcStream)
        self.assertIs(ops[0].tensor, tensor)
        self.assertIs(ops[0].op, p2p._send_on_calc_stream)
        self.assertEqual(ops[0].peer, 5)
        self.assertIs(ops[0].group, group)
        self.assertEqual(ops[0].nranks, 2)
        self.assertEqual(ops[0].rank_id, 1)

    def test_tuple_preserves_order_and_identity(self):
        # Distinct objects so a reorder or mis-wire is observable.
        t0 = paddle.ones([2], dtype="float32")
        t1 = paddle.ones([3], dtype="float32")
        t2 = paddle.ones([4], dtype="float32")
        group = object()
        ops = p2p._batch_p2p_tuple_or_tensor(
            (t0, t1, t2),
            p2p._recv_on_calc_stream,
            7,
            group,
            mp_degree=4,
            mp_rank=2,
        )
        self.assertEqual(len(ops), 3)
        for op, expected_tensor in zip(ops, (t0, t1, t2)):
            self.assertIs(op.tensor, expected_tensor)
            self.assertIs(op.op, p2p._recv_on_calc_stream)
            self.assertEqual(op.peer, 7)
            self.assertIs(op.group, group)
            self.assertEqual(op.nranks, 4)
            self.assertEqual(op.rank_id, 2)


@unittest.skipUnless(
    _HAS_DEPS,
    f"paddle/paddlefleet not importable in this environment: {_IMPORT_ERROR!r}",
)
class TestSendRecvMeta(unittest.TestCase):
    """SendRecvMeta extracts send messages and guards against shape/dtype
    drift; the stop-gradient filter applies only to the tuple branch."""

    def test_fresh_meta_is_erased(self):
        meta = p2p.SendRecvMeta()
        self.assertIsNone(meta.send_shape_message)
        self.assertIsNone(meta.send_dtype_message)
        self.assertIsNone(meta.send_key_message)
        self.assertIsNone(meta.recv_shape_message)
        self.assertIsNone(meta.recv_dtype_message)
        self.assertIsNone(meta.recv_stop_gradient)
        self.assertIsNone(meta.recv_key_message)
        self.assertFalse(meta.has_send_meta)
        self.assertFalse(meta.has_recv_meta)

    def test_single_tensor_send_message(self):
        meta = p2p.SendRecvMeta()
        tensor = paddle.ones([2, 3], dtype="float32")
        meta.set_send_message(tensor)
        self.assertEqual(list(meta.send_shape_message), [2, 3])
        # float32 encodes to 1 in the independent table.
        self.assertEqual(meta.send_dtype_message, 1)
        self.assertIsNone(meta.send_key_message)

    def test_single_tensor_keeps_stop_gradient_tensor(self):
        # The single-tensor branch does NOT drop stop_gradient tensors.
        meta = p2p.SendRecvMeta()
        tensor = paddle.ones([5], dtype="int64")
        tensor.stop_gradient = True
        meta.set_send_message(tensor)
        self.assertEqual(list(meta.send_shape_message), [5])
        self.assertEqual(meta.send_dtype_message, 4)  # int64 -> 4

    def test_key_attribute_is_captured(self):
        meta = p2p.SendRecvMeta()
        tensor = paddle.ones([2], dtype="float32")
        tensor.key = "hidden_states"
        meta.set_send_message(tensor)
        self.assertEqual(meta.send_key_message, "hidden_states")

    def test_tuple_filters_stop_gradient_and_keeps_order(self):
        # t1 is stop_gradient=True and must be dropped; t0, t2 remain in order.
        t0 = paddle.ones([2], dtype="float32")
        t0.stop_gradient = False
        t1 = paddle.ones([3, 4], dtype="float32")
        t1.stop_gradient = True
        t2 = paddle.ones([5], dtype="float32")
        t2.stop_gradient = False
        meta = p2p.SendRecvMeta()
        meta.set_send_message((t0, t1, t2))
        self.assertEqual([list(s) for s in meta.send_shape_message], [[2], [5]])
        self.assertEqual(meta.send_dtype_message, (1, 1))
        self.assertEqual(meta.send_key_message, (None, None))

    def test_check_send_message_noop_before_set(self):
        # With no stored shape/dtype the guard is a no-op (must not raise).
        meta = p2p.SendRecvMeta()
        meta.check_send_message(paddle.ones([9, 9], dtype="float32"))

    def test_check_send_message_accepts_matching(self):
        meta = p2p.SendRecvMeta()
        meta.set_send_message(paddle.ones([2, 3], dtype="float32"))
        # Same shape and dtype -> passes silently.
        meta.check_send_message(paddle.zeros([2, 3], dtype="float32"))

    def test_check_send_message_rejects_shape_mismatch(self):
        meta = p2p.SendRecvMeta()
        meta.set_send_message(paddle.ones([2, 3], dtype="float32"))
        with self.assertRaises(AssertionError):
            meta.check_send_message(paddle.ones([4, 5], dtype="float32"))

    def test_check_send_message_rejects_dtype_mismatch(self):
        meta = p2p.SendRecvMeta()
        meta.set_send_message(paddle.ones([2, 3], dtype="float32"))
        with self.assertRaises(AssertionError):
            meta.check_send_message(paddle.ones([2, 3], dtype="int64"))

    def test_init_or_erase_resets_populated_meta(self):
        meta = p2p.SendRecvMeta()
        meta.set_send_message(paddle.ones([2, 3], dtype="float32"))
        meta.has_send_meta = True
        meta.init_or_erase_meta()
        self.assertIsNone(meta.send_shape_message)
        self.assertIsNone(meta.send_dtype_message)
        self.assertFalse(meta.has_send_meta)


if __name__ == "__main__":
    unittest.main()

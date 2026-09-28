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

"""CPU-only behavior tests for device-independent helpers of
``paddlefleet.pipeline_parallel.pp_utils.four_directions_p2p_communication``
and its dtype-protocol collaborators in ``pp_utils.utils``.

Covered here (all pure, non-communicating, CPU-executable):

* ``paddle_2_number`` / ``number_2_dtype`` -- the on-wire dtype code protocol
  used by ``SendRecvMeta`` to serialize/deserialize tensor metadata. Expected
  codes and names are hand-written from the protocol table, NOT produced by
  calling the production helpers, so test and code are not the same source.
* ``allgather_partial`` -- the identity fall-through when the tensor is not
  partition-eligible (``mp_degree == 1``): it must return the *same* tensor
  object without touching any process group.
* ``_xpu_comm_group_start`` / ``_xpu_comm_group_end`` -- the no-op guard on a
  build that is not compiled with XPU.
* ``SendRecvMeta.set_send_message`` edge cases: an all-``stop_gradient`` tuple
  collapses to empty metadata, and an unrecognized input type leaves the
  metadata untouched.

Deliberately NOT covered: the real ``_p2p_helper`` scheduling, meta exchange,
and the partition-eligible ``allgather_partial`` path. Those require a real
pipeline process group across multiple ranks; faking ``world_size`` and mocking
the collectives would only exercise local orchestration and could not reject a
wrong peer, direction, split size, or missing reduction. They belong to a real
multi-card run, not to this CPU-only file.
"""

import unittest

try:
    import paddle

    from paddlefleet.pipeline_parallel.pp_utils import (
        four_directions_p2p_communication as p2p,
        utils as pp_utils,
    )

    _IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:  # honest dependency probe
    paddle = None
    p2p = None
    pp_utils = None
    _IMPORT_ERROR = exc

_HAS_DEPS = _IMPORT_ERROR is None
_SKIP_REASON = (
    f"paddle / paddlefleet not importable on this CPU host: {_IMPORT_ERROR!r}"
)

# Independent transcription of the on-wire dtype protocol as
# (paddle dtype, integer code, dtype name). Written by hand from the protocol
# definition; production ``paddle_2_number`` / ``number_2_dtype`` are the code
# under test, so they are never used to build this table.
_PROTOCOL = None
if _HAS_DEPS:
    _PROTOCOL = [
        (paddle.float16, 0, "float16"),
        (paddle.float32, 1, "float32"),
        (paddle.float64, 2, "float64"),
        (paddle.int32, 3, "int32"),
        (paddle.int64, 4, "int64"),
        (paddle.bfloat16, 5, "bfloat16"),
        (paddle.bool, 6, "bool"),
    ]


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestDtypeProtocolCollaborators(unittest.TestCase):
    """`paddle_2_number` / `number_2_dtype` implement the wire dtype codes."""

    def test_paddle_2_number_exact_codes(self):
        for dtype, code, _name in _PROTOCOL:
            self.assertEqual(
                pp_utils.paddle_2_number(dtype),
                code,
                msg=f"{dtype} should encode to {code}",
            )

    def test_number_2_dtype_exact_names(self):
        for _dtype, code, name in _PROTOCOL:
            self.assertEqual(
                pp_utils.number_2_dtype(code),
                name,
                msg=f"code {code} should decode to {name}",
            )

    def test_round_trip_is_consistent(self):
        # Encoding then decoding must return the canonical dtype name, proving
        # the two tables are mutual inverses on every supported dtype.
        for dtype, _code, name in _PROTOCOL:
            self.assertEqual(
                pp_utils.number_2_dtype(pp_utils.paddle_2_number(dtype)),
                name,
            )

    def test_codes_are_unique(self):
        codes = [pp_utils.paddle_2_number(d) for d, _c, _n in _PROTOCOL]
        self.assertEqual(sorted(codes), [0, 1, 2, 3, 4, 5, 6])

    def test_unsupported_dtype_object_asserts(self):
        # A string is not one of the paddle dtype keys; the explicit assert in
        # ``paddle_2_number`` must reject it rather than silently return junk.
        with self.assertRaises(AssertionError):
            pp_utils.paddle_2_number("float32")

    def test_unsupported_code_asserts(self):
        # 99 is outside the protocol table; decoding it must assert.
        with self.assertRaises(AssertionError):
            pp_utils.number_2_dtype(99)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestAllgatherPartialIdentityFallthrough(unittest.TestCase):
    """`allgather_partial` returns the tensor unchanged when not splittable.

    With ``nranks == 1`` the partition-eligibility gate is False, so the
    function must short-circuit to ``return tensor`` and perform no gather.
    The partition-eligible path (nranks > 1) needs a real process group and is
    intentionally not exercised here.
    """

    def setUp(self):
        paddle.set_device("cpu")

    def test_returns_same_object_and_preserves_content(self):
        t = paddle.arange(6, dtype="float32").reshape([2, 3])
        before = t.tolist()
        out = p2p.allgather_partial(t, nranks=1, rank_id=0, group=None)
        # Identity: no new tensor is allocated and the group is never touched.
        self.assertIs(out, t)
        # Content is untouched by the no-op path.
        self.assertEqual(out.tolist(), before)
        self.assertEqual(before, [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestXpuCommGroupGuard(unittest.TestCase):
    """`_xpu_comm_group_start` / `_xpu_comm_group_end` no-op off XPU builds."""

    def setUp(self):
        if paddle.is_compiled_with_xpu():
            self.skipTest(
                "paddle build is compiled with XPU; the start/end helpers are "
                "not no-ops here and require real BKCL process groups."
            )
        # Guard the module global so a regression that flips it cannot leak
        # into other tests in the same process.
        self._orig_started = p2p._xpu_comm_group_started
        self.addCleanup(
            setattr, p2p, "_xpu_comm_group_started", self._orig_started
        )

    def test_start_and_end_are_noops_without_xpu(self):
        # On a non-XPU build both calls must return without starting a BKCL
        # group, leaving the started-flag untouched.
        self.assertFalse(p2p._xpu_comm_group_started)
        self.assertIsNone(p2p._xpu_comm_group_start())
        self.assertFalse(p2p._xpu_comm_group_started)
        self.assertIsNone(p2p._xpu_comm_group_end())
        self.assertFalse(p2p._xpu_comm_group_started)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestSetSendMessageEdgeCases(unittest.TestCase):
    """`SendRecvMeta.set_send_message` filtering and type-guard edges."""

    def setUp(self):
        paddle.set_device("cpu")

    def test_all_stop_gradient_tuple_collapses_to_empty(self):
        # Every element is stop_gradient=True, so the ``not d.stop_gradient``
        # filter drops all of them: both messages are empty tuples, not None
        # and not the full set of shapes.
        meta = p2p.SendRecvMeta()
        a = paddle.zeros([2, 3], dtype="float32")
        a.stop_gradient = True
        b = paddle.zeros([4, 5], dtype="int64")
        b.stop_gradient = True
        meta.set_send_message((a, b))
        self.assertEqual(meta.send_shape_message, ())
        self.assertEqual(meta.send_dtype_message, ())

    def test_unrecognized_type_leaves_metadata_untouched(self):
        # A list is neither a paddle.Tensor nor a tuple, so no branch fires and
        # the fields keep their initial None. A regression that treated list
        # like tuple would populate them and fail this.
        meta = p2p.SendRecvMeta()
        meta.set_send_message([1, 2, 3])
        self.assertIsNone(meta.send_shape_message)
        self.assertIsNone(meta.send_dtype_message)


if __name__ == "__main__":
    unittest.main()

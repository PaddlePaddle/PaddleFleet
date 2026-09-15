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

"""CPU-observable behaviour of trainer/utils/reshard/common.py bucketing.

Scope and oracle
----------------
The reshard "bucketed broadcast" splits a state dict into per-(rank, dtype)
byte-bounded buckets, coalesces buckets into broadcast chunks, packs each bucket
into one contiguous buffer, broadcasts it from its owning rank, and scatters the
slices back to their tensors. Only the *partition logic* is observable in a
single CPU process:

  - _shape_numel / _normalize_np_dtype_str : pure element-count and dtype rules.
  - _build_state_dict_broadcast_buckets    : grouping, byte boundaries, per-item
                                              offsets, ownership, empty split,
                                              oversized-stays-alone.
  - _iter_state_dict_bucket_chunks          : count cap, byte cap, and the
                                              documented "oversized bucket is
                                              emitted whole, never split" rule.
  - set_broadcast_max_chunk_bytes           : the mutable peak-memory knob and
                                              its floor at one bucket size.
  - all_gather_state_dict (nranks < 2)      : the genuinely supported
                                              world-size-1 fast path.

Every expectation here is hand-derived from the algorithm by an independent
re-implementation (dtype item sizes taken as the well-known constants
float32=4B, int64=8B), never by calling the production helper that is under
test. Buckets are compared field-by-field including per-item (begin, end)
offsets, so a swap of ownership, a wrong offset, or a dropped/merged oversized
bucket is rejected -- not just a shape check.

NOT verified here: the cross-rank numerics (pack -> multi-root broadcast ->
unpack) only exist once >=2 real ranks each hold different bytes and exchange
them. A single CPU process cannot exercise that, and faking a 2-rank group while
stubbing the collective would only prove local mapping, not that peers receive
the right bytes. That path is covered by the launch-based test below, which
honestly skips unless a real >=2 rank process group is present.
"""

import unittest
from collections import OrderedDict

try:
    import numpy as np
    import paddle

    from paddlefleet.trainer.utils.reshard import common as reshard_common
    from paddlefleet.trainer.utils.reshard.common import (
        _build_state_dict_broadcast_buckets,
        _iter_state_dict_bucket_chunks,
        _normalize_np_dtype_str,
        _shape_numel,
        all_gather_state_dict,
        set_broadcast_max_chunk_bytes,
    )

    _IMPORT_ERROR = None
except ImportError as exc:  # paddle / paddlefleet not installed in this env
    reshard_common = None
    _IMPORT_ERROR = exc


# Independent oracle: hand-known item sizes, deliberately NOT the production
# _dtype_itemsize helper, so byte expectations do not borrow the code under test.
_ITEMSIZE = {"float32": 4, "int64": 8}


def setUpModule():
    if _IMPORT_ERROR is not None:
        raise unittest.SkipTest(
            f"paddle/paddlefleet unavailable, cannot import reshard.common: "
            f"{_IMPORT_ERROR!r}"
        )


class _SingleRankGroup:
    """Minimal stand-in for the world-size-1 group.

    all_gather_state_dict's fast path reads only ``group.nranks``; nranks == 1
    is the real, supported local path (no collective is started), so exercising
    it with a one-field object is genuine, not a faked multi-rank topology.
    """

    nranks = 1


class TestShapeNumel(unittest.TestCase):
    """_shape_numel: empty shape (scalar) counts as 1; a zero dim makes 0."""

    def test_scalar_shape_is_one_element(self):
        self.assertEqual(_shape_numel([]), 1)

    def test_regular_shapes(self):
        self.assertEqual(_shape_numel([5]), 5)
        self.assertEqual(_shape_numel([2, 3]), 6)
        self.assertEqual(_shape_numel([2, 3, 4]), 24)

    def test_zero_dims_are_empty(self):
        self.assertEqual(_shape_numel([0]), 0)
        self.assertEqual(_shape_numel([2, 0]), 0)
        self.assertEqual(_shape_numel([0, 5]), 0)


class TestNormalizeNpDtypeStr(unittest.TestCase):
    """_normalize_np_dtype_str records the paddle-effective dtype: numpy has no
    native bf16, so a bf16 checkpoint loads as uint16 and must map to bfloat16;
    every other dtype string passes through unchanged."""

    def test_uint16_maps_to_bfloat16(self):
        self.assertEqual(_normalize_np_dtype_str("uint16"), "bfloat16")

    def test_other_dtypes_unchanged(self):
        for dt in ("float32", "float16", "int64", "int32", "bfloat16", "bool"):
            self.assertEqual(_normalize_np_dtype_str(dt), dt)


class TestBuildBroadcastBuckets(unittest.TestCase):
    """_build_state_dict_broadcast_buckets: grouping, byte boundaries, per-item
    offsets, ownership, empty split and the oversized-stays-alone rule.

    meta_list entries are ``(key, (dtype, shape, rank))``. Expected buckets are
    written out literally, derived by hand from the algorithm; item sizes come
    from _ITEMSIZE (float32=4B, int64=8B), never from the production helper.
    """

    def test_byte_boundary_and_contiguous_offsets(self):
        # rank 0, float32 (4B). cap = 100B = 25 elements.
        # a(40B)+b(40B)=80B fits; adding c(40B) would reach 120B>100B so [a,b]
        # flushes first, then c starts a fresh bucket at offset 0.
        meta = [
            ("a", ("float32", [10], 0)),
            ("b", ("float32", [10], 0)),
            ("c", ("float32", [10], 0)),
        ]
        buckets, empty = _build_state_dict_broadcast_buckets(meta, 100)
        self.assertEqual(empty, [])
        self.assertEqual(
            buckets,
            [
                {
                    "rank": 0,
                    "dtype": "float32",
                    "numel": 20,
                    "nbytes": 80,
                    "items": [("a", [10], 0, 10), ("b", [10], 10, 20)],
                },
                {
                    "rank": 0,
                    "dtype": "float32",
                    "numel": 10,
                    "nbytes": 40,
                    "items": [("c", [10], 0, 10)],
                },
            ],
        )

    def test_oversized_tensor_stays_alone(self):
        # A tensor whose own bytes reach the cap gets its own bucket and does not
        # pull neighbours in: small0 flushes before big, big flushes after
        # itself, so small1 cannot merge onto big.
        meta = [
            ("small0", ("float32", [5], 0)),
            ("big", ("float32", [50], 0)),  # 200B >= 100B cap
            ("small1", ("float32", [5], 0)),
        ]
        buckets, empty = _build_state_dict_broadcast_buckets(meta, 100)
        self.assertEqual(empty, [])
        self.assertEqual([b["nbytes"] for b in buckets], [20, 200, 20])
        self.assertEqual(
            [[it[0] for it in b["items"]] for b in buckets],
            [["small0"], ["big"], ["small1"]],
        )
        self.assertEqual(buckets[1]["items"], [("big", [50], 0, 50)])

    def test_grouping_ownership_empty_split_and_scalar(self):
        # Distinct (rank, dtype) never share a bucket; numel==0 tensors are
        # split out verbatim (not bucketed); a scalar (shape []) counts as one
        # element and is packed. cap is huge so no byte splitting happens.
        meta = [
            ("r0f_a", ("float32", [4], 0)),
            ("r0i_b", ("int64", [4], 0)),
            ("r1f_c", ("float32", [4], 1)),
            ("empty0", ("float32", [0], 0)),
            ("empty2d", ("float32", [2, 0], 0)),
            ("scalar", ("float32", [], 0)),
        ]
        buckets, empty = _build_state_dict_broadcast_buckets(meta, 10**9)
        self.assertEqual(
            buckets,
            [
                {
                    "rank": 0,
                    "dtype": "float32",
                    "numel": 5,
                    "nbytes": 5 * _ITEMSIZE["float32"],
                    "items": [("r0f_a", [4], 0, 4), ("scalar", [], 4, 5)],
                },
                {
                    "rank": 0,
                    "dtype": "int64",
                    "numel": 4,
                    "nbytes": 4 * _ITEMSIZE["int64"],
                    "items": [("r0i_b", [4], 0, 4)],
                },
                {
                    "rank": 1,
                    "dtype": "float32",
                    "numel": 4,
                    "nbytes": 4 * _ITEMSIZE["float32"],
                    "items": [("r1f_c", [4], 0, 4)],
                },
            ],
        )
        # Empty tensors carry their full meta and never enter a bucket.
        self.assertEqual(
            empty,
            [
                ("empty0", ("float32", [0], 0)),
                ("empty2d", ("float32", [2, 0], 0)),
            ],
        )
        bucketed_keys = {it[0] for b in buckets for it in b["items"]}
        self.assertNotIn("empty0", bucketed_keys)
        self.assertNotIn("empty2d", bucketed_keys)

    def test_rejects_nonpositive_bucket_size(self):
        with self.assertRaises(AssertionError):
            _build_state_dict_broadcast_buckets([("a", ("float32", [4], 0))], 0)


class TestIterBucketChunks(unittest.TestCase):
    """_iter_state_dict_bucket_chunks: buckets coalesce into chunks under both a
    count cap and a byte cap, but a single bucket larger than the byte cap is
    emitted whole in its own chunk (documented: the cap bounds aggregation of
    many buckets, it never splits one bucket).

    The function reads only ``bucket['nbytes']``; each synthetic bucket carries a
    unique ``tag`` so chunk membership is checked by object identity, and the
    concatenation of all chunks must reproduce the input order exactly.
    """

    @staticmethod
    def _mk(nbytes_seq):
        return [{"nbytes": n, "tag": i} for i, n in enumerate(nbytes_seq)]

    def _tags(self, chunks):
        return [[b["tag"] for b in c] for c in chunks]

    def _assert_exact_partition(self, chunks, buckets):
        flat = [b for c in chunks for b in c]
        self.assertEqual(len(flat), len(buckets))
        for got, original in zip(flat, buckets):
            self.assertIs(got, original)  # same objects, original order

    def test_count_cap(self):
        buckets = self._mk([1, 1, 1, 1, 1])
        chunks = list(
            _iter_state_dict_bucket_chunks(
                buckets, chunk_size=2, max_chunk_bytes=10**9
            )
        )
        self.assertEqual(self._tags(chunks), [[0, 1], [2, 3], [4]])
        self._assert_exact_partition(chunks, buckets)

    def test_byte_cap_aggregation(self):
        buckets = self._mk([100, 100, 100])
        chunks = list(
            _iter_state_dict_bucket_chunks(
                buckets, chunk_size=1000, max_chunk_bytes=250
            )
        )
        # 100+100=200<=250 aggregate; +100=300>250 so the third starts anew.
        self.assertEqual(self._tags(chunks), [[0, 1], [2]])
        self._assert_exact_partition(chunks, buckets)

    def test_oversized_bucket_emitted_whole(self):
        buckets = self._mk([50, 500, 50])  # 500 > 200 cap
        chunks = list(
            _iter_state_dict_bucket_chunks(
                buckets, chunk_size=1000, max_chunk_bytes=200
            )
        )
        # The 500B bucket is neither split nor merged with its neighbours.
        self.assertEqual(self._tags(chunks), [[0], [1], [2]])
        self.assertEqual(chunks[1][0]["nbytes"], 500)
        self._assert_exact_partition(chunks, buckets)

    def test_rejects_nonpositive_caps(self):
        with self.assertRaises(AssertionError):
            list(_iter_state_dict_bucket_chunks(self._mk([1]), 0, 10))
        with self.assertRaises(AssertionError):
            list(_iter_state_dict_bucket_chunks(self._mk([1]), 2, 0))


class TestSetBroadcastMaxChunkBytes(unittest.TestCase):
    """set_broadcast_max_chunk_bytes: a value above one bucket size is kept as
    is; a smaller positive value is floored to one bucket size (a chunk must
    hold at least one whole bucket); a non-positive value resets to the compiled
    default. The module global is saved and restored so the mutation cannot leak
    into other tests."""

    def setUp(self):
        self._orig = reshard_common._broadcast_max_chunk_bytes
        self.addCleanup(
            setattr,
            reshard_common,
            "_broadcast_max_chunk_bytes",
            self._orig,
        )
        self._bucket = reshard_common._STATE_DICT_BROADCAST_BUCKET_SIZE_BYTES
        self._default = reshard_common._STATE_DICT_BROADCAST_MAX_CHUNK_BYTES

    def test_value_above_bucket_kept(self):
        target = self._bucket * 3
        set_broadcast_max_chunk_bytes(target)
        self.assertEqual(reshard_common._broadcast_max_chunk_bytes, target)

    def test_value_below_bucket_floored(self):
        set_broadcast_max_chunk_bytes(self._bucket // 2)
        self.assertEqual(
            reshard_common._broadcast_max_chunk_bytes, self._bucket
        )

    def test_zero_resets_to_default(self):
        set_broadcast_max_chunk_bytes(self._bucket * 5)  # move off default
        set_broadcast_max_chunk_bytes(0)
        self.assertEqual(
            reshard_common._broadcast_max_chunk_bytes, self._default
        )

    def test_negative_resets_to_default(self):
        set_broadcast_max_chunk_bytes(self._bucket * 5)
        set_broadcast_max_chunk_bytes(-123)
        self.assertEqual(
            reshard_common._broadcast_max_chunk_bytes, self._default
        )


class TestSingleRankFastPath(unittest.TestCase):
    """all_gather_state_dict at nranks < 2 is the supported world-size-1 path:
    no collective runs, but it still filters, sorts keys, converts numpy inputs
    to CPU paddle tensors, and returns paddle tensor inputs unchanged (identity).
    This is a genuine local path; it says nothing about cross-rank exchange."""

    def test_filters_sorts_and_converts_numpy(self):
        sd = OrderedDict()
        sd["z_keep"] = np.arange(3, dtype="float32")
        sd["a_drop"] = np.arange(3, dtype="float32") + 100.0
        sd["m_keep"] = np.arange(4, dtype="float32") + 10.0
        out = all_gather_state_dict(
            sd, lambda k: k.endswith("keep"), _SingleRankGroup()
        )
        # dropped key gone, remaining keys sorted.
        self.assertEqual(list(out.keys()), ["m_keep", "z_keep"])
        for k in out:
            self.assertIsInstance(out[k], paddle.Tensor)
            self.assertTrue(out[k].place.is_cpu_place())
        np.testing.assert_array_equal(
            out["z_keep"].numpy(), np.arange(3, dtype="float32")
        )
        np.testing.assert_array_equal(
            out["m_keep"].numpy(), np.arange(4, dtype="float32") + 10.0
        )

    def test_existing_tensor_passed_through_by_identity(self):
        t = paddle.to_tensor(np.arange(4, dtype="float32"))
        out = all_gather_state_dict(
            OrderedDict(w=t), lambda k: True, _SingleRankGroup()
        )
        self.assertIs(out["w"], t)


class TestCrossRankNumericsNotVerified(unittest.TestCase):
    """The pack -> multi-root broadcast -> unpack numerics require >=2 real ranks
    each holding different bytes. This single CPU process cannot exercise them;
    faking a 2-rank group and stubbing the collective would only prove local
    mapping, not that peers receive the right bytes. So we skip honestly unless a
    real process group is present, and only then run a genuine round-trip."""

    def test_cross_rank_round_trip(self):
        world_size = paddle.distributed.get_world_size()
        if world_size < 2:
            self.skipTest(
                "cross-rank bucketed-broadcast numerics NOT verified: needs a "
                "real >=2 rank process group started by a launcher; "
                f"world_size={world_size}"
            )
        rank = paddle.distributed.get_rank()
        group = paddle.distributed.new_group(list(range(world_size)))
        # Each rank owns a distinct, content-identifiable tensor.
        sd = OrderedDict()
        sd[f"w_rank{rank}"] = (
            np.arange(4, dtype="float32") + 100.0 * rank
        ).copy()
        out = all_gather_state_dict(sd, lambda k: True, group)
        # After the real broadcast every rank must hold every rank's tensor with
        # the correct owner's content (not merely the right shape).
        for r in range(world_size):
            key = f"w_rank{r}"
            self.assertIn(key, out)
            expected = np.arange(4, dtype="float32") + 100.0 * r
            np.testing.assert_array_equal(
                out[key].astype("float32").numpy(), expected
            )


if __name__ == "__main__":
    unittest.main()

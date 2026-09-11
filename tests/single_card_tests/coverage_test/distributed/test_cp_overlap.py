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

"""Tests for the overlapped FlashMask context parallel path.

The overlap itself lives in the paddlefleet_ops FA-4 kernel, which this build
may predate, so the kernel entry points are stubbed throughout: what is under
test is the layer that feeds them -- the mask order, the argument passthrough
and the gradient slots -- not the communication.

Covers:
  - the import-time capability probe, in all four of its outcomes;
  - the "_overlap"/"_nonoverlap" suffix of cp_balance_mode, and that configs
    written without a suffix keep meaning non-overlap;
  - the gathered-KV traversal, against the validated FM-4 wrapper pipeline
    (preprocess_index_dual_chunks + rearrange_blocks + roll) and its
    hierarchical variant, plus the contiguous layout against a slice-and-concat
    oracle;
  - the PyLayer forward/backward: mask order per direction, argument
    passthrough and the learnable-sink gradient slots;
  - dispatch from flashmask_attention_cp and from DotProductAttention;
  - rejection of the features the overlapped kernel does not implement.
"""

import contextlib
import importlib.util
import os
import sys
import types
import unittest
from unittest import mock

# Insert local src/ before site-packages so we test the dev version
_project_root = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    )
)
sys.path.insert(0, os.path.join(_project_root, "src"))

import paddle

from paddlefleet import (
    context_parallel_utils as cpu,
    overlap_context_parallel as ocp,
)
from paddlefleet.transformer import dot_product_attention as dpa
from paddlefleet.transformer.enums import AttnMaskType
from paddlefleet.transformer.transformer_config import (
    TransformerConfig,
)

DUALCHUNK = "dualchunk_allgather_overlap"
CONTIGUOUS = "contiguous_allgather_overlap"


class _Group:
    """Stand-in for a CP process group; the layer only reads these two."""

    def __init__(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size


def _config(mode, **kwargs):
    return TransformerConfig(
        hidden_size=8, num_attention_heads=1, cp_balance_mode=mode, **kwargs
    )


@contextlib.contextmanager
def _stub_kernel(sink_grad=True):
    """Stub the FA-4 entry points and the CP group the layer resolves.

    ``create=True`` throughout: on a build without the overlapped kernel the
    module never bound these names, which is exactly the build these tests have
    to run on.
    """
    calls = {}

    def fwd(query, key, value, **kwargs):
        calls["fwd"] = kwargs
        return paddle.zeros_like(query), paddle.zeros(
            [query.shape[0], query.shape[2], query.shape[1]], dtype="float32"
        )

    def bwd(query, key, value, output, output_grad, lse, info, **kwargs):
        calls["bwd"] = kwargs
        calls["bwd_info"] = info
        return (
            paddle.full_like(query, 1.0),
            paddle.full_like(key, 2.0),
            paddle.full_like(value, 3.0),
            paddle.full([1], 4.0, dtype=query.dtype) if sink_grad else None,
        )

    class Info:
        def __init__(self, startend_row_indices, is_causal):
            self.startend_row_indices = startend_row_indices
            self.is_causal = is_causal

    fleet = mock.MagicMock()
    fleet.get_hybrid_communicate_group.return_value.get_context_parallel_group.return_value = _Group(
        1, 4
    )
    with (
        mock.patch.object(ocp, "OVERLAP_SUPPORTED", True, create=True),
        mock.patch.object(ocp, "_flash_attn_fwd", fwd, create=True),
        mock.patch.object(ocp, "_flash_attn_bwd", bwd, create=True),
        mock.patch.object(ocp, "FlashMaskInfoPaddle", Info, create=True),
        mock.patch.object(ocp, "fleet", fleet),
    ):
        yield calls


def _hier_map_chunk(logical_pos, my_pe, total_n_pes, gpus_per_node):
    """Rank mapping of dist_flashmask_dev/overlap_flashmask_fm4.py."""
    my_pe_node = my_pe % gpus_per_node
    my_node_id = my_pe // gpus_per_node
    num_nodes = total_n_pes // gpus_per_node
    if logical_pos < num_nodes:
        return (
            my_pe_node
            + ((my_node_id + logical_pos) % num_nodes) * gpus_per_node
        )
    adj_pos = logical_pos - num_nodes
    slot = adj_pos // num_nodes + 1
    sub = adj_pos % num_nodes
    base = (my_pe_node + slot) % gpus_per_node
    return base + ((my_node_id + sub) % num_nodes) * gpus_per_node


def _reference_dualchunk(
    startend_row_indices, rank, cp_size, seqlen_local, gpus_per_node=0
):
    """Mask pipeline of dist_flashmask_dev/overlap_flashmask_fm4.py.

    Returns the (forward, backward) gathered masks. The circular traversal is
    expressed as a roll, which is independent of the permutation the layer
    builds.
    """
    mask = cpu.preprocess_index_dual_chunks(
        startend_row_indices,
        chunk_id_first=rank,
        chunk_id_second=2 * cp_size - rank - 1,
        seq_blocksize=seqlen_local // 2,
        max_seqlen_q=seqlen_local // 2,
    )
    # rearrange_blocks: natural order -> DualChunkSwap rank order.
    batch_size, _, seqlen, _ = mask.shape
    n_blocks = 2 * cp_size
    order = []
    for i in range(cp_size):
        order += [i, n_blocks - 1 - i]
    mask = (
        mask.reshape([batch_size, -1, n_blocks, seqlen // n_blocks, 2])
        .index_select(paddle.to_tensor(order, dtype="int64"), axis=2)
        .reshape([batch_size, -1, seqlen, 2])
    )

    if not gpus_per_node:
        return (
            paddle._C_ops.roll(mask, shifts=-seqlen_local * (rank + 1), axis=2),
            paddle._C_ops.roll(mask, shifts=-seqlen_local * rank, axis=2),
        )

    def permute(positions):
        perm = paddle.to_tensor(
            [
                _hier_map_chunk(pos, rank, cp_size, gpus_per_node)
                for pos in positions
            ],
            dtype="int64",
        )
        return (
            mask.reshape([batch_size, -1, cp_size, seqlen_local, 2])
            .index_select(perm, axis=2)
            .reshape([batch_size, -1, seqlen, 2])
        )

    return permute(range(cp_size - 1, -1, -1)), permute(range(cp_size))


def _reference_contiguous(startend_row_indices, rank, cp_size, seqlen_local):
    """Contiguous-layout oracle: concatenate whole owner chunks by slicing.

    Under the contiguous layout a rank owns one contiguous run of keys, so the
    gathered buffer is just those runs in traversal order.
    """
    mask = cpu.preprocess_index(
        startend_row_indices,
        chunk_id=rank,
        seq_blocksize=seqlen_local,
        max_seqlen_q=seqlen_local,
    )

    def concat(positions):
        return paddle.concat(
            [
                mask[
                    :,
                    :,
                    ((rank + pos) % cp_size) * seqlen_local : (
                        (rank + pos) % cp_size + 1
                    )
                    * seqlen_local,
                    :,
                ]
                for pos in positions
            ],
            axis=2,
        )

    return concat([*range(1, cp_size), 0]), concat(range(cp_size))


def _gathered(startend_row_indices, rank, cp_size, seqlen_local, mode):
    group = _Group(rank, cp_size)
    mask = ocp.localize_mask(startend_row_indices, seqlen_local, group, mode)
    return (
        ocp.gathered_kv_order(mask, group, False, mode),
        ocp.gathered_kv_order(mask, group, True, mode),
    )


def _probe_overlap_supported(capability, interface, device_count=1):
    """Re-run the module's import-time capability probe under a fake build.

    Loaded as a private module: the probe decides an import-time constant, so
    the only way to see the other outcomes is to execute the module body again,
    and doing that to the shared instance would rebind its kernel entry points
    for every later test.
    """
    modules = {
        name: types.ModuleType(name)
        for name in (
            "paddlefleet_ops",
            "paddlefleet_ops.flash_mask",
            "paddlefleet_ops.flash_mask.cute",
        )
    }
    utils = types.ModuleType("paddlefleet_ops.flash_mask.cute.flashmask_utils")
    utils.FlashMaskInfoPaddle = object
    modules[utils.__name__] = utils
    # ``None`` makes the import machinery raise ImportError.
    modules["paddlefleet_ops.flash_mask.cute.interface"] = interface

    spec = importlib.util.spec_from_file_location("ocp_probe", ocp.__file__)
    module = importlib.util.module_from_spec(spec)
    with (
        mock.patch.object(
            paddle.device.cuda, "device_count", return_value=device_count
        ),
        mock.patch.object(
            paddle.device.cuda,
            "get_device_capability",
            return_value=capability,
        ),
        mock.patch.dict(sys.modules, modules),
    ):
        spec.loader.exec_module(module)
    return module.OVERLAP_SUPPORTED


def _fake_interface(with_group):
    module = types.ModuleType("paddlefleet_ops.flash_mask.cute.interface")
    if with_group:

        def _flash_attn_fwd(query, key, value, group=None):
            pass

    else:

        def _flash_attn_fwd(query, key, value):
            pass

    module._flash_attn_fwd = _flash_attn_fwd
    module._flash_attn_bwd = _flash_attn_fwd
    return module


class TestOverlapCapabilityProbe(unittest.TestCase):
    """The probe must be false unless the build really has the overlap."""

    def test_no_device(self):
        self.assertFalse(
            _probe_overlap_supported(
                (10, 3), _fake_interface(True), device_count=0
            )
        )

    def test_wrong_architecture(self):
        self.assertFalse(
            _probe_overlap_supported((9, 0), _fake_interface(True))
        )

    def test_kernel_without_group_parameter(self):
        # An SM100 build that predates the overlap: the entry point exists but
        # has no `group`, which is the entry point of the in-kernel overlap.
        self.assertFalse(
            _probe_overlap_supported((10, 3), _fake_interface(False))
        )

    def test_kernel_with_group_parameter(self):
        self.assertTrue(
            _probe_overlap_supported((10, 3), _fake_interface(True))
        )

    def test_missing_ops_package(self):
        self.assertFalse(_probe_overlap_supported((10, 3), None))


class TestCpOverlapConfig(unittest.TestCase):
    def test_suffix_normalized_into_cp_overlap(self):
        for layout in ("dualchunk_allgather", "contiguous_allgather"):
            for suffix, expected in (
                ("", False),
                ("_nonoverlap", False),
                ("_overlap", True),
            ):
                config = _config(layout + suffix)
                self.assertEqual(config.cp_balance_mode, layout)
                self.assertEqual(config.cp_overlap, expected)

    def test_suffixless_mode_untouched(self):
        config = _config("contiguous_a2a")
        self.assertEqual(config.cp_balance_mode, "contiguous_a2a")
        self.assertFalse(config.cp_overlap)

    def test_overlap_requires_an_allgather_layout(self):
        # The suffix is stripped before the layout whitelist, so the layout
        # itself stays valid and only the overlap restriction rejects it.
        with self.assertRaises(ValueError):
            _config("contiguous_a2a_overlap")
        with self.assertRaises(ValueError):
            _config("contiguous_a2a", cp_overlap=True)

    def test_unrecognized_suffix_still_rejected(self):
        # Only the exact "_overlap"/"_nonoverlap" suffixes are recognized; a
        # near miss must fall through to the layout whitelist.
        with self.assertRaises(ValueError):
            _config("dualchunk_allgather_overlapping")


class TestTraversal(unittest.TestCase):
    def test_hierarchical_switch_follows_kernel_semantics(self):
        # Unset, unrecognized and false-ish values leave the traversal circular;
        # a single node does too, because the kernel falls back to it as well.
        for value, cp_size, expected in (
            (None, 8, 0),
            ("", 8, 0),
            ("0", 8, 0),
            ("off", 8, 0),
            ("maybe", 8, 0),
            ("TRUE", 8, 0),
            ("TRUE", 16, 8),
            ("yes", 16, 8),
        ):
            environ = (
                {} if value is None else {"FLASHMASK_USE_HIERARCHICAL": value}
            )
            with mock.patch.dict(os.environ, environ, clear=True):
                self.assertEqual(
                    ocp.hierarchical_gpus_per_node(cp_size), expected
                )

    def test_node_size_read_from_environment(self):
        with mock.patch.dict(
            os.environ,
            {
                "FLASHMASK_USE_HIERARCHICAL": "1",
                "HIERARCHICAL_GPUS_PER_NODE": "4",
            },
        ):
            self.assertEqual(ocp.hierarchical_gpus_per_node(8), 4)
            # cp_size <= gpus_per_node is a single node, hence circular.
            self.assertEqual(ocp.hierarchical_gpus_per_node(4), 0)

    def test_traversal_visits_every_rank_once_from_the_local_one(self):
        for cp_size, gpus_per_node in ((8, 0), (8, 2), (8, 4), (16, 8)):
            for rank in range(cp_size):
                visited = [
                    ocp.traversal_rank(pos, rank, cp_size, gpus_per_node)
                    for pos in range(cp_size)
                ]
                self.assertEqual(visited[0], rank)
                self.assertEqual(sorted(visited), list(range(cp_size)))

    def test_hierarchical_traversal_starts_with_the_congruence_group(self):
        cp_size, gpus_per_node = 8, 2
        num_nodes = cp_size // gpus_per_node
        for rank in range(cp_size):
            group = [
                ocp.traversal_rank(pos, rank, cp_size, gpus_per_node)
                for pos in range(num_nodes)
            ]
            # Same intra-node slot, successive nodes: the cross-node phase.
            self.assertEqual(
                {peer % gpus_per_node for peer in group}, {rank % gpus_per_node}
            )
            self.assertEqual(len(set(group)), num_nodes)


class TestGatheredKvOrder(unittest.TestCase):
    def setUp(self):
        # The traversal cache samples the environment on the first miss, like
        # the kernel does; tests that switch the environment must invalidate it.
        ocp.BLOCK_ORDER_CACHE.clear()
        self.addCleanup(ocp.BLOCK_ORDER_CACHE.clear)

    @staticmethod
    def _random_mask(seqlen_total):
        return paddle.randint(
            0, seqlen_total, [2, 1, seqlen_total, 2], dtype="int32"
        )

    def _check(self, cp_size, mode, gpus_per_node=0):
        seqlen_local = 32
        startend_row_indices = self._random_mask(seqlen_local * cp_size)
        for rank in range(cp_size):
            if mode == DUALCHUNK:
                expected = _reference_dualchunk(
                    startend_row_indices,
                    rank,
                    cp_size,
                    seqlen_local,
                    gpus_per_node,
                )
            else:
                expected = _reference_contiguous(
                    startend_row_indices, rank, cp_size, seqlen_local
                )
            got = _gathered(
                startend_row_indices, rank, cp_size, seqlen_local, mode
            )
            for direction, (actual, ref) in zip(
                ("fwd", "bwd"), zip(got, expected)
            ):
                self.assertTrue(
                    bool((actual == ref).all()),
                    f"{mode} {direction} mismatch at rank {rank}/{cp_size}",
                )

    def test_dualchunk_matches_reference_pipeline(self):
        for cp_size in (1, 2, 4, 8):
            self._check(cp_size, DUALCHUNK)

    def test_dualchunk_hierarchical_matches_reference_pipeline(self):
        for gpus_per_node, cp_size in ((2, 4), (2, 8), (4, 8), (8, 16)):
            ocp.BLOCK_ORDER_CACHE.clear()
            with mock.patch.dict(
                os.environ,
                {
                    "FLASHMASK_USE_HIERARCHICAL": "true",
                    "HIERARCHICAL_GPUS_PER_NODE": str(gpus_per_node),
                },
            ):
                self._check(cp_size, DUALCHUNK, gpus_per_node)

    def test_contiguous_matches_slice_oracle(self):
        for cp_size in (1, 2, 4, 8):
            self._check(cp_size, CONTIGUOUS)

    def test_block_order_is_cached_per_key(self):
        first = ocp.block_order(4, 1, False, DUALCHUNK)
        self.assertIs(first, ocp.block_order(4, 1, False, DUALCHUNK))
        for key in (
            (4, 2, False, DUALCHUNK),
            (4, 1, True, DUALCHUNK),
            (4, 1, False, CONTIGUOUS),
        ):
            self.assertIsNot(first, ocp.block_order(*key))

    def test_unsupported_layout_rejected(self):
        with self.assertRaises(ValueError):
            ocp.localize_mask(
                paddle.zeros([1, 1, 8, 2], dtype="int32"),
                4,
                _Group(0, 2),
                "contiguous_a2a_overlap",
            )


class TestOverlapLayer(unittest.TestCase):
    """Forward/backward of the PyLayer, over a stubbed FA-4 kernel."""

    CP_SIZE = 4
    RANK = 1
    SEQLEN_LOCAL = 8

    def setUp(self):
        ocp.BLOCK_ORDER_CACHE.clear()
        self.addCleanup(ocp.BLOCK_ORDER_CACHE.clear)
        self.group = _Group(self.RANK, self.CP_SIZE)
        self.mask = paddle.randint(
            0,
            self.SEQLEN_LOCAL * self.CP_SIZE,
            [1, 1, self.SEQLEN_LOCAL * self.CP_SIZE, 2],
            dtype="int32",
        )
        self.qkv = []
        for _ in range(3):
            tensor = paddle.randn(
                [1, self.SEQLEN_LOCAL, 2, 16], dtype="bfloat16"
            )
            tensor.stop_gradient = False
            self.qkv.append(tensor)

    def _expected(self, backward, mode=DUALCHUNK):
        return _gathered(
            self.mask, self.RANK, self.CP_SIZE, self.SEQLEN_LOCAL, mode
        )[int(backward)]

    def _run(self, learnable_sink=None, mode=DUALCHUNK, softmax_scale=0.25):
        with _stub_kernel(sink_grad=learnable_sink is not None) as calls:
            output = ocp.overlap_flashmask_attention_cp(
                *self.qkv,
                self.mask,
                learnable_sink=learnable_sink,
                softmax_scale=softmax_scale,
                mode=mode,
            )
            output.backward(paddle.ones_like(output))
        return calls

    def test_forward_gets_the_forward_traversal_and_the_group(self):
        calls = self._run()
        kwargs = calls["fwd"]
        self.assertEqual(kwargs["group"].rank, self.RANK)
        self.assertEqual(kwargs["group"].world_size, self.CP_SIZE)
        self.assertFalse(kwargs["causal"])
        self.assertTrue(kwargs["return_lse"])
        self.assertFalse(kwargs["pack_gqa"])
        self.assertEqual(kwargs["softmax_scale"], 0.25)
        self.assertTrue(
            bool(
                (kwargs["startend_row_indices"] == self._expected(False)).all()
            )
        )

    def test_backward_gets_the_backward_traversal(self):
        calls = self._run()
        info = calls["bwd_info"]
        self.assertFalse(info.is_causal)
        self.assertTrue(
            bool((info.startend_row_indices == self._expected(True)).all())
        )
        # The two directions really differ, so the assertion above is not
        # satisfied by the forward order as well.
        self.assertFalse(
            bool((info.startend_row_indices == self._expected(False)).all())
        )

    def test_backward_passes_the_group_and_the_deterministic_flag(self):
        flag = paddle.get_flags(["FLAGS_cudnn_deterministic"])[
            "FLAGS_cudnn_deterministic"
        ]
        kwargs = self._run()["bwd"]
        self.assertEqual(kwargs["group"].rank, self.RANK)
        self.assertFalse(kwargs["causal"])
        self.assertEqual(kwargs["softmax_scale"], 0.25)
        self.assertEqual(kwargs["deterministic"], flag)

    def test_contiguous_layout_reaches_the_kernel(self):
        calls = self._run(mode=CONTIGUOUS)
        self.assertTrue(
            bool(
                (
                    calls["fwd"]["startend_row_indices"]
                    == self._expected(False, CONTIGUOUS)
                ).all()
            )
        )

    def test_query_key_value_gradients_come_from_the_kernel(self):
        self._run()
        for tensor, expected in zip(self.qkv, (1.0, 2.0, 3.0)):
            self.assertIsNotNone(tensor.grad)
            self.assertEqual(tensor.grad.shape, tensor.shape)
            self.assertAlmostEqual(
                float(tensor.grad.astype("float32").mean()), expected, places=3
            )

    def test_trainable_sink_gets_its_gradient_slot(self):
        sink = paddle.zeros([1], dtype="bfloat16")
        sink.stop_gradient = False
        self._run(learnable_sink=sink)
        self.assertIsNotNone(sink.grad)
        self.assertAlmostEqual(
            float(sink.grad.astype("float32").sum()), 4.0, places=3
        )

    def test_frozen_sink_omits_the_gradient_slot(self):
        # The kernel returns dsink whenever a sink is passed, but a frozen sink
        # has no slot to receive it: returning one anyway makes the PyLayer
        # reject the gradient arity.
        sink = paddle.zeros([1], dtype="bfloat16")
        sink.stop_gradient = True
        self._run(learnable_sink=sink)
        self.assertIsNone(sink.grad)
        for tensor in self.qkv:
            self.assertIsNotNone(tensor.grad)


class TestOverlapRejections(unittest.TestCase):
    @staticmethod
    def _call(seqlen=8, **kwargs):
        query = paddle.zeros([1, seqlen, 1, 8], dtype="bfloat16")
        return ocp.overlap_flashmask_attention_cp(
            query,
            query,
            query,
            paddle.zeros([1, 1, seqlen, 2], dtype="int32"),
            **kwargs,
        )

    def test_rejects_unimplemented_features(self):
        with _stub_kernel():
            for kwargs in (
                {"dropout": 0.1},
                {"causal": True},
                {"fixed_seed_offset": paddle.zeros([1], dtype="int64")},
            ):
                with self.assertRaises(NotImplementedError):
                    self._call(**kwargs)

    def test_rejects_odd_local_sequence_length(self):
        with _stub_kernel(), self.assertRaises(AssertionError):
            self._call(seqlen=7)

    def test_requires_capable_build(self):
        with (
            mock.patch.object(ocp, "OVERLAP_SUPPORTED", False),
            self.assertRaises(AssertionError),
        ):
            self._call()


class TestFlashmaskAttentionCpDispatch(unittest.TestCase):
    @staticmethod
    def _qkv():
        return paddle.zeros([1, 8, 1, 8], dtype="bfloat16")

    def test_overlap_mode_routes_to_the_overlap_layer(self):
        with mock.patch.object(
            ocp, "overlap_flashmask_attention_cp", return_value="overlapped"
        ) as patched:
            output = cpu.flashmask_attention_cp(
                self._qkv(),
                self._qkv(),
                self._qkv(),
                paddle.zeros([1, 1, 8, 2], dtype="int32"),
                mode=DUALCHUNK,
            )
        self.assertEqual(output, "overlapped")
        patched.assert_called_once()
        # The overlap layer dispatches on the full name, suffix included.
        self.assertEqual(patched.call_args[0][-1], DUALCHUNK)

    def test_plain_mode_never_reaches_the_overlap_layer(self):
        with (
            mock.patch.object(ocp, "overlap_flashmask_attention_cp") as patched,
            self.assertRaises(ValueError),
        ):
            cpu.flashmask_attention_cp(
                self._qkv(),
                self._qkv(),
                self._qkv(),
                paddle.zeros([1, 1, 8, 2], dtype="int32"),
                mode="unsupported_mode",
            )
        patched.assert_not_called()


class TestDotProductAttentionOverlapSuffix(unittest.TestCase):
    """cp_overlap is what puts the suffix back on for flashmask_attention_cp."""

    SEQLEN = 8
    CP_SIZE = 2
    HEADS = 4
    HEAD_DIM = 128

    def _attention(self, mode):
        config = TransformerConfig(
            num_hidden_layers=1,
            hidden_size=self.HEADS * self.HEAD_DIM,
            num_attention_heads=self.HEADS,
            cp_balance_mode=mode,
        )
        config.bf16 = True
        attention = dpa.DotProductAttention(
            config=config,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
        )
        # The real world size needs a process group; only its value is read.
        attention.context_parallel_size = self.CP_SIZE
        return attention

    def _dispatched_mode(self, mode, **kwargs):
        attention = self._attention(mode)
        qkv = paddle.randn(
            [1, self.SEQLEN, self.HEADS, self.HEAD_DIM], dtype="bfloat16"
        )
        with mock.patch.object(
            dpa, "flashmask_attention_cp", return_value=paddle.zeros_like(qkv)
        ) as patched:
            attention(
                query=qkv,
                key=qkv,
                value=qkv,
                attention_mask=None,
                attn_mask_startend_row_indices=paddle.zeros(
                    [1, 1, self.SEQLEN * self.CP_SIZE, 1], dtype="int32"
                ),
                attn_mask_type=AttnMaskType.causal,
                **kwargs,
            )
        return patched.call_args.kwargs["mode"]

    def test_overlap_config_appends_the_suffix(self):
        self.assertEqual(
            self._dispatched_mode("dualchunk_allgather_overlap"), DUALCHUNK
        )

    def test_plain_config_keeps_the_layout_name(self):
        self.assertEqual(
            self._dispatched_mode("dualchunk_allgather"), "dualchunk_allgather"
        )

    def test_overlap_rejects_refined_recompute(self):
        with self.assertRaises(AssertionError):
            self._dispatched_mode(
                "dualchunk_allgather_overlap", use_rr_flash_attention=True
            )


if __name__ == "__main__":
    unittest.main()

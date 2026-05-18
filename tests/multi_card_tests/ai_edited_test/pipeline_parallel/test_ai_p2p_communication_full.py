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
import unittest

import numpy as np
import paddle
from paddle.distributed import fleet

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)

from paddlefleet.pipeline_parallel.pp_utils.p2p_communication import (
    P2pHelper,
    P2PonCalcStream,
    SendRecvMeta,
    _batch_p2p_tuple_or_tensor,
    _batched_p2p_ops,
    _is_valid_send_recv_partial,
    _p2p_helper,
    _p2p_ops,
    _recv_on_calc_stream,
    _send_on_calc_stream,
    allgather_partial,
    batch_send_recv_on_calc_stream,
    initialize_p2p_groups,
)
from paddlefleet.pipeline_parallel.pp_utils.utils import paddle_2_number
from paddlefleet.training.initialize import initialize_fleet

PP_DEGREE = 2


def _init_pp():
    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": 1,
        "mp_degree": 1,
        "pp_degree": PP_DEGREE,
        "sharding_degree": 1,
        "sep_degree": 1,
        "cp_degree": 1,
        "ep_degree": 1,
        "moe_sharding_degree": 1,
        "order": [
            "sharding",
            "moe_sharding",
            "pp",
            "sep",
            "cp",
            "dp",
            "ep",
            "mp",
        ],
    }
    initialize_fleet(strategy)


def setUpModule():
    """Initialize fleet once for all tests in this module (PP=2)."""
    _init_pp()
    hcg = fleet.get_hybrid_communicate_group()
    initialize_p2p_groups(
        hcg, enable_partial_send_recv=True, enable_timer=False
    )
    np.random.seed(42)
    paddle.seed(42)


class TestInitializeP2PGroupsFull(unittest.TestCase):
    """Test initialize_p2p_groups with timer enabled."""

    def test_initialize_p2p_groups_basic(self):
        """initialize_p2p_groups should succeed with a valid HCG."""
        hcg = fleet.get_hybrid_communicate_group()
        initialize_p2p_groups(
            hcg, enable_partial_send_recv=True, enable_timer=False
        )
        initialize_p2p_groups(
            hcg, enable_partial_send_recv=False, enable_timer=False
        )


class TestSendRecvMetaBasic(unittest.TestCase):
    """Test SendRecvMeta basic functionality."""

    def test_send_recv_meta_init(self):
        """Verify SendRecvMeta initializes with None and False values."""
        meta = SendRecvMeta()
        self.assertIsNone(meta.send_shape_message)
        self.assertIsNone(meta.send_dtype_message)
        self.assertIsNone(meta.recv_shape_message)
        self.assertIsNone(meta.recv_dtype_message)
        self.assertIsNone(meta.recv_stop_gradient)
        self.assertFalse(meta.has_send_meta)
        self.assertFalse(meta.has_recv_meta)

    def test_set_send_message_single_tensor(self):
        """Verify set_send_message correctly populates shape and dtype."""
        meta = SendRecvMeta()
        tensor = paddle.randn([3, 6, 9], dtype="float32")
        meta.set_send_message(tensor)
        self.assertEqual(meta.send_shape_message, [3, 6, 9])
        self.assertEqual(
            meta.send_dtype_message, paddle_2_number(paddle.float32)
        )

    def test_set_send_message_tuple_tensor(self):
        """Verify set_send_message works with tuple input."""
        meta = SendRecvMeta()
        t1 = paddle.randn([2, 4], dtype="float32")
        t1.stop_gradient = False
        t2 = paddle.randn([2, 4], dtype="float16")
        t2.stop_gradient = False
        meta.set_send_message((t1, t2))
        self.assertIsInstance(meta.send_shape_message, tuple)
        self.assertEqual(len(meta.send_shape_message), 2)

    def test_obtain_send_message_with_stop_gradient(self):
        """Test _obtain_send_message with stop_gradient tensor."""
        meta = SendRecvMeta()
        t1 = paddle.randn([2, 4], dtype="float32")
        t1.stop_gradient = True
        t2 = paddle.randn([2, 4], dtype="float32")
        t2.stop_gradient = False
        meta.set_send_message((t1, t2))
        self.assertIsInstance(meta.send_shape_message, tuple)
        self.assertEqual(len(meta.send_shape_message), 1)

    def test_send_meta_list_tensor(self):
        """Test send_meta with list of tensors."""
        meta = SendRecvMeta()
        t1 = paddle.randn([2, 4], dtype="float32")
        t1.stop_gradient = False
        t2 = paddle.randn([2, 4], dtype="float16")
        t2.stop_gradient = False
        tensors = [t1, t2]
        meta.set_send_message(tensors)
        self.assertIsInstance(meta.send_shape_message, tuple)
        self.assertEqual(len(meta.send_shape_message), 2)

    def test_init_or_erase_meta(self):
        """Test init_or_erase_meta resets all fields."""
        meta = SendRecvMeta()
        meta.send_shape_message = [2, 3]
        meta.send_dtype_message = 1
        meta.recv_shape_message = [4, 5]
        meta.recv_dtype_message = 3
        meta.has_send_meta = True
        meta.has_recv_meta = True
        meta.init_or_erase_meta()
        self.assertIsNone(meta.send_shape_message)
        self.assertIsNone(meta.send_dtype_message)
        self.assertIsNone(meta.recv_shape_message)
        self.assertIsNone(meta.recv_dtype_message)
        self.assertFalse(meta.has_send_meta)
        self.assertFalse(meta.has_recv_meta)

    def test_repr(self):
        """Test __repr__ output."""
        meta = SendRecvMeta()
        repr_str = repr(meta)
        self.assertIn("send_shape_message", repr_str)
        self.assertIn("send_dtype_message", repr_str)
        self.assertIn("recv_shape_message", repr_str)
        self.assertIn("recv_dtype_message", repr_str)

    def test_check_send_message_match(self):
        """check_send_message should pass when tensor matches."""
        meta = SendRecvMeta()
        tensor = paddle.randn([4, 8], dtype="float32")
        meta.set_send_message(tensor)
        meta.check_send_message(tensor)

    def test_check_send_message_mismatch(self):
        """check_send_message should raise AssertionError on shape mismatch."""
        meta = SendRecvMeta()
        t1 = paddle.randn([4, 8], dtype="float32")
        meta.set_send_message(t1)
        t2 = paddle.randn([4, 16], dtype="float32")
        with self.assertRaises(AssertionError):
            meta.check_send_message(t2)

    def test_check_send_message_none(self):
        """check_send_message should not raise when messages are None."""
        meta = SendRecvMeta()
        tensor = paddle.randn([4, 8], dtype="float32")
        meta.check_send_message(tensor)


class TestIsValidSendRecvPartial(unittest.TestCase):
    """Test _is_valid_send_recv_partial with various tensor shapes and mp_degree."""

    def test_is_valid_send_recv_partial_not_divisible(self):
        """Should return False when tensor numel is not divisible by mp_degree."""
        tensor = paddle.randn([3, 7], dtype="float32")
        self.assertFalse(_is_valid_send_recv_partial(tensor, 4))

    def test_is_valid_send_recv_partial_mp_degree_one(self):
        """Should return False when mp_degree is 1."""
        tensor = paddle.randn([4, 8], dtype="float32")
        self.assertFalse(_is_valid_send_recv_partial(tensor, 1))

    def test_is_valid_send_recv_partial_divisible(self):
        """Should return True when tensor numel is divisible by mp_degree > 1."""
        import paddlefleet.pipeline_parallel.pp_utils.p2p_communication as p2p_mod

        p2p_mod._enable_partial_send_recv = True
        tensor = paddle.randn([4, 8], dtype="float32")
        self.assertTrue(_is_valid_send_recv_partial(tensor, 2))

    def test_is_valid_send_recv_partial_large_mp(self):
        """Should return True when tensor is large enough and divisible."""
        import paddlefleet.pipeline_parallel.pp_utils.p2p_communication as p2p_mod

        p2p_mod._enable_partial_send_recv = True
        tensor = paddle.randn([16, 16], dtype="float32")
        self.assertTrue(_is_valid_send_recv_partial(tensor, 8))

    def test_is_valid_send_recv_partial_odd_numel(self):
        """Should return False for odd numel with even mp_degree."""
        tensor = paddle.randn([3], dtype="float32")
        self.assertFalse(_is_valid_send_recv_partial(tensor, 2))

    def test_is_valid_send_recv_partial_disabled(self):
        """Should return False when partial send/recv is disabled."""
        import paddlefleet.pipeline_parallel.pp_utils.p2p_communication as p2p_mod

        p2p_mod._enable_partial_send_recv = False
        tensor = paddle.randn([4, 8], dtype="float32")
        self.assertFalse(_is_valid_send_recv_partial(tensor, 2))
        p2p_mod._enable_partial_send_recv = True


class TestAllgatherPartial(unittest.TestCase):
    """Test allgather_partial function."""

    def test_allgather_partial_no_op(self):
        """allgather_partial returns tensor unchanged when not divisible by nranks."""
        tensor = paddle.randn([3, 7], dtype="float32")
        result = allgather_partial(tensor, nranks=4, rank_id=0)
        self.assertIs(result, tensor)

    def test_allgather_partial_mp_degree_one(self):
        """allgather_partial returns tensor unchanged when nranks=1."""
        tensor = paddle.randn([4, 8], dtype="float32")
        result = allgather_partial(tensor, nranks=1, rank_id=0)
        self.assertIs(result, tensor)


class TestP2PonCalcStream(unittest.TestCase):
    """Test P2PonCalcStream class."""

    def test_p2p_on_calc_stream_creation_send(self):
        """P2PonCalcStream creation should succeed with _send_on_calc_stream."""
        from paddlefleet.pipeline_parallel.pp_utils.p2p_communication import (
            _send_on_calc_stream,
        )

        tensor = paddle.randn([4], dtype="float32")
        p2p_op = P2PonCalcStream(_send_on_calc_stream, tensor, 0, None)
        self.assertIs(p2p_op.tensor, tensor)
        self.assertEqual(p2p_op.peer, 0)

    def test_p2p_on_calc_stream_creation_recv(self):
        """P2PonCalcStream creation should succeed with _recv_on_calc_stream."""
        from paddlefleet.pipeline_parallel.pp_utils.p2p_communication import (
            _recv_on_calc_stream,
        )

        tensor = paddle.randn([4], dtype="float32")
        p2p_op = P2PonCalcStream(_recv_on_calc_stream, tensor, 0, None)
        self.assertIs(p2p_op.tensor, tensor)

    def test_p2p_on_calc_stream_creation_invalid(self):
        """P2PonCalcStream should raise RuntimeError for invalid op."""
        tensor = paddle.randn([4], dtype="float32")
        with self.assertRaises(RuntimeError):
            P2PonCalcStream(lambda x: x, tensor, 0, None)

    def test_p2p_on_calc_stream_with_mp_params(self):
        """P2PonCalcStream with model parallel parameters."""
        from paddlefleet.pipeline_parallel.pp_utils.p2p_communication import (
            _send_on_calc_stream,
        )

        tensor = paddle.randn([4], dtype="float32")
        p2p_op = P2PonCalcStream(
            _send_on_calc_stream, tensor, 0, None, nranks=2, rank_id=0
        )
        self.assertEqual(p2p_op.nranks, 2)
        self.assertEqual(p2p_op.rank_id, 0)


class TestBatchP2pTupleOrTensor(unittest.TestCase):
    """Test _batch_p2p_tuple_or_tensor function."""

    def test_batch_p2p_tuple_or_tensor_single(self):
        """Test _batch_p2p_tuple_or_tensor with single tensor."""
        hcg = fleet.get_hybrid_communicate_group()
        pipe_group = hcg.get_pipe_parallel_group()
        from paddlefleet.pipeline_parallel.pp_utils.p2p_communication import (
            _send_on_calc_stream,
        )

        tensor = paddle.randn([4, 8], dtype="float32")
        ops = _batch_p2p_tuple_or_tensor(
            tensor, _send_on_calc_stream, 0, pipe_group
        )
        self.assertEqual(len(ops), 1)
        self.assertIsInstance(ops[0], P2PonCalcStream)

    def test_batch_p2p_tuple_or_tensor_tuple(self):
        """Test _batch_p2p_tuple_or_tensor with tuple of tensors."""
        hcg = fleet.get_hybrid_communicate_group()
        pipe_group = hcg.get_pipe_parallel_group()
        from paddlefleet.pipeline_parallel.pp_utils.p2p_communication import (
            _send_on_calc_stream,
        )

        t1 = paddle.randn([2, 4], dtype="float32")
        t2 = paddle.randn([2, 4], dtype="float16")
        ops = _batch_p2p_tuple_or_tensor(
            (t1, t2), _send_on_calc_stream, 0, pipe_group
        )
        self.assertEqual(len(ops), 2)

    def test_batch_p2p_tuple_or_tensor_with_mp(self):
        """Test _batch_p2p_tuple_or_tensor with model parallel parameters."""
        hcg = fleet.get_hybrid_communicate_group()
        pipe_group = hcg.get_pipe_parallel_group()
        from paddlefleet.pipeline_parallel.pp_utils.p2p_communication import (
            _send_on_calc_stream,
        )

        tensor = paddle.randn([4, 8], dtype="float32")
        ops = _batch_p2p_tuple_or_tensor(
            tensor, _send_on_calc_stream, 0, pipe_group, mp_degree=2, mp_rank=0
        )
        self.assertEqual(len(ops), 1)


class TestP2pHelperClass(unittest.TestCase):
    """Test P2pHelper class initialization and basic methods."""

    def test_p2p_helper_init_default(self):
        """P2pHelper should initialize with default use_cache=True."""
        helper = P2pHelper()
        self.assertTrue(helper._use_cache)
        self.assertIsInstance(helper._send_recv_meta, SendRecvMeta)

    def test_p2p_helper_init_no_cache(self):
        """P2pHelper should respect use_cache=False parameter."""
        helper = P2pHelper(use_cache=False)
        self.assertFalse(helper._use_cache)

    def test_p2p_helper_init_dynamic_shape(self):
        """P2pHelper should initialize with dynamic_shape=True."""
        helper = P2pHelper(use_cache=True, dynamic_shape=True)
        self.assertTrue(helper._dynamic_shape)
        self.assertEqual(helper._dynamic_cnt, 0)

    def test_p2p_helper_clear_meta_cache(self):
        """Test P2pHelper.clear_meta_cache."""
        helper = P2pHelper(use_cache=True, dynamic_shape=False)
        helper._send_recv_meta.send_shape_message = [2, 3]
        helper._send_recv_meta.send_dtype_message = 1
        helper.clear_meta_cache()
        self.assertIsNone(helper._send_recv_meta.send_shape_message)
        self.assertIsNone(helper._send_recv_meta.send_dtype_message)

    def test_p2p_helper_repr(self):
        """Test P2pHelper.__repr__."""
        helper = P2pHelper(use_cache=True, dynamic_shape=False)
        repr_str = repr(helper)
        self.assertIn("using cache", repr_str)


class TestP2pHelperStageMethods(unittest.TestCase):
    """Test P2pHelper methods that return early for certain stages."""

    def test_send_forward_last_stage(self):
        """send_forward should do nothing for last stage."""
        helper = P2pHelper()
        tensor = paddle.randn([2, 4], dtype="float32")
        helper.send_forward(tensor, pp_last_stage=True)

    def test_recv_forward_first_stage(self):
        """recv_forward should return None for first stage."""
        helper = P2pHelper()
        result = helper.recv_forward(pp_first_stage=True, sync_recv=True)
        self.assertIsNone(result)

    def test_recv_backward_last_stage(self):
        """recv_backward should return None for last stage."""
        helper = P2pHelper()
        result = helper.recv_backward(pp_last_stage=True, sync_recv=True)
        self.assertIsNone(result)

    def test_send_backward_first_stage(self):
        """send_backward should do nothing for first stage."""
        helper = P2pHelper()
        tensor = paddle.randn([2, 4], dtype="float32")
        helper.send_backward(tensor, pp_first_stage=True)

    def test_send_forward_recv_backward_last_stage(self):
        """send_forward_recv_backward should return None for last stage."""
        helper = P2pHelper()
        tensor = paddle.randn([2, 4], dtype="float32")
        result = helper.send_forward_recv_backward(tensor, pp_last_stage=True)
        self.assertIsNone(result)

    def test_send_backward_recv_forward_first_stage(self):
        """send_backward_recv_forward should return None for first stage."""
        helper = P2pHelper()
        tensor = paddle.randn([2, 4], dtype="float32")
        result = helper.send_backward_recv_forward(tensor, pp_first_stage=True)
        self.assertIsNone(result)


class TestSendRecvMetaCommunication(unittest.TestCase):
    """Test SendRecvMeta send_meta and recv_meta with coordinated communication."""

    def test_send_recv_meta_single_tensor(self):
        """Test send_meta and recv_meta with single tensor."""
        hcg = fleet.get_hybrid_communicate_group()
        pp_rank = hcg.get_stage_id()
        pp_group = hcg.get_pipe_parallel_group()

        meta = SendRecvMeta()
        tensor = paddle.randn([2, 4], dtype="float32")

        if pp_rank == 0:
            meta.send_meta(tensor, pp_group)
        else:
            meta.recv_meta(pp_group)
            self.assertEqual(meta.recv_shape_message, [2, 4])
            self.assertEqual(
                meta.recv_dtype_message, paddle_2_number(paddle.float32)
            )

    def test_send_recv_meta_tuple_tensor(self):
        """Test send_meta and recv_meta with tuple of tensors."""
        hcg = fleet.get_hybrid_communicate_group()
        pp_rank = hcg.get_stage_id()
        pp_group = hcg.get_pipe_parallel_group()

        meta = SendRecvMeta()
        t1 = paddle.randn([2, 4], dtype="float32")
        t2 = paddle.randn([2, 4], dtype="float16")

        if pp_rank == 0:
            meta.send_meta((t1, t2), pp_group)
        else:
            meta.recv_meta(pp_group)
            self.assertIsInstance(meta.recv_shape_message, tuple)
            self.assertEqual(len(meta.recv_shape_message), 2)

    def test_send_recv_meta_reverse(self):
        """Test send_meta and recv_meta with reverse=True."""
        hcg = fleet.get_hybrid_communicate_group()
        pp_rank = hcg.get_stage_id()
        pp_group = hcg.get_pipe_parallel_group()

        meta = SendRecvMeta()
        tensor = paddle.randn([3, 6], dtype="float32")

        if pp_rank == 1:
            meta.send_meta(tensor, pp_group, reverse=True)
        else:
            meta.recv_meta(pp_group, reverse=True)
            self.assertEqual(meta.recv_shape_message, [3, 6])

    def test_send_recv_meta_with_key(self):
        """Test send_meta and recv_meta with tensor key attribute."""
        hcg = fleet.get_hybrid_communicate_group()
        pp_rank = hcg.get_stage_id()
        pp_group = hcg.get_pipe_parallel_group()

        meta = SendRecvMeta()
        tensor = paddle.randn([4, 8], dtype="float32")
        tensor.key = "test_tensor_key"

        if pp_rank == 0:
            meta.send_meta(tensor, pp_group)
        else:
            meta.recv_meta(pp_group)
            self.assertEqual(meta.recv_key_message, "test_tensor_key")


class TestSendRecvOnCalcStream(unittest.TestCase):
    """Test _send_on_calc_stream and _recv_on_calc_stream."""

    def test_send_recv_on_calc_stream_basic(self):
        """Test send and recv on calc stream with coordination."""
        hcg = fleet.get_hybrid_communicate_group()
        pp_rank = hcg.get_stage_id()
        pp_group = hcg.get_pipe_parallel_group()

        send_tensor = paddle.randn([4, 8], dtype="float32")
        recv_tensor = paddle.empty([4, 8], dtype="float32")
        peer = 1 - pp_rank

        if pp_rank == 0:
            _send_on_calc_stream(send_tensor, pp_group, peer)
        else:
            _recv_on_calc_stream(recv_tensor, pp_group, peer)

        paddle.distributed.barrier()

    def test_send_recv_on_calc_stream_with_nranks(self):
        """Test send and recv with nranks parameter."""
        hcg = fleet.get_hybrid_communicate_group()
        pp_rank = hcg.get_stage_id()
        pp_group = hcg.get_pipe_parallel_group()

        send_tensor = paddle.randn([4, 8], dtype="float32")
        recv_tensor = paddle.empty([4, 8], dtype="float32")
        peer = 1 - pp_rank

        if pp_rank == 0:
            _send_on_calc_stream(
                send_tensor, pp_group, peer, nranks=1, rank_id=0
            )
        else:
            _recv_on_calc_stream(
                recv_tensor, pp_group, peer, nranks=1, rank_id=0
            )

        paddle.distributed.barrier()


class TestBatchSendRecvOnCalcStream(unittest.TestCase):
    """Test batch_send_recv_on_calc_stream."""

    def test_batch_send_recv_on_calc_stream_send_recv(self):
        """Test batch_send_recv_on_calc_stream with send and recv operations."""
        hcg = fleet.get_hybrid_communicate_group()
        pp_rank = hcg.get_stage_id()
        pp_group = hcg.get_pipe_parallel_group()

        send_tensor = paddle.randn([4, 8], dtype="float32")
        recv_tensor = paddle.empty([4, 8], dtype="float32")
        peer = 1 - pp_rank

        if pp_rank == 0:
            send_op = P2PonCalcStream(
                _send_on_calc_stream, send_tensor, peer, pp_group
            )
            recv_op = P2PonCalcStream(
                _recv_on_calc_stream, recv_tensor, peer, pp_group
            )
            batch_send_recv_on_calc_stream([send_op, recv_op])
        else:
            recv_op = P2PonCalcStream(
                _recv_on_calc_stream, recv_tensor, peer, pp_group
            )
            send_op = P2PonCalcStream(
                _send_on_calc_stream, send_tensor, peer, pp_group
            )
            batch_send_recv_on_calc_stream([recv_op, send_op])

        paddle.distributed.barrier()


class TestBatchedP2pOps(unittest.TestCase):
    """Test _batched_p2p_ops."""

    def test_batched_p2p_ops_all_none(self):
        """Test _batched_p2p_ops with all None tensors."""
        hcg = fleet.get_hybrid_communicate_group()
        _batched_p2p_ops(None, None, None, None, hcg)
        paddle.distributed.barrier()

    def test_batched_p2p_ops_send_next_recv_prev(self):
        """Test _batched_p2p_ops with send_next and recv_prev."""
        hcg = fleet.get_hybrid_communicate_group()
        pp_rank = hcg.get_stage_id()

        send_tensor = paddle.randn([4, 8], dtype="float32")
        recv_tensor = paddle.empty([4, 8], dtype="float32")

        if pp_rank == 0:
            _batched_p2p_ops(None, None, send_tensor, recv_tensor, hcg)
        else:
            _batched_p2p_ops(recv_tensor, send_tensor, None, None, hcg)

        paddle.distributed.barrier()

    def test_batched_p2p_ops_tuple_tensors(self):
        """Test _batched_p2p_ops with tuple of tensors."""
        hcg = fleet.get_hybrid_communicate_group()
        pp_rank = hcg.get_stage_id()

        t1 = paddle.randn([2, 4], dtype="float32")
        t2 = paddle.randn([2, 4], dtype="float32")
        recv1 = paddle.empty([2, 4], dtype="float32")
        recv2 = paddle.empty([2, 4], dtype="float32")

        if pp_rank == 0:
            _batched_p2p_ops(None, None, (t1, t2), (recv1, recv2), hcg)
        else:
            _batched_p2p_ops((recv1, recv2), (t1, t2), None, None, hcg)

        paddle.distributed.barrier()


class TestP2pOps(unittest.TestCase):
    """Test _p2p_ops."""

    def test_p2p_ops_all_none(self):
        """Test _p2p_ops with all None tensors."""
        hcg = fleet.get_hybrid_communicate_group()
        reqs = _p2p_ops(None, None, None, None, hcg)
        self.assertEqual(len(reqs), 0)
        paddle.distributed.barrier()

    def test_p2p_ops_send_recv(self):
        """Test _p2p_ops with send and recv."""
        hcg = fleet.get_hybrid_communicate_group()
        pp_rank = hcg.get_stage_id()

        send_tensor = paddle.randn([4, 8], dtype="float32")
        recv_tensor = paddle.empty([4, 8], dtype="float32")

        if pp_rank == 0:
            reqs = _p2p_ops(None, None, send_tensor, recv_tensor, hcg)
        else:
            reqs = _p2p_ops(recv_tensor, send_tensor, None, None, hcg)

        for req in reqs:
            req.wait()

        paddle.distributed.barrier()


class TestP2pHelperFunction(unittest.TestCase):
    """Test _p2p_helper with coordinated communication."""

    def test_p2p_helper_recv_prev_single(self):
        """Test _p2p_helper with recv_prev=True."""
        hcg = fleet.get_hybrid_communicate_group()
        pp_rank = hcg.get_stage_id()

        meta = SendRecvMeta()
        meta.send_shape_message = [4, 8]
        meta.send_dtype_message = paddle_2_number(paddle.float32)
        meta.send_key_message = None
        meta.recv_shape_message = [4, 8]
        meta.recv_dtype_message = paddle_2_number(paddle.float32)
        meta.recv_stop_gradient = False
        meta.recv_key_message = None

        if pp_rank == 0:
            tensor = paddle.randn([4, 8], dtype="float32")
            meta.send_meta(tensor, hcg.get_pipe_parallel_group())
            _p2p_helper(
                tensor_send_next=tensor,
                tensor_send_prev=None,
                recv_prev=False,
                recv_next=False,
                send_recv_meta=meta,
                batch_p2p_comm=True,
            )
        else:
            recv_prev, recv_next, reqs = _p2p_helper(
                tensor_send_next=None,
                tensor_send_prev=None,
                recv_prev=True,
                recv_next=False,
                send_recv_meta=meta,
                batch_p2p_comm=True,
            )
            self.assertIsNotNone(recv_prev)
            self.assertEqual(recv_prev.shape, [4, 8])

        paddle.distributed.barrier()


class TestP2pHelperDistributedMethods(unittest.TestCase):
    """Test P2pHelper methods with distributed communication."""

    def test_p2p_helper_send_recv_forward(self):
        """Test P2pHelper send_forward and recv_forward."""
        hcg = fleet.get_hybrid_communicate_group()
        pp_rank = hcg.get_stage_id()

        helper = P2pHelper(use_cache=True, dynamic_shape=False)
        tensor = paddle.randn([4, 8], dtype="float32")

        if pp_rank == 0:
            helper.send_forward(tensor, pp_last_stage=False)
        else:
            result = helper.recv_forward(pp_first_stage=False, sync_recv=True)
            self.assertIsNotNone(result)
            self.assertEqual(result.shape, [4, 8])

        paddle.distributed.barrier()

    def test_p2p_helper_send_recv_backward(self):
        """Test P2pHelper send_backward and recv_backward."""
        hcg = fleet.get_hybrid_communicate_group()
        pp_rank = hcg.get_stage_id()

        helper = P2pHelper(use_cache=True, dynamic_shape=False)
        tensor = paddle.randn([4, 8], dtype="float32")

        if pp_rank == 1:
            helper.send_backward(tensor, pp_first_stage=False)
        else:
            result = helper.recv_backward(pp_last_stage=False, sync_recv=True)
            self.assertIsNotNone(result)
            self.assertEqual(result.shape, [4, 8])

        paddle.distributed.barrier()

    def test_p2p_helper_dynamic_shape_recv_forward(self):
        """Test P2pHelper.recv_forward with dynamic_shape=True."""
        hcg = fleet.get_hybrid_communicate_group()
        pp_rank = hcg.get_stage_id()

        helper = P2pHelper(use_cache=True, dynamic_shape=True)
        tensor = paddle.randn([4, 8], dtype="float32")

        if pp_rank == 0:
            helper.send_forward(tensor, pp_last_stage=False)
        else:
            result = helper.recv_forward(pp_first_stage=False, sync_recv=True)
            self.assertIsNotNone(result)

        paddle.distributed.barrier()

    def test_p2p_helper_dynamic_shape_recv_backward(self):
        """Test P2pHelper.recv_backward with dynamic_shape=True."""
        hcg = fleet.get_hybrid_communicate_group()
        pp_rank = hcg.get_stage_id()

        helper = P2pHelper(use_cache=True, dynamic_shape=True)
        tensor = paddle.randn([4, 8], dtype="float32")

        if pp_rank == 1:
            helper.send_backward(tensor, pp_first_stage=False)
        else:
            result = helper.recv_backward(pp_last_stage=False, sync_recv=True)
            self.assertIsNotNone(result)

        paddle.distributed.barrier()


if __name__ == "__main__":
    unittest.main()

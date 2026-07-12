# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import unittest

import paddle
from paddle.distributed.fleet.meta_parallel.pp_utils.forward_backward_overlap_utils import (
    ScheduleNode,
)
from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
    SendRecvMeta,
)

import paddlefleet  # noqa: F401 - installs the IndexCache pipeline patch
from paddlefleet.indexcache_pp_runtime_patch import (
    _dict_to_tuple_helper,
    _get_pipeline_key,
    _normalize_pipeline_input_gradients,
    _tuple_to_dict_helper,
)


def _make_distill_state(offset):
    return (
        paddle.to_tensor([offset], dtype="int64"),
        paddle.to_tensor([1.0 + offset], stop_gradient=False),
        paddle.to_tensor([2.0 + offset], stop_gradient=False),
        paddle.to_tensor([3.0 + offset], stop_gradient=False),
        paddle.to_tensor([offset], dtype="int64"),
        paddle.to_tensor([0.5 + offset]),
        paddle.to_tensor([offset], dtype="int64"),
        paddle.to_tensor([0], dtype="int64"),
    )


class TestIndexCachePipelineBackward(unittest.TestCase):
    def test_replaced_distill_state_uses_sendable_zero_gradients(self):
        old_state = _make_distill_state(0)
        new_state = _make_distill_state(10)
        node = ScheduleNode(
            lambda inputs: {
                "hidden": inputs["hidden"] * 2,
                "indexcache_state": new_state,
            },
            name="replace_indexcache_producer",
        )

        outputs = node.forward(
            {
                "hidden": paddle.to_tensor([4.0], stop_gradient=False),
                "indexcache_state": old_state,
            }
        )
        output_grads = tuple(
            paddle.ones_like(tensor)
            for tensor in _dict_to_tuple_helper(outputs)
            if not tensor.stop_gradient
        )
        input_grads = node.backward(output_grads)

        self.assertEqual(len(input_grads), 4)
        self.assertTrue(all(isinstance(grad, paddle.Tensor) for grad in input_grads))
        self.assertTrue(all(not grad.stop_gradient for grad in input_grads))
        self.assertEqual(input_grads[0].item(), 2.0)
        for grad in input_grads[1:]:
            self.assertEqual(grad.item(), 0.0)

        meta = SendRecvMeta()
        meta.set_send_message(input_grads)
        self.assertEqual(len(meta.send_shape_message), 4)

    def test_missing_non_indexcache_gradient_fails_fast(self):
        node = ScheduleNode(
            lambda inputs: {"hidden": inputs["hidden"] * 2},
            name="drop_regular_pipeline_input",
        )
        outputs = node.forward(
            {
                "hidden": paddle.to_tensor([4.0], stop_gradient=False),
                "unused": paddle.to_tensor([5.0], stop_gradient=False),
            }
        )
        output_grads = tuple(
            paddle.ones_like(tensor)
            for tensor in _dict_to_tuple_helper(outputs)
            if not tensor.stop_gradient
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "missing a gradient outside IndexCache state",
        ):
            node.backward(output_grads)

    def test_outer_pipeline_boundary_uses_sendable_zero_gradients(self):
        pipeline_inputs = _dict_to_tuple_helper(
            {
                "hidden_states": paddle.to_tensor(
                    [4.0], stop_gradient=False
                ),
                "indexcache_state": _make_distill_state(0),
            }
        )
        converted_inputs, use_dict = _tuple_to_dict_helper(pipeline_inputs)

        self.assertTrue(use_dict)
        self.assertIn("indexcache_state", converted_inputs)
        self.assertTrue(all(not hasattr(tensor, "key") for tensor in pipeline_inputs))

        hidden_grad = paddle.ones_like(pipeline_inputs[0])
        hidden_grad.stop_gradient = False
        input_grads = _normalize_pipeline_input_gradients(
            pipeline_inputs,
            (hidden_grad, None, None, None),
        )

        self.assertEqual(len(input_grads), 4)
        self.assertTrue(all(isinstance(grad, paddle.Tensor) for grad in input_grads))
        self.assertTrue(all(not grad.stop_gradient for grad in input_grads))
        self.assertEqual(input_grads[0].item(), 1.0)
        for grad in input_grads[1:]:
            self.assertEqual(grad.item(), 0.0)

        meta = SendRecvMeta()
        meta.set_send_message(input_grads)
        self.assertEqual(len(meta.send_shape_message), 4)

    def test_outer_pipeline_boundary_preserves_baseline_none_gradient(self):
        hidden = paddle.to_tensor([4.0], stop_gradient=False)
        unused = paddle.to_tensor([5.0], stop_gradient=False)
        input_grads = (paddle.ones_like(hidden), None)

        normalized = _normalize_pipeline_input_gradients(
            (hidden, unused),
            input_grads,
        )

        self.assertIs(normalized, input_grads)

    def test_outer_pipeline_boundary_handles_released_state(self):
        pipeline_inputs = _dict_to_tuple_helper(
            {
                "hidden_states": paddle.to_tensor(
                    [4.0], stop_gradient=False
                ),
                "indexcache_state": _make_distill_state(0),
            }
        )
        _tuple_to_dict_helper(pipeline_inputs)
        released_inputs = [
            tensor
            for tensor in pipeline_inputs
            if _get_pipeline_key(tensor)
            in {
                "indexcache_state 1",
                "indexcache_state 2",
                "indexcache_state 3",
            }
        ]
        expected_metadata = [
            (list(tensor.shape), tensor.dtype) for tensor in released_inputs
        ]
        for tensor in released_inputs:
            tensor._clear_dataptr()

        hidden_grad = paddle.ones_like(pipeline_inputs[0])
        hidden_grad.stop_gradient = False
        input_grads = _normalize_pipeline_input_gradients(
            pipeline_inputs,
            (hidden_grad, None, None, None),
        )

        self.assertEqual(len(released_inputs), 3)
        self.assertEqual(len(input_grads), 4)
        for grad, (shape, dtype) in zip(input_grads[1:], expected_metadata):
            self.assertEqual(list(grad.shape), shape)
            self.assertEqual(grad.dtype, dtype)
            self.assertFalse(grad.stop_gradient)
            self.assertEqual(float(grad.sum().item()), 0.0)

        meta = SendRecvMeta()
        meta.set_send_message(input_grads)
        self.assertEqual(len(meta.send_shape_message), 4)

    def test_outer_pipeline_boundary_rejects_missing_hidden_gradient(self):
        pipeline_inputs = _dict_to_tuple_helper(
            {
                "hidden_states": paddle.to_tensor(
                    [4.0], stop_gradient=False
                ),
                "indexcache_state": _make_distill_state(0),
            }
        )
        _tuple_to_dict_helper(pipeline_inputs)

        with self.assertRaisesRegex(
            RuntimeError,
            "missing a gradient outside IndexCache state",
        ):
            _normalize_pipeline_input_gradients(
                pipeline_inputs,
                (None, None, None, None),
            )


if __name__ == "__main__":
    unittest.main()

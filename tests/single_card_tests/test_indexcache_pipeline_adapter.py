# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import unittest
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

import paddle
from paddle.distributed.fleet.meta_parallel import pipeline_parallel
from paddle.distributed.fleet.meta_parallel.pp_utils.forward_backward_overlap_utils import (
    ScheduleNode,
)
from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
    SendRecvMeta,
)

from paddlefleet.pipeline_parallel.indexcache_adapter import (
    _clone_and_clear_dataptr,
    _debug_state5_gradient,
    _detach_and_requires_grad,
    _dict_to_tuple_helper,
    _get_pipeline_key,
    _indexcache_producer_layer,
    _normalize_pipeline_input_gradients,
    _tuple_to_dict_helper,
    register_indexcache_pipeline_adapter,
)

_EMPTY_CONFIG = SimpleNamespace(
    index_topk_pattern=None,
    indexcache_train_debug=False,
)
_ADAPTER_CONFIG = SimpleNamespace(
    index_topk_pattern="FS",
    indexcache_train_debug=False,
)
_EMPTY_PATTERN_REGISTRATION = register_indexcache_pipeline_adapter(
    _EMPTY_CONFIG
)
_FIRST_REGISTRATION = register_indexcache_pipeline_adapter(_ADAPTER_CONFIG)
_SECOND_REGISTRATION = register_indexcache_pipeline_adapter(_ADAPTER_CONFIG)


def _make_distill_state(offset):
    return (
        paddle.to_tensor([offset], dtype="int64"),
        paddle.zeros([1]),
        paddle.zeros([1]),
        paddle.zeros([1]),
        paddle.to_tensor([offset], dtype="int64"),
        paddle.to_tensor([0.5 + offset], stop_gradient=False),
        paddle.to_tensor([offset], dtype="int64"),
        paddle.to_tensor([0], dtype="int64"),
    )


class TestIndexCachePipelineBackward(unittest.TestCase):
    def test_adapter_registration_is_explicit_and_idempotent(self):
        self.assertFalse(_EMPTY_PATTERN_REGISTRATION)
        self.assertTrue(_FIRST_REGISTRATION)
        self.assertFalse(_SECOND_REGISTRATION)
        self.assertIs(
            pipeline_parallel.dict_to_tuple_helper,
            _dict_to_tuple_helper,
        )
        self.assertIs(
            pipeline_parallel.tuple_to_dict_helper,
            _tuple_to_dict_helper,
        )

    def test_clear_dataptr_clone_uses_zero_allocation_alias_gradient(self):
        class ForbiddenAllocatingClone:
            @staticmethod
            def apply(_value):
                raise AssertionError("allocating FakeClone must not run")

        value = paddle.to_tensor([1.0, 2.0], stop_gradient=False)
        cloned = _clone_and_clear_dataptr(
            value,
            ForbiddenAllocatingClone,
            clear_dataptr=True,
        )
        paddle.autograd.backward(cloned, paddle.ones_like(value))

        self.assertTrue(
            paddle.equal_all(value.grad, paddle.ones_like(value)).item()
        )

    def test_state5_debug_marker_reports_finite_and_nonzero(self):
        cases = (
            ("normal", paddle.to_tensor([0.0, 2.0]), True, True),
            ("zero", paddle.zeros([2]), True, False),
            ("nan", paddle.to_tensor([float("nan")]), False, False),
            ("missing", None, False, False),
        )
        for name, grad, finite, nonzero in cases:
            with self.subTest(name=name):
                output = StringIO()
                with (
                    patch.object(
                        _ADAPTER_CONFIG, "indexcache_train_debug", True
                    ),
                    redirect_stdout(output),
                ):
                    _debug_state5_gradient(
                        "indexcache_state 5", grad, "test", 2
                    )
                marker = output.getvalue()
                self.assertIn("producer_layer=2", marker)
                self.assertIn(f"grad_finite={finite}", marker)
                self.assertIn(f"grad_nonzero={nonzero}", marker)

    def test_pipeline_metadata_recovers_state5_producer_layer(self):
        with patch.object(_ADAPTER_CONFIG, "indexcache_train_debug", True):
            pipeline_inputs = _dict_to_tuple_helper(
                {
                    "hidden_states": paddle.to_tensor(
                        [4.0], stop_gradient=False
                    ),
                    "indexcache_state": _make_distill_state(7),
                }
            )
            _tuple_to_dict_helper(pipeline_inputs)
            self.assertEqual(_indexcache_producer_layer(pipeline_inputs), 7)

    def test_pipeline_metadata_avoids_item_after_dataptr_clear(self):
        with patch.object(_ADAPTER_CONFIG, "indexcache_train_debug", True):
            pipeline_inputs = _dict_to_tuple_helper(
                {
                    "hidden_states": paddle.to_tensor(
                        [4.0], stop_gradient=False
                    ),
                    "indexcache_state": _make_distill_state(7),
                }
            )
            state5 = next(
                tensor
                for tensor in pipeline_inputs
                if _get_pipeline_key(tensor) == "indexcache_state 5"
            )
            producer_tensor = next(
                tensor
                for tensor in pipeline_inputs
                if _get_pipeline_key(tensor) == "indexcache_state 6"
            )
            delattr(state5, "_paddlefleet_indexcache_producer_layer")
            _tuple_to_dict_helper(pipeline_inputs)
            producer_tensor._clear_dataptr()

            output = StringIO()
            with redirect_stdout(output):
                gradients = _normalize_pipeline_input_gradients(
                    pipeline_inputs,
                    (paddle.ones([1]), paddle.ones([1])),
                )

            self.assertEqual(len(gradients), 2)
            self.assertEqual(_indexcache_producer_layer(pipeline_inputs), 7)
            self.assertIn("producer_layer=7", output.getvalue())

    def test_missing_metadata_skips_released_producer_tensor(self):
        with patch.object(_ADAPTER_CONFIG, "indexcache_train_debug", True):
            state = _make_distill_state(7)
            state[6]._clear_dataptr()
            pipeline_inputs = _dict_to_tuple_helper({"indexcache_state": state})

            self.assertIsNone(_indexcache_producer_layer(pipeline_inputs))

    def test_producer_metadata_survives_detach_and_clone(self):
        class SimpleClone:
            @staticmethod
            def apply(value):
                return value.clone()

        with patch.object(_ADAPTER_CONFIG, "indexcache_train_debug", True):
            pipeline_inputs = _dict_to_tuple_helper(
                {"indexcache_state": _make_distill_state(7)}
            )
            state5 = next(
                tensor
                for tensor in pipeline_inputs
                if _get_pipeline_key(tensor) == "indexcache_state 5"
            )
            detached = _detach_and_requires_grad(state5)
            cloned = _clone_and_clear_dataptr(state5, SimpleClone)

            self.assertEqual(_indexcache_producer_layer((detached,)), 7)
            self.assertEqual(_indexcache_producer_layer((cloned,)), 7)

    def test_fresh_state_dict_captures_metadata_before_detach(self):
        with patch.object(_ADAPTER_CONFIG, "indexcache_train_debug", True):
            inputs = {"indexcache_state": _make_distill_state(7)}
            detached = _detach_and_requires_grad(inputs)
            detached_state = detached["indexcache_state"]
            detached_state[6]._clear_dataptr()

            self.assertEqual(
                _indexcache_producer_layer((detached_state[5],)), 7
            )

    def test_fresh_state_dict_captures_metadata_before_alias_clear(self):
        class ForbiddenAllocatingClone:
            @staticmethod
            def apply(_value):
                raise AssertionError("allocating FakeClone must not run")

        with patch.object(_ADAPTER_CONFIG, "indexcache_train_debug", True):
            cloned = _clone_and_clear_dataptr(
                {"indexcache_state": _make_distill_state(7)},
                ForbiddenAllocatingClone,
                clear_dataptr=True,
            )
            cloned_state = cloned["indexcache_state"]

            self.assertEqual(_indexcache_producer_layer((cloned_state[5],)), 7)

    def test_schedule_node_marker_uses_captured_producer_metadata(self):
        node = ScheduleNode(
            lambda inputs: {
                "score": inputs["indexcache_state"][5] * 2,
            },
            name="indexcache_state5_gradient",
        )
        output = StringIO()
        with (
            patch.object(_ADAPTER_CONFIG, "indexcache_train_debug", True),
            redirect_stdout(output),
        ):
            outputs = node.forward({"indexcache_state": _make_distill_state(7)})
            node.backward(paddle.ones_like(outputs["score"]))

        marker = output.getvalue()
        self.assertIn("source=schedule_node", marker)
        self.assertIn("producer_layer=7", marker)
        self.assertIn("grad_finite=True", marker)
        self.assertIn("grad_nonzero=True", marker)

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

        self.assertEqual(len(input_grads), 2)
        self.assertTrue(
            all(isinstance(grad, paddle.Tensor) for grad in input_grads)
        )
        self.assertTrue(all(not grad.stop_gradient for grad in input_grads))
        self.assertEqual(input_grads[0].item(), 2.0)
        self.assertEqual(input_grads[1].item(), 0.0)

        meta = SendRecvMeta()
        meta.set_send_message(input_grads)
        self.assertEqual(len(meta.send_shape_message), 2)

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
                "hidden_states": paddle.to_tensor([4.0], stop_gradient=False),
                "indexcache_state": _make_distill_state(0),
            }
        )
        converted_inputs, use_dict = _tuple_to_dict_helper(pipeline_inputs)

        self.assertTrue(use_dict)
        self.assertIn("indexcache_state", converted_inputs)
        self.assertTrue(
            all(not hasattr(tensor, "key") for tensor in pipeline_inputs)
        )

        hidden_grad = paddle.ones_like(pipeline_inputs[0])
        hidden_grad.stop_gradient = False
        input_grads = _normalize_pipeline_input_gradients(
            pipeline_inputs,
            (hidden_grad, None),
        )

        self.assertEqual(len(input_grads), 2)
        self.assertTrue(
            all(isinstance(grad, paddle.Tensor) for grad in input_grads)
        )
        self.assertTrue(all(not grad.stop_gradient for grad in input_grads))
        self.assertEqual(input_grads[0].item(), 1.0)
        self.assertEqual(input_grads[1].item(), 0.0)

        meta = SendRecvMeta()
        meta.set_send_message(input_grads)
        self.assertEqual(len(meta.send_shape_message), 2)

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
                "hidden_states": paddle.to_tensor([4.0], stop_gradient=False),
                "indexcache_state": _make_distill_state(0),
            }
        )
        _tuple_to_dict_helper(pipeline_inputs)
        released_inputs = [
            tensor
            for tensor in pipeline_inputs
            if _get_pipeline_key(tensor)
            in {
                "indexcache_state 5",
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
            (hidden_grad, None),
        )

        self.assertEqual(len(released_inputs), 1)
        self.assertEqual(len(input_grads), 2)
        for grad, (shape, dtype) in zip(input_grads[1:], expected_metadata):
            self.assertEqual(list(grad.shape), shape)
            self.assertEqual(grad.dtype, dtype)
            self.assertFalse(grad.stop_gradient)
            self.assertEqual(float(grad.sum().item()), 0.0)

        meta = SendRecvMeta()
        meta.set_send_message(input_grads)
        self.assertEqual(len(meta.send_shape_message), 2)

    def test_outer_pipeline_boundary_rejects_missing_hidden_gradient(self):
        pipeline_inputs = _dict_to_tuple_helper(
            {
                "hidden_states": paddle.to_tensor([4.0], stop_gradient=False),
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
                (None, None),
            )


if __name__ == "__main__":
    unittest.main()

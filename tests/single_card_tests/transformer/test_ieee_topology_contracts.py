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

"""Exercise production topology dispatch with native tensors and explicit groups."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import paddle

ROOT = Path(__file__).resolve().parents[3] / "src/paddlefleet"


def load(path, names, namespace):
    source = ROOT / path
    tree = ast.parse(source.read_text())
    nodes = []
    for name in names:
        if "." in name:
            owner, method = name.split(".")
            cls = next(
                n
                for n in tree.body
                if isinstance(n, ast.ClassDef) and n.name == owner
            )
            nodes.append(
                next(
                    n
                    for n in cls.body
                    if isinstance(n, ast.FunctionDef) and n.name == method
                )
            )
        else:
            nodes.append(
                next(n for n in tree.body if getattr(n, "name", None) == name)
            )
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[future, *nodes], type_ignores=[])
    )
    exec(compile(module, str(source), "exec"), namespace)
    return namespace


class TestIEEETopologyContracts(unittest.TestCase):
    def test_mtp_uses_own_indexer_only_for_ieee_tp2(self):
        state = SimpleNamespace(tp=1, ieee=True)
        ns = load(
            "transformer/dsa_attention.py",
            [
                "_ieee_tp1",
                "is_dsa_skip_topk_layer",
                "source_dsa_compute_layer",
                "decoder_dsa_logical_layer",
                "decoder_dsa_topk_producer_layer",
                "_decoder_layer_publishes_shared_topk",
                "resolve_dsa_indexer_layout",
            ],
            {
                "ieee_kernel_enabled": lambda: state.ieee,
                "parallel_state": SimpleNamespace(
                    get_tensor_model_parallel_world_size=lambda: state.tp
                ),
            },
        )
        config = SimpleNamespace(
            use_accuracy_compatible=True,
            num_hidden_layers=4,
            dsa_index_share_for_mtp_iteration=True,
            dsa_indexer_types=["full", "full", "full", "shared"],
        )
        for ieee, tp, expected in [
            (True, 1, ("shared", True, True, 2)),
            (True, 2, ("full", False, False, 4)),
            (False, 1, ("shared", True, True, 2)),
            (False, 2, ("shared", True, True, 2)),
        ]:
            state.ieee, state.tp = ieee, tp
            with self.subTest(ieee=ieee, tp=tp):
                self.assertEqual(
                    ns["resolve_dsa_indexer_layout"](config, 4, True), expected
                )
        state.ieee, state.tp = True, 2
        config.use_accuracy_compatible = False
        self.assertEqual(
            ns["resolve_dsa_indexer_layout"](config, 4, True),
            ("shared", True, True, 2),
        )

    def test_router_topk_keeps_tp1_take_along_axis_gradient(self):
        state = SimpleNamespace(ieee=True)
        ns = load(
            "transformer/moe/moe_router.py",
            ["StandardMoERouter._topk_noaux_tc"],
            {
                "paddle": paddle,
                "ieee_kernel_enabled": lambda: state.ieee,
            },
        )
        router = SimpleNamespace(
            config=SimpleNamespace(gpt_model_use_experimental_version=False),
            e_score_correction_bias=paddle.arange(16, dtype="float32") * 0.01,
            use_accuracy_compatible=True,
            tensor_model_parallel_size=1,
        )
        paddle.seed(514)
        original = paddle.randn([4, 16])
        for ieee, tp, calls in [(True, 1, 0), (True, 2, 1), (False, 1, 1)]:
            state.ieee = ieee
            router.tensor_model_parallel_size = tp
            scores = original.detach().clone()
            scores.stop_gradient = False
            with patch.object(
                paddle, "gather_nd", wraps=paddle.gather_nd
            ) as gather:
                weights, indices = ns["_topk_noaux_tc"](router, scores, 4, 1, 1)
                self.assertEqual(gather.call_count, calls)
            reference = original.detach().clone()
            reference.stop_gradient = False
            expected = reference.take_along_axis(indices, axis=1)
            dy = paddle.arange(16, dtype="float32").reshape([4, 4])
            weights.backward(dy)
            expected.backward(dy)
            self.assertEqual(
                weights.numpy().tobytes(), expected.numpy().tobytes()
            )
            self.assertEqual(
                scores.grad.numpy().tobytes(), reference.grad.numpy().tobytes()
            )

    def test_zero_coefficient_indexer_graph_and_input_edges(self):
        state = SimpleNamespace(ieee=True, tp=1)
        indices = paddle.zeros([1, 4, 1], dtype="int64")
        loss = SimpleNamespace(
            apply=Mock(return_value=paddle.zeros([], dtype="float32")),
            _last_topk_indices=indices,
        )
        scaler = SimpleNamespace(apply=Mock(side_effect=lambda out, loss: out))
        ns = load(
            "transformer/dsa_attention.py",
            ["_ieee_tp1", "DSAttention.forward"],
            {
                "paddle": paddle,
                "ieee_kernel_enabled": lambda: state.ieee,
                "parallel_state": SimpleNamespace(
                    get_tensor_model_parallel_world_size=lambda: state.tp
                ),
                "FusedDSAIndexerLoss": loss,
                "DSAIndexerLossAutoScaler": scaler,
                "_unfused_dsa_attention": lambda q, k, v, mask, scale: v,
            },
        )
        x = paddle.ones([1, 4, 8], dtype="bfloat16")
        qr = paddle.ones([1, 4, 4], dtype="bfloat16")
        x.stop_gradient = qr.stop_gradient = False
        q = paddle.ones([1, 4, 2, 4], dtype="bfloat16")
        for ieee, tp, fused_calls, attach_calls in [
            (True, 1, 1, 0),
            (True, 2, 1, 1),
            (False, 1, 0, 0),
        ]:
            state.ieee, state.tp = ieee, tp
            indexer = SimpleNamespace(
                forward_before_topk=Mock(return_value=(x, qr, qr)),
                forward=Mock(return_value=(None, indices)),
                index_topk=1,
            )
            layer = SimpleNamespace(
                config=SimpleNamespace(sequence_parallel=False),
                pg_collection=SimpleNamespace(tp=None),
                index_share=False,
                skip_topk=False,
                training=True,
                dsa_indexer_loss_coeff=0.0,
                ieee_indexer_loss=ieee,
                dsa_indexer_use_sparse_loss=False,
                indexer=indexer,
                softmax_scale=0.5,
            )
            loss.apply.reset_mock()
            scaler.apply.reset_mock()
            ns["forward"](layer, q, q, q, None, x=x, qr=qr)
            self.assertEqual(loss.apply.call_count, fused_calls)
            self.assertEqual(scaler.apply.call_count, attach_calls)
            call = (
                indexer.forward_before_topk if fused_calls else indexer.forward
            )
            consumed_x, consumed_qr = call.call_args.args[:2]
            if ieee and tp == 1:
                self.assertIs(consumed_x, x)
                self.assertIs(consumed_qr, qr)
            else:
                self.assertIsNot(consumed_x, x)
                self.assertIsNot(consumed_qr, qr)

    def test_tp1_gather_preserves_tensor_identity_only_in_ieee_mode(self):
        state = SimpleNamespace(ieee=True)
        gather = Mock(return_value=object())
        ns = load(
            "tensor_parallel/mappings.py",
            ["gather_from_tensor_model_parallel_region"],
            {
                "ieee_kernel_enabled": lambda: state.ieee,
                "get_tensor_model_parallel_group_if_none": lambda group: group,
                "_GatherFromModelParallelRegion": SimpleNamespace(apply=gather),
            },
        )
        x = paddle.ones([2, 4])
        fn = ns["gather_from_tensor_model_parallel_region"]
        for group in (None, SimpleNamespace(nranks=1)):
            self.assertIs(fn(x, group), x)
        gather.assert_not_called()
        fn(x, SimpleNamespace(nranks=2))
        self.assertEqual(gather.call_count, 1)
        state.ieee = False
        fn(x, SimpleNamespace(nranks=1))
        self.assertEqual(gather.call_count, 2)

    def test_tp1_linear_uses_native_forward_and_backward(self):
        state = SimpleNamespace(ieee=True)
        communication = Mock(return_value=object())
        ns = load(
            "tensor_parallel/layers.py",
            [
                "general_gemm",
                "linear_with_grad_accumulation_and_async_allreduce",
            ],
            {
                "paddle": paddle,
                "ieee_kernel_enabled": lambda: state.ieee,
                "get_tensor_model_parallel_group_if_none": lambda group: group,
                "get_pg_size": lambda group: (
                    1 if group is None else group.nranks
                ),
                "LinearWithGradAccumulationAndAsyncCommunication": SimpleNamespace(
                    apply=communication
                ),
            },
        )
        fn = ns["linear_with_grad_accumulation_and_async_allreduce"]
        fn.warned = True
        paddle.seed(515)
        x = paddle.randn([2, 7, 32]).cast("bfloat16")
        w = paddle.randn([32, 16]).cast("bfloat16")
        bias = paddle.randn([16]).cast("bfloat16")
        for use_bias in (False, True):
            left = [a.detach().clone() for a in (x, w, bias)]
            right = [a.detach().clone() for a in (x, w, bias)]
            for a in left + right:
                a.stop_gradient = False
            y = fn(
                left[0],
                left[1],
                left[2] if use_bias else None,
                False,
                False,
                False,
                use_accuracy_compatible=True,
            )
            expected = paddle.nn.functional.linear(
                right[0], right[1], right[2] if use_bias else None
            )
            dy = paddle.randn(y.shape).cast("bfloat16")
            y.backward(dy)
            expected.backward(dy)
            self.assertEqual(
                y.cast("float32").numpy().tobytes(),
                expected.cast("float32").numpy().tobytes(),
            )
            for a, b in zip(
                left[: 3 if use_bias else 2], right[: 3 if use_bias else 2]
            ):
                self.assertEqual(
                    a.grad.cast("float32").numpy().tobytes(),
                    b.grad.cast("float32").numpy().tobytes(),
                )
        communication.assert_not_called()
        for overrides in [
            {"tp_group": SimpleNamespace(nranks=2)},
            {"sequence_parallel": True},
            {"allreduce_dgrad": True},
            {"gradient_accumulation_fusion": True},
            {"grad_output_buffer": []},
        ]:
            args = {
                "gradient_accumulation_fusion": False,
                "allreduce_dgrad": False,
                "sequence_parallel": False,
                "use_accuracy_compatible": True,
            }
            args.update(overrides)
            fn(x, w, None, **args)
        self.assertEqual(communication.call_count, 5)
        state.ieee = False
        fn(x, w, None, False, False, False, use_accuracy_compatible=True)
        self.assertEqual(communication.call_count, 6)

    def test_moe_does_not_stack_input_fanouts(self):
        class ReachedExperts(Exception):
            pass

        primary = SimpleNamespace(
            apply=Mock(side_effect=lambda x: (x.clone(), x.clone(), x.clone()))
        )
        fallback = SimpleNamespace(
            apply=Mock(side_effect=lambda x: (x.clone(), x.clone(), x.clone()))
        )
        ns = load(
            "transformer/moe/moe_layer.py",
            ["MoELayer.forward"],
            {
                "paddle": paddle,
                "ieee_kernel_enabled": lambda: True,
                "ThreePathCloneAlignMG": primary,
                "_AccuracyCompatibleMoEInputBranches": fallback,
                "inspect_tensor_set_current_layer": lambda *args: None,
                "_log_moe_md5": lambda *args: None,
            },
        )
        layer = SimpleNamespace(
            expert_model_parallel_size=1,
            sequence_parallel=False,
            shared_experts=object(),
            use_accuracy_compatible=True,
            _supports_three_path_clone=lambda: True,
            _maybe_pre_allgather_overlap=Mock(side_effect=ReachedExperts),
        )
        x = paddle.ones([2, 4])
        x.stop_gradient = False
        with self.assertRaises(ReachedExperts):
            ns["forward"](layer, x)
        primary.apply.assert_called_once()
        fallback.apply.assert_not_called()
        layer.use_accuracy_compatible = False
        with self.assertRaises(ReachedExperts):
            ns["forward"](layer, x)
        fallback.apply.assert_called_once()


if __name__ == "__main__":
    paddle.set_device("gpu:0")
    unittest.main()

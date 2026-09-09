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

"""Exercise the actual constructor selection without creating a model or group."""

import ast
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


class TestIEEEExpertDispatchSelection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = (
            Path(__file__).resolve().parents[3]
            / "src/paddlefleet/transformer/moe/moe_layer.py"
        )
        tree = ast.parse(source.read_text())
        moe = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "MoELayer"
        )
        init = next(
            n
            for n in moe.body
            if isinstance(n, ast.FunctionDef) and n.name == "__init__"
        )
        choices = [
            n
            for n in init.body
            if isinstance(n, ast.If)
            and any(
                isinstance(x, ast.Attribute)
                and x.attr == "use_accuracy_compatible"
                for x in ast.walk(n.test)
            )
            and len(n.body) == 1
            and isinstance(n.body[0], ast.Assign)
            and isinstance(n.body[0].value, ast.Constant)
            and n.body[0].value.value == "alltoall"
        ]
        assert len(choices) == 1
        cls.code = compile(
            ast.Module(body=choices, type_ignores=[]), str(source), "exec"
        )

    def test_constructor_respects_actual_group_and_default_off_contract(self):
        cases = [
            # compatible, IEEE, fused, actual EP, requested, expected
            (True, True, True, 2, "deepep", "deepep"),
            (True, False, True, 2, "deepep", "deepep"),
            (True, True, False, 2, "deepep", "alltoall"),
            (True, True, True, 1, "deepep", "alltoall"),
            (True, True, True, None, "deepep", "alltoall"),
            (True, True, True, 2, "alltoall", "alltoall"),
            (True, True, True, 2, "hybridep", "alltoall"),
            (False, False, True, 2, "hybridep", "hybridep"),
        ]
        for compatible, ieee, fused, ep, requested, expected in cases:
            with self.subTest(case=(compatible, ieee, fused, ep, requested)):
                instance = SimpleNamespace(
                    use_accuracy_compatible=compatible,
                    moe_token_dispatcher_type=requested,
                )
                namespace = {
                    "self": instance,
                    # A declared EP2 must not override an actual local/absent group.
                    "config": SimpleNamespace(
                        moe_expert_fusion=fused,
                        expert_model_parallel_size=2,
                        use_accuracy_compatible=compatible,
                    ),
                    "pg_collection": SimpleNamespace(
                        ep=None if ep is None else SimpleNamespace(nranks=ep)
                    ),
                    "utils": SimpleNamespace(
                        get_pg_size=lambda group: group.nranks
                    ),
                    "ieee_kernel_enabled": lambda: ieee,
                }
                instance.config = namespace["config"]
                exec(self.code, namespace)
                self.assertEqual(instance.moe_token_dispatcher_type, expected)


class TestIEEEFusionForwardSelection(unittest.TestCase):
    def test_actual_forward_routes_fused_dispatch_and_preserves_defaults(self):
        source = (
            Path(__file__).resolve().parents[3]
            / "src/paddlefleet/transformer/moe/moe_layer.py"
        )
        tree = ast.parse(source.read_text())
        moe = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "MoELayer"
        )
        method = next(
            n
            for n in moe.body
            if isinstance(n, ast.FunctionDef) and n.name == "fusion_moe_forward"
        )
        for arg in method.args.args:
            arg.annotation = None
        method.returns = None
        code = compile(
            ast.Module(body=[method], type_ignores=[]), str(source), "exec"
        )
        cases = [
            # compatibility, IEEE, fusion, backend, EP, MTP, selected path
            (True, True, True, "deepep", 2, False, "fused"),
            (True, True, True, "deepep", 2, True, "fused"),
            (True, False, True, "deepep", 2, True, "fused"),
            (True, True, False, "deepep", 2, True, "custom"),
            (True, True, True, "alltoall", 2, True, "custom"),
            (True, True, True, "deepep", 1, True, "custom"),
            (False, True, True, "deepep", 2, True, "ordinary"),
        ]
        for compatible, ieee, fused, backend, ep, mtp, path in cases:
            with self.subTest(case=(compatible, ieee, fused, backend, ep, mtp)):
                hs = Mock(dtype="bfloat16")
                fp32, recast, fusion_output, cloned = (
                    Mock(),
                    Mock(),
                    Mock(),
                    Mock(),
                )
                hs.cast.return_value = fp32
                fp32.cast.return_value = recast
                fusion_output.clone.return_value = cloned
                instance = SimpleNamespace(
                    use_accuracy_compatible=compatible,
                    moe_expert_fusion=fused,
                    moe_token_dispatcher_type=backend,
                    expert_model_parallel_size=ep,
                    is_mtp_layer=mtp,
                    custom_forward=Mock(return_value="custom-result"),
                    _project_to_latent=Mock(side_effect=lambda value: value),
                    dispatch=Mock(return_value=("dispatched", None)),
                    token_dispatcher=SimpleNamespace(
                        get_dispatched_routing=lambda: ("indices", "probs", [1])
                    ),
                    fp8_dispatch_bwd=False,
                    _use_hybrid_ep_fusion=lambda: False,
                    using_sonic_moe=False,
                    combine=Mock(return_value="combined"),
                    use_latent_moe=False,
                    num_experts_per_tok=2,
                    config=SimpleNamespace(
                        activation_func_clamp_value=None,
                        use_accuracy_compatible=compatible,
                    ),
                )
                for key in (
                    "fp8",
                    "moe_deep_gemm",
                    "recompute_moe_gate_up",
                    "recompute_moe_premute",
                    "fp8_wgrad",
                    "use_auto_subbatch",
                    "auto_subbatch_mode",
                    "moe_subbatch_token_num_after_dispatch",
                    "moe_subbatch_diag",
                    "use_ue8m0",
                    "defer_expert_up_gate_dw",
                    "defer_expert_down_dw",
                    "use_w4a8",
                    "use_w4a8_fused_quant",
                ):
                    setattr(instance, key, False)
                fusion = Mock(return_value=fusion_output)
                namespace = {
                    "ieee_kernel_enabled": lambda: ieee,
                    "inspect_tensor": lambda name, layer, value, **kwargs: (
                        value
                    ),
                    "framework": SimpleNamespace(
                        _dygraph_tracer=lambda: SimpleNamespace(_has_grad=True)
                    ),
                    "profile": lambda name: nullcontext(),
                    "global_moe_balance_training_logs_enabled": lambda: False,
                    "FusionMoePyLayer": SimpleNamespace(apply=fusion),
                }
                exec(code, namespace)
                overlap = {"fn": Mock(return_value=("shared",)), "fn_args": ()}
                result = namespace["fusion_moe_forward"](
                    instance,
                    hs,
                    "dense-probs",
                    "routing",
                    overlap,
                    "topk-weights",
                    "topk-indices",
                )
                if path == "custom":
                    self.assertEqual(result, "custom-result")
                    instance.dispatch.assert_not_called()
                    self.assertEqual(overlap["fn_out"], ("shared",))
                    continue
                self.assertEqual(result, "combined")
                instance.custom_forward.assert_not_called()
                overlap["fn"].assert_not_called()
                dispatch_hs = recast if path == "fused" and mtp else hs
                operands = (
                    (None, None)
                    if path == "fused"
                    else ("topk-weights", "topk-indices")
                )
                instance.dispatch.assert_called_once_with(
                    dispatch_hs, "dense-probs", "routing", *operands
                )
                fusion.assert_called_once()
                self.assertIs(
                    instance.combine.call_args.args[0],
                    cloned if path == "fused" else fusion_output,
                )
                if path == "fused" and mtp:
                    hs.cast.assert_called_once_with("float32")
                    fp32.cast.assert_called_once_with("bfloat16")
                else:
                    hs.cast.assert_not_called()


if __name__ == "__main__":
    unittest.main()

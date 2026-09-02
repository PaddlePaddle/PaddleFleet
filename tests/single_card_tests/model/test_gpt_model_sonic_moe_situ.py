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

"""Precision of the SonicMoE fp8 SiTU-GLU expert path.

The reference is the *non*-SonicMoE SiTU path (``GroupedMLPExpert``, bf16,
``moe_deep_gemm=False``), which evaluates ``situ_glu`` in fp32 outside the
GEMM. The tolerance is not invented: the same measurement is taken for
SwiGLU, whose fp8 SonicMoE epilogue is already validated, and SiTU is
required to stay within a small multiple of it. That way the gate tracks
the fp8 rounding floor of whatever hardware runs it instead of encoding
one machine's numbers.

Why the reference is bf16 rather than the fp8 non-SonicMoE path: on a
single card, ``fp8`` + ``moe_deep_gemm=True`` produces an output that does
not depend on the activation function nor on fp8 at all (the batched
DeepGEMM bug that ``test_gpt_model_sonic_moe.py`` also works around), and
``fp8`` + ``moe_deep_gemm=False`` disables weight fusion and never builds
``grouped_gemm_experts``. Neither is a usable numeric oracle here; the fp8
standalone-op path is only exercised end to end with EP > 1.
"""

import unittest

import paddle
import paddle.nn.functional as F
import paddlefleet_ops
from paddle.distributed import fleet
from paddle.distributed.fleet.utils import mix_precision_utils

import paddlefleet.parallel_state as ps
from paddlefleet.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from paddlefleet.process_groups_config import ProcessGroupCollection
from paddlefleet.tensor_parallel.random import model_parallel_cuda_manual_seed
from paddlefleet.transformer.activations import situ
from paddlefleet.transformer.moe.moe_layer import MoELayer
from paddlefleet.transformer.transformer_config import TransformerConfig

if paddlefleet_ops.is_sonic_moe_available():
    from paddlefleet_ops.sonicmoe.functional import (
        clear_all_fp8_weight_caches,
    )


def situ_epilogue_available():
    """Whether the installed paddlefleet_ops ships the SiTU-GLU epilogue.

    ``SonicMoEExpert.__init__`` imports ``encode_situ_activation`` lazily, so
    an older build fails there with ImportError by design rather than running
    SwiGLU numerics under a SiTU config.
    """
    if not paddlefleet_ops.is_sonic_moe_available():
        return False
    try:
        from paddlefleet_ops.sonicmoe.quack_utils.activation_situ import (  # noqa: F401
            encode_situ_activation,
        )
    except ImportError:
        return False
    return True


def calc_diff(x: paddle.Tensor, y: paddle.Tensor):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    if denominator.item() == 0:
        return 0.0
    sim = 2 * (x * y).sum() / denominator
    return (1 - sim).item()


# ── Module-level fleet initialization (only once) ─────────────────────
_strategy = fleet.DistributedStrategy()
_strategy.hybrid_configs = {
    "dp_degree": 1,
    "mp_degree": 1,
    "pp_degree": 1,
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
fleet.init(is_collective=True, strategy=_strategy)
_hcg = fleet.get_hybrid_communicate_group()
ps.initialize_model_parallel(_hcg)


@unittest.skipUnless(
    paddlefleet_ops.is_sonic_moe_available(),
    "Sonic-MoE not available (requires Python>=3.12, CUDA>=12.9, SM>=90)",
)
class TestSonicMoESituPrecision(unittest.TestCase):
    """fp8 SonicMoE SiTU-GLU vs the non-SonicMoE SiTU reference."""

    # Absolute ceiling, shared with test_gpt_model_sonic_moe.py.
    fp8_tol = 5e-3
    # SiTU is allowed this multiple of the measured SwiGLU fp8 error. SiTU-GLU
    # is SwiGLU's shape with two extra tanh factors, both with derivative
    # bounded by 1, so the fp8 rounding error may not grow by more than a
    # small constant. 4x leaves room for the up-branch clamp and the beta
    # scaling while still catching an order-of-magnitude regression.
    situ_over_swiglu_ratio = 4.0

    def setUp(self):
        self.pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        self.seed = 46
        self.hidden_size = 512
        self.n_routed_experts = 8

    @staticmethod
    def _small_init_method(tensor):
        paddle.nn.initializer.Uniform(-0.001, 0.001)(tensor)

    def _build_transformer_config(self, act, using_sonic_moe, fp8):
        kwargs = {
            "hidden_size": self.hidden_size,
            "num_attention_heads": 4,
            "n_routed_experts": self.n_routed_experts,
            "use_cpu_initialization": False,
            "num_experts_per_tok": 2,
            "tensor_model_parallel_size": 1,
            "expert_model_parallel_size": 1,
            "sequence_parallel": False,
            "bf16": True,
            "params_dtype": paddle.bfloat16,
            "moe_intermediate_size": 1024,
            "gated_linear_unit": True,
            "n_shared_experts": 0,
            "hidden_act": situ if act == "situ" else F.silu,
            "moe_expert_fusion": True,
            "bias_activation_fusion": True,
            "moe_token_dispatcher_type": "alltoall",
            "moe_use_fusion_node": True,
            "using_sonic_moe": using_sonic_moe,
            "fp8": fp8,
            # SiTU + fp8 rejects fp8_wgrad, so both activations run with bf16
            # weight gradients to keep the comparison apples to apples.
            "fp8_wgrad": False,
            "init_method": self._small_init_method,
            "output_layer_init_method": self._small_init_method,
        }
        if act == "situ":
            kwargs["activation_situ_beta"] = 4.0
            kwargs["activation_situ_linear_beta"] = 25.0
        if not using_sonic_moe:
            kwargs["moe_deep_gemm"] = False
        return TransformerConfig(**kwargs)

    def _build_moe_layer(self, act, using_sonic_moe=False, fp8=None):
        # Same seed for every layer: SonicMoEExpert reuses GroupedMLPExpert
        # initialization, so expert weights end up identical in both paths.
        paddle.seed(self.seed)
        model_parallel_cuda_manual_seed(self.seed)
        config = self._build_transformer_config(act, using_sonic_moe, fp8)
        transformer_layer_spec = get_gpt_layer_local_spec(
            config,
            num_experts=self.n_routed_experts,
        )
        moe_layer = MoELayer(
            config,
            transformer_layer_spec.sublayers_spec.mlp.extra_kwargs["sublayers"],
            self.pg_collection,
        )
        mix_precision_utils.MixPrecisionLayer(moe_layer, dtype="bfloat16")
        for param in moe_layer.parameters():
            if hasattr(param, "main_grad") and param.main_grad is None:
                param.main_grad = paddle.zeros_like(param, dtype=paddle.float32)
        return moe_layer

    @staticmethod
    def _collect_grads(layer):
        grads = {}
        for name, param in layer.named_parameters():
            grad = getattr(param, "main_grad", None)
            if grad is None:
                grad = param.grad
            if grad is not None:
                grads[name] = grad.detach().clone()
        return grads

    def _forward_backward(self, moe_layer, input_data):
        hidden_states = input_data.detach().clone()
        hidden_states.stop_gradient = False
        with paddle.amp.auto_cast(level="O2", dtype="bfloat16"):
            output = moe_layer(hidden_states)[0]
            loss = output.sum()
        loss.backward()
        expert = getattr(moe_layer, "grouped_gemm_experts", None)
        if hasattr(expert, "flush_to_grouped_layout"):
            expert.flush_to_grouped_layout()
        return (
            loss.item(),
            output.detach().clone(),
            self._collect_grads(moe_layer),
        )

    def _input_data(self):
        paddle.seed(self.seed)
        return paddle.randn([2, 64, self.hidden_size], dtype=paddle.bfloat16)

    def _measure_fp8_error(self, act):
        """Diff of the fp8 SonicMoE path against the non-SonicMoE bf16 one.

        Returns ``{"output": diff, "<param>": diff, ...}``.
        """
        input_data = self._input_data()

        reference = self._build_moe_layer(act, using_sonic_moe=False)
        loss_ref, output_ref, grads_ref = self._forward_backward(
            reference, input_data
        )

        sonic = self._build_moe_layer(act, using_sonic_moe=True, fp8="e4m3")
        sonic.grouped_gemm_experts.sonic_moe_config.enabled = True
        sonic.grouped_gemm_experts.quant_weight()
        loss_fp8, output_fp8, grads_fp8 = self._forward_backward(
            sonic, input_data
        )
        clear_all_fp8_weight_caches()

        diffs = {"output": calc_diff(output_fp8, output_ref)}
        common = set(grads_ref) & set(grads_fp8)
        self.assertTrue(common, f"no common grad tensors for act={act}")
        for name in sorted(common):
            diffs[name] = calc_diff(grads_fp8[name], grads_ref[name])
        print(
            f"[{act}] fp8 sonic vs bf16 non-sonic: "
            f"loss_ref={loss_ref:.6e} loss_fp8={loss_fp8:.6e}"
        )
        for name, diff in diffs.items():
            print(f"[{act}]   {name}: {diff:.6e}")
        return diffs

    def test_swiglu_fp8_error_is_within_tolerance(self):
        """Calibration baseline, and a self-test of the harness above.

        The SiTU test below is a ratio against these numbers, so they have to
        be bounded on their own: a SwiGLU regression would otherwise let SiTU
        pass vacuously.
        """
        for name, diff in self._measure_fp8_error("swiglu").items():
            with self.subTest(tensor=name):
                self.assertLess(diff, self.fp8_tol, f"{name} diff {diff:.6e}")

    @unittest.skipUnless(
        situ_epilogue_available(),
        "installed paddlefleet_ops has no SiTU-GLU fp8 epilogue",
    )
    def test_situ_fp8_error_not_worse_than_swiglu(self):
        """SiTU on SonicMoE must not be noisier than SwiGLU on SonicMoE.

        Both are measured against their own non-SonicMoE bf16 SiTU/SwiGLU
        reference, so the ratio isolates the epilogue's fp8 behaviour from
        the activation's own conditioning.
        """
        swiglu = self._measure_fp8_error("swiglu")
        situ_diffs = self._measure_fp8_error("situ")

        for name, diff in situ_diffs.items():
            with self.subTest(tensor=name):
                self.assertLess(
                    diff, self.fp8_tol, f"{name} situ diff {diff:.6e}"
                )
                # Floor so a degenerate zero denominator cannot turn the
                # budget into "must match bit-exactly".
                budget = self.situ_over_swiglu_ratio * max(swiglu[name], 1e-5)
                self.assertLessEqual(
                    diff,
                    budget,
                    f"{name}: situ fp8 diff {diff:.6e} exceeds "
                    f"{self.situ_over_swiglu_ratio}x the swiglu fp8 diff "
                    f"({swiglu[name]:.6e})",
                )


if __name__ == "__main__":
    unittest.main()

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
"""Single-card tests for fine-grained activation offloading on a real GPTModel.

Offloading must be numerically invisible: it only changes *where* an activation
lives between forward and backward. So the bar is bit-exact loss and gradients
versus a run with the feature off -- not "close enough". Anything looser cannot
tell a harmless floating-point reordering from a real stream/event race, and
those races are exactly what this machinery can get wrong.

Run with the repository copy of PaddleFleet, not the one in site-packages::

    PYTHONPATH=$PWD/src CUDA_VISIBLE_DEVICES=0 python -m pytest \
        tests/single_card_tests/model/test_gpt_model_activation_offload.py -v
"""

from __future__ import annotations

import functools
import random
import unittest

import numpy as np
import paddle
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import NoPipelineParallel

import paddlefleet.parallel_state as ps
from paddlefleet.activation_offload import (
    manager_from_config,
    reset_offload_manager,
)
from paddlefleet.gpt_builders import gpt_builder
from paddlefleet.models.gpt import GPTConfig

_HAS_GPU = (
    paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0
)
_REQUIRE_GPU = unittest.skipUnless(_HAS_GPU, "Requires a CUDA device")

ATTENTION_MODULES = ["attn_norm", "qkv_linear", "core_attn", "attn_proj"]


@_REQUIRE_GPU
class TestGPTModelActivationOffload(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {
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
        fleet.init(is_collective=True, strategy=strategy)
        ps.initialize_model_parallel(fleet.get_hybrid_communicate_group())

    def setUp(self):
        seed = 46
        random.seed(seed)
        np.random.seed(seed)
        paddle.manual_seed(seed)
        self.strategy = fleet.fleet._user_defined_strategy

    # ---------------- fixtures ----------------

    @staticmethod
    def _base_config(**overrides):
        kwargs = {
            "num_hidden_layers": 2,
            "hidden_size": 512,
            "vocab_size": 100,
            "max_sequence_length": 64,
            "num_attention_heads": 4,
            "moe_expert_fusion": False,
            "intermediate_size": 1024,
            "normalization": "RMSNorm",
            "hidden_dropout_prob": 0.0,
            "first_k_dense_replace": 1,
            "attention_dropout": 0.0,
            "n_routed_experts": 8,
            "use_bias": False,
            "rotary_percent": 1.0,
            "rotary_base": 10000,
            "rope_scaling": 1.0,
            "moe_intermediate_size": 1024,
            "moe_token_dispatcher_type": "alltoall",
            "n_shared_experts": 1,
            "init_method": functools.partial(
                paddle.nn.init.xavier_uniform_, gain=1.0
            ),
            "output_layer_init_method": functools.partial(
                paddle.nn.init.xavier_uniform_, gain=1.0
            ),
            "tie_word_embeddings": True,
            "use_qk_norm": True,
            "recompute_granularity": None,
            "recompute_modules": [],
            # Binding NUMA would change the CPU affinity of the whole test
            # process, which a test must not do as a side effect.
            "activation_offload_numa_bind": False,
        }
        kwargs.update(overrides)  # overrides may replace any of the above
        return GPTConfig(**kwargs)

    @classmethod
    def _offload_config(cls, modules, **kw):
        # The offloading fields have to go in as constructor arguments. GPTConfig
        # is a dataclass, so setting them afterwards would need a second
        # __post_init__ to re-validate, and __post_init__ is not idempotent: it
        # derives moe_layer_freq into a non-int, and the second run then trips
        # "Cannot specify both first_k_dense_replace and moe_layer_freq".
        return cls._base_config(
            fine_grained_activation_offloading=True,
            offload_modules=list(modules),
            # Activations at this shape are a few hundred KB, so the 2MB default
            # threshold would filter out every one of them.
            **{"min_offloaded_tensor_bytes": 1, **kw},
        )

    def _prepare_input_data(self, config):
        seq = config.max_sequence_length
        data = list(range(seq))
        ids = paddle.to_tensor(data, dtype=paddle.int64).repeat((1, 1))
        labels = paddle.to_tensor(
            list(range(1, seq + 1)), dtype=paddle.int64
        ).repeat((1, 1))
        mask = paddle.ones((1, 1, seq, seq), dtype=bool)
        return (
            {
                "input_ids": [ids],
                "position_ids": [ids],
                "attention_mask": [mask],
            },
            [labels],
        )

    # ---------------- driver ----------------

    @staticmethod
    def _manager(config, offload):
        """Build the manager from a config.

        It has to happen before the model is built: ``TransformerLayer.__init__``
        fetches the singleton itself, and it must get this one.
        ``reset_offload_manager()`` is needed because only the first call's kwargs
        apply -- without it, several configurations in one process would all run
        with the first one.
        """
        reset_offload_manager()
        mgr = manager_from_config(config)
        mgr.enabled = offload
        return mgr

    def _run(self, config, offload, mgr=None):
        """One forward+backward. Returns (loss, {param: grad}, mgr.stats)."""
        if mgr is None:
            mgr = self._manager(config, offload)
        mgr.reset_stats()
        # On a single card there are no fleet micro-step callbacks, so open the
        # group by hand; the key is arbitrary as long as it is consistent.
        if offload:
            mgr.begin_forward_group(0)

        model = gpt_builder(config, num_stages=1)
        data = self._prepare_input_data(config)
        # No manual scope here: TransformerLayer.forward opens the regions itself
        # according to the config, and that production path is what is under test.
        loss = NoPipelineParallel(
            model, self.strategy
        ).forward_backward_pipeline(data)

        grads = {
            name: param.grad.detach().clone()
            for name, param in model.named_parameters()
            if param.grad is not None
        }
        stats = dict(mgr.stats)
        if offload:
            mgr.clear_current_group()
            mgr.end_iteration()
        return loss.item(), grads, stats

    def _assert_bit_exact(self, ref, got, tag):
        ref_loss, ref_grads, _ = ref
        got_loss, got_grads, _ = got
        self.assertEqual(
            ref_loss, got_loss, f"{tag}: loss differs, {ref_loss} vs {got_loss}"
        )
        self.assertEqual(
            set(ref_grads),
            set(got_grads),
            f"{tag}: different sets of parameters received gradients",
        )
        for name, ref_grad in ref_grads.items():
            np.testing.assert_array_equal(
                ref_grad.astype("float32").numpy(),
                got_grads[name].astype("float32").numpy(),
                err_msg=f"{tag}: gradient of {name} is not bit-exact",
            )

    # ---------------- tests ----------------

    def test_attention_modules_are_bit_exact(self):
        """Every attention region on: loss and every gradient must be bit-exact."""
        self.setUp()
        ref = self._run(self._base_config(), offload=False)
        self.setUp()
        got = self._run(self._offload_config(ATTENTION_MODULES), offload=True)
        self.assertGreater(
            got[2]["packed"],
            0,
            "nothing was offloaded: either no region was entered or the size "
            "threshold filtered every activation out",
        )
        self._assert_bit_exact(ref, got, "all attention modules")

    def test_single_module_offloads_less_than_all_modules(self):
        """One region must offload strictly less than all of them.

        This is what shows the module selection is read at all.
        """
        self.setUp()
        one = self._run(self._offload_config(["core_attn"]), offload=True)
        self.setUp()
        every = self._run(self._offload_config(ATTENTION_MODULES), offload=True)
        self.assertGreater(one[2]["packed"], 0, "core_attn offloaded nothing")
        self.assertLess(
            one[2]["packed"],
            every[2]["packed"],
            "one region offloaded as many tensors as all of them, so the region "
            "boundaries are not being applied",
        )

    def test_region_offloads_whole_module_not_one_tensor(self):
        """A region offloads everything its module saves, not a single tensor.

        ``core_attn``'s backward needs q/k/v and the softmax intermediates, so on
        a two-layer model ``packed`` has to be well above 2. A regression to
        "tag the module's output" would drop it straight back to 2.
        """
        self.setUp()
        got = self._run(self._offload_config(["core_attn"]), offload=True)
        n_layers = self._base_config().num_hidden_layers
        self.assertGreater(
            got[2]["packed"],
            n_layers,
            f"{n_layers} layers offloaded only {got[2]['packed']} tensors, "
            "which is the old one-tensor-per-name behaviour",
        )

    def test_expert_fc1_is_bit_exact(self):
        """The MoE experts' fc1 input, the largest MoE activation, must offload.

        ``moe_expert_fusion=True`` is required: that boundary lives in
        ``GroupedMLPExpert``, and the default ``False`` runs one
        ``StandardMLPExpert`` per expert, which has no such point at all.
        """
        self.setUp()
        # moe_deep_gemm=False because DeepGEMM only takes bf16, and the fused
        # experts' weights are bf16 already, so the whole model has to run bf16
        # or the two operands of the grouped GEMM disagree on dtype. bf16 does
        # not weaken the bit-exactness bar: offloading changes no computation.
        fused = {
            "moe_expert_fusion": True,
            "moe_deep_gemm": False,
            "bf16": True,
            "params_dtype": paddle.bfloat16,
        }
        ref = self._run(self._base_config(**fused), offload=False)
        self.setUp()
        got = self._run(
            self._offload_config(["expert_fc1"], **fused), offload=True
        )
        self.assertGreater(
            got[2]["packed"],
            0,
            "expert_fc1 offloaded nothing: either GroupedMLPExpert was not used "
            "(the fused SonicMoE path has no such boundary on the Python side) "
            "or the region is not being entered",
        )
        self._assert_bit_exact(ref, got, "expert_fc1")

    def test_moe_act_is_bit_exact(self):
        """``moe_act`` offloads fc1's output, a different tensor from
        ``expert_fc1``'s.

        With a gated linear unit, fc1's output is gate and up concatenated, twice
        ``moe_intermediate_size`` wide, which makes it the largest activation in
        the MoE. It too only exists on ``GroupedMLPExpert``.
        """
        self.setUp()
        fused = {
            "moe_expert_fusion": True,
            "moe_deep_gemm": False,
            "bf16": True,
            "params_dtype": paddle.bfloat16,
        }
        ref = self._run(self._base_config(**fused), offload=False)
        self.setUp()
        got = self._run(
            self._offload_config(["moe_act"], **fused), offload=True
        )
        self.assertGreater(got[2]["packed"], 0, "moe_act offloaded nothing")
        self._assert_bit_exact(ref, got, "moe_act")

    def test_moe_act_and_expert_fc1_offload_different_tensors(self):
        """Both MoE regions together must offload more than either alone."""
        self.setUp()
        fused = {
            "moe_expert_fusion": True,
            "moe_deep_gemm": False,
            "bf16": True,
            "params_dtype": paddle.bfloat16,
        }
        ref = self._run(self._base_config(**fused), offload=False)
        self.setUp()
        fc1 = self._run(self._offload_config(["expert_fc1"], **fused), True)
        self.setUp()
        act = self._run(self._offload_config(["moe_act"], **fused), True)
        self.setUp()
        both = self._run(
            self._offload_config(["expert_fc1", "moe_act"], **fused), True
        )
        self.assertEqual(
            both[2]["packed"],
            fc1[2]["packed"] + act[2]["packed"],
            "the two boundaries should target different tensors, so enabling "
            "both should offload the sum of what each one does",
        )
        self._assert_bit_exact(ref, both, "expert_fc1+moe_act")

    def test_fraction_zero_offloads_nothing(self):
        """fraction=0 means "keep every boundary resident", still bit-exact."""
        self.setUp()
        ref = self._run(self._base_config(), offload=False)
        config = self._offload_config(
            ATTENTION_MODULES, activation_offload_fraction=0.0
        )
        # Both iterations must share one manager: what the learning iteration
        # observed (boundary index -> bytes) lives on the manager, so a new
        # instance would start learning from scratch.
        mgr = self._manager(config, offload=True)
        self.setUp()
        # Learning iteration: everything is offloaded and only observed.
        first = self._run(config, offload=True, mgr=mgr)
        self.setUp()
        # Settled: the policy is in force and nothing should be offloaded.
        second = self._run(config, offload=True, mgr=mgr)
        self.assertGreater(first[2]["packed"], 0)
        self.assertEqual(
            second[2]["packed"],
            0,
            "fraction=0 must leave every boundary on the device",
        )
        self._assert_bit_exact(ref, second, "fraction=0")

    def test_fraction_half_offloads_some_boundaries_but_not_all(self):
        """A fraction between the extremes must land between them too.

        0 and 1 can both be reached by ignoring the knob in one direction or the
        other; a middle value cannot.
        """
        self.setUp()
        ref = self._run(self._base_config(), offload=False)
        config = self._offload_config(
            ATTENTION_MODULES, activation_offload_fraction=0.5
        )
        mgr = self._manager(config, offload=True)
        self.setUp()
        learning = self._run(config, offload=True, mgr=mgr)
        self.setUp()
        settled = self._run(config, offload=True, mgr=mgr)
        self.assertGreater(
            settled[2]["packed"],
            0,
            "fraction=0.5 offloaded nothing, which is what fraction=0 means",
        )
        self.assertLess(
            settled[2]["packed"],
            learning[2]["packed"],
            "fraction=0.5 offloaded every boundary, which is what fraction=1 "
            "means",
        )
        self.assertGreater(settled[2]["skipped_policy"], 0)
        self._assert_bit_exact(ref, settled, "fraction=0.5")

    def test_a_full_pinned_pool_leaves_activations_on_the_device(self):
        """Running out of pinned memory must degrade, not fail.

        A capacity this small is exhausted partway through the forward pass, so
        the run covers both branches: activations offloaded before the pool filled
        up, and activations that had to stay resident afterwards. Correctness is
        what matters here -- the fallback is on the same numerical path.
        """
        self.setUp()
        ref = self._run(self._base_config(), offload=False)
        self.setUp()
        got = self._run(
            self._offload_config(
                ATTENTION_MODULES,
                activation_offload_pool_capacity_bytes=1 << 20,
            ),
            offload=True,
        )
        self.assertGreater(got[2]["packed"], 0, "nothing was offloaded at all")
        self.assertGreater(
            got[2]["pool_oom"],
            0,
            "a 1MB pool was never exhausted, so the degradation path did not "
            "run and this test proves nothing",
        )
        self._assert_bit_exact(ref, got, "pool at capacity")

    def test_prefetch_budget_settings_are_all_bit_exact(self):
        """The three budget settings the config exposes, on a real model.

        ``0`` disables prefetch, an explicit value is honoured exactly as given,
        and ``None`` asks the manager to pick one, which it can only do once the
        first iteration has measured a group. What is checked here is that the
        knob arrives intact and that none of the three changes a number; whether
        prefetch actually hides the copies is driven by the pipeline schedule and
        belongs to the PP test, since there is no backward anchor on this path.
        """
        self.setUp()
        ref = self._run(self._base_config(), offload=False)

        for budget in (0, 64 * 1024):
            self.setUp()
            config = self._offload_config(
                ATTENTION_MODULES,
                activation_offload_prefetch_budget_bytes=budget,
            )
            mgr = self._manager(config, offload=True)
            self.assertEqual(
                mgr.prefetch_budget_bytes,
                budget,
                "an explicit budget must reach the manager unchanged",
            )
            self.assertFalse(
                mgr.auto_budget, "an explicit budget must not be re-tuned"
            )
            got = self._run(config, offload=True, mgr=mgr)
            self.assertGreater(got[2]["packed"], 0)
            self.assertEqual(
                mgr.prefetch_budget_bytes,
                budget,
                "the budget was adjusted even though it was pinned",
            )
            self._assert_bit_exact(ref, got, f"prefetch budget {budget}")

        self.setUp()
        config = self._offload_config(ATTENTION_MODULES)
        mgr = self._manager(config, offload=True)
        self.assertIsNone(
            mgr.prefetch_budget_bytes,
            "an unset budget must stay unset until a group has been measured",
        )
        auto = self._run(config, offload=True, mgr=mgr)
        self.assertGreater(
            mgr.prefetch_budget_bytes,
            0,
            "no budget was picked, so every later iteration keeps reloading "
            "whole groups",
        )
        self._assert_bit_exact(ref, auto, "auto prefetch budget")

    def test_a_threshold_above_every_activation_offloads_nothing(self):
        """The size threshold has to be able to filter everything out.

        This is the configuration to reach for when offloading has to be ruled
        out as the cause of something, so it must be inert rather than
        half-applied.
        """
        self.setUp()
        ref = self._run(self._base_config(), offload=False)
        self.setUp()
        got = self._run(
            self._offload_config(
                ATTENTION_MODULES, min_offloaded_tensor_bytes=1 << 30
            ),
            offload=True,
        )
        self.assertEqual(
            got[2]["packed"],
            0,
            "an activation larger than 1GB cannot exist at this model size, so "
            "the threshold is not being applied",
        )
        self.assertGreater(got[2]["skipped"], 0, "no region was entered at all")
        self._assert_bit_exact(ref, got, "threshold above every activation")


class TestOffloadConfigValidation(unittest.TestCase):
    """The rules in ``_validate_activation_offloading``. No GPU needed.

    Recompute and offloading are **not** mutually exclusive: what a region
    offloads is its input, which is exactly the tensor recompute has to keep
    resident, so the two together use less memory than either alone (measured on
    a 4-layer GPTModel: peak activations 713.5 -> 542.5 -> 507.2MB, bit-exact).
    The one case that must be rejected is a region nested *inside* a checkpoint:
    forward then saves nothing, the region is only entered during the backward
    replay, and every offload becomes a D2H/H2D round trip for nothing. The
    per-combination measurements are in ``probes/probe_region_inside_recompute.py``.
    """

    @staticmethod
    def _config(offload, recompute):
        return TestGPTModelActivationOffload._offload_config(
            offload,
            recompute_granularity="selective" if recompute else None,
            recompute_modules=list(recompute),
            recompute_num_layers=None,
        )

    def test_same_module_in_recompute_and_offload_is_allowed(self):
        for name in ("core_attn", "attn_norm", "mlp_norm"):
            with self.subTest(name=name):
                # The names that line up on both sides -- core_attn and the two
                # norms -- are allowed to appear in both lists.
                rc = "norm" if name.endswith("norm") else name
                self._config([name] if name != "attn_proj" else [], [rc])

    def test_moe_gate_up_recompute_allowed_with_expert_fc1(self):
        # moe_gate_up recomputes fc1's output, which is not the tensor the
        # expert_fc1 region offloads.
        self._config(["expert_fc1"], ["moe_gate_up"])

    def test_mlp_recompute_rejects_moe_internal_offload(self):
        for name in ("expert_fc1", "moe_act", "fused_group_mlp"):
            with (
                self.subTest(name=name),
                self.assertRaisesRegex(ValueError, "backward replay"),
            ):
                self._config([name], ["mlp"])

    def test_mlp_recompute_allows_attention_offload(self):
        # Recomputing the MLP covers no region in the attention block.
        self._config(ATTENTION_MODULES, ["mlp"])

    def test_full_recompute_is_rejected(self):
        # full wraps the whole layer in a checkpoint, so every region is only
        # entered during the backward replay: measured as 0 packs in forward and
        # all of them in backward, a pure round trip. It has to be a hard error
        # rather than a silent degradation.
        with self.assertRaisesRegex(ValueError, "recompute_granularity"):
            TestGPTModelActivationOffload._offload_config(
                ["core_attn"],
                recompute_granularity="full",
                recompute_method="uniform",
                recompute_num_layers=1,
            )

    def test_fused_group_mlp_excludes_finer_moe_boundaries(self):
        # fused_group_mlp already covers the whole fused node's cached tensors,
        # so it is mutually exclusive with the two finer MoE boundaries.
        for name in ("expert_fc1", "moe_act"):
            with (
                self.subTest(name=name),
                self.assertRaisesRegex(ValueError, "fused_group_mlp"),
            ):
                self._config(["fused_group_mlp", name], [])
        # On its own it is fine.
        self._config(["fused_group_mlp"], [])

    def test_none_knobs_fall_back_to_defaults(self):
        """A None from the outer config layer must restore the dataclass default.

        PaddleFormers' LlmMetaConfig whitelist writes *every* whitelisted key onto
        the model config, including the ones the YAML never mentioned, so these
        fields arrive here as None while their defaults are not None. Without the
        fallback the range checks blow up -- seen for real as
        ``'<' not supported between NoneType and int``.
        """
        config = TestGPTModelActivationOffload._base_config(
            fine_grained_activation_offloading=True,
            offload_modules=["core_attn", "attn_proj"],
            min_offloaded_tensor_bytes=None,
            activation_offload_fraction=None,
            delta_offload_bytes_across_pp_ranks=None,
            activation_offload_numa_bind=None,
            activation_offload_prefetch_budget_bytes=None,
            activation_offload_pool_capacity_bytes=None,
        )
        self.assertEqual(config.min_offloaded_tensor_bytes, 2 * 1024 * 1024)
        self.assertEqual(config.activation_offload_fraction, 1.0)
        self.assertEqual(config.delta_offload_bytes_across_pp_ranks, 0)
        self.assertIs(config.activation_offload_numa_bind, True)
        # These two default to None already and must be left alone.
        self.assertIsNone(config.activation_offload_prefetch_budget_bytes)
        self.assertIsNone(config.activation_offload_pool_capacity_bytes)

    def test_attn_proj_requires_core_attn(self):
        with self.assertRaisesRegex(ValueError, "requires 'core_attn'"):
            self._config(["attn_proj"], [])

    def test_offload_modules_without_master_switch(self):
        with self.assertRaisesRegex(ValueError, "master switch"):
            TestGPTModelActivationOffload._base_config(
                offload_modules=["core_attn"]
            )

    def test_unknown_offload_module(self):
        with self.assertRaisesRegex(ValueError, "Invalid offload_modules"):
            self._config(["not_a_module"], [])


if __name__ == "__main__":
    unittest.main()

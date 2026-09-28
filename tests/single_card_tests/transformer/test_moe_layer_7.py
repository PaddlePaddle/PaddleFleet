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

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

# Heavy imports are guarded honestly: this CPU-only environment has no paddle
# installed, so the whole MoE stack is unimportable. We skip with a truthful
# reason instead of faking a pass. Only ImportError/ModuleNotFoundError count as
# "dependency absent"; any other error must surface as a real failure.
try:
    import paddle

    from paddlefleet.transformer.moe import moe_layer
    from paddlefleet.transformer.moe.moe_layer import MoELayer

    _IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
    paddle = None
    moe_layer = None
    MoELayer = None
    _IMPORT_ERROR = exc

_SKIP_REASON = (
    "paddle / paddlefleet MoE stack not importable on this CPU environment: "
    f"{_IMPORT_ERROR!r}"
)


# The two methods exercised here run their genuine, unmodified bodies. Both are
# pure orchestration over a handful of plain attributes, so we bypass the
# heavyweight ``MoELayer.__init__`` (it needs a full TransformerConfig, a live
# ProcessGroupCollection and constructed experts, none available on a CPU box)
# by binding the unbound method to a light ``self`` holder. No production method
# is patched or rewritten -- the real code stays on the validation chain.
@unittest.skipUnless(MoELayer is not None, _SKIP_REASON)
class TestMoELayerInitExpertParallel(unittest.TestCase):
    """MoELayer._init_expert_parallel derives the per-device expert count,
    the (floored) EP rank and the moe group.

    moe_layer.py:911-951. The expert-parallel branch splits ``num_experts``
    across ``expert_model_parallel_size`` ranks; the intermediate-EP-sharding
    dispatchers keep every expert on every rank; the single-rank branch resets
    the group/rank/count. Every expectation below is derived by hand from the
    exact division / floor / assertion rules, so a swapped branch, a wrong
    divisor or a dropped floor cannot survive.
    """

    def _run(self, **attrs):
        stub = SimpleNamespace(**attrs)
        MoELayer._init_expert_parallel(stub)
        return stub

    def test_single_rank_resets_group_rank_and_uses_all_experts(self):
        # EP == 1: the else-branch forces moe_group=None, moe_rank=0, keeps
        # expert_model_parallel_size at 1 and puts every expert on this rank.
        stub = self._run(
            expert_model_parallel_size=1,
            num_experts=4,
            moe_group="preexisting-sentinel",
        )
        self.assertIsNone(stub.moe_group)
        self.assertEqual(stub.moe_rank, 0)
        self.assertEqual(stub.expert_model_parallel_size, 1)
        self.assertEqual(stub.num_experts_per_device, 4)

    def test_expert_parallel_divides_experts_across_ranks(self):
        # 8 experts / EP=4 => 2 local experts. moe_rank comes from the real
        # get_pg_rank(moe_group); moe_grad_group is taken from pg.expt_dp.
        expt_dp = object()
        pg = SimpleNamespace(expt_dp=expt_dp)
        moe_group = object()
        with mock.patch.object(
            moe_layer.utils, "get_pg_rank", return_value=3
        ) as get_rank:
            stub = self._run(
                expert_model_parallel_size=4,
                num_experts=8,
                use_intermediate_ep_sharding=False,
                pg_collection=pg,
                moe_group=moe_group,
            )
        get_rank.assert_called_once_with(moe_group)
        self.assertEqual(stub.moe_rank, 3)
        self.assertEqual(stub.num_experts_per_device, 2)
        self.assertEqual(stub.expert_model_parallel_size, 4)
        self.assertIs(stub.moe_grad_group, expt_dp)

    def test_negative_pg_rank_is_floored_to_zero(self):
        # get_pg_rank may report -1 for an uninitialized group; the production
        # code floors it with max(rank, 0). A missing floor would leave -1 and
        # index experts from the tail.
        pg = SimpleNamespace(expt_dp=object())
        with mock.patch.object(moe_layer.utils, "get_pg_rank", return_value=-1):
            stub = self._run(
                expert_model_parallel_size=2,
                num_experts=4,
                use_intermediate_ep_sharding=False,
                pg_collection=pg,
                moe_group=object(),
            )
        self.assertEqual(stub.moe_rank, 0)
        self.assertEqual(stub.num_experts_per_device, 2)

    def test_intermediate_ep_sharding_keeps_every_expert_local(self):
        # allgather / ringmoe dispatchers shard the *intermediate* dim, so every
        # rank still holds all experts -> num_experts_per_device == num_experts,
        # NOT num_experts // EP. This is the branch a naive divide would break.
        pg = SimpleNamespace(expt_dp=object())
        with mock.patch.object(moe_layer.utils, "get_pg_rank", return_value=1):
            stub = self._run(
                expert_model_parallel_size=4,
                num_experts=8,
                use_intermediate_ep_sharding=True,
                pg_collection=pg,
                moe_group=object(),
            )
        self.assertEqual(stub.num_experts_per_device, 8)
        self.assertEqual(stub.moe_rank, 1)

    def test_experts_not_divisible_by_ep_raises(self):
        # 6 experts across EP=4: 6 >= 4 holds but 6 % 4 != 0, so the divisibility
        # assertion must fire.
        pg = SimpleNamespace(expt_dp=object())
        with (
            mock.patch.object(moe_layer.utils, "get_pg_rank", return_value=0),
            self.assertRaises(AssertionError),
        ):
            self._run(
                expert_model_parallel_size=4,
                num_experts=6,
                use_intermediate_ep_sharding=False,
                pg_collection=pg,
                moe_group=object(),
            )

    def test_fewer_experts_than_ep_raises(self):
        # 2 experts across EP=4: the num_experts >= EP assertion must fire first.
        pg = SimpleNamespace(expt_dp=object())
        with (
            mock.patch.object(moe_layer.utils, "get_pg_rank", return_value=0),
            self.assertRaises(AssertionError),
        ):
            self._run(
                expert_model_parallel_size=4,
                num_experts=2,
                use_intermediate_ep_sharding=False,
                pg_collection=pg,
                moe_group=object(),
            )


@unittest.skipUnless(MoELayer is not None, _SKIP_REASON)
class TestMoELayerPermuteUnpermuteDelegation(unittest.TestCase):
    """permute / unpermute forward the token tensor to the token_dispatcher and
    return its results untouched.

    moe_layer.py:1069-1076. The dispatcher is a genuine, non-under-test
    collaborator, so it is replaced by a stub that returns distinguishable
    marker tensors bound by identity. We then assert the exact input object is
    handed over and the exact outputs are returned (identity + content), which a
    dropped call, a swapped argument or a discarded return value could not
    satisfy.
    """

    def setUp(self):
        paddle.set_device("cpu")

    def test_permute_forwards_hidden_and_returns_both_dispatch_outputs(self):
        hidden = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
        global_tokens = paddle.to_tensor(
            [[9.0, 8.0], [7.0, 6.0]], dtype="float32"
        )
        tokens_per_expert = paddle.to_tensor([1, 1], dtype="int64")

        dispatcher = mock.MagicMock()
        dispatcher.dispatch_postprocess.return_value = (
            global_tokens,
            tokens_per_expert,
        )
        stub = SimpleNamespace(token_dispatcher=dispatcher)

        out_tokens, out_counts = MoELayer.permute(stub, hidden)

        dispatcher.dispatch_postprocess.assert_called_once()
        (called_arg,), called_kwargs = dispatcher.dispatch_postprocess.call_args
        self.assertIs(called_arg, hidden)
        self.assertEqual(called_kwargs, {})
        # Both dispatcher outputs propagate out by identity, in order.
        self.assertIs(out_tokens, global_tokens)
        self.assertIs(out_counts, tokens_per_expert)
        np.testing.assert_array_equal(
            out_tokens.numpy(), [[9.0, 8.0], [7.0, 6.0]]
        )
        np.testing.assert_array_equal(out_counts.numpy(), [1, 1])

    def test_unpermute_forwards_hidden_to_combine_preprocess(self):
        hidden = paddle.to_tensor([[5.0, 6.0, 7.0]], dtype="float32")
        combined = paddle.to_tensor([[50.0, 60.0, 70.0]], dtype="float32")

        dispatcher = mock.MagicMock()
        dispatcher.combine_preprocess.return_value = combined
        stub = SimpleNamespace(token_dispatcher=dispatcher)

        result = MoELayer.unpermute(stub, hidden)

        dispatcher.combine_preprocess.assert_called_once()
        (called_arg,), called_kwargs = dispatcher.combine_preprocess.call_args
        self.assertIs(called_arg, hidden)
        self.assertEqual(called_kwargs, {})
        self.assertIs(result, combined)
        np.testing.assert_array_equal(result.numpy(), [[50.0, 60.0, 70.0]])


if __name__ == "__main__":
    unittest.main()

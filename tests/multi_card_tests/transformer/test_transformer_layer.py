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

"""Real multi-card behavior tests for ``TransformerLayer`` and ``tensors_clone``.

Topology: 8 GPUs, tensor-model-parallel size 4, sharding size 2
(``mp_degree=4``, ``sharding_degree=2``). Launch with::

    python -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 \
        tests/multi_card_tests/transformer/test_transformer_layer.py
"""

import functools

import numpy as np
import paddle
import paddle.distributed as dist

from paddlefleet.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from paddlefleet.parallel_state import get_tensor_model_parallel_group
from paddlefleet.process_groups_config import ProcessGroupCollection
from paddlefleet.tensor_parallel.random import model_parallel_cuda_manual_seed
from paddlefleet.transformer.transformer_config import TransformerConfig
from paddlefleet.transformer.transformer_layer import tensors_clone
from tests.multi_card_tests.tensor_parallel.test_utilities import Utils

HIDDEN_SIZE = 64
NUM_HEADS = 4
SEQ_LEN = 8
BATCH = 2


def _build_tp_transformer_layer():
    """Construct a real tensor-parallel dense transformer layer (TP=4)."""
    config = TransformerConfig(
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=NUM_HEADS,
        intermediate_size=128,
        use_cpu_initialization=True,
        tensor_model_parallel_size=4,
        sequence_parallel=False,
        bf16=False,
        params_dtype=paddle.float32,
        rms_norm_eps=1e-5,
        hidden_dropout_prob=0.0,
        attention_dropout=0.0,
        init_method=functools.partial(paddle.nn.init.xavier_uniform_, gain=1.0),
        output_layer_init_method=functools.partial(
            paddle.nn.init.xavier_uniform_, gain=1.0
        ),
    )
    pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    layer_spec = get_gpt_layer_local_spec(config)
    layer = layer_spec.layer(
        config,
        layer_spec.sublayers_spec,
        layer_number=1,
        pg_collection=pg_collection,
    )
    return layer


def _replicated_input():
    """Deterministic input, identical on every rank (no RNG involved).

    Megatron-style TP consumes a *replicated* activation on every rank of the
    TP group; building it from ``arange`` guarantees byte-identical inputs so
    the only source of any cross-rank divergence would be a broken collective.
    """
    numel = BATCH * SEQ_LEN * HIDDEN_SIZE
    hidden = paddle.arange(numel, dtype="float32").reshape(
        [BATCH, SEQ_LEN, HIDDEN_SIZE]
    )
    hidden = hidden / float(numel)
    return hidden.cuda()


def test_transformer_layer_tp_output_is_replicated():
    """TP dense layer output must be replicated across the TP group.

    The self-attention and MLP output projections are RowParallelLinear whose
    trailing all-reduce sums each rank's partial (head / intermediate) shard.
    With a replicated input and dropout disabled, that all-reduce makes the
    final hidden_states identical on all 4 ranks of the TP group. The
    reference here is the TP replication invariant itself (independent of the
    layer's internal math): if the row-parallel all-reduce were dropped or
    reduced the wrong shard, ranks would hold different partial sums and the
    cross-rank comparison below would fail. Real collectives run on GPU.
    """
    layer = _build_tp_transformer_layer()
    hidden = _replicated_input()

    result = layer({"hidden_states": hidden})
    out = result["hidden_states"]

    # Contract 1: a transformer layer preserves the [b, s, h] shape.
    assert list(out.shape) == [BATCH, SEQ_LEN, HIDDEN_SIZE], out.shape

    # Contract 2: the layer is not a degenerate identity — it must actually
    # transform the activation, otherwise "replicated" would be vacuous.
    assert not paddle.equal_all(out, hidden)

    # Contract 3: output is replicated across the whole TP group. Gather every
    # rank's output through a REAL all-gather and require exact agreement
    # (all-reduce yields the same reduced value on every participating rank).
    tp_group = get_tensor_model_parallel_group()
    gathered = []
    dist.all_gather(gathered, out.contiguous(), group=tp_group)
    assert len(gathered) == tp_group.nranks == 4, len(gathered)
    ref = gathered[0]
    for i, other in enumerate(gathered[1:], start=1):
        assert paddle.equal_all(other, ref), (
            f"TP rank {i} output diverged from rank 0; "
            "row-parallel all-reduce is not replicating the activation"
        )


def test_tensors_clone_list_branch_passthrough():
    """The list/tuple branch clones tensors and passes non-tensors through.

    Hand-derived reference: tensors are deep-copied (equal value, new object,
    gradients severed), while ints / strings / None / nested dicts are handled
    without ever calling ``.clone()`` on a non-tensor.
    """
    base = paddle.arange(6, dtype="float32").reshape([2, 3]).cuda()
    nested_tensor = (paddle.ones([2, 2]) * 3.0).cuda()

    out = tensors_clone([base, 5, "keep", None, {"n": nested_tensor}])

    assert isinstance(out, list)
    assert len(out) == 5
    # cloned tensor: same values, distinct storage
    assert paddle.equal_all(out[0], base)
    assert out[0] is not base
    # non-tensors are forwarded unchanged
    assert out[1] == 5
    assert out[2] == "keep"
    assert out[3] is None
    # nested dict of tensors is recursively cloned
    assert isinstance(out[4], dict)
    assert paddle.equal_all(out[4]["n"], nested_tensor)
    assert out[4]["n"] is not nested_tensor

    # tuple input preserves the tuple container type
    tup = tensors_clone((base, 7))
    assert isinstance(tup, tuple)
    assert paddle.equal_all(tup[0], base)
    assert tup[1] == 7


def test_tensors_clone_dict_all_tensor_values():
    """All-tensor dicts clone every value (values equal, objects distinct)."""
    a = paddle.arange(4, dtype="float32").reshape([2, 2]).cuda()
    b = (paddle.ones([2, 2]) * -2.0).cuda()

    out = tensors_clone({"a": a, "b": b})

    assert set(out.keys()) == {"a", "b"}
    assert paddle.equal_all(out["a"], a) and out["a"] is not a
    assert paddle.equal_all(out["b"], b) and out["b"] is not b


def test_tensors_clone_dict_with_nontensor_is_known_bug():
    """KNOWN BUG (transformer_layer.py:129-133): dict branch is asymmetric.

    The list/tuple branch guards each element with ``isinstance(item, Tensor)``
    and forwards non-tensors untouched, but the dict branch calls
    ``value.clone()`` unconditionally. A dict holding any non-tensor value
    therefore raises ``AttributeError``. The CORRECT behavior would mirror the
    list branch and pass the non-tensor through. We do NOT edit production;
    instead we lock the current (buggy) behavior with an assertRaises-style
    check so that a future fix makes this test fail and flags the update.
    """
    payload = {"hidden": (paddle.ones([2, 2])).cuda(), "flag": True}

    raised = False
    try:
        tensors_clone(payload)
    except AttributeError:
        raised = True
    assert raised, (
        "expected AttributeError from tensors_clone dict branch on a "
        "non-tensor value (known asymmetry bug at "
        "transformer_layer.py:129-133)"
    )


def _run_all():
    test_transformer_layer_tp_output_is_replicated()
    test_tensors_clone_list_branch_passthrough()
    test_tensors_clone_dict_all_tensor_values()
    test_tensors_clone_dict_with_nontensor_is_known_bug()


if __name__ == "__main__":
    Utils.initialize_model_parallel(
        tensor_parallel_size=4, sharding_parallel_size=2
    )
    np.random.seed(42)
    paddle.seed(42)
    model_parallel_cuda_manual_seed(42)
    _run_all()

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

import functools

import numpy as np
import paddle
import paddle.distributed as dist
import paddle.nn.functional as F

from paddlefleet.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from paddlefleet.process_groups_config import ProcessGroupCollection
from paddlefleet.transformer.mlp import MLP
from paddlefleet.transformer.transformer_config import TransformerConfig
from paddlefleet.utils import get_tensor_model_parallel_group_if_none
from tests.multi_card_tests.tensor_parallel.test_utilities import Utils

HIDDEN = 16
INTERMEDIATE = 32  # global; column/row parallel shard this by TP=4


def _dense_config(**overrides):
    # A dense (num_experts=None) SwiGLU MLP with TP=4. bias_activation_fusion
    # stays at its default (False) and use_bias is False, so MLP.forward takes
    # the plain gated path: silu(gate) * up with glu_linear_offset 0 and no
    # clamp -- exactly the swiglu expression reproduced by hand below.
    defaults = dict(  # noqa: C408
        hidden_size=HIDDEN,
        num_attention_heads=4,
        intermediate_size=INTERMEDIATE,
        gated_linear_unit=True,
        hidden_act=F.silu,
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
    defaults.update(overrides)
    return TransformerConfig(**defaults)


def _build_dense_mlp(config):
    pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    layer_spec = get_gpt_layer_local_spec(config)
    layer = layer_spec.layer(
        config,
        layer_spec.sublayers_spec,
        layer_number=1,
        pg_collection=pg_collection,
    )
    return layer.mlp.cuda()


def _silu(z):
    return z / (1.0 + np.exp(-z))


def test_dense_mlp_swiglu_matches_independent_reference():
    """Real TP=4 forward of the dense SwiGLU MLP built by the spec.

    The row-parallel down_proj all-reduces each rank's partial across the 4
    ranks, so the returned output is the full MLP result. We rebuild that
    result independently in numpy from the *actual* per-rank weights gathered
    off every rank -- summing each rank's ``silu(x@Wg_r)*(x@Wu_r) @ Wd_r``
    contribution. A dropped or wrong all-reduce changes the summed magnitude,
    which the scale-sensitive check rejects (see the negative controls). The
    expected is never produced by the production forward.
    """
    config = _dense_config()
    mlp = _build_dense_mlp(config)
    assert isinstance(mlp, MLP)

    rank = Utils.rank
    tp_group = get_tensor_model_parallel_group_if_none(tp_group=None)
    world_size = tp_group.world_size

    ug_out = 2 * INTERMEDIATE // world_size  # local [gate_r | up_r]
    dn_in = INTERMEDIATE // world_size

    # Distinct, reproducible weights per rank so a wrong gather/peer or a
    # dropped rank changes the summed reference.
    wrng = np.random.RandomState(1000 + rank)
    ug_val = (wrng.standard_normal((HIDDEN, ug_out)) * 0.1).astype("float32")
    dn_val = (wrng.standard_normal((dn_in, HIDDEN)) * 0.1).astype("float32")
    assert list(mlp.up_gate_proj.weight.shape) == [HIDDEN, ug_out]
    assert list(mlp.down_proj.weight.shape) == [dn_in, HIDDEN]
    mlp.up_gate_proj.weight.set_value(paddle.to_tensor(ug_val).cuda())
    mlp.down_proj.weight.set_value(paddle.to_tensor(dn_val).cuda())

    # Replicated input: identical on every rank (fixed rank-independent seed).
    xrng = np.random.RandomState(7)
    x_np = xrng.standard_normal((2, 3, HIDDEN)).astype("float32")
    hidden_states = paddle.to_tensor(x_np).cuda()

    with paddle.no_grad():
        output, output_bias = mlp(hidden_states)
    assert output_bias is None
    actual = np.asarray(output.astype("float32").numpy(), dtype=np.float64)

    # Gather the real per-rank weights and rebuild the reference by hand.
    ug_list, dn_list = [], []
    dist.all_gather(ug_list, mlp.up_gate_proj.weight, group=tp_group)
    dist.all_gather(dn_list, mlp.down_proj.weight, group=tp_group)

    half = ug_out // 2  # == dn_in
    y_ref = np.zeros((2, 3, HIDDEN), dtype=np.float64)
    for ug_t, dn_t in zip(ug_list, dn_list):
        ug_r = np.asarray(ug_t.astype("float32").numpy(), dtype=np.float64)
        dn_r = np.asarray(dn_t.astype("float32").numpy(), dtype=np.float64)
        fc1 = np.matmul(x_np.astype(np.float64), ug_r)
        gate, up = fc1[..., :half], fc1[..., half:]
        act = _silu(gate) * up
        y_ref = y_ref + np.matmul(act, dn_r)

    ref_norm = np.linalg.norm(y_ref)
    assert ref_norm > 1e-6
    np.testing.assert_allclose(actual, y_ref, rtol=1e-4, atol=1e-5)
    assert np.linalg.norm(actual - y_ref) / ref_norm < 1e-3

    # The comparison must reject magnitude errors: a half/double/zeroed
    # all-reduce result differs from the true sum.
    for bad in (0.5 * y_ref, 2.0 * y_ref, np.zeros_like(y_ref)):
        rejected = False
        try:
            np.testing.assert_allclose(bad, y_ref, rtol=1e-4, atol=1e-5)
        except AssertionError:
            rejected = True
        assert rejected


def test_hy_sparse_without_mla_is_rejected():
    """get_gpt_layer_local_spec must reject HySparse without MLA.

    HySparseTransformerLayer passes a ``shared_kv`` kwarg into the attention
    forward, which only the MLA-absorbed MQA path accepts; a plain
    self-attention layer would raise TypeError at runtime. The spec guards this
    at build time, so a valid-looking config with hy-sparse on but MLA off must
    raise ValueError from the real production entry rather than build a layer
    that breaks later.
    """
    config = _dense_config(
        enable_hy_sparse_attention=True,
        multi_latent_attention=False,
    )
    raised = False
    try:
        get_gpt_layer_local_spec(config)
    except ValueError:
        raised = True
    assert raised, (
        "expected get_gpt_layer_local_spec to raise ValueError when "
        "enable_hy_sparse_attention is set without multi-latent attention"
    )


if __name__ == "__main__":
    Utils.initialize_model_parallel(4, 1)
    test_dense_mlp_swiglu_matches_independent_reference()
    test_hy_sparse_without_mla_is_rejected()

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
import math

import paddle
import paddle.distributed as dist
from paddle.distributed.fleet import distributed_model

from paddlefleet import parallel_state
from paddlefleet.gpt_builders import gpt_builder
from paddlefleet.models.gpt import GPTConfig
from paddlefleet.transformer.transformer_layer import TransformerLayer
from tests.multi_card_tests.tensor_parallel.test_utilities import Utils

# Topology under test: pure pipeline parallelism over 4 GPUs. With 8 dense
# decoder layers evenly split across 4 stages, every stage owns exactly 2
# TransformerLayers, so the layer partition has an exact hand-derived form
# (see test_gpt_builder_pipeline_layer_partition).
PP_DEGREE = 4
NUM_LAYERS = 8
HIDDEN = 64
HEADS = 4
FFN = 128
VOCAB = 128
SEQ_LEN = 16
SEED = 2024

LAYERS_PER_STAGE = NUM_LAYERS // PP_DEGREE
SEG_METHOD = "layer:TransformerLayer|EmptyLayer"


def _build_config(**overrides):
    """Build a small dense GPT config driving the real pipeline builder.

    Dropout is disabled and CPU initialization is used so that a forward
    pass is deterministic given a fixed seed. All parallel/MoE features are
    left off, exercising the dense branch of ``gpt_builder``.
    """
    xavier = functools.partial(paddle.nn.init.xavier_uniform_, gain=1.0)
    defaults = dict(  # noqa: C408
        vocab_size=VOCAB,
        max_sequence_length=SEQ_LEN,
        num_hidden_layers=NUM_LAYERS,
        hidden_size=HIDDEN,
        num_attention_heads=HEADS,
        intermediate_size=FFN,
        normalization="RMSNorm",
        hidden_dropout_prob=0.0,
        attention_dropout=0.0,
        use_cpu_initialization=True,
        parallel_output=True,
        tie_word_embeddings=True,
        position_embedding_type="rope",
        rotary_percent=1.0,
        init_method=xavier,
        output_layer_init_method=xavier,
        pipeline_model_parallel_size=PP_DEGREE,
    )
    defaults.update(overrides)
    return GPTConfig(**defaults)


def _local_transformer_layers(model):
    """Return this rank's local TransformerLayer instances in order.

    ``GPTModel`` is a ``PipelineLayer``; ``run_function`` holds only the
    submodules that were partitioned onto the current pipeline stage.
    """
    return [
        layer
        for layer in model.run_function
        if isinstance(layer, TransformerLayer)
    ]


def _make_lm_batch(micro_batch_size, num_acc):
    """Deterministic next-token-prediction batch placed on GPU."""
    data = paddle.randint(
        low=0, high=VOCAB, shape=(micro_batch_size, SEQ_LEN + 1)
    ).cuda()
    input_ids = data[:, :-1]
    labels = data[:, 1:]
    position_ids = (
        paddle.arange(0, SEQ_LEN, dtype="int64")
        .unsqueeze(0)
        .expand([micro_batch_size, -1])
        .cuda()
    )
    inputs = (
        {
            "input_ids": [input_ids] * num_acc,
            "position_ids": [position_ids] * num_acc,
        },
        [labels] * num_acc,
    )
    return inputs


def test_gpt_builder_pipeline_layer_partition():
    """Each pipeline stage owns a distinct, contiguous block of layers.

    This is a genuine cross-rank sharding contract: the builder must hand
    stage ``r`` exactly the decoder layers ``[r * k, r * k + k)`` (with
    ``k = LAYERS_PER_STAGE``), so that gathering every rank's owned global
    layer numbers reconstructs ``0..NUM_LAYERS-1`` with no gaps or overlap.
    A wrong split size, a dropped layer, or a duplicated layer changes the
    exact per-rank set and the gathered union, so the assertions below fail.
    """
    config = _build_config()
    model = gpt_builder(config, num_stages=PP_DEGREE, seg_method=SEG_METHOD)

    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    pp_world = parallel_state.get_pipeline_model_parallel_world_size()
    assert pp_world == PP_DEGREE

    local_layers = _local_transformer_layers(model)
    local_numbers = sorted(int(layer.layer_number) for layer in local_layers)

    # Hand-derived expectation for this stage (head offset defaults to 0).
    start = pp_rank * LAYERS_PER_STAGE
    expected_local = list(range(start, start + LAYERS_PER_STAGE))
    assert local_numbers == expected_local, (
        f"stage {pp_rank} owns {local_numbers}, expected {expected_local}"
    )

    # Gather each stage's owned layer numbers over the real pp process group
    # and verify the union is an exact, non-overlapping partition.
    gathered = []
    dist.all_gather_object(gathered, local_numbers)
    assert len(gathered) == PP_DEGREE

    flat = [num for per_rank in gathered for num in per_rank]
    assert sorted(flat) == list(range(NUM_LAYERS)), (
        f"gathered partition {gathered} is not a clean cover of "
        f"0..{NUM_LAYERS - 1}"
    )
    # No layer number appears on two stages.
    assert len(flat) == len(set(flat)), (
        f"layers duplicated across stages: {gathered}"
    )
    # Stages are ordered: stage r holds strictly larger indices than r-1.
    for r in range(PP_DEGREE):
        assert gathered[r] == list(
            range(r * LAYERS_PER_STAGE, (r + 1) * LAYERS_PER_STAGE)
        ), f"stage {r} block {gathered[r]} is not contiguous/in-order"


def _finite_nonzero_grad_seen(parameters):
    """True if any parameter carries a finite, non-zero gradient.

    Fleet may route gradients through ``.grad`` or the fused ``main_grad``
    buffer, so both are inspected. This provides positive evidence that the
    backward pass actually ran on this stage rather than being skipped.
    """
    for param in parameters:
        for grad in (
            getattr(param, "grad", None),
            getattr(param, "main_grad", None),
        ):
            if grad is None:
                continue
            grad = grad.astype("float32")
            if not bool(paddle.isfinite(grad).all()):
                continue
            if float(paddle.abs(grad).max()) > 0.0:
                return True
    return False


def test_gpt_builder_pipeline_forward_backward():
    """Real pipeline forward+backward with an independent magnitude anchor.

    The exact transformer math is not re-derived here (that would require a
    same-weight non-pipeline reference, which is not run in this file). What
    IS asserted, using real collectives across the 4 pipeline stages:

    * the loss is finite and strictly positive (rejects NaN/+inf and a
      collapsed zero loss),
    * the loss is of the order of an untrained language model's cross
      entropy, ``ln(VOCAB)`` -- a hand-derived anchor from a near-uniform
      softmax over ``VOCAB`` classes; a grossly wrong reduction (e.g. summed
      instead of token-averaged, or a factor-of-many scaling) leaves this
      band,
    * the loss is bit-reproducible when the identical batch is replayed
      with unchanged weights (the pp loss reduction is deterministic),
    * every stage's parameters receive a finite, non-zero gradient, so the
      backward pass genuinely executed on all four ranks.
    """
    config = _build_config()
    model = gpt_builder(config, num_stages=PP_DEGREE, seg_method=SEG_METHOD)
    pp_model = distributed_model(model)

    micro_batch_size = 1
    num_acc = 4
    inputs = _make_lm_batch(micro_batch_size, num_acc)

    loss = pp_model.forward_backward_pipeline(inputs, None)
    assert loss is not None

    loss_value = float(loss)
    assert math.isfinite(loss_value), f"non-finite pp loss: {loss_value}"
    assert loss_value > 0.0, f"pp loss should be positive, got {loss_value}"

    uniform_ce = math.log(VOCAB)
    assert 0.5 * uniform_ce <= loss_value <= 3.0 * uniform_ce, (
        f"untrained pp loss {loss_value} is far from the uniform-softmax "
        f"anchor ln(VOCAB)={uniform_ce:.4f}; the reduction magnitude is "
        "likely wrong"
    )

    # Replaying the same batch without any optimizer step must reproduce the
    # exact reduced loss: the collective reduction is deterministic.
    loss_again = float(pp_model.forward_backward_pipeline(inputs, None))
    assert loss_again == loss_value, (
        f"pp loss not reproducible: {loss_value} vs {loss_again}"
    )

    local_grad_seen = _finite_nonzero_grad_seen(pp_model.parameters())
    per_rank_grad = []
    dist.all_gather_object(per_rank_grad, bool(local_grad_seen))
    assert all(per_rank_grad), (
        f"some pipeline stage produced no usable gradient: {per_rank_grad}"
    )


def test_use_fp8_vpp_missing_return_false_is_a_bug():
    """Locks a real production bug: GPTModel.use_fp8 under VPP returns None.

    In ``gpt_model.py`` ``GPTModel.use_fp8`` (around line 986) the
    virtual-pipeline branch iterates the model chunks and returns ``True``
    on the first fp8 layer, but -- unlike the non-VPP branch which ends in
    ``return False`` -- it has no trailing ``return False``. So when fp8 is
    disabled (``config.fp8 is None``) and no layer reports fp8, the VPP
    branch falls off the end and returns ``None`` instead of ``False``.

    A model built with ``virtual_pipeline_model_parallel_size > 1`` and fp8
    off must therefore currently return ``None`` here. We assert that buggy
    return so the defect is documented without editing production. When the
    missing ``return False`` is added, this call yields ``False``, the
    assertion fails, and the fix surfaces as an expected regression here.
    """
    config = _build_config(virtual_pipeline_model_parallel_size=2)
    assert config.fp8 is None
    model = gpt_builder(config, num_stages=PP_DEGREE, seg_method=SEG_METHOD)

    # Confirm we exercise the virtual-pipeline branch that carries the bug.
    assert getattr(model, "_num_virtual_pipeline_stages", 1) > 1, (
        "expected a virtual-pipeline model to reach the buggy branch"
    )

    result = model.use_fp8()
    assert result is None, (
        "GPTModel.use_fp8 under VPP with fp8 disabled is expected to return "
        f"None (missing 'return False'); got {result!r}. If this now returns "
        "False the production bug has been fixed -- update this test."
    )


if __name__ == "__main__":
    Utils.initialize_model_parallel(
        tensor_parallel_size=1, pipeline_parallel_size=PP_DEGREE
    )
    paddle.seed(SEED)
    test_gpt_builder_pipeline_layer_partition()
    test_gpt_builder_pipeline_forward_backward()
    test_use_fp8_vpp_missing_return_false_is_a_bug()

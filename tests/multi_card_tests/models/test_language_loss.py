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

"""Real multi-card behavior tests for ``LanguageLoss`` under PP=2.

Topology: world_size=2, tensor_model_parallel_size=1, pipeline_model_parallel
_size=2. With TP=1 the module takes the plain (non parallel-cross-entropy)
branch of ``forward_impl``: per-token ``CrossEntropyLoss(reduction="none")``,
``ignored_index=-100`` masking, then a token-weighted mean over the valid
tokens. Each rank feeds distinct, fully-deterministic inputs and every expected
value is derived by hand with numpy in float64 -- never from the module under
test nor from any coverage_test file -- so that magnitude errors (2x/0.5x),
sign flips, a dropped reduction term, or an accidentally-zeroed loss are all
rejected. The cross-rank test exercises a real ``dist.all_reduce`` over both
GPUs and checks the reconstructed global token-weighted mean against an
independent numpy reference built from both ranks' known inputs.
"""

import functools

import numpy as np
import paddle
import paddle.distributed as dist

from paddlefleet.models.common.language_loss.language_loss import LanguageLoss
from paddlefleet.transformer.transformer_config import TransformerConfig
from tests.multi_card_tests.tensor_parallel.test_utilities import Utils

IGNORE_INDEX = -100
VOCAB_SIZE = 5
SEQ_LEN = 4
BATCH_SIZE = 1


def _build_config():
    """Minimal TransformerConfig that drives the plain (TP=1) loss branch.

    ``LanguageLoss`` builds no trainable parameters, so only the fields the
    loss path actually reads matter: ``parallel_output`` and TP size select the
    cross-entropy kernel, ``loss_subbatch_sequence_length=0`` disables sub-
    batching, and ``gpt_model_use_experimental_version=False`` keeps the plain
    token-weighted-mean reduction (not the experimental line-wise path).
    """
    return TransformerConfig(
        hidden_size=64,
        num_attention_heads=4,
        use_cpu_initialization=True,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=2,
        sequence_parallel=False,
        params_dtype=paddle.float32,
        parallel_output=True,
        hidden_dropout_prob=0.0,
        attention_dropout=0.0,
        loss_subbatch_sequence_length=0,
        gpt_model_use_experimental_version=False,
        init_method=functools.partial(paddle.nn.init.xavier_uniform_, gain=1.0),
        output_layer_init_method=functools.partial(
            paddle.nn.init.xavier_uniform_, gain=1.0
        ),
    )


def _local_inputs(rank):
    """Deterministic, rank-distinguishable logits/labels (no randomness).

    The per-class additive tilt ``arange(V) * 0.31 * (rank + 1)`` breaks the
    softmax shift-invariance across ranks, and the per-rank scale changes the
    distribution sharpness, so rank 0 and rank 1 produce genuinely different
    per-token losses -- a dropped rank in the cross-rank reduction therefore
    changes the global value. Exactly one label per rank is ``IGNORE_INDEX`` so
    the masking branch is exercised while valid tokens remain.
    """
    base = np.arange(
        BATCH_SIZE * SEQ_LEN * VOCAB_SIZE, dtype=np.float64
    ).reshape([BATCH_SIZE, SEQ_LEN, VOCAB_SIZE])
    tilt = np.arange(VOCAB_SIZE, dtype=np.float64) * 0.31 * (rank + 1)
    logits = base * (0.1 + 0.05 * rank) + tilt
    labels = np.array(
        [
            [
                (rank + 1) % VOCAB_SIZE,
                (rank + 3) % VOCAB_SIZE,
                IGNORE_INDEX,
                (rank + 2) % VOCAB_SIZE,
            ]
        ],
        dtype=np.int64,
    )
    return logits, labels


def _numpy_masked_ce(logits, labels):
    """Independent float64 reference: masked cross-entropy sum and token count.

    Returns ``(loss_sum, valid_count)`` where ``loss_sum`` is the sum over
    valid tokens of ``-log_softmax(logits)[label]`` and ``valid_count`` is the
    number of non-ignored tokens. The token-weighted mean is
    ``loss_sum / valid_count``.
    """
    shifted = logits - logits.max(axis=-1, keepdims=True)
    log_softmax = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    loss_sum = 0.0
    valid_count = 0
    for b in range(logits.shape[0]):
        for s in range(logits.shape[1]):
            label = int(labels[b, s])
            if label == IGNORE_INDEX:
                continue
            loss_sum += -float(log_softmax[b, s, label])
            valid_count += 1
    return loss_sum, valid_count


def _to_cuda(logits_np, labels_np):
    logits = paddle.to_tensor(logits_np, dtype="float32").cuda()
    labels = paddle.to_tensor(labels_np, dtype="int64").cuda()
    return logits, labels


def test_local_masked_mean_matches_numpy():
    """Each rank's real LanguageLoss output equals an independent numpy mean.

    Distinct per-rank inputs go through the real production ``forward`` on the
    GPU. The expected value is the hand-derived token-weighted mean; the tight
    tolerance plus the ``> 0.1`` magnitude guard reject a halved/doubled loss,
    a sign flip, or an accidental zero.
    """
    config = _build_config()
    loss_fn = LanguageLoss(config=config)

    logits_np, labels_np = _local_inputs(Utils.rank)
    logits, labels = _to_cuda(logits_np, labels_np)

    loss = loss_fn(logits, labels)

    loss_sum, valid_count = _numpy_masked_ce(logits_np, labels_np)
    assert valid_count == SEQ_LEN - 1  # exactly one IGNORE_INDEX token
    expected = loss_sum / valid_count
    assert expected > 0.1  # non-degenerate: a zeroed loss would fail below

    actual = float(loss.numpy())
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_all_ignored_returns_exact_zero():
    """A fully-ignored label row makes the real forward return exactly 0.0.

    Exercises the ``(~lossmask).all()`` branch of ``forward_impl`` which returns
    ``paddle.mean(loss) * 0.0``. The assertion is exact zero, so a branch that
    leaked a nonzero mean or divided by a zero count would be caught.
    """
    config = _build_config()
    loss_fn = LanguageLoss(config=config)

    logits_np = _local_inputs(Utils.rank)[0]
    labels_np = np.full([BATCH_SIZE, SEQ_LEN], IGNORE_INDEX, dtype=np.int64)
    logits, labels = _to_cuda(logits_np, labels_np)

    loss = loss_fn(logits, labels)
    assert float(loss.numpy()) == 0.0


def test_cross_rank_token_weighted_mean_reduction():
    """Real all_reduce over both GPUs reconstructs the global loss mean.

    Each rank computes its real production loss and pairs it with its own valid
    token count to recover the loss numerator (``mean * count``). A real
    ``dist.all_reduce`` (SUM) over the whole 2-GPU world combines the
    ``(numerator, count)`` pairs; the global token-weighted mean is compared to
    an independent numpy reference built from BOTH ranks' known inputs. Because
    the two ranks contribute distinct numerators and counts, a dropped rank, a
    wrong reduction op, or a swapped term changes the result and fails here.
    """
    config = _build_config()
    loss_fn = LanguageLoss(config=config)

    logits_np, labels_np = _local_inputs(Utils.rank)
    logits, labels = _to_cuda(logits_np, labels_np)

    local_loss = float(loss_fn(logits, labels).numpy())
    _, local_count = _numpy_masked_ce(logits_np, labels_np)

    # Carry the production loss through a real collective as (numerator, count).
    stats = paddle.to_tensor(
        [local_loss * local_count, float(local_count)], dtype="float32"
    ).cuda()
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    global_mean_actual = float(stats[0].numpy()) / float(stats[1].numpy())

    # Independent global reference from both ranks' deterministic inputs.
    total_sum = 0.0
    total_count = 0
    for rank in range(Utils.world_size):
        r_logits, r_labels = _local_inputs(rank)
        r_sum, r_count = _numpy_masked_ce(r_logits, r_labels)
        total_sum += r_sum
        total_count += r_count
    expected_global = total_sum / total_count

    assert expected_global > 0.1
    np.testing.assert_allclose(
        global_mean_actual, expected_global, rtol=1e-5, atol=1e-6
    )


if __name__ == "__main__":
    Utils.initialize_model_parallel(
        tensor_parallel_size=1, pipeline_parallel_size=2
    )
    test_local_masked_mean_matches_numpy()
    test_all_ignored_returns_exact_zero()
    test_cross_rank_token_weighted_mean_reduction()

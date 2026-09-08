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

import copy
import dataclasses
import functools
import hashlib
import random
import unittest

import numpy as np
import paddle
from paddle.distributed import fleet
from paddle.distributed.fleet import distributed_model

import paddlefleet
from paddlefleet.gpt_builders import gpt_builder
from paddlefleet.models.gpt import GPTConfig
from paddlefleet.models.gpt.mtp_embedding_layer import mtp_magic_instance
from paddlefleet.training.initialize import initialize_fleet

# Same determinism flags as ci/multi-card_test.sh.
paddle.set_flags(
    {"FLAGS_embedding_deterministic": 1, "FLAGS_cudnn_deterministic": 1}
)


@dataclasses.dataclass(frozen=True)
class PPResult:
    """One PP pass: loss value + md5s of loss / param-grad."""

    loss: float
    md5s: tuple[str, str]


def _seed_everything(seed=46):
    # per-rank offset -> each PP stage builds distinct, reproducible weights
    seed += 100 * paddlefleet.parallel_state.get_pipeline_model_parallel_rank()
    random.seed(seed)
    np.random.seed(seed)
    paddle.manual_seed(seed)


def _make_recompute_config(
    *,
    magic_send: bool,
    train_mtp_only: bool,
    num_nextn: int = 1,
    hidden_size: int = 128,
    seq_len: int = 32,
    vocab_size: int = 512,
) -> GPTConfig:
    # full/uniform/num_layers=1 -> need_full_recompute() is True
    return GPTConfig(
        vocab_size=vocab_size,
        max_sequence_length=seq_len,
        num_hidden_layers=4,
        hidden_size=hidden_size,
        num_attention_heads=4,
        num_key_value_heads=4,
        first_k_dense_replace=1,
        intermediate_size=hidden_size * 2,
        normalization="RMSNorm",
        hidden_dropout_prob=0.0,
        attention_dropout=0.0,
        use_cpu_initialization=True,
        parallel_output=True,
        tie_word_embeddings=not magic_send,  # magic send forbids weight tying
        position_embedding_type="rope",
        rotary_percent=1.0,
        rotary_base=10000,
        rope_scaling=1.0,
        init_method=functools.partial(paddle.nn.init.xavier_uniform_, gain=1.0),
        output_layer_init_method=functools.partial(
            paddle.nn.init.xavier_uniform_, gain=1.0
        ),
        use_qk_norm=True,
        # magic-send validation requires pp > 1 at construction
        pipeline_model_parallel_size=4,
        enable_mtp_magic_send=magic_send,
        train_mtp_only=train_mtp_only,
        num_nextn_predict_layers=num_nextn,
        recompute_granularity="full",
        recompute_method="uniform",
        recompute_num_layers=1,
    )


def _baseline_config(config):
    # Same model with recompute disabled.
    baseline = copy.deepcopy(config)
    baseline.recompute_granularity = None
    baseline.recompute_method = None
    baseline.recompute_num_layers = None
    return baseline


def _make_inputs(config):
    paddle.manual_seed(46)  # plain seed -> identical inputs on every rank
    batch_size = 2
    num_nextn = config.num_nextn_predict_layers
    seq_len, vocab_size = config.max_sequence_length, config.vocab_size
    data = paddle.randint(0, vocab_size, [batch_size, seq_len + num_nextn + 1])
    position_ids = (
        paddle.arange(seq_len + num_nextn).unsqueeze(0).expand([batch_size, -1])
    )
    return (
        {"input_ids": [data[:, :-1]], "position_ids": [position_ids]},
        [data[:, 1:]],
    )


@functools.cache
def _init_fleet_once():
    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": 1,
        "mp_degree": 1,
        "pp_degree": 4,
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
    initialize_fleet(strategy)


def run_pp(config, inputs):
    """One PP forward/backward pass; returns loss + gradient md5 digests."""
    run_config = copy.deepcopy(config)  # carries pipeline_model_parallel_size=4
    _init_fleet_once()
    _seed_everything()
    model = distributed_model(
        gpt_builder(
            run_config,
            num_stages=4,
            seg_method="layer:TransformerLayer",
        )
    )
    loss = model.forward_backward_pipeline(inputs)
    loss_v = float(loss)
    assert np.isfinite(loss_v), f"loss not finite: {loss_v}"

    param_md5s = []
    for name, param in sorted(dict(model.named_parameters()).items()):
        grad = getattr(param, "main_grad", None)
        if grad is None:
            grad = param.grad
        assert grad is not None, f"parameter {name} has no gradient"
        param_md5s.append(grad._md5sum())
    return PPResult(
        loss=loss_v,
        md5s=(
            loss._md5sum(),
            hashlib.md5("".join(param_md5s).encode()).hexdigest(),
        ),
    )


class TestMTPRecomputePaths(unittest.TestCase):
    def _assert_bit_exact(self, tag, config, inputs):
        baseline = run_pp(_baseline_config(config), inputs)
        recompute = run_pp(config, inputs)
        print(
            f"[{tag}] PP=4 OK loss={baseline.loss:.6f} "
            f"match={tuple(b == r for b, r in zip(baseline.md5s, recompute.md5s))}"
        )
        self.assertEqual(
            recompute.md5s,
            baseline.md5s,
            msg=(
                f"[{tag}] not bit-identical\n"
                f"  baseline : {baseline.md5s}\n"
                f"  recompute: {recompute.md5s}"
            ),
        )

    def test_magic_send_recompute(self):
        # full_recompute call site: magic-send branch (multi_token_prediction.py:1445)
        config = _make_recompute_config(magic_send=True, train_mtp_only=False)
        inputs = _make_inputs(config)
        # no dataloader in standalone PP test; every rank holds the full tensor
        mtp_magic_instance.set_data({"input_ids": [inputs[0]["input_ids"][0]]})
        self._assert_bit_exact("magic_send", config, inputs)

    def test_train_mtp_only_recompute(self):
        # full_recompute call site: train_mtp_only loop (multi_token_prediction.py:1658)
        config = _make_recompute_config(magic_send=False, train_mtp_only=True)
        self._assert_bit_exact("train_mtp_only", config, _make_inputs(config))

    def test_pp_with_mtp(self):
        # full_recompute call site: concat+split default path (multi_token_prediction.py:1725)
        config = _make_recompute_config(
            magic_send=False,
            train_mtp_only=False,
            num_nextn=2,
            hidden_size=512,
            seq_len=128,
            vocab_size=1024,
        )
        self._assert_bit_exact("concat+split", config, _make_inputs(config))


if __name__ == "__main__":
    unittest.main()

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

import functools
import random
import unittest

import numpy as np
import paddle
from paddle.distributed import fleet
from paddle.distributed.fleet import distributed_model
from paddle.distributed.fleet.meta_parallel import NoPipelineParallel

import paddlefleet
from paddlefleet.gpt_builders import gpt_builder
from paddlefleet.models.gpt import GPTConfig
from paddlefleet.models.gpt.mtp_embedding_layer import mtp_magic_instance
from paddlefleet.training.initialize import initialize_fleet

PP_DEGREE = 4
# fleet can only be initialized once per process
_FLEET_READY = False
_XAVIER = functools.partial(paddle.nn.init.xavier_uniform_, gain=1.0)


def _set_random_seed(seed):
    seed += 100 * paddlefleet.parallel_state.get_pipeline_model_parallel_rank()
    random.seed(seed)
    np.random.seed(seed)
    paddle.manual_seed(seed)


def _make_config(
    *,
    magic_send: bool,
    train_mtp_only: bool,
    num_nextn: int = 1,
    pp_size: int = 1,
    hidden_size: int = 128,
    seq_len: int = 32,
    vocab_size: int = 512,
) -> GPTConfig:
    # recompute full/uniform/num_layers=1 makes _use_checkpoint() return True
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
        init_method=_XAVIER,
        output_layer_init_method=_XAVIER,
        use_qk_norm=True,
        pipeline_model_parallel_size=pp_size,
        enable_mtp_magic_send=magic_send,
        train_mtp_only=train_mtp_only,
        num_nextn_predict_layers=num_nextn,
        recompute_granularity="full",
        recompute_method="uniform",
        recompute_num_layers=1,
    )


def _make_inputs(batch_size, config):
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


def run_pp(batch_size, config, inputs=None):
    """Run PP forward/backward, comparing against a single-device baseline.

    magic send requires pipeline_model_parallel_size > 1, so no single-device
    baseline is possible and the loss is only checked to be finite.
    """
    global _FLEET_READY
    if not config.enable_mtp_magic_send:
        # Single-device baseline (PP=1), same seeds as the PP run below.
        config.pipeline_model_parallel_size = 1
        random.seed(46)
        np.random.seed(46)
        paddle.manual_seed(46)
        base_model = NoPipelineParallel(
            gpt_builder(config, num_stages=1), fleet.DistributedStrategy()
        )
        paddle.manual_seed(46)
        loss_baseline = base_model.forward_backward_pipeline(
            _make_inputs(batch_size, config)
        )

    config.pipeline_model_parallel_size = PP_DEGREE
    if not _FLEET_READY:
        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": 1,
            "pp_degree": config.pipeline_model_parallel_size,
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
        _FLEET_READY = True

    _set_random_seed(46)
    gpt_pipe_model = distributed_model(
        gpt_builder(
            config,
            num_stages=config.pipeline_model_parallel_size,
            seg_method="layer:TransformerLayer",
        )
    )
    paddle.manual_seed(46)
    if inputs is None:
        inputs = _make_inputs(batch_size, config)
    loss = gpt_pipe_model.forward_backward_pipeline(inputs)
    loss_v = float(loss)
    assert np.isfinite(loss_v), f"loss not finite: {loss_v}"
    if not config.enable_mtp_magic_send:
        assert loss == loss_baseline
    return loss_v


class TestMTPRecomputePaths(unittest.TestCase):
    def test_magic_send_recompute(self):
        # _use_checkpoint() call site: magic-send branch (:1342)
        config = _make_config(
            magic_send=True, train_mtp_only=False, pp_size=PP_DEGREE
        )
        # no dataloader in standalone PP test; every rank holds the full tensor
        paddle.manual_seed(46)
        inputs = _make_inputs(2, config)
        mtp_magic_instance.set_data({"input_ids": [inputs[0]["input_ids"][0]]})
        loss_v = run_pp(2, config, inputs=inputs)
        print(f"[magic_send] PP={PP_DEGREE} OK loss={loss_v:.6f}")

    def test_train_mtp_only_recompute(self):
        # _use_checkpoint() call site: train_mtp_only loop (:1552)
        config = _make_config(magic_send=False, train_mtp_only=True)
        loss_v = run_pp(2, config)
        print(f"[train_mtp_only] PP={PP_DEGREE} OK loss={loss_v:.6f}")

    def test_pp_with_mtp(self):
        # _use_checkpoint() call site: concat+split default path (:1619)
        config = _make_config(
            magic_send=False,
            train_mtp_only=False,
            num_nextn=2,
            hidden_size=512,
            seq_len=128,
            vocab_size=1024,
        )
        loss_v = run_pp(2, config)
        print(f"[concat+split] PP={PP_DEGREE} OK loss={loss_v:.6f}")


if __name__ == "__main__":
    unittest.main()

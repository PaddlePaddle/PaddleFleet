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
import pprint
import random
import subprocess
import unittest

import numpy as np
import paddle
from paddle.distributed import fleet
from paddle.distributed.fleet import distributed_model

import paddlefleet
from paddlefleet.gpt_builders import gpt_builder
from paddlefleet.models.gpt import GPTConfig
from paddlefleet.training.initialize import initialize_fleet

PP_DEGREE = 4
MTP_DEGREE = 3


def get_gpu_models_via_nvidia_smi():
    try:
        output = subprocess.check_output(
            "nvidia-smi --query-gpu=name --format=csv,noheader", shell=True
        )
        models = output.decode().strip().replace("NVIDIA", "")
        return models
    except Exception as e:
        return ["Unknown"]


def judge_machine_type():
    if not paddle.is_compiled_with_cuda():
        return "No CUDA GPU"
    models = get_gpu_models_via_nvidia_smi()
    for model in models:
        name = model.upper()
        if "V" in name:
            return "V"
        elif "H" in name:
            return "H"
        elif "B" in name:
            return "B"


def _set_random_seed(
    seed_: int,
    data_parallel_random_init: bool = False,
    te_rng_tracker: bool = False,
    inference_rng_tracker: bool = False,
    use_cudagraphable_rng: bool = False,
):
    """Set random seed for reproducibility."""
    if seed_ is not None and seed_ > 0:
        # Ensure that different pipeline MP stages get different seeds.
        seed = seed_ + (
            100 * paddlefleet.parallel_state.get_pipeline_model_parallel_rank()
        )
        # Ensure different data parallel ranks get different seeds
        if data_parallel_random_init:
            seed = seed + (
                10 * paddlefleet.parallel_state.get_data_parallel_rank()
            )
        random.seed(seed)
        np.random.seed(seed)
        paddle.manual_seed(seed)

        if (
            paddle.distributed.is_initialized()
            and paddle.cuda.device_count() > 0
        ):
            paddlefleet.tensor_parallel.model_parallel_cuda_manual_seed(
                seed,
                te_rng_tracker,
                inference_rng_tracker,
                use_cudagraphable_rng,
            )
    else:
        raise ValueError(f"Seed ({seed_}) should be a positive integer.")


def run_pp(
    seed,
    batch_size,
    seq_len,
    vocab_size,
    config,
    forward_backward_overlap_scheduler=False,
):
    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": 1,
        "mp_degree": config.tensor_model_parallel_size,
        "pp_degree": config.pipeline_model_parallel_size,
        "sharding_degree": 1,
        "sep_degree": 1,
        "cp_degree": 1,
        "ep_degree": config.tensor_model_parallel_size,
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
        "pp_configs": {
            "forward_backward_overlap_scheduler": forward_backward_overlap_scheduler,
            "overlap_p2p_comm": True,
            "enable_dynamic_shape": True,
        },
    }
    micro_batch_size = 1
    num_acc = batch_size // micro_batch_size
    strategy.pipeline_configs = {
        "accumulate_steps": num_acc,
        "micro_batch_size": micro_batch_size,
    }
    initialize_fleet(strategy)

    _set_random_seed(seed)

    gpt_model = gpt_builder(
        config,
        num_stages=config.pipeline_model_parallel_size,
        seg_method="layer:TransformerLayer|EmptyLayer",
    )
    gpt_model = paddle.amp.decorate(
        models=gpt_model, optimizers=None, level="O2", dtype="bfloat16"
    )

    gpt_pipe_model = distributed_model(gpt_model)

    data = paddle.randint(
        low=0,
        high=vocab_size,
        shape=(micro_batch_size, seq_len + MTP_DEGREE + 1),
    )
    input_ids = data[:, :-1]
    labels = data[:, 1:]
    position_ids = paddle.to_tensor(data, dtype=paddle.int64).repeat(
        (micro_batch_size, 1)
    )

    inputs = (
        {
            "input_ids": [input_ids] * num_acc,
            "position_ids": [position_ids] * num_acc,
        },
        [labels] * num_acc,
    )

    loss = gpt_pipe_model.forward_backward_pipeline(inputs, None)
    return loss, gpt_pipe_model


class TestPP(unittest.TestCase):
    def setUp(self):
        self.seed = 46
        self.batch_size = 12
        self.seq_len = 128
        self.vocab_size = 1024

    def test_pp(self):
        config = GPTConfig(
            moe_expert_fusion=False,
            vocab_size=self.vocab_size,
            max_sequence_length=self.seq_len,
            num_hidden_layers=11,
            hidden_size=512,
            num_attention_heads=4,
            intermediate_size=1024,
            normalization="RMSNorm",
            hidden_dropout_prob=0.0,
            attention_dropout=0.0,
            use_cpu_initialization=True,
            parallel_output=True,
            tie_word_embeddings=True,
            position_embedding_type="rope",
            rotary_percent=1.0,
            rotary_base=10000,
            rope_scaling=1.0,
            init_method=functools.partial(
                paddle.nn.init.xavier_uniform_, gain=1.0
            ),
            output_layer_init_method=functools.partial(
                paddle.nn.init.xavier_uniform_, gain=1.0
            ),
            use_qk_norm=True,
            num_empty_layers_add_in_head=2,
            num_empty_layers_add_in_tail=3,
            pipeline_model_parallel_size=PP_DEGREE,
            virtual_pipeline_model_parallel_size=2,
            tensor_model_parallel_size=2,
            expert_model_parallel_size=2,
            sequence_parallel=True,
            n_shared_experts=1,
            n_routed_experts=8,
            moe_intermediate_size=1024,
            bf16=True,
            gated_linear_unit=True,
            bias_activation_fusion=True,
            num_nextn_predict_layers=MTP_DEGREE,
        )

        overlap_loss, overlap_gpt_model = run_pp(
            self.seed,
            self.batch_size,
            self.seq_len,
            self.vocab_size,
            config,
        )

        print("PP loss MD5:", overlap_loss._md5sum())

        rst = {}
        for name, param in overlap_gpt_model.named_parameters():
            if param.grad is not None:
                rst[name] = param.grad._md5sum()

        pp = pprint.PrettyPrinter(depth=None, width=200, compact=False)
        pp.pprint(rst)

        if judge_machine_type() == "H":
            assert overlap_loss._md5sum() == "14052829ee7ced24f3b794e6eadf5c00"
            if paddle.distributed.get_rank() == 0:
                baseline = {
                    "_layers.shared_layers.embed.embedding.embed_tokens.weight": "3ee515720ce20d3ed1866c3d3c4037a9",
                    "_layers.9.0.input_layernorm.weight": "dfa7d675ec2a1519f1bd30895fda4f66",
                    "_layers.9.0.self_attn.o_proj.weight": "aca7d8a8f0f4034c08680d07e362fa22",
                    "_layers.9.0.self_attn.qkv_proj.weight": "e7c0917441d6b2e8d9474ea8cdeae097",
                    "_layers.9.0.self_attn.q_norm.weight": "ca1916ef879246092a57f1c67e963267",
                    "_layers.9.0.self_attn.k_norm.weight": "c6d102894ea24247db65dbb1ff73d0d4",
                    "_layers.9.0.post_attention_layernorm.weight": "845024ef9bd2992f1a88daf8bd26cdbf",
                    "_layers.9.0.mlp.gate.weight": "4ddef3d3662d386db0124b8125c15cc0",
                    "_layers.9.0.mlp.experts.0.up_gate_proj.weight": "de4ede532923ee80a868f7fad7e00a9a",
                    "_layers.9.0.mlp.experts.0.down_proj.weight": "5b74eeeb430da7ee94478093a0bc7e41",
                    "_layers.9.0.mlp.experts.1.up_gate_proj.weight": "6f230e10676f1c3f7d9e8fcf70d6764c",
                    "_layers.9.0.mlp.experts.1.down_proj.weight": "0c8d4b17b7050f31dabbaffa8a900b9b",
                    "_layers.9.0.mlp.experts.2.up_gate_proj.weight": "e1f4db2de1ba0cbbab51e878c0f88693",
                    "_layers.9.0.mlp.experts.2.down_proj.weight": "1aa273dfb1b7752a8aabe5230bf09ae7",
                    "_layers.9.0.mlp.experts.3.up_gate_proj.weight": "04dfa44fccf5a50e13d3002cd2107ce1",
                    "_layers.9.0.mlp.experts.3.down_proj.weight": "58704896976f2b1171c442f3637a1f91",
                    "_layers.9.0.mlp.shared_experts.up_gate_proj.weight": "6911aaf197970dfa7f5503c337ad4fa7",
                    "_layers.9.0.mlp.shared_experts.down_proj.weight": "787bd23319f2c48afbe93bcdbb50bfba",
                    "_layers.9.1.input_layernorm.weight": "eea3047af2b6b0750c3fb9fcd4c90c10",
                    "_layers.9.1.self_attn.o_proj.weight": "323d90b3845f4e84146631321f39d953",
                    "_layers.9.1.self_attn.qkv_proj.weight": "29540e45a51e37bb1ffc234534699767",
                    "_layers.9.1.self_attn.q_norm.weight": "a229b23e0f019698c33d89686dd67122",
                    "_layers.9.1.self_attn.k_norm.weight": "a35b9de4a0303c0341b16a4e496ded9b",
                    "_layers.9.1.post_attention_layernorm.weight": "95e478361ec7b8994c5465a571f5e1fd",
                    "_layers.9.1.mlp.gate.weight": "547e09c8118ee9df6aaa3ca345f2b2ab",
                    "_layers.9.1.mlp.experts.0.up_gate_proj.weight": "53e1068c7a960ae4e4d02dcf4c63027b",
                    "_layers.9.1.mlp.experts.0.down_proj.weight": "44e23c91eb053937055ffcfb11cb0dc4",
                    "_layers.9.1.mlp.experts.1.up_gate_proj.weight": "0cdab3c1b8896c3640a65955ab4979a3",
                    "_layers.9.1.mlp.experts.1.down_proj.weight": "7215ffc7b17060084a1ad3c43e751a61",
                    "_layers.9.1.mlp.experts.2.up_gate_proj.weight": "323b93acdb0eafc15567857ee7029971",
                    "_layers.9.1.mlp.experts.2.down_proj.weight": "d9591a97771ce9a9ef309ad12626a5e4",
                    "_layers.9.1.mlp.experts.3.up_gate_proj.weight": "f2cce559c596a19f29f3091075c4a030",
                    "_layers.9.1.mlp.experts.3.down_proj.weight": "af027e845be2002018dd9b79f45be0bf",
                    "_layers.9.1.mlp.shared_experts.up_gate_proj.weight": "555a90ebba047b581d337c4c4d3ce257",
                    "_layers.9.1.mlp.shared_experts.down_proj.weight": "bd57563560fe3eb30baec77d7fea0555",
                }
                for name, param in overlap_gpt_model.named_parameters():
                    assert param.grad._md5sum() == baseline[name], (
                        f"{name}'s grad has diff"
                    )
        elif judge_machine_type() == "B":
            assert overlap_loss._md5sum() == "f7d9cb770c3de0a6f647b44a56080a66"
            if paddle.distributed.get_rank() == 0:
                baseline = {
                    "_layers.shared_layers.embed.embedding.embed_tokens.weight": "9a17046eda296d1701e118f60e170425",
                    "_layers.9.0.input_layernorm.weight": "8d8c5c38c1cd66028746abdcf0ddda80",
                    "_layers.9.0.self_attn.o_proj.weight": "8c16aead6fe11a60018481c5496b355b",
                    "_layers.9.0.self_attn.qkv_proj.weight": "b02dacc17cd0bd8628efe179251c4efe",
                    "_layers.9.0.self_attn.q_norm.weight": "103940b05c7865e190a27b2d93237a81",
                    "_layers.9.0.self_attn.k_norm.weight": "3ea9da49daf21d46ac1585da2074c5db",
                    "_layers.9.0.post_attention_layernorm.weight": "227729d66686f40e920dca574660beee",
                    "_layers.9.0.mlp.gate.weight": "e3cf3188c37ba65371179e8f95ce1745",
                    "_layers.9.0.mlp.experts.0.up_gate_proj.weight": "2f22b26b3c9f2646da6b84dda263b763",
                    "_layers.9.0.mlp.experts.0.down_proj.weight": "66fc655f9592d2105a2d45ea759d33df",
                    "_layers.9.0.mlp.experts.1.up_gate_proj.weight": "6d7db6fa048f923e9aafdc8df89150f8",
                    "_layers.9.0.mlp.experts.1.down_proj.weight": "af643d90f73f2d71254d1a68ddc9da4f",
                    "_layers.9.0.mlp.experts.2.up_gate_proj.weight": "50f20bf98c9ec6f579d6e9e211667052",
                    "_layers.9.0.mlp.experts.2.down_proj.weight": "000da3155e150c5146e8ec20776d609c",
                    "_layers.9.0.mlp.experts.3.up_gate_proj.weight": "620b08c58e72e2362cb0c34f4cca9a2e",
                    "_layers.9.0.mlp.experts.3.down_proj.weight": "927b47f263c0a79b9d5faa0fa112d39d",
                    "_layers.9.0.mlp.shared_experts.up_gate_proj.weight": "78122a5edcb082d26bb50c8e76427359",
                    "_layers.9.0.mlp.shared_experts.down_proj.weight": "f3e1fe4e26b932ad6dc28c5c7fd664af",
                    "_layers.9.1.input_layernorm.weight": "71e8e4829bbce5ae8422e9c24c02a3cc",
                    "_layers.9.1.self_attn.o_proj.weight": "6c634637ac2cb5a3c4a1161963600258",
                    "_layers.9.1.self_attn.qkv_proj.weight": "e67d9340a8fbd308995e22fb18cd240f",
                    "_layers.9.1.self_attn.q_norm.weight": "b3d77c93f82111b60d987f52cb652ac0",
                    "_layers.9.1.self_attn.k_norm.weight": "cd850cd399f3d923f60a6bfa393bd342",
                    "_layers.9.1.post_attention_layernorm.weight": "4dc17dfc4f0c887b650297ce16a7c66d",
                    "_layers.9.1.mlp.gate.weight": "c25852619827b20ac10bc488be197378",
                    "_layers.9.1.mlp.experts.0.up_gate_proj.weight": "f6b06c203ffb5b4d3c66fc2d4c4b46ac",
                    "_layers.9.1.mlp.experts.0.down_proj.weight": "53df49e1f5d5b271566210cc786b8774",
                    "_layers.9.1.mlp.experts.1.up_gate_proj.weight": "deae484d844ac4f01c93c0b536e2a8ee",
                    "_layers.9.1.mlp.experts.1.down_proj.weight": "9a359baaa9accaee485372f7c8d38964",
                    "_layers.9.1.mlp.experts.2.up_gate_proj.weight": "d82ee7d55a0999dcfd6669b8c89eb1e9",
                    "_layers.9.1.mlp.experts.2.down_proj.weight": "39bda613164d88e96581db740188c9d8",
                    "_layers.9.1.mlp.experts.3.up_gate_proj.weight": "825f03f096100c90140d695662107e0e",
                    "_layers.9.1.mlp.experts.3.down_proj.weight": "7f18f7bc55b6f5560b3932149c63f9f4",
                    "_layers.9.1.mlp.shared_experts.up_gate_proj.weight": "ff00c06aead3d2ac6e02fdfb7eb1ccec",
                    "_layers.9.1.mlp.shared_experts.down_proj.weight": "114d9dafe46c2a193d9ec048170cd115",
                }
                for name, param in overlap_gpt_model.named_parameters():
                    assert param.grad._md5sum() == baseline[name], (
                        f"{name}'s grad has diff"
                    )


if __name__ == "__main__":
    unittest.main()

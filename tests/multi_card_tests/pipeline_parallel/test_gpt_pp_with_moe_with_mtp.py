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
import os
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
# skip test for paddle pr 79368 merge
REPO_FLAG = os.getenv("repo_flag")
SKIP_TESTS = REPO_FLAG != "paddlefleet"


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


def judge_h_subtype():
    """Distinguish H800 vs H20 within the Hopper ("H") family."""
    name = "".join(get_gpu_models_via_nvidia_smi()).upper()
    if "H800" in name:
        return "H800"
    if "H20" in name:
        return "H20"
    return None


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
            moe_token_dispatcher_type="deepep",
            num_nextn_predict_layers=MTP_DEGREE,
            mtp_loss_scaling_factor=0.3,
            overlap_p2p_comm=False,
            batch_p2p_comm=True,
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
            actual_md5 = overlap_loss._md5sum()
            if judge_h_subtype() == "H800":
                expected_md5 = "d0cc18f8919d2968ac0ad5d577650d39"
            else:
                expected_md5 = "74ede7d44c9d2232b5931be27fbd3fdc"
            print(
                f"PP loss MD5 - Actual: {actual_md5}, Expected: {expected_md5}"
            )
            assert actual_md5 == expected_md5, (
                f"PP loss MD5 mismatch! Actual: {actual_md5}, Expected: {expected_md5}"
            )
            if paddle.distributed.get_rank() == 0:
                if judge_h_subtype() == "H800":
                    baseline = {
                        "_layers.shared_layers.embed.embedding.embed_tokens.weight": "60e28736b728cd43e8ced6e14129dec0",
                        "_layers.9.0.input_layernorm.weight": "e253a133c9ab652d77e2e640096ccdfa",
                        "_layers.9.0.self_attn.o_proj.weight": "8153acd0cfe6b3e7a4d8c4ca2a2bf658",
                        "_layers.9.0.self_attn.qkv_proj.weight": "fbff0aa3575ff970bcd35ddce4e51122",
                        "_layers.9.0.self_attn.q_norm.weight": "928e8002da650e48ae5f091087785667",
                        "_layers.9.0.self_attn.k_norm.weight": "10165e00c2567c1fe1f5e3a8078ab1c0",
                        "_layers.9.0.post_attention_layernorm.weight": "d9b7700033d8da3198f163c92ec0bed9",
                        "_layers.9.0.mlp.gate.weight": "7e5190c81644416bdc2689c64e102489",
                        "_layers.9.0.mlp.experts.0.up_gate_proj.weight": "aa3fe661b26badc953d9aa6066ea145c",
                        "_layers.9.0.mlp.experts.0.down_proj.weight": "91efad2ebb3e49c0e1ceaf28347bb50b",
                        "_layers.9.0.mlp.experts.1.up_gate_proj.weight": "2c2c96ce461aa28223280e749f30019b",
                        "_layers.9.0.mlp.experts.1.down_proj.weight": "e9c4f6b3e53de5f06a1eacc1a3a4dcfe",
                        "_layers.9.0.mlp.experts.2.up_gate_proj.weight": "877816e7b626d34d34f080f10b415023",
                        "_layers.9.0.mlp.experts.2.down_proj.weight": "1990ffedad8cf1acbc7f4b68a86ee379",
                        "_layers.9.0.mlp.experts.3.up_gate_proj.weight": "da2c711d3b55c5da0c40ecc4b49334a1",
                        "_layers.9.0.mlp.experts.3.down_proj.weight": "75364c05636e1d773d902fe73291254d",
                        "_layers.9.0.mlp.shared_experts.up_gate_proj.weight": "067473ff9dfcdecbcf949942a3caeba9",
                        "_layers.9.0.mlp.shared_experts.down_proj.weight": "9f47f96c26d81b1a6647d553b3b3af3f",
                        "_layers.9.1.input_layernorm.weight": "5a0ec38bbfa92c17ddf5dcb5cf4ff2ad",
                        "_layers.9.1.self_attn.o_proj.weight": "85f4cece84fb8cd7616538aee2acd57f",
                        "_layers.9.1.self_attn.qkv_proj.weight": "9c9123fd23bce9bba931b971b2eedb21",
                        "_layers.9.1.self_attn.q_norm.weight": "5ed69f04dfd86bbbf91cdfeac650b52d",
                        "_layers.9.1.self_attn.k_norm.weight": "00dc120fab3d152b284730d9b6efe8c2",
                        "_layers.9.1.post_attention_layernorm.weight": "0967ad75c77c929226e2bc1d39ce0986",
                        "_layers.9.1.mlp.gate.weight": "bf09882de3516988b785ab1ec9c65882",
                        "_layers.9.1.mlp.experts.0.up_gate_proj.weight": "1f520ba660871b094d2fb313f3568696",
                        "_layers.9.1.mlp.experts.0.down_proj.weight": "40a7ed164e23b0b62ee32dc858396212",
                        "_layers.9.1.mlp.experts.1.up_gate_proj.weight": "c2434f9506de466dab2e756678e77012",
                        "_layers.9.1.mlp.experts.1.down_proj.weight": "7dece11d41ee3ec2aa7abf287f6c7d60",
                        "_layers.9.1.mlp.experts.2.up_gate_proj.weight": "ea0ab48ef0723398fbc0743c6240772c",
                        "_layers.9.1.mlp.experts.2.down_proj.weight": "15a5fc94657b70269a9666ba85108120",
                        "_layers.9.1.mlp.experts.3.up_gate_proj.weight": "73020700f0284cc6e15a65462bafd2c5",
                        "_layers.9.1.mlp.experts.3.down_proj.weight": "bf038050b26286ea04fd86f811f13495",
                        "_layers.9.1.mlp.shared_experts.up_gate_proj.weight": "d772b98cd764a61c73966e7110edb5c9",
                        "_layers.9.1.mlp.shared_experts.down_proj.weight": "2c39784fd16c4a4dd700eb7848dc35fb",
                    }
                else:
                    baseline = {
                        "_layers.9.0.input_layernorm.weight": "3f3dff970ad76cbf9c939be66fea8822",
                        "_layers.9.0.mlp.experts.0.down_proj.weight": "1fbc9e3fa7263b31ba6c38c819540dc4",
                        "_layers.9.0.mlp.experts.0.up_gate_proj.weight": "9ec2b0ae147ebd1bd0e55c9dde2a16ea",
                        "_layers.9.0.mlp.experts.1.down_proj.weight": "cfa3a2355deb957f1757d156fd4f775d",
                        "_layers.9.0.mlp.experts.1.up_gate_proj.weight": "8970fec2d4d65be324c110f91222c465",
                        "_layers.9.0.mlp.experts.2.down_proj.weight": "d01014b65d37a83b60e9e012804ce11b",
                        "_layers.9.0.mlp.experts.2.up_gate_proj.weight": "82c96dbf7b5fbcb57f26367f818d7cc7",
                        "_layers.9.0.mlp.experts.3.down_proj.weight": "9508a9a1c5fc86501d2d0ffbf55be6a1",
                        "_layers.9.0.mlp.experts.3.up_gate_proj.weight": "054dcd1b7b2b30d55c4cd0e95d1fe912",
                        "_layers.9.0.mlp.gate.weight": "326c4255f2e078df3220816b9614a2bf",
                        "_layers.9.0.mlp.shared_experts.down_proj.weight": "c410d043dd60c8e23af426a035ef56fa",
                        "_layers.9.0.mlp.shared_experts.up_gate_proj.weight": "4c884ccb4457d1e2712f18935636d5c6",
                        "_layers.9.0.post_attention_layernorm.weight": "5b7a485224453d4875ce3c29cc4ad90a",
                        "_layers.9.0.self_attn.k_norm.weight": "2080002bd2a3e9ca095b6c37e583e9bc",
                        "_layers.9.0.self_attn.o_proj.weight": "87a59a37587e456740517a57e0e12c30",
                        "_layers.9.0.self_attn.q_norm.weight": "49fc9ce43146b21dbba4784c814b058b",
                        "_layers.9.0.self_attn.qkv_proj.weight": "71fc902c827c1b83f129641ecd13b471",
                        "_layers.9.1.input_layernorm.weight": "04d1fa20a047589fd448b8caa48be127",
                        "_layers.9.1.mlp.experts.0.down_proj.weight": "9675017a1ba8a3ac6c4d562a4c3abce9",
                        "_layers.9.1.mlp.experts.0.up_gate_proj.weight": "6f04e60729974610cebded619c73dd16",
                        "_layers.9.1.mlp.experts.1.down_proj.weight": "86ff9427e6e1c640de7f4ad1ba9b699e",
                        "_layers.9.1.mlp.experts.1.up_gate_proj.weight": "652767184301223999761c00b4b83db0",
                        "_layers.9.1.mlp.experts.2.down_proj.weight": "b2d1236c286a3c0704224fe4105eca49",
                        "_layers.9.1.mlp.experts.2.up_gate_proj.weight": "b5cfa9d6c8febd618f91ac2843d50a1c",
                        "_layers.9.1.mlp.experts.3.down_proj.weight": "9fdb795a3eecba579eabdafe2d396b86",
                        "_layers.9.1.mlp.experts.3.up_gate_proj.weight": "7f9d3dbe18f41b2c012423ab5c2c0d41",
                        "_layers.9.1.mlp.gate.weight": "ebe1540d629f60d73509c43f50f3fe86",
                        "_layers.9.1.mlp.shared_experts.down_proj.weight": "33fa9c430425bb1ea4a41ee9053c4a16",
                        "_layers.9.1.mlp.shared_experts.up_gate_proj.weight": "3981e0c595ccf4149e858a25b44f684c",
                        "_layers.9.1.post_attention_layernorm.weight": "0743e39d4538d86f4231a00c4c2d9d93",
                        "_layers.9.1.self_attn.k_norm.weight": "7f315cec90cf648f938fdf5ccfde8ec4",
                        "_layers.9.1.self_attn.o_proj.weight": "1ff6d2ea8c3aab99adf7df186f22a27c",
                        "_layers.9.1.self_attn.q_norm.weight": "1edf3c866392e553217fdf24ef2cea21",
                        "_layers.9.1.self_attn.qkv_proj.weight": "e6562d4a5d8e15d16f455878c5a4a156",
                        "_layers.shared_layers.embed.embedding.embed_tokens.weight": "c711f99438d25725a43db0cd54588109",
                    }
                for name, param in overlap_gpt_model.named_parameters():
                    assert param.grad._md5sum() == baseline[name], (
                        f"{name}'s grad has diff"
                    )
        elif judge_machine_type() == "B":
            assert overlap_loss._md5sum() == "20417b4850693406ba21c6163f437ff5"
            if paddle.distributed.get_rank() == 0:
                baseline = {
                    "_layers.9.0.input_layernorm.weight": "fa39ce310e5df49a15154e343baab49d",
                    "_layers.9.0.mlp.experts.0.down_proj.weight": "68907583a113a85861ed8713267489cf",
                    "_layers.9.0.mlp.experts.0.up_gate_proj.weight": "2779d45dda5d976c2bd4fc01b5ea0bdd",
                    "_layers.9.0.mlp.experts.1.down_proj.weight": "d9334943c2d13473c329dcfd10e8fe46",
                    "_layers.9.0.mlp.experts.1.up_gate_proj.weight": "ac311162c065e3586a4bd8ac501f5680",
                    "_layers.9.0.mlp.experts.2.down_proj.weight": "05902078993f3930556b359c80b4d562",
                    "_layers.9.0.mlp.experts.2.up_gate_proj.weight": "9d633bc2c1f04aa0114a874c644b863a",
                    "_layers.9.0.mlp.experts.3.down_proj.weight": "e6f68a3cd18983ce2fd2c2afe9b1e65f",
                    "_layers.9.0.mlp.experts.3.up_gate_proj.weight": "f925e26fd619e86f618a94548f50437c",
                    "_layers.9.0.mlp.gate.weight": "fc38e5b35d9c31f427e15ecdb14e89a7",
                    "_layers.9.0.mlp.shared_experts.down_proj.weight": "97de0df4fc1bd4f34d95d051545e33d3",
                    "_layers.9.0.mlp.shared_experts.up_gate_proj.weight": "c5db587a8aa49af6493975681659ea30",
                    "_layers.9.0.post_attention_layernorm.weight": "b28dc1d61ca7944ea09780f935a3b167",
                    "_layers.9.0.self_attn.k_norm.weight": "f7a5e968ea12a454a6d159f2cec26b01",
                    "_layers.9.0.self_attn.o_proj.weight": "8d745df57fcc2b9b2075c3f0dbd9e945",
                    "_layers.9.0.self_attn.q_norm.weight": "b93863d48b030e4c9a4357eeb3374855",
                    "_layers.9.0.self_attn.qkv_proj.weight": "495087bf6c3b33df92bb46c41b3b9b21",
                    "_layers.9.1.input_layernorm.weight": "bc42431947258990dcb6121ea185411a",
                    "_layers.9.1.mlp.experts.0.down_proj.weight": "5d072dca7bec57ad89de03d05f9d72f1",
                    "_layers.9.1.mlp.experts.0.up_gate_proj.weight": "04dab70cb4caaae68ab1c41f1d9f3917",
                    "_layers.9.1.mlp.experts.1.down_proj.weight": "c10a4f51fcd85ab30f036be1bdf32782",
                    "_layers.9.1.mlp.experts.1.up_gate_proj.weight": "96acf0149d612c945ccd57af55f7968b",
                    "_layers.9.1.mlp.experts.2.down_proj.weight": "8486f3979555f03a032a87c324065509",
                    "_layers.9.1.mlp.experts.2.up_gate_proj.weight": "c9519f8f93740e351b0f8e9b09eb2dc5",
                    "_layers.9.1.mlp.experts.3.down_proj.weight": "a1ba35f42c0c75ac5021355c7ff93bd8",
                    "_layers.9.1.mlp.experts.3.up_gate_proj.weight": "00e5c59df6cbfc2c6a7936f8e063b27f",
                    "_layers.9.1.mlp.gate.weight": "bab81f505d2ea1e4847f4780ccf57f9c",
                    "_layers.9.1.mlp.shared_experts.down_proj.weight": "5de69600a1f082acbe09b189be70b66b",
                    "_layers.9.1.mlp.shared_experts.up_gate_proj.weight": "e35c1974cba8d89f54d54b96c7abc33e",
                    "_layers.9.1.post_attention_layernorm.weight": "1db9a16ff23f09071e0ab78a7ede67bc",
                    "_layers.9.1.self_attn.k_norm.weight": "8cfc1772b40342ae1b0f119817827065",
                    "_layers.9.1.self_attn.o_proj.weight": "47b5a7b254a16dee488385202e353a6f",
                    "_layers.9.1.self_attn.q_norm.weight": "92b92f322298279dbc1404281931409c",
                    "_layers.9.1.self_attn.qkv_proj.weight": "f17dc2c4efa90c399eee33606fedf48b",
                    "_layers.shared_layers.embed.embedding.embed_tokens.weight": "418d8e565f3124eb2b46c97839000bd3",
                }
                mismatches = {}
                actual_all = {}
                for name, param in overlap_gpt_model.named_parameters():
                    if param.grad is None:
                        continue
                    actual_md5 = param.grad._md5sum()
                    actual_all[name] = actual_md5
                    expected = baseline.get(name)
                    if expected != actual_md5:
                        mismatches[name] = {
                            "actual": actual_md5,
                            "expected": expected,
                        }

                if mismatches:
                    print("===== MISMATCHED KEYS =====")
                    pp = pprint.PrettyPrinter(
                        depth=None, width=200, compact=False
                    )
                    pp.pprint(mismatches)

                    print("===== FULL ACTUAL DICT =====")
                    pp.pprint(actual_all)

                assert not mismatches, (
                    f"{len(mismatches)} param(s) grad mismatch: {list(mismatches.keys())}"
                )


if __name__ == "__main__":
    unittest.main()

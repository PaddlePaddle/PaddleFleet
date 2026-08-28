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
        low=0, high=vocab_size, shape=(micro_batch_size, seq_len + 1)
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
        if (
            not paddle.device.current_device_is_cpu
            and paddle.device.get_device_capability()[0] < 9
        ):
            return
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
            moe_token_dispatcher_type="deepep",
            gated_linear_unit=True,
            bias_activation_fusion=True,
            norm_topk_prob=False,
        )

        overlap_loss, overlap_gpt_model = run_pp(
            self.seed,
            self.batch_size,
            self.seq_len,
            self.vocab_size,
            config,
            forward_backward_overlap_scheduler=False,
        )

        print("Overlap PP loss MD5:", overlap_loss._md5sum())
        rst = {}
        for name, param in overlap_gpt_model.named_parameters():
            if param.grad is not None:
                rst[name] = param.grad._md5sum()

        pp = pprint.PrettyPrinter(depth=None, width=200, compact=False)
        pp.pprint(rst)

        if judge_machine_type() == "H":
            actual_md5 = overlap_loss._md5sum()
            if judge_h_subtype() == "H800":
                expected_md5 = "ac1c324951d04405f159fe60a1b02f77"
            else:
                expected_md5 = "1ccdc1e3ec2f1b03a3634a468e9ef234"
            print(
                f"Overlap PP loss MD5 - Actual: {actual_md5}, Expected: {expected_md5}"
            )
            assert actual_md5 == expected_md5, (
                f"Overlap PP loss MD5 mismatch! Actual: {actual_md5}, Expected: {expected_md5}"
            )
            if paddle.distributed.get_rank() == 0:
                if judge_h_subtype() == "H800":
                    baseline = {
                        "_layers.9.0.input_layernorm.weight": "0aeebb3b5ac42c1faadb299fb398a396",
                        "_layers.9.0.mlp.experts.0.down_proj.weight": "b2d1236c286a3c0704224fe4105eca49",
                        "_layers.9.0.mlp.experts.0.up_gate_proj.weight": "b5cfa9d6c8febd618f91ac2843d50a1c",
                        "_layers.9.0.mlp.experts.1.down_proj.weight": "bb2d39b853c6cde7efc6affda92e6970",
                        "_layers.9.0.mlp.experts.1.up_gate_proj.weight": "6c54f1ed3be5b09ec747f30f1c291ec1",
                        "_layers.9.0.mlp.experts.2.down_proj.weight": "46b72fbb4e114b757fe20cd8aeeed345",
                        "_layers.9.0.mlp.experts.2.up_gate_proj.weight": "cdcc7000f0ec04f286387d3250ac7cbd",
                        "_layers.9.0.mlp.experts.3.down_proj.weight": "597ccdee5c1c51a9185c4532c151f36e",
                        "_layers.9.0.mlp.experts.3.up_gate_proj.weight": "9d7a7ca9763ca2b4e367b57f6430fd7c",
                        "_layers.9.0.mlp.gate.weight": "8186b2b41857c3eabb57900cf6a7ceb5",
                        "_layers.9.0.mlp.shared_experts.down_proj.weight": "6d218d68bcb6dde5cbb1b3da6e6341f4",
                        "_layers.9.0.mlp.shared_experts.up_gate_proj.weight": "506abb1bc4d2136cfe932c56df3838b3",
                        "_layers.9.0.post_attention_layernorm.weight": "56ae029b77d9d95b6d7632d1c9a94f6d",
                        "_layers.9.0.self_attn.k_norm.weight": "4cca721a4ad6e6d5055193924c0e238e",
                        "_layers.9.0.self_attn.o_proj.weight": "563d48636a28f11ddf0f9a8640eb8731",
                        "_layers.9.0.self_attn.q_norm.weight": "e42400164ec518f2e859474a5fa6918a",
                        "_layers.9.0.self_attn.qkv_proj.weight": "6a14886ed069063fee422d0d78c31ed7",
                        "_layers.9.1.input_layernorm.weight": "cf4b7d64e2bd7e5a446bdb066038a979",
                        "_layers.9.1.mlp.experts.0.down_proj.weight": "a3d6ca99029a542ba95b965a18f62e61",
                        "_layers.9.1.mlp.experts.0.up_gate_proj.weight": "50f974c4a34d885b6ca471a5ace6b72d",
                        "_layers.9.1.mlp.experts.1.down_proj.weight": "587dd9fb61c30338fa396d8b95492c75",
                        "_layers.9.1.mlp.experts.1.up_gate_proj.weight": "4328c6ab3666d39bb00f049491de420c",
                        "_layers.9.1.mlp.experts.2.down_proj.weight": "3e5835a883f4cc758b76df5586208cd0",
                        "_layers.9.1.mlp.experts.2.up_gate_proj.weight": "98f6954c2e8b52bbc1fddcca23252de9",
                        "_layers.9.1.mlp.experts.3.down_proj.weight": "b821d3e53f5af013480662fe797c798b",
                        "_layers.9.1.mlp.experts.3.up_gate_proj.weight": "1f25c9dc954557d42d0c53a0813134f3",
                        "_layers.9.1.mlp.gate.weight": "10c7b77099afa415214b95cbf175a529",
                        "_layers.9.1.mlp.shared_experts.down_proj.weight": "7806ce29cc56e200c5c12bcc193bf2ea",
                        "_layers.9.1.mlp.shared_experts.up_gate_proj.weight": "5193abcc40d069213862054e86109972",
                        "_layers.9.1.post_attention_layernorm.weight": "5d1bff4c423a8e164169cb8960e9e332",
                        "_layers.9.1.self_attn.k_norm.weight": "6c72294d51d9a9327a681b8b9762246e",
                        "_layers.9.1.self_attn.o_proj.weight": "4df0fe86b348813e18f5b4be879b1e5e",
                        "_layers.9.1.self_attn.q_norm.weight": "bcdff6ad6a46be0452ac20ccafa6b469",
                        "_layers.9.1.self_attn.qkv_proj.weight": "4950b65afdbd50308dc190c013635586",
                        "_layers.shared_layers.embed.embedding.embed_tokens.weight": "bcdb228f9924e079397e73a58a2ce638",
                    }
                else:
                    baseline = {
                        "_layers.9.0.input_layernorm.weight": "ec0237c803deb1d5c8e7383868cafed6",
                        "_layers.9.0.mlp.experts.0.down_proj.weight": "685fb4180087336f1c53fcd3dbabe6c9",
                        "_layers.9.0.mlp.experts.0.up_gate_proj.weight": "0dce4c8eee56416298e5a0736f1fbda2",
                        "_layers.9.0.mlp.experts.1.down_proj.weight": "d3ae1e44d829996273bb5a876ab61bc1",
                        "_layers.9.0.mlp.experts.1.up_gate_proj.weight": "11f1d19bb63e2a84b5b49880b8fd27b1",
                        "_layers.9.0.mlp.experts.2.down_proj.weight": "fbf89a0bcdc1a07e9db9859edd58982c",
                        "_layers.9.0.mlp.experts.2.up_gate_proj.weight": "c3f11f221118ac386865b59977666fd2",
                        "_layers.9.0.mlp.experts.3.down_proj.weight": "02f3544a655e065d356c118ff0030264",
                        "_layers.9.0.mlp.experts.3.up_gate_proj.weight": "702807491eb4fdc675199332a2865714",
                        "_layers.9.0.mlp.gate.weight": "c884c8a4023c28bdd991e6780af976cf",
                        "_layers.9.0.mlp.shared_experts.down_proj.weight": "557a9cf582b71c2314c624a58ba4f59e",
                        "_layers.9.0.mlp.shared_experts.up_gate_proj.weight": "1c79825703c71441ac4891140f12ddc6",
                        "_layers.9.0.post_attention_layernorm.weight": "bb24f2a9073af75bf216143dd0bea8d6",
                        "_layers.9.0.self_attn.k_norm.weight": "db1dd94571e2c28a5028bdc02a09ee9c",
                        "_layers.9.0.self_attn.o_proj.weight": "63f255e307c8995f29229347d93d48eb",
                        "_layers.9.0.self_attn.q_norm.weight": "cac43a18a4fea711fcf6be9a33157bd8",
                        "_layers.9.0.self_attn.qkv_proj.weight": "b982a039cef9d1cd3743b8480e4c9a07",
                        "_layers.9.1.input_layernorm.weight": "5f8e8972e45b94a7e575600d6c1c9162",
                        "_layers.9.1.mlp.experts.0.down_proj.weight": "bd03e97f15a8f8189d8ef32dea2a0c3d",
                        "_layers.9.1.mlp.experts.0.up_gate_proj.weight": "56573dd41653ddf0e6bdae3d9159bec4",
                        "_layers.9.1.mlp.experts.1.down_proj.weight": "1f194657715255dddcddecc61b785d30",
                        "_layers.9.1.mlp.experts.1.up_gate_proj.weight": "af9e70b171834f1de3ae59ba477894db",
                        "_layers.9.1.mlp.experts.2.down_proj.weight": "b2d1236c286a3c0704224fe4105eca49",
                        "_layers.9.1.mlp.experts.2.up_gate_proj.weight": "b5cfa9d6c8febd618f91ac2843d50a1c",
                        "_layers.9.1.mlp.experts.3.down_proj.weight": "1509a9d0495708a232db6c132de941da",
                        "_layers.9.1.mlp.experts.3.up_gate_proj.weight": "211c558638bc279cd97ab52a1a2aa1e9",
                        "_layers.9.1.mlp.gate.weight": "8fccefebc64a088e03aa5e06776c09d6",
                        "_layers.9.1.mlp.shared_experts.down_proj.weight": "381a680b9df4c6d299b0f8c87cbe6959",
                        "_layers.9.1.mlp.shared_experts.up_gate_proj.weight": "77aab284604803f18f7a40e8ee79c1a1",
                        "_layers.9.1.post_attention_layernorm.weight": "d869ac3d57ddb0e53a7bee5b7186e423",
                        "_layers.9.1.self_attn.k_norm.weight": "ee75ae274f5a0d60ce3a2f327f5da59c",
                        "_layers.9.1.self_attn.o_proj.weight": "89dca0341d0809626368dd887d908dd6",
                        "_layers.9.1.self_attn.q_norm.weight": "442e5b8eb0d81e47a17dfa771045c1b1",
                        "_layers.9.1.self_attn.qkv_proj.weight": "91ac2636d4968ff58a720b65622db864",
                        "_layers.shared_layers.embed.embedding.embed_tokens.weight": "40e721e098a0d7db467b3d699018afb1",
                    }
                for name, param in overlap_gpt_model.named_parameters():
                    assert param.grad._md5sum() == baseline[name], (
                        f"{name}'s grad has diff"
                    )
        elif judge_machine_type() == "B":
            assert overlap_loss._md5sum() == "85aa36e75cc0bee02411bf7d210e5cbe"
            if paddle.distributed.get_rank() == 0:
                baseline = {
                    "_layers.9.0.input_layernorm.weight": "f971381506276a757e7d5df17abdab6a",
                    "_layers.9.0.mlp.experts.0.down_proj.weight": "4356e79343d78774dbd2b8bf12827bc1",
                    "_layers.9.0.mlp.experts.0.up_gate_proj.weight": "313e7056304403e5f338b9f0d7457be3",
                    "_layers.9.0.mlp.experts.1.down_proj.weight": "b18edebcccbaa4014b63d9832ad52e3e",
                    "_layers.9.0.mlp.experts.1.up_gate_proj.weight": "22776461fe4175605574d25f53b469a9",
                    "_layers.9.0.mlp.experts.2.down_proj.weight": "78e8d02fd1ac168b08c6b834458d43f2",
                    "_layers.9.0.mlp.experts.2.up_gate_proj.weight": "4ddf51b904de48bb457da227055e4cb9",
                    "_layers.9.0.mlp.experts.3.down_proj.weight": "66e2528cf116a0b1d3a9e3b30376ec07",
                    "_layers.9.0.mlp.experts.3.up_gate_proj.weight": "f96f6bac687fcf8a0790553f74cbe392",
                    "_layers.9.0.mlp.gate.weight": "5b7b566fd7cb976253bd55c547d5d5db",
                    "_layers.9.0.mlp.shared_experts.down_proj.weight": "2b3f14a50fde3272ceaf4464115afe4f",
                    "_layers.9.0.mlp.shared_experts.up_gate_proj.weight": "f9f8be5c083d20c16375087c3609e6bf",
                    "_layers.9.0.post_attention_layernorm.weight": "023931c3bd2b9280839b4dbf5808f893",
                    "_layers.9.0.self_attn.k_norm.weight": "a8d5e0deb2b1f8704d4ba7a1e266d31c",
                    "_layers.9.0.self_attn.o_proj.weight": "00b442588685723b165d750453a0a7b2",
                    "_layers.9.0.self_attn.q_norm.weight": "a7f8ea46ae3fc6f3e1575edbf6722b04",
                    "_layers.9.0.self_attn.qkv_proj.weight": "ae58af4b0932a118eba51c14de8fb464",
                    "_layers.9.1.input_layernorm.weight": "8d6f8a1d6c1df6840e91c542e76923d2",
                    "_layers.9.1.mlp.experts.0.down_proj.weight": "5cf79d603c4afd562bede2a3ef1a8f40",
                    "_layers.9.1.mlp.experts.0.up_gate_proj.weight": "0014c42b06a2313940db98b361d38a25",
                    "_layers.9.1.mlp.experts.1.down_proj.weight": "646208faaea723db1929b84f0f276119",
                    "_layers.9.1.mlp.experts.1.up_gate_proj.weight": "d09d58788294d33577d03b62ebcc03c8",
                    "_layers.9.1.mlp.experts.2.down_proj.weight": "b2d1236c286a3c0704224fe4105eca49",
                    "_layers.9.1.mlp.experts.2.up_gate_proj.weight": "b5cfa9d6c8febd618f91ac2843d50a1c",
                    "_layers.9.1.mlp.experts.3.down_proj.weight": "4b67a6784880b90fec22142deff38e98",
                    "_layers.9.1.mlp.experts.3.up_gate_proj.weight": "35b4023dca25dae2f7c354068dbf7fc3",
                    "_layers.9.1.mlp.gate.weight": "476f933ef1cf1c84e7f6e869f278b4a2",
                    "_layers.9.1.mlp.shared_experts.down_proj.weight": "ddb798cce31fa60df67aa17233fd12ca",
                    "_layers.9.1.mlp.shared_experts.up_gate_proj.weight": "8921fad3165828bd443851e6c037d39f",
                    "_layers.9.1.post_attention_layernorm.weight": "0ce80ed39049cd9ec0ff94f2fe0923a1",
                    "_layers.9.1.self_attn.k_norm.weight": "523f6ae4609f89dc8ffaf62477a3bd65",
                    "_layers.9.1.self_attn.o_proj.weight": "f05cc44c22e8558f472df0ed8bc8434e",
                    "_layers.9.1.self_attn.q_norm.weight": "db8a54dd36edc25b83a7bb0345a4dd86",
                    "_layers.9.1.self_attn.qkv_proj.weight": "0e69f6456f90aef715731da7435f6b6e",
                    "_layers.shared_layers.embed.embedding.embed_tokens.weight": "af8e06a62800ed3126f2e945e71bfcf6",
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

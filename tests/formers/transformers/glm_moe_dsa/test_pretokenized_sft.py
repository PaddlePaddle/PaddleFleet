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

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from paddlefleet.cli.train.sft.workflow import (
    apply_glm_moe_dsa_training_contract,
    save_final_hf_model_if_requested,
    validate_pretokenized_offline_dataset,
)
from paddlefleet.datasets.collate import collate_fn
from paddlefleet.datasets.SFTDataset import TextSequence


def fixed_sequence():
    return TextSequence(
        token_ids=[154820, 42, 42, 17, 99, 42, 8],
        labels=[42, 42, 17, 99, 42, 8, 3],
        position_ids=[0, 1, 2, 3, 4, 5, 6],
        num_examples=1,
    )


def test_final_hf_export_always_writes_last_fc_to_hf():
    trainer = MagicMock()
    args = SimpleNamespace(save_to_hf=False, tensor_model_parallel_size=1)
    assert save_final_hf_model_if_requested(trainer, args) is True
    trainer.save_model.assert_called_once_with(
        merge_tensor_parallel=False,
        last_fc_to_hf=True,
    )

    trainer.reset_mock()
    args.save_to_hf = True
    args.tensor_model_parallel_size = 2
    assert save_final_hf_model_if_requested(trainer, args) is True
    trainer.save_model.assert_called_once_with(
        merge_tensor_parallel=True,
        last_fc_to_hf=True,
    )


def test_validate_pretokenized_offline_dataset_rejects_wrong_length():
    sequence = fixed_sequence()
    sequence.position_ids = sequence.position_ids[:-1]

    with pytest.raises(ValueError, match="position_ids length 6 != 7"):
        validate_pretokenized_offline_dataset([[sequence]], expected_length=7)


def test_glm_moe_dsa_training_contract_propagates_frozen_provider_fields():
    model_config = SimpleNamespace(model_type="glm_moe_dsa")
    training_args = SimpleNamespace(
        num_nextn_predict_layers=1,
        mtp_num_layers=1,
        fp32_residual_connection=False,
        moe_token_dispatcher_type="alltoall",
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=4,
        context_parallel_size=1,
        expert_model_parallel_size=2,
        expert_tensor_model_parallel_size=1,
        sequence_parallel=True,
    )
    model_args = SimpleNamespace(
        mtp_attention_flexible=True, persist_layer_norm=False
    )
    data_args = SimpleNamespace(pretokenized_dataset=True)

    apply_glm_moe_dsa_training_contract(
        model_config, training_args, model_args, data_args
    )

    assert not hasattr(model_config, "mtp_num_layers")
    assert training_args.mtp_num_layers == 0
    assert model_config.num_nextn_predict_layers == 1
    assert model_config.mtp_enabled is True
    assert model_config.fp32_residual_connection is False
    assert model_config.moe_token_dispatcher_type == "alltoall"
    assert model_config.tensor_model_parallel_size == 2
    assert model_config.pipeline_model_parallel_size == 4
    assert model_config.context_parallel_size == 1
    assert model_config.expert_model_parallel_size == 2
    assert model_config.expert_tensor_parallel_size == 1
    assert model_config.sequence_parallel is True
    assert model_config.persist_layer_norm is False


def test_glm_moe_dsa_training_contract_rejects_invalid_expert_tensor_parallel_size():
    model_config = SimpleNamespace(model_type="glm_moe_dsa")
    training_args = SimpleNamespace(
        num_nextn_predict_layers=1,
        mtp_num_layers=1,
        fp32_residual_connection=False,
        moe_token_dispatcher_type="alltoall",
        expert_tensor_model_parallel_size=0,
    )
    model_args = SimpleNamespace(
        mtp_attention_flexible=True, persist_layer_norm=False
    )
    data_args = SimpleNamespace(pretokenized_dataset=True)

    with pytest.raises(
        ValueError,
        match="expert_tensor_model_parallel_size must be -1 or at least 1",
    ):
        apply_glm_moe_dsa_training_contract(
            model_config, training_args, model_args, data_args
        )


def test_glm_moe_dsa_pretokenized_mtp_requires_flexible_mask():
    model_config = SimpleNamespace(model_type="glm_moe_dsa")
    training_args = SimpleNamespace(
        num_nextn_predict_layers=1,
        mtp_num_layers=1,
        fp32_residual_connection=False,
        moe_token_dispatcher_type="alltoall",
    )
    model_args = SimpleNamespace(
        mtp_attention_flexible=False, persist_layer_norm=False
    )
    data_args = SimpleNamespace(pretokenized_dataset=True)

    with pytest.raises(ValueError, match="mtp_attention_flexible=true"):
        apply_glm_moe_dsa_training_contract(
            model_config, training_args, model_args, data_args
        )


def test_pretokenized_mtp_padding_uses_explicit_zero_sentinel():
    tokenizer = SimpleNamespace(pad_token_id=154820)
    training_args = SimpleNamespace(
        num_nextn_predict_layers=1,
        context_parallel_size=1,
        tensor_model_parallel_size=1,
        sequence_parallel=False,
        fp8=False,
        max_seq_len=7,
    )
    model_args = SimpleNamespace(
        mtp_attention_flexible=False,
        use_attn_mask_startend_row_indices=False,
        use_global_causal_attn=False,
    )

    batch = collate_fn(
        [[fixed_sequence()]],
        tokenizer=tokenizer,
        training_args=training_args,
        model_args=model_args,
        max_seq_len=None,
        padding_free=False,
        input_pad_token_id=0,
    )

    assert batch["input_ids"].tolist() == [[154820, 42, 42, 17, 99, 42, 8, 0]]
    assert batch["labels"].tolist() == [[42, 42, 17, 99, 42, 8, 3, -100]]
    assert batch["position_ids"].tolist() == [[0, 1, 2, 3, 4, 5, 6, 0]]


def test_pretokenized_mtp_flexible_mask_matches_main_stream_length():
    tokenizer = SimpleNamespace(pad_token_id=154820)
    training_args = SimpleNamespace(
        num_nextn_predict_layers=1,
        context_parallel_size=1,
        tensor_model_parallel_size=1,
        sequence_parallel=False,
        fp8=False,
        max_seq_len=7,
    )
    model_args = SimpleNamespace(
        mtp_attention_flexible=True,
        use_attn_mask_startend_row_indices=False,
        use_global_causal_attn=False,
    )

    batch = collate_fn(
        [[fixed_sequence()]],
        tokenizer=tokenizer,
        training_args=training_args,
        model_args=model_args,
        max_seq_len=None,
        padding_free=False,
        input_pad_token_id=0,
    )

    assert list(batch["input_ids"].shape) == [1, 8]
    assert list(batch["attention_mask"].shape) == [1, 1, 7, 7]

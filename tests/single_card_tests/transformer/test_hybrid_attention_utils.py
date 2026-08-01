# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_MODULE_PATH = (
    Path(__file__).parents[3]
    / "src/paddlefleet/transformer/hybrid_attention_utils.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "hybrid_attention_utils", _MODULE_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

get_effective_mtp_layers = _MODULE.get_effective_mtp_layers
resolve_hybrid_attention_layer = _MODULE.resolve_hybrid_attention_layer
resolve_layer_attention_config = _MODULE.resolve_layer_attention_config


MLA_INDICES = {2, 10, 18, 26, 34, 42, 43}


def _config(head_offset=0, mtp_num_layers=0, nextn_layers=1):
    ratios = [128] * 44
    for index in MLA_INDICES:
        ratios[index] = -2
    ratios[1] = 0
    ratios[3] = 16
    return SimpleNamespace(
        num_hidden_layers=43,
        num_empty_layers_add_in_head=head_offset,
        mtp_num_layers=mtp_num_layers,
        num_nextn_predict_layers=nextn_layers,
        csa_compress_ratios=ratios,
        q_lora_rank=1024,
        v_head_dim=512,
        head_dim=128,
        qk_pos_emb_head_dim=64,
        num_attention_heads=64,
        num_key_value_heads=1,
        tensor_model_parallel_size=1,
        hybrid_mla_q_lora_rank=1536,
        hybrid_mla_kv_lora_rank=512,
        hybrid_mla_qk_nope_head_dim=192,
        hybrid_mla_qk_rope_head_dim=64,
        hybrid_mla_v_head_dim=256,
        hybrid_mla_num_attention_heads=64,
        hybrid_mla_num_key_value_heads=64,
    )


@pytest.mark.parametrize("head_offset", [0, 1])
def test_decoder_physical_numbers_map_to_canonical_logical_indices(head_offset):
    config = _config(head_offset=head_offset)

    for logical_index in range(43):
        info = resolve_hybrid_attention_layer(
            config, logical_index + head_offset, is_mtp_layer=False
        )
        assert info.logical_index == logical_index
        assert info.compress_ratio == config.csa_compress_ratios[logical_index]

    assert (
        resolve_hybrid_attention_layer(config, head_offset).layer_kind == "hca"
    )
    assert (
        resolve_hybrid_attention_layer(config, head_offset + 1).layer_kind
        == "window"
    )
    assert (
        resolve_hybrid_attention_layer(config, head_offset + 2).layer_kind
        == "mla"
    )
    assert (
        resolve_hybrid_attention_layer(config, head_offset + 3).layer_kind
        == "csa"
    )
    assert (
        resolve_hybrid_attention_layer(config, head_offset + 42).layer_kind
        == "mla"
    )


@pytest.mark.parametrize("head_offset", [0, 1])
def test_mtp_zero_maps_after_last_decoder_without_head_offset(head_offset):
    config = _config(head_offset=head_offset)

    info = resolve_hybrid_attention_layer(config, 0, is_mtp_layer=True)

    assert info == (43, "mla", -2)


@pytest.mark.parametrize(
    ("mtp_num_layers", "nextn_layers", "expected"),
    [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1)],
)
def test_effective_mtp_count_presence_combinations(
    mtp_num_layers, nextn_layers, expected
):
    config = _config(mtp_num_layers=mtp_num_layers, nextn_layers=nextn_layers)

    assert get_effective_mtp_layers(config) == expected


def test_mismatched_positive_mtp_counts_are_rejected():
    config = _config(mtp_num_layers=2, nextn_layers=1)

    with pytest.raises(ValueError, match="must be equal"):
        get_effective_mtp_layers(config)
    with pytest.raises(ValueError, match="must be equal"):
        resolve_hybrid_attention_layer(config, 0, is_mtp_layer=True)


@pytest.mark.parametrize(
    ("layer_number", "is_mtp_layer"),
    [(-1, False), (43, False), (-1, True), (1, True)],
)
def test_layer_boundaries_are_checked(layer_number, is_mtp_layer):
    with pytest.raises(IndexError):
        resolve_hybrid_attention_layer(
            _config(), layer_number, is_mtp_layer=is_mtp_layer
        )


@pytest.mark.parametrize("ratio", [1, -3, 129, 2.5, True, False])
def test_unknown_ratio_does_not_silently_select_an_attention_kind(ratio):
    config = _config()
    config.csa_compress_ratios[0] = ratio

    with pytest.raises(ValueError):
        resolve_hybrid_attention_layer(config, 0)


def test_mla_local_dimensions_are_explicit_and_immutable():
    local = resolve_layer_attention_config(_config(), 2)

    assert local.layer_kind == "mla"
    assert (local.q_lora_rank, local.kv_lora_rank) == (1536, 512)
    assert (local.qk_nope_head_dim, local.qk_rope_head_dim) == (192, 64)
    assert local.v_head_dim == 256
    assert (local.num_attention_heads, local.num_key_value_heads) == (64, 64)
    assert local.qk_pos_emb_head_dim is None
    with pytest.raises(AttributeError):
        local.v_head_dim = 512


def test_mla_local_dimensions_must_be_explicit_positive_integers():
    config = _config()
    config.hybrid_mla_v_head_dim = None
    with pytest.raises(ValueError, match="hybrid_mla_v_head_dim"):
        resolve_layer_attention_config(config, 2)

    config = _config()
    config.hybrid_mla_q_lora_rank = 0
    with pytest.raises(ValueError, match="hybrid_mla_q_lora_rank"):
        resolve_layer_attention_config(config, 2)


def test_hybrid_mla_rejects_tensor_parallelism():
    config = _config()
    config.tensor_model_parallel_size = 2
    with pytest.raises(
        ValueError, match="requires tensor_model_parallel_size=1"
    ):
        resolve_layer_attention_config(config, 2)


def test_mla_local_dimensions_accept_noncanonical_values():
    config = _config()
    config.hybrid_mla_q_lora_rank = 768
    config.hybrid_mla_kv_lora_rank = 384
    config.hybrid_mla_qk_nope_head_dim = 96
    config.hybrid_mla_qk_rope_head_dim = 32
    config.hybrid_mla_v_head_dim = 128
    config.hybrid_mla_num_attention_heads = 32
    config.hybrid_mla_num_key_value_heads = 8

    local = resolve_layer_attention_config(config, 2)

    assert local.q_lora_rank == 768
    assert local.kv_lora_rank == 384
    assert (local.qk_nope_head_dim, local.qk_rope_head_dim) == (96, 32)
    assert local.v_head_dim == 128
    assert (local.num_attention_heads, local.num_key_value_heads) == (32, 8)


@pytest.mark.parametrize("logical_index", [0, 1, 3])
def test_dsv4_local_dimensions_use_only_dsv4_namespace(logical_index):
    local = resolve_layer_attention_config(_config(), logical_index)

    assert local.layer_kind in {"hca", "csa", "window"}
    assert local.q_lora_rank == 1024
    assert local.v_head_dim == 512
    assert local.qk_pos_emb_head_dim == 64
    assert local.qk_nope_head_dim is None
    assert local.qk_rope_head_dim is None
    assert local.kv_lora_rank is None
    assert local.v_head_dim - local.qk_pos_emb_head_dim == 448

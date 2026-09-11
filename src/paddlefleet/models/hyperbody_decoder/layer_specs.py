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

"""HyperBody decoder 主干的 layer spec —— PaddleFleet 侧提供的**组件**。

## 分工

| 侧 | 职责 |
|---|---|
| **PaddleFleet（本文件）** | 提供 12 层 backbone 的 `LayerSpec` 列表与 block spec |
| PaddleFormers `transformers/hyperbody_decoder/` | HF-style config → ``GPTConfig`` 的转换 + 组网装配 |

这与参考实现 ``fleet_formers`` 的 hyperencoder 完全同形：那边 fleet 侧也只有
``layer_specs.py`` + ``modality_encoders.py``（组件），config 装配与顶层 Model
都在 formers 的 ``transformers/hyperencoder/``。

## 与 hyperencoder 共享的 backbone

decoder 与 encoder 是**同一族 DeepSeekV2-Lite MoE 主干**：12 层 / hidden 1280 /
MHA 10×128 / dense FFN 6848（只 layer 0）/ 64 experts top-6 / moe FFN 896 /
2 shared experts / RMSNorm 1e-6 / RoPE base 10000。所以本文件的骨架直接复用
``models/hyperencoder/layer_specs.py``：同样的 :func:`_is_moe_layer`（只认逐层
list）、同样的「逐层调 ``get_gpt_layer_local_spec``」循环。

**两处刻意不复用**，因为它们是 encoder 的双向 prefix-LM 专属机制，decoder 不占：

1. ``input_layernorm`` / ``post_attention_layernorm`` **不换成**
   ``HyperEncoderRMSNorm``。那是 encoder 为「恒定 fp32 norm」做的替换
   （源侧 ``native_hyperbody_modality_submodules.py:356-357`` 的
   ``_SequenceParallelRMSNorm``）；decoder 的源侧是未改动的 mcore ``GPTModel``，
   norm 就用 ``get_gpt_layer_local_spec`` 按 ``normalization="RMSNorm"`` 给的那个。
2. ``attn_mask_type`` 用 **``causal``** 而不是 encoder 的 ``no_mask``。decoder 是
   标准因果 LM；encoder 传 ``no_mask`` 是因为它自己喂完整的 dense prefix-LM 掩码，
   不能让框架再叠一层因果（见 hyperencoder/layer_specs.py 的模块 docstring）。
   同理 decoder **不需要** ``_attn_implementation="eager"`` 那道断言 ——
   ``DotProductAttention`` 的 SDPA 分支强制 ``is_causal=True`` 对我们正好是对的。

## 为什么 decoder 也要有这个文件

源侧 ``_build_model_specs``（``megatron_mimo_training_hyperbody.py:154-192``）只传
``ModuleSpec(module=GPTModel, params={config, vocab_size, max_sequence_length,
position_embedding_type})``，连 ``transformer_layer_spec`` 都没给 —— 所以
``paddlefleet.gpt_builders.gpt_builder`` 的通用路径（它内部会调
``get_gpt_decoder_layers_spec``）在**行为上**本来就够用。

但把 spec 的构造显式收在这里有两个实际好处：

* ``moe_layer_freq`` 传 int 时 ``get_gpt_decoder_layers_spec``
  （``gpt_layer_specs.py:803-807``）会按 ``i % N`` 生成 pattern，而 Megatron 侧是
  另一套取模语义 —— 静默错位。:func:`_is_moe_layer` 直接**拒绝** int，让它变成
  显式报错（这条与 hyperencoder 那份逐字一致）。
* backbone 的构造点是 encoder / decoder 的公共面，放在一处才能保证两边同步演进。
"""

from __future__ import annotations

from paddle.distributed.fleet.meta_parallel import LayerSpec

from paddlefleet.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from paddlefleet.transformer.enums import AttnMaskType
from paddlefleet.transformer.transformer_block import TransformerBlockSublayersSpec
from paddlefleet.transformer.transformer_config import TransformerConfig

__all__ = [
    "get_hyperbody_decoder_layer_specs",
    "get_hyperbody_decoder_block_spec",
]


def _is_moe_layer(config: TransformerConfig, layer_idx: int) -> bool:
    """第 i 层是不是 MoE 层。

    源侧 ``moe_layer_freq = [0] + [1]*11``（``megatron_mimo_training_hyperbody.py:100``）
    是**逐层的 0/1 列表**，判据就是取列表第 i 项。

    ⚠️ 与 hyperencoder 那份同样**拒绝 int**：传 int 时 Paddle 走
    ``i % N``（``gpt_layer_specs.py:803-807``），Megatron 走另一套语义，必然错位。
    """
    freq = config.moe_layer_freq
    if isinstance(freq, (list, tuple)):
        if len(freq) != config.num_hidden_layers:
            raise ValueError(
                f"moe_layer_freq 长度 {len(freq)} != num_hidden_layers "
                f"{config.num_hidden_layers}"
            )
        return bool(freq[layer_idx])
    raise TypeError(
        f"HyperBody decoder 要求 moe_layer_freq 是逐层的 list，得到 {type(freq).__name__}。"
        "传 int 会走 `i % N` 的语义（与 Megatron 侧不同），必然错位。"
    )


def get_hyperbody_decoder_layer_specs(config: TransformerConfig) -> list[LayerSpec]:
    """产出 ``num_hidden_layers`` 层的 :class:`LayerSpec` 列表。

    layer 0 是 dense FFN（``intermediate_size``），其余是 MoE
    （``n_routed_experts`` × ``moe_intermediate_size`` + shared experts）。

    ``moe_expert_fusion`` 只对 MoE 层生效 —— dense 层传 ``False``，否则
    ``get_mlp_layer_spec_for_backend`` 会去建 grouped GEMM 的 3-D 权重
    （对齐 ``get_gpt_decoder_layers_spec`` 的做法）。
    """
    if config.multi_latent_attention:
        raise ValueError(
            "HyperBody decoder 是纯 MHA（源侧 num_query_groups == num_attention_heads == 10，"
            "且 _make_language_config 没设任何 MLA 字段）。multi_latent_attention 必须为 False。"
        )

    specs: list[LayerSpec] = []
    for i in range(config.num_hidden_layers):
        is_moe = _is_moe_layer(config, i)
        specs.append(
            get_gpt_layer_local_spec(
                config=config,
                layer_number=i + config.num_empty_layers_add_in_head,
                num_experts=config.n_routed_experts if is_moe else None,
                moe_expert_fusion=config.moe_expert_fusion if is_moe else False,
                use_qk_norm=config.use_qk_norm,
                normalization=config.normalization,
                # 标准因果 LM。与 encoder 的 no_mask 相反，见模块 docstring。
                attn_mask_type=AttnMaskType.causal,
            )
        )
    return specs


def get_hyperbody_decoder_block_spec(
    config: TransformerConfig,
) -> TransformerBlockSublayersSpec:
    """产出整个 block 的 spec（N 层 + 最终 norm）。

    ``layer_norm=None`` 表示沿用 ``TransformerBlock`` 按 ``config.normalization``
    自己建的那一个 —— decoder 不像 encoder 需要换成自写的 fp32 norm。

    走 ``paddleformers-cli`` 的 pipeline 路径时用不到这个函数（那条路要的是裸的
    ``list[LayerSpec]``，见 ``get_hyperbody_decoder_layer_specs``）；它是给不走
    ``PipelineLayer`` 的单卡脚本/单测直接建 ``TransformerBlock`` 用的。
    """
    return TransformerBlockSublayersSpec(
        layer_specs=get_hyperbody_decoder_layer_specs(config),
        layer_norm=None,
    )

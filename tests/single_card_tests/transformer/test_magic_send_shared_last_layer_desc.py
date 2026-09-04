# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""``mtp_shared_last_layer`` must survive ``enable_mtp_magic_send``.

``GPTModel.get_layer_desc_list`` emits two descs for the last-layer tie:

  * the *pivot* -- the last backbone TransformerLayer, keyed
    ``mtp_reuse_transformer``;
  * the *MTP layer* -- keyed the same way, with
    ``shared_submodule_weight_only=True`` so paddle's ``_alias_shared_layer``
    aliases only the parameters under ``transformer_layer``.

Both are required: paddle only aliases when a shared key has more than one
member. The MTP-side branch used to test ``enable_mtp_magic_send`` first and
short-circuit to a plain ``LayerDesc``, so with both flags on the tie silently
did nothing -- the model trained an extra full MTP attention block while the
config claimed it was shared. That is a wrong-parameter-count bug that no assert
caught, which is why it is pinned here.

The two mechanisms are orthogonal: magic send owns ``mtp_embed`` (a separate
``SharedLayerDesc`` keyed ``mtp_embed``, synced through gpt_model's dedicated
``_mtp_embed_global_group``), and the tie touches only ``transformer_layer``
params. Their parameter sets do not overlap.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from paddle import nn
from paddle.distributed.fleet.meta_parallel import LayerDesc, SharedLayerDesc

from paddlefleet.models.gpt.gpt_model import GPTModel


class _Stub(nn.Layer):
    """Placeholder layer class; only its identity matters to LayerDesc."""


def _make_spec(num_transformer_layers=2, num_mtp=1):
    return SimpleNamespace(
        embedding=_Stub,
        head_empty_layers=[],
        tail_empty_layers=[],
        mhc_expand=None,
        mhc_contract=None,
        transformer_layers=[_Stub] * num_transformer_layers,
        output_block_attn_res=None,
        layer_norm=_Stub,
        mtp=[_Stub] * num_mtp,
        mtp_lm_head=None,
        mtp_loss=None,
        lm_head=_Stub,
    )


def _make_config(**overrides):
    cfg = dict(
        model_type="gpt",
        enable_mtp_magic_send=False,
        mtp_shared_last_layer=False,
        gpt_model_use_experimental_version=False,
        num_nextn_predict_layers=1,
        multimax_modules=None,
        separate_mtp_headloss=False,
    )
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


def _layer_descs(**config_overrides):
    """Call get_layer_desc_list without building a real GPTModel.

    The method only touches ``self.config`` and ``self.add_sequential_layer``,
    so a stub carrying those two is enough and keeps this a single-card test.
    """
    fake_self = SimpleNamespace(
        config=_make_config(**config_overrides),
        add_sequential_layer=lambda layers, desc, name_prefix="": layers.append(
            {"layer": desc, "name_prefix": name_prefix}
        ),
    )
    return GPTModel.get_layer_desc_list(
        fake_self, _make_spec(), tie_word_embeddings=False
    )


def _shared_descs(layers, key):
    return [
        entry["layer"]
        for entry in layers
        if isinstance(entry["layer"], SharedLayerDesc)
        and entry["layer"].layer_name == key
    ]


class TestMagicSendSharedLastLayerDesc(unittest.TestCase):
    def test_tie_emits_two_members_under_magic_send(self) -> None:
        layers = _layer_descs(
            enable_mtp_magic_send=True, mtp_shared_last_layer=True
        )
        tied = _shared_descs(layers, "mtp_reuse_transformer")
        self.assertEqual(
            len(tied),
            2,
            "the tie needs both the pivot and the MTP desc; with one member "
            "paddle's _alias_shared_layer never runs and the tie is a no-op",
        )
        # Exactly one of them aliases a submodule only: the MTP side. The pivot
        # is a whole TransformerLayer and shares itself.
        submodule_only = [
            d for d in tied if getattr(d, "shared_submodule_weight_only", False)
        ]
        self.assertEqual(len(submodule_only), 1)

    def test_magic_send_still_emits_its_own_mtp_embed(self) -> None:
        # The tie must not displace magic send's embedding desc -- that table is
        # the only reason magic send works at all on the last PP stage.
        layers = _layer_descs(
            enable_mtp_magic_send=True, mtp_shared_last_layer=True
        )
        self.assertEqual(len(_shared_descs(layers, "mtp_embed")), 1)

    def test_no_tie_keeps_plain_layer_desc(self) -> None:
        layers = _layer_descs(
            enable_mtp_magic_send=True, mtp_shared_last_layer=False
        )
        self.assertEqual(_shared_descs(layers, "mtp_reuse_transformer"), [])
        # ... and the MTP layer is a plain desc, not shared with anything.
        self.assertTrue(
            any(
                isinstance(entry["layer"], LayerDesc)
                and not isinstance(entry["layer"], SharedLayerDesc)
                for entry in layers
            )
        )

    def test_tie_without_magic_send_is_unchanged(self) -> None:
        # Regression guard for the branch reorder: the non-magic-send behaviour
        # must be byte-for-byte the same shape it always was.
        layers = _layer_descs(
            enable_mtp_magic_send=False, mtp_shared_last_layer=True
        )
        self.assertEqual(len(_shared_descs(layers, "mtp_reuse_transformer")), 2)
        self.assertEqual(_shared_descs(layers, "mtp_embed"), [])


if __name__ == "__main__":
    unittest.main()

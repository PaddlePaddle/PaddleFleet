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
"""Empty-layer naming decoupling: empty (placeholder) layers live in their own
``empty_layers.<n>`` namespace while real transformer/MTP layers are numbered
from ``layers.0`` with no ``num_empty_layers_add_in_head`` offset.

We drive ``get_layer_desc_list`` / ``get_encoder_layer_desc_list`` with a stub
self so we observe only the ``name_prefix`` each slot is assigned, independent
of any distributed model build.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

from paddlefleet.models.gpt import gpt_model as gm
from paddlefleet.transformer import transformer_encoder as te


def _spec(num_hidden, n_head_empty, n_tail_empty, n_mtp):
    return SimpleNamespace(
        embedding=object(),
        head_empty_layers=[object()] * n_head_empty,
        mhc_expand=None,
        transformer_layers=[object()] * num_hidden,
        mhc_contract=None,
        output_block_attn_res=None,
        layer_norm=object(),
        mtp=([object()] * n_mtp) if n_mtp else None,
        mtp_lm_head=None,
        mtp_loss=None,
        tail_empty_layers=[object()] * n_tail_empty,
        lm_head=object(),
    )


def _config(num_hidden, n_mtp):
    return SimpleNamespace(
        enable_mtp_magic_send=False,
        mtp_shared_last_layer=False,
        gpt_model_use_experimental_version=False,
        num_nextn_predict_layers=n_mtp,
        num_hidden_layers=num_hidden,
        multimax_modules=[],
        separate_mtp_headloss=False,
    )


class _StubGPT:
    """Records the name_prefix passed to each add_sequential_layer call."""

    def __init__(self, config):
        self.config = config

    def _model_name_prefix(self):
        return "model"

    def add_sequential_layer(self, layers, desc, name_prefix):
        layers.append(name_prefix)


class _StubEncoder:
    modal = None

    def add_sequential_layer(self, layers, desc, name_prefix):
        layers.append(name_prefix)


class TestEmptyLayerNamingDecoupled(unittest.TestCase):
    def test_gpt_model_empty_and_real_namespaces(self):
        num_hidden, head_empty, tail_empty, mtp = 3, 2, 2, 1
        with (
            mock.patch.object(gm, "LayerDesc", lambda *a, **k: None),
            mock.patch.object(
                gm,
                "SharedLayerDesc",
                lambda *a, **k: SimpleNamespace(layer_name=a[0] if a else None),
            ),
        ):
            names = gm.GPTModel.get_layer_desc_list(
                _StubGPT(_config(num_hidden, mtp)),
                _spec(num_hidden, head_empty, tail_empty, mtp),
                tie_word_embeddings=False,
            )

        empties = [n for n in names if ".empty_layers." in n]
        reals = [n for n in names if n.rsplit(".", 2)[-2:-1] == ["layers"]]
        # head(2) + tail(2) empties share one 0-based empty_layers counter
        self.assertEqual(
            empties,
            [
                "model.empty_layers.0",
                "model.empty_layers.1",
                "model.empty_layers.2",
                "model.empty_layers.3",
            ],
        )
        # transformer 0..num_hidden-1 then MTP at num_hidden.. (no +H offset)
        self.assertEqual(
            reals,
            [
                "model.layers.0",
                "model.layers.1",
                "model.layers.2",
                "model.layers.3",
            ],
        )

    def test_encoder_tower_empty_and_real_namespaces(self):
        num_hidden, head_empty, tail_empty = 3, 2, 2
        enc_layers = []
        with mock.patch.object(te, "LayerDesc", lambda *a, **k: None):
            te.TransformerEncoder.get_encoder_layer_desc_list(
                _StubEncoder(),
                enc_layers,
                _spec(num_hidden, head_empty, tail_empty, 0),
                "model.visual",
            )
        self.assertEqual(
            enc_layers,
            [
                "model.visual.empty_layers.0",
                "model.visual.empty_layers.1",
                "model.visual.layers.0",
                "model.visual.layers.1",
                "model.visual.layers.2",
                "model.visual.empty_layers.2",
                "model.visual.empty_layers.3",
            ],
        )


if __name__ == "__main__":
    unittest.main()

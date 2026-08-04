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

"""Regression tests for the MTP layer (logical layer 43) under non-absorbed
MQA + DSA. Owned by audit workstream W6.

Production shape: ``num_hidden_layers=43``, ``num_nextn_predict_layers=1`` and
``csa_compress_ratios`` with ``-2`` at logical indices [8,17,26,34,42,43]. The
single MTP layer is built with ``layer_number=0`` / ``is_mtp_layer=True`` and
its ``MQALatentAttention`` must (a) dispatch to ``multi_latent_attention`` with
a live DSA indexer, not the dense HCA path; (b) build its indexer-loss row mask
purely from ``input_ids != pad`` (so the depth+1 MTP token shift cannot change
the row count for a fixed pad count); and (c) keep the indexer aux gradient
confined to ``indexer.*`` even when the downstream output loss is zero.

These guard the one layer where the MTP token shift is the only thing that
changes what ``x`` / ``input_ids`` the indexer sees.
"""

import types
import unittest

import numpy as np
import paddle

from paddlefleet.models.gpt.gpt_layer_specs import (
    _get_dsv4_hybrid_attention_layer_type,
)

from . import hybrid_mla_utils as U

SEQ = 512


def _prod_csa_ratios():
    """44-slot production ratios: -2 at [8,17,26,34,42,43], 128 elsewhere."""
    r = [128] * 8 + [-2] + [128] * 8 + [-2] + [128] * 8 + [-2]
    r += [128] * 7 + [-2] + [128] * 7 + [-2, -2]
    assert len(r) == 44
    assert [i for i, v in enumerate(r) if v == -2] == [8, 17, 26, 34, 42, 43]
    return r


def _build_mtp_module():
    """The single MTP layer: ``layer_number=0`` + ``is_mtp_layer=True``."""
    cfg = U._create_mqa_config("mqa_dsa", loss_coeff=0.01, num_hidden_layers=43)
    cfg.num_nextn_predict_layers = 1
    cfg.pad_token_id = 0
    return U._build_module(cfg, layer_number=0, bf16=True, is_mtp=True)


class TestMTPLayer43Dispatch(unittest.TestCase):
    def test_mtp_depth0_routes_to_mla_ratio_minus2(self):
        cfg = types.SimpleNamespace(
            num_hidden_layers=43,
            csa_compress_ratios=_prod_csa_ratios(),
            num_nextn_predict_layers=1,
            mtp_num_layers=0,
            num_empty_layers_add_in_head=0,
        )
        li, atype, ratio = _get_dsv4_hybrid_attention_layer_type(
            cfg, 0, is_mtp_layer=True
        )
        self.assertEqual((li, atype, ratio), (43, "multi_latent_attention", -2))


class TestMTPLayer43IndexerRowMask(unittest.TestCase):
    """Row mask is driven by ``input_ids != pad`` only (mqa_latent_attention.py
    :560-576), so the depth+1 MTP shift cannot move the row count.
    """

    def test_row_count_is_nonpad_count(self):
        m = _build_mtp_module()
        # 512 == no padding at all, 1 == a single real token: the two boundaries
        # where an off-by-one in the mask would show up.
        for real in (400, 512, 1):
            with self.subTest(real=real):
                ids = np.zeros([1, SEQ], dtype="int64")
                ids[0, :real] = np.arange(1, real + 1)
                _, vr = m._indexer_loss_mask(paddle.to_tensor(ids), 1, SEQ)
                self.assertEqual(int(vr), real)


@U._GPU
class TestMTPLayer43AuxGradConfinement(unittest.TestCase):
    """On layer 43, the indexer aux loss is injected via the
    ``TileLangCSAIndexerLossAutoScaler`` PyLayer and must (a) fire even when the
    downstream output loss is scaled to 0, and (b) stay confined to
    ``indexer.*`` because of the ``x.detach()`` at mqa_latent_attention.py:406.
    """

    def _run(self, upstream_scale):
        paddle.set_device("gpu:0")
        m = _build_mtp_module()
        m.train()
        q, k, wv, x, qr = U._make_inputs(SEQ, seed=1, with_hidden=True)

        def leaf(t):
            t = t.clone().detach()
            t.stop_gradient = False
            return t

        q, k, wv, x, qr = map(leaf, (q, k, wv, x, qr))
        ids = np.zeros([1, SEQ], dtype="int64")
        ids[0, :400] = np.arange(1, 401)
        row_end = U._row_end([SEQ], SEQ)
        m.clear_gradients()
        out = m(
            q,
            k,
            None,
            None,
            row_end,
            x=x,
            qr=qr,
            v_b_proj_weight=wv,
            input_ids=paddle.to_tensor(ids),
        )
        (out.cast("float32") * upstream_scale).sum().backward()
        idx_ids = {id(p) for _, p in m.indexer.named_parameters()}
        idx_nz = other_nz = 0
        for _, p in m.named_parameters():
            g = 0.0 if p.grad is None else float(p.grad.cast("float32").norm())
            if id(p) in idx_ids:
                idx_nz += g > 0
            else:
                other_nz += g > 0
        xg = 0.0 if x.grad is None else float(x.grad.cast("float32").norm())
        return idx_nz, other_nz, xg

    def test_aux_grad_survives_zero_output_loss(self):
        idx_nz, other_nz, xg = self._run(upstream_scale=0.0)
        self.assertGreater(
            idx_nz, 0, "indexer aux grad must fire at 0 output loss"
        )
        self.assertEqual(other_nz, 0, "aux grad leaked into non-indexer params")
        self.assertEqual(
            xg, 0.0, "aux grad leaked into backbone x (detach broke)"
        )


if __name__ == "__main__":
    unittest.main()

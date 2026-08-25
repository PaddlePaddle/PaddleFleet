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

"""Cache-on/off equivalence for ``csa_share_docmask_meta`` /
``mqa_share_docmask_meta``.

The switch never changes any arithmetic: it only decides *who* builds the
document-mask metadata (``CSADocMaskMetadata`` / ``MQADocMeta`` is a pure
function of the mask, ratio and sequence length). With the switch ON and the
metadata preloaded, the layer reads the shared slot; with it OFF (or on a mask
group the trainer never prebuilds -- the MTP depths), the layer builds the same
pure function itself. Either way the tables are bit-identical, so forward AND
backward must be bit-identical too.

Which phases actually consume the metadata, on the current architecture:

* ``mqa_full_causal`` (no indexer) runs as dense FA4 flashmask off the row-end
  mask alone -- no index table at all, so the cache only proves the preload ->
  lookup plumbing disturbs nothing (``test_full_causal_plumbing``).
* the DSA ``warmup`` phase (indexer present, ``indexer_use_sparse_loss`` off)
  reads ``meta.indexer_valid_range`` for the full-causal indexer KL.
* the DSA ``sparse`` phase reads ``meta.window_topk_idxs`` +
  ``meta.indexer_valid_range`` + ``meta.cu_seqlens_arg``.

Each test therefore:

1. runs one step (``accumulate_steps`` micro-batches) with the switch OFF --
   the reference;
2. runs the same step with the switch ON, driving the producer side exactly
   like the trainer does (``begin_step`` -> ``preload``/``preload_mqa`` per
   micro-batch -> ``check`` after the step);
3. asserts the results match: the forward output **bit-equal** (the metadata
   only selects who builds the same pure function, so it must be), and the
   gradients within a small relative tolerance. The tolerance is not cache
   related: the sparse backward kernel is nondeterministic in the atomic
   accumulation order -- repeating the *same* switch-off run twice already
   perturbs ``dkey`` by ~1.5e-2 (measured on B300), so gradients cannot be
   bit-compared.

Three production paths are pinned this way:

* MQA main  group -- slot hit (``get_mqa``), in all three phases.
* MQA MTP  group -- slot absent by design -> falls back to building its own.
* CSA  main group -- slot hit (``get``); CSA MTP group -> same fallback. CSA
  layers go through the HCA auto-advance (no explicit ``docmask_mb_idx``).

GPU-gated: the MQA/DSv4 forwards run the real kernels (sparse or FA4).
"""

import unittest

import numpy as np
import paddle
from paddle.distributed.fleet.meta_parallel import build_spec_layer

from paddlefleet.models.gpt.gpt_layer_specs import get_attention_spec
from paddlefleet.tensor_parallel.random import model_parallel_cuda_manual_seed
from paddlefleet.transformer.doc_mask_meta_registry import (
    doc_mask_meta_registry,
)
from paddlefleet.transformer.enums import AttnMaskType
from paddlefleet.transformer.transformer_config import TransformerConfig

from .hybrid_mla_utils import (
    _GPU,
    WINDOW,
    _build_module,
    _create_mqa_config,
    _fa4_module_hooks,
    _make_inputs,
    _row_end,
)

_ACC = 2  # gradient accumulation steps, matching the registry's step model

# The MQA full-causal / warmup phases run dense FA4 flashmask, which the flag
# pin reproduces (``_fa4_pin`` stands down where FA4 cannot serve).
setUpModule, tearDownModule = _fa4_module_hooks()


def _leaf(t):
    x = t.clone().detach()
    x.stop_gradient = False
    return x


def _assert_bitequal(actual, expected, what):
    a = np.asarray(actual, dtype="float32")
    e = np.asarray(expected, dtype="float32")
    assert a.shape == e.shape, f"{what} shape: {a.shape} != {e.shape}"
    assert np.array_equal(a, e), (
        f"{what} not bit-equal: max_abs_diff="
        f"{(np.abs(a - e)).max() if a.size else 0}"
    )


def _assert_relclose(actual, expected, what, tol=1e-2):
    """Gradient: relative, under ``tol``.

    The sparse bwd kernel is nondeterministic in its atomic accumulation order
    (the same switch-off run reproduced twice perturbs ``dkey`` by ~1.5e-2 on
    B300), so gradients cannot be bit-compared independent of the cache.
    """
    a = np.asarray(actual, dtype="float32")
    e = np.asarray(expected, dtype="float32")
    assert a.shape == e.shape, f"{what} shape: {a.shape} != {e.shape}"
    denom = float(np.linalg.norm(e.reshape(-1)))
    rel = float(np.linalg.norm((a - e).reshape(-1)) / max(denom, 1e-12))
    assert rel < tol, f"{what} rel err {rel:.3e} >= {tol}"


@_GPU
class TestMQADocCacheEquivalence(unittest.TestCase):
    """``mqa_share_docmask_meta`` on == off, bit-equal on out and grads."""

    def setUp(self):
        # One module, reused for both runs: parameter tensors stay identical,
        # only the cache path changes between the two passes.
        self.module = self._mqa_module()
        self.seqlen, self.doc_lens = 128, [10, 118]
        self.row_end = _row_end(self.doc_lens, self.seqlen)

    def _mqa_module(self, mode="mqa", bf16=False, is_mtp=False):
        # ``hybrid_mla_utils``'s factory already sizes k/v for the FA4 whitelist
        # (kv_lora 512 + rope 64 -> (576, 512)), so no dim overrides here.
        cfg = _create_mqa_config(
            mode, loss_coeff=0.05 if mode == "mqa_dsa" else 0.0
        )
        return _build_module(cfg, bf16=bf16, is_mtp=is_mtp)

    def _skip_unless_fa4_576_512(self):
        # The dense full-causal / warmup paths run FA4 with the (576, 512)
        # pair. Wheels that only carry the older (192, 128) cute kernel answer
        # FA2 for it -- skip there, run on the SM100+ boxes with the new
        # kernels.
        from paddlefleet_ops.flash_mask_facade import get_fa_version

        if get_fa_version(576, 512) != 4:
            self.skipTest(
                "FA4 (576, 512) is not served by this paddlefleet_ops build; "
                "the dense full-causal / warmup phases cannot run here."
            )

    def _preload(self):
        doc_mask_meta_registry.begin_step(_ACC)
        for mb in range(_ACC):
            doc_mask_meta_registry.preload_mqa(
                mb,
                1,
                self.seqlen,
                self.row_end,
                ("main",),
                WINDOW,
            )

    def _run(self, use_cache):
        outs, grads = [], []
        # The switch lives on the config: with it off the layer always builds
        # its metadata privately; with it on (plus the preload below) the
        # main-group slot is really read from the registry. The reference run
        # keeps the switch off.
        self.module.config.mqa_share_docmask_meta = use_cache
        if use_cache:
            self._preload()
        for mb in range(_ACC):
            use_hidden = bool(
                getattr(self.module.config, "_build_dsa_indexer", False)
            )
            if use_hidden:
                query, key, w_v, x, qr = _make_inputs(
                    self.seqlen, seed=mb + 1, with_hidden=True
                )
            else:
                query, key, w_v = _make_inputs(self.seqlen, seed=mb + 1)
            q, k, wv = _leaf(query), _leaf(key), _leaf(w_v)
            self.module.clear_gradients()
            if use_hidden:
                out = self.module(
                    q,
                    k,
                    None,
                    None,
                    self.row_end,
                    v_b_proj_weight=wv,
                    x=x,
                    qr=qr,
                    docmask_mb_idx=mb if use_cache else -1,
                )
            else:
                out = self.module(
                    q,
                    k,
                    None,
                    None,
                    self.row_end,
                    v_b_proj_weight=wv,
                    docmask_mb_idx=mb if use_cache else -1,
                )
            out.sum().backward()
            outs.append(out.cast("float32").numpy())
            grads.append(
                [
                    q.grad.cast("float32").numpy(),
                    k.grad.cast("float32").numpy(),
                    wv.grad.cast("float32").numpy(),
                ]
            )
        if use_cache:
            doc_mask_meta_registry.check()
        return outs, grads

    def _assert_step_equal(
        self, ref_out, ref_grads, cache_out, cache_grads, what
    ):
        for mb in range(_ACC):
            _assert_bitequal(cache_out[mb], ref_out[mb], f"{what} out mb{mb}")
            for name, g, rg in zip(
                ("query", "key", "w_v"), cache_grads[mb], ref_grads[mb]
            ):
                _assert_relclose(g, rg, f"{what} d{name} mb{mb}")

    def test_full_causal_plumbing_matches(self):
        self._skip_unless_fa4_576_512()
        # full_causal phase: dense FA4 off the row-end mask, no meta table
        # consumed. The preloaded slot is still looked up (warm build, slot
        # keying) -- the test pins that this plumbing never perturbs the
        # output bit-pattern.
        ref_out, ref_grads = self._run(use_cache=False)
        cache_out, cache_grads = self._run(use_cache=True)
        self._assert_step_equal(
            ref_out, ref_grads, cache_out, cache_grads, "plumbing"
        )

    def test_warmup_main_group_matches(self):
        self._skip_unless_fa4_576_512()
        # DSA warmup phase (indexer present, sparse loss off): attention runs
        # the dense FA4 full-causal path while the indexer is supervised over
        # the whole causal span -- both halves read the same shared slot.
        self.module = self._mqa_module("mqa_dsa", bf16=True)
        self.module.config.dsa_indexer_use_sparse_loss = False
        self.module.indexer_use_sparse_loss = False
        ref_out, ref_grads = self._run(use_cache=False)
        cache_out, cache_grads = self._run(use_cache=True)
        self._assert_step_equal(
            ref_out, ref_grads, cache_out, cache_grads, "warmup"
        )

    def test_sparse_main_group_matches(self):
        # sparse phase (``dsa_indexer_use_sparse_loss=True``, the default of
        # the fixtures): window + top-k attention and the restricted KL -- the
        # consumer of window_topk / valid_range / cu_seqlens.
        self.module = self._mqa_module("mqa_dsa", bf16=True)
        self.module.config.dsa_indexer_use_sparse_loss = True
        self.module.indexer_use_sparse_loss = True
        ref_out, ref_grads = self._run(use_cache=False)
        cache_out, cache_grads = self._run(use_cache=True)
        self._assert_step_equal(
            ref_out, ref_grads, cache_out, cache_grads, "sparse"
        )

    def test_mp_group_fallback_matches(self):
        # MTP consumer: its ("mtp", ...) group is never preloaded; the lookup
        # misses and the layer builds privately -- the result must equal the
        # switch-off run on the same weights.
        self.module = self._mqa_module("mqa_dsa", bf16=True, is_mtp=True)
        self.module.config.dsa_indexer_use_sparse_loss = True
        self.module.indexer_use_sparse_loss = True
        ref_out, ref_grads = self._run(use_cache=False)
        cache_out, cache_grads = self._run(use_cache=True)
        self._assert_step_equal(
            ref_out, ref_grads, cache_out, cache_grads, "mtp"
        )

    def test_forward_without_mask_builds_privately(self):
        self._skip_unless_fa4_576_512()
        # attn_mask_startend_row_indices=None means one document covering the
        # whole sequence: the layer falls back to a full row_end. Keep the
        # slot off so the fallback build actually runs.
        self.module.config.mqa_share_docmask_meta = True
        query, key, w_v = _make_inputs(self.seqlen, seed=7)
        q, k, wv = _leaf(query), _leaf(key), _leaf(w_v)
        out = self.module(
            q,
            k,
            None,
            None,
            None,  # no mask -> whole sequence is one document
            v_b_proj_weight=wv,
            docmask_mb_idx=-1,
        )
        out.sum().backward()
        self.assertEqual(next(iter(out.shape)), 1)


def _make_csa_config():
    """Minimal DSv4-hybrid config: one compressed-ratio-4 layer, unfused
    kernels so the test depends on no extra fused-cuda backend."""
    config = TransformerConfig(
        num_hidden_layers=1,
        num_nextn_predict_layers=1,
        mtp_num_layers=1,
        hidden_size=128,
        num_attention_heads=4,
        params_dtype=paddle.bfloat16,
        bf16=True,
        use_bias=False,
        multi_latent_attention=True,
        experimental_attention_variant="dsv4_hybrid",
        q_lora_rank=32,
        kv_lora_rank=16,
        qk_nope_head_dim=16,
        qk_rope_head_dim=8,
        qk_pos_emb_head_dim=8,
        v_head_dim=16,
        o_groups=2,
        o_lora_rank=16,
        normalization="RMSNorm",
        use_qk_norm=True,
        csa_compress_ratios=[
            4,
            4,
        ],  # [main, MTP]: MTP reads ratios[num_hidden_layers + depth]
        csa_window_size=8,
        dsa_index_n_heads=8,
        dsa_index_head_dim=16,
        dsa_index_topk=8,
        dsa_indexer_loss_coeff=0.0,
        dsa_indexer_use_sparse_loss=False,
        apply_rope_fusion=False,
        attention_dropout=0.0,
        attention_softmax_in_fp32=True,
        masked_softmax_fusion=False,
        csa_indexer_backend="unfused",
        csa_sparse_attn_backend="unfused",
        tensor_model_parallel_size=1,
        context_parallel_size=1,
        csa_dense_mode=False,
    )
    config.sequence_parallel = False
    config.rotary_interleaved = False
    config.rotary_percent = 1.0
    config.rope_type = "rope"
    config.rotary_base = 10000.0
    return config


def _make_docmask(doc_lens, seqlen):
    values = []
    end = 0
    for length in doc_lens:
        end += length
        values.extend([end] * length)
    if len(values) < seqlen:
        values.extend([end] * (seqlen - len(values)))
    return paddle.to_tensor(values, dtype="int32").reshape([1, 1, seqlen, 1])


def _build_csa(config, layer_number=0, is_mtp=False):
    spec = get_attention_spec(
        config=config,
        attention_layer_type="dsv4_hybrid_attention",
        attn_mask_type=AttnMaskType.causal,
        is_mtp_layer=is_mtp,
    )
    return build_spec_layer(spec, config=config, layer_number=layer_number)


@_GPU
class TestCSADocCacheEquivalence(unittest.TestCase):
    """``csa_share_docmask_meta`` on == off, bit-equal on out and grads.

    The dsv4 layer picks its micro-batch slot via the registry's own
    ``advance`` (``TransformerLayer.forward``), so no ``docmask_mb_idx`` is
    passed by the test -- driving the same entry point production uses.
    """

    def setUp(self):
        paddle.set_flags({"FLAGS_cudnn_deterministic": True})
        model_parallel_cuda_manual_seed(42)
        self.config = _make_csa_config()
        # Built with the switch already on so the layer registers its audit
        # key; ``_run`` toggles the live config per pass.
        self.config.csa_share_docmask_meta = True
        self.module = _build_csa(self.config, layer_number=0)
        self.seqlen, self.doc_lens = 32, [7, 25]
        self.mask = _make_docmask(self.doc_lens, self.seqlen)

    def _preload(self):
        doc_mask_meta_registry.begin_step(_ACC)
        for mb in range(_ACC):
            doc_mask_meta_registry.preload(
                mb,
                4,
                1,
                self.seqlen,
                self.mask,
                dense_mode=False,
                mask_group=("main",),
                window_size=self.config.csa_window_size,
            )

    def _run(self, use_cache):
        outs, grads = [], []
        self.config.csa_share_docmask_meta = use_cache
        if use_cache:
            self._preload()
        for mb in range(_ACC):
            paddle.seed(100 + mb)
            hidden = _leaf(
                paddle.randn([1, self.seqlen, 128], dtype="bfloat16")
            )
            self.module.clear_gradients()
            out, _bias = self.module(
                hidden,
                attn_mask_startend_row_indices=self.mask,
            )
            out.sum().backward()
            outs.append(out.cast("float32").numpy())
            grads.append(hidden.grad.cast("float32").numpy())
        if use_cache:
            doc_mask_meta_registry.check()
        return outs, grads

    def test_main_group_matches(self):
        ref_out, ref_grads = self._run(use_cache=False)
        cache_out, cache_grads = self._run(use_cache=True)
        for mb in range(_ACC):
            _assert_bitequal(cache_out[mb], ref_out[mb], f"csa out mb{mb}")
            _assert_relclose(cache_grads[mb], ref_grads[mb], f"csa dx mb{mb}")

    def test_mpg_group_fallback_matches(self):
        # MTP consumer group ("mtp", depth) is never preloaded: the lookup
        # fails and the layer builds privately; value must equal switch-off.
        self.module = _build_csa(self.config, layer_number=0, is_mtp=True)
        ref_out, ref_grads = self._run(use_cache=False)
        cache_out, cache_grads = self._run(use_cache=True)
        for mb in range(_ACC):
            _assert_bitequal(cache_out[mb], ref_out[mb], f"mtp csa out mb{mb}")
            _assert_relclose(
                cache_grads[mb], ref_grads[mb], f"mtp csa dx mb{mb}"
            )


if __name__ == "__main__":
    unittest.main()

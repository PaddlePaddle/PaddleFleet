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

"""A5 GRADIENT HEALTH validation for hybrid-MLA attention.

Adversarial gradient validation of ``non_absorbed_mqa`` for the hybrid-MLA
layer (csa ratio == -2, experimental_attention_variant == "dsv4_hybrid").

  * non_absorbed_mqa=False -- dense MLA (MLASelfAttention + DotProductAttention).
  * non_absorbed_mqa=True  -- runtime activation-level absorption on the shared
      KV latent PLUS a cuDNN block-sparse DSA indexer (gpt_layer_specs always
      builds the indexer for such a layer). With the forced local window
      (``csa_window_size``) >= sequence length the indexer selects no extra
      tokens, so the absorbed path degenerates to full per-document causal
      attention -- mathematically equal to the dense MHA phase, which is what
      the equivalence items below rely on.

The old 3-state ``hybrid_mla_attn_mode`` {mha, mqa, mqa_dsa} collapsed onto
this single boolean: "mha" -> non_absorbed_mqa=False and both "mqa"/"mqa_dsa"
-> non_absorbed_mqa=True (the DSA-less "mqa" is no longer reachable from
config, so its redundant parametrisation was dropped). The mode strings below
are kept only as readable labels for ``_build``.

The tests below deliberately look past cosine similarity and assert on
gradient *magnitude* (absolute + relative norms, per parameter), on the
analytic attention-sink gradient (validated against finite differences),
on gradient-flow completeness, on numerical health under stress, on the
bf16-vs-fp32 precision floor, and on multi-step training stability.

Run (SM100+ / Blackwell required for the absorbed cuDNN block-sparse kernel):

    PYTHONPATH=./third_party/PaddleFleet/src:./third_party/PaddleFormers \\
        CUDA_VISIBLE_DEVICES=4 python -m pytest <file> -q -p no:warnings -s
"""

import contextlib
import math
import unittest

import numpy as np
import paddle


def _try_use_cuda_device():
    if not paddle.is_compiled_with_cuda():
        return False
    try:
        paddle.set_device("gpu:0")
        place = str(paddle.empty([1]).place).lower()
    except Exception:
        return False
    return paddle.get_device().startswith("gpu") and (
        "gpu" in place or "cuda" in place
    )


_HAS_USABLE_CUDA = _try_use_cuda_device()

if not _HAS_USABLE_CUDA:
    # Keep import of the spec machinery from crashing on a CPU-only box.
    paddle.cuda = getattr(paddle, "cuda", paddle.device.cuda)
    paddle.cuda.get_device_capability = lambda device=None: (0, 0)
    paddle.device.cuda.get_device_capability = lambda device=None: (0, 0)

from paddle.distributed.fleet.meta_parallel import (
    build_spec_layer,
)

from paddlefleet.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
)
from paddlefleet.tensor_parallel.random import (
    model_parallel_cuda_manual_seed,
)
from paddlefleet.transformer.multi_latent_attention import (
    MLASelfAttention,  # noqa: F401  (import forces hybrid_mla_* field registration)
)
from paddlefleet.transformer.transformer_config import (
    TransformerConfig,
)

from .hybrid_mla_utils import _row_end as _row_end_from_docs

try:
    from paddlefleet.cudnn_ops.block_sparse_mqa_dsa import is_dsa_available
except Exception:  # pragma: no cover - kernel module absent

    def is_dsa_available():
        return False


_SEED = 42
_HAS_DSA = bool(_HAS_USABLE_CUDA and is_dsa_available())


def _skip_if_no_cuda(obj):
    return unittest.skipUnless(
        _HAS_USABLE_CUDA, "requires a usable CUDA device"
    )(obj)


# --------------------------------------------------------------------------- #
# Fake process-group collection (single card, no TP / CP).
# --------------------------------------------------------------------------- #
class _FakeGroup:
    def __init__(self, nranks=1):
        self.nranks = nranks
        self.world_size = nranks
        self.ranks = list(range(nranks))
        self.rank = 0


class _FakePGCollection:
    def __init__(self, tp_nranks=1, cp_nranks=1):
        self.tp = _FakeGroup(tp_nranks)
        self.cp = _FakeGroup(cp_nranks)


# --------------------------------------------------------------------------- #
# Config + module construction.
#
# Geometry is deliberately small (hidden 256) but keeps the *hybrid-MLA*
# invariants the kernels require: >= 64 heads, head_dim 128 indexer, topk % 128
# == 0, kv_lora_rank 512 + qk_rope 64 == 576 latent key width, v == 512.
# --------------------------------------------------------------------------- #
def _make_config(mode, sink, indexer=False, dtype=paddle.bfloat16):
    # ``indexer`` is vestigial: the hybrid-MLA layer's DSA indexer is now built
    # unconditionally whenever ``non_absorbed_mqa`` is True (see
    # gpt_layer_specs), so it is retained only for call-site compatibility.
    del indexer
    non_absorbed_mqa = mode != "mha"
    cfg = TransformerConfig(
        num_hidden_layers=2,
        num_nextn_predict_layers=0,
        hidden_size=256,
        num_attention_heads=8,
        params_dtype=dtype,
        bf16=(dtype == paddle.bfloat16),
        use_bias=False,
        multi_latent_attention=True,
        experimental_attention_variant="dsv4_hybrid",
        q_lora_rank=64,
        kv_lora_rank=16,
        qk_nope_head_dim=16,
        qk_rope_head_dim=16,
        qk_pos_emb_head_dim=16,
        v_head_dim=32,
        hybrid_mla_q_lora_rank=1536,
        hybrid_mla_kv_lora_rank=512,
        hybrid_mla_qk_nope_head_dim=192,
        hybrid_mla_qk_rope_head_dim=64,
        hybrid_mla_v_head_dim=256,
        hybrid_mla_num_attention_heads=64,
        hybrid_mla_num_key_value_heads=64,
        non_absorbed_mqa=non_absorbed_mqa,
        add_full_attention_sink_bias=sink,
        o_groups=4,
        o_lora_rank=32,
        rope_type="rope",
        rotary_base=10000.0,
        rotary_percent=1.0,
        normalization="RMSNorm",
        use_qk_norm=True,
        csa_compress_ratios=[-2, 128],
        csa_window_size=128,
        # Indexer dims are model-wide now (HF json ``index_*``). The hybrid MLA
        # -2 layer's cuDNN indexer requires head_dim == 128 and topk % 128 == 0
        # (<= 2048); these carry the values the old ``hybrid_index_*`` fields
        # held and satisfy the ``non_absorbed_mqa`` config validation.
        dsa_index_n_heads=64,
        dsa_index_head_dim=128,
        dsa_index_topk=128,
        dsa_indexer_loss_coeff=1.0,
        dsa_indexer_use_sparse_loss=False,
        dsa_indexer_rotary_interleaved=False,
        apply_rope_fusion=False,
        attention_dropout=0.0,
        attention_softmax_in_fp32=True,
        masked_softmax_fusion=False,
        softmax_type="vanilla",
        csa_indexer_backend="unfused",
        csa_sparse_attn_backend="unfused",
        tensor_model_parallel_size=1,
        context_parallel_size=1,
        csa_dense_mode=False,
    )
    return cfg


def _build(mode, sink=False, indexer=False, dtype=paddle.bfloat16):
    """Build the layer-0 (hybrid-MLA) self-attention for ``mode``."""
    model_parallel_cuda_manual_seed(_SEED)
    cfg = _make_config(mode, sink, indexer, dtype=dtype)
    spec = get_gpt_layer_local_spec(
        config=cfg,
        normalization=cfg.normalization,
        layer_number=0,
    ).sublayers_spec.self_attn
    with _fa4_for_mha_sink(mode, sink):
        module = build_spec_layer(
            spec, config=cfg, layer_number=0, pg_collection=_FakePGCollection()
        )
    return module


@contextlib.contextmanager
def _fa4_for_mha_sink(mode, sink):
    """Satisfy the ``mha`` + sink FA4 requirement for the duration of a build.

    ``MultiLatentAttention.__init__`` refuses to create the sink for ``mha``
    unless ``FLAGS_flash_attn_version in (3, 4)``, because ``mha`` consumes it as
    ``flashmask_attention_func(learnable_sink=...)`` which only exists on the
    cute path (multi_latent_attention.py, guard next to the parameter creation).
    The default flag value in this image is 2. These tests only inspect the
    parameter, so we flip the flag around construction and restore it -- the
    process-global default must stay untouched for the other suites.
    """
    if not (sink and mode == "mha"):
        yield
        return
    previous = paddle.get_flags(["FLAGS_flash_attn_version"])[
        "FLAGS_flash_attn_version"
    ]
    paddle.set_flags({"FLAGS_flash_attn_version": 4})
    try:
        yield
    finally:
        paddle.set_flags({"FLAGS_flash_attn_version": previous})


# Reachable modes (label, non_absorbed_mqa).  Via config there are only two:
# dense MHA (non_absorbed_mqa=False) and the absorbed MQA+DSA-indexer path
# (non_absorbed_mqa=True).  The old DSA-less "mqa" collapsed onto "mqa_dsa"
# (indexer always built), so its redundant entry was dropped.
_MODES = (("mha", False), ("mqa_dsa", True))

# Parameters we expect to carry gradient in the hybrid-MLA layer (weights only;
# the sink is handled separately because mha's is inert here, see item 3).
_EXPECTED_PARAMS = (
    "q_a_proj.weight",
    "q_b_proj.weight",
    "kv_a_proj_with_mqa.weight",
    "kv_b_proj.weight",
    "o_proj.weight",
    "q_a_layernorm.weight",
    "kv_a_layernorm.weight",
)


def _row_end(seqlen, doc_lens=None):
    """Exclusive per-token document-end index, shape [1, 1, s, 1] int32.

    Argument order is flipped w.r.t. the shared fixture because every call site
    here passes only the sequence length.
    """
    return _row_end_from_docs(doc_lens or [seqlen], seqlen)


def _hidden(seqlen, scale=1.0, seed=0, dtype=paddle.bfloat16):
    rng = np.random.RandomState(1234 + seed)
    x = rng.randn(1, seqlen, 256).astype("float32") * scale
    t = paddle.to_tensor(x, dtype=dtype)
    t.stop_gradient = False
    return t


def _forward_loss(module, x, row_end, weighted=False, wseed=7):
    """Run forward and return a scalar loss (fp32)."""
    module.eval()  # autoscaler PyLayer inplace-safety (see test_mqa_latent)
    out, _bias = module(
        x, attention_mask=None, attn_mask_startend_row_indices=row_end
    )
    out = out.astype("float32")
    if weighted:
        rng = np.random.RandomState(wseed)
        w = paddle.to_tensor(
            rng.randn(*out.shape).astype("float32"), dtype="float32"
        )
        return (out * w).sum()
    return out.sum()


def _grads(module, x, row_end, weighted=False, wseed=7):
    """Backward once; return {name: fp32 numpy grad} for params with grad."""
    module.clear_gradients()
    loss = _forward_loss(module, x, row_end, weighted=weighted, wseed=wseed)
    loss.backward()
    grads = {}
    for name, p in module.named_parameters():
        if p.grad is not None:
            grads[name] = p.grad.astype("float32").numpy()
    return float(loss.astype("float32")), grads


def _cos(a, b):
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 and nb == 0:
        return 1.0
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _norm(a):
    return float(np.linalg.norm(a.ravel().astype(np.float64)))


def _rel_err(a, b):
    """Relative L2 error of a vs reference b."""
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    denom = max(np.linalg.norm(b), 1e-12)
    return float(np.linalg.norm(a - b) / denom)


def _short(name):
    """Trailing-most identifying piece of a long qualified param name."""
    parts = name.split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else name


@_skip_if_no_cuda
class TestItem1StateDictAndGradientMatch(unittest.TestCase):
    """Item 1: an MHA checkpoint is a strict subset of the MQA key set and
    loads unchanged (byte-identical), the sink shares its name across phases;
    per-param gradient cosine / rel-err / norm-RATIO of mqa vs mha reference
    (the forced local window >= seq makes the absorbed path full-causal, i.e.
    mathematically equal to dense MHA)."""

    SEQ = 128

    def test_mqa_consumes_mha_state_dict_unchanged(self):
        # Build-only (no forward), so this runs without the DSA kernel.
        mha = _build("mha", sink=True)
        mqa = _build("mqa", sink=True)
        mha_sd = mha.state_dict()
        mqa_sd = mqa.state_dict()
        mha_keys = set(mha_sd)
        mqa_keys = set(mqa_sd)
        # Activation-level absorption keeps every dense-MLA parameter
        # byte-identical and shares the sink name; the MQA phase only ADDS the
        # DSA indexer's weights.  So MHA keys are a subset and an MHA
        # checkpoint loads unchanged.
        self.assertTrue(
            mha_keys <= mqa_keys, "mqa dropped an mha parameter name"
        )
        extra = mqa_keys - mha_keys
        self.assertTrue(extra, "mqa must add the DSA indexer parameters")
        self.assertTrue(
            all("indexer" in k for k in extra),
            f"mqa added non-indexer keys: {sorted(extra)}",
        )
        # The sink parameter is present in both phases under the same name.
        sink_keys = [k for k in mha_keys if k.endswith("softmax_offset")]
        self.assertTrue(sink_keys, "mha sink parameter missing")
        for k in sink_keys:
            self.assertIn(k, mqa_keys, "sink name diverged between phases")
        # Load the MHA checkpoint into the MQA module; shared keys stay
        # byte-identical (the indexer keys are left at their init values).
        mqa.set_state_dict(mha_sd)
        reloaded = mqa.state_dict()
        for name in mha_sd:
            ref = mha_sd[name].astype("float32").numpy()
            got = reloaded[name].astype("float32").numpy()
            np.testing.assert_array_equal(
                got, ref, err_msg=f"{name} changed on load"
            )

    def test_mqa_consumes_mha_state_dict_when_sink_is_enabled_together(self):
        """The production migration flips TWO json fields at once.

        ``ernielite_layer43_mla_hca`` -> ``..._non_absorbed_mqa_hca_dsa`` adds
        ``non_absorbed_mqa: true`` *and* ``add_full_attention_sink_bias: true``,
        so the phase-1 checkpoint has no ``softmax_offset`` at all. The test
        above builds both phases with the sink already on; this one covers the
        real delta: the newly initialised set must be exactly the indexer's
        weights plus the one sink vector, with nothing dropped or renamed, which
        is what makes the phase-1 checkpoint loadable with no conversion step.
        """
        mha = _build("mha", sink=False)
        mqa = _build("mqa", sink=True)
        mha_sd = mha.state_dict()
        mha_keys, mqa_keys = set(mha_sd), set(mqa.state_dict())
        self.assertEqual(
            mha_keys - mqa_keys, set(), "mqa dropped an mha parameter name"
        )
        extra = mqa_keys - mha_keys
        sink = {k for k in extra if k.endswith("softmax_offset")}
        self.assertEqual(len(sink), 1, f"expected one sink, got {sorted(sink)}")
        self.assertTrue(
            all("indexer" in k for k in extra - sink),
            f"mqa added non-indexer, non-sink keys: {sorted(extra - sink)}",
        )
        mqa.set_state_dict(mha_sd)
        reloaded = mqa.state_dict()
        for name in mha_sd:
            np.testing.assert_array_equal(
                reloaded[name].astype("float32").numpy(),
                mha_sd[name].astype("float32").numpy(),
                err_msg=f"{name} changed on load",
            )

    def test_per_param_gradient_match_mqa_vs_mha(self):
        if not _HAS_DSA:
            self.skipTest("absorbed MQA forward requires SM100+ DSA kernel")
        re = _row_end(self.SEQ)
        mha = _build("mha", sink=False)
        mqa = _build("mqa", sink=False)
        mqa.set_state_dict(mha.state_dict())
        _, g_mha = _grads(mha, _hidden(self.SEQ, seed=0), re, weighted=True)
        _, g_mqa = _grads(mqa, _hidden(self.SEQ, seed=0), re, weighted=True)
        for key in _EXPECTED_PARAMS:
            n_mha = [k for k in g_mha if k.endswith(key)]
            n_mqa = [k for k in g_mqa if k.endswith(key)]
            self.assertTrue(n_mha and n_mqa, f"missing grad for {key}")
            a, b = g_mqa[n_mqa[0]], g_mha[n_mha[0]]
            cos = _cos(a, b)
            nb = _norm(b)
            ratio = (_norm(a) / nb) if nb > 0 else float("nan")
            self.assertGreater(cos, 0.999, f"{key} cosine too low ({cos})")
            self.assertTrue(
                0.95 < ratio < 1.05, f"{key} grad-norm ratio off ({ratio})"
            )


@_skip_if_no_cuda
class TestItem2AbsoluteMagnitudeSanity(unittest.TestCase):
    """Item 2: guard against missing / double-applied softmax scale (0.0625),
    a missing 1/sqrt(d) in the de-absorption einsum, or a sum-vs-mean-over-
    heads error.  All of these show up as a *forward activation* mismatch
    between mqa and mha, and as a per-param grad-norm ratio clustered on a
    suspicious constant (num_heads=64, scale=0.0625, 1/scale=16, sqrt(256)=16).
    """

    SEQ = 128

    def test_forward_activation_equivalence(self):
        if not _HAS_DSA:
            self.skipTest("absorbed MQA forward requires SM100+ DSA kernel")
        x = _hidden(self.SEQ, seed=3)
        re = _row_end(self.SEQ)
        mha = _build("mha", sink=False)
        mqa = _build("mqa", sink=False)
        mqa.set_state_dict(mha.state_dict())
        mha.eval()
        mqa.eval()
        o_mha = mha(x, attention_mask=None, attn_mask_startend_row_indices=re)[
            0
        ].astype("float32")
        o_mqa = mqa(x, attention_mask=None, attn_mask_startend_row_indices=re)[
            0
        ].astype("float32")
        a = o_mha.numpy()
        b = o_mqa.numpy()
        rel = _rel_err(b, a)
        # Measured 3.96e-3 .. 4.05e-3 over seeds 0/3/7 at seq 128 -- the two
        # kernels (dense flashmask vs FlashMLA sparse) accumulate bf16 in a
        # different order; each is bit-reproducible against itself (self rel
        # 0.0). The bound leaves ~2.5x headroom over that floor. If the softmax
        # scale were dropped/doubled, or the einsum lost a 1/sqrt(d), the
        # outputs would differ by orders of magnitude more.
        self.assertLess(
            rel, 1e-2, "mqa forward diverges from mha -> scale/einsum bug"
        )

    def test_grad_norm_ratio_not_on_a_suspicious_constant(self):
        if not _HAS_DSA:
            self.skipTest("absorbed MQA forward requires SM100+ DSA kernel")
        re = _row_end(self.SEQ)
        mha = _build("mha", sink=False)
        mqa = _build("mqa", sink=False)
        mqa.set_state_dict(mha.state_dict())
        _, g_mha = _grads(mha, _hidden(self.SEQ, seed=4), re, weighted=True)
        _, g_mqa = _grads(mqa, _hidden(self.SEQ, seed=4), re, weighted=True)
        suspicious = {
            "num_heads": 64.0,
            "scale": 0.0625,
            "1/scale": 16.0,
            "sqrt_qhd": 16.0,
        }
        for key in _EXPECTED_PARAMS:
            a = g_mqa[next(k for k in g_mqa if k.endswith(key))]
            b = g_mha[next(k for k in g_mha if k.endswith(key))]
            nb = _norm(b)
            ratio = _norm(a) / nb if nb else float("nan")
            for label, c in suspicious.items():
                self.assertFalse(
                    abs(ratio - c) < 0.2 * c,
                    f"{key} grad ratio {ratio:.4f} ~= {label}({c})",
                )


@_skip_if_no_cuda
class TestItem3SinkGradient(unittest.TestCase):
    """Item 3: analytic attention-sink gradient.

    Kernel computes d_sink analytically (cuDNN returns zeros):
        Delta[b,s,h] = sum_dv(out * dO)
        p_sink[b,s,h] = exp(sink[h] - lse_full[b,s,h])
        d_sink[h]     = -sum_{b,s}(p_sink * Delta)
    We validate it against a finite-difference directional derivative and
    quantify d(sink)/d(o_proj).  mha's sink is compared where live.
    """

    SEQ = 64

    def _sink_grad(self, mode, indexer):
        m = _build(mode, sink=True, indexer=indexer)
        x = _hidden(self.SEQ, seed=1)
        re = _row_end(self.SEQ)
        _, g = _grads(m, x, re, weighted=False)
        keys = [k for k in g if k.endswith("softmax_offset")]
        return (g[keys[0]] if keys else None), m

    def test_mqa_sink_grad_finite_nonzero_and_matches_finite_diff(self):
        if not _HAS_DSA:
            self.skipTest("absorbed MQA forward requires SM100+ DSA kernel")
        g, m = self._sink_grad("mqa", indexer=False)
        self.assertIsNotNone(g, "mqa sink must receive gradient")
        self.assertTrue(np.all(np.isfinite(g)), "mqa sink grad non-finite")
        self.assertGreater(np.abs(g).max(), 0.0, "mqa sink grad all-zero")
        analytic_sum = float(g.sum())

        # Finite-difference directional derivative d/dt L(sink = t*1) at t=0.
        # NB: mutate the sink IN PLACE (``add_``). The absorbed DSA kernel binds
        # to the parameter's tensor storage on its first forward (already run by
        # ``_grads`` above); a fresh-tensor ``set_value`` would swap that storage
        # out and the kernel would read a stale buffer (observably: zero output),
        # so the perturbation must be applied to the existing storage.
        x = _hidden(self.SEQ, seed=1)
        re = _row_end(self.SEQ)
        sink = m.core_attention.softmax_offset
        h = 0.25
        with paddle.no_grad():
            sink.add_(paddle.full_like(sink, h))
        lp = float(_forward_loss(m, x, re).astype("float32"))
        with paddle.no_grad():
            sink.add_(paddle.full_like(sink, -2 * h))
        lm = float(_forward_loss(m, x, re).astype("float32"))
        with paddle.no_grad():
            sink.add_(paddle.full_like(sink, h))
        fd = (lp - lm) / (2 * h)
        rel = abs(fd - analytic_sum) / max(abs(fd), abs(analytic_sum), 1e-6)
        self.assertLess(
            rel, 0.30, "analytic sink grad disagrees with finite diff"
        )


@_skip_if_no_cuda
class TestItem4GradientFlowCompleteness(unittest.TestCase):
    """Item 4: every parameter that should receive a gradient does -- no None,
    no all-zero -- in all reachable modes, sink on and off."""

    SEQ = 64

    def _check(self, mode, sink, indexer):
        m = _build(mode, sink=sink, indexer=indexer)
        _, g = _grads(
            m, _hidden(self.SEQ, seed=2), _row_end(self.SEQ), weighted=True
        )
        missing, dead = [], []
        for key in _EXPECTED_PARAMS:
            hit = [k for k in g if k.endswith(key)]
            if not hit:
                missing.append(key)
            elif float(np.abs(g[hit[0]]).max()) == 0.0:
                dead.append(key)
        sink_state = "n/a"
        if sink:
            sk = [k for k in g if k.endswith("softmax_offset")]
            if not sk:
                sink_state = "MISSING"
            elif float(np.abs(g[sk[0]]).max()) == 0.0:
                sink_state = "DEAD"
            else:
                sink_state = "live"
        return missing, dead, sink_state

    def test_all_weight_params_receive_gradient(self):
        for mode, idx in _MODES:
            if mode == "mqa_dsa" and not _HAS_DSA:
                continue
            for sink in (False, True):
                if mode == "mha" and sink:
                    # mha + sink cannot run here (FA4 cute kernel absent); the
                    # forward raises.  Documented under item 3; skip flow check.
                    continue
                with self.subTest(mode=mode, sink=sink):
                    missing, dead, sink_state = self._check(mode, sink, idx)
                    self.assertEqual(missing, [], f"{mode}: missing grads")
                    self.assertEqual(dead, [], f"{mode}: all-zero grads")
                    if sink and mode != "mha":
                        self.assertEqual(
                            sink_state,
                            "live",
                            f"{mode}: sink got no usable gradient",
                        )


@_skip_if_no_cuda
class TestItem5NumericalHealthUnderStress(unittest.TestCase):
    """Item 5: no non-finite gradients under large / tiny inputs, long
    sequences, doc-length-1 masking, and saturated sink logits (+/-30)."""

    def _run(self, mode, indexer, seq, scale, doc_lens, sink_fill):
        m = _build(mode, sink=(sink_fill is not None), indexer=indexer)
        if sink_fill is not None:
            sk = m.core_attention.softmax_offset
            with paddle.no_grad():
                sk.set_value(
                    paddle.full(sk.shape, float(sink_fill), dtype=sk.dtype)
                )
        x = _hidden(seq, scale=scale, seed=5)
        re = _row_end(seq, doc_lens=doc_lens)
        try:
            _, g = _grads(m, x, re, weighted=True)
        except Exception as e:
            return "EXC:" + type(e).__name__, {}
        bad = [_short(k) for k, v in g.items() if not np.all(np.isfinite(v))]
        return ("finite" if not bad else "NONFINITE"), bad

    def test_mqa_numerical_stress(self):
        if not _HAS_DSA:
            self.skipTest("absorbed MQA forward requires SM100+ DSA kernel")
        seq = 256
        configs = [
            ("large_x100", seq, 100.0, None, None),
            ("tiny_x1e-3", seq, 1e-3, None, None),
            ("long_seq_512", 512, 1.0, None, None),
            ("doclen1", seq, 1.0, [1] * seq, None),
            ("sink_+30", seq, 1.0, None, 30.0),
            ("sink_-30", seq, 1.0, None, -30.0),
        ]
        first_bad = None
        for label, s, scale, docs, sink in configs:
            status, _ = self._run("mqa", False, s, scale, docs, sink)
            if status != "finite" and first_bad is None:
                first_bad = label
        self.assertIsNone(
            first_bad, f"non-finite gradients first at: {first_bad}"
        )

    def test_mqa_dsa_numerical_stress(self):
        if not _HAS_DSA:
            self.skipTest("mqa_dsa requires SM100+ DSA kernel")
        seq = 256
        configs = [
            ("large_x100", seq, 100.0, None, None),
            ("tiny_x1e-3", seq, 1e-3, None, None),
            ("sink_+30", seq, 1.0, None, 30.0),
            ("sink_-30", seq, 1.0, None, -30.0),
        ]
        first_bad = None
        for label, s, scale, docs, sink in configs:
            status, _ = self._run("mqa_dsa", True, s, scale, docs, sink)
            if status != "finite" and first_bad is None:
                first_bad = label
        self.assertIsNone(first_bad)


@_skip_if_no_cuda
class TestItem6DtypePrecisionFloor(unittest.TestCase):
    """Item 6: bf16 vs fp32.  The mha path runs on generic matmuls and can be
    built in fp32, giving a precision-floor reference for the bf16 gradients.
    (The mqa/mqa_dsa flash kernels are bf16-only, so their floor is measured
    against the bf16 mha reference instead -- see item 1.)"""

    SEQ = 64

    def test_bf16_vs_fp32_mha_grad_floor(self):
        re = _row_end(self.SEQ)
        try:
            m32 = _build("mha", sink=False, dtype=paddle.float32)
            _, g32 = _grads(
                m32,
                _hidden(self.SEQ, seed=0, dtype=paddle.float32),
                re,
                weighted=True,
            )
        except Exception as e:
            self.skipTest(f"fp32 mha path unavailable: {type(e).__name__}: {e}")
            return
        m16 = _build("mha", sink=False, dtype=paddle.bfloat16)
        # Load fp32 weights (cast) so both modules share identical weights.
        sd = {k: v.astype("bfloat16") for k, v in m32.state_dict().items()}
        m16.set_state_dict(sd)
        _, g16 = _grads(m16, _hidden(self.SEQ, seed=0), re, weighted=True)
        worst = 0.0
        for key in _EXPECTED_PARAMS:
            a = g16[next(k for k in g16 if k.endswith(key))]
            b = g32[next(k for k in g32 if k.endswith(key))]
            rel = _rel_err(a.astype(np.float64), b.astype(np.float64))
            worst = max(worst, rel)
        # bf16 has ~3 decimal digits; a few % relative error is the floor.
        self.assertLess(worst, 0.10, "bf16 grad error exceeds bf16 floor")


@_skip_if_no_cuda
class TestItem7MultiStepStability(unittest.TestCase):
    """Item 7: ~20 SGD steps on a fixed batch for mqa_dsa (with sink) and mha
    as a control; loss and grad-norm must stay finite and not blow up."""

    SEQ = 64
    STEPS = 20
    LR = 1e-3

    def _run_steps(self, mode, indexer, sink):
        m = _build(mode, sink=sink, indexer=indexer)
        x = _hidden(self.SEQ, seed=8)
        re = _row_end(self.SEQ)
        opt = paddle.optimizer.SGD(
            learning_rate=self.LR, parameters=m.parameters()
        )
        # Bounded supervised objective (MSE to a fixed random target) so that
        # divergence reflects instability, not an unbounded linear loss.
        rng = np.random.RandomState(99)
        target = paddle.to_tensor(
            rng.randn(1, self.SEQ, 256).astype("float32"), dtype="float32"
        )
        losses, gnorms = [], []
        for _ in range(self.STEPS):
            m.clear_gradients()
            m.eval()
            out, _ = m(
                x, attention_mask=None, attn_mask_startend_row_indices=re
            )
            loss = ((out.astype("float32") - target) ** 2).mean()
            loss.backward()
            gn = 0.0
            for p in m.parameters():
                if p.grad is not None:
                    gn += float((p.grad.astype("float32") ** 2).sum())
            gn = math.sqrt(gn)
            opt.step()
            losses.append(float(loss.astype("float32")))
            gnorms.append(gn)
        return losses, gnorms

    def _report(self, tag, losses, gnorms):
        self.assertTrue(all(np.isfinite(losses)), f"{tag}: non-finite loss")
        self.assertTrue(all(np.isfinite(gnorms)), f"{tag}: non-finite gnorm")
        # No runaway: final grad norm within 100x of the first.
        self.assertLess(
            gnorms[-1], 100.0 * gnorms[0] + 1e-6, f"{tag}: grad norm blew up"
        )

    def test_mha_control(self):
        losses, gnorms = self._run_steps("mha", False, sink=False)
        self._report("mha", losses, gnorms)

    def test_mqa_dsa_with_sink(self):
        if not _HAS_DSA:
            self.skipTest("mqa_dsa requires SM100+ DSA kernel")
        losses, gnorms = self._run_steps("mqa_dsa", True, sink=True)
        self._report("mqa_dsa", losses, gnorms)


@_skip_if_no_cuda
class TestItem8WeightDecayInteraction(unittest.TestCase):
    """Item 8: the sink is 1-D so Muon skips it, but the AdamW weight-decay
    filter (ernie5/src/trainers/pretraining_trainer.py) only excludes
    bias/norm/multimax names -- so the sink RECEIVES weight decay.  Quantify
    the decay pull vs the true gradient signal at lr=3.2e-4, wd=0.1."""

    SEQ = 64
    LR = 3.2e-4
    WD = 0.1

    def test_weight_decay_pull_on_sink(self):
        if not _HAS_DSA:
            self.skipTest("absorbed MQA forward requires SM100+ DSA kernel")
        m = _build("mqa", sink=True, indexer=False)
        _, g = _grads(
            m, _hidden(self.SEQ, seed=1), _row_end(self.SEQ), weighted=False
        )
        gs = g[next(k for k in g if k.endswith("softmax_offset"))]
        self.assertTrue(np.all(np.isfinite(gs)), "sink grad non-finite")
        # Emulate a trained sink logit magnitude of O(1) (grads above were
        # single-digit at zero-init after one step).
        sink_val = 1.0
        grad_step = self.LR * float(np.abs(gs).mean())
        decay_step = self.LR * self.WD * sink_val
        # The denominator must be a real measurement, not a clamp floor:
        # otherwise the ratio below is a division artifact and asserting on it
        # proves nothing.
        self.assertGreater(grad_step, 1e-12, "sink grad is all-zero")
        ratio = decay_step / grad_step
        # Decay is a real, non-negligible systematic pull toward 0 on the sink:
        # |g_sink|.mean() is O(1) here, so ratio lands near 1e-1. Bound at 1e-3
        # (100x headroom) -- it only fails if the gradient signal grows to
        # ~100x the emulated logit, which would make decay genuinely
        # negligible and invalidate this item's conclusion.
        self.assertGreater(ratio, 1e-3)


@_skip_if_no_cuda
class TestItem9InputIdsReachTheIndexerLossMask(unittest.TestCase):
    """``input_ids`` must reach the MQA core attention, and only that one.

    The indexer-loss row mask has to come from ``input_ids != pad_token_id``:
    ``attn_mask_startend_row_indices`` folds a packed sequence's trailing
    padding into the last document, so the document metadata alone reports those
    rows as valid. ``MultiLatentAttention.forward`` therefore forwards
    ``input_ids`` to ``core_attention`` -- but only under ``mqa_latent``, since
    ``DotProductAttention.forward`` has no such parameter and no ``**kwargs``.
    """

    SEQ = 64

    def test_mqa_core_attention_receives_input_ids(self):
        if not _HAS_DSA:
            self.skipTest("absorbed MQA forward requires SM100+ DSA kernel")
        module = _build("mqa_dsa")
        seen = {}
        inner = module.core_attention.forward

        def recording(*args, **kwargs):
            seen["input_ids"] = kwargs.get("input_ids")
            return inner(*args, **kwargs)

        module.core_attention.forward = recording
        module.eval()
        input_ids = paddle.ones([1, self.SEQ], dtype="int64")
        module(
            _hidden(self.SEQ),
            attention_mask=None,
            attn_mask_startend_row_indices=_row_end(self.SEQ),
            input_ids=input_ids,
        )
        self.assertIsNotNone(seen["input_ids"])
        self.assertEqual(list(seen["input_ids"].shape), [1, self.SEQ])

    def test_mha_core_attention_is_not_given_input_ids(self):
        # Would raise TypeError if the kwarg were forwarded unconditionally.
        module = _build("mha")
        module.eval()
        module(
            _hidden(self.SEQ),
            attention_mask=None,
            attn_mask_startend_row_indices=_row_end(self.SEQ),
            input_ids=paddle.ones([1, self.SEQ], dtype="int64"),
        )


class TestItem10IndexerKLValueAndDenominator(unittest.TestCase):
    """The indexer KL scalar and its reduction denominator, at production settings.

    Item 9 pins that ``input_ids`` reaches the layer; the existing width tests
    pin the loss-table width. Neither pins the *numeric value* of the forward
    indexer KL nor that its reduction denominator is the number of non-pad
    tokens (``input_ids != pad_token_id``) rather than ``B*Sq``. Production runs
    ``dsa_indexer_use_sparse_loss=False`` and ``dsa_indexer_loss_coeff=0.01``
    with packed sequences whose tail is padding, so this fixes that path:

      * the logged KL equals an independent fp32 recomputation from the same
        (topk_probs, target, row-mask) the layer used, to < 1e-4;
      * the reduction denominator (``num_rows_override`` handed to the backward)
        equals the non-pad token count, so forward and backward agree exactly;
      * the coefficient is applied exactly once (linear in coeff).
    """

    SEQ = 256
    NPAD = 48  # trailing padding folded into the last document's row range

    def _run_and_capture(self, coeff):
        import paddlefleet.transformer.mqa_latent_attention as mqamod
        from paddlefleet.transformer.dsa_attention import (
            DSAIndexerLossLoggingHelper as LOG,
        )

        cap = {}
        real = mqamod.TileLangCSAIndexerLossAutoScaler

        class Spy:
            @staticmethod
            def apply(
                output,
                index_q,
                weights,
                index_k_comp,
                topk_indices,
                topk_probs,
                target,
                loss_coeff,
                backend="tilelang",
                num_rows_override=None,
                loss_mask=None,
            ):
                cap["probs"] = topk_probs.detach().astype("float32").numpy()
                cap["target"] = target.detach().astype("float32").numpy()
                cap["mask"] = (
                    None
                    if loss_mask is None
                    else loss_mask.detach().astype("float32").numpy()
                )
                cap["num_rows"] = (
                    None
                    if num_rows_override is None
                    else float(num_rows_override)
                )
                cap["coeff"] = float(loss_coeff)
                return real.apply(
                    output,
                    index_q,
                    weights,
                    index_k_comp,
                    topk_indices,
                    topk_probs,
                    target,
                    loss_coeff,
                    backend,
                    num_rows_override,
                    loss_mask,
                )

        module = _build("mqa_dsa")
        module.core_attention.indexer_loss_coeff = float(coeff)
        module.core_attention.indexer_use_sparse_loss = False
        module.train()
        ids = np.ones((1, self.SEQ), dtype="int64")
        ids[0, self.SEQ - self.NPAD :] = 0  # pad_token_id == 0
        input_ids = paddle.to_tensor(ids)
        mqamod.TileLangCSAIndexerLossAutoScaler = Spy
        LOG.clean_loss_in_tracker()
        try:
            module(
                _hidden(self.SEQ),
                attention_mask=None,
                attn_mask_startend_row_indices=_row_end(
                    self.SEQ, doc_lens=[self.SEQ]
                ),
                input_ids=input_ids,
            )
        finally:
            mqamod.TileLangCSAIndexerLossAutoScaler = real
        logged = float(LOG.tracker["values"].astype("float32").sum())
        return logged, cap

    def test_kl_value_denominator_and_coeff(self):
        if not _HAS_DSA:
            self.skipTest("absorbed MQA forward requires SM100+ DSA kernel")
        logged, cap = self._run_and_capture(coeff=0.01)
        nonpad = float(self.SEQ - self.NPAD)

        # denominator is the non-pad token count, shared with the backward.
        self.assertIsNotNone(cap["mask"])
        self.assertEqual(float(cap["mask"].sum()), nonpad)
        self.assertEqual(cap["num_rows"], nonpad)

        # loss-table width is widened (sparse_loss=False) beyond attn top-k 128.
        self.assertGreater(cap["target"].shape[-1], 128)

        # independent fp32 recomputation of the masked KL mean.
        t, p = cap["target"], cap["probs"]
        eps = 1e-10
        kl_per_pos = (t * (np.log(t + eps) - np.log(p + eps))).sum(axis=-1)
        ref = (kl_per_pos * cap["mask"]).sum() / nonpad * cap["coeff"]
        self.assertLess(abs(logged - ref) / max(abs(ref), 1e-12), 1e-4)

        # coefficient applied exactly once: 10x coeff -> ~10x logged KL.
        logged10, _ = self._run_and_capture(coeff=0.1)
        self.assertAlmostEqual(logged10 / logged, 10.0, delta=0.05)


if __name__ == "__main__":
    unittest.main()

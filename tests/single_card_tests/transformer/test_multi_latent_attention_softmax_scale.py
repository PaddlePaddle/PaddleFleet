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

"""Behaviour tests for the MLA softmax-scale computation and its propagation
to the core-attention build.

Two independently verifiable surfaces of
``src/paddlefleet/transformer/multi_latent_attention.py``:

1. ``_yarn_get_mscale`` (``yarn_rotary_pos_embedding.py:295-298``) -- the YaRN
   concentration factor that feeds the scale, checked against hand-derived
   numeric constants covering both of its branches.
2. The ``MultiLatentAttention.__init__`` propagation
   (``multi_latent_attention.py:409-414`` and the core-attention build at
   :512-527): ``softmax_scale = mscale**2 / sqrt(q_head_dim)`` and
   ``_softmax_scale_arg = None if mscale == 1.0 else softmax_scale``, the latter
   being what is actually handed to ``build_spec_layer(core_attention, ...)``.

Surface 2 constructs a *real* ``MLASelfAttention`` and captures the ``softmax_scale``
keyword the production ``__init__`` passes to the core-attention build, rather
than re-deriving the branch in the test body. Only ``build_spec_layer`` -- a
genuine, not-under-test collaborator -- is replaced, and by a recorder that
returns a stub whose ``softmax_offset`` is ``None`` so the real sink guard
(:557-584) stays on its no-sink path. The YaRN rotary module is left real.

CPU-only; skips honestly when Paddle is unavailable.
"""

import math
import unittest
import unittest.mock

try:
    import paddle  # noqa: F401

    from paddlefleet.models.common.embeddings.yarn_rotary_pos_embedding import (
        _yarn_get_mscale,
    )
    from paddlefleet.transformer import (
        attention as attention_mod,
        multi_latent_attention as mla_mod,
    )
    from paddlefleet.transformer.enums import AttnMaskType
    from paddlefleet.transformer.multi_latent_attention import (
        MLASelfAttention,
        MLASelfAttentionSublayersSpec,
    )
    from paddlefleet.transformer.transformer_config import TransformerConfig

    _IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
    _IMPORT_ERROR = str(exc)


_SKIP_MSG = (
    f"paddle/paddlefleet import unavailable ({_IMPORT_ERROR}); "
    "run on the single-card (H20) CI where Paddle is installed"
    if _IMPORT_ERROR
    else ""
)


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_MSG)
class TestYarnGetMscale(unittest.TestCase):
    """``_yarn_get_mscale`` against hand-derived numeric constants.

    Formula (production): ``scale <= 1 -> 1.0`` else
    ``0.1 * mscale * ln(scale) + 1.0`` where ``mscale`` is ``mscale_all_dim``.
    """

    def test_scale_at_or_below_one_is_exactly_one(self):
        # scale <= 1 short-circuits to 1.0 regardless of the second argument.
        self.assertEqual(_yarn_get_mscale(1.0, 0.0), 1.0)
        self.assertEqual(_yarn_get_mscale(1.0, 5.0), 1.0)
        self.assertEqual(_yarn_get_mscale(0.5, 5.0), 1.0)

    def test_mscale_all_dim_zero_collapses_to_one(self):
        # 0.1 * 0.0 * ln(scale) + 1.0 == 1.0 for any scale > 1: the factor
        # vanishes even though the scale branch is taken. This is why the
        # default config (rotary_scaling_factor=40, mscale_all_dim=0.0) still
        # lands on the None arm downstream.
        self.assertEqual(_yarn_get_mscale(40.0, 0.0), 1.0)
        self.assertEqual(_yarn_get_mscale(4.0, 0.0), 1.0)

    def test_scale_gt_one_matches_hand_value(self):
        # 0.1 * 1.0 * ln(4) + 1.0. ln(4) = 1.3862943611198906, so the result
        # is 1.1386294361119891 -- pinned as a literal, derived by hand.
        self.assertAlmostEqual(
            _yarn_get_mscale(4.0, 1.0), 1.1386294361119891, places=12
        )
        # A second point: 0.1 * 0.5 * ln(16) + 1.0 = 0.05 * 2.772588722... + 1.
        self.assertAlmostEqual(
            _yarn_get_mscale(16.0, 0.5), 1.1386294361119891, places=12
        )


class _FakeGroup:
    """World-size-1 process group. ``get_pg_size`` returns 1 whenever
    ``paddle.distributed`` is uninitialised, so these attributes only need to
    satisfy the ``hasattr(tp)/hasattr(cp)`` checks in ``Attention.__init__``.
    """

    def __init__(self, nranks=1):
        self.nranks = nranks
        self.world_size = nranks
        self.ranks = list(range(nranks))
        self.rank = 0


class _FakePGCollection:
    def __init__(self):
        self.tp = _FakeGroup(1)
        self.cp = _FakeGroup(1)


class _CoreStub:
    """Return value for the patched ``build_spec_layer``.

    ``softmax_offset`` must be ``None`` so the production sink guard
    (:557-584) stays on its no-sink branch; a bare ``MagicMock`` would expose a
    truthy auto-attribute there and trip an unrelated FA4/dtype error.
    """

    softmax_offset = None


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_MSG)
class TestMLASoftmaxScalePropagation(unittest.TestCase):
    """Construct a real ``MLASelfAttention`` and capture the ``softmax_scale``
    argument its ``__init__`` hands to the core-attention ``build_spec_layer``.
    """

    Q_HEAD_DIM = 128  # qk_nope_head_dim(64) + qk_rope_head_dim(64)

    def _make_config(self, rotary_scaling_factor, mscale_all_dim):
        # __post_init__ fills head_dim/v_head_dim/num_key_value_heads from these
        # three; the MLA-specific dims are set afterwards, matching the
        # convention in hybrid_mla_utils.py (bypass post-init validation).
        config = TransformerConfig(
            num_hidden_layers=1,
            hidden_size=256,
            num_attention_heads=2,
        )
        config.q_lora_rank = 512
        config.kv_lora_rank = 512
        config.qk_nope_head_dim = 64
        config.qk_rope_head_dim = 64
        config.v_head_dim = 128
        config.rope_type = "yarn"
        config.rotary_scaling_factor = rotary_scaling_factor
        config.mscale = 1.0
        config.mscale_all_dim = mscale_all_dim
        config.softmax_scale = None
        return config

    def _build_and_capture(self, config):
        """Build the module with ``build_spec_layer`` recorded in *both* the
        base ``attention`` and the ``multi_latent_attention`` namespaces, then
        return ``(module, core_softmax_scale)`` for the MLA core build.
        """
        calls = []

        def _recorder(*args, **kwargs):
            calls.append((args, kwargs))
            return _CoreStub()

        with (
            unittest.mock.patch.object(
                mla_mod, "build_spec_layer", new=_recorder
            ),
            unittest.mock.patch.object(
                attention_mod, "build_spec_layer", new=_recorder
            ),
        ):
            module = MLASelfAttention(
                config=config,
                sublayers_spec=MLASelfAttentionSublayersSpec(),
                layer_number=1,
                attn_mask_type=AttnMaskType.causal,
                pg_collection=_FakePGCollection(),
            )

        # The MLA core build (:512-527) is the one carrying k_channels ==
        # q_head_dim and num_key_value_heads == 1; the base build (:339-354)
        # uses the dense head_dim and the config's kv-head count instead.
        core_calls = [
            kwargs
            for _, kwargs in calls
            if kwargs.get("k_channels") == self.Q_HEAD_DIM
            and kwargs.get("num_key_value_heads") == 1
            and "softmax_scale" in kwargs
        ]
        self.assertEqual(
            len(core_calls),
            1,
            "expected exactly one MLA core-attention build carrying "
            f"k_channels={self.Q_HEAD_DIM}; got {len(core_calls)}",
        )
        return module, core_calls[0]["softmax_scale"]

    def test_geometry(self):
        module, _ = self._build_and_capture(self._make_config(1.0, 0.0))
        # Independent: q_head_dim = qk_nope_head_dim + qk_rope_head_dim.
        self.assertEqual(module.q_head_dim, self.Q_HEAD_DIM)

    def test_mscale_one_propagates_none(self):
        # rotary_scaling_factor=1.0 -> _yarn_get_mscale short-circuits to 1.0,
        # so the None arm is taken and the kernel is left to use its own
        # 1/sqrt(d) default.
        module, core_scale = self._build_and_capture(
            self._make_config(1.0, 0.0)
        )
        self.assertIsNone(core_scale)
        self.assertIsNone(module._softmax_scale_arg)
        # The suppressed explicit value still equals the kernel default, which
        # is exactly why production is allowed to pass None here.
        self.assertAlmostEqual(
            module.softmax_scale,
            1.0 / math.sqrt(self.Q_HEAD_DIM),
            places=12,
        )

    def test_mscale_not_one_propagates_scaled_value(self):
        # rotary_scaling_factor=4.0, mscale_all_dim=1.0 -> mscale = 0.1*ln(4)+1.
        module, core_scale = self._build_and_capture(
            self._make_config(4.0, 1.0)
        )
        # Independent reference from the documented formula (not a call into the
        # production line under test): mscale**2 / sqrt(q_head_dim).
        mscale = 0.1 * math.log(4.0) + 1.0
        expected = mscale * mscale / math.sqrt(self.Q_HEAD_DIM)
        self.assertIsNotNone(core_scale)
        self.assertAlmostEqual(core_scale, expected, places=12)
        self.assertAlmostEqual(module.softmax_scale, expected, places=12)
        self.assertEqual(core_scale, module._softmax_scale_arg)
        # And it is genuinely distinct from the plain 1/sqrt(d) default, so a
        # regression that dropped the YaRN factor would change this value.
        self.assertNotAlmostEqual(
            core_scale, 1.0 / math.sqrt(self.Q_HEAD_DIM), places=6
        )


if __name__ == "__main__":
    unittest.main()

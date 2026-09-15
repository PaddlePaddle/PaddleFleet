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

"""Behavior tests for the DeepSeek HF-checkpoint loader name/tensor mapping.

Module under test:
``paddlefleet.cli.train.deepseek_v3_pretrain.utils.load_hf_ckpt``.

This is a Checkpoint / weight-management concern. Before any tensor is copied,
the loader must rewrite each Paddle parameter *name* into the exact Hugging
Face name(s) that own the corresponding weight, and ``prepare_tensor`` must
lay each source tensor into the destination with the correct transpose and the
correct gate/up ownership when a fused projection is split. The load-bearing
contract is *identity and content* (which layer, which expert, which half),
not shape -- a mapping that dropped the layer index, swapped gate/up, or
transposed the wrong axis would still preserve list lengths and shapes. Every
oracle below is hand-derived from the documented HF naming scheme, with layer
and expert indices deliberately distinct so an index swap/drop is rejected.

Import strategy. The real package import is attempted first so a full
environment exercises the exact entry point callers use. On a Paddle-less
no-card host, ``paddlefleet/__init__`` pulls in Paddle and raises ImportError;
we then load the *identical* production source file directly, binding the
*real* ``paddlefleet.utils.log.Logger`` (colorlog + stdlib only, no Paddle) so
the logger's real call signature is exercised faithfully -- this is what the
flagged production bug depends on. The top-level ``import paddle`` in the
module is satisfied by a bare stub only because the name-mapping functions
never touch Paddle; ``prepare_tensor`` performs real tensor ops and therefore
runs only when real Paddle is importable, otherwise those cases skip with an
honest reason. No production code is modified.
"""

import importlib.util
import os
import sys
import types
import unittest

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "src"))
_PROD_PATH = os.path.join(
    _SRC,
    "paddlefleet",
    "cli",
    "train",
    "deepseek_v3_pretrain",
    "utils",
    "load_hf_ckpt.py",
)
_LOG_PATH = os.path.join(_SRC, "paddlefleet", "utils", "log.py")

# Detect *real* Paddle before any stub is installed, so prepare_tensor cases
# can be gated honestly and never run against a stub.
try:
    import paddle as _real_paddle  # noqa: F401

    _HAS_REAL_PADDLE = True
except ImportError:
    _real_paddle = None
    _HAS_REAL_PADDLE = False


def _load_production_module():
    """Return (module, error). Prefer real package import; fall back to file.

    The fallback binds the genuine Logger (needed to observe the logger bug)
    and, only when real Paddle is unavailable, a bare ``paddle`` stub that
    merely satisfies the unused module-level import.
    """
    try:
        from paddlefleet.cli.train.deepseek_v3_pretrain.utils import (
            load_hf_ckpt as mod,
        )

        return mod, None
    except ImportError:
        pass
    except Exception as exc:  # a non-dependency failure must not be hidden
        return None, exc

    try:
        if not _HAS_REAL_PADDLE and "paddle" not in sys.modules:
            sys.modules["paddle"] = types.ModuleType("paddle")
        for pkg in ("paddlefleet", "paddlefleet.utils"):
            if pkg not in sys.modules:
                stub = types.ModuleType(pkg)
                stub.__path__ = []
                sys.modules[pkg] = stub
        if "paddlefleet.utils.log" not in sys.modules:
            log_spec = importlib.util.spec_from_file_location(
                "paddlefleet.utils.log", _LOG_PATH
            )
            log_mod = importlib.util.module_from_spec(log_spec)
            sys.modules["paddlefleet.utils.log"] = log_mod
            log_spec.loader.exec_module(log_mod)
        spec = importlib.util.spec_from_file_location(
            "paddlefleet_load_hf_ckpt_under_test", _PROD_PATH
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod, None
    except ImportError as exc:
        return None, exc


_MOD, _LOAD_ERROR = _load_production_module()


class _Base(unittest.TestCase):
    def setUp(self):
        if _MOD is None:
            self.skipTest(
                "load_hf_ckpt import failed (dependency unavailable): "
                "{!r}".format(_LOAD_ERROR)
            )


class TestGetHfPrefix(_Base):
    """``_get_hf_prefix`` maps (segment_id, id_in_segment) to an HF prefix.

    The four special (segment, id) pairs must win over the arithmetic
    ``segment + id - 1`` fallback; oracles are chosen so the fallback would
    produce a *different* string, proving the special branch is taken.
    """

    def test_normal_pairs_use_segment_plus_id_minus_one(self):
        self.assertEqual(_MOD._get_hf_prefix(4, 3), "model.layers.6")
        self.assertEqual(_MOD._get_hf_prefix(7, 2), "model.layers.8")
        self.assertEqual(_MOD._get_hf_prefix(2, 1), "model.layers.2")

    def test_special_first_block_is_bare_model(self):
        # Fallback would give model.layers.-1; the special case wins.
        self.assertEqual(_MOD._get_hf_prefix(0, 0), "model")

    def test_special_final_norm_is_bare_model(self):
        # Fallback would give model.layers.62; the special case wins.
        self.assertEqual(_MOD._get_hf_prefix(60, 3), "model")

    def test_special_lm_head(self):
        # Fallback would give model.layers.63; the special case wins.
        self.assertEqual(_MOD._get_hf_prefix(60, 4), "lm_head")

    def test_special_layer61(self):
        self.assertEqual(_MOD._get_hf_prefix(60, 2), "model.layers.61")


class TestHandleExpertWeights(_Base):
    """``_handle_expert_weights`` splits w1 into gate/up, w2 into down."""

    def test_w1_splits_into_gate_then_up(self):
        self.assertEqual(
            _MOD._handle_expert_weights(
                "model.layers.6", "mlp.experts.4.w1.weight"
            ),
            [
                "model.layers.6.mlp.experts.4.gate_proj.weight",
                "model.layers.6.mlp.experts.4.up_proj.weight",
            ],
        )

    def test_w2_maps_to_down(self):
        self.assertEqual(
            _MOD._handle_expert_weights(
                "model.layers.6", "mlp.experts.4.w2.weight"
            ),
            ["model.layers.6.mlp.experts.4.down_proj.weight"],
        )

    def test_expert_id_is_integer_normalized(self):
        # Capture group is fed through int(), so a zero-padded id collapses.
        self.assertEqual(
            _MOD._handle_expert_weights(
                "model.layers.2", "mlp.experts.07.w1.weight"
            ),
            [
                "model.layers.2.mlp.experts.7.gate_proj.weight",
                "model.layers.2.mlp.experts.7.up_proj.weight",
            ],
        )

    def test_weight_suffix_is_optional(self):
        self.assertEqual(
            _MOD._handle_expert_weights("model.layers.2", "mlp.experts.4.w1"),
            [
                "model.layers.2.mlp.experts.4.gate_proj.weight",
                "model.layers.2.mlp.experts.4.up_proj.weight",
            ],
        )

    def test_non_expert_returns_none(self):
        self.assertIsNone(
            _MOD._handle_expert_weights(
                "model.layers.6", "self_attn.q_proj.weight"
            )
        )


class TestHandleSharedExpertWeights(_Base):
    """``_handle_shared_expert_weights`` maps the shared-expert MLP."""

    def test_w1_splits_into_gate_then_up(self):
        self.assertEqual(
            _MOD._handle_shared_expert_weights(
                "model.layers.9", "mlp.shared_experts.w1.weight"
            ),
            [
                "model.layers.9.mlp.shared_experts.gate_proj.weight",
                "model.layers.9.mlp.shared_experts.up_proj.weight",
            ],
        )

    def test_w2_maps_to_down(self):
        self.assertEqual(
            _MOD._handle_shared_expert_weights(
                "model.layers.9", "mlp.shared_experts.w2.weight"
            ),
            ["model.layers.9.mlp.shared_experts.down_proj.weight"],
        )

    def test_non_shared_returns_none(self):
        self.assertIsNone(
            _MOD._handle_shared_expert_weights(
                "model.layers.9", "self_attn.q_proj.weight"
            )
        )


class TestHandleMlpWeights(_Base):
    """``_handle_mlp_weights`` matches the *bare* dense keys exactly."""

    def test_w1_splits_into_gate_then_up(self):
        self.assertEqual(
            _MOD._handle_mlp_weights("model.layers.5", "mlp.w1"),
            [
                "model.layers.5.mlp.gate_proj.weight",
                "model.layers.5.mlp.up_proj.weight",
            ],
        )

    def test_w2_maps_to_down(self):
        self.assertEqual(
            _MOD._handle_mlp_weights("model.layers.5", "mlp.w2"),
            ["model.layers.5.mlp.down_proj.weight"],
        )

    def test_requires_exact_bare_key_not_dotted_weight(self):
        # This helper uses exact string equality (not the optional-.weight
        # regex), so "mlp.w1.weight" must NOT match here.
        self.assertIsNone(
            _MOD._handle_mlp_weights("model.layers.5", "mlp.w1.weight")
        )

    def test_non_mlp_returns_none(self):
        self.assertIsNone(
            _MOD._handle_mlp_weights(
                "model.layers.5", "self_attn.q_proj.weight"
            )
        )


class TestPaddleNameToHfNames(_Base):
    """End-to-end pipeline-parallel Paddle name -> HF name(s).

    Segment/id pairs are chosen so the resolved layer index (8 via 4+5-1) is
    distinct from every expert index (2), so a mapping that confused the two
    would fail. Each branch of the dispatcher is covered with an exact oracle.
    """

    def test_embed_tokens_local_shared_alias(self):
        self.assertEqual(
            _MOD.paddle_name_to_hf_names(
                "_layers.local_shared_layers."
                "DeepseekV2_shared_weight.embed_tokens.weight"
            ),
            ["model.embed_tokens.weight"],
        )

    def test_embed_tokens_deepseek_v2_alias(self):
        self.assertEqual(
            _MOD.paddle_name_to_hf_names(
                "_layers.deepseek_v2.embed_tokens.weight"
            ),
            ["model.embed_tokens.weight"],
        )

    def test_attention_passthrough_uses_resolved_layer_index(self):
        self.assertEqual(
            _MOD.paddle_name_to_hf_names("_layers.4.5.self_attn.o_proj.weight"),
            ["model.layers.8.self_attn.o_proj.weight"],
        )

    def test_custom_name_map_input_layernorm(self):
        self.assertEqual(
            _MOD.paddle_name_to_hf_names(
                "_layers.4.5.self_attn.input_layernorm.weight"
            ),
            ["model.layers.8.input_layernorm.weight"],
        )

    def test_custom_name_map_fused_kv_down(self):
        # A non-trivial rename: fused kv_down_weight -> kv_a_proj_with_mqa.
        self.assertEqual(
            _MOD.paddle_name_to_hf_names(
                "_layers.2.1.self_attn.fused_rms_norm_linear.kv_down_weight"
            ),
            ["model.layers.2.self_attn.kv_a_proj_with_mqa.weight"],
        )

    def test_routed_expert_w1(self):
        self.assertEqual(
            _MOD.paddle_name_to_hf_names("_layers.4.5.mlp.experts.2.w1.weight"),
            [
                "model.layers.8.mlp.experts.2.gate_proj.weight",
                "model.layers.8.mlp.experts.2.up_proj.weight",
            ],
        )

    def test_routed_expert_w2(self):
        self.assertEqual(
            _MOD.paddle_name_to_hf_names("_layers.4.5.mlp.experts.2.w2.weight"),
            ["model.layers.8.mlp.experts.2.down_proj.weight"],
        )

    def test_routed_expert_fused_gate_up(self):
        self.assertEqual(
            _MOD.paddle_name_to_hf_names(
                "_layers.4.5.mlp.experts.2.gate_up_fused_proj.weight"
            ),
            [
                "model.layers.8.mlp.experts.2.gate_proj.weight",
                "model.layers.8.mlp.experts.2.up_proj.weight",
            ],
        )

    def test_shared_expert_w1(self):
        self.assertEqual(
            _MOD.paddle_name_to_hf_names(
                "_layers.4.5.mlp.shared_experts.w1.weight"
            ),
            [
                "model.layers.8.mlp.shared_experts.gate_proj.weight",
                "model.layers.8.mlp.shared_experts.up_proj.weight",
            ],
        )

    def test_shared_expert_w2(self):
        self.assertEqual(
            _MOD.paddle_name_to_hf_names(
                "_layers.4.5.mlp.shared_experts.w2.weight"
            ),
            ["model.layers.8.mlp.shared_experts.down_proj.weight"],
        )

    def test_shared_expert_fused_gate_up(self):
        self.assertEqual(
            _MOD.paddle_name_to_hf_names(
                "_layers.4.5.mlp.shared_experts.gate_up_fused_proj.weight"
            ),
            [
                "model.layers.8.mlp.shared_experts.gate_proj.weight",
                "model.layers.8.mlp.shared_experts.up_proj.weight",
            ],
        )

    def test_dense_mlp_w1(self):
        self.assertEqual(
            _MOD.paddle_name_to_hf_names("_layers.4.5.mlp.w1"),
            [
                "model.layers.8.mlp.gate_proj.weight",
                "model.layers.8.mlp.up_proj.weight",
            ],
        )

    def test_dense_mlp_w2(self):
        self.assertEqual(
            _MOD.paddle_name_to_hf_names("_layers.4.5.mlp.w2"),
            ["model.layers.8.mlp.down_proj.weight"],
        )

    def test_dense_mlp_fused_gate_up(self):
        self.assertEqual(
            _MOD.paddle_name_to_hf_names(
                "_layers.4.5.mlp.gate_up_fused_proj.weight"
            ),
            [
                "model.layers.8.mlp.gate_proj.weight",
                "model.layers.8.mlp.up_proj.weight",
            ],
        )

    def test_final_norm_via_special_prefix(self):
        # (60, 3) resolves to bare "model", so norm.weight -> model.norm.weight.
        self.assertEqual(
            _MOD.paddle_name_to_hf_names("_layers.60.3.norm.weight"),
            ["model.norm.weight"],
        )

    def test_lm_head_via_special_prefix(self):
        # (60, 4) resolves to "lm_head".
        self.assertEqual(
            _MOD.paddle_name_to_hf_names("_layers.60.4.weight"),
            ["lm_head.weight"],
        )


class TestPaddleNameToHfNamesDsV2(_Base):
    """DeepSeek-V2 flat naming (``_layers.deepseek_v2.layers.N....``)."""

    def test_embed_tokens(self):
        self.assertEqual(
            _MOD.paddle_name_to_hf_names_ds_v2(
                "_layers.deepseek_v2.embed_tokens.weight"
            ),
            ["model.embed_tokens.weight"],
        )

    def test_norm(self):
        self.assertEqual(
            _MOD.paddle_name_to_hf_names_ds_v2(
                "_layers.deepseek_v2.norm.weight"
            ),
            ["model.norm.weight"],
        )

    def test_lm_head(self):
        self.assertEqual(
            _MOD.paddle_name_to_hf_names_ds_v2("_layers.lm_head.weight"),
            ["lm_head.weight"],
        )

    def test_attention_passthrough_keeps_layer_index(self):
        self.assertEqual(
            _MOD.paddle_name_to_hf_names_ds_v2(
                "_layers.deepseek_v2.layers.7.self_attn.o_proj.weight"
            ),
            ["model.layers.7.self_attn.o_proj.weight"],
        )

    def test_custom_name_map_memory_recompute_q_up(self):
        self.assertEqual(
            _MOD.paddle_name_to_hf_names_ds_v2(
                "_layers.deepseek_v2.layers.7.self_attn."
                "memory_recompute_att.q_up_weight"
            ),
            ["model.layers.7.self_attn.q_b_proj.weight"],
        )

    def test_routed_expert_w1(self):
        self.assertEqual(
            _MOD.paddle_name_to_hf_names_ds_v2(
                "_layers.deepseek_v2.layers.9.mlp.experts.3.w1.weight"
            ),
            [
                "model.layers.9.mlp.experts.3.gate_proj.weight",
                "model.layers.9.mlp.experts.3.up_proj.weight",
            ],
        )

    def test_routed_expert_w2(self):
        self.assertEqual(
            _MOD.paddle_name_to_hf_names_ds_v2(
                "_layers.deepseek_v2.layers.9.mlp.experts.3.w2.weight"
            ),
            ["model.layers.9.mlp.experts.3.down_proj.weight"],
        )

    def test_routed_expert_fused_gate_up(self):
        self.assertEqual(
            _MOD.paddle_name_to_hf_names_ds_v2(
                "_layers.deepseek_v2.layers.9.mlp.experts.3.gate_up_fused_proj.weight"
            ),
            [
                "model.layers.9.mlp.experts.3.gate_proj.weight",
                "model.layers.9.mlp.experts.3.up_proj.weight",
            ],
        )

    def test_shared_expert_w1(self):
        self.assertEqual(
            _MOD.paddle_name_to_hf_names_ds_v2(
                "_layers.deepseek_v2.layers.9.mlp.shared_experts.w1.weight"
            ),
            [
                "model.layers.9.mlp.shared_experts.gate_proj.weight",
                "model.layers.9.mlp.shared_experts.up_proj.weight",
            ],
        )

    def test_shared_expert_w2(self):
        self.assertEqual(
            _MOD.paddle_name_to_hf_names_ds_v2(
                "_layers.deepseek_v2.layers.9.mlp.shared_experts.w2.weight"
            ),
            ["model.layers.9.mlp.shared_experts.down_proj.weight"],
        )

    def test_shared_expert_fused_gate_up(self):
        self.assertEqual(
            _MOD.paddle_name_to_hf_names_ds_v2(
                "_layers.deepseek_v2.layers.9.mlp.shared_experts."
                "gate_up_fused_proj.weight"
            ),
            [
                "model.layers.9.mlp.shared_experts.gate_proj.weight",
                "model.layers.9.mlp.shared_experts.up_proj.weight",
            ],
        )

    def test_dense_mlp_w1(self):
        self.assertEqual(
            _MOD.paddle_name_to_hf_names_ds_v2(
                "_layers.deepseek_v2.layers.9.mlp.w1"
            ),
            [
                "model.layers.9.mlp.gate_proj.weight",
                "model.layers.9.mlp.up_proj.weight",
            ],
        )

    def test_dense_mlp_w2(self):
        self.assertEqual(
            _MOD.paddle_name_to_hf_names_ds_v2(
                "_layers.deepseek_v2.layers.9.mlp.w2"
            ),
            ["model.layers.9.mlp.down_proj.weight"],
        )


class TestRegexCaptureGroups(_Base):
    """The compiled patterns must capture the indices names are built from."""

    def test_layer_re_captures_segment_id_and_rest(self):
        m = _MOD._LAYER_RE.match("_layers.11.7.self_attn.k_proj.weight")
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1), "11")
        self.assertEqual(m.group(2), "7")
        self.assertEqual(m.group(3), "self_attn.k_proj.weight")

    def test_layer_re_rejects_unrelated(self):
        self.assertIsNone(_MOD._LAYER_RE.match("weights.for.something.else"))

    def test_expert_w1_re_captures_id_and_optional_weight(self):
        self.assertEqual(
            _MOD._EXPERT_W1_RE.match("mlp.experts.42.w1.weight").group(1), "42"
        )
        self.assertEqual(
            _MOD._EXPERT_W1_RE.match("mlp.experts.42.w1").group(1), "42"
        )

    def test_expert_w2_re_captures_id(self):
        self.assertEqual(
            _MOD._EXPERT_W2_RE.match("mlp.experts.13.w2.weight").group(1), "13"
        )

    def test_layer_re_v2_captures_layer_and_rest(self):
        m = _MOD._LAYER_RE_v2.match("_layers.deepseek_v2.layers.13.mlp.w2")
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1), "13")
        self.assertEqual(m.group(2), "mlp.w2")


@unittest.skipUnless(
    _HAS_REAL_PADDLE,
    "prepare_tensor performs real tensor ops; Paddle not installed",
)
class TestPrepareTensor(_Base):
    """``prepare_tensor`` decides transpose/split and preserves content.

    Inputs carry distinguishable ``arange`` content so a wrong axis, a missing
    transpose, or a gate/up ownership swap is detected -- shape alone would
    not reveal any of these.
    """

    def test_matching_shape_1d_is_returned_verbatim(self):
        import paddle

        t = paddle.arange(6, dtype="float32")
        out = _MOD.prepare_tensor(t, [6])
        np.testing.assert_array_equal(
            out.numpy(), np.arange(6, dtype="float32")
        )

    def test_matching_shape_2d_is_not_transposed(self):
        import paddle

        base = np.arange(6, dtype="float32").reshape(2, 3)
        t = paddle.to_tensor(base)
        out = _MOD.prepare_tensor(t, [2, 3])
        # Same shape => returned as-is; a spurious transpose would reorder.
        np.testing.assert_array_equal(out.numpy(), base)

    def test_transpose_when_only_transpose_matches_dst(self):
        import paddle

        base = np.arange(6, dtype="float32").reshape(2, 3)
        t = paddle.to_tensor(base)
        out = _MOD.prepare_tensor(t, [3, 2])
        np.testing.assert_array_equal(out.numpy(), base.T)

    def test_force_transpose_overrides_matching_shape(self):
        import paddle

        base = np.arange(6, dtype="float32").reshape(2, 3)
        t = paddle.to_tensor(base)
        # Without force this would be identity; force_transpose must flip it.
        out = _MOD.prepare_tensor(t, [2, 3], force_transpose=True)
        self.assertEqual(list(out.shape), [3, 2])
        np.testing.assert_array_equal(out.numpy(), base.T)

    def test_fused_pair_transposes_each_then_concats_gate_then_up(self):
        import paddle

        gate = np.arange(128, dtype="float32").reshape(16, 8)
        up = np.arange(128, 256, dtype="float32").reshape(16, 8)
        out = _MOD.prepare_tensor(
            [paddle.to_tensor(gate), paddle.to_tensor(up)], [8, 32]
        )
        self.assertEqual(list(out.shape), [8, 32])
        got = out.numpy()
        # Gate half owns columns [0:16], up half owns [16:32]; each is the
        # transpose of its source. A swap would flip these two blocks.
        np.testing.assert_array_equal(got[:, :16], gate.T)
        np.testing.assert_array_equal(got[:, 16:], up.T)


class TestUnmatchedNameLoggerBug(_Base):
    """A genuine production defect (no code modified here).

    When a Paddle name matches none of the branches, both mappers are meant to
    log a warning and return ``[]``. They call
    ``logger.warning("not match here !!", paddle_name)`` with two positional
    arguments, but ``paddlefleet.utils.log.Logger.__call__`` accepts only
    ``(log_level, msg)``. The extra argument raises ``TypeError`` before the
    intended ``return []`` is reached, so an unmatched parameter crashes the
    checkpoint load instead of being reported and skipped. These tests assert
    the intended contract and are expected to fail until the logging call is
    fixed in production.
    """

    @unittest.expectedFailure
    def test_unmatched_name_returns_empty_list(self):
        self.assertEqual(
            _MOD.paddle_name_to_hf_names("random.unmatched.name"), []
        )

    @unittest.expectedFailure
    def test_unmatched_name_ds_v2_returns_empty_list(self):
        self.assertEqual(
            _MOD.paddle_name_to_hf_names_ds_v2("totally.unrelated.name"), []
        )


if __name__ == "__main__":
    unittest.main()

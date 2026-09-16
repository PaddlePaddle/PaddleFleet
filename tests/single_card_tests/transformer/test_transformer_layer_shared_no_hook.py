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

"""Behavior tests for the MTP-shared-last-layer "no-hook" parameter coloring.

Three real production entry points are exercised directly (no re-implemented
formulas, no patched methods):

  * ``is_mtp_shared_last_layer`` -- the predicate both coloring sites consult.
  * ``MoELayer._color_expert_params`` -- the single site that colors routed
    expert params (moe_weight_no_hook on the shared last layer, else moe_expert).
  * ``TransformerLayer._mark_shared_no_hook_params`` -- the site that colors the
    remaining uncolored dense params with dense_weight_no_hook.

The two methods only read plain data attributes off ``self`` and mutate each
param's ``color``. They are invoked unbound against a minimal stub ``self`` so
the genuine branching / skip / assignment logic runs; only the not-under-test
container objects (experts, param list) are lightweight doubles carrying just
the ``color`` attribute production touches. ``is_mtp_shared_last_layer`` is NOT
patched -- the real predicate runs inside both methods.

These are CPU-only, world-size-independent checks of the local coloring and
config-branch logic. They do NOT execute or verify any sharding-stage1
communication / overlap-hook behavior, which requires a real process group.
"""

import types
import unittest

try:
    from paddlefleet.transformer.moe.moe_layer import MoELayer
    from paddlefleet.transformer.transformer_layer import (
        TransformerLayer,
        is_mtp_shared_last_layer,
    )

    HAS_PADDLE = True
    IMPORT_ERROR = ""
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
    HAS_PADDLE = False
    IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

_SKIP_REASON = (
    "cannot import paddlefleet.transformer (CPU-only run without paddle): "
    + IMPORT_ERROR
)


class _FakeParam:
    """Minimal parameter stand-in carrying only the mutable ``color`` attribute
    that the coloring methods read (via getattr) and write."""

    def __init__(self, color=-1, name="p"):
        self.color = color
        self.name = name


def _make_config(**overrides):
    """Config double exposing exactly the attributes the predicate reads.

    ``is_mtp_shared_last_layer`` accesses everything through ``getattr`` with
    defaults, so a namespace is a faithful stand-in; the real predicate logic
    still runs against it.
    """
    cfg = {
        "mtp_shared_last_layer": True,
        "stage1_overlap": True,
        "num_nextn_predict_layers": 1,
        "num_hidden_layers": 2,
        "num_empty_layers_add_in_head": 0,
    }
    cfg.update(overrides)
    return types.SimpleNamespace(**cfg)


def _make_moe_stub(
    config,
    layer_number,
    is_mtp_layer,
    params,
    ep_size=2,
    fusion=True,
    grad_group=None,
):
    """Build a stub ``self`` for the unbound ``MoELayer._color_expert_params``.

    The stub carries only the attributes that method reads. Expert params are
    exposed through the fusion (grouped_gemm_experts) or non-fusion (experts)
    container exactly as production selects them.
    """
    experts = types.SimpleNamespace(parameters=lambda: list(params))
    ns = types.SimpleNamespace(
        config=config,
        layer_number=layer_number,
        is_mtp_layer=is_mtp_layer,
        expert_model_parallel_size=ep_size,
        moe_grad_group=grad_group,
    )
    if fusion:
        ns.grouped_gemm_experts = experts
    else:
        # No grouped_gemm_experts attribute -> production falls back to experts.
        ns.experts = experts
    return ns


def _make_layer_stub(config, layer_number, is_mtp_layer, params):
    """Build a stub ``self`` for ``TransformerLayer._mark_shared_no_hook_params``.

    ``params`` is captured by reference so a test can append late-created params
    (mirroring HyperConnectionTransformerLayer, whose submodules are built after
    the base ``_mark_shared_no_hook_params`` call) and re-invoke the method.
    """
    return types.SimpleNamespace(
        config=config,
        layer_number=layer_number,
        is_mtp_layer=is_mtp_layer,
        parameters=lambda: list(params),
    )


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestIsMtpSharedLastLayer(unittest.TestCase):
    """Predicate gating both coloring sites. Each guard is exercised in a
    region where the wrong answer is observable, and the last-layer boundary is
    checked against a hand-derived ``num_hidden_layers - 1 + head_offset``."""

    def test_sharing_disabled_returns_false(self):
        cfg = _make_config(mtp_shared_last_layer=False)
        # last layer number would be 1, but sharing off short-circuits first.
        self.assertFalse(is_mtp_shared_last_layer(cfg, 1, False))

    def test_stage1_overlap_disabled_returns_false(self):
        cfg = _make_config(stage1_overlap=False)
        self.assertFalse(is_mtp_shared_last_layer(cfg, 1, False))

    def test_stage1_overlap_missing_attr_returns_false(self):
        cfg = _make_config()
        del cfg.stage1_overlap  # getattr default False
        self.assertFalse(is_mtp_shared_last_layer(cfg, 1, False))

    def test_no_mtp_layers_returns_false(self):
        cfg = _make_config(num_nextn_predict_layers=0)
        self.assertFalse(is_mtp_shared_last_layer(cfg, 1, False))

    def test_mtp_layer_itself_returns_false(self):
        # Even on the last layer number, the MTP layer's params are aliases.
        self.assertFalse(is_mtp_shared_last_layer(_make_config(), 1, True))

    def test_boundary_default_offset(self):
        # H=2, E=0 -> last_layer_number = 2 - 1 + 0 = 1.
        cfg = _make_config(num_hidden_layers=2, num_empty_layers_add_in_head=0)
        self.assertTrue(is_mtp_shared_last_layer(cfg, 1, False))
        self.assertFalse(is_mtp_shared_last_layer(cfg, 0, False))
        self.assertFalse(is_mtp_shared_last_layer(cfg, 2, False))

    def test_boundary_with_head_offset(self):
        # H=2, E=3 -> last_layer_number = 2 - 1 + 3 = 4. The offset must move
        # the boundary; a naive H-1=1 would wrongly fire here.
        cfg = _make_config(num_hidden_layers=2, num_empty_layers_add_in_head=3)
        self.assertTrue(is_mtp_shared_last_layer(cfg, 4, False))
        self.assertFalse(is_mtp_shared_last_layer(cfg, 1, False))
        self.assertFalse(is_mtp_shared_last_layer(cfg, 3, False))
        self.assertFalse(is_mtp_shared_last_layer(cfg, 5, False))

    def test_boundary_deeper_stack(self):
        # H=5, E=0 -> last_layer_number = 4.
        cfg = _make_config(num_hidden_layers=5, num_empty_layers_add_in_head=0)
        self.assertTrue(is_mtp_shared_last_layer(cfg, 4, False))
        self.assertFalse(is_mtp_shared_last_layer(cfg, 3, False))


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestColorExpertParams(unittest.TestCase):
    """``MoELayer._color_expert_params``: the single site that colors routed
    expert params. Assertions pin the full color dict (key AND the exact
    moe_grad_group object threaded through), not just presence."""

    def test_shared_last_layer_uses_no_hook_with_grad_group(self):
        grp = object()  # sentinel to prove the real grad group is threaded in
        p = _FakeParam(-1)
        stub = _make_moe_stub(_make_config(), 1, False, [p], grad_group=grp)
        MoELayer._color_expert_params(stub)
        self.assertEqual(p.color["color"], "moe_weight_no_hook")
        self.assertIs(p.color["group"], grp)
        self.assertEqual(set(p.color), {"color", "group"})

    def test_non_last_layer_uses_moe_expert(self):
        grp = object()
        p = _FakeParam(-1)
        # layer 0 with default H=2,E=0 (last=1) -> not the shared last layer.
        stub = _make_moe_stub(_make_config(), 0, False, [p], grad_group=grp)
        MoELayer._color_expert_params(stub)
        self.assertEqual(p.color, {"color": "moe_expert", "group": grp})

    def test_sharing_off_uses_moe_expert(self):
        grp = object()
        p = _FakeParam(-1)
        cfg = _make_config(mtp_shared_last_layer=False)
        stub = _make_moe_stub(cfg, 1, False, [p], grad_group=grp)
        MoELayer._color_expert_params(stub)
        self.assertEqual(p.color, {"color": "moe_expert", "group": grp})

    def test_stage1_overlap_off_uses_moe_expert(self):
        grp = object()
        p = _FakeParam(-1)
        cfg = _make_config(stage1_overlap=False)
        stub = _make_moe_stub(cfg, 1, False, [p], grad_group=grp)
        MoELayer._color_expert_params(stub)
        self.assertEqual(p.color, {"color": "moe_expert", "group": grp})

    def test_non_fusion_branch_reads_experts(self):
        # Without grouped_gemm_experts, production must color the params
        # exposed by self.experts instead.
        grp = object()
        p = _FakeParam(-1)
        stub = _make_moe_stub(
            _make_config(), 1, False, [p], fusion=False, grad_group=grp
        )
        MoELayer._color_expert_params(stub)
        self.assertEqual(p.color, {"color": "moe_weight_no_hook", "group": grp})

    def test_ep_size_one_is_noop(self):
        # No expert parallelism -> params stay untouched (sentinel -1).
        p = _FakeParam(-1)
        stub = _make_moe_stub(_make_config(), 1, False, [p], ep_size=1)
        MoELayer._color_expert_params(stub)
        self.assertEqual(p.color, -1)

    def test_already_colored_param_is_left_untouched(self):
        # Paddle forbids reassigning a non-None color: an already-colored param
        # must be skipped even on the shared last layer.
        preset_group = object()
        preset = {"color": "moe_expert", "group": preset_group}
        p = _FakeParam(preset)
        stub = _make_moe_stub(
            _make_config(), 1, False, [p], grad_group=object()
        )
        MoELayer._color_expert_params(stub)
        self.assertIs(p.color, preset)
        self.assertEqual(p.color["color"], "moe_expert")
        self.assertIs(p.color["group"], preset_group)

    def test_only_uncolored_params_recolored_in_mixed_list(self):
        grp = object()
        fresh = _FakeParam(-1, name="fresh")
        preset_group = object()
        colored = _FakeParam(
            {"color": "moe_expert", "group": preset_group}, name="colored"
        )
        stub = _make_moe_stub(
            _make_config(), 1, False, [fresh, colored], grad_group=grp
        )
        MoELayer._color_expert_params(stub)
        self.assertEqual(
            fresh.color, {"color": "moe_weight_no_hook", "group": grp}
        )
        self.assertIs(colored.color["group"], preset_group)
        self.assertEqual(colored.color["color"], "moe_expert")


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestMarkSharedNoHookDense(unittest.TestCase):
    """``TransformerLayer._mark_shared_no_hook_params``: colors only the
    uncolored dense params with dense_weight_no_hook (no group), skipping any
    param already colored (dict or non-sentinel), and no-ops off the shared
    last layer."""

    def test_not_shared_last_layer_is_noop(self):
        p = _FakeParam(-1)
        stub = _make_layer_stub(_make_config(), 0, False, [p])
        TransformerLayer._mark_shared_no_hook_params(stub)
        self.assertEqual(p.color, -1)

    def test_stage1_overlap_off_is_noop(self):
        p = _FakeParam(-1)
        cfg = _make_config(stage1_overlap=False)
        stub = _make_layer_stub(cfg, 1, False, [p])
        TransformerLayer._mark_shared_no_hook_params(stub)
        self.assertEqual(p.color, -1)

    def test_uncolored_sentinel_param_gets_dense_no_hook(self):
        p = _FakeParam(-1)
        stub = _make_layer_stub(_make_config(), 1, False, [p])
        TransformerLayer._mark_shared_no_hook_params(stub)
        self.assertEqual(p.color, {"color": "dense_weight_no_hook"})

    def test_missing_color_attr_param_gets_dense_no_hook(self):
        p = _FakeParam(-1)
        del p.color  # getattr(p, "color", None) -> None -> still colored
        stub = _make_layer_stub(_make_config(), 1, False, [p])
        TransformerLayer._mark_shared_no_hook_params(stub)
        self.assertEqual(p.color, {"color": "dense_weight_no_hook"})

    def test_moe_colored_param_is_skipped(self):
        preset_group = object()
        moe = _FakeParam({"color": "moe_weight_no_hook", "group": preset_group})
        stub = _make_layer_stub(_make_config(), 1, False, [moe])
        TransformerLayer._mark_shared_no_hook_params(stub)
        self.assertEqual(moe.color["color"], "moe_weight_no_hook")
        self.assertIs(moe.color["group"], preset_group)

    def test_mixed_dense_and_moe_only_dense_colored(self):
        dense = _FakeParam(-1, name="dense")
        preset_group = object()
        moe = _FakeParam(
            {"color": "moe_expert", "group": preset_group}, name="moe"
        )
        stub = _make_layer_stub(_make_config(), 1, False, [dense, moe])
        TransformerLayer._mark_shared_no_hook_params(stub)
        self.assertEqual(dense.color, {"color": "dense_weight_no_hook"})
        self.assertEqual(moe.color["color"], "moe_expert")
        self.assertIs(moe.color["group"], preset_group)

    def test_second_call_colors_late_created_params_idempotently(self):
        # Mirrors HyperConnectionTransformerLayer: the base call colors the
        # params present at super().__init__(); a later call must color the
        # newly created hyper-connection params while leaving base params as-is
        # (Paddle forbids reassigning color).
        dense = _FakeParam(-1, name="dense")
        preset_group = object()
        moe = _FakeParam(
            {"color": "moe_weight_no_hook", "group": preset_group}, name="moe"
        )
        params = [dense, moe]
        stub = _make_layer_stub(_make_config(), 1, False, params)

        TransformerLayer._mark_shared_no_hook_params(stub)
        self.assertEqual(dense.color, {"color": "dense_weight_no_hook"})

        # Late-created hyper-connection submodule params appear now.
        late = [
            _FakeParam(-1, name="mapping_proj.weight"),
            _FakeParam(-1, name="alpha_res"),
            _FakeParam(-1, name="bias"),
        ]
        params.extend(late)
        TransformerLayer._mark_shared_no_hook_params(stub)

        for p in late:
            self.assertEqual(
                p.color, {"color": "dense_weight_no_hook"}, msg=p.name
            )
        # Base params untouched by the second pass.
        self.assertEqual(dense.color, {"color": "dense_weight_no_hook"})
        self.assertEqual(moe.color["color"], "moe_weight_no_hook")
        self.assertIs(moe.color["group"], preset_group)

    def test_second_call_noop_when_not_shared_last_layer(self):
        dense = _FakeParam(-1)
        params = [dense]
        stub = _make_layer_stub(_make_config(), 0, False, params)
        TransformerLayer._mark_shared_no_hook_params(stub)
        late = _FakeParam(-1, name="mapping_proj.weight")
        params.append(late)
        TransformerLayer._mark_shared_no_hook_params(stub)
        self.assertEqual(dense.color, -1)
        self.assertEqual(late.color, -1)


if __name__ == "__main__":
    unittest.main()

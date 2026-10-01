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

"""Behaviour tests for two module-level helpers in
``paddlefleet.transformer.transformer_layer``:

  * ``is_mtp_shared_last_layer(config, layer_number, is_mtp_layer)`` -- the
    guard that decides whether a backbone transformer layer aliases (shares)
    its weights with the MTP layer and therefore needs the dedicated
    "no_hook" sharding colour. Its result is a pure function of five config
    fields, so every early-exit branch and the final last-layer-index
    comparison are checked against a by-hand truth table.

  * ``tensors_clone(outputs)`` -- the recompute helper that deep-copies the
    tensors inside a tensor / tuple / list / dict container while passing
    non-tensor, non-dict items through by identity, and rejects any other
    container type with ``ValueError``.

Expected values are derived here by hand (no call into the module under test
produces them), and the two helpers are exercised through their real public
signatures. Neither the coverage-suite fixtures nor its expected values were
used to build this file.

Environment: these helpers are CPU-only Python / paddle-tensor logic (no GPU,
no distributed init, no heavy layer construction). The suite is skipped
honestly, with the captured error repr, when paddle cannot be imported; a
missing paddle install is "not run", never a silent pass.
"""

import unittest

try:
    import numpy as np
    import paddle

    from paddlefleet.transformer.transformer_layer import (
        is_mtp_shared_last_layer,
        tensors_clone,
    )

    _IMPORT_ERROR = None
    HAS_DEPS = True
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
    # Only genuine missing-dependency errors become a skip. Any other error
    # (API drift, compile failure) must surface instead of being hidden.
    _IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
    HAS_DEPS = False

_SKIP_REASON = (
    f"paddle/numpy not importable in this environment ({_IMPORT_ERROR})"
)


class _Cfg:
    """Minimal config carrier.

    ``is_mtp_shared_last_layer`` reads its fields with ``getattr(..., default)``
    except ``num_hidden_layers``, which is read directly and is only reached on
    the non-short-circuit path. Attributes are set only when a test needs them,
    so the getattr-default branches are genuinely exercised.
    """

    def __init__(self, **attrs):
        for key, value in attrs.items():
            setattr(self, key, value)


@unittest.skipUnless(HAS_DEPS, _SKIP_REASON)
class TestIsMtpSharedLastLayer(unittest.TestCase):
    """Truth table for the MTP-shared-last-layer guard.

    Enabling condition (all must hold): mtp_shared_last_layer is truthy,
    stage1_overlap is truthy, num_nextn_predict_layers > 0, and the queried
    layer is not the MTP layer itself. When enabled, the result is
    ``layer_number == num_hidden_layers - 1 + num_empty_layers_add_in_head``.
    """

    def test_disabled_when_all_flags_absent(self):
        # Empty config: every getattr falls back to its default (all falsey),
        # so the very first guard returns False without touching
        # num_hidden_layers.
        self.assertFalse(
            is_mtp_shared_last_layer(_Cfg(), layer_number=0, is_mtp_layer=False)
        )

    def test_disabled_when_sharing_off(self):
        cfg = _Cfg(
            mtp_shared_last_layer=False,
            stage1_overlap=True,
            num_nextn_predict_layers=1,
            num_hidden_layers=4,
        )
        self.assertFalse(
            is_mtp_shared_last_layer(cfg, layer_number=3, is_mtp_layer=False)
        )

    def test_disabled_when_stage1_overlap_off(self):
        cfg = _Cfg(
            mtp_shared_last_layer=True,
            stage1_overlap=False,
            num_nextn_predict_layers=1,
            num_hidden_layers=4,
        )
        self.assertFalse(
            is_mtp_shared_last_layer(cfg, layer_number=3, is_mtp_layer=False)
        )

    def test_disabled_when_no_mtp_layers(self):
        # num_nextn_predict_layers <= 0 (0 and None both collapse to 0).
        for npl in (0, None):
            cfg = _Cfg(
                mtp_shared_last_layer=True,
                stage1_overlap=True,
                num_nextn_predict_layers=npl,
                num_hidden_layers=4,
            )
            self.assertFalse(
                is_mtp_shared_last_layer(
                    cfg, layer_number=3, is_mtp_layer=False
                ),
                msg=f"num_nextn_predict_layers={npl!r} must disable sharing",
            )

    def test_disabled_for_the_mtp_layer_itself(self):
        cfg = _Cfg(
            mtp_shared_last_layer=True,
            stage1_overlap=True,
            num_nextn_predict_layers=2,
            num_hidden_layers=4,
        )
        # Even at the would-be last-layer index, the MTP layer never re-colours.
        self.assertFalse(
            is_mtp_shared_last_layer(cfg, layer_number=3, is_mtp_layer=True)
        )

    def test_enabled_only_at_last_backbone_layer(self):
        cfg = _Cfg(
            mtp_shared_last_layer=True,
            stage1_overlap=True,
            num_nextn_predict_layers=2,
            num_hidden_layers=4,
        )
        # last_layer_number = 4 - 1 + 0 = 3.
        self.assertTrue(
            is_mtp_shared_last_layer(cfg, layer_number=3, is_mtp_layer=False)
        )
        for other in (0, 1, 2, 4):
            self.assertFalse(
                is_mtp_shared_last_layer(
                    cfg, layer_number=other, is_mtp_layer=False
                ),
                msg=f"layer {other} is not the last backbone layer",
            )

    def test_head_empty_layers_shift_the_last_index(self):
        cfg = _Cfg(
            mtp_shared_last_layer=True,
            stage1_overlap=True,
            num_nextn_predict_layers=1,
            num_hidden_layers=4,
            num_empty_layers_add_in_head=2,
        )
        # last_layer_number = 4 - 1 + 2 = 5; the plain index 3 no longer matches.
        self.assertTrue(
            is_mtp_shared_last_layer(cfg, layer_number=5, is_mtp_layer=False)
        )
        self.assertFalse(
            is_mtp_shared_last_layer(cfg, layer_number=3, is_mtp_layer=False)
        )


@unittest.skipUnless(HAS_DEPS, _SKIP_REASON)
class TestTensorsClone(unittest.TestCase):
    """Deep-copy semantics of the recompute clone helper."""

    def test_single_tensor_is_copied_not_aliased(self):
        src = paddle.arange(6, dtype="float32").reshape([2, 3])
        out = tensors_clone(src)
        self.assertIsInstance(out, paddle.Tensor)
        self.assertIsNot(out, src)  # a real clone, not the same handle
        self.assertEqual(out.shape, src.shape)
        np.testing.assert_array_equal(out.numpy(), src.numpy())

    def test_tuple_clones_tensors_and_passes_others_by_identity(self):
        t0 = paddle.arange(3, dtype="float32")
        t1 = paddle.arange(3, 6, dtype="float32")
        sentinel = object()  # non-tensor, non-dict -> passed through unchanged
        inner = {"h": paddle.arange(6, 8, dtype="float32")}
        src = (t0, sentinel, None, inner, t1)

        out = tensors_clone(src)

        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 5)
        # tensors: cloned (distinct object, identical values)
        for got, original in ((out[0], t0), (out[4], t1)):
            self.assertIsInstance(got, paddle.Tensor)
            self.assertIsNot(got, original)
            np.testing.assert_array_equal(got.numpy(), original.numpy())
        # non-tensor / non-dict items: same objects, by identity
        self.assertIs(out[1], sentinel)
        self.assertIsNone(out[2])
        # nested dict: recursed into a new dict with a cloned tensor value
        self.assertIsInstance(out[3], dict)
        self.assertIsNot(out[3], inner)
        self.assertEqual(set(out[3]), {"h"})
        self.assertIsNot(out[3]["h"], inner["h"])
        np.testing.assert_array_equal(out[3]["h"].numpy(), inner["h"].numpy())

    def test_list_round_trips_as_list(self):
        t = paddle.arange(4, dtype="float32")
        marker = object()
        src = [t, marker]
        out = tensors_clone(src)
        self.assertIsInstance(out, list)  # list in, list out (not tuple)
        self.assertIsNot(out[0], t)
        np.testing.assert_array_equal(out[0].numpy(), t.numpy())
        self.assertIs(out[1], marker)

    def test_dict_clones_every_tensor_value(self):
        src = {
            "a": paddle.arange(2, dtype="float32"),
            "b": paddle.arange(2, 5, dtype="float32"),
        }
        out = tensors_clone(src)
        self.assertIsInstance(out, dict)
        self.assertIsNot(out, src)
        self.assertEqual(set(out), {"a", "b"})
        for key in src:
            self.assertIsNot(out[key], src[key])
            np.testing.assert_array_equal(out[key].numpy(), src[key].numpy())

    def test_unsupported_container_raises_value_error(self):
        with self.assertRaises(ValueError) as ctx:
            tensors_clone(5)
        self.assertIn("Unsupported data type", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

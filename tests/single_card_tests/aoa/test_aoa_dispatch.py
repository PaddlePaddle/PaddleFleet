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
#
# Scope: consumer-side AOA dispatch (``resolve_aoa_config`` /
# ``resolve_inv_aoa_config``). Pins the three discriminants -- hand-written
# ``_gen_(inv_)aoa_config``, modular ``gen_(inv_)aoa_statements`` *override*
# (not mere presence, which every ``nn.Layer`` has), and the no-match error --
# plus the hand-written-over-modular priority and the derive-from-forward
# fallback's shallow copy (it must not mutate the caller's dict).
import os
import sys
import unittest

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)

from paddle.nn import Layer

from paddlefleet.models.gpt.aoa_dispatch import (
    resolve_aoa_config,
    resolve_inv_aoa_config,
)


class _HandWritten:
    """A model that hand-writes its AOA config (not necessarily a Layer)."""

    def __init__(self, fwd=None, inv=None):
        self._fwd = fwd
        self._inv = inv
        if fwd is not None:
            self._gen_aoa_config = self._call_fwd
        if inv is not None:
            self._gen_inv_aoa_config = self._call_inv

    def _call_fwd(self, config=None):
        return self._fwd(config)

    def _call_inv(self, config=None):
        return self._inv(config)


class _Modular(Layer):
    """A modular model: overrides the whole-model recursion entry points."""

    def gen_aoa_statements(self, config=None):
        return {"aoa_statements": ["modular_fwd", config]}

    def gen_inv_aoa_statements(self, config=None):
        return {"aoa_statements": ["modular_inv", config]}


class _ModularHandWritten(_Modular):
    """Both worlds: a modular override *and* a bolted-on hand-written config
    (mirrors kimi_k3 attaching ``_gen_(inv_)aoa_config`` onto its GPTModel)."""

    def __init__(self):
        super().__init__()
        self._gen_aoa_config = lambda config=None: {"src": "handwritten_fwd"}
        self._gen_inv_aoa_config = lambda config=None: {
            "src": "handwritten_inv"
        }


class TestResolveAoaConfig(unittest.TestCase):
    def test_handwritten_forward(self):
        model = _HandWritten(fwd=lambda c: {"src": "hw", "cfg": c})
        self.assertEqual(
            resolve_aoa_config(model, "CFG"), {"src": "hw", "cfg": "CFG"}
        )

    def test_modular_override_forward(self):
        self.assertEqual(
            resolve_aoa_config(_Modular(), "CFG"),
            {"aoa_statements": ["modular_fwd", "CFG"]},
        )

    def test_bare_layer_is_not_modular(self):
        # ``gen_aoa_statements`` exists on every nn.Layer, so a bare Layer that
        # neither hand-writes nor overrides must raise -- not be treated modular.
        with self.assertRaises(RuntimeError):
            resolve_aoa_config(Layer())

    def test_handwritten_beats_modular(self):
        self.assertEqual(
            resolve_aoa_config(_ModularHandWritten()),
            {"src": "handwritten_fwd"},
        )


class TestResolveInvAoaConfig(unittest.TestCase):
    def test_handwritten_inverse(self):
        model = _HandWritten(
            fwd=lambda c: {"src": "fwd"},
            inv=lambda c: {"src": "inv", "cfg": c},
        )
        self.assertEqual(
            resolve_inv_aoa_config(model, "CFG"), {"src": "inv", "cfg": "CFG"}
        )

    def test_modular_override_inverse(self):
        self.assertEqual(
            resolve_inv_aoa_config(_Modular(), "CFG"),
            {"aoa_statements": ["modular_inv", "CFG"]},
        )

    def test_bare_layer_is_not_modular(self):
        with self.assertRaises(RuntimeError):
            resolve_inv_aoa_config(Layer())

    def test_handwritten_inverse_beats_modular(self):
        self.assertEqual(
            resolve_inv_aoa_config(_ModularHandWritten()),
            {"src": "handwritten_inv"},
        )

    def test_derive_inverse_from_forward(self):
        model = _HandWritten(fwd=lambda c: {"aoa_statements": ["s"]})
        result = resolve_inv_aoa_config(model)
        self.assertEqual(
            result, {"aoa_statements": ["s"], "aoa_config_reverse": True}
        )

    def test_derive_prefers_forward_over_modular(self):
        # A hand-written forward (no inverse) that is *also* a modular subclass
        # derives from forward rather than falling through to the modular entry.
        model = _ModularHandWritten()
        del model._gen_inv_aoa_config
        result = resolve_inv_aoa_config(model)
        self.assertEqual(
            result, {"src": "handwritten_fwd", "aoa_config_reverse": True}
        )

    def test_derive_does_not_mutate_shared_forward_dict(self):
        # ernie5/pretrain.py hands back the *same* closure-captured dict each
        # call; an in-place reverse flag would poison later forward loads.
        shared = {"aoa_statements": ["s"]}
        model = _HandWritten(fwd=lambda c: shared)

        first = resolve_inv_aoa_config(model)
        second = resolve_inv_aoa_config(model)

        self.assertNotIn("aoa_config_reverse", shared)
        self.assertIsNot(first, shared)
        self.assertIsNot(second, shared)
        self.assertTrue(first["aoa_config_reverse"])
        self.assertTrue(second["aoa_config_reverse"])


if __name__ == "__main__":
    unittest.main()

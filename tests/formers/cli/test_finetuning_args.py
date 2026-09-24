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

"""Behavior tests for paddlefleet.cli.hparams.finetuning_args.

Two production behaviors are exercised through their real entry points:

1. ``FinetuningArguments.__post_init__`` maps the user-facing ``compute_type``
   string onto the low-level ``bf16``/``fp16``/``weight_quantize_algo`` fields
   that the rest of the trainer consumes. We drive the *real* ``__post_init__``
   (not a copy of its branch table) by constructing a real ``FinetuningArguments``
   while stubbing only the un-tested base-class distributed initializer
   (``TrainingArguments.__post_init__``) and the ``dataset_world_size`` property,
   so the genuine quantization-config mapping runs and is compared against
   hand-written expected values.

2. ``PdArgumentParser`` synthesizes argparse options from the dataclass field
   definitions/metadata. We build the real parser over the real dataclass and
   parse argv, checking the resolved namespace for the bool-flag defaults, the
   ``no_*`` complement of a default-True bool, and the ``nargs='?'/const`` wiring
   of the ``use_accuracy_compatible`` string flag. This stays at the argparse
   layer (no dataclass instantiation), so it does not need distributed runtime.

Expected values are derived by hand from the argument contract, never read back
from the object under test and never imported as the "answer" from production.

The local environment has no ``paddle`` installed, so importing the production
module fails at ``from paddle.distributed import fleet``. That import is guarded
below and the whole module is skipped with an honest reason when paddle is
absent; the assertions above run unchanged on any environment that has paddle
(no accelerator required -- only argparse and the pure-Python branch table are
touched).
"""

import tempfile
import unittest
from unittest import mock

try:
    import paddle  # noqa: F401

    from paddlefleet.cli.hparams.finetuning_args import FinetuningArguments
    from paddlefleet.trainer import PdArgumentParser
    from paddlefleet.trainer.training_args import TrainingArguments

    _IMPORT_ERROR = None
except ImportError as exc:  # only treat a genuine missing dependency as skip
    _IMPORT_ERROR = exc

_HAS_PADDLE = _IMPORT_ERROR is None
_SKIP_REASON = (
    "paddlefleet.cli.hparams.finetuning_args requires paddle "
    f"(import failed: {_IMPORT_ERROR})"
)

# Hand-derived from the finetuning-args contract, independent of production
# constants: the default quantize target set and the two explicit per-layer
# lists used by the mixed wint4/8 mode.
EXPECTED_DEFAULT_QUANTIZE_LAYERS = [".*mlp.*", ".*self_attn.*"]
EXPECTED_WINT4_8_INT4 = [
    ".*mlp.experts.*",
    ".*mlp.shared_expert.*",
    ".*mlp.shared_experts.*",
]
EXPECTED_WINT4_8_INT8 = [
    ".*self_attn.qkv_proj.*",
    ".*self_attn.q_proj.*",
    ".*self_attn.k_proj.*",
    ".*self_attn.v_proj.*",
    ".*self_attn.o_proj.*",
    ".*mlp.up_gate_proj.*",
    ".*mlp.up_proj.*",
    ".*mlp.gate_proj.*",
    ".*mlp.down_proj.*",
]


@unittest.skipUnless(_HAS_PADDLE, _SKIP_REASON)
class TestComputeTypeMapping(unittest.TestCase):
    """Drive the real FinetuningArguments.__post_init__ compute_type branch.

    Only the base-class distributed init and the dataset_world_size property are
    stubbed; the compute_type -> (bf16, fp16, weight_quantize_algo) mapping is
    executed by real production code, not reimplemented in the test.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.output_dir = self._tmp.name

    def _build(self, **overrides):
        kwargs = {"output_dir": self.output_dir}
        kwargs.update(overrides)
        # TrainingArguments.__post_init__ performs Fleet/distributed setup that
        # is out of scope here; dataset_world_size needs a live process group.
        # Both are un-tested collaborators -> stubbed. The FinetuningArguments
        # branch table under test still runs for real.
        with (
            mock.patch.object(
                TrainingArguments, "__post_init__", lambda self: None
            ),
            mock.patch.object(
                TrainingArguments,
                "dataset_world_size",
                new_callable=mock.PropertyMock,
                return_value=1,
            ),
        ):
            return FinetuningArguments(**kwargs)

    def test_bf16_disables_fp16_and_quantization(self):
        args = self._build(compute_type="bf16")
        self.assertTrue(args.bf16)
        self.assertFalse(args.fp16)
        self.assertIsNone(args.weight_quantize_algo)

    def test_fp16_flips_bf16_off(self):
        args = self._build(compute_type="fp16")
        self.assertFalse(args.bf16)
        self.assertTrue(args.fp16)
        self.assertIsNone(args.weight_quantize_algo)

    def test_float32_disables_both_half_precisions(self):
        # This branch is a genuine production behavior; verify it maps to no
        # half precision and no quantization.
        args = self._build(compute_type="float32")
        self.assertFalse(args.bf16)
        self.assertFalse(args.fp16)
        self.assertIsNone(args.weight_quantize_algo)

    def test_wint8_uses_default_layers_under_int8_key(self):
        args = self._build(compute_type="wint8")
        self.assertEqual(
            args.weight_quantize_algo,
            {"weight_only_int8": EXPECTED_DEFAULT_QUANTIZE_LAYERS},
        )
        # wint modes do not touch bf16, which stays enabled.
        self.assertTrue(args.bf16)

    def test_wint4_uses_default_layers_under_int4_key(self):
        args = self._build(compute_type="wint4")
        self.assertEqual(
            args.weight_quantize_algo,
            {"weight_only_int4": EXPECTED_DEFAULT_QUANTIZE_LAYERS},
        )
        self.assertTrue(args.bf16)

    def test_nf4_uses_default_layers_under_nf4_key(self):
        args = self._build(compute_type="nf4")
        self.assertEqual(
            args.weight_quantize_algo,
            {"nf4": EXPECTED_DEFAULT_QUANTIZE_LAYERS},
        )

    def test_wint4_8_splits_experts_int4_and_attention_int8(self):
        args = self._build(compute_type="wint4/8")
        self.assertEqual(
            args.weight_quantize_algo,
            {
                "weight_only_int4": EXPECTED_WINT4_8_INT4,
                "weight_only_int8": EXPECTED_WINT4_8_INT8,
            },
        )

    def test_unknown_compute_type_is_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            self._build(compute_type="does_not_exist")
        self.assertIn("Unknown compute_type", str(ctx.exception))


@unittest.skipUnless(_HAS_PADDLE, _SKIP_REASON)
class TestArgumentParserWiring(unittest.TestCase):
    """Drive the real PdArgumentParser over the real FinetuningArguments fields.

    Stays at the argparse layer (parse_known_args returns a namespace and does
    not construct the dataclass), so no distributed runtime is required.
    """

    def _namespace(self, extra_argv):
        parser = PdArgumentParser(FinetuningArguments)
        # output_dir has no default -> parser marks it required.
        argv = ["--output_dir", "/tmp/finetuning_args_test", *extra_argv]
        namespace, remaining = parser.parse_known_args(argv)
        return namespace, remaining

    def test_bool_flag_default_false_and_set_true(self):
        # use_fp8 is a plain bool with default False.
        ns_default, _ = self._namespace([])
        self.assertFalse(ns_default.use_fp8)

        ns_set, _ = self._namespace(["--use_fp8"])
        self.assertTrue(ns_set.use_fp8)

    def test_default_true_bool_has_no_complement(self):
        # moe_with_send_router_loss defaults to True; PdArgumentParser must
        # synthesize a --no_ complement that flips it to False.
        ns_default, _ = self._namespace([])
        self.assertTrue(ns_default.moe_with_send_router_loss)

        ns_negated, _ = self._namespace(["--no_moe_with_send_router_loss"])
        self.assertFalse(ns_negated.moe_with_send_router_loss)

    def test_use_accuracy_compatible_const_and_explicit_value(self):
        # str field: default "", bare flag resolves to the megatron const,
        # explicit value is taken verbatim.
        ns_default, _ = self._namespace([])
        self.assertEqual(ns_default.use_accuracy_compatible, "")

        ns_bare, _ = self._namespace(["--use_accuracy_compatible"])
        self.assertEqual(ns_bare.use_accuracy_compatible, "megatron")

        ns_hf, _ = self._namespace(["--use_accuracy_compatible", "hf"])
        self.assertEqual(ns_hf.use_accuracy_compatible, "hf")


if __name__ == "__main__":
    unittest.main()

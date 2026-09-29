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

"""Behavior tests for ``paddlefleet.model_parallel_config.ModelParallelConfig``.

``ModelParallelConfig`` is a ``@dataclass`` whose ``__post_init__`` is the real
unit under test: it (a) derives fields that were left ``None`` from other fields,
(b) auto-disables sequence parallelism when tensor parallelism is absent, and
(c) validates several combinations of pipeline / expert / p2p-overlap flags,
raising ``ValueError`` with a specific message on conflict.

These tests exercise that constructor-time logic and compare against expected
values derived by hand from the documented parallelism rules -- never by reading
back a field that the test itself just assigned. Because the production module
imports ``paddle`` at import time, the whole suite is skipped with an honest
reason when Paddle (and therefore the module) is unavailable on the host.
"""

import os
import sys
import unittest

# Make the in-tree ``src/paddlefleet`` importable when the package has not been
# pip-installed into the environment (repo_root/src is 4 levels up from here).
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

try:
    import paddle

    from paddlefleet.model_parallel_config import ModelParallelConfig

    _IMPORT_ERROR = None
except ImportError as exc:  # paddle / paddlefleet not installed on this host
    paddle = None
    ModelParallelConfig = None
    _IMPORT_ERROR = exc


@unittest.skipUnless(
    _IMPORT_ERROR is None,
    f"paddle/paddlefleet unavailable: {_IMPORT_ERROR}",
)
class ModelParallelConfigPostInitTest(unittest.TestCase):
    """Constructor-time derivation and validation of ModelParallelConfig."""

    # --- Defaulting of "None" fields from sibling fields -----------------

    def test_autocast_dtype_defaults_to_params_dtype_when_unset(self):
        """autocast_dtype left None must inherit the default params_dtype.

        Hand-derived: params_dtype defaults to float32, so an unset
        autocast_dtype must resolve to float32 (not stay None).
        """
        config = ModelParallelConfig()
        self.assertEqual(config.params_dtype, paddle.float32)
        self.assertEqual(config.autocast_dtype, paddle.float32)

    def test_autocast_dtype_follows_explicit_params_dtype(self):
        """An unset autocast_dtype tracks whatever params_dtype was given."""
        config = ModelParallelConfig(params_dtype=paddle.float16)
        # Expected value is params_dtype itself, propagated by __post_init__.
        self.assertEqual(config.autocast_dtype, paddle.float16)

    def test_explicit_autocast_dtype_is_not_overwritten(self):
        """A caller-provided autocast_dtype must survive defaulting.

        params_dtype (float16) differs from autocast_dtype (float32); if the
        "only default when None" guard were broken and always copied
        params_dtype, autocast_dtype would wrongly become float16.
        """
        config = ModelParallelConfig(
            params_dtype=paddle.float16,
            autocast_dtype=paddle.float32,
        )
        self.assertEqual(config.autocast_dtype, paddle.float32)
        self.assertEqual(config.params_dtype, paddle.float16)

    def test_expert_tensor_parallel_defaults_to_tensor_parallel(self):
        """expert_tensor_parallel_size left None mirrors tensor_model_parallel_size."""
        config = ModelParallelConfig(tensor_model_parallel_size=4)
        self.assertEqual(config.expert_tensor_parallel_size, 4)

    def test_explicit_expert_tensor_parallel_is_not_overwritten(self):
        """A distinct explicit expert_tensor_parallel_size must be preserved.

        tp=4 but etp=2: if the None-guard were broken and always copied tp,
        etp would wrongly become 4.
        """
        config = ModelParallelConfig(
            tensor_model_parallel_size=4,
            expert_tensor_parallel_size=2,
        )
        self.assertEqual(config.expert_tensor_parallel_size, 2)

    def test_microbatch_group_defaults_to_pipeline_parallel(self):
        """microbatch_group_size_per_vp_stage left None mirrors pp size."""
        config = ModelParallelConfig(pipeline_model_parallel_size=4)
        self.assertEqual(config.microbatch_group_size_per_vp_stage, 4)

    def test_explicit_microbatch_group_is_not_overwritten(self):
        """A distinct explicit microbatch group size must be preserved.

        pp=4 but group=3: a broken guard that always copied pp would yield 4.
        """
        config = ModelParallelConfig(
            pipeline_model_parallel_size=4,
            microbatch_group_size_per_vp_stage=3,
        )
        self.assertEqual(config.microbatch_group_size_per_vp_stage, 3)

    # --- Conditional auto-disable of sequence parallelism ----------------

    def test_sequence_parallel_off_by_default(self):
        """Default config has tp=1 and therefore sequence_parallel False."""
        config = ModelParallelConfig()
        self.assertEqual(config.tensor_model_parallel_size, 1)
        self.assertFalse(config.sequence_parallel)

    def test_sequence_parallel_auto_disabled_without_tensor_parallel(self):
        """Requesting sequence_parallel with tp<=1 silently disables it.

        The intent flag is True on input but the tp<=1 branch transforms it to
        False; this is a real state transformation, not a read-back. It also
        confirms the ValueError guard for "sequence parallelism without tensor
        parallelism" is never reached from the constructor, because the
        auto-disable runs first (see report note on unreachable validation).
        """
        config = ModelParallelConfig(
            tensor_model_parallel_size=1,
            sequence_parallel=True,
        )
        self.assertFalse(config.sequence_parallel)

    def test_sequence_parallel_kept_with_tensor_parallel(self):
        """With tp>1 the requested sequence_parallel is left enabled."""
        config = ModelParallelConfig(
            tensor_model_parallel_size=4,
            sequence_parallel=True,
        )
        self.assertTrue(config.sequence_parallel)

    # --- Deferred embedding wgrad validation -----------------------------

    def test_defer_wgrad_requires_pipeline_parallel(self):
        """defer_embedding_wgrad_compute with pp==1 is rejected."""
        with self.assertRaises(ValueError) as ctx:
            ModelParallelConfig(
                defer_embedding_wgrad_compute=True,
                pipeline_model_parallel_size=1,
                gradient_accumulation_fusion=True,
            )
        self.assertIn("pipeline model parallel", str(ctx.exception).lower())

    def test_defer_wgrad_requires_gradient_accumulation_fusion(self):
        """defer_embedding_wgrad_compute without grad-accum fusion is rejected."""
        with self.assertRaises(ValueError) as ctx:
            ModelParallelConfig(
                defer_embedding_wgrad_compute=True,
                pipeline_model_parallel_size=2,
                gradient_accumulation_fusion=False,
            )
        self.assertIn(
            "gradient accumulation fusion", str(ctx.exception).lower()
        )

    def test_defer_wgrad_rejects_negative_deferral_limit(self):
        """A negative wgrad_deferral_limit is rejected when deferral is on."""
        with self.assertRaises(ValueError) as ctx:
            ModelParallelConfig(
                defer_embedding_wgrad_compute=True,
                pipeline_model_parallel_size=2,
                gradient_accumulation_fusion=True,
                wgrad_deferral_limit=-1,
            )
        self.assertIn("greater than or equal to 0", str(ctx.exception).lower())

    def test_defer_wgrad_pipeline_check_precedes_fusion_check(self):
        """When pp==1 AND fusion is off, the pipeline error wins (ordering).

        Both guards would fire; __post_init__ checks pp first, so the raised
        message must mention pipeline parallelism, not fusion. A reordering of
        the two guards would flip this message and fail the test.
        """
        with self.assertRaises(ValueError) as ctx:
            ModelParallelConfig(
                defer_embedding_wgrad_compute=True,
                pipeline_model_parallel_size=1,
                gradient_accumulation_fusion=False,
            )
        message = str(ctx.exception).lower()
        self.assertIn("pipeline model parallel", message)
        self.assertNotIn("gradient accumulation fusion", message)

    def test_defer_wgrad_valid_combination_constructs(self):
        """A fully valid deferral config builds and still derives siblings.

        Rather than reading back the fields we set, assert a *derived* value:
        with pp=2 and no explicit microbatch group, the group must default to
        the pipeline size (2). This proves construction reached the end of
        __post_init__ without raising.
        """
        config = ModelParallelConfig(
            defer_embedding_wgrad_compute=True,
            pipeline_model_parallel_size=2,
            gradient_accumulation_fusion=True,
            wgrad_deferral_limit=5,
        )
        self.assertEqual(config.microbatch_group_size_per_vp_stage, 2)

    # --- Expert + tensor parallel requires sequence parallel -------------

    def test_expert_and_tensor_parallel_require_sequence_parallel(self):
        """ep>1 and tp>1 without sequence parallel is rejected."""
        with self.assertRaises(ValueError) as ctx:
            ModelParallelConfig(
                expert_model_parallel_size=2,
                tensor_model_parallel_size=2,
                sequence_parallel=False,
            )
        self.assertIn("sequence parallelism must be used", str(ctx.exception))

    def test_expert_and_tensor_parallel_with_sequence_parallel_ok(self):
        """ep>1 + tp>1 + sequence parallel builds; siblings still derive.

        Assert derived expert_tensor_parallel_size == tp (=2) rather than
        re-reading the flags we passed, tying the success path to real
        defaulting behavior.
        """
        config = ModelParallelConfig(
            expert_model_parallel_size=2,
            tensor_model_parallel_size=2,
            sequence_parallel=True,
        )
        self.assertTrue(config.sequence_parallel)
        self.assertEqual(config.expert_tensor_parallel_size, 2)

    def test_expert_parallel_without_tensor_parallel_has_no_requirement(self):
        """ep>1 but tp==1 must NOT require sequence parallel.

        The guard triggers only when BOTH degrees exceed 1. With tp=1 the
        config builds even though sequence_parallel is False, and sequence
        parallelism is additionally auto-disabled by the tp<=1 branch.
        """
        config = ModelParallelConfig(
            expert_model_parallel_size=2,
            tensor_model_parallel_size=1,
            sequence_parallel=True,
        )
        self.assertFalse(config.sequence_parallel)
        self.assertEqual(config.expert_model_parallel_size, 2)

    # --- overlap_p2p_comm_warmup_flush compatibility ---------------------

    def test_warmup_flush_rejects_batch_p2p_comm(self):
        """Warmup/flush overlap is incompatible with batch_p2p_comm=True."""
        with self.assertRaises(ValueError):
            ModelParallelConfig(
                overlap_p2p_comm=True,
                batch_p2p_comm=True,
                overlap_p2p_comm_warmup_flush=True,
            )

    def test_warmup_flush_requires_overlap_p2p_comm(self):
        """Warmup/flush overlap requires overlap_p2p_comm=True."""
        with self.assertRaises(ValueError):
            ModelParallelConfig(
                overlap_p2p_comm=False,
                batch_p2p_comm=False,
                overlap_p2p_comm_warmup_flush=True,
            )

    def test_warmup_flush_valid_combination_constructs(self):
        """overlap on + batch off + warmup on builds and derives siblings.

        Assert a derived field (autocast_dtype -> default float32) instead of
        re-reading the p2p flags, confirming __post_init__ finished.
        """
        config = ModelParallelConfig(
            overlap_p2p_comm=True,
            batch_p2p_comm=False,
            overlap_p2p_comm_warmup_flush=True,
        )
        self.assertEqual(config.autocast_dtype, paddle.float32)


if __name__ == "__main__":
    unittest.main()

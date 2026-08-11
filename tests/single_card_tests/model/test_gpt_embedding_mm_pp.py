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
"""Tests for the Qwen3.5 PP refactor of ``GPTEmbedding``.

Covers:
  * ``GPTEmbedding._merge_multimodal`` (extracted helper): image-only,
    video-only, image+video (deepstack join) and the no-visual early exit.
  * ``forward``: multimodal merge now runs *before* the MTP split, the SP
    scatter happens once after it, and ``visual_pos_masks`` is truncated when
    MTP shortens the main branch.
  * ``forward``: mRoPE ``position_ids`` ([3, B, S]) are sliced on the last
    axis for MTP.
  * ``forward``: every tensor in ``preproc_output`` is made contiguous for
    pipeline P2P send.
"""

import os
import subprocess
import sys
import unittest
from unittest.mock import MagicMock, patch

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), *[".."] * 3)
)
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import numpy as np
import paddle

from paddlefleet.models.gpt.gpt_embedding import GPTEmbedding

IMAGE_TOKEN_ID = 101
VIDEO_TOKEN_ID = 102


class _Cfg:
    """Minimal stand-in for ``TransformerConfig``.

    A plain object (not ``MagicMock``) so that unset attributes raise instead
    of silently evaluating to a truthy mock.
    """

    def __init__(self, **overrides):
        self.gpt_model_use_experimental_version = False
        self.multi_latent_attention = False
        self.max_sequence_length = 16
        self.multimodal_embedding = True
        self.sequence_parallel = False
        self.expert_model_parallel_size = 1
        self.tensor_model_parallel_size = 1
        self.pad_token_id = 0
        self.num_nextn_predict_layers = 0
        self.mtp_load_weight_only = False
        self.enable_mtp_magic_send = False
        self.experimental_dataflow = False
        self.cp_balance_mode = "padding"
        self.clone_scatter_output_in_embedding = False
        self.apply_rope_fusion = False
        self.image_token_id = IMAGE_TOKEN_ID
        self.video_token_id = VIDEO_TOKEN_ID
        self.__dict__.update(overrides)


def _make_embedding(config=None, **attrs):
    """Build a ``GPTEmbedding`` without running ``__init__``."""
    emb = GPTEmbedding.__new__(GPTEmbedding)
    emb.__dict__.setdefault("_parameters", {})
    emb.__dict__.setdefault("_buffers", {})
    emb.__dict__.setdefault("_sub_layers", {})
    emb.__dict__.setdefault("_loaddict_holder", {})
    emb.__dict__.setdefault("_non_persistable_buffers", set())
    emb.__dict__.setdefault("_non_persistable_buffer_names_set", set())
    defaults = {
        "config": config if config is not None else _Cfg(),
        "position_embedding_type": "none",
        "rotary_pos_emb": None,
        "swa_rotary_pos_emb": None,
        "mrope_section": None,
        "sequence_parallel": False,
        "multimodal_embedding": True,
        "training": False,
    }
    defaults.update(attrs)
    for key, value in defaults.items():
        object.__setattr__(emb, key, value)
    return emb


class TestMergeMultimodal(unittest.TestCase):
    """``GPTEmbedding._merge_multimodal`` replaces placeholder embeddings."""

    def setUp(self):
        self.hidden = 4
        # [image, image, video, video, pad, pad]
        self.input_ids = paddle.to_tensor(
            [
                [
                    IMAGE_TOKEN_ID,
                    IMAGE_TOKEN_ID,
                    VIDEO_TOKEN_ID,
                    VIDEO_TOKEN_ID,
                    0,
                    0,
                ]
            ],
            dtype="int64",
        )
        self.decoder_input = paddle.ones([1, 6, self.hidden], dtype="float32")

    def test_no_visual_features_is_a_noop(self):
        emb = _make_embedding()
        out, masks, deepstack = emb._merge_multimodal(
            {}, self.input_ids, self.decoder_input, None, None
        )
        self.assertIs(out, self.decoder_input)
        self.assertIsNone(masks)
        self.assertIsNone(deepstack)

    def test_image_only_scatters_image_embeds(self):
        emb = _make_embedding()
        image_embeds = paddle.full([2, self.hidden], 7.0, dtype="float32")
        deepstack_image = [paddle.zeros([2, self.hidden])]

        out, masks, deepstack = emb._merge_multimodal(
            {"image_embeds": image_embeds},
            self.input_ids,
            self.decoder_input,
            deepstack_image,
            None,
        )

        self.assertEqual(list(out.shape), [1, 6, self.hidden])
        # image positions replaced, everything else untouched
        np.testing.assert_allclose(
            out[0, :2].numpy(), np.full([2, self.hidden], 7.0)
        )
        np.testing.assert_allclose(
            out[0, 2:].numpy(), np.ones([4, self.hidden], dtype="float32")
        )
        self.assertEqual(list(masks.shape), [1, 6])
        self.assertEqual(masks.numpy().tolist(), [[1, 1, 0, 0, 0, 0]])
        self.assertIs(deepstack, deepstack_image)

    def test_video_only_scatters_video_embeds(self):
        emb = _make_embedding()
        video_embeds = paddle.full([2, self.hidden], 3.0, dtype="float32")
        deepstack_video = [paddle.zeros([2, self.hidden])]

        out, masks, deepstack = emb._merge_multimodal(
            {"video_embeds": video_embeds},
            self.input_ids,
            self.decoder_input,
            None,
            deepstack_video,
        )

        np.testing.assert_allclose(
            out[0, 2:4].numpy(), np.full([2, self.hidden], 3.0)
        )
        self.assertEqual(masks.numpy().tolist(), [[0, 0, 1, 1, 0, 0]])
        self.assertIs(deepstack, deepstack_video)

    def test_image_and_video_join_masks_and_deepstack(self):
        emb = _make_embedding()
        image_embeds = paddle.full([2, self.hidden], 7.0, dtype="float32")
        video_embeds = paddle.full([2, self.hidden], 3.0, dtype="float32")
        # deepstack embeds live in the joint visual space: [N_visual, H]
        n_visual = 4
        deepstack_image = [paddle.full([n_visual, self.hidden], 11.0)]
        deepstack_video = [paddle.full([n_visual, self.hidden], 13.0)]

        out, masks, deepstack = emb._merge_multimodal(
            {"image_embeds": image_embeds, "video_embeds": video_embeds},
            self.input_ids,
            self.decoder_input,
            deepstack_image,
            deepstack_video,
        )

        np.testing.assert_allclose(
            out[0, :2].numpy(), np.full([2, self.hidden], 7.0)
        )
        np.testing.assert_allclose(
            out[0, 2:4].numpy(), np.full([2, self.hidden], 3.0)
        )
        self.assertEqual(masks.numpy().tolist(), [[1, 1, 1, 1, 0, 0]])
        # joint deepstack: image rows keep the image value, video rows the video
        self.assertEqual(len(deepstack), 1)
        joint = deepstack[0].numpy()
        self.assertEqual(joint.shape, (n_visual, self.hidden))
        np.testing.assert_allclose(joint[:2], np.full([2, self.hidden], 11.0))
        np.testing.assert_allclose(joint[2:], np.full([2, self.hidden], 13.0))


class TestForwardMultimodalOrdering(unittest.TestCase):
    """Multimodal merge happens before the MTP split; SP scatter happens once."""

    def setUp(self):
        self.hidden = 4
        self.seq = 6
        self.input_ids = paddle.to_tensor(
            [[IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, 5, 6, 0, 0]], dtype="int64"
        )
        self.image_embeds = paddle.full([2, self.hidden], 7.0, dtype="float32")

    def _embedding_mock(self):
        mock = MagicMock(
            return_value=paddle.ones([1, self.seq, self.hidden], "float32")
        )
        mock.tp_group = object()
        return mock

    @patch(
        "paddlefleet.models.gpt.gpt_embedding.scatter_to_sequence_parallel_region"
    )
    def test_sequence_parallel_scatter_runs_after_merge(self, mock_scatter):
        """No MTP: the SP scatter closes the multimodal branch exactly once."""
        scattered = paddle.zeros([self.seq, 1, self.hidden], "float32")
        mock_scatter.return_value = scattered
        cfg = _Cfg(
            sequence_parallel=True, clone_scatter_output_in_embedding=True
        )
        emb = _make_embedding(
            cfg, sequence_parallel=True, embedding=self._embedding_mock()
        )

        result = emb.forward(
            {
                "input_ids": self.input_ids,
                "image_embeds": self.image_embeds,
            }
        )

        mock_scatter.assert_called_once()
        # merge already replaced the image tokens before the scatter ran
        merged = mock_scatter.call_args[0][0]
        self.assertEqual(list(merged.shape), [self.seq, 1, self.hidden])
        np.testing.assert_allclose(
            merged[:2, 0].numpy(), np.full([2, self.hidden], 7.0)
        )
        # clone_scatter_output_in_embedding=True -> a copy, not the same tensor
        self.assertIsNot(result["hidden_states"], scattered)
        self.assertEqual(
            list(result["hidden_states"].shape), [self.seq, 1, self.hidden]
        )
        self.assertEqual(list(result["visual_pos_masks"].shape), [1, self.seq])

    @patch(
        "paddlefleet.models.gpt.gpt_embedding.scatter_to_sequence_parallel_region"
    )
    def test_sequence_parallel_scatter_without_clone(self, mock_scatter):
        """clone_scatter_output_in_embedding=False returns the scatter output."""
        scattered = paddle.zeros([self.seq, 1, self.hidden], "float32")
        mock_scatter.return_value = scattered
        cfg = _Cfg(
            sequence_parallel=True, clone_scatter_output_in_embedding=False
        )
        emb = _make_embedding(
            cfg, sequence_parallel=True, embedding=self._embedding_mock()
        )

        result = emb.forward(
            {"input_ids": self.input_ids, "image_embeds": self.image_embeds}
        )

        self.assertIs(result["hidden_states"], scattered)

    def test_multimodal_without_sequence_parallel_is_not_scattered(self):
        """No SP: the merged embedding keeps its [B, S, H] layout."""
        emb = _make_embedding(_Cfg(), embedding=self._embedding_mock())

        result = emb.forward(
            {"input_ids": self.input_ids, "image_embeds": self.image_embeds}
        )

        self.assertEqual(
            list(result["hidden_states"].shape), [1, self.seq, self.hidden]
        )
        np.testing.assert_allclose(
            result["hidden_states"][0, :2].numpy(),
            np.full([2, self.hidden], 7.0),
        )

    def test_plain_path_without_multimodal(self):
        """multimodal_embedding=False skips both merge and mask bookkeeping."""
        embedding = self._embedding_mock()
        emb = _make_embedding(
            _Cfg(multimodal_embedding=False),
            multimodal_embedding=False,
            embedding=embedding,
        )

        result = emb.forward({"input_ids": self.input_ids})

        self.assertNotIn("visual_pos_masks", result)
        self.assertNotIn("deepstack_visual_emb", result)
        np.testing.assert_allclose(
            result["hidden_states"].numpy(), embedding.return_value.numpy()
        )

    def test_mtp_without_visual_embeds_leaves_masks_unset(self):
        """MTP + multimodal but no image/video features: nothing to truncate."""
        cfg = _Cfg(num_nextn_predict_layers=2)
        emb = _make_embedding(cfg, embedding=self._embedding_mock())

        result = emb.forward({"input_ids": self.input_ids})

        self.assertNotIn("visual_pos_masks", result)
        self.assertEqual(
            list(result["hidden_states"].shape), [3, self.seq - 2, self.hidden]
        )

    def test_mtp_position_ids_already_truncated_are_kept(self):
        """Ids that already match the main branch length are passed as-is."""
        num_mtp = 2
        cfg = _Cfg(num_nextn_predict_layers=num_mtp)
        rope = MagicMock()
        rope.get_rotary_seq_len = MagicMock(return_value=4)
        rope.return_value = paddle.zeros([1, 4, 1, 8], "float32")
        emb = _make_embedding(
            cfg,
            position_embedding_type="rope",
            rotary_pos_emb=rope,
            embedding=self._embedding_mock(),
        )
        main_seq = self.seq - num_mtp
        position_ids = paddle.arange(3 * main_seq, dtype="int64").reshape(
            [3, 1, main_seq]
        )

        emb.forward(
            {
                "input_ids": self.input_ids,
                "image_embeds": self.image_embeds,
                "position_ids": position_ids,
            }
        )

        passed = rope.call_args.kwargs["position_ids"]
        self.assertEqual(list(passed.shape), [3, 1, main_seq])

    def test_mtp_truncates_visual_pos_masks(self):
        num_mtp = 2
        cfg = _Cfg(
            num_nextn_predict_layers=num_mtp,
            expert_model_parallel_size=2,
            tensor_model_parallel_size=1,
        )
        emb = _make_embedding(cfg, embedding=self._embedding_mock())

        result = emb.forward(
            {
                "input_ids": self.input_ids,
                "image_embeds": self.image_embeds,
            }
        )

        main_seq = self.seq - num_mtp
        # masks are indexed against the (shortened) main branch
        self.assertEqual(list(result["visual_pos_masks"].shape), [1, main_seq])
        self.assertEqual(
            result["visual_pos_masks"].numpy().tolist(), [[1, 1, 0, 0]]
        )
        # hidden_states holds main + per-depth MTP chunks concatenated
        self.assertEqual(
            list(result["hidden_states"].shape),
            [num_mtp + 1, main_seq, self.hidden],
        )
        # the fill_feature/EP branch published input_ids for the MoE mask
        self.assertEqual(list(result["input_ids"].shape), [1, main_seq])
        self.assertEqual(
            list(result["mtp_input_ids_for_moe_mask"].shape),
            [1, num_mtp, main_seq],
        )

    def test_mtp_with_deepstack_raises(self):
        cfg = _Cfg(num_nextn_predict_layers=1)
        emb = _make_embedding(cfg, embedding=self._embedding_mock())

        with self.assertRaises(ValueError) as ctx:
            emb.forward(
                {
                    "input_ids": self.input_ids,
                    "image_embeds": self.image_embeds,
                    "deepstack_image_embeds": [
                        paddle.zeros([2, self.hidden], "float32")
                    ],
                }
            )
        self.assertIn("deepstack", str(ctx.exception))

    def test_mtp_with_deepstack_raises_under_optimized_mode(self):
        """``python -O`` strips asserts, so this guard must be a real raise.

        Re-runs the in-process case above in an optimized subprocess instead of
        duplicating the embedding harness.
        """
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [os.path.join(REPO_ROOT, "src"), env.get("PYTHONPATH", "")]
        )
        proc = subprocess.run(
            [
                sys.executable,
                "-O",
                "-m",
                "pytest",
                "-q",
                os.path.abspath(__file__),
                "-k",
                "test_mtp_with_deepstack_raises and not optimized",
            ],
            capture_output=True,
            text=True,
            env=env,
        )

        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("1 passed", proc.stdout)

    def test_mrope_position_ids_sliced_on_last_axis(self):
        """[3, B, S] mRoPE ids must be sliced on the sequence (last) axis."""
        num_mtp = 2
        cfg = _Cfg(num_nextn_predict_layers=num_mtp)
        rope = MagicMock()
        rope.get_rotary_seq_len = MagicMock(return_value=4)
        rope.return_value = paddle.zeros([1, 4, 1, 8], "float32")
        emb = _make_embedding(
            cfg,
            position_embedding_type="rope",
            rotary_pos_emb=rope,
            embedding=self._embedding_mock(),
        )
        position_ids = paddle.arange(3 * self.seq, dtype="int64").reshape(
            [3, 1, self.seq]
        )

        emb.forward(
            {
                "input_ids": self.input_ids,
                "image_embeds": self.image_embeds,
                "position_ids": position_ids,
            }
        )

        passed = rope.call_args.kwargs["position_ids"]
        self.assertEqual(list(passed.shape), [3, 1, self.seq - num_mtp])
        np.testing.assert_array_equal(
            passed.numpy(), position_ids.numpy()[..., : self.seq - num_mtp]
        )


class TestForwardMultimodalMTPSequenceParallelMRope(unittest.TestCase):
    """All four features at once: multimodal + MTP + SP + real mRoPE.

    The individual regressions this guards:
      * the visual features must survive into *every* MTP depth (the merge runs
        before the MTP split, so each shifted chunk carries them);
      * the multimodal SP scatter must not run a second time when the MTP
        branch already scattered each chunk;
      * ``visual_pos_masks`` must be truncated to the main branch length;
      * the real mRoPE branch must produce ``[S, B, head_dim]`` under SP.
    """

    hidden = 4
    seq = 6
    num_mtp = 2
    head_dim = 8
    text_value = 1.0
    image_value = 7.0

    def setUp(self):
        # image tokens sit in the middle so that every MTP shift lands on a
        # different mix of visual/text positions
        self.input_ids = paddle.to_tensor(
            [[5, 6, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, 0, 0]], dtype="int64"
        )
        self.image_embeds = paddle.full(
            [2, self.hidden], self.image_value, dtype="float32"
        )
        self.position_ids = paddle.arange(3 * self.seq, dtype="int64").reshape(
            [3, 1, self.seq]
        )

    @patch(
        "paddlefleet.models.gpt.gpt_embedding.scatter_to_sequence_parallel_region"
    )
    @patch("paddlefleet.models.gpt.gpt_embedding.ScatterOp")
    def test_visual_features_reach_every_mtp_depth(
        self, mock_scatter_op, mock_sp_scatter
    ):
        from paddlefleet.models.common.embeddings.rotary_pos_embedding import (
            MultimodalRotaryEmbedding,
        )

        # tp=1: the per-chunk MTP scatter is an identity
        mock_scatter_op.apply = staticmethod(lambda x: x)
        cfg = _Cfg(
            num_nextn_predict_layers=self.num_mtp,
            sequence_parallel=True,
        )
        emb = _make_embedding(
            cfg,
            sequence_parallel=True,
            position_embedding_type="mrope",
            rotary_pos_emb=MultimodalRotaryEmbedding(
                head_dim=self.head_dim, rotary_percent=1.0
            ),
            mrope_section=[2, 1, 1],
            embedding=MagicMock(
                return_value=paddle.full(
                    [1, self.seq, self.hidden],
                    self.text_value,
                    dtype="float32",
                )
            ),
        )

        result = emb.forward(
            {
                "input_ids": self.input_ids,
                "image_embeds": self.image_embeds,
                "position_ids": self.position_ids,
            }
        )

        main_seq = self.seq - self.num_mtp
        # the MTP branch scattered each chunk; the multimodal branch must not
        # scatter the already-scattered tensor a second time
        mock_sp_scatter.assert_not_called()

        # hidden_states = [main, depth0, depth1] each [S_main, B, H]
        hidden = result["hidden_states"]
        self.assertEqual(
            list(hidden.shape), [(self.num_mtp + 1) * main_seq, 1, self.hidden]
        )
        t, v = self.text_value, self.image_value
        # merged embedding rows: [t, t, v, v, t, t]
        #   main    = rows 0..3        -> t t v v
        #   depth 0 = rows 1..4        -> t v v t
        #   depth 1 = rows 2..5        -> v v t t
        expected = [
            [t, t, v, v],
            [t, v, v, t],
            [v, v, t, t],
        ]
        for depth, row_values in enumerate(expected):
            chunk = hidden[depth * main_seq : (depth + 1) * main_seq, 0]
            np.testing.assert_allclose(
                chunk.numpy(),
                np.array(
                    [[val] * self.hidden for val in row_values],
                    dtype="float32",
                ),
                err_msg=f"visual features wrong in MTP chunk {depth}",
            )

        # masks are indexed against the (truncated) main branch
        self.assertEqual(list(result["visual_pos_masks"].shape), [1, main_seq])
        self.assertEqual(
            result["visual_pos_masks"].numpy().tolist(), [[0, 0, 1, 1]]
        )

        # real mRoPE output, transposed to [S, B, head_dim] for SP
        rope_emb = result["rotary_pos_emb"]
        self.assertEqual(list(rope_emb.shape), [self.seq, 1, self.head_dim])
        self.assertTrue(rope_emb.is_contiguous())


class TestForwardPreprocContiguous(unittest.TestCase):
    """Every tensor handed to PP P2P send must be contiguous."""

    def test_non_contiguous_tensors_are_made_contiguous(self):
        emb = _make_embedding(_Cfg(multimodal_embedding=False))
        decoder_input = paddle.randn([2, 4, 8])
        # a transposed view is not contiguous
        attention_mask = paddle.randn([2, 4, 4]).transpose([0, 2, 1])
        self.assertFalse(attention_mask.is_contiguous())

        result = emb.forward(
            {"input_ids": None, "attention_mask": attention_mask},
            decoder_input=decoder_input,
        )

        self.assertTrue(result["attention_mask"].is_contiguous())
        self.assertTrue(result["hidden_states"].is_contiguous())
        np.testing.assert_allclose(
            result["attention_mask"].numpy(), attention_mask.numpy()
        )

    def test_tensors_inside_lists_are_made_contiguous(self):
        """deepstack features travel as a list, which must be swept too."""
        emb = _make_embedding(
            _Cfg(),
            embedding=MagicMock(return_value=paddle.ones([1, 6, 4], "float32")),
        )
        input_ids = paddle.to_tensor(
            [[IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, 5, 6, 0, 0]], dtype="int64"
        )
        # a transposed [4, 2] view -> non-contiguous [2, 4] deepstack feature
        deepstack = paddle.randn([4, 2]).T
        self.assertFalse(deepstack.is_contiguous())

        result = emb.forward(
            {
                "input_ids": input_ids,
                "image_embeds": paddle.full([2, 4], 7.0, dtype="float32"),
                "deepstack_image_embeds": [deepstack],
            }
        )

        self.assertTrue(result["deepstack_visual_emb"][0].is_contiguous())
        np.testing.assert_allclose(
            result["deepstack_visual_emb"][0].numpy(), deepstack.numpy()
        )

    def test_make_contiguous_passes_non_tensors_through(self):
        from paddlefleet.models.gpt.gpt_embedding import make_contiguous

        sentinel = object()
        self.assertIs(make_contiguous(sentinel), sentinel)
        self.assertIsNone(make_contiguous(None))
        swept = make_contiguous((paddle.randn([2, 3]).T,))
        self.assertIsInstance(swept, tuple)
        self.assertTrue(swept[0].is_contiguous())

    def test_build_schedule_node(self):
        from paddle.distributed.fleet.meta_parallel import ScheduleNode

        emb = _make_embedding()
        self.assertIsInstance(emb.build_schedule_node(), ScheduleNode)


if __name__ == "__main__":
    unittest.main()

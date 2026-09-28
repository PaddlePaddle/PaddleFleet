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

"""Dump-layout canonicalizers used to align train vs inference top-k dumps.

Two ``pre_save_func`` helpers rewrite the training-side index layout so it
lines up slot-by-slot with what inference emits, so the two ``.npy`` dumps can
be compared directly:

* ``_sort_topk_indices_pad_last`` (``mqa_latent_attention``) -- tag
  ``mla_indexer_topk_indices``: sort each row ascending (inference emits
  increasing key order) while pinning ``-1`` padding to the row tail.
* ``_fold_window_into_topk_pad`` (``mqa_latent_attention``) -- tag
  ``mla_indexer_token_indices``: training lays slots out as ``[top-k |
  window]`` with the top-k block's ``-1`` padding sitting between the two runs;
  inference emits ``[valid top-k | window]`` with the padding trailing, so
  splice the window block into the top-k gap and push the padding to the tail.
* ``_reverse_window_and_topk`` (``csa_attention``) -- tag
  ``attn_compressor_topk_idxs``: training lays slots out as ``[window,
  compress]``; inference concatenates ``[compress, window]``, so re-lay to the
  inference order.

Neither is exercised by the default (probe-off) tests, and the reorder is the
whole point of the alignment, so pin it here. Both are pure integer reorders
run before any kernel, so the checks are exact. The last test drives a real
``inspect_tensor`` save->load round-trip to prove the layout the consumer reads
back is the canonical one, not just the training order.
"""

from __future__ import annotations

import tempfile
import unittest

import paddle

import paddlefleet.train_infer_consistent_ops.inspect_util as iu
from paddlefleet.transformer.csa_attention import _reverse_window_and_topk
from paddlefleet.transformer.mqa_latent_attention import (
    _fold_window_into_topk_pad,
    _sort_topk_indices_pad_last,
)


def _i32(rows):
    return paddle.to_tensor(rows, dtype="int32")


@unittest.skipUnless(
    paddle.device.is_compiled_with_cuda(),
    "paddlefleet import chain needs a CUDA build",
)
class TestDumpLayoutAlignment(unittest.TestCase):
    def setUp(self) -> None:
        paddle.set_device("gpu")

    # -- _sort_topk_indices_pad_last -------------------------------------

    def test_valid_indices_sorted_ascending(self) -> None:
        """A padding-free row comes back in strictly ascending key order."""
        out = _sort_topk_indices_pad_last(_i32([[5, 0, 9, 2, 7]]))
        self.assertEqual(out.numpy().tolist(), [[0, 2, 5, 7, 9]])

    def test_neg1_padding_kept_at_row_tail(self) -> None:
        """``-1`` padding stays at the tail; valid keys sort ahead of it.

        The valid entries land ascending and every ``-1`` is pushed past
        them, independently per row -- the ``INT32_MAX`` sort-key trick must
        not let a real (necessarily smaller) index outrank the padding.
        """
        out = _sort_topk_indices_pad_last(
            _i32([[3, -1, 1, -1, 2], [-1, -1, 8, 4, -1]])
        )
        self.assertEqual(
            out.numpy().tolist(),
            [[1, 2, 3, -1, -1], [4, 8, -1, -1, -1]],
        )

    def test_all_padding_row_is_untouched(self) -> None:
        """An all-``-1`` row round-trips to all ``-1`` (no stray MAX leaks)."""
        out = _sort_topk_indices_pad_last(_i32([[-1, -1, -1]]))
        self.assertEqual(out.numpy().tolist(), [[-1, -1, -1]])

    # -- _fold_window_into_topk_pad --------------------------------------

    def test_window_block_folds_into_the_topk_padding_gap(self) -> None:
        """``[valid|pad|window]`` becomes ``[valid|window|pad]`` per row.

        Training lays the row out as ``[top-k block | window block]`` with the
        top-k block's ``-1`` padding sitting *between* the valid top-k entries
        and the window run. Inference emits the two valid runs back to back
        with the padding trailing, so the window block must slide up into the
        first ``-1`` slot and the top-k padding must fall to the tail.
        """
        # topk_width=3 (valid 5, pad -1, valid 2), window_width=2 (100, 101).
        out = _fold_window_into_topk_pad(_i32([[5, -1, 2, 100, 101]]), 3, 2)
        self.assertEqual(out.numpy().tolist(), [[5, 2, 100, 101, -1]])

    def test_multiple_topk_pads_all_land_at_the_row_tail(self) -> None:
        """Every top-k ``-1`` ends up behind the window block, order kept.

        The window run keeps its own order and stays contiguous; the two
        top-k pads collapse to the tail rather than interleaving the window.
        """
        out = _fold_window_into_topk_pad(_i32([[-1, 7, -1, 8, 9]]), 3, 2)
        self.assertEqual(out.numpy().tolist(), [[7, 8, 9, -1, -1]])

    def test_a_full_topk_block_leaves_the_layout_untouched(self) -> None:
        """No top-k padding means there is no gap to fold the window into."""
        out = _fold_window_into_topk_pad(_i32([[5, 0, 9, 100, 101]]), 3, 2)
        self.assertEqual(out.numpy().tolist(), [[5, 0, 9, 100, 101]])

    # -- _reverse_window_and_topk ----------------------------------------

    def test_window_compress_reordered_to_compress_window(self) -> None:
        """``[window | compress]`` is re-laid to inference's ``[compress
        | window]``.

        The training tensor is built ``concat([window, compress])``; the
        helper must emit ``concat([compress, window])`` so the slots line up
        with the inference dump.
        """
        window = _i32([[0, 1, 2]])
        compress = _i32([[10, 11]])
        train_layout = paddle.concat([window, compress], axis=-1)
        out = _reverse_window_and_topk(train_layout, compress, window)
        self.assertEqual(out.numpy().tolist(), [[10, 11, 0, 1, 2]])

    def test_no_compress_branch_returns_origin_unchanged(self) -> None:
        """``top_k is None`` hands back the original object untouched.

        With no compress block there is nothing to move to the front, so the
        function must be a no-op -- same values *and* same object identity.
        """
        window = _i32([[0, 1, 2]])
        origin = _i32([[7, 8, 9]])
        out = _reverse_window_and_topk(origin, None, window)
        self.assertIs(out, origin)
        self.assertEqual(out.numpy().tolist(), [[7, 8, 9]])

    # -- load-side round-trip --------------------------------------------

    def test_saved_layout_loads_back_in_consumer_order(self) -> None:
        """A real ``inspect_tensor`` save->load returns the canonical layout.

        The helpers run as ``pre_save_func``: what gets written to ``.npy`` --
        and therefore what the consumer loads back -- is the *reordered* view,
        not the raw training order. Drive one save+load through
        ``inspect_tensor`` for each tag and assert the value handed back is the
        inference-convention layout.
        """
        saved = {
            "_ENABLED": iu._ENABLED,
            "_SAVE_PATH": iu._SAVE_PATH,
            "_LOAD_PATH": iu._LOAD_PATH,
            "_WHITELIST": iu._WHITELIST,
            "_BLACKLIST": iu._BLACKLIST,
            "_DUMP_SKIP_TAGS": iu._DUMP_SKIP_TAGS,
        }
        with tempfile.TemporaryDirectory() as tmp:
            iu._ENABLED = True
            iu._SAVE_PATH = tmp
            iu._LOAD_PATH = tmp
            iu._WHITELIST = iu._EMPTY_FILTER
            iu._BLACKLIST = iu._EMPTY_FILTER
            iu._DUMP_SKIP_TAGS = iu._EMPTY_FILTER
            try:
                window = _i32([[0, 1, 2]])
                compress = _i32([[10, 11]])
                train_layout = paddle.concat([window, compress], axis=-1)
                back = iu.inspect_tensor(
                    "attn_compressor_topk_idxs",
                    0,
                    train_layout,
                    save=True,
                    load=True,
                    pre_save_func=lambda t: _reverse_window_and_topk(
                        t, compress, window
                    ),
                )
                self.assertEqual(
                    back.astype("int32").numpy().tolist(),
                    [[10, 11, 0, 1, 2]],
                )

                raw = _i32([[3, -1, 1, -1, 2]])
                back = iu.inspect_tensor(
                    "mla_indexer_topk_indices",
                    1,
                    raw,
                    save=True,
                    load=True,
                    pre_save_func=lambda t: _sort_topk_indices_pad_last(t),
                )
                self.assertEqual(
                    back.astype("int32").numpy().tolist(),
                    [[1, 2, 3, -1, -1]],
                )
            finally:
                for name, val in saved.items():
                    setattr(iu, name, val)
        print(
            "[dsa] dump-layout helpers round-trip to consumer order via "
            "inspect_tensor",
            flush=True,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)

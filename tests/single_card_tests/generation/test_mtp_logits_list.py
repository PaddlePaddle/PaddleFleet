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

"""Main-head selection when the model returns a list/tuple of logits.

MTP (``num_nextn_predict_layers > 0``) models return one logits tensor per
head, so ``GreedyGenerator`` must unwrap ``logits[0]`` (the main head) before
sampling. There are three independent forward sites that need this:

* ``_generate_no_cache`` -- the ``no_cache=True`` full-prefill-per-step loop.
* ``generate`` prefill    -- the single KV-cache priming pass.
* ``generate`` decode     -- the per-step incremental pass.

Each is covered below with a stub model that returns a container *only* at the
site under test, so a missing unwrap fails that test alone. The auxiliary head
peaks at a different token than the main head, which is what makes "index 0 was
used" observable in the generated ids.
"""

import types
import unittest

import paddle

from paddlefleet.generation.greedy_generator import GreedyGenerator

MAIN_TOKEN = 7  # argmax of the main head
AUX_TOKEN = 11  # argmax of every auxiliary (MTP) head
VOCAB_SIZE = 16
NUM_LAYERS = 2


class _MTPStubModel(paddle.nn.Layer):
    """Minimal stand-in for a Fleet MTP model.

    ``forward`` ignores the hidden states entirely and emits constant logits:
    the main head peaks at ``MAIN_TOKEN``, auxiliary heads at ``AUX_TOKEN``.
    Whether the return value is wrapped in a container is decided per call site
    via ``container_by_kind``, so a single stub can isolate any of the three
    ``isinstance(logits, (list, tuple))`` branches.

    Call sites are labelled:

    * ``"no_cache"`` -- ``use_cache=False``
    * ``"prefill"``  -- first ``use_cache=True`` call
    * ``"decode"``   -- every later ``use_cache=True`` call
    """

    def __init__(self, container_by_kind: dict, num_heads: int = 2):
        super().__init__()
        self.config = types.SimpleNamespace(
            num_hidden_layers=NUM_LAYERS,
            vocab_size=VOCAB_SIZE,
            sequence_parallel=False,
            apply_rope_fusion=False,
        )
        self.container_by_kind = container_by_kind
        self.num_heads = num_heads
        self.calls: list[str] = []

    def _kind(self, use_cache: bool) -> str:
        if not use_cache:
            return "no_cache"
        return "prefill" if "prefill" not in self.calls else "decode"

    def _head_logits(self, bsz: int, seq_len: int, peak: int):
        logits = paddle.zeros([bsz, seq_len, VOCAB_SIZE], dtype="float32")
        logits[:, :, peak] = 10.0
        return logits

    def forward(self, dict_args):
        input_ids = dict_args["input_ids"]
        bsz, seq_len = input_ids.shape
        kind = self._kind(bool(dict_args.get("use_cache", False)))
        self.calls.append(kind)

        # Keep the cache's sequence bookkeeping advancing, otherwise the decode
        # loop would hand every step ``position_ids=0``.
        cache = dict_args.get("past_key_values")
        if cache is not None and dict_args.get("use_cache", False):
            for layer_idx in range(NUM_LAYERS):
                cache.update(
                    paddle.zeros([bsz, seq_len, 1, 1], dtype="float32"),
                    paddle.zeros([bsz, seq_len, 1, 1], dtype="float32"),
                    layer_idx,
                )

        main = self._head_logits(bsz, seq_len, MAIN_TOKEN)
        container = self.container_by_kind.get(kind)
        if container is None:
            return main
        heads = [main] + [
            self._head_logits(bsz, seq_len, AUX_TOKEN)
            for _ in range(self.num_heads - 1)
        ]
        return container(heads)


def _input_ids(bsz: int = 1, prompt_len: int = 3):
    return paddle.arange(bsz * prompt_len, dtype="int64").reshape(
        [bsz, prompt_len]
    )


def _generated_tokens(out, prompt_len: int) -> list[int]:
    return out[0, prompt_len:].tolist()


class _MTPBranchTestMixin:
    """Shared assertions: main head wins, aux head is never sampled."""

    def _run(self, container_by_kind, max_new_tokens, no_cache, num_heads=2):
        model = _MTPStubModel(container_by_kind, num_heads=num_heads)
        gen = GreedyGenerator(model)
        input_ids = _input_ids()
        out = gen.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=None,
            no_cache=no_cache,
        )
        return model, out, input_ids.shape[1]

    def _assert_main_head_sampled(self, out, prompt_len, n_expected):
        tokens = _generated_tokens(out, prompt_len)
        self.assertEqual(len(tokens), n_expected)
        self.assertEqual(tokens, [MAIN_TOKEN] * n_expected)
        self.assertNotIn(AUX_TOKEN, tokens)


class TestNoCacheLogitsList(_MTPBranchTestMixin, unittest.TestCase):
    """``_generate_no_cache``: greedy_generator.py:559."""

    def test_list_unwrapped_to_main_head(self):
        model, out, prompt_len = self._run(
            {"no_cache": list}, max_new_tokens=3, no_cache=True
        )
        self._assert_main_head_sampled(out, prompt_len, 3)
        self.assertEqual(model.calls, ["no_cache"] * 3)

    def test_tuple_unwrapped_to_main_head(self):
        _, out, prompt_len = self._run(
            {"no_cache": tuple}, max_new_tokens=3, no_cache=True
        )
        self._assert_main_head_sampled(out, prompt_len, 3)

    def test_three_heads_still_picks_index_zero(self):
        _, out, prompt_len = self._run(
            {"no_cache": list}, max_new_tokens=2, no_cache=True, num_heads=3
        )
        self._assert_main_head_sampled(out, prompt_len, 2)

    def test_matches_plain_tensor_output(self):
        """Wrapping the logits must not change what gets generated."""
        _, wrapped, _ = self._run(
            {"no_cache": list}, max_new_tokens=3, no_cache=True
        )
        _, plain, _ = self._run({}, max_new_tokens=3, no_cache=True)
        self.assertEqual(wrapped.tolist(), plain.tolist())


class TestPrefillLogitsList(_MTPBranchTestMixin, unittest.TestCase):
    """``generate`` prefill: greedy_generator.py:775.

    ``max_new_tokens=1`` runs the prefill only (the decode loop is
    ``range(0)``), so the prefill branch is exercised in isolation.
    """

    def test_list_unwrapped_to_main_head(self):
        model, out, prompt_len = self._run(
            {"prefill": list}, max_new_tokens=1, no_cache=False
        )
        self._assert_main_head_sampled(out, prompt_len, 1)
        self.assertEqual(model.calls, ["prefill"])

    def test_tuple_unwrapped_to_main_head(self):
        _, out, prompt_len = self._run(
            {"prefill": tuple}, max_new_tokens=1, no_cache=False
        )
        self._assert_main_head_sampled(out, prompt_len, 1)

    def test_prefill_container_does_not_leak_into_decode(self):
        """Only the prefill returns a list; decode returns a bare tensor."""
        model, out, prompt_len = self._run(
            {"prefill": list}, max_new_tokens=3, no_cache=False
        )
        self._assert_main_head_sampled(out, prompt_len, 3)
        self.assertEqual(model.calls, ["prefill", "decode", "decode"])

    def test_log_probs_collected_from_main_head(self):
        """``return_log_probs`` reads the unwrapped logits too."""
        model = _MTPStubModel({"prefill": list})
        gen = GreedyGenerator(model)
        input_ids = _input_ids()
        out, log_probs = gen.generate(
            input_ids,
            max_new_tokens=1,
            eos_token_id=None,
            return_log_probs=True,
            logprob_start_len=0,
        )
        self.assertEqual(len(log_probs), 1)
        # positions 1..prompt_len-1 (prompt) + the single generated token
        self.assertEqual(len(log_probs[0]), input_ids.shape[1])
        self.assertEqual(
            _generated_tokens(out, input_ids.shape[1]), [MAIN_TOKEN]
        )


class TestDecodeLogitsList(_MTPBranchTestMixin, unittest.TestCase):
    """``generate`` decode: greedy_generator.py:862."""

    def test_list_unwrapped_to_main_head(self):
        model, out, prompt_len = self._run(
            {"decode": list}, max_new_tokens=3, no_cache=False
        )
        self._assert_main_head_sampled(out, prompt_len, 3)
        self.assertEqual(model.calls, ["prefill", "decode", "decode"])

    def test_tuple_unwrapped_to_main_head(self):
        _, out, prompt_len = self._run(
            {"decode": tuple}, max_new_tokens=3, no_cache=False
        )
        self._assert_main_head_sampled(out, prompt_len, 3)

    def test_every_decode_step_unwraps(self):
        model, out, prompt_len = self._run(
            {"prefill": list, "decode": list}, max_new_tokens=5, no_cache=False
        )
        self._assert_main_head_sampled(out, prompt_len, 5)
        self.assertEqual(model.calls.count("decode"), 4)

    def test_eos_stops_after_first_decode_step(self):
        """EOS handling sees main-head tokens (MAIN_TOKEN is the eos here).

        Prefill does not check eos, so exactly one decode step runs and then
        the loop breaks -- proof the decode branch fed a real tensor to the
        eos comparison rather than a container.
        """
        model = _MTPStubModel({"prefill": list, "decode": list})
        gen = GreedyGenerator(model)
        input_ids = _input_ids()
        out = gen.generate(input_ids, max_new_tokens=8, eos_token_id=MAIN_TOKEN)
        self.assertEqual(model.calls, ["prefill", "decode"])
        self.assertEqual(
            _generated_tokens(out, input_ids.shape[1]), [MAIN_TOKEN] * 2
        )


if __name__ == "__main__":
    unittest.main()

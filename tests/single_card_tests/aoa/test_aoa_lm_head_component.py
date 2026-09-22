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
# Scope: an LM head's AOA contract. The live tree names the head after the role
# it plays, so this test pins: all three spellings resolve to the one checkpoint
# name the mapping declares; the head carries no ``^T``, unlike the rest of its
# Linear family; ``bias`` and the multimax extras go through the very same
# machinery as ``weight``; a tied head is filled from the embedding's tensor and
# writes nothing back; and when the two paired roles hold one tensor the MTP
# head is the one that writes it while the main head discards.
import os
import sys

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)

import unittest

import paddle
from paddle.distributed.flex_checkpoint.aoa.generation import AOAContext

# The three entries a real model declares for this boundary: the head's own two
# sit outside the checkpoint root (the ``ForCausalLM`` layout keeps ``lm_head`` a
# top-level sibling of the backbone), and the third is the entry the embedding
# leaf goes through, which is what a tied head has to follow.
_MAPPING = {
    "model.lm_head.weight": "lm_head.weight",
    "model.lm_head.bias": "lm_head.bias",
    "model.embedding.embed_tokens.weight": "model.embed_tokens.weight",
}

# Every name the live tree gives the head: its own, the one pipeline weight
# sharing splits it off under, and the one an MTP head of its own gets.
_ALIASES = ("lm_head", "shared_head", "shared_mtp_lm_head")


class _Cfg:
    def __init__(self, *, tie_word_embeddings=False):
        self.tie_word_embeddings = tie_word_embeddings


def _ctx(*, tie_word_embeddings=False):
    return AOAContext(
        config=_Cfg(tie_word_embeddings=tie_word_embeddings),
        checkpoint_name_prefix="model",
        pp_to_single_mapping={},
        checkpoint_name_mapping=_MAPPING,
        model_name_prefix="model",
    )


def _make_head(cls):
    """A structural stand-in for an LM head of the given class.

    Building the real head needs a live config plus TP process groups, so this
    subclass keeps the class (hence both AOA overrides, the canonical-segment
    collapse they share, and the writer flag) and only the tensors that decide
    which statements come out.
    """

    class _HeadStandIn(cls):
        def __init__(self):
            paddle.nn.Layer.__init__(self)
            self.weight = self.create_parameter(shape=[2, 2])
            self.bias = self.create_parameter(shape=[2])
            self.multimax_ranges = self.create_parameter(shape=[4])

    return _HeadStandIn()


def _main_head():
    from paddlefleet.models.gpt.lm_head import GPTMainLMHead

    return _make_head(GPTMainLMHead)


def _mtp_head():
    from paddlefleet.models.gpt.lm_head import GPTMTPLMHead

    return _make_head(GPTMTPLMHead)


def _fwd(head, alias, **kwargs):
    return head.gen_aoa_statements(
        _ctx(**kwargs), structured_name_prefix=f"model.{alias}."
    )


def _inv(head, alias, **kwargs):
    return head.gen_inv_aoa_statements(
        _ctx(**kwargs), structured_name_prefix=f"model.{alias}."
    )


class TestHeadClassContract(unittest.TestCase):
    def test_the_head_overrides_both_directions(self):
        from paddlefleet.models.gpt.lm_head import GPTLMHead

        self.assertIn("gen_aoa_statements", GPTLMHead.__dict__)
        self.assertIn("gen_inv_aoa_statements", GPTLMHead.__dict__)

    def test_both_roles_share_the_overrides_and_differ_only_on_writing(self):
        from paddlefleet.models.gpt.lm_head import (
            GPTLMHead,
            GPTMainLMHead,
            GPTMTPLMHead,
        )

        for cls in (GPTMainLMHead, GPTMTPLMHead):
            self.assertIs(cls.gen_aoa_statements, GPTLMHead.gen_aoa_statements)
            self.assertIs(
                cls.gen_inv_aoa_statements, GPTLMHead.gen_inv_aoa_statements
            )
        # The two roles only ever come as a pair, and the MTP one is the head
        # that is guaranteed to exist: both are registered under the same shared
        # pipeline key, whose earlier desc wins per rank, and the MTP desc comes
        # first. So it is the writer and the main head gives its copy up.
        self.assertFalse(GPTMainLMHead._aoa_writes_checkpoint_tensor)
        self.assertTrue(GPTMTPLMHead._aoa_writes_checkpoint_tensor)
        # A head that is alone in the tree writes, so the base default must not
        # be the given-up one.
        self.assertTrue(GPTLMHead._aoa_writes_checkpoint_tensor)


class TestHeadForward(unittest.TestCase):
    def test_every_role_alias_reads_the_one_checkpoint_tensor(self):
        for alias in _ALIASES:
            self.assertIn(
                f"lm_head.weight -> model.{alias}.weight",
                _fwd(_main_head(), alias),
            )

    def test_bias_goes_through_the_same_machinery_as_weight(self):
        for alias in _ALIASES:
            self.assertIn(
                f"lm_head.bias -> model.{alias}.bias",
                _fwd(_main_head(), alias),
            )

    def test_the_multimax_extras_follow_the_collapse_unmapped(self):
        # No entry of their own, so the identity fallback keeps them under the
        # backbone root -- but the collapse still applies, so an aliased head
        # reads them from the head's canonical name rather than its live one.
        self.assertIn(
            "model.lm_head.multimax_ranges "
            "-> model.shared_head.multimax_ranges",
            _fwd(_main_head(), "shared_head"),
        )
        # Under the canonical spelling the two names coincide, and an identical
        # statement is what the engine's same-name passthrough already does.
        self.assertIn(
            "model.lm_head.multimax_ranges -> model.lm_head.multimax_ranges",
            _fwd(_main_head(), "lm_head"),
        )

    def test_no_transpose_anywhere(self):
        # Unlike the rest of the ColumnParallelLinear family: the head stores
        # ``weight`` in the checkpoint's own layout and transposes in forward.
        for alias in _ALIASES:
            self.assertFalse(any("^T" in s for s in _fwd(_main_head(), alias)))

    def test_a_non_writing_head_still_reads_everything(self):
        # Two heads holding one tensor both get filled; only writing is elected.
        self.assertEqual(
            _fwd(_main_head(), "shared_head"),
            _fwd(_mtp_head(), "shared_head"),
        )

    def test_the_statement_set_is_exactly_the_head_tensors(self):
        self.assertEqual(
            _fwd(_main_head(), "shared_head"),
            [
                "lm_head.weight -> model.shared_head.weight",
                "lm_head.bias -> model.shared_head.bias",
                "model.lm_head.multimax_ranges "
                "-> model.shared_head.multimax_ranges",
            ],
        )


class TestHeadInverse(unittest.TestCase):
    def test_the_writer_exports_every_tensor_under_the_collapsed_name(self):
        self.assertEqual(
            _inv(_mtp_head(), "shared_mtp_lm_head"),
            [
                "model.shared_mtp_lm_head.weight -> lm_head.weight",
                "model.shared_mtp_lm_head.bias -> lm_head.bias",
                "model.shared_mtp_lm_head.multimax_ranges "
                "-> model.lm_head.multimax_ranges",
            ],
        )

    def test_a_non_writing_head_discards_rather_than_exports(self):
        # Without a statement the identity passthrough would export the head's
        # live keys, giving the checkpoint a second copy of one tensor.
        stmts = _inv(_main_head(), "shared_head")
        self.assertTrue(all(s.endswith(" -> _") for s in stmts))
        self.assertEqual(len(stmts), 3)

    def test_no_transpose_anywhere(self):
        # Read off the writer, whose statements name a checkpoint tensor on the
        # right; a discarded head's ``-> _`` would carry no ``^T`` either way.
        for alias in _ALIASES:
            self.assertFalse(any("^T" in s for s in _inv(_mtp_head(), alias)))


class TestTiedHead(unittest.TestCase):
    def test_the_weight_is_filled_from_the_embedding_tensor(self):
        # The checkpoint holds no head tensor at all, and the one it does hold
        # is named by the entry the embedding leaf itself goes through.
        self.assertIn(
            "model.embed_tokens.weight -> model.shared_head.weight",
            _fwd(_main_head(), "shared_head", tie_word_embeddings=True),
        )

    def test_the_extras_come_from_nothing(self):
        # They are not in the checkpoint either, mirroring the inverse
        # direction, which writes none of them.
        stmts = _fwd(_main_head(), "shared_head", tie_word_embeddings=True)
        self.assertIn("_ -> model.shared_head.bias", stmts)
        self.assertIn("_ -> model.shared_head.multimax_ranges", stmts)

    def test_nothing_is_written_back(self):
        # The embedding already wrote that tensor.
        for head in (_main_head(), _mtp_head()):
            stmts = _inv(head, "shared_head", tie_word_embeddings=True)
            self.assertTrue(all(s.endswith(" -> _") for s in stmts))
            self.assertEqual(len(stmts), 3)

    def test_the_head_never_names_a_checkpoint_tensor_of_its_own(self):
        stmts = _fwd(
            _main_head(), "shared_head", tie_word_embeddings=True
        ) + _inv(_main_head(), "shared_head", tie_word_embeddings=True)
        self.assertFalse(any("lm_head.weight ->" in s for s in stmts))


if __name__ == "__main__":
    unittest.main()

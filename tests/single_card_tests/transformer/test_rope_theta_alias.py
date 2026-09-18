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

"""Migration tests for the ``rotary_base`` -> ``rope_theta`` alias (B2).

``rope_theta`` is the single RoPE base field; ``rotary_base`` is its
deprecated alias. ``TransformerConfig.__post_init__`` rejects the historical
silent split where a config dual-writes both fields with different values --
including the migration case where only the old alias is set to a non-default
value while ``rope_theta`` still sits on its dataclass default (10000.0),
which would otherwise silently keep two different bases alive.
"""

import pytest

from paddlefleet.transformer.transformer_config import TransformerConfig


def _make_config(**kwargs):
    return TransformerConfig(**kwargs)


def test_default_neither_set():
    config = _make_config()
    assert config.rope_theta == 10000.0
    assert config.rotary_base is None
    assert "rotary_base" not in vars(config)


def test_alias_equal_to_rope_theta_accepted_and_stripped():
    config = _make_config(rotary_base=10000)
    assert config.rope_theta == 10000.0
    # The alias is removed from the instance dict so __dict__-based
    # serialization (HF to_dict / vars) no longer dual-writes the field.
    assert "rotary_base" not in vars(config)


def test_dual_write_conflict_raises():
    with pytest.raises(
        ValueError, match="please set rope_theta and delete rotary_base"
    ):
        _make_config(rope_theta=10000.0, rotary_base=100000)


def test_only_alias_set_with_default_rope_theta_raises():
    """Migration case: only the old alias is set and the values differ.

    ``rope_theta`` is left at its dataclass default (10000.0) while the alias
    carries a different base. This is a dual-write inequality on the default
    path and must raise rather than silently keep two bases: rejecting it
    forces the legacy config to be rewritten to ``rope_theta`` explicitly.
    """
    with pytest.raises(
        ValueError, match="rope_theta=10000.0, rotary_base=500000"
    ):
        _make_config(rotary_base=500000)

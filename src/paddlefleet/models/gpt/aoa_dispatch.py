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
"""Consumer-side AOA dispatch: pick hand-written vs modular whole-model AOA.

Kept out of :mod:`aoa_generator` (the modular generator) so that module stays
direction-independent -- the ``aoa_config_reverse`` derive-from-forward fallback
below is a hand-written-model concern, not part of modular generation.
"""

import logging

from paddle.nn import Layer

logger = logging.getLogger(__name__)


def resolve_aoa_config(model, config=None):
    """Consumer-side dispatch for the checkpoint -> model (load) direction.

    Hand-written models attach ``_gen_aoa_config``; a modular model instead
    overrides the whole-model entry :meth:`GPTModel.gen_aoa_statements`.
    ``gen_aoa_statements`` exists on every ``nn.Layer`` (the recursion
    protocol), so its mere presence cannot flag a modular model -- an override
    of the base method does.
    """
    if hasattr(model, "_gen_aoa_config"):
        return model._gen_aoa_config(config)
    if type(model).gen_aoa_statements is not Layer.gen_aoa_statements:
        return model.gen_aoa_statements(config)
    raise RuntimeError(
        "model provides neither a hand-written _gen_aoa_config nor a modular "
        "whole-model gen_aoa_statements override."
    )


def resolve_inv_aoa_config(model, config=None):
    """Consumer-side dispatch for the model -> checkpoint (save) direction.

    Mirror of :func:`resolve_aoa_config`, with one extra fallback: a model that
    only defines the forward ``_gen_aoa_config`` (no hand-written inverse) has
    its inverse auto-derived by running the forward config in reverse.
    """
    if hasattr(model, "_gen_inv_aoa_config"):
        return model._gen_inv_aoa_config(config)
    if hasattr(model, "_gen_aoa_config"):
        logger.warning(
            "There is no _gen_inv_aoa_config, so we auto-derived it from _gen_aoa_config."
        )
        # Shallow-copy rather than mutate in place: some models hand back a
        # shared, closure-captured dict (e.g. the EC/HF bridge lambda in
        # ernie5/pretrain.py returns the same _hf_aoa every call), so an
        # in-place reverse flag would poison later forward loads.
        return {
            **model._gen_aoa_config(config),
            "aoa_config_reverse": True,
        }
    if type(model).gen_inv_aoa_statements is not Layer.gen_inv_aoa_statements:
        return model.gen_inv_aoa_statements(config)
    raise RuntimeError(
        "model provides neither a hand-written _gen_inv_aoa_config / _gen_aoa_config "
        "nor a modular whole-model gen_inv_aoa_statements override."
    )

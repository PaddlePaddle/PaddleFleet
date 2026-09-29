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

import paddle
import paddle.distributed as dist
from typing import Optional, TYPE_CHECKING, Union
if TYPE_CHECKING:
    from collections.abc import Callable


def in_auto_parallel_mode() -> bool:
    return False


@dataclass
class MixedPrecisionPolicy:
    param_dtype: Optional[paddle.dtype] = None
    reduce_dtype: Optional[paddle.dtype] = None
    output_dtype: Optional[paddle.dtype] = None
    cast_forward_inputs: bool = True


@dataclass
class OffloadPolicy:
    pin_memory: bool = True


def _fully_shard_manual_parallel(
    module,
    mesh,
    reshard_after_forward,
    shard_placement_fn,
    mp_policy,
    offload_policy,
    ignored_params,
):
    # TODO: Implement manual parallel logic
    pass


def _fully_shard_auto_parallel(
    module,
    mesh,
    reshard_after_forward,
    shard_placement_fn,
    mp_policy,
    offload_policy,
    ignored_params,
):
    # TODO: Implement auto parallel logic
    pass


def fully_shard(
    module: paddle.nn.Layer,
    *,
    mesh: dist.ProcessMesh,
    reshard_after_forward: Optional[Union[bool, int]] = None,
    shard_placement_fn: Optional[Callable[[paddle.Tensor],
                                          Optional[dist.Shard]]] = None,
    mp_policy: Optional[MixedPrecisionPolicy] = None,
    offload_policy: Optional[OffloadPolicy] = None,
    ignored_params: Optional[Iterable[paddle.Tensor]] = None,
) -> paddle.nn.Layer:
    if mp_policy is None:
        mp_policy = MixedPrecisionPolicy()
    if offload_policy is None:
        offload_policy = OffloadPolicy()
    ignored_params_set: Set[paddle.Tensor] = set(
        ignored_params) if ignored_params else set()

    args = (
        module,
        mesh,
        reshard_after_forward,
        shard_placement_fn,
        mp_policy,
        offload_policy,
        ignored_params_set,
    )
    if in_auto_parallel_mode():
        return _fully_shard_auto_parallel(*args)
    else:
        return _fully_shard_manual_parallel(*args)

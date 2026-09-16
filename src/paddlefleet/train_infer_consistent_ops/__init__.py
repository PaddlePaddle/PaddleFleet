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

"""train_infer_consistent_inspect tooling, one module per operator family.

* `inspect_util` -- the `inspect_tensor` probe itself plus the current-layer
  context it stamps dumps with.
* `permute` -- expert-contiguous row order (`canonical_rows` and its inverse
  `scatter_canonical_rows`).
* `ffn_act` -- the fused SwiGLU+fp8-quant activation: unit routing weights and
  the blockwise dequant back to bf16.
* `slice_util` -- one segment of the last dim (`last_dim_segment` and its inverse
  `scatter_last_dim_segment`), for the MLA nope / RoPE split.

`inspect_tensor` is the only probe the network definition calls to compare a
tensor; it takes an `index` when the tensor travels inside a tuple / list / dict,
and the layout bridges above go in through its `pre_save_func` /
`post_load_func`. The remaining `inspect_tensor_*` entry points only publish
context or change what the model computes, so grepping that prefix still lists
the whole surface. A module reused in several roles takes the role as an
`inspect_name` constructor argument (default `moe_shared`) and prefixes its own
tags with it: `inspect_tensor(f"{self.inspect_name}_ffn1_output", ...)` -- so the
role is fixed at build time and the forward only reads it.

Everything here is diagnostic. With `ABLATION_INSPECT_TENSOR` unset every entry
point returns immediately without touching the network, so the network
definition can call them unconditionally -- no `if inspect_enabled():` at the
call site. Work that would
have to happen *before* a probe (dequant, row gather, a slice of the last dim)
goes in through `inspect_tensor(..., pre_save_func=...)` so it is skipped too --
including the write-back, which a `post_load_func` turns into a fresh tensor
instead of an in-place edit of the live buffer.
"""

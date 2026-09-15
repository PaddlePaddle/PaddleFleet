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

"""HyperBody 多模输入 <-> 纯 Tensor 打包工具（fleet pipeline 兼容）。

背景
----
fleet pipeline 的 micro-batch loader（``paddle/.../pipeline_parallel.py``）只接受
**Tensor/None** 字段（对每个字段逐个 ``.detach()`` / 切片），不能传 python list。
而 HyperBody 的 packed 多模输入按段是 list::

    image          = [Tensor(C, H, W) | None, ...]   # 每个 context 段一个(变尺寸)
    audio          = [Tensor(M, L)    | None, ...]
    use_long_query = [bool, ...]                      # 每个 decoder 段一个

本模块把它们打包成"纯 Tensor + 元信息"，过完 loader 后再无损恢复成原 list。

元信息设计（ragged/jagged tensor，自描述，模态无关）
----------------------------------------------------
对一个"每段可选张量"的 list，用 3 个 Tensor 表示::

    {prefix}_values    : 1-D, 原 dtype, 所有存在张量 flatten 后拼接
    {prefix}_shapes    : int64 [K, R], 每个存在张量的形状(R=该模态 rank, 自描述)
    {prefix}_seg_index : int32 [S], 第 s 段 -> 存在列表下标 k, 无则 -1

K=存在张量数, S=段数; 每段字节偏移由 prod(shapes) 累加得到, 不单独存(最小冗余)。
空 list: values=[0] float32, shapes=[0,0], seg_index 全 -1。

调用点
------
* 数据侧(erniebot ``_hyperbody_hack_inputs`` / 未来真实 collator):
  ``batch.update(pack_hyperbody_mm(image, audio, use_long_query))`` 后删除原 list 键,
  使 batch 只剩 Tensor, 可进 fleet pipeline。
* 组网侧(``HyperBodyEncoderFrontEnd.forward`` 顶部):
  ``unpack_hyperbody_mm(dict_args)`` 原地还原 image/audio/use_long_query 并清理打包键,
  之后既有 list-based frontend 逻辑完全不变。
"""

from __future__ import annotations

import paddle

__all__ = [
    "pack_optional_tensor_list",
    "unpack_optional_tensor_list",
    "pack_use_long_query",
    "unpack_use_long_query",
    "pack_hyperbody_mm",
    "unpack_hyperbody_mm",
    "HB_IMAGE_PREFIX",
    "HB_AUDIO_PREFIX",
    "HB_USE_LONG_QUERY_KEY",
]

HB_IMAGE_PREFIX = "hb_image"
HB_AUDIO_PREFIX = "hb_audio"
HB_USE_LONG_QUERY_KEY = "hb_use_long_query"


def pack_optional_tensor_list(items, prefix):
    """list[Tensor|None] -> {prefix}_values/_shapes/_seg_index 三个 Tensor。"""
    seg_index = [-1] * len(items)
    flats, shapes, dtype, k = [], [], None, 0
    for s, t in enumerate(items):
        if t is None:
            continue
        if dtype is None:
            dtype = t.dtype
        seg_index[s] = k
        shapes.append([int(d) for d in t.shape])
        flats.append(t.reshape([-1]).astype(dtype))
        k += 1
    if k == 0:
        values = paddle.zeros([0], dtype="float32")
        shapes_t = paddle.zeros([0, 0], dtype="int64")
    else:
        values = paddle.concat(flats, axis=0)
        shapes_t = paddle.to_tensor(shapes, dtype="int64")
    return {
        f"{prefix}_values": values,
        f"{prefix}_shapes": shapes_t,
        f"{prefix}_seg_index": paddle.to_tensor(seg_index, dtype="int32"),
    }


def unpack_optional_tensor_list(packed, prefix):
    """还原 pack_optional_tensor_list 的结果 -> list[Tensor|None]（长度 S）。"""
    values = packed[f"{prefix}_values"]
    shapes_t = packed[f"{prefix}_shapes"]
    seg_idx = packed[
        f"{prefix}_seg_index"
    ].tolist()  # 小 int 张量, host 侧控制流
    present, off = [], 0
    if int(shapes_t.shape[0]) > 0:
        for shp in shapes_t.tolist():
            shp = [int(x) for x in shp]
            n = 1
            for d in shp:
                n *= d
            present.append(values[off : off + n].reshape(shp))
            off += n
    return [present[i] if i >= 0 else None for i in seg_idx]


def pack_use_long_query(flags):
    """list[bool] -> int32 Tensor [S] (1=long, 0=short)。"""
    return paddle.to_tensor([1 if f else 0 for f in flags], dtype="int32")


def unpack_use_long_query(t):
    """int32 Tensor [S] -> list[bool]。"""
    return [bool(x) for x in t.tolist()]


def pack_hyperbody_mm(image, audio, use_long_query):
    """把 HyperBody 的三个 list 字段打包成纯 Tensor dict（便于并入 batch）。"""
    packed = {}
    packed.update(pack_optional_tensor_list(image, HB_IMAGE_PREFIX))
    packed.update(pack_optional_tensor_list(audio, HB_AUDIO_PREFIX))
    packed[HB_USE_LONG_QUERY_KEY] = pack_use_long_query(use_long_query)
    return packed


def unpack_hyperbody_mm(dict_args):
    """原地还原 image/audio/use_long_query 并清理打包键；无打包键时原样返回。"""
    if f"{HB_IMAGE_PREFIX}_seg_index" not in dict_args:
        return dict_args
    dict_args["image"] = unpack_optional_tensor_list(dict_args, HB_IMAGE_PREFIX)
    dict_args["audio"] = unpack_optional_tensor_list(dict_args, HB_AUDIO_PREFIX)
    dict_args["use_long_query"] = unpack_use_long_query(
        dict_args[HB_USE_LONG_QUERY_KEY]
    )
    for key in (
        f"{HB_IMAGE_PREFIX}_values",
        f"{HB_IMAGE_PREFIX}_shapes",
        f"{HB_IMAGE_PREFIX}_seg_index",
        f"{HB_AUDIO_PREFIX}_values",
        f"{HB_AUDIO_PREFIX}_shapes",
        f"{HB_AUDIO_PREFIX}_seg_index",
        HB_USE_LONG_QUERY_KEY,
    ):
        dict_args.pop(key, None)
    return dict_args

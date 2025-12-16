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

from paddle import _C_ops
from paddle.base.layer_helper import LayerHelper
from paddle.framework import in_dynamic_or_pir_mode
from paddle.jit.marker import unified


@unified
def tokens_zip_unique_add(x_zipped, x_unzipped, idx_unzipped, zipped_rows):
    # The output variable's dtype use default value 'float32',
    # and the actual dtype of output variable will be inferred in runtime.
    if in_dynamic_or_pir_mode():
        outs = _C_ops._run_custom_op(
            "tokens_zip_unique_add",
            x_zipped,
            x_unzipped,
            idx_unzipped,
            zipped_rows,
        )
        res = []
        start_idx = 0
        res.append(outs[start_idx])
        start_idx += 1
        return res[0] if len(res) == 1 else res
    else:
        ins = {}
        ins_map = {
            "x_zipped": x_zipped,
            "x_unzipped": x_unzipped,
            "idx_unzipped": idx_unzipped,
        }
        outs = {}
        outs_list = ["y_zipped"]
        for key, value in ins_map.items():
            # handle optional inputs
            if value is not None:
                ins[key] = value
        helper = LayerHelper("tokens_zip_unique_add", **locals())

        outs["y_zipped"] = helper.create_variable(dtype="float32")
        helper.append_op(
            type="tokens_zip_unique_add",
            inputs=ins,
            outputs=outs,
            attrs={"zipped_rows": zipped_rows},
        )
        res = [
            outs[out_name] if out_name in outs.keys() else None
            for out_name in outs_list
        ]
        return res[0] if len(res) == 1 else res


from paddle.jit.marker import unified


@unified
def tokens_unzip_gather(
    x,
    x_scale,
    zipped_expertwise_rowmap,
    expert_id,
    tokens_per_expert,
    padding_multiplex,
):
    # The output variable's dtype use default value 'float32',
    # and the actual dtype of output variable will be inferred in runtime.
    if in_dynamic_or_pir_mode():
        outs = _C_ops._run_custom_op(
            "tokens_unzip_gather",
            x,
            x_scale,
            zipped_expertwise_rowmap,
            expert_id,
            tokens_per_expert,
            padding_multiplex,
        )
        res = []
        start_idx = 0
        res.append(outs[start_idx])
        start_idx += 1
        res.append(outs[start_idx])
        start_idx += 1
        res.append(outs[start_idx])
        start_idx += 1
        return res[0] if len(res) == 1 else res
    else:
        ins = {}
        ins_map = {
            "x": x,
            "x_scale@OPTIONAL": x_scale,
            "zipped_expertwise_rowmap": zipped_expertwise_rowmap,
        }
        outs = {}
        outs_list = ["x_unzipped", "x_scale_unzipped@OPTIONAL", "idx_unzipped"]
        for key, value in ins_map.items():
            # handle optional inputs
            if value is not None:
                ins[key] = value
        helper = LayerHelper("tokens_unzip_gather", **locals())

        outs["x_unzipped"] = helper.create_variable(dtype="float32")
        outs["x_scale_unzipped@OPTIONAL"] = helper.create_variable(
            dtype="float32"
        )
        outs["idx_unzipped"] = helper.create_variable(dtype="float32")
        helper.append_op(
            type="tokens_unzip_gather",
            inputs=ins,
            outputs=outs,
            attrs={
                "expert_id": expert_id,
                "tokens_per_expert": tokens_per_expert,
                "padding_multiplex": padding_multiplex,
            },
        )
        res = [
            outs[out_name] if out_name in outs.keys() else None
            for out_name in outs_list
        ]
        return res[0] if len(res) == 1 else res


from paddle.jit.marker import unified


@unified
def merge_subbatch_cast(x, dtype):
    # The output variable's dtype use default value 'float32',
    # and the actual dtype of output variable will be inferred in runtime.
    if in_dynamic_or_pir_mode():
        outs = _C_ops._run_custom_op("merge_subbatch_cast", x, dtype)
        res = []
        start_idx = 0
        res.append(outs[start_idx])
        start_idx += 1
        return res[0] if len(res) == 1 else res
    else:
        ins = {}
        ins_map = {"x@VECTOR": x}
        outs = {}
        outs_list = ["y"]
        for key, value in ins_map.items():
            # handle optional inputs
            if value is not None:
                ins[key] = value
        helper = LayerHelper("merge_subbatch_cast", **locals())

        outs["y"] = helper.create_variable(dtype="float32")
        helper.append_op(
            type="merge_subbatch_cast",
            inputs=ins,
            outputs=outs,
            attrs={"dtype": dtype},
        )
        res = [
            outs[out_name] if out_name in outs.keys() else None
            for out_name in outs_list
        ]
        return res[0] if len(res) == 1 else res


from paddle.jit.marker import unified


@unified
def fused_swiglu_scale_bwd(x, scale, dout):
    # The output variable's dtype use default value 'float32',
    # and the actual dtype of output variable will be inferred in runtime.
    if in_dynamic_or_pir_mode():
        outs = _C_ops._run_custom_op("fused_swiglu_scale_bwd", x, scale, dout)
        res = []
        start_idx = 0
        res.append(outs[start_idx])
        start_idx += 1
        res.append(outs[start_idx])
        start_idx += 1
        return res[0] if len(res) == 1 else res
    else:
        ins = {}
        ins_map = {"X": x, "Scale": scale, "DOut": dout}
        outs = {}
        outs_list = ["DX", "DScale"]
        for key, value in ins_map.items():
            # handle optional inputs
            if value is not None:
                ins[key] = value
        helper = LayerHelper("fused_swiglu_scale_bwd", **locals())

        outs["DX"] = helper.create_variable(dtype="float32")
        outs["DScale"] = helper.create_variable(dtype="float32")
        helper.append_op(
            type="fused_swiglu_scale_bwd", inputs=ins, outputs=outs, attrs={}
        )
        res = [
            outs[out_name] if out_name in outs.keys() else None
            for out_name in outs_list
        ]
        return res[0] if len(res) == 1 else res


from paddle.jit.marker import unified


@unified
def tokens_unzip_stable(
    x,
    xscale,
    expert_routemap_topk,
    expert_prob_topk,
    topk,
    num_experts,
    tokens_per_expert,
    padding_multiplex,
    fill_output,
):
    # The output variable's dtype use default value 'float32',
    # and the actual dtype of output variable will be inferred in runtime.
    if in_dynamic_or_pir_mode():
        outs = _C_ops._run_custom_op(
            "tokens_unzip_stable",
            x,
            xscale,
            expert_routemap_topk,
            expert_prob_topk,
            topk,
            num_experts,
            tokens_per_expert,
            padding_multiplex,
            fill_output,
        )
        res = []
        start_idx = 0
        res.append(outs[start_idx])
        start_idx += 1
        res.append(outs[start_idx])
        start_idx += 1
        res.append(outs[start_idx])
        start_idx += 1
        res.append(outs[start_idx])
        start_idx += 1
        return res[0] if len(res) == 1 else res
    else:
        ins = {}
        ins_map = {
            "X": x,
            "Xscale@OPTIONAL": xscale,
            "expert_routemap_topk": expert_routemap_topk,
            "expert_prob_topk": expert_prob_topk,
        }
        outs = {}
        outs_list = [
            "X_unzipped",
            "zipped_expertwise_rowmap",
            "token_prob_unzipped",
            "XScale_unzipped@OPTIONAL",
        ]
        for key, value in ins_map.items():
            # handle optional inputs
            if value is not None:
                ins[key] = value
        helper = LayerHelper("tokens_unzip_stable", **locals())

        outs["X_unzipped"] = helper.create_variable(dtype="float32")
        outs["zipped_expertwise_rowmap"] = helper.create_variable(
            dtype="float32"
        )
        outs["token_prob_unzipped"] = helper.create_variable(dtype="float32")
        outs["XScale_unzipped@OPTIONAL"] = helper.create_variable(
            dtype="float32"
        )
        helper.append_op(
            type="tokens_unzip_stable",
            inputs=ins,
            outputs=outs,
            attrs={
                "topk": topk,
                "num_experts": num_experts,
                "tokens_per_expert": tokens_per_expert,
                "padding_multiplex": padding_multiplex,
                "fill_output": fill_output,
            },
        )
        res = [
            outs[out_name] if out_name in outs.keys() else None
            for out_name in outs_list
        ]
        return res[0] if len(res) == 1 else res


from paddle.jit.marker import unified


@unified
def fused_swiglu_scale(x, scale):
    # The output variable's dtype use default value 'float32',
    # and the actual dtype of output variable will be inferred in runtime.
    if in_dynamic_or_pir_mode():
        outs = _C_ops._run_custom_op("fused_swiglu_scale", x, scale)
        res = []
        start_idx = 0
        res.append(outs[start_idx])
        start_idx += 1
        return res[0] if len(res) == 1 else res
    else:
        ins = {}
        ins_map = {"X": x, "Scale": scale}
        outs = {}
        outs_list = ["Out"]
        for key, value in ins_map.items():
            # handle optional inputs
            if value is not None:
                ins[key] = value
        helper = LayerHelper("fused_swiglu_scale", **locals())

        outs["Out"] = helper.create_variable(dtype="float32")
        helper.append_op(
            type="fused_swiglu_scale", inputs=ins, outputs=outs, attrs={}
        )
        res = [
            outs[out_name] if out_name in outs.keys() else None
            for out_name in outs_list
        ]
        return res[0] if len(res) == 1 else res


from paddle.jit.marker import unified


@unified
def tokens_zip_prob_seq_subbatch(
    unzipped_prob, zipped_expertwise_rowmap, dispatched_indices, subbatch_rows
):
    # The output variable's dtype use default value 'float32',
    # and the actual dtype of output variable will be inferred in runtime.
    if in_dynamic_or_pir_mode():
        outs = _C_ops._run_custom_op(
            "tokens_zip_prob_seq_subbatch",
            unzipped_prob,
            zipped_expertwise_rowmap,
            dispatched_indices,
            subbatch_rows,
        )
        res = []
        start_idx = 0
        res.append(outs[start_idx])
        start_idx += 1
        return res[0] if len(res) == 1 else res
    else:
        ins = {}
        ins_map = {
            "unzipped_prob@VECTOR": unzipped_prob,
            "zipped_expertwise_rowmap": zipped_expertwise_rowmap,
            "dispatched_indices": dispatched_indices,
        }
        outs = {}
        outs_list = ["zipped_prob"]
        for key, value in ins_map.items():
            # handle optional inputs
            if value is not None:
                ins[key] = value
        helper = LayerHelper("tokens_zip_prob_seq_subbatch", **locals())

        outs["zipped_prob"] = helper.create_variable(dtype="float32")
        helper.append_op(
            type="tokens_zip_prob_seq_subbatch",
            inputs=ins,
            outputs=outs,
            attrs={"subbatch_rows": subbatch_rows},
        )
        res = [
            outs[out_name] if out_name in outs.keys() else None
            for out_name in outs_list
        ]
        return res[0] if len(res) == 1 else res


from paddle.jit.marker import unified


@unified
def tokens_zip_prob(
    unzipped_prob, zipped_expertwise_rowmap, dispatched_indices
):
    # The output variable's dtype use default value 'float32',
    # and the actual dtype of output variable will be inferred in runtime.
    if in_dynamic_or_pir_mode():
        outs = _C_ops._run_custom_op(
            "tokens_zip_prob",
            unzipped_prob,
            zipped_expertwise_rowmap,
            dispatched_indices,
        )
        res = []
        start_idx = 0
        res.append(outs[start_idx])
        start_idx += 1
        return res[0] if len(res) == 1 else res
    else:
        ins = {}
        ins_map = {
            "unzipped_prob@VECTOR": unzipped_prob,
            "zipped_expertwise_rowmap": zipped_expertwise_rowmap,
            "dispatched_indices": dispatched_indices,
        }
        outs = {}
        outs_list = ["zipped_prob"]
        for key, value in ins_map.items():
            # handle optional inputs
            if value is not None:
                ins[key] = value
        helper = LayerHelper("tokens_zip_prob", **locals())

        outs["zipped_prob"] = helper.create_variable(dtype="float32")
        helper.append_op(
            type="tokens_zip_prob", inputs=ins, outputs=outs, attrs={}
        )
        res = [
            outs[out_name] if out_name in outs.keys() else None
            for out_name in outs_list
        ]
        return res[0] if len(res) == 1 else res


from paddle.jit.marker import unified


@unified
def tokens_zip_unique_add_subbatch(
    x_zipped, x_unzipped, idx_unzipped, zipped_rows, subbatch_rows
):
    # The output variable's dtype use default value 'float32',
    # and the actual dtype of output variable will be inferred in runtime.
    if in_dynamic_or_pir_mode():
        outs = _C_ops._run_custom_op(
            "tokens_zip_unique_add_subbatch",
            x_zipped,
            x_unzipped,
            idx_unzipped,
            zipped_rows,
            subbatch_rows,
        )
        res = []
        start_idx = 0
        res.append(outs[start_idx : start_idx + len(x_zipped)])
        start_idx += len(x_zipped)
        return res[0] if len(res) == 1 else res
    else:
        ins = {}
        ins_map = {
            "x_zipped@VECTOR": x_zipped,
            "x_unzipped": x_unzipped,
            "idx_unzipped": idx_unzipped,
        }
        outs = {}
        outs_list = ["y_zipped@VECTOR"]
        for key, value in ins_map.items():
            # handle optional inputs
            if value is not None:
                ins[key] = value
        helper = LayerHelper("tokens_zip_unique_add_subbatch", **locals())

        outs["y_zipped@VECTOR"] = x_zipped
        helper.append_op(
            type="tokens_zip_unique_add_subbatch",
            inputs=ins,
            outputs=outs,
            attrs={"zipped_rows": zipped_rows, "subbatch_rows": subbatch_rows},
        )
        res = [
            outs[out_name] if out_name in outs.keys() else None
            for out_name in outs_list
        ]
        return res[0] if len(res) == 1 else res


from paddle.jit.marker import unified


@unified
def tokens_unzip_slice(
    x,
    zipped_expertwise_rowmap,
    num_experts,
    total_unzipped_rows,
    start_idx,
    end_idx,
):
    # The output variable's dtype use default value 'float32',
    # and the actual dtype of output variable will be inferred in runtime.
    if in_dynamic_or_pir_mode():
        outs = _C_ops._run_custom_op(
            "tokens_unzip_slice",
            x,
            zipped_expertwise_rowmap,
            num_experts,
            total_unzipped_rows,
            start_idx,
            end_idx,
        )
        res = []
        start_idx = 0
        res.append(outs[start_idx])
        start_idx += 1
        return res[0] if len(res) == 1 else res
    else:
        ins = {}
        ins_map = {"x": x, "zipped_expertwise_rowmap": zipped_expertwise_rowmap}
        outs = {}
        outs_list = ["idx_unzipped"]
        for key, value in ins_map.items():
            # handle optional inputs
            if value is not None:
                ins[key] = value
        helper = LayerHelper("tokens_unzip_slice", **locals())

        outs["idx_unzipped"] = helper.create_variable(dtype="float32")
        helper.append_op(
            type="tokens_unzip_slice",
            inputs=ins,
            outputs=outs,
            attrs={
                "num_experts": num_experts,
                "total_unzipped_rows": total_unzipped_rows,
                "start_idx": start_idx,
                "end_idx": end_idx,
            },
        )
        res = [
            outs[out_name] if out_name in outs.keys() else None
            for out_name in outs_list
        ]
        return res[0] if len(res) == 1 else res


import importlib.abc
import importlib.util
import os
import sys
import types

import paddle

cur_dir = os.path.dirname(os.path.abspath(__file__))
so_path = os.path.join(cur_dir, "ops_pd_.so")


def __bootstrap__():
    assert os.path.exists(so_path)
    # load custom op shared library with abs path
    custom_ops = paddle.utils.cpp_extension.load_op_meta_info_and_register_op(
        so_path
    )

    if os.name == "nt" or sys.platform.startswith("darwin"):
        # Cpp Extension only support Linux now
        mod = types.ModuleType(__name__)
    else:
        try:
            spec = importlib.util.spec_from_file_location(__name__, so_path)
            assert spec is not None
            mod = importlib.util.module_from_spec(spec)
            assert isinstance(spec.loader, importlib.abc.Loader)
            spec.loader.exec_module(mod)
        except ImportError:
            mod = types.ModuleType(__name__)

    for custom_op in custom_ops:
        setattr(mod, custom_op, eval(custom_op))


__bootstrap__()

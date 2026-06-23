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

"""
GLM4.5-Air 精度对齐工具函数 - Paddle (PaddleFleet) 侧

集中放置 PF/MG 跨框架对齐用的 tensor dump / md5 打印 / 梯度 hook /
强制对齐 hook / spike capture 等辅助函数, 便于 paddlefleet 内各模块统一从这里 import,
避免之前 sys.path hack 散落在各处。

环境变量 (默认全 0, 不影响原始训练):
  GLM_ALIGN_BIT_EXACT  逻辑级 bit-exact 对齐开关 (deterministic unpermute / fp64 sum / bf16 matmul 等)
  GLM_ALIGN_LOG        持续性插桩 (cp* tensor & grad info)
  GLM_ALIGN_DUMP_DATA  一次性 dump (输入数据 md5/shape, 初始权重 md5/norm)

打印过滤控制(SAVE_TENSOR_NAMES 是逗号分隔关键字, 不设则全部打印):
  SAVE_TENSOR_NAMES    只保存名称含指定关键字的 checkpoint, 例如 "cp12b,cp12c"

迁移自:
  /root/paddlejob/share-storage/gpfs/system-public/zhanghonggeng/glm_45/save_tensor_paddle.py
"""

import hashlib as _hashlib
import os

import numpy as np
import paddle


# ==================== GLM 精度对齐三档总开关 ====================
def is_bit_exact() -> bool:
    """逻辑级 bit-exact 对齐开关 (默认关)"""
    return os.environ.get("GLM_ALIGN_BIT_EXACT", "0") == "1"


def is_log_enabled() -> bool:
    """持续性插桩开关 (cp* tensor/grad info; 默认关)"""
    return os.environ.get("GLM_ALIGN_LOG", "0") == "1"


def is_dump_data_enabled() -> bool:
    """一次性 dump 开关 (输入数据/初始权重 md5/norm; 默认关)"""
    return os.environ.get("GLM_ALIGN_DUMP_DATA", "0") == "1"


def _is_enabled(name, subdir):
    """根据环境变量白名单判断是否需要保存(PF 侧目前不做过滤, 保留接口形状一致)"""
    return True


def _pf_tensor_info(name, tensor, layer_num=None, prefix="PF MoE"):
    """打印 PF 侧 tensor 信息(表格格式), 受 GLM_ALIGN_LOG 与 SAVE_TENSOR_NAMES 过滤控制"""
    if not is_log_enabled():
        return
    allowed_names = os.environ.get("SAVE_TENSOR_NAMES", "")
    if allowed_names and not any(
        k.strip() in name for k in allowed_names.split(",")
    ):
        return
    layer_str = f"L{layer_num}:" if layer_num is not None else ""
    label = f"{prefix}:{layer_str}{name}"
    if tensor is None:
        print(f"| {label:<40s} | {'None':<16s} | {'N/A':<20s} | {'N/A':<20s} |")
        return
    data = tensor.detach().cast("float32").numpy()
    md5 = _hashlib.md5(data.tobytes()).hexdigest()[:16]
    shape_str = str(list(tensor.shape))
    dtype_str = str(tensor.dtype)
    print(
        f"| {label:<40s} | {md5:<16s} | {shape_str:<20s} | {dtype_str:<20s} |"
    )


def _pf_grad_info(name, layer_num=None, prefix="GRAD PF"):
    """
    返回一个 hook, backward 时打印梯度的 md5/shape/dtype/abs_mean/abs_max 信息。
    用法: paddle.autograd.backward 中注册

    打印格式与 _mg_grad_info 一致, 方便两侧对比。
    受 GLM_ALIGN_LOG 控制: 关闭时返回的 hook 直接透传, 不做任何打印。
    """
    if not is_log_enabled():

        def _noop_hook(grad):
            return grad

        return _noop_hook

    def hook(grad):
        if grad is None:
            return
        layer_str = f"L{layer_num}:" if layer_num is not None else ""
        label = f"{prefix}:{layer_str}{name}"
        data = grad.detach().cast("float32").numpy()
        md5 = _hashlib.md5(data.tobytes()).hexdigest()[:16]
        shape_str = str(list(grad.shape))
        dtype_str = str(grad.dtype)
        abs_mean = float(np.abs(data).mean())
        abs_max = float(np.abs(data).max())
        # 黄色高亮 GRAD 行
        print(
            f"\033[33m| {label:<40s} | md5={md5} | {shape_str:<20s} | {dtype_str:<14s} | "
            f"abs_mean={abs_mean:.6e} | abs_max={abs_max:.6e} |\033[0m"
        )
        return grad

    return hook


def _print_tensor_info(tensor, name, subdir, fname, tag="Saved"):
    """打印张量的详细信息"""
    shape = tuple(tensor.shape)
    dtype = str(tensor.dtype)
    print(f"  📌 [PF][{subdir}] {tag}: {fname}  shape={shape}, dtype={dtype}")


def save_tensor(tensor, name, subdir, layer_idx=None):
    """[NO-SAVE] 仅打印 tensor 信息, 不再写 npy. 保留签名以兼容老调用方."""
    if tensor is None:
        return
    if not _is_enabled(name, subdir):
        return
    if layer_idx is not None:
        fname = f"{name}_{layer_idx}.npy"
    else:
        fname = f"{name}.npy"
    _print_tensor_info(tensor, name, subdir, fname, tag="Print(no-save)")


def save_tensor_grad(name, subdir, layer_idx=None):
    """[NO-SAVE] 返回 no-op hook, 不再写 npy. 保留签名以兼容老调用方."""

    def hook(grad):
        return grad

    return hook


def force_align_from_npy(tensor, name, base_path, layer_id=None):
    """
    从 npy 文件加载数据并替换 tensor 的值, 用于跳过不对齐层直接注入对齐数据。
    """
    if layer_id is not None:
        fname = f"{name}_{layer_id}.npy"
    else:
        fname = f"{name}.npy"

    fpath = os.path.join(base_path, fname)
    if not os.path.exists(fpath):
        print(f"  ⚠️ [PF] force_align: file not found: {fpath}")
        return tensor

    np_data = np.load(fpath)
    aligned = paddle.to_tensor(np_data, dtype="float32").cast(tensor.dtype)
    aligned = aligned.reshape(tensor.shape)
    print(
        f"  🔄 [PF] force_align: {fname} -> shape={list(aligned.shape)}, dtype={aligned.dtype}"
    )
    return aligned


def create_force_align_grad_hook(name, base_path, layer_id=None):
    """
    返回一个 hook, backward 时用 npy 文件中的梯度替换当前梯度。
    用于梯度注入对齐。
    """

    def hook(grad):
        if grad is None:
            return grad
        if layer_id is not None:
            fname = f"{name}_{layer_id}.npy"
        else:
            fname = f"{name}.npy"

        fpath = os.path.join(base_path, fname)
        if not os.path.exists(fpath):
            print(f"  ⚠️ [PF] force_align_grad: file not found: {fpath}")
            return grad

        np_data = np.load(fpath)
        aligned_grad = paddle.to_tensor(np_data, dtype="float32").cast(
            grad.dtype
        )
        aligned_grad = aligned_grad.reshape(grad.shape)
        print(
            f"  🔄 [PF] force_align_grad: {fname} -> shape={list(aligned_grad.shape)}"
        )
        return aligned_grad

    return hook


# ==================== Step-indexed force align (for backward alignment) ====================
# 每个 (name, subdir, layer_id) 的 hook 触发次数计数器, 按 step 读取对应 npy
# 命名约定与 align_dump_utils.save_tensor_grad_step_indexed 对齐:
#   {name}_L{layer_id}_step{N}.npy
_force_align_counters = {}


def create_force_align_grad_hook_step_indexed(
    name, base_path, layer_id=None, verbose=True, fallback_keep_grad=True
):
    """
    返回一个 hook, 每次 backward 触发时按调用次数读取 step{N} 的 npy 替换梯度。

    用法(PF 侧):
        gates.register_hook(
            create_force_align_grad_hook_step_indexed(
                "cp11c_scores_after_sigmoid_grad",
                "/root/.../diff_test/mg/router_align",
                layer_id=2,  # 注意: PF L1 对应 MG L2, 需要传 MG 侧的 layer_id
            )
        )
    """
    key = (name, base_path, layer_id)
    _force_align_counters.setdefault(key, 0)

    def hook(grad):
        if grad is None:
            return grad
        _force_align_counters[key] += 1
        step_n = _force_align_counters[key]

        layer_part = f"_L{layer_id}" if layer_id is not None else ""
        fname = f"{name}{layer_part}_step{step_n}.npy"
        fpath = os.path.join(base_path, fname)

        if not os.path.exists(fpath):
            print(
                f"  ⚠️ [PF] force_align_step_indexed: file not found: {fpath} (step={step_n})"
            )
            if fallback_keep_grad:
                return grad
            return grad

        np_data = np.load(fpath)
        aligned_grad = paddle.to_tensor(np_data, dtype="float32").cast(
            grad.dtype
        )
        aligned_grad = aligned_grad.reshape(grad.shape)
        if verbose:
            print(
                f"  🔄 [PF] force_align_step_indexed: {fname} -> shape={list(aligned_grad.shape)} "
                f"dtype={aligned_grad.dtype}"
            )
        return aligned_grad

    return hook


def create_force_align_and_print_grad_hook_step_indexed(
    name,
    base_path,
    layer_id=None,
    print_layer_num=None,
    prefix="GRAD PF Router(ALIGNED)",
):
    """
    组合 hook: 先用 npy 文件中的梯度替换当前梯度, 再以 _pf_grad_info 同样的格式打印替换后的 grad。
    避免与已有 print hook 注册顺序歧义。
    """
    key = ("align_and_print", name, base_path, layer_id)
    _force_align_counters.setdefault(key, 0)

    def hook(grad):
        if grad is None:
            return grad
        _force_align_counters[key] += 1
        step_n = _force_align_counters[key]

        layer_part = f"_L{layer_id}" if layer_id is not None else ""
        fname = f"{name}{layer_part}_step{step_n}.npy"
        fpath = os.path.join(base_path, fname)

        if not os.path.exists(fpath):
            print(
                f"  ⚠️ [PF] force_align_and_print: file not found: {fpath} (step={step_n})"
            )
            data = grad.detach().cast("float32").numpy()
            md5 = _hashlib.md5(data.tobytes()).hexdigest()[:16]
            layer_str = (
                f"L{print_layer_num}:" if print_layer_num is not None else ""
            )
            label = f"{prefix}(NO_ALIGN_FILE):{layer_str}{name}"
            abs_mean = float(np.abs(data).mean())
            abs_max = float(np.abs(data).max())
            print(
                f"\033[31m| {label:<40s} | md5={md5} | {list(grad.shape)!s:<20s} | "
                f"{grad.dtype!s:<14s} | abs_mean={abs_mean:.6e} | abs_max={abs_max:.6e} |\033[0m"
            )
            return grad

        np_data = np.load(fpath)
        aligned_grad = paddle.to_tensor(np_data, dtype="float32").cast(
            grad.dtype
        )
        aligned_grad = aligned_grad.reshape(grad.shape)

        # 打印替换后的梯度(绿色高亮, 便于在 log 中区分对齐过的 grad)
        data = aligned_grad.detach().cast("float32").numpy()
        md5 = _hashlib.md5(data.tobytes()).hexdigest()[:16]
        layer_str = (
            f"L{print_layer_num}:" if print_layer_num is not None else ""
        )
        label = f"{prefix}:{layer_str}{name}(step{step_n})"
        abs_mean = float(np.abs(data).mean())
        abs_max = float(np.abs(data).max())
        print(
            f"\033[32m| {label:<40s} | md5={md5} | {list(aligned_grad.shape)!s:<20s} | "
            f"{aligned_grad.dtype!s:<14s} | abs_mean={abs_mean:.6e} | abs_max={abs_max:.6e} |\033[0m"
        )
        return aligned_grad

    return hook


# ==================== 跨框架 forward/grad dump (仅打印 md5/shape) ====================
_pf_dump_fwd_counter = {}
_pf_dump_grad_counter = {}


def _pf_dump_to_np(t):
    if t is None:
        return None
    return t.detach().cast("float32").numpy()


def pf_dump_forward(tag_layer, tensors, scalars=None):
    """forward 入参 dump (仅打印 md5/shape, 不写盘). tag_layer 形如 'auxloss_LPF'."""
    if not is_log_enabled():
        return 0
    key = ("pf", tag_layer)
    _pf_dump_fwd_counter[key] = _pf_dump_fwd_counter.get(key, 0) + 1
    step = _pf_dump_fwd_counter[key]
    for name, t in tensors.items():
        arr = _pf_dump_to_np(t)
        if arr is None:
            continue
        md5 = _hashlib.md5(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]
        print(
            f"  [DUMP PF] forward step{step} {tag_layer}/{name} "
            f"md5={md5} shape={tuple(arr.shape)}",
            flush=True,
        )
    return step


def pf_dump_grad_hook(tag_layer, name):
    """返回 backward hook (仅打印 md5/shape, 不写盘). GLM_ALIGN_LOG=0 时返回 no-op hook."""
    if not is_log_enabled():

        def _noop(grad):
            return grad

        return _noop
    key = ("pf", tag_layer, name)

    def hook(grad):
        if grad is None:
            return grad
        _pf_dump_grad_counter[key] = _pf_dump_grad_counter.get(key, 0) + 1
        step = _pf_dump_grad_counter[key]
        arr = _pf_dump_to_np(grad)
        md5 = _hashlib.md5(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]
        print(
            f"  [DUMP PF] grad {tag_layer}/{name} step{step} "
            f"md5={md5} shape={tuple(arr.shape)}",
            flush=True,
        )
        return grad

    return hook


# ==================== Embed I/O probe (centralized, PF side, 仅打印) ====================
# 入侵 VocabParallelEmbedding.forward, 打印 weight + ids + dy + wgrad 的 md5/norm
_pf_embed_step = {"pf": 0}


def _pf_embed_md5(arr_np):
    return _hashlib.md5(np.ascontiguousarray(arr_np).tobytes()).hexdigest()[:16]


def _pf_embed_print_row(label, arr):
    md5 = _pf_embed_md5(arr)
    af32 = arr.astype(np.float32)
    norm = float(np.linalg.norm(af32))
    am = float(np.abs(af32).mean())
    mx = float(np.abs(af32).max())
    print(
        f"\033[36m[EMBED PROBE] {label:<55s} md5={md5} shape={tuple(arr.shape)} "
        f"norm={norm:.6f} abs_mean={am:.4e} abs_max={mx:.4e}\033[0m",
        flush=True,
    )
    return md5


def pf_dump_embed_io(weight, masked_input, output, weight_param=None):
    """PF VocabParallelEmbedding.forward 调用一次 (仅打印, 不写盘).
    forward 立即打印 weight+ids; 给 output 注册 dy hook, 给 weight_param 注册 wgrad hook.
    GLM_ALIGN_LOG=0 时 no-op.
    """
    if not is_log_enabled():
        return
    _pf_embed_step["pf"] += 1
    n = _pf_embed_step["pf"]

    w_np = weight.detach().cast("float32").numpy()
    ids_np = masked_input.detach().cast("int64").numpy()
    _pf_embed_print_row(f"pf step{n} weight", w_np)
    _pf_embed_print_row(f"pf step{n} ids   ", ids_np)

    def _dy_hook(grad):
        if grad is None:
            return grad
        try:
            dy_np = grad.detach().cast("float32").numpy()
            _pf_embed_print_row(f"pf step{n} dy    ", dy_np)
        except Exception as e:
            print(f"[EMBED PROBE] pf step{n} dy hook fail: {e}", flush=True)
        return grad

    try:
        output.register_hook(_dy_hook)
    except Exception as e:
        print(
            f"[EMBED PROBE] pf step{n} register output hook fail: {e}",
            flush=True,
        )

    if weight_param is not None:

        def _wgrad_hook(grad):
            if grad is None:
                return grad
            try:
                wg_np = grad.detach().cast("float32").numpy()
                _pf_embed_print_row(f"pf step{n} wgrad ", wg_np)
            except Exception as e:
                print(
                    f"[EMBED PROBE] pf step{n} wgrad hook fail: {e}", flush=True
                )
            return grad

        try:
            weight_param.register_hook(_wgrad_hook)
        except Exception as e:
            print(
                f"[EMBED PROBE] pf step{n} register wgrad hook fail: {e}",
                flush=True,
            )


# ==================== Optimizer state probe (GLM_ALIGN_OPTIM_PROBE) ====================
# 与 GLM_ALIGN_LOG 解耦, 单独控制 optimizer 前/后状态对齐打印, 与 MG 侧
# mg_dump_optim_probe_print 输出格式一致, 便于跨框架 md5/norm 对比。
#
# Paddle 不像 Megatron 把 cast 拆成单独一步 (_copy_main_params_to_model_params),
# AdamW python 实现里 master_weight[:] = p ; param[:] = p.astype(param.dtype) 是
# 原子的, 所以这里只暴露 pre / post 两个 phase, 不再单独打 mid。
_optim_probe_step_pf = {"pf": 0}


def is_optim_probe_enabled() -> bool:
    """Optimizer 前后状态对齐打印开关 (默认关)"""
    return os.environ.get("GLM_ALIGN_OPTIM_PROBE", "0") == "1"


def _optim_probe_print_row_pf(label, arr):
    if arr is None:
        print(f"\033[35m[OPTIM PROBE] {label:<60s} None\033[0m", flush=True)
        return
    md5 = _hashlib.md5(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]
    af32 = arr.astype(np.float32)
    norm = float(np.linalg.norm(af32))
    am = float(np.abs(af32).mean())
    mx = float(np.abs(af32).max())
    print(
        f"\033[35m[OPTIM PROBE] {label:<60s} md5={md5} shape={tuple(arr.shape)} "
        f"norm={norm:.6f} abs_mean={am:.4e} abs_max={mx:.4e}\033[0m",
        flush=True,
    )


def _pf_iter_optim_params(optimizer):
    """从 (可能经多层包装的) optimizer 里取出所有 param。
    顺序兼容: HybridParallelOptimizer -> MixPrecisionOptimizer -> AdamW。
    返回 (params_list, inner_adamw) ; inner_adamw 用于取 _master_weights / 累加器。
    """
    inner = optimizer
    # 穿透到底层 AdamW: MixPrecisionOptimizer / HybridParallelOptimizer 都暴露 _inner_opt
    for _ in range(4):
        if hasattr(inner, "_inner_opt") and inner._inner_opt is not None:
            inner = inner._inner_opt
        else:
            break

    plist = getattr(optimizer, "_parameter_list", None)
    if plist is None:
        plist = getattr(inner, "_parameter_list", None)

    params = []
    if plist is not None and len(plist) > 0:
        if isinstance(plist[0], dict):
            for g in plist:
                params.extend(g.get("params", []))
        else:
            params = list(plist)
    elif hasattr(inner, "_param_groups"):
        for g in inner._param_groups:
            if isinstance(g, dict):
                params.extend(g.get("params", []))
            else:
                params.append(g)
    return params, inner


def pf_dump_optim_probe_print(phase, optimizer):
    """
    在 PF 侧 optimizer.step 不同 phase 调用一次, 紫色 [OPTIM PROBE] 行输出
    md5/shape/norm/abs_mean/abs_max, 与 MG 侧 mg_dump_optim_probe_print 同源。

    Args:
        phase: "pre" | "post"
            pre  = optimizer.step() 前 (master_weight fp32 + main_grad fp32 + moments)
            post = optimizer.step() 后 (master_weight fp32 已更新, working bf16 已 cast)
        optimizer: trainer.self.optimizer (任意层包装均可)
    """
    if not is_optim_probe_enabled():
        return
    if phase == "pre":
        _optim_probe_step_pf["pf"] += 1
    n = _optim_probe_step_pf["pf"]

    try:
        params, inner = _pf_iter_optim_params(optimizer)
    except Exception as e:
        print(f"[OPTIM PROBE PF] iter params fail: {e}", flush=True)
        return

    master_weights = getattr(inner, "_master_weights", None) or {}
    moment1_str = getattr(inner, "_moment1_acc_str", "moment1")
    moment2_str = getattr(inner, "_moment2_acc_str", "moment2")
    get_acc = getattr(inner, "_get_accumulator_master", None)

    for pi, p in enumerate(params):
        if getattr(p, "stop_gradient", False):
            continue
        tag = f"p{pi}"
        # working param (原 dtype, 一般 bf16)
        try:
            wp_np = p.detach().contiguous().cast("float32").numpy()
            _optim_probe_print_row_pf(
                f"pf step{n} {phase} {tag} working    ", wp_np
            )
        except Exception as e:
            print(f"[OPTIM PROBE PF] {tag} working fail: {e}", flush=True)
        # master weight (fp32)
        pname = getattr(p, "name", None)
        mw = master_weights.get(pname) if pname else None
        if mw is not None:
            try:
                mw_np = mw.detach().contiguous().cast("float32").numpy()
                _optim_probe_print_row_pf(
                    f"pf step{n} {phase} {tag} master     ", mw_np
                )
            except Exception as e:
                print(f"[OPTIM PROBE PF] {tag} master fail: {e}", flush=True)
        # main_grad (fp32, 仅 pre 有意义)
        if (
            phase == "pre"
            and hasattr(p, "main_grad")
            and p.main_grad is not None
        ):
            try:
                g_np = p.main_grad.detach().contiguous().cast("float32").numpy()
                _optim_probe_print_row_pf(
                    f"pf step{n} {phase} {tag} main_grad  ", g_np
                )
            except Exception as e:
                print(f"[OPTIM PROBE PF] {tag} grad fail: {e}", flush=True)
        # moments (Adam exp_avg / exp_avg_sq)
        if get_acc is not None:
            for sk_label, sk in (
                ("exp_avg     ", moment1_str),
                ("exp_avg_sq  ", moment2_str),
            ):
                try:
                    m = get_acc(sk, p)
                    if m is not None:
                        m_np = m.detach().contiguous().cast("float32").numpy()
                        _optim_probe_print_row_pf(
                            f"pf step{n} {phase} {tag} {sk_label}", m_np
                        )
                except Exception as e:
                    print(f"[OPTIM PROBE PF] {tag} {sk} fail: {e}", flush=True)

    print(f"  [OPTIM PROBE PF] phase={phase} step{n} printed", flush=True)

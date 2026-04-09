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

__all__ = []

# ============================================================
# FA3: C++/CUDA compiled extension (requires paddle + .so)
# ============================================================
_fa3_available = False
try:
    import os as _os
    import paddle

    _so_loaded = False

    # 尝试从已安装的模块中加载
    try:
        import flash_mask as _flash_mask_module
        _so_path = _flash_mask_module.__file__
        if _so_path and _so_path.endswith('.so'):
            paddle.utils.cpp_extension.load_op_meta_info_and_register_op(_so_path)
            _so_loaded = True
    except Exception:
        pass

    # 如果还没加载，尝试从 build 目录加载
    if not _so_loaded:
        _curr_dir = _os.path.dirname(_os.path.abspath(__file__))
        _parent_dir = _os.path.dirname(_curr_dir)
        _possible_paths = [
            _os.path.join(_parent_dir, "build", "flash_mask",
                          "lib.linux-x86_64-cpython-310", "flash_mask.so"),
            _os.path.join(_parent_dir, "flash_mask.so"),
        ]
        for _so_path in _possible_paths:
            if _os.path.exists(_so_path):
                paddle.utils.cpp_extension.load_op_meta_info_and_register_op(_so_path)
                _so_loaded = True
                break

    if _so_loaded:
        from .flashmask_attention_v3.interface import flashmask_attention as flashmask_attention_v3
        __all__.append("flashmask_attention_v3")
        _fa3_available = True
    else:
        print("[WARNING] flash_mask.so not found, FA3 custom ops not available")
except ImportError:
    pass  # paddle not installed, skip FA3

# ============================================================
# FA4: Pure Python + CUTLASS DSL (no .so needed)
# ============================================================
_fa4_available = False
try:
    from .cute import flash_attention, flashmask_attention
    __all__ += ["flash_attention", "flashmask_attention"]
    _fa4_available = True
except ImportError:
    pass  # cute module not installed or dependencies missing

if not _fa3_available and not _fa4_available:
    print("[WARNING] flash_mask: neither FA3 nor FA4 is available. "
          "Check your installation.")

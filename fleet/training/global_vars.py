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

"""PaddleFleet global variables."""

_GLOBAL_ARGS = None


def get_args():
    """Return arguments."""
    _ensure_var_is_initialized(_GLOBAL_ARGS, "args")
    return _GLOBAL_ARGS


def set_args(args):
    global _GLOBAL_ARGS
    _GLOBAL_ARGS = args


def _ensure_var_is_initialized(var, name):
    """Make sure the input variable is not None."""
    assert var is not None, f"{name} is not initialized."


def _ensure_var_is_not_initialized(var, name):
    """Make sure the input variable is not None."""
    assert var is None, f"{name} is already initialized."


def destroy_global_vars():
    global _GLOBAL_ARGS
    _GLOBAL_ARGS = None


def set_global_variables(args):
    """Set args, etc."""

    assert args is not None

    _ensure_var_is_not_initialized(_GLOBAL_ARGS, "args")
    set_args(args)


def unset_global_variables():
    global _GLOBAL_ARGS
    _GLOBAL_ARGS = None

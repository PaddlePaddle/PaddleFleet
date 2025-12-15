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

import importlib
import inspect
import logging


def import_custom_ops(package, module_name, global_ns):
    """
    Imports custom operations from a specified module within a package and adds them to a global namespace.

    Args:
        package (str): The name of the package containing the module.
        module_name (str): The name of the module within the package.
        global_ns (dict): The global namespace to add the imported functions to.
    """
    try:
        module = importlib.import_module(module_name, package=package)
        functions = inspect.getmembers(module)
        for func_name, func in functions:
            if func_name.startswith("__") or func_name == "_C_ops":
                continue
            logging.debug(f"Import {func_name} from {package}")
            try:
                global_ns[func_name] = func
            except Exception as e:
                logging.warning(f"Failed to import op {func_name}: {e}")

    except Exception as e:
        logging.warning(
            f"Ops of {package} import failed, it may be not compiled."
        )

# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import os
import sys
from contextlib import suppress
from typing import TYPE_CHECKING

from . import (
    parallel_state as parallel_state,
    training as training,
    version as version,
)
from .package_info import (
    __contact_emails__,
    __contact_names__,
    __description__,
    __download_url__,
    __homepage__,
    __keywords__,
    __license__,
    __package_name__,
    __repository_url__,
    __version__,
)
from .timers import Timers
from .utils.lazy_import import _LazyModule

from paddlefleet.utils.log import logger

mpu = parallel_state

__all__ = [
    "training",
    "parallel_state",
    "Timers",
    "__contact_emails__",
    "__contact_names__",
    "__description__",
    "__download_url__",
    "__homepage__",
    "__keywords__",
    "__license__",
    "__package_name__",
    "__repository_url__",
    "__version__",
]

try:
    from importlib import metadata
except ImportError:
    import importlib_metadata as metadata


def compare_version(v1, v2):
    for a, b in zip(v1.split("."), v2.split(".")):
        if a.isnumeric() and b.isnumeric():
            if a != b:
                return 1 if int(a) > int(b) else -1
        else:
            return 1 if a.isnumeric() else -1
    return 0


def _check_dependency_versions():
    for pkg_names, min_version in [(["paddlepaddle-gpu", "paddlepaddle"], "3.3")]:
        for pkg_name in pkg_names:
            try:
                _version = metadata.version(pkg_name)
                if compare_version(_version, min_version) < 0:
                    logger.warning(
                        "Version check warning:\n" + f"{pkg_name} version {_version}, recommended >= {min_version}"
                    )
            except:
                pass


_check_dependency_versions()


with suppress(Exception):
    import paddle

    from .utils.paddle_patch import *

    paddle.disable_signal_handler()

PADDLEFLEET_TESTING = os.environ.get("PADDLEFLEET_TESTING", False)
_disabled_optional_modules = ["torchcodec"]
if "torch" not in sys.modules and not PADDLEFLEET_TESTING:
    _disabled_optional_modules.extend(["torch", "torchvision"])
_disabled_optional_modules = [
    name for name in _disabled_optional_modules if name not in sys.modules
]
try:
    for name in _disabled_optional_modules:
        sys.modules[name] = None
    import transformers  # qa
finally:
    for name in _disabled_optional_modules:
        sys.modules.pop(name, None)

logger.warning(
    """Due to potential compatibility issues between PaddlePaddle and PyTorch in PaddleFleet, PaddleFleet defaults `transformers.utils.import_utils.is_torch_available` and `transformers.utils.import_utils.is_torchvision_available` to False. If you need to use PyTorch in transformers or torchvision, please add `del sys.modules['transformers']` before using them."""
)

if "datasets" in sys.modules.keys():

    logger.warning(
        "Detected that datasets module was imported before paddlefleet. "
        "This may cause PaddleFleet datasets to be unavailable in intranet. "
        "Please import paddlefleet before datasets module to avoid download issues"
    )

# module index
modules = [
    "cli",
    "data",
    "datasets",
    "generation",
    "package_info",
    "parallel_state",
    "nn",
    "mergekit",
    "ops",
    "peft",
    "quantization",
    "trainer",
    "trl",
    "utils",
    "version",
    "transformers",
    "training",
    "timers",
]

import_structure = {module: [] for module in modules}
import_structure["package_info"] = [
    "__contact_emails__",
    "__contact_names__",
    "__description__",
    "__download_url__",
    "__homepage__",
    "__keywords__",
    "__license__",
    "__package_name__",
    "__repository_url__",
]
import_structure["timers"] = ["Timers"]
import_structure["transformers.tokenizer_utils"] = ["PreTrainedTokenizer"]

if TYPE_CHECKING:
    from . import datasets  # noqa
    from . import transformers  # noqa
    from . import (
        cli,
        data,
        generation,
        mergekit,
        nn,
        ops,
        package_info,
        parallel_state,
        peft,
        quantization,
        trainer,
        training,
        trl,
        utils,
        version,
    )
    from .timers import Timers
else:
    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        import_structure,
        module_spec=__spec__,
        extra_objects={
            "__all__": __all__,
            "mpu": mpu,
            **{name: globals()[name] for name in __all__},
        },
    )

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

from setuptools import build_meta as orig

from build_utils import Artifact, EcosystemLibrary

LIBRARIES: list[EcosystemLibrary] = [
    EcosystemLibrary(
        name="DeepGEMM",
        source_rel_path="third_party/DeepGEMM",
        artifacts=[
            # Updated paths to point to the installation directory
            Artifact("deep_gemm", "deep_gemm"),
            Artifact("deep_gemm_cpp", "deep_gemm_cpp"),
        ],
    ),
]


def _prepare_ecosystem(use_symlinks: bool):
    """Iterates over all registered libraries and prepares them."""
    for lib in LIBRARIES:
        lib.build()
        lib.install(use_symlinks=use_symlinks)


def get_requires_for_build_wheel(config_settings=None):
    return orig.get_requires_for_build_wheel(config_settings)


def get_requires_for_build_sdist(config_settings=None):
    return orig.get_requires_for_build_sdist(config_settings)


def get_requires_for_build_editable(config_settings=None):
    return orig.get_requires_for_build_editable(config_settings)


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    return orig.prepare_metadata_for_build_wheel(
        metadata_directory, config_settings
    )


def prepare_metadata_for_build_editable(
    metadata_directory, config_settings=None
):
    return orig.prepare_metadata_for_build_editable(
        metadata_directory, config_settings
    )


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    _prepare_ecosystem(use_symlinks=False)
    return orig.build_wheel(
        wheel_directory, config_settings, metadata_directory
    )


def build_editable(
    wheel_directory, config_settings=None, metadata_directory=None
):
    _prepare_ecosystem(use_symlinks=True)
    return orig.build_editable(
        wheel_directory, config_settings, metadata_directory
    )


def build_sdist(sdist_directory, config_settings=None):
    return orig.build_sdist(sdist_directory, config_settings)

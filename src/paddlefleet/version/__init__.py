# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

from paddlefleet.version import git

commit = "unknown"

paddlefleet_dir = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if commit.endswith("unknown") and git.is_git_repo(paddlefleet_dir) and git.have_git():
    commit = git.git_revision(paddlefleet_dir).decode("utf-8")
    if git.is_dirty(paddlefleet_dir):
        commit += ".dirty"
del paddlefleet_dir


__all__ = ["__version__", "commit", "show"]


def show():
    """Get the corresponding commit id of paddlefleet.

    Returns:
        The commit-id of paddlefleet will be output.

        full_version: version of paddlefleet


    Examples:
        .. code-block:: python

            import paddlefleet

            paddlefleet.version.show()
            # commit: 1ef5b94a18773bb0b1bba1651526e5f5fc5b16fa

    """
    print("commit:", commit)


from .._fleet_version import __version__ as __version__
from .._fleet_version import commit as commit

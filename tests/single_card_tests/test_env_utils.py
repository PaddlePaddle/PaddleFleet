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


from paddlefleet.env_utils import (
    get_env_int,
    get_env_str,
    is_ci_env,
)


def test_get_env_int_default():
    assert get_env_int("NOT_EXIST_ENV") == 0
    assert get_env_int("NOT_EXIST_ENV", default=5) == 5


def test_get_env_int_valid(monkeypatch):
    monkeypatch.setenv("TEST_INT_ENV", "123")
    assert get_env_int("TEST_INT_ENV") == 123


def test_get_env_int_invalid(monkeypatch):
    monkeypatch.setenv("TEST_INT_ENV", "abc")
    assert get_env_int("TEST_INT_ENV", default=7) == 7


def test_get_env_str_default():
    assert get_env_str("NOT_EXIST_STR") is None
    assert get_env_str("NOT_EXIST_STR", default="hello") == "hello"


def test_get_env_str_valid(monkeypatch):
    monkeypatch.setenv("TEST_STR_ENV", "world")
    assert get_env_str("TEST_STR_ENV") == "world"


def test_is_ci_env_false(monkeypatch):
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.delenv("GITLAB_CI", raising=False)
    assert is_ci_env() is False


def test_is_ci_env_true(monkeypatch):
    monkeypatch.setenv("CI", "true")
    assert is_ci_env() is True
import json

import pytest

from scripts.convert_hf_to_paddle import infer_model_type


def write_config(path, model_type):
    path.mkdir()
    (path / "config.json").write_text(json.dumps({"model_type": model_type}), encoding="utf-8")


def test_infer_model_type_rejects_qwen3(tmp_path):
    model_dir = tmp_path / "qwen3"
    write_config(model_dir, "qwen3")

    with pytest.raises(ValueError, match="Qwen3"):
        infer_model_type(model_dir)


def test_infer_model_type_supports_ernie_moe(tmp_path):
    model_dir = tmp_path / "ernie"
    write_config(model_dir, "ernie4_5_moe")

    assert infer_model_type(model_dir) == "ernie_moe"

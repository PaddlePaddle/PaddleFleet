from rrattn.checkpoint_utils import (
    flex_checkpoint_load_lock,
    is_hf_safetensors_checkpoint,
    is_paddle_checkpoint,
    load_pretrained_checkpoint,
)


class FakeConfig:
    base_config_key = None

    @classmethod
    def get_config_dict(cls, path):
        return (
            {
                "rope_scaling": {
                    "type": "yarn",
                    "original_max_position_embeddings": 16,
                    "factor": 2,
                }
            },
            {},
        )

    @classmethod
    def from_dict(cls, config_dict):
        config = cls()
        config.max_position_embeddings = config_dict["max_position_embeddings"]
        config.rope_scaling = config_dict["rope_scaling"]
        return config


class FakeModel:
    config_class = FakeConfig
    calls = []

    @classmethod
    def from_pretrained(cls, model_name_or_path, **kwargs):
        cls.calls.append((model_name_or_path, kwargs))
        return kwargs


def test_checkpoint_format_detection(tmp_path):
    hf_dir = tmp_path / "hf"
    hf_dir.mkdir()
    (hf_dir / "model.safetensors.index.json").write_text("{}")

    paddle_dir = tmp_path / "paddle"
    paddle_dir.mkdir()
    (paddle_dir / "model_state.pdparams.index.json").write_text("{}")

    assert is_hf_safetensors_checkpoint(hf_dir)
    assert not is_paddle_checkpoint(hf_dir)
    assert is_paddle_checkpoint(paddle_dir)
    assert not is_hf_safetensors_checkpoint(paddle_dir)


def test_load_pretrained_checkpoint_routes_by_format(tmp_path):
    FakeModel.calls = []

    paddle_dir = tmp_path / "paddle"
    paddle_dir.mkdir()
    (paddle_dir / "model_state.pdparams").write_text("")
    load_pretrained_checkpoint(FakeModel, paddle_dir, dtype="bfloat16")
    assert FakeModel.calls[-1][1]["load_checkpoint_format"] == "sharding_io"
    assert FakeModel.calls[-1][1]["convert_from_hf"] is False
    assert FakeModel.calls[-1][1]["use_safetensors"] is False

    hf_dir = tmp_path / "hf"
    hf_dir.mkdir()
    (hf_dir / "model.safetensors.index.json").write_text("{}")
    load_pretrained_checkpoint(FakeModel, hf_dir, dtype="bfloat16")
    assert FakeModel.calls[-1][1]["load_checkpoint_format"] == "flex_checkpoint"
    assert FakeModel.calls[-1][1]["config"].max_position_embeddings == 32

    unknown_dir = tmp_path / "unknown"
    unknown_dir.mkdir()
    load_pretrained_checkpoint(FakeModel, unknown_dir, dtype="bfloat16")
    assert FakeModel.calls[-1][1] == {
        "dtype": "bfloat16",
        "load_checkpoint_format": "sharding_io",
    }


def test_flex_checkpoint_load_lock_does_not_delete_paddle_lock(tmp_path):
    metadata_lock = tmp_path / "flex-ckpt.auto_generated.metadata.lock"
    metadata_lock.write_text("")

    with flex_checkpoint_load_lock(tmp_path):
        assert metadata_lock.exists()

    assert metadata_lock.exists()

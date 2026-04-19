import pytest


paddle = pytest.importorskip("paddle")
pytest.importorskip("paddleformers")


def test_public_api_imports():
    import rrattn

    assert callable(rrattn.rrattn_estimate)
    assert callable(rrattn.rrattn_prefill)
    assert callable(rrattn.patch_llama_attention)
    assert callable(rrattn.patch_qwen_attention)
    assert callable(rrattn.patch_ernie_attention)


def test_rrattn_version_default():
    from rrattn.llama_patch import patch_llama_attention

    assert patch_llama_attention.__defaults__[-1] == "v1"

from __future__ import annotations

import contextlib
import importlib.util


_COMPAT_PROXY_SCOPE_ENABLED = False


def is_package_installed(package_name: str) -> bool:
    return importlib.util.find_spec(package_name) is not None


def enable_tilelang_paddle_compat_before_import():
    import paddle

    if not hasattr(paddle, "enable_compat"):
        raise RuntimeError("paddle.enable_compat is required before importing tilelang-paddle")
    paddle.enable_compat(scope={"tilelang"})


@contextlib.contextmanager
def _paddle_current_device_guard():
    try:
        from tilelang.jit.adapter.base import BaseKernelAdapter
    except Exception:
        yield
        return

    original = BaseKernelAdapter.get_current_device_functor

    def paddle_current_device_functor(*args, **kwargs):
        import paddle

        if paddle.is_compiled_with_cuda():
            return lambda: paddle.framework._current_expected_place()
        return lambda: "cpu"

    BaseKernelAdapter.get_current_device_functor = staticmethod(paddle_current_device_functor)
    try:
        yield
    finally:
        BaseKernelAdapter.get_current_device_functor = original


@contextlib.contextmanager
def paddle_tilelang_compat_guard():
    """Enable Paddle's torch proxy only around DSv4 TileLang compat launches."""
    import paddle

    if not hasattr(paddle, "compat") or not hasattr(paddle.compat, "enable_torch_proxy"):
        raise RuntimeError("paddle.compat.enable_torch_proxy is required by attention_paddle_compat")
    if not hasattr(paddle.compat, "disable_torch_proxy"):
        raise RuntimeError("paddle.compat.disable_torch_proxy is required by attention_paddle_compat")

    global _COMPAT_PROXY_SCOPE_ENABLED
    if not _COMPAT_PROXY_SCOPE_ENABLED:
        scope = {"tilelang", "paddlefleet.ops.tilelang_dsv4"}
        paddle.compat.enable_torch_proxy(scope=scope)
        _COMPAT_PROXY_SCOPE_ENABLED = True
    with _paddle_current_device_guard():
        yield

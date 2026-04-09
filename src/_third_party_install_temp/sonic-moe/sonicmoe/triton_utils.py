import sys

import paddle

original_paddle_empty = paddle.empty


def torch_compat_empty(*args, **kwargs):
    if "device" in kwargs and kwargs["device"] == "cuda":
        del kwargs["device"]
    return original_paddle_empty(*args, **kwargs)


def swap_torch_guard(fn):
    def wrapped_fn(*args, **kwargs):
        if "torch" not in sys.modules:
            return fn(*args, **kwargs)
        torch_module = sys.modules["torch"]
        original_paddle_empty = paddle.empty
        sys.modules["torch"] = paddle
        paddle.empty = torch_compat_empty
        try:
            return fn(*args, **kwargs)
        finally:
            sys.modules["torch"] = torch_module
            paddle.empty = original_paddle_empty

    return wrapped_fn


def wrap_triton_kernel(triton_kernel):
    class WrappedTritonKernel:
        def __init__(self, kernel):
            self.kernel = kernel

        def __getitem__(self, index):
            return swap_torch_guard(self.kernel[index])

        def __getattr__(self, name):
            return getattr(self.kernel, name)

    return WrappedTritonKernel(triton_kernel)

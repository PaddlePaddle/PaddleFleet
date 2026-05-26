_ENABLE_COMPAT_CALLED = False


def enable_tilelang_paddle_compat_before_import():
    global _ENABLE_COMPAT_CALLED
    if _ENABLE_COMPAT_CALLED:
        return
    import paddle
    paddle.enable_compat(scope={"tilelang"}, silent=True)
    _ENABLE_COMPAT_CALLED = True
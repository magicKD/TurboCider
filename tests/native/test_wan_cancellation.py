"""Real native Wan cancellation/recovery; Python is only the test harness.

Set TURBOCIDER_WAN_TEST_MODEL to a complete native model package. A missing
fixture or unavailable GPU is a skip, not evidence of successful inference.
"""
import ctypes as C
import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
LIB = C.CDLL(str(ROOT / "build/native/libturbocider.dylib"))
CALLBACK = C.CFUNCTYPE(None, C.c_char_p, C.c_void_p)
LIB.tc_system_json.restype = C.c_void_p
LIB.tc_string_free.argtypes = [C.c_void_p]
LIB.tc_engine_create_model.argtypes = [C.c_char_p, C.c_char_p, C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)]
LIB.tc_engine_generate.argtypes = [C.c_void_p, C.c_char_p, CALLBACK, C.c_void_p,
                                  C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)]
LIB.tc_engine_cancel.argtypes = [C.c_void_p]
LIB.tc_engine_free.argtypes = [C.c_void_p]
LIB.tc_engine_unload.argtypes = [C.c_void_p, C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)]


def consume(pointer):
    if not pointer.value:
        return None
    result = C.string_at(pointer).decode()
    LIB.tc_string_free(pointer)
    pointer.value = None
    return result


@pytest.mark.parametrize("phase", ["model_load", "umt5_text_encode", "wan_dit_block", "taehv", "export"])
def test_native_wan_cancel_and_recover(phase, tmp_path):
    model = os.environ.get("TURBOCIDER_WAN_TEST_MODEL")
    if not model:
        pytest.skip("explicit native Wan fixture not configured")
    root = Path(model).resolve(strict=True)
    assert (root / "vae/taew2_1.safetensors").is_file()
    system = json.loads(consume(C.c_void_p(LIB.tc_system_json())))
    if not system.get("gpu_available"):
        pytest.skip("Metal device unavailable")
    engine, error = C.c_void_p(), C.c_void_p()
    assert LIB.tc_engine_create_model(b"wan2.1-1.3b-qad", str(root).encode(),
                                      C.byref(engine), C.byref(error)) == 0, consume(error)
    output = tmp_path / "video.mp4"
    request = {"model": "wan2.1-1.3b-qad", "operation": "video.generate",
               "prompt": "A red boat on a calm lake.", "output": str(output),
               "width": 832, "height": 480, "frames": 5, "fps": 16, "steps": 3,
               "audio": False, "execution": "gpu", "compile_gpu": False}
    triggered = False

    @CALLBACK
    def cancel_at(raw, _):
        nonlocal triggered
        event = json.loads(raw)
        if event.get("phase") == phase and not triggered:
            triggered = True
            LIB.tc_engine_cancel(engine)

    @CALLBACK
    def progress(raw, _):
        pass

    def generate(callback):
        result, failure = C.c_void_p(), C.c_void_p()
        code = LIB.tc_engine_generate(engine, json.dumps(request).encode(), callback,
                                      None, C.byref(result), C.byref(failure))
        return code, consume(result), consume(failure)

    try:
        # A cancelled request must leave an existing destination intact.
        output.write_bytes(b"existing-user-output")
        code, result, failure = generate(cancel_at)
        assert triggered, f"phase {phase} was not exercised"
        assert (code, result, failure) == (2, None, "generation cancelled")
        assert output.read_bytes() == b"existing-user-output"
        assert not list(tmp_path.glob("*.tmp.*.mp4"))
        code, result, failure = generate(progress)
        assert code == 0, failure
        metrics = json.loads(result)
        assert metrics["runtime_dependency"] == "native"
        assert metrics["frames"] == 5 and metrics["runtime_backend"] == "wan-mlx"
        assert output.stat().st_size > 128
        # Repeated requests reuse conditioning, not stale cancellation state.
        code, result, failure = generate(progress)
        assert code == 0, failure
        assert json.loads(result)["prompt_cache_hit"] is True
        unloaded = C.c_void_p()
        assert LIB.tc_engine_unload(engine, C.byref(unloaded), C.byref(error)) == 0, consume(error)
        consume(unloaded)
        code, result, failure = generate(progress)
        assert code == 0, failure
        assert json.loads(result)["prompt_cache_hit"] is False
    finally:
        LIB.tc_engine_free(engine)

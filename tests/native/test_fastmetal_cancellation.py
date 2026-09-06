"""Fixture-backed cancellation regression for the persistent FastMetal child.

The fake worker never imports MLX or loads model weights.  The real FastMetal
model directory is used only for the native Session's pinned asset checks, so
this test exercises the production pipe/poll/SIGTERM path without running a
denoise.  It skips when the validated local fixture or a visible Metal device
is unavailable.
"""

from __future__ import annotations

import ctypes as C
import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
LIB = C.CDLL(str(ROOT / "build/native/libturbocider.dylib"))
CALLBACK = C.CFUNCTYPE(None, C.c_char_p, C.c_void_p)
LIB.tc_system_json.restype = C.c_void_p
LIB.tc_string_free.argtypes = [C.c_void_p]
LIB.tc_engine_create_model.argtypes = [
    C.c_char_p,
    C.c_char_p,
    C.POINTER(C.c_void_p),
    C.POINTER(C.c_void_p),
]
LIB.tc_engine_generate.argtypes = [
    C.c_void_p,
    C.c_char_p,
    CALLBACK,
    C.c_void_p,
    C.POINTER(C.c_void_p),
    C.POINTER(C.c_void_p),
]
LIB.tc_engine_cancel.argtypes = [C.c_void_p]
LIB.tc_engine_free.argtypes = [C.c_void_p]


def consume(pointer: C.c_void_p) -> str | None:
    if not pointer.value:
        return None
    value = C.string_at(pointer).decode()
    LIB.tc_string_free(pointer)
    return value


def local_model_root() -> Path:
    configured = os.environ.get("TURBOCIDER_FASTMETAL_TEST_MODEL")
    if configured:
        return Path(configured).resolve()
    return (
        ROOT.parent
        / "gpu_ane/fastmetal-runtime/models/FastMetal-1.3B-QAD"
    ).resolve()


def test_active_cancel_terminates_fastmetal_worker_and_publishes_no_output() -> None:
    system_pointer = C.c_void_p(LIB.tc_system_json())
    system = json.loads(consume(system_pointer) or "{}")
    if not system.get("gpu_available"):
        pytest.skip("Metal device is unavailable")

    model_root = local_model_root()
    required = [
        model_root / "mlx_dit.json",
        model_root / "mlx_dit.safetensors",
        model_root / "tokenizer",
        model_root / "text_encoder",
        model_root / "vae",
    ]
    if not all(path.exists() for path in required):
        pytest.skip("validated FastMetal fixture is unavailable")

    with tempfile.TemporaryDirectory(prefix="turbocider-fastmetal-cancel-") as raw:
        temporary = Path(raw)
        fake_entrypoint = temporary / "entrypoint.py"
        fake_entrypoint.write_text("# protocol fixture\n")
        pid_path = temporary / "worker.pid"
        fake_worker = temporary / "worker.py"
        fake_worker.write_text(
            """#!/usr/bin/env python3
import json
import os
import sys
import time

with open(os.environ["TURBOCIDER_FASTMETAL_FAKE_PID"], "w") as stream:
    stream.write(str(os.getpid()))
print(json.dumps({"type": "ready"}), flush=True)
for line in sys.stdin:
    request = json.loads(line)
    if request.get("action") == "generate":
        print(json.dumps({"type": "progress", "phase": "dit", "completed": 0, "total": 3}), flush=True)
        while True:
            time.sleep(1)
"""
        )
        profile = temporary / "profile.json"
        profile.write_text(
            json.dumps(
                {
                    "schema": "turbocider-fastmetal-v1",
                    "python": sys.executable,
                    "engine_root": str(temporary),
                    "script": str(fake_entrypoint),
                    "worker": str(fake_worker),
                    "decode_backend": "taehv",
                    "mlx_compile": True,
                    "prompt_cache": True,
                }
            )
        )
        output = temporary / "cancelled.mp4"
        request = {
            "model": "fastmetal-1.3b-qad",
            "operation": "video.generate",
            "prompt": "cancel fixture",
            "output": str(output),
            "execution": "gpu",
            "width": 832,
            "height": 480,
            "frames": 81,
            "fps": 16,
            "steps": 3,
            "seed": 1024,
            "residency": "resident",
        }

        old_profile = os.environ.get("TURBOCIDER_FASTMETAL_CONFIG")
        old_pid = os.environ.get("TURBOCIDER_FASTMETAL_FAKE_PID")
        os.environ["TURBOCIDER_FASTMETAL_CONFIG"] = str(profile)
        os.environ["TURBOCIDER_FASTMETAL_FAKE_PID"] = str(pid_path)
        engine = C.c_void_p()
        error = C.c_void_p()
        result = C.c_void_p()
        progress = threading.Event()
        outcome: dict[str, object] = {}

        @CALLBACK
        def callback(raw_event: bytes, _context: C.c_void_p) -> None:
            event = json.loads(raw_event)
            if event.get("phase") == "denoise":
                progress.set()

        try:
            code = LIB.tc_engine_create_model(
                b"fastmetal-1.3b-qad",
                str(model_root).encode(),
                C.byref(engine),
                C.byref(error),
            )
            assert code == 0, consume(error)

            def run_generate() -> None:
                code = LIB.tc_engine_generate(
                    engine,
                    json.dumps(request).encode(),
                    callback,
                    None,
                    C.byref(result),
                    C.byref(error),
                )
                outcome["code"] = code
                outcome["result"] = consume(result)
                outcome["error"] = consume(error)

            thread = threading.Thread(target=run_generate, daemon=True)
            thread.start()
            assert progress.wait(15), "fake worker did not enter active generation"
            LIB.tc_engine_cancel(engine)
            thread.join(15)
            assert not thread.is_alive(), "cancel did not unblock native generation"
            assert outcome == {
                "code": 2,
                "result": None,
                "error": "generation cancelled",
            }
            assert not output.exists()
            assert not list(temporary.glob("cancelled.mp4.tmp.*.mp4"))

            worker_pid = int(pid_path.read_text())
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                try:
                    os.kill(worker_pid, 0)
                except ProcessLookupError:
                    break
                time.sleep(0.05)
            else:
                pytest.fail("cancelled FastMetal worker process is still alive")
        finally:
            if engine.value:
                LIB.tc_engine_cancel(engine)
                LIB.tc_engine_free(engine)
            if old_profile is None:
                os.environ.pop("TURBOCIDER_FASTMETAL_CONFIG", None)
            else:
                os.environ["TURBOCIDER_FASTMETAL_CONFIG"] = old_profile
            if old_pid is None:
                os.environ.pop("TURBOCIDER_FASTMETAL_FAKE_PID", None)
            else:
                os.environ["TURBOCIDER_FASTMETAL_FAKE_PID"] = old_pid

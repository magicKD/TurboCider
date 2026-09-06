"""CPU-only regression tests for the persistent FastMetal JSONL protocol."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
WORKER_PATH = ROOT / "tools/native/fastmetal_worker.py"
SPEC = importlib.util.spec_from_file_location("turbocider_fastmetal_worker", WORKER_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FakeWorker:
    def __init__(self) -> None:
        self.requests: list[dict[str, object]] = []

    def generate(self, request: dict[str, object]) -> dict[str, object]:
        self.requests.append(request)
        return {"prompt": request["prompt"], "persistent_model": True}


def test_protocol_control_messages_do_not_enter_generation() -> None:
    worker = FakeWorker()
    assert MODULE.dispatch_protocol_request(worker, {"action": "ping"}) == {
        "type": "pong"
    }
    assert MODULE.dispatch_protocol_request(worker, {"action": "shutdown"}) == {
        "type": "stopped"
    }
    assert worker.requests == []


def test_protocol_generate_forwards_one_request_and_wraps_result() -> None:
    worker = FakeWorker()
    request = {"prompt": "misty forest", "seed": 1024}
    assert MODULE.dispatch_protocol_request(
        worker, {"action": "generate", "request": request}
    ) == {
        "type": "result",
        "result": {"prompt": "misty forest", "persistent_model": True},
    }
    assert worker.requests == [request]


@pytest.mark.parametrize("payload", [{}, {"action": "unknown"}])
def test_protocol_rejects_unknown_actions(payload: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="unknown action"):
        MODULE.dispatch_protocol_request(FakeWorker(), payload)


def test_worker_peak_rss_metric_uses_bytes() -> None:
    assert MODULE.process_peak_rss_bytes() > 1024 * 1024
    current = MODULE.process_rss_bytes()
    assert current is None or current > 1024 * 1024


def test_worker_accepts_separate_model_and_mlx_checkpoint_roots(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model_root = tmp_path / "model"
    checkpoint = tmp_path / "merged"
    entrypoint = tmp_path / "entrypoint.py"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(WORKER_PATH),
            "--model-root",
            str(model_root),
            "--mlx-checkpoint",
            str(checkpoint),
            "--entrypoint",
            str(entrypoint),
        ],
    )
    args = MODULE.parse_args()
    assert args.model_root == model_root
    assert args.mlx_checkpoint == checkpoint
    source = WORKER_PATH.read_text()
    assert "load_mlx_dit_checkpoint(\n            self.mlx_checkpoint" in source
    assert "model_root=self.model_root" in source


def test_worker_rejects_incomplete_mlx_checkpoint_before_importing_mlx(
    tmp_path: Path,
) -> None:
    entrypoint = tmp_path / "entrypoint.py"
    entrypoint.write_text("# not imported before checkpoint validation\n")
    args = type(
        "Args",
        (),
        {
            "model_root": tmp_path / "model",
            "mlx_checkpoint": tmp_path / "missing-checkpoint",
            "entrypoint": entrypoint,
        },
    )()
    with pytest.raises(ValueError, match="must contain mlx_dit.json"):
        MODULE.FastMetalWorker(args)

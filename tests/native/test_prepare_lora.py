import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools/native/prepare_lora.py"


def fake_workspace(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    tools = tmp_path / "h3.c/tools"
    tools.mkdir(parents=True)
    for name in ("merge_h3_lora.py", "merge_ltx_refiner.py"):
        (tools / name).write_text("# fixture\n")
    environment = os.environ.copy()
    environment["TURBOCIDER_WORKSPACE"] = str(tmp_path)
    return tools, environment


def run_tool(tmp_path: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    _, environment = fake_workspace(tmp_path)
    return subprocess.run(
        [sys.executable, str(TOOL), *arguments],
        text=True,
        capture_output=True,
        env=environment,
        check=False,
    )


def test_h3_dispatch_preserves_upstream_options(tmp_path: Path) -> None:
    result = run_tool(
        tmp_path,
        "minimax-h3-turbo",
        "base",
        "adapter.safetensors",
        "merged",
        "--check-only",
        "--device",
        "cpu",
        "--profile",
        "lightx2v-4step",
        "--strength",
        "0.0625",
        "--source-revision",
        "revision",
        "--print-command",
    )
    assert result.returncode == 0, result.stderr
    value = json.loads(result.stdout)
    assert value["model"] == "h3"
    assert value["upstream"].endswith("h3.c/tools/merge_h3_lora.py")
    assert value["command"][-9:] == [
        "--check-only",
        "--device",
        "cpu",
        "--profile",
        "lightx2v-4step",
        "--strength",
        "0.0625",
        "--source-revision",
        "revision",
    ]


def test_ltx_dispatch_uses_existing_refiner_merger(tmp_path: Path) -> None:
    result = run_tool(
        tmp_path,
        "ltx-2.5-distilled",
        "base.safetensors",
        "adapter.safetensors",
        "merged.safetensors",
        "--device",
        "auto",
        "--print-command",
    )
    assert result.returncode == 0, result.stderr
    value = json.loads(result.stdout)
    assert value["model"] == "ltx"
    assert value["upstream"].endswith("h3.c/tools/merge_ltx_refiner.py")
    assert value["command"][-2:] == ["--device", "auto"]


def test_ltx_rejects_h3_only_merge_options(tmp_path: Path) -> None:
    result = run_tool(
        tmp_path,
        "ltx",
        "base.safetensors",
        "adapter.safetensors",
        "merged.safetensors",
        "--strength",
        "0.8",
        "--print-command",
    )
    assert result.returncode == 2
    assert "only valid for H3" in result.stderr


def runtime_fixture(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, str]]:
    tools = tmp_path / "workspace/h3.c/tools"
    tools.mkdir(parents=True)
    counter = tmp_path / "workspace/merge-count.txt"
    merger = tools / "merge_h3_lora.py"
    merger.write_text(
        """#!/usr/bin/env python3
import argparse
from pathlib import Path
import time
p=argparse.ArgumentParser()
p.add_argument('base');p.add_argument('lora');p.add_argument('output')
p.add_argument('--profile');p.add_argument('--strength');p.add_argument('--device')
a=p.parse_args()
counter=Path(__file__).parents[2]/'merge-count.txt'
with counter.open('a') as stream: stream.write('1\\n')
time.sleep(0.15)
out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
(out/'h3-turbo-merge-manifest.json').write_text('{}')
"""
    )
    base = tmp_path / "base-model/FL2VA/transformer"
    base.mkdir(parents=True)
    (base / "model.safetensors.index.json").write_text("index")
    (base / "model-00001.safetensors").write_bytes(b"base")
    adapter = tmp_path / "adapter.safetensors"
    adapter.write_bytes(b"adapter-v1")
    environment = os.environ.copy()
    environment["TURBOCIDER_WORKSPACE"] = str(tmp_path / "workspace")
    environment["TURBOCIDER_PREPARE_PYTHON"] = sys.executable
    return base, adapter, counter, environment


def runtime_command(tmp_path: Path, base: Path, adapter: Path) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "tools/native/lora_runtime_cache.py"),
        "h3",
        str(base),
        str(adapter),
        "--strength",
        "0.0625",
        "--profile",
        "lightx2v-4step",
        "--cache-dir",
        str(tmp_path / "cache"),
    ]


def test_runtime_cache_miss_hit_and_content_invalidation(tmp_path: Path) -> None:
    base, adapter, counter, environment = runtime_fixture(tmp_path)
    command = runtime_command(tmp_path, base, adapter)
    first = subprocess.run(command, text=True, capture_output=True,
                           env=environment, check=True)
    first_value = json.loads(first.stdout)
    assert first_value["cache_hit"] is False
    assert Path(first_value["artifact"]).is_dir()
    second = subprocess.run(command, text=True, capture_output=True,
                            env=environment, check=True)
    second_value = json.loads(second.stdout)
    assert second_value["cache_hit"] is True
    assert second_value["cache_key"] == first_value["cache_key"]
    assert counter.read_text().splitlines() == ["1"]

    adapter.write_bytes(b"adapter-v2")
    third = subprocess.run(command, text=True, capture_output=True,
                           env=environment, check=True)
    third_value = json.loads(third.stdout)
    assert third_value["cache_hit"] is False
    assert third_value["cache_key"] != first_value["cache_key"]
    assert counter.read_text().splitlines() == ["1", "1"]


def test_runtime_cache_lock_builds_once_across_processes(tmp_path: Path) -> None:
    base, adapter, counter, environment = runtime_fixture(tmp_path)
    command = runtime_command(tmp_path, base, adapter)
    first = subprocess.Popen(command, text=True, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, env=environment)
    time.sleep(0.03)
    second = subprocess.Popen(command, text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, env=environment)
    first_out, first_error = first.communicate(timeout=10)
    second_out, second_error = second.communicate(timeout=10)
    assert first.returncode == 0, first_error
    assert second.returncode == 0, second_error
    values = [json.loads(first_out), json.loads(second_out)]
    assert sorted(value["cache_hit"] for value in values) == [False, True]
    assert values[0]["cache_key"] == values[1]["cache_key"]
    assert counter.read_text().splitlines() == ["1"]

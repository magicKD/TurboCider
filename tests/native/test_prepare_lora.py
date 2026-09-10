import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools/native/prepare_lora.py"


def local_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.pop("TURBOCIDER_WORKSPACE", None)
    return environment


def run_tool(tmp_path: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    environment = local_environment()
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
    assert value["upstream"] == str(ROOT / "tools/native/merge_h3_lora.py")
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
    assert value["upstream"] == str(ROOT / "tools/native/merge_ltx_refiner.py")
    assert value["command"][-2:] == ["--device", "auto"]


def test_ltx_dispatch_forwards_explicit_strength(tmp_path: Path) -> None:
    result = run_tool(
        tmp_path,
        "ltx",
        "base.safetensors",
        "adapter.safetensors",
        "merged.safetensors",
        "--device",
        "auto",
        "--strength",
        "1.25",
        "--print-command",
    )
    assert result.returncode == 0, result.stderr
    value = json.loads(result.stdout)
    assert value["model"] == "ltx"
    assert value["command"][-4:] == ["--device", "auto", "--strength", "1.25"]


def test_ltx_rejects_h3_only_merge_options(tmp_path: Path) -> None:
    result = run_tool(
        tmp_path,
        "ltx",
        "base.safetensors",
        "adapter.safetensors",
        "merged.safetensors",
        "--profile",
        "native",
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
    environment.pop("TURBOCIDER_WORKSPACE", None)
    environment["TURBOCIDER_PREPARE_PYTHON"] = sys.executable
    environment["TURBOCIDER_LORA_TOOL_DIR"] = str(tools)
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


def ltx_runtime_fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, str]]:
    tools = tmp_path / "tools"
    tools.mkdir()
    merger = tools / "merge_ltx_refiner.py"
    merger.write_text(
        """#!/usr/bin/env python3
import argparse
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('base');p.add_argument('lora');p.add_argument('output')
p.add_argument('--strength');p.add_argument('--device')
a=p.parse_args()
Path(__file__).with_name('captured.txt').write_text(a.strength)
out=Path(a.output)
out.write_bytes(b'artifact')
Path(str(out)+'.manifest.json').write_text('{}')
"""
    )
    merger.chmod(0o755)
    base = tmp_path / "base.safetensors"
    adapter = tmp_path / "adapter.safetensors"
    base.write_bytes(b"base")
    adapter.write_bytes(b"adapter")
    environment = local_environment()
    environment["TURBOCIDER_PREPARE_PYTHON"] = sys.executable
    environment["TURBOCIDER_LORA_TOOL_DIR"] = str(tools)
    return base, adapter, environment


def test_runtime_cache_ltx_identity_forwards_strength_and_role(tmp_path: Path) -> None:
    base, adapter, environment = ltx_runtime_fixture(tmp_path)
    command = [
        sys.executable,
        str(ROOT / "tools/native/lora_runtime_cache.py"),
        "ltx", str(base), str(adapter), "--strength", "1.25",
        "--role", "refiner", "--cache-dir", str(tmp_path / "cache"),
    ]
    result = subprocess.run(command, text=True, capture_output=True,
                            env=environment, check=True)
    value = json.loads(result.stdout)
    assert value["cache_hit"] is False
    assert value["identity"]["role"] == "refiner"
    assert value["identity"]["strength"] == 1.25
    assert Path(value["artifact"]).is_file()
    assert Path(value["artifact"]).name == (
        "ltx-2.5-22b-runtime-refiner-comfy-int8-convrot.safetensors"
    )
    assert Path(value["manifest"]).name == (
        "ltx-2.5-22b-runtime-refiner-comfy-int8-convrot.safetensors.manifest.json"
    )
    package = Path(value["artifact"]).parent
    assert (package / base.name).resolve() == base.resolve()
    assert (package / adapter.name).resolve() == adapter.resolve()
    assert (tmp_path / "tools/captured.txt").read_text() == "1.25"


def test_runtime_cache_rejects_nonfinite_strength(tmp_path: Path) -> None:
    base, adapter, environment = ltx_runtime_fixture(tmp_path)
    command = [
        sys.executable,
        str(ROOT / "tools/native/lora_runtime_cache.py"),
        "ltx", str(base), str(adapter), "--strength", "nan",
        "--cache-dir", str(tmp_path / "cache"),
    ]
    result = subprocess.run(command, text=True, capture_output=True,
                            env=environment, check=False)
    assert result.returncode != 0
    assert "finite" in result.stderr

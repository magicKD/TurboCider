"""Direct-engine versus TurboCider performance and quality benchmarking."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
from array import array
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, MutableMapping, Optional

from turbocider.errors import TurboCiderError, ValidationError
from turbocider.models import GenerationRequest
from turbocider.paths import PACKAGE_ROOT, engine_root, workspace_root
from turbocider.runtime import TurboCiderRuntime


def _expand(value: Any, context: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        expanded = value.replace("${TURBOCIDER}", str(PACKAGE_ROOT))
        expanded = expanded.replace("${WORKSPACE}", str(workspace_root()))
        expanded = expanded.replace("${TURBOCIDER_ENGINES}", str(engine_root()))
        expanded = os.path.expandvars(os.path.expanduser(expanded))
        for key, replacement in context.items():
            expanded = expanded.replace("{%s}" % key, replacement)
        return expanded
    if isinstance(value, list):
        return [_expand(item, context) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item, context) for key, item in value.items()}
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _derived_output(path: Path, label: str, index: int) -> Path:
    return path.with_name("%s-%s%d%s" % (path.stem, label, index, path.suffix))


@contextlib.contextmanager
def _temporary_environment(values: Mapping[str, Any]) -> Iterator[None]:
    previous: Dict[str, Optional[str]] = {}
    try:
        for key, value in values.items():
            name = str(key)
            previous[name] = os.environ.get(name)
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = str(value)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _summary(values: Iterable[float]) -> Dict[str, Any]:
    samples = [float(value) for value in values]
    return {
        "samples": samples,
        "count": len(samples),
        "min": min(samples) if samples else None,
        "median": statistics.median(samples) if samples else None,
        "max": max(samples) if samples else None,
        "mean": statistics.fmean(samples) if samples else None,
    }


def _metric_value(metrics: Mapping[str, Any], path: str) -> Optional[float]:
    value: Any = metrics
    for component in path.split("."):
        if not isinstance(value, Mapping) or component not in value:
            return None
        value = value[component]
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _engine_seconds(
    target: Mapping[str, Any], metrics: Mapping[str, Any], stdout: str
) -> Optional[float]:
    metric_path = str(target.get("engine_metric_path", "wall_seconds"))
    value = _metric_value(metrics, metric_path)
    if value is not None:
        return value
    pattern = target.get("engine_metric_regex")
    if pattern:
        match = re.search(str(pattern), stdout)
        if match:
            try:
                return float(match.group(1))
            except (IndexError, TypeError, ValueError):
                pass
    return None


def _output_exists(path: Path, kind: str) -> bool:
    if kind == "file":
        return path.is_file()
    if kind == "directory":
        return path.is_dir()
    raise ValidationError("benchmark output_kind must be file or directory")


def _run_command_target(
    target: Mapping[str, Any], label: str, warmups: int, runs: int
) -> Dict[str, Any]:
    base_output = Path(str(_expand(target["output"], {}))).resolve()
    cwd = Path(str(_expand(target.get("cwd", PACKAGE_ROOT), {}))).resolve()
    environment = os.environ.copy()
    environment.update({
        str(key): str(_expand(value, {}))
        for key, value in target.get("environment", {}).items()
    })
    output_kind = str(target.get("output_kind", "file"))
    all_rows = []
    for ordinal in range(1, warmups + runs + 1):
        measured = ordinal > warmups
        run_index = ordinal - warmups if measured else ordinal
        run_label = "run" if measured else "warmup"
        output = _derived_output(base_output, run_label, run_index)
        output.parent.mkdir(parents=True, exist_ok=True)
        context = {
            "output": str(output),
            "output_stem": str(output.with_suffix("")),
            "run": str(run_index),
        }
        command = [str(item) for item in _expand(target["command"], context)]
        sidecar_value = target.get("sidecar", "{output_stem}.engine.json")
        sidecar = Path(str(_expand(sidecar_value, context))).resolve()
        started = time.perf_counter()
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        wall = time.perf_counter() - started
        if completed.returncode != 0:
            raise TurboCiderError(
                "%s command failed with exit code %d:\n%s"
                % (label, completed.returncode, completed.stdout[-4000:])
            )
        if not _output_exists(output, output_kind):
            raise TurboCiderError("%s did not create %s" % (label, output))
        metrics = (
            json.loads(sidecar.read_text(encoding="utf-8"))
            if sidecar.is_file()
            else {}
        )
        all_rows.append({
            "kind": run_label,
            "index": run_index,
            "output": str(output),
            "sidecar": str(sidecar) if sidecar.is_file() else None,
            "process_wall_seconds": wall,
            "engine_wall_seconds": _engine_seconds(
                target, metrics, completed.stdout
            ),
            "metrics": metrics,
            "stdout_tail": completed.stdout[-2000:],
        })
    measured_rows = [row for row in all_rows if row["kind"] == "run"]
    return {
        "kind": "command",
        "runs": all_rows,
        "process_wall": _summary(row["process_wall_seconds"] for row in measured_rows),
        "engine_wall": _summary(
            row["engine_wall_seconds"]
            for row in measured_rows
            if row["engine_wall_seconds"] is not None
        ),
        "comparison_output": measured_rows[-1]["output"],
    }


def _run_batch_command_target(
    target: Mapping[str, Any], label: str, warmups: int, runs: int
) -> Dict[str, Any]:
    output_dir = Path(str(_expand(target["output_dir"], {}))).resolve()
    report_path = Path(str(_expand(target["report"], {}))).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    context = {
        "output_dir": str(output_dir),
        "report": str(report_path),
        "total_runs": str(warmups + runs),
    }
    command = [str(item) for item in _expand(target["command"], context)]
    environment = os.environ.copy()
    environment.update({
        str(key): str(_expand(value, context))
        for key, value in target.get("environment", {}).items()
    })
    cwd = Path(str(_expand(target.get("cwd", PACKAGE_ROOT), context))).resolve()
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    process_wall = time.perf_counter() - started
    if completed.returncode != 0:
        raise TurboCiderError(
            "%s batch command failed with exit code %d:\n%s"
            % (label, completed.returncode, completed.stdout[-4000:])
        )
    raw_rows = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(raw_rows, list) or len(raw_rows) != warmups + runs:
        raise TurboCiderError(
            "%s batch report has %d rows; expected %d"
            % (label, len(raw_rows) if isinstance(raw_rows, list) else -1, warmups + runs)
        )
    all_rows = []
    for ordinal, metrics in enumerate(raw_rows, 1):
        measured = ordinal > warmups
        output = Path(str(metrics["output"])).resolve()
        if not output.is_file():
            raise TurboCiderError("%s did not create %s" % (label, output))
        all_rows.append({
            "kind": "run" if measured else "warmup",
            "index": ordinal - warmups if measured else ordinal,
            "output": str(output),
            "engine_wall_seconds": _engine_seconds(target, metrics, ""),
            "metrics": metrics,
        })
    measured_rows = [row for row in all_rows if row["kind"] == "run"]
    return {
        "kind": "batch-command",
        "command": command,
        "process_wall_total_seconds": process_wall,
        "runs": all_rows,
        "process_wall": _summary([]),
        "engine_wall": _summary(
            row["engine_wall_seconds"]
            for row in measured_rows
            if row["engine_wall_seconds"] is not None
        ),
        "comparison_output": measured_rows[-1]["output"],
        "stdout_tail": completed.stdout[-2000:],
    }


def _run_turbocider_target(
    target: Mapping[str, Any], label: str, warmups: int, runs: int
) -> Dict[str, Any]:
    raw_request = dict(_expand(target["request"], {}))
    output_spec = dict(raw_request.get("output", {}))
    if not output_spec.get("path"):
        raise ValidationError("benchmark TurboCider request requires output.path")
    base_output = Path(str(output_spec["path"])).resolve()
    environment = {
        str(key): (_expand(value, {}) if value is not None else None)
        for key, value in target.get("environment", {}).items()
    }
    all_rows = []
    with _temporary_environment(environment):
        runtime = TurboCiderRuntime()
        try:
            for ordinal in range(1, warmups + runs + 1):
                measured = ordinal > warmups
                run_index = ordinal - warmups if measured else ordinal
                run_label = "run" if measured else "warmup"
                output = _derived_output(base_output, run_label, run_index)
                request_value = json.loads(json.dumps(raw_request))
                request_value.setdefault("output", {})["path"] = str(output)
                request = GenerationRequest.from_dict(request_value)
                started = time.perf_counter()
                record = runtime.submit(request)
                record = runtime.jobs.wait(record.id, timeout=target.get("timeout"))
                outer_wall = time.perf_counter() - started
                row = record.as_dict()
                if row["state"] != "succeeded":
                    raise TurboCiderError(
                        "%s job %s ended as %s: %s"
                        % (label, row["id"], row["state"], row.get("error"))
                    )
                metrics = dict(row.get("metrics", {}))
                context = {
                    "output": str(output),
                    "output_stem": str(output.with_suffix("")),
                    "run": str(run_index),
                }
                sidecar_value = target.get("sidecar", "{output_stem}.engine.json")
                sidecar = Path(str(_expand(sidecar_value, context))).resolve()
                sidecar_metrics = (
                    json.loads(sidecar.read_text(encoding="utf-8"))
                    if sidecar.is_file()
                    else {}
                )
                if "engine_metric_path" in target:
                    engine_wall = _engine_seconds(target, metrics, "")
                else:
                    engine_wall = metrics.get("engine_wall_seconds")
                if engine_wall is None:
                    engine_wall = _engine_seconds(target, sidecar_metrics, "")
                all_rows.append({
                    "kind": run_label,
                    "index": run_index,
                    "output": str(output),
                    "job_id": row["id"],
                    "process_wall_seconds": outer_wall,
                    "job_wall_seconds": metrics.get("job_wall_seconds"),
                    "engine_wall_seconds": engine_wall,
                    "sidecar": str(sidecar) if sidecar.is_file() else None,
                    "engine_metrics": sidecar_metrics,
                    "plan_id": row.get("plan_id"),
                    "metrics": metrics,
                })
        finally:
            runtime.close()
    measured_rows = [row for row in all_rows if row["kind"] == "run"]
    return {
        "kind": "turbocider",
        "runs": all_rows,
        "process_wall": _summary(row["process_wall_seconds"] for row in measured_rows),
        "job_wall": _summary(
            row["job_wall_seconds"]
            for row in measured_rows
            if row["job_wall_seconds"] is not None
        ),
        "engine_wall": _summary(
            row["engine_wall_seconds"]
            for row in measured_rows
            if row["engine_wall_seconds"] is not None
        ),
        "comparison_output": measured_rows[-1]["output"],
    }


def _ffprobe(path: Path) -> Dict[str, Any]:
    completed = subprocess.run(
        [
            "ffprobe", "-v", "error", "-count_frames", "-show_streams",
            "-show_format", "-of", "json", str(path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise TurboCiderError("ffprobe failed for %s: %s" % (path, completed.stderr))
    raw = json.loads(completed.stdout)
    streams = []
    for stream in raw.get("streams", []):
        streams.append({
            key: stream.get(key)
            for key in (
                "index", "codec_type", "codec_name", "width", "height",
                "pix_fmt", "sample_rate", "channels", "channel_layout",
                "r_frame_rate", "avg_frame_rate", "duration", "nb_frames",
                "nb_read_frames",
            )
            if stream.get(key) is not None
        })
    return {
        "format_name": raw.get("format", {}).get("format_name"),
        "duration": raw.get("format", {}).get("duration"),
        "size": raw.get("format", {}).get("size"),
        "streams": streams,
    }


def _decode(path: Path, output: Path, stream: str) -> bool:
    if stream == "video":
        command = [
            "ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0",
            "-f", "rawvideo", "-pix_fmt", "rgb24", str(output),
        ]
    else:
        command = [
            "ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0?",
            "-f", "s16le", "-acodec", "pcm_s16le", str(output),
        ]
    completed = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
    )
    if completed.returncode != 0:
        if stream == "audio":
            return False
        raise TurboCiderError(
            "ffmpeg %s decode failed for %s: %s"
            % (stream, path, completed.stderr.decode(errors="replace"))
        )
    return output.is_file() and output.stat().st_size > 0


def _numeric_compare(first: Path, second: Path, dtype: str) -> Dict[str, Any]:
    if first.stat().st_size != second.stat().st_size:
        return {
            "same_length": False,
            "first_bytes": first.stat().st_size,
            "second_bytes": second.stat().st_size,
        }
    first_hash = _sha256(first)
    second_hash = _sha256(second)
    item_bytes = 1 if dtype == "u8" else 2
    if first_hash == second_hash:
        return {
            "same_length": True,
            "samples": first.stat().st_size // item_bytes,
            "identical": True,
            "mae": 0.0,
            "mse": 0.0,
            "psnr_db": None,
            "cosine_similarity": 1.0,
            "decoded_sha256_first": first_hash,
            "decoded_sha256_second": second_hash,
        }
    count = 0
    absolute = 0.0
    squared = 0.0
    dot = 0.0
    norm_first = 0.0
    norm_second = 0.0
    identical = True
    typecode = "B" if dtype == "u8" else "h"
    with first.open("rb") as left, second.open("rb") as right:
        while True:
            a_raw = left.read(4 << 20)
            b_raw = right.read(len(a_raw))
            if not a_raw:
                break
            identical = identical and a_raw == b_raw
            a_values = array(typecode)
            b_values = array(typecode)
            a_values.frombytes(a_raw[: len(a_raw) // item_bytes * item_bytes])
            b_values.frombytes(b_raw[: len(b_raw) // item_bytes * item_bytes])
            if item_bytes == 2 and sys.byteorder != "little":
                a_values.byteswap()
                b_values.byteswap()
            for a, b in zip(a_values, b_values):
                difference = float(a) - float(b)
                count += 1
                absolute += abs(difference)
                squared += difference * difference
                dot += float(a) * float(b)
                norm_first += float(a) * float(a)
                norm_second += float(b) * float(b)
    mse = squared / count if count else 0.0
    peak = 255.0 if dtype == "u8" else 32767.0
    cosine = (
        dot / math.sqrt(norm_first * norm_second)
        if norm_first > 0 and norm_second > 0
        else (1.0 if identical else 0.0)
    )
    return {
        "same_length": True,
        "samples": count,
        "identical": identical,
        "mae": absolute / count if count else 0.0,
        "mse": mse,
        "psnr_db": None if mse == 0 else 20.0 * math.log10(peak) - 10.0 * math.log10(mse),
        "cosine_similarity": cosine,
        "decoded_sha256_first": first_hash,
        "decoded_sha256_second": second_hash,
    }


def compare_media(first: Path, second: Path) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "first": str(first),
        "second": str(second),
        "file_sha256_first": _sha256(first),
        "file_sha256_second": _sha256(second),
        "file_identical": (
            first.stat().st_size == second.stat().st_size
            and _sha256(first) == _sha256(second)
        ),
        "probe_first": _ffprobe(first),
        "probe_second": _ffprobe(second),
    }
    with tempfile.TemporaryDirectory(prefix="turbocider-benchmark-") as temporary:
        temporary_root = Path(temporary)
        first_video = temporary_root / "first.rgb"
        second_video = temporary_root / "second.rgb"
        _decode(first, first_video, "video")
        _decode(second, second_video, "video")
        result["decoded_visual"] = _numeric_compare(first_video, second_video, "u8")
        first_audio = temporary_root / "first.pcm"
        second_audio = temporary_root / "second.pcm"
        has_first_audio = _decode(first, first_audio, "audio")
        has_second_audio = _decode(second, second_audio, "audio")
        result["audio_presence_match"] = has_first_audio == has_second_audio
        if has_first_audio and has_second_audio:
            result["decoded_audio"] = _numeric_compare(first_audio, second_audio, "s16")
    return result


def _run_target(
    target: Mapping[str, Any], label: str, warmups: int, runs: int
) -> Dict[str, Any]:
    kind = str(target.get("kind", "command"))
    if kind == "command":
        return _run_command_target(target, label, warmups, runs)
    if kind == "batch-command":
        return _run_batch_command_target(target, label, warmups, runs)
    if kind == "turbocider":
        return _run_turbocider_target(target, label, warmups, runs)
    raise ValidationError("unknown benchmark target kind: %s" % kind)


def run_benchmark(spec: Mapping[str, Any]) -> Dict[str, Any]:
    warmups = int(spec.get("warmup_runs", 0))
    runs = int(spec.get("runs", 1))
    if warmups < 0 or runs < 1:
        raise ValidationError("warmup_runs must be nonnegative and runs positive")
    started = time.time()
    direct = _run_target(spec["direct"], "direct", warmups, runs)
    integrated = _run_target(spec["turbocider"], "turbocider", warmups, runs)
    direct_engine = direct["engine_wall"]["median"]
    integrated_engine = integrated["engine_wall"]["median"]
    direct_process = direct["process_wall"]["median"]
    integrated_process = integrated["process_wall"]["median"]
    performance = {
        "direct": direct,
        "turbocider": integrated,
        "engine_overhead_seconds": (
            integrated_engine - direct_engine
            if direct_engine is not None and integrated_engine is not None
            else None
        ),
        "engine_overhead_ratio": (
            integrated_engine / direct_engine
            if direct_engine not in (None, 0) and integrated_engine is not None
            else None
        ),
        "process_overhead_seconds": (
            integrated_process - direct_process
            if direct_process is not None and integrated_process is not None
            else None
        ),
        "process_overhead_ratio": (
            integrated_process / direct_process
            if direct_process not in (None, 0) and integrated_process is not None
            else None
        ),
    }
    comparison = spec.get("comparison", {})
    if not isinstance(comparison, Mapping):
        raise ValidationError("benchmark comparison must be an object")
    comparison_kind = str(comparison.get("kind", "media"))
    if comparison_kind == "media":
        quality = compare_media(
            Path(direct["comparison_output"]),
            Path(integrated["comparison_output"]),
        )
    elif comparison_kind == "artifacts":
        quality = {
            "kind": "artifacts",
            "media_comparison_skipped": True,
        }
    else:
        raise ValidationError(
            "benchmark comparison.kind must be media or artifacts"
        )
    artifact_rows = []
    artifact_context = {
        "direct_output": direct["comparison_output"],
        "direct_output_stem": str(Path(direct["comparison_output"]).with_suffix("")),
        "turbocider_output": integrated["comparison_output"],
        "turbocider_output_stem": str(Path(integrated["comparison_output"]).with_suffix("")),
    }
    for pair in spec.get("artifact_pairs", []):
        first = Path(str(_expand(pair["direct"], artifact_context))).resolve()
        second = Path(str(_expand(pair["turbocider"], artifact_context))).resolve()
        row = {
            "name": str(pair.get("name", first.name)),
            "direct": str(first),
            "turbocider": str(second),
            "direct_exists": first.is_file(),
            "turbocider_exists": second.is_file(),
        }
        if first.is_file() and second.is_file():
            first_hash = _sha256(first)
            second_hash = _sha256(second)
            row.update({
                "direct_bytes": first.stat().st_size,
                "turbocider_bytes": second.stat().st_size,
                "direct_sha256": first_hash,
                "turbocider_sha256": second_hash,
                "identical": (
                    first.stat().st_size == second.stat().st_size
                    and first_hash == second_hash
                ),
            })
            comparison_spec = pair.get("comparison")
            if isinstance(comparison_spec, Mapping):
                comparison_context = dict(artifact_context)
                comparison_context.update({
                    "direct_artifact": str(first),
                    "turbocider_artifact": str(second),
                })
                command = [
                    str(item)
                    for item in _expand(
                        comparison_spec.get("command", []),
                        comparison_context,
                    )
                ]
                if not command:
                    raise ValidationError(
                        "artifact comparison command must not be empty"
                    )
                completed = subprocess.run(
                    command,
                    cwd=str(PACKAGE_ROOT),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
                if completed.returncode != 0:
                    raise TurboCiderError(
                        "artifact comparison failed with exit code %d:\n%s"
                        % (completed.returncode, completed.stdout[-4000:])
                    )
                parsed_metrics = {}
                for name, pattern in comparison_spec.get(
                    "metric_regexes", {}
                ).items():
                    match = re.search(str(pattern), completed.stdout)
                    parsed_metrics[str(name)] = (
                        float(match.group(1)) if match else None
                    )
                row["comparison"] = {
                    "command": command,
                    "stdout": completed.stdout.strip(),
                    "metrics": parsed_metrics,
                }
        artifact_rows.append(row)
    metric_rows = []
    direct_runs = [row for row in direct["runs"] if row["kind"] == "run"]
    integrated_runs = [
        row for row in integrated["runs"] if row["kind"] == "run"
    ]
    for pair in spec.get("metric_pairs", []):
        direct_path = str(pair["direct"])
        integrated_path = str(pair.get("turbocider", direct_path))
        direct_values = [
            value
            for row in direct_runs
            for value in [_metric_value(row.get("metrics", {}), direct_path)]
            if value is not None
        ]
        integrated_values = [
            value
            for row in integrated_runs
            for value in [
                _metric_value(row.get("metrics", {}), integrated_path)
            ]
            if value is not None
        ]
        direct_summary = _summary(direct_values)
        integrated_summary = _summary(integrated_values)
        direct_median = direct_summary["median"]
        integrated_median = integrated_summary["median"]
        metric_rows.append({
            "name": str(pair.get("name", direct_path)),
            "unit": str(pair.get("unit", "seconds")),
            "direct_path": direct_path,
            "turbocider_path": integrated_path,
            "direct": direct_summary,
            "turbocider": integrated_summary,
            "overhead_seconds": (
                integrated_median - direct_median
                if direct_median is not None and integrated_median is not None
                else None
            ),
            "overhead_ratio": (
                integrated_median / direct_median
                if direct_median not in (None, 0) and integrated_median is not None
                else None
            ),
        })
    return {
        "schema": "turbocider-benchmark-v1",
        "id": str(spec.get("id", "benchmark")),
        "started_at_unix": started,
        "duration_seconds": time.time() - started,
        "warmup_runs": warmups,
        "runs": runs,
        "performance": performance,
        "quality": quality,
        "artifacts": artifact_rows,
        "metric_comparisons": metric_rows,
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="turbocider-benchmark")
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    report = run_benchmark(spec)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(rendered, encoding="utf-8")
        os.replace(temporary, args.output)
    print(rendered, end="")


if __name__ == "__main__":
    main()

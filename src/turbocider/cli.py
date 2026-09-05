"""TurboCider command-line interface and daemon entry point."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

from turbocider import __version__
from turbocider.benchmark import run_benchmark
from turbocider.errors import TurboCiderError
from turbocider.model_management import (
    ModelPreparer,
    preparation_options_from_namespace,
)
from turbocider.models import GenerationRequest
from turbocider.runtime import TurboCiderRuntime
from turbocider.service import serve


DEFAULT_API_URL = "http://127.0.0.1:11435"


def _json_print(value: Any) -> None:
    print(json.dumps(value, indent=2))


def _api_json(
    base_url: str,
    path: str,
    *,
    method: str = "GET",
    body: Optional[Dict[str, Any]] = None,
    token: Optional[str] = None,
) -> Dict[str, Any]:
    payload = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {"Accept": "application/json"}
    if payload is not None:
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = "Bearer " + token
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=payload,
        method=method,
        headers=headers,
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            parsed = json.load(response)
    except urllib.error.HTTPError as error:
        try:
            detail = json.loads(error.read().decode("utf-8")).get("error")
        except Exception:
            detail = None
        raise TurboCiderError(
            "TurboCider API returned HTTP %d%s"
            % (error.code, ": " + str(detail) if detail else "")
        ) from error
    except urllib.error.URLError as error:
        raise TurboCiderError(
            "TurboCider API is unavailable at %s: %s"
            % (base_url, error.reason)
        ) from error
    if not isinstance(parsed, dict):
        raise TurboCiderError("TurboCider API returned a non-object response")
    return parsed


def _watch_job(runtime: TurboCiderRuntime, job_id: str, timeout: Optional[float]):
    deadline = time.monotonic() + timeout if timeout is not None else None
    last_status = None
    while True:
        record = runtime.jobs.get(job_id)
        rendered = "%s %.1f%% %s" % (
            record.state.value,
            record.progress * 100.0,
            record.phase,
        )
        if record.started_at is not None and record.finished_at is None:
            current = record.as_dict()
            remaining = current.get("estimated_remaining_seconds")
            if remaining is not None:
                rendered += " ETA %.0fs" % float(remaining)
        if rendered != last_status:
            print(rendered, file=sys.stderr, flush=True)
            last_status = rendered
        if record.state.value in {"succeeded", "failed", "cancelled"}:
            return record
        if deadline is not None and time.monotonic() >= deadline:
            runtime.jobs.cancel(job_id)
            return runtime.jobs.wait(job_id, timeout=10)
        time.sleep(0.2)


def _request_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--request", type=Path, help="load the complete request from JSON")
    parser.add_argument("--model", default="minimax-h3-turbo")
    parser.add_argument("--task", choices=("image", "video", "audio"), default="video")
    parser.add_argument("--prompt")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--frames", type=int)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--audio", action=argparse.BooleanOptionalAction)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guidance", type=float)
    parser.add_argument("--execution", choices=("auto", "gpu", "gpu_ane"), default="auto")
    parser.add_argument("--profile", choices=("quality", "balanced", "preview"), default="quality")
    parser.add_argument("--approximation", choices=("exact", "validated", "experimental"), default="validated")
    parser.add_argument("--persistent", action="store_true")
    parser.add_argument("--no-fallback", action="store_true")
    parser.add_argument("--first-frame", action="append", default=[])
    parser.add_argument("--last-frame", action="append", default=[])
    parser.add_argument("--ref-image", action="append", default=[])
    parser.add_argument("--ref-video", action="append", default=[])
    parser.add_argument("--ref-silent-video", action="append", default=[])
    parser.add_argument(
        "--ref-video-audio",
        action="append",
        nargs=2,
        default=[],
        metavar=("VIDEO", "AUDIO"),
    )
    parser.add_argument("--ref-audio", action="append", default=[])
    parser.add_argument("--engine-arg", action="append", default=[])
    parser.add_argument("--engine-env", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument(
        "--engine-options",
        help="namespaced JSON object, or @PATH to a JSON file",
    )


def _engine_options(value: Optional[str]) -> Dict[str, Any]:
    if value is None:
        return {}
    if value.startswith("@"):
        payload = Path(value[1:]).expanduser().read_text(encoding="utf-8")
    else:
        payload = value
    parsed = json.loads(payload)
    if not isinstance(parsed, dict):
        raise ValueError("--engine-options must decode to a JSON object")
    return parsed


def _parse_request(args) -> GenerationRequest:
    if args.request:
        return GenerationRequest.from_dict(json.loads(args.request.read_text(encoding="utf-8")))
    inputs: List[Dict[str, Any]] = []
    for path in args.first_frame:
        inputs.append({"type": "image", "role": "first_frame", "path": path})
    for path in args.last_frame:
        inputs.append({"type": "image", "role": "last_frame", "path": path})
    for path in args.ref_image:
        inputs.append({"type": "image", "role": "reference", "path": path})
    for path in args.ref_video:
        inputs.append({"type": "video", "role": "reference", "path": path})
    for path in args.ref_silent_video:
        inputs.append({
            "type": "video",
            "role": "reference",
            "path": path,
            "include_embedded_audio": False,
        })
    for video, audio in args.ref_video_audio:
        inputs.append({
            "type": "video",
            "role": "reference",
            "path": video,
            "audio_path": audio,
        })
    for path in args.ref_audio:
        inputs.append({"type": "audio", "role": "reference", "path": path})
    environment = {}
    for item in args.engine_env:
        if "=" not in item:
            raise ValueError("--engine-env must be NAME=VALUE")
        key, value = item.split("=", 1)
        environment[key] = value
    namespace = {
        "minimax-h3-turbo": "h3",
        "ltx-2.5-distilled": "ltx",
        "flux2-klein-4b": "flux2",
        "fastmetal-1.3b-qad": "fastmetal",
    }.get(args.model)
    engine_options = _engine_options(args.engine_options)
    if namespace:
        native = engine_options.get(namespace, {})
        if not isinstance(native, dict):
            raise ValueError("engine_options.%s must be an object" % namespace)
        native = dict(native)
        if args.engine_arg:
            native["args"] = list(args.engine_arg)
        if environment:
            native["env"] = environment
        if native:
            engine_options[namespace] = native
    elif args.engine_arg or environment:
        raise ValueError(
            "--engine-arg/--engine-env require a bundled model id; "
            "use --engine-options for custom model packs"
        )
    return GenerationRequest.from_dict({
        "model": args.model,
        "task": args.task,
        "prompt": args.prompt or "",
        "inputs": inputs,
        "output": {
            "type": args.task,
            "path": str(args.output.resolve()) if args.output else None,
            "width": args.width,
            "height": args.height,
            "frames": args.frames if args.frames is not None else (1 if args.task == "image" else 22),
            "fps": args.fps,
            "audio": (
                args.audio
                if args.audio is not None
                else args.task == "video" and args.model != "fastmetal-1.3b-qad"
            ),
        },
        "sampling": {"seed": args.seed, "steps": args.steps, "guidance": args.guidance},
        "policy": {
            "execution": args.execution,
            "profile": args.profile,
            "approximation": args.approximation,
            "persistent": args.persistent,
            "allow_fallback": not args.no_fallback,
        },
        "engine_options": engine_options,
    })


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="turbocider")
    parser.add_argument("--version", action="version", version="%(prog)s " + __version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="inspect engines, models, and ANE artifacts")
    doctor.add_argument("--json", action="store_true")

    models = subparsers.add_parser("models", help="list registered model packs")
    models.add_argument("--json", action="store_true")

    prepare_model = subparsers.add_parser(
        "prepare-model",
        help="download pinned weights and prepare device-specific ANE artifacts",
    )
    prepare_model.add_argument("model")
    prepare_model.add_argument("--download", action="store_true")
    prepare_model.add_argument("--ane", action="store_true")
    prepare_model.add_argument(
        "--cache", action="store_true",
        help="persist compiled Core ML models when supported",
    )
    prepare_model.add_argument("--dry-run", action="store_true")
    prepare_model.add_argument("--force", action="store_true")
    prepare_model.add_argument("--model-dir", type=Path)
    prepare_model.add_argument(
        "--source-model", type=Path,
        help="existing base/runtime model or transformer checkpoint",
    )
    prepare_model.add_argument("--lora", type=Path, help="existing H3 Turbo LoRA")
    prepare_model.add_argument("--ane-output", type=Path)
    prepare_model.add_argument("--cache-dir", type=Path)
    prepare_model.add_argument(
        "--h3-rows", type=int,
        help="exact fixed token-row count required by an H3 ANE export",
    )
    prepare_model.add_argument(
        "--fastmetal-rows", type=int, default=32760,
        help="fixed FastMetal token rows (32760 for 480x832x81)",
    )
    prepare_model.add_argument(
        "--fastmetal-ane-width", type=int, default=4096,
        help="64-aligned FastMetal FFN channels assigned to ANE",
    )
    prepare_model.add_argument(
        "--blocks", help="comma-separated indexes/ranges, such as 0-19,23"
    )
    prepare_model.add_argument("--ltx-text-rows", type=int, default=1024)
    prepare_model.add_argument("--bucket", type=int, action="append")
    prepare_model.add_argument("--workers", type=int, default=1)
    prepare_model.add_argument("--minimum-free-gib", type=float, default=5.0)
    prepare_model.add_argument("--python", type=Path)

    benchmark = subparsers.add_parser(
        "benchmark", help="compare a direct engine with TurboCider"
    )
    benchmark.add_argument("--spec", type=Path, required=True)
    benchmark.add_argument("--output", type=Path)

    plans = subparsers.add_parser("plans", help="show candidate execution plans")
    _request_parser(plans)

    generate = subparsers.add_parser("generate", help="generate image/video/audio")
    _request_parser(generate)
    generate.add_argument("--dry-run", action="store_true")
    generate.add_argument("--detach", action="store_true")
    generate.add_argument("--timeout", type=float)
    generate.add_argument(
        "--api-url",
        default=os.environ.get("TURBOCIDER_API_URL", DEFAULT_API_URL),
    )
    generate.add_argument("--api-token", default=os.environ.get("TURBOCIDER_API_TOKEN"))

    jobs = subparsers.add_parser("jobs", help="list jobs from a running TurboCider service")
    jobs.add_argument("--json", action="store_true")
    jobs.add_argument("--id")
    jobs.add_argument("--cancel", action="store_true")
    jobs.add_argument(
        "--api-url",
        default=os.environ.get("TURBOCIDER_API_URL", DEFAULT_API_URL),
    )
    jobs.add_argument("--api-token", default=os.environ.get("TURBOCIDER_API_TOKEN"))

    serve_parser = subparsers.add_parser("serve", help="run the local HTTP API")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=11435)
    serve_parser.add_argument("--token", default=os.environ.get("TURBOCIDER_API_TOKEN"))
    serve_parser.add_argument("--workers", type=int, default=1)
    serve_parser.add_argument(
        "--allow-input-root", type=Path, action="append", default=[]
    )
    serve_parser.add_argument("--allow-external-output", action="store_true")
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "prepare-model":
        try:
            preparer = ModelPreparer()
            requested = args.download or args.ane or args.cache
            report = (
                preparer.prepare(
                    args.model, preparation_options_from_namespace(args)
                )
                if requested
                else preparer.inspect(args.model)
            )
            _json_print(report)
            return
        except (TurboCiderError, ValueError, OSError, json.JSONDecodeError) as error:
            print("turbocider: %s" % error, file=sys.stderr)
            raise SystemExit(2) from error
    if args.command == "benchmark":
        try:
            specification = json.loads(args.spec.read_text(encoding="utf-8"))
            report = run_benchmark(specification)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                temporary = args.output.with_suffix(args.output.suffix + ".tmp")
                temporary.write_text(
                    json.dumps(report, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                temporary.replace(args.output)
            _json_print(report)
            return
        except (TurboCiderError, ValueError, OSError, json.JSONDecodeError) as error:
            print("turbocider: %s" % error, file=sys.stderr)
            raise SystemExit(2) from error
    if args.command == "jobs":
        try:
            path = "/v1/jobs" + (("/" + args.id) if args.id else "")
            if args.cancel and not args.id:
                raise ValueError("jobs --cancel requires --id JOB_ID")
            report = _api_json(
                args.api_url,
                path,
                method="DELETE" if args.cancel else "GET",
                token=args.api_token,
            )
            _json_print(report)
            return
        except (TurboCiderError, ValueError) as error:
            print("turbocider: %s" % error, file=sys.stderr)
            raise SystemExit(2) from error
    if args.command == "generate" and args.detach:
        try:
            if args.dry_run:
                raise ValueError("--detach and --dry-run cannot be combined")
            report = _api_json(
                args.api_url,
                "/v1/jobs",
                method="POST",
                body=_parse_request(args).as_dict(),
                token=args.api_token,
            )
            _json_print(report)
            return
        except (TurboCiderError, ValueError, OSError, json.JSONDecodeError) as error:
            print("turbocider: %s" % error, file=sys.stderr)
            raise SystemExit(2) from error
    runtime = TurboCiderRuntime(max_workers=getattr(args, "workers", 1))
    try:
        if args.command == "doctor":
            report = runtime.doctor()
            _json_print(report)
        elif args.command == "models":
            _json_print({"data": [item.public_dict() for item in runtime.registry.all()]})
        elif args.command == "plans":
            _json_print({"data": runtime.plans(_parse_request(args))})
        elif args.command == "generate":
            request = _parse_request(args)
            prepared = runtime.prepare(request)
            if args.dry_run:
                _json_print(prepared.as_dict())
                return
            record = runtime.jobs.submit(request, prepared.plan.id, prepared.spec)
            if args.detach:
                _json_print(record.as_dict())
                return
            record = _watch_job(runtime, record.id, args.timeout)
            _json_print(record.as_dict())
            if record.state.value != "succeeded":
                raise SystemExit(1)
        elif args.command == "serve":
            print("TurboCider API listening on http://%s:%d" % (args.host, args.port))
            previous_handlers = {}

            def stop_service(signum, frame):
                raise KeyboardInterrupt

            for signum in (signal.SIGINT, signal.SIGTERM):
                previous_handlers[signum] = signal.signal(signum, stop_service)
            try:
                serve(
                    runtime,
                    args.host,
                    args.port,
                    args.token,
                    allowed_input_roots=args.allow_input_root,
                    allow_external_output=args.allow_external_output,
                )
            except KeyboardInterrupt:
                pass
            finally:
                for signum, handler in previous_handlers.items():
                    signal.signal(signum, handler)
    except (TurboCiderError, ValueError, OSError, json.JSONDecodeError) as error:
        print("turbocider: %s" % error, file=sys.stderr)
        raise SystemExit(2) from error
    finally:
        runtime.close()


def daemon_main() -> None:
    main(["serve"] + sys.argv[1:])


if __name__ == "__main__":
    main()

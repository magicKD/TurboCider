#!/usr/bin/env python3
"""Warm compatible FLUX4 Q4 references only; never reloads native baselines."""
import argparse
import base64
import json
import math
from pathlib import Path
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request

from flux9_bf16_framework_screen import PROMPTS, generation, check_generations, registered_hashes, write_json
from mlxserve_local_model_probe import listener_pid
from runtime_ane_common import benchmark_environment, sha256_file
from runtime_ane_memory import run_sampled

ROOT = Path(__file__).resolve().parents[2]
ROUTES = ("mflux-q4", "mlxserve-q4")


def validate_trial(trial):
    report = trial["report"]
    if trial["route"] not in ROUTES or len(report["runs"]) != 4 or trial["memory"]["verified"]["complete"] is not True:
        raise ValueError("complete known four-PNG reference and independent memory required")
    if trial["route"] == "mlxserve-q4" and report.get("server_terminal") is not True:
        raise ValueError("owned server must exit")
    for i, row in enumerate(report["runs"]):
        if (row["index"], row["prompt"], row["seed"]) != (i, PROMPTS[i], 43 if i == 3 else 42):
            raise ValueError("actual cold/two fresh/repeated prompt new seed differs")
        if (type(row["request_wall_including_export"]) not in (float, int) or
                not math.isfinite(row["request_wall_including_export"]) or row["request_wall_including_export"] <= 0):
            raise ValueError("positive complete image wall required")
        if trial["route"] == "mflux-q4":
            if row["actual_steps"] != [0, 1, 2, 3] or row["actual_quantization"] != 4 or row["loras"]:
                raise ValueError("actual saved Q4 complete base steps required")
        elif row["denoise_progress"] not in ([[i, 4] for i in range(1, 5)], [[i, 4] for i in range(5)]):
            raise ValueError("actual four denoise completion events required")


def server_worker(output):
    config = json.loads((output / "launch.json").read_text())
    address = f"http://127.0.0.1:{config['port']}"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    report = dict(status="running", runs=[], version=config["version"])
    with (output / "server.stdout.txt").open("x") as out, (output / "server.stderr.txt").open("x") as err:
        server = subprocess.Popen(config["command"], cwd=ROOT, stdout=out, stderr=err)
        start = time.perf_counter()
        try:
            while True:
                if server.poll() is not None:
                    raise RuntimeError("owned media server exited before ready")
                if listener_pid(config["port"]) == {server.pid}:
                    try:
                        with opener.open(address + "/health", timeout=1) as response:
                            if response.status == 200:
                                break
                    except (urllib.error.URLError, TimeoutError):
                        pass
                if time.perf_counter() - start > 120:
                    raise TimeoutError("owned readiness deadline exceeded")
                time.sleep(.2)
            report["ready_seconds"] = time.perf_counter() - start
            for index, prompt in enumerate(PROMPTS):
                if listener_pid(config["port"]) != {server.pid}:
                    raise RuntimeError("owned listener identity changed")
                payload = dict(prompt=prompt, seed=43 if index == 3 else 42, size="512x512", steps=4, stream=True)
                write_json(output / f"request-{index}.json", payload)
                events, complete = [], None
                start = time.perf_counter()
                with opener.open(urllib.request.Request(address + "/v1/images/generations", data=json.dumps(payload).encode(),
                    headers={"Content-Type": "application/json"}), timeout=180) as response:
                    for line in response:
                        if not line.startswith(b"data:"):
                            continue
                        item = json.loads(line[5:])
                        if item.get("type") == "complete":
                            complete = item
                            break
                        events.append(item)
                        if item.get("type") == "error":
                            raise RuntimeError(json.dumps(item))
                if complete is None or len(complete.get("data", [])) != 1:
                    raise ValueError("complete media SSE PNG response required")
                png = base64.b64decode(complete["data"][0]["b64_json"], validate=True)
                (output / f"{index}.png").write_bytes(png)
                wall = time.perf_counter() - start
                write_json(output / f"events-{index}.json", events)
                report["runs"].append(dict(index=index, prompt=prompt, seed=payload["seed"], request_wall_including_export=wall,
                    denoise_progress=[[e["step"], e["total"]] for e in events if e.get("type") == "progress" and e.get("stage") == "Generating"],
                    png_sha256=sha256_file(output / f"{index}.png")))
                write_json(output / "report.json", report)
            report["status"] = "complete_reference_only"
        except BaseException as error:
            report.update(status="failed", error=str(error))
            raise
        finally:
            if server.poll() is None:
                server.terminate()
                try:
                    server.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait(timeout=5)
            report["server_terminal"] = server.poll() is not None
            write_json(output / "report.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for n in ("model", "mflux", "python", "binary", "output"):
        parser.add_argument("--" + n, type=Path)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if not args.output:
        parser.error("explicit fresh output required")
    args.output = args.output.resolve()
    if args.worker:
        server_worker(args.output)
        return
    if args.output.exists() or any(getattr(args,n) is None for n in ("model", "mflux", "python", "binary")):
        parser.error("existing explicit references and fresh output required")
    args.model, args.mflux, args.binary = (p.resolve(strict=True) for p in (args.model, args.mflux, args.binary))
    args.python = args.python.absolute()
    sources = sorted(p for p in args.model.rglob("*") if p.is_file() and ".cache" not in p.relative_to(args.model).parts
                     and p.suffix != ".lock" and p.stat().st_size > 0)
    tools = [Path(__file__).resolve(), ROOT / "tools/validation/flux4_mflux_saved_precheck.py", args.binary,
             *sorted((args.mflux / "src").rglob("*.py"))]
    files = sorted(set(sources + tools))
    identities = {str(p): generation(p) for p in files}
    hashes = registered_hashes(files, ROOT / "outputs/benchmark-source-registrations/flux4-q4.json")
    args.output.mkdir(parents=True)
    data = dict(schema="tc-flux4-warm-references-v1", status="running", date="2026-10-12", reference_only=True,
        qualification_passed=False, native_retested=False, model_sha256={str(p): hashes[str(p)] for p in sources}, source_sha256=hashes,
        order=list(ROUTES), trials=[], scope="same original savedQ4/g64 pack incl embedding/8VAEattention linears;512/4,full PNG;constructor/readiness/cold/twofresh/repeat new seed; existing native results reused separately,not new paired ranking")
    target = args.output / "summary.json"
    write_json(target, data)
    try:
        for index, route in enumerate(ROUTES):
            check_generations(identities)
            output = args.output / f"{index}-{route}"
            output.mkdir()
            env = benchmark_environment()
            env.update(PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
            if route == "mflux-q4":
                # The reference tool creates its output and retains its stock
                # saved weights/compile path. The sampler owns only the parent.
                tool_output = output / "request-output"
                command = [str(args.python), "-B", str(ROOT / "tools/validation/flux4_mflux_saved_precheck.py"),
                    "--model", str(args.model), "--mflux", str(args.mflux), "--output", str(tool_output), "--request-count", "4"]
                roles = (("python_helper", str(args.python.resolve(strict=True))),)
            else:
                tool_output = output
                with socket.socket() as sock:
                    sock.bind(("127.0.0.1", 0))
                    port = sock.getsockname()[1]
                version = subprocess.check_output([str(args.binary), "--version"], text=True, stderr=subprocess.STDOUT)
                launch = [str(args.binary), "--serve", "--host", "127.0.0.1", "--port", str(port), "--model", str(args.model), "--log-file", "off"]
                write_json(output / "launch.json", dict(command=launch, port=port, version=version))
                command = [str(args.python), "-B", str(Path(__file__).resolve()), "--worker", "--output", str(output)]
                roles = (("media_server", str(args.binary)), ("listener_probe", "/usr/sbin/lsof"))
            print(json.dumps(dict(starting=index, route=route)), flush=True)
            with (output / "stdout.txt").open("x") as out, (output / "stderr.txt").open("x") as err:
                memory = run_sampled(command, repo=ROOT, output=output, stem="run", env=env, stdout=out, stderr=err,
                    timeout=600, interval_ms=100, max_gap_ms=500, async_evidence=True, child_roles=roles)
            check_generations(identities)
            report = json.loads((tool_output / "report.json").read_text())
            trial = dict(route=route, index=index, report=report, memory=memory)
            validate_trial(trial)
            from PIL import Image
            for i, row in enumerate(report["runs"]):
                path = tool_output / f"{i}.png"
                with Image.open(path) as image:
                    image.load()
                    if image.size != (512,512):
                        raise ValueError("full size PNG required")
                if sha256_file(path) != row["png_sha256"]:
                    raise ValueError("original PNG differs")
            data["trials"].append(trial)
            write_json(target, data)
        data["medians"] = {t["route"]: statistics.median(r["request_wall_including_export"] for r in t["report"]["runs"][1:3]) for t in data["trials"]}
        data["status"] = "complete_reference_only"
    except BaseException as error:
        data.update(status="failed", error=str(error))
        raise
    finally:
        write_json(target, data)
    print(json.dumps(dict(status=data["status"], medians=data["medians"])))


if __name__ == "__main__":
    main()

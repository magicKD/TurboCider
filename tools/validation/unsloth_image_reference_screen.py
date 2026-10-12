#!/usr/bin/env python3
"""Owned pinned Unsloth sd-server complete-image screen; no UI or failed latency ranking."""
import argparse
import base64
import json
import os
from pathlib import Path
import re
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request

from flux9_bf16_framework_screen import generation, check_generations, registered_hashes, write_json
from mlxserve_local_model_probe import listener_pid
from qwen21_gguf_unsloth_screen import measured_time
from runtime_ane_common import benchmark_environment, sha256_file
from runtime_ane_memory import run_sampled
from z_base_formats_screen import PROMPTS

ROOT = Path(__file__).resolve().parents[2]
STEPS = {"z-image-turbo": 8, "flux2-klein-4b": 4}


def request(model, prompt):
    if model not in STEPS:
        raise ValueError("known current-scope image family required")
    return dict(prompt=prompt,width=512,height=512,steps=STEPS[model],seed=42,
        cfg_scale=1,batch_size=1,sampler_name="euler")


def validate_generation(info, payload, log):
    if not isinstance(info,dict) or any(info.get(k) != payload[k] for k in ("prompt","width","height","steps","seed","cfg_scale")):
        raise ValueError("actual complete reference generation metadata differs")
    progress = [(int(a),int(b)) for a,b in re.findall(r"\|\s*(\d+)/(\d+)\s*-\s*[0-9.]+s/it",log)]
    if progress != [(index,payload["steps"]) for index in range(1,payload["steps"]+1)]:
        raise ValueError("every actual sampler step must execute exactly once")
    return {name: measured_time(log,marker) for name,marker in
        (("condition_seconds","get_learned_condition completed"),
         ("sampling_seconds","sampling completed"),("generation_seconds","generate_image completed"))}


def placement(log, encoder):
    expected = "MTL0, params MTL0" if encoder == "gpu" else "CPU, params cpu"
    if not re.search(r"Conditioner[^\n]*-> compute " + re.escape(expected),log):
        raise ValueError("actual requested text placement missing")
    if not re.search(r"DiT[^\n]*-> compute MTL0, params MTL0",log):
        raise ValueError("actual Metal DiT placement missing")


def sampler_roles(binary):
    return (("reference_inference",str(binary)),("listener_probe","/usr/sbin/lsof"))


def launch_command(reference,binary,dit,encoder,vae,model,encoder_placement,port):
    if model not in STEPS or encoder_placement not in ("gpu","default"):
        raise ValueError("known model and text placement required")
    sys.dont_write_bytecode = True
    sys.path.insert(0,str(reference / "studio/backend"))
    from core.inference.sd_cpp_args import metal_text_encoder_flags,native_speed_flags,text_encoder_flags_for_family
    family = "z-image" if model == "z-image-turbo" else "flux.2-klein"
    if text_encoder_flags_for_family(family) != ("--llm",):
        raise ValueError("original Unsloth family flag differs")
    setting = "UNSLOTH_DIFFUSION_SD_CPP_METAL_TE_GPU"
    previous = os.environ.get(setting)
    try:
        os.environ[setting] = "1" if encoder_placement == "gpu" else "0"
        flags = metal_text_encoder_flags()+native_speed_flags("max")
    finally:
        if previous is None:
            os.environ.pop(setting,None)
        else:
            os.environ[setting] = previous
    return [str(binary),"--diffusion-model",str(dit),"--llm",str(encoder),"--vae",str(vae),*flags,
        "--threads","12","--verbose","--cfg-scale","1","--listen-ip","127.0.0.1","--listen-port",str(port)]


def worker(output):
    launch = json.loads((output / "launch.json").read_text())
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    address = f"http://127.0.0.1:{launch['port']}"
    paths = [output / "server.stdout.txt",output / "server.stderr.txt"]
    report = dict(status="running",runs=[],latency_comparison_valid=False,performance_verified=False)
    target = output / "report.json"
    with paths[0].open("x") as out, paths[1].open("x") as err:
        # Inherit the sampler-owned process group, so cancellation cannot
        # orphan the server. Requests additionally verify the listener PID.
        server = subprocess.Popen(launch["command"],cwd=ROOT,stdout=out,stderr=err)
        started = time.perf_counter()
        try:
            while True:
                if server.poll() is not None:
                    raise RuntimeError("owned server exited before readiness")
                if listener_pid(launch["port"]) == {server.pid}:
                    try:
                        with opener.open(address+"/v1/models",timeout=1) as response:
                            if not json.load(response).get("data"):
                                raise ValueError("actual loaded reference model missing")
                        break
                    except (urllib.error.URLError,TimeoutError):
                        pass
                if time.perf_counter()-started >= 180:
                    raise TimeoutError("owned server readiness deadline exceeded")
                time.sleep(.2)
            report["ready_seconds"] = time.perf_counter()-started
            placement("".join(p.read_text() for p in paths),launch["encoder_placement"])
            for index,prompt in enumerate(PROMPTS[:launch["count"]]):
                if listener_pid(launch["port"]) != {server.pid}:
                    raise RuntimeError("owned listener identity changed")
                offsets = [p.stat().st_size for p in paths]
                payload = request(launch["model"],prompt)
                write_json(output / f"request-{index}.json",payload)
                start = time.perf_counter()
                with opener.open(urllib.request.Request(address+"/sdapi/v1/txt2img",
                    data=json.dumps(payload).encode(),headers={"Content-Type":"application/json"}),timeout=600) as response:
                    result = json.load(response)
                http = time.perf_counter()-start
                images = result.get("images")
                if not isinstance(images,list) or len(images) != 1:
                    raise ValueError("one complete PNG response required")
                png = base64.b64decode(images[0],validate=True)
                if not png.startswith(b"\x89PNG\r\n\x1a\n"):
                    raise ValueError("original PNG payload required")
                image_path = output / f"{index}.png"
                image_path.write_bytes(png)
                wall = time.perf_counter()-start
                info = result.get("info")
                if isinstance(info,str):
                    info = json.loads(info)
                log = ""
                for path,offset in zip(paths,offsets):
                    with path.open("rb") as stream:
                        stream.seek(offset)
                        log += stream.read().decode(errors="replace")
                (output / f"request-{index}.log").write_text(log)
                timings = validate_generation(info,payload,log)
                report["runs"].append(dict(index=index,request_wall_including_export=wall,http_request_seconds=http,
                    prompt_is_repeat=index==3,png_sha256=sha256_file(image_path),info=info,**timings))
                write_json(target,report)
            report.update(status="complete_precheck_only",runtime_verified=True)
        except BaseException as error:
            report.update(status="failed",runtime_verified=False,error=str(error))
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
            write_json(target,report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference","binary","dit","encoder","vae"):
        parser.add_argument("--"+name,type=Path)
    parser.add_argument("--model",choices=tuple(STEPS),default="z-image-turbo")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--encoder-placement",choices=("gpu","default"),default="gpu")
    parser.add_argument("--count",type=int,choices=(1,4),default=1)
    parser.add_argument("--worker",action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.worker:
        worker(args.output)
        return
    if args.output.exists() or any(getattr(args,name) is None for name in ("reference","binary","dit","encoder","vae")):
        parser.error("all source paths and fresh output required")
    reference,binary,dit,encoder,vae = (getattr(args,name).resolve(strict=True) for name in ("reference","binary","dit","encoder","vae"))
    sources = [binary,dit,encoder,vae,reference / "studio/backend/core/inference/sd_cpp_args.py",
        reference / "studio/backend/core/inference/diffusion_memory.py"]
    identities = {str(p):generation(p) for p in sources}
    hashes = registered_hashes(sources,ROOT / "outputs/benchmark-source-registrations/unsloth-image.json")
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1",0))
        port = reservation.getsockname()[1]
    command = launch_command(reference,binary,dit,encoder,vae,args.model,args.encoder_placement,port)
    args.output.mkdir(parents=True)
    write_json(args.output / "launch.json",dict(command=command,port=port,model=args.model,count=args.count,encoder_placement=args.encoder_placement))
    data = dict(schema="tc-unsloth-image-reference-screen-v1",status="running",qualification_passed=False,
        latency_comparison_valid=False,performance_verified=False,model=args.model,encoder_placement=args.encoder_placement,
        source_sha256={str(p):hashes[str(p)] for p in sources},count=args.count,steps=STEPS[args.model],command=command,
        scope="owned Unsloth native Metal backend, not Desktop UI; complete PNG reply/export; own process cold/fresh/repeat only, no paired native cohort; exact component recipe separately labelled")
    target = args.output / "summary.json"
    write_json(target,data)
    try:
        with (args.output / "driver.stdout.txt").open("x") as out,(args.output / "driver.stderr.txt").open("x") as err:
            data["memory"] = run_sampled([sys.executable,"-B",str(Path(__file__).resolve()),"--worker","--output",str(args.output)],
                repo=ROOT,output=args.output,stem="reference",env=benchmark_environment(),stdout=out,stderr=err,timeout=1500,
                interval_ms=100,max_gap_ms=500,async_evidence=True,child_roles=sampler_roles(binary))
        check_generations(identities)
        report = json.loads((args.output / "report.json").read_text())
        if report["status"] != "complete_precheck_only" or report["server_terminal"] is not True or len(report["runs"]) != args.count:
            raise ValueError("complete terminal reference image sequence required")
        from PIL import Image
        for index in range(args.count):
            with Image.open(args.output / f"{index}.png") as image:
                image.load()
                if image.size != (512,512):
                    raise ValueError("actual full-size512 PNG required")
        data.update(status="complete_precheck_only",runtime_verified=True,report=report)
        if args.count == 4:
            data["one_process_two_fresh_median_seconds"] = statistics.median(row["request_wall_including_export"] for row in report["runs"][1:3])
    except BaseException as error:
        data.update(status="failed",runtime_verified=False,error=str(error))
        raise
    finally:
        write_json(target,data)
    print(json.dumps(dict(status=data["status"],runs=[row["request_wall_including_export"] for row in data["report"]["runs"]])))


if __name__ == "__main__":
    main()

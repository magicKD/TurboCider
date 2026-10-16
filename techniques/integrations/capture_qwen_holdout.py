#!/usr/bin/env python3
"""Opt-in unseen seed/prompt capture with an existing source-matched runtime.

No rebuild/download/private backend. The fixed native hook writes nine real BF16
inputs; the reference image is instrumented and not a request performance score.
Only this task's TMPDIR is supplied to native artifact-lease cleanup.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time

from capture_qwen_ffn import digest, source_match

PROMPT = "A red bicycle leaning against a yellow wall, soft overcast light, realistic photograph."


def text_sha(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def different_workload(seed, prompt, baseline_seed, baseline_prompt):
    if (type(seed) is not int or not 0 <= seed < 2**32 or seed == baseline_seed or not isinstance(prompt,str)
            or not 1 <= len(prompt.encode("utf-8")) <= 4096 or "\0" in prompt or text_sha(prompt) == text_sha(baseline_prompt)):
        raise ValueError("holdout capture requires a different uint32 seed AND a different bounded prompt")


def stop_child(child):
    for action, grace in ((child.terminate,4),(child.kill,4)):
        if child.poll() is not None: return child.returncode
        try: action()
        except ProcessLookupError: pass
        try: return child.wait(timeout=grace)
        except subprocess.TimeoutExpired: pass
    raise RuntimeError("owned native child termination not confirmed")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device",action="store_true")
    parser.add_argument("--native-dir",type=Path,required=True)
    parser.add_argument("--model-root",type=Path,required=True)
    parser.add_argument("--ane-manifest",type=Path,required=True)
    parser.add_argument("--baseline-capture",type=Path,required=True)
    parser.add_argument("--scenario-id",default="heldout-bicycle")
    parser.add_argument("--seed",type=int,default=43)
    parser.add_argument("--prompt",default=PROMPT)
    parser.add_argument("--timeout",type=int,default=180)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    if not re.fullmatch("[a-z][a-z0-9-]{2,63}",args.scenario_id) or not 60 <= args.timeout <= 240:
        parser.error("bounded scenario ID and 60..240 second timeout required")
    baseline_root=args.baseline_capture.resolve()
    baseline=json.loads((baseline_root/"receipt.json").read_text())
    old_request=json.loads((baseline_root/"request.json").read_text())
    different_workload(args.seed,args.prompt,baseline["seed"],old_request["prompt"])
    native,model,output=(value.resolve() for value in (args.native_dir,args.model_root,args.output_dir))
    build_path=native/"runtime-build/runtime-build-manifest.json"; build=json.loads(build_path.read_text())
    count=source_match(build)
    if build["runtime_build_id"] != baseline["runtime_build_id"]:
        parser.error("reuse the explicit same source-qualified runtime; do not auto-rebuild")
    for name,key in (("turbocider","cli_sha256"),("libturbocider.dylib","library_sha256")):
        if digest(native/name) != baseline[key]: parser.error("runtime binary differs from baseline capture")
    manifest=json.loads(args.ane_manifest.read_text())
    if any(manifest.get(key)!=value for key,value in {"backend":"runtime_weight_fp16","kind":"swiglu","rows":1024,"hidden":4096,"width":12288}.items()):
        parser.error("fixed public full-width runtime SwiGLU manifest required")
    output.mkdir(parents=True,exist_ok=False)
    own=Path(__file__); helper=own.with_name("capture_qwen_ffn.py")
    report={"schema":"tc-qwen-heldout-capture-v1","status":"starting","not_performance":True,"production_qualified":False,
            "scenario_id":args.scenario_id,"seed":args.seed,"prompt_sha256":text_sha(args.prompt),"steps":20,"loras":[],
            "shape":[1,1024,4096],"original_dtype":"bfloat16","cache":"off","records":[],"device_requested":args.device,
            "baseline_capture_receipt_sha256":digest(baseline_root/"receipt.json"),"baseline_seed":baseline["seed"],
            "baseline_prompt_sha256":text_sha(old_request["prompt"]),"source_sha256":{own.name:digest(own),helper.name:digest(helper)},
            "runtime_build_id":build["runtime_build_id"],"native_build_manifest_sha256":digest(build_path),"native_source_files_matched":count,
            "cli_sha256":digest(native/"turbocider"),"library_sha256":digest(native/"libturbocider.dylib"),
            "ane_manifest_sha256":digest(args.ane_manifest),"actual_ffn_route":"not-confirmed","errors":[],"owned_child_exit_confirmed":False}
    checkpoint=model/"diffusion_models/qwen_image_2.1_bf16.safetensors"; before=checkpoint.stat()
    report["checkpoint_stat_before"]={key:getattr(before,key) for key in ("st_size","st_mtime_ns","st_ctime_ns","st_ino","st_dev")}

    def save():
        (output/"receipt.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")

    save(); child=None; started=time.monotonic()
    try:
        # 73 MiB real matrices + 32 MiB sparse stats/serialization + 32 MiB task
        # temporary allowance + 2 MiB reference PNG + 128 MiB remaining reserve.
        report["admission_minimum_free_mib"]=267
        if shutil.disk_usage(output).free < 267*1024**2:
            raise ValueError("267 MiB capture admission required; no existing data deleted")
        (output/"source").mkdir()
        for path in (own,helper): shutil.copy2(path,output/"source"/path.name)
        request=dict(model="qwen-image-2.1",operation="image.generate",width=512,height=512,prompt=args.prompt,
            output=str(output/"reference.png"),steps=20,seed=args.seed,frames=1,audio=False,prompt_enhance=False,
            execution="gpu_ane",residency="resident",allow_approximation=True,hybrid_mlp_mode="runtime",
            ane_manifest=str(args.ane_manifest.resolve()),qwen21_dit_cache="off",qwen21_denoiser_backend="mlx")
        with (output/"request.json").open("x",encoding="utf-8") as stream: stream.write(json.dumps(request,indent=2)+"\n")
        report["request_sha256"]=digest(output/"request.json")
        if not args.device:
            report["status"]="planned-only-no-device"
        else:
            temporary=output/"task-tmp"; temporary.mkdir(mode=0o700)
            environment={key:value for key,value in os.environ.items() if not key.startswith(("TURBOCIDER_","ANE_","ANEFORGE_","DYLD_","MPSGRAPH_","MLX_DISABLE_COMPILE"))}
            environment.update(TMPDIR=str(temporary),TURBOCIDER_ANE_CALIBRATION_DIR=str(output/"captures"),
                TURBOCIDER_ANE_CALIBRATION_REPLAY="1",TURBOCIDER_RUNTIME_ANE_CHUNKS="0",TURBOCIDER_RUNTIME_ANE_PROFILE="0",TURBOCIDER_ANE_BACKEND="public")
            report["task_tmpdir_supplied"]=True
            # Stay in the supervisor's owned process group. Only our Popen child
            # is stopped here; the outer supervisor can stop its full group.
            with (output/"stdout.json").open("x") as stdout,(output/"stderr.log").open("x") as stderr:
                child=subprocess.Popen([str(native/"turbocider"),"generate",str(model),str(output/"request.json")],env=environment,stdout=stdout,stderr=stderr)
                report["status"]="running"; save()
                deadline=time.monotonic()+args.timeout
                while child.poll() is None:
                    if time.monotonic()>=deadline or shutil.disk_usage(output).free<128*1024**2:
                        raise RuntimeError("capture time/disk reserve reached; owned child stopped")
                    try: child.wait(timeout=1)
                    except subprocess.TimeoutExpired: pass
            report["returncode"]=child.returncode
            if child.returncode: raise RuntimeError("native heldout capture failed; inspect owned log")
            source_match(build)
            if any(getattr(checkpoint.stat(),key)!=value for key,value in report["checkpoint_stat_before"].items()):
                raise ValueError("checkpoint stat generation changed")
            completions=list((output/"captures").glob("*-replay/complete.json"))
            if len(completions)!=1: raise ValueError("exactly one complete replay required")
            directory=completions[0].parent; files=list(directory.glob("*.safetensors"))
            expected={f"layer-{layer}-step-{step}.safetensors" for layer in (0,15,31) for step in (1,10,19)}
            if {p.name for p in files}!=expected or sum(p.stat().st_size for p in files)>73*1024**2:
                raise ValueError("all nine complete actual BF16 points/byte cap required")
            report["records"]=[{"path":str(p.relative_to(output)),"bytes":p.stat().st_size,"sha256":digest(p)} for p in sorted(files)]
            report["completion_sha256"]=digest(completions[0]); report["reference_image_sha256"]=digest(output/"reference.png")
            report["actual_ffn_route"]="forced-gpu-runtime-boundary"; report["status"]="captured"
        report["source_inputs_unchanged"]=all(digest(path)==report["source_sha256"][path.name] for path in (own,helper))
        report["runtime_binaries_unchanged"]=digest(native/"turbocider")==report["cli_sha256"] and digest(native/"libturbocider.dylib")==report["library_sha256"]
    except (Exception,KeyboardInterrupt) as error:
        report["status"]="failed"; report["errors"].append({"type":type(error).__name__,"message":str(error)})
    finally:
        if child is not None:
            if child.poll() is None: report["returncode"]=stop_child(child)
            report["owned_child_exit_confirmed"]=child.poll() is not None
        report["instrumented_wall_seconds"]=time.monotonic()-started
        report["physical_global_device_drain"]="not observed; no cross-child buffer reuse"
        save()
    print(json.dumps({"status":report["status"],"scenario_id":args.scenario_id,"seed":args.seed,"records":len(report["records"]),"errors":report["errors"]}))
    return 0 if report["status"] in ("captured","planned-only-no-device") else 1


if __name__=="__main__":
    def stop_signal(signum,frame): raise KeyboardInterrupt("owned capture supervisor signal")
    signal.signal(signal.SIGTERM,stop_signal)
    raise SystemExit(main())

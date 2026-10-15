#!/usr/bin/env python3
"""Build/run the explicit bounded original-input Qwen attention replay client.

Optional product integration only. Existing libraries/models remain read-only;
every recorded build source/MLX asset must match before and after execution.
No full native rebuild, App replacement, download, private ANE or image export.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sysconfig

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(1<<20),b""):
            value.update(block)
    return value.hexdigest()


def build_match(build, mlx):
    counts = {}
    for group, root in (("sources",ROOT),("mlx",mlx)):
        for item in build["inputs"][group]:
            relative = Path(item["path"])
            target = (root / relative).resolve()
            if (relative.is_absolute() or ".." in relative.parts or not target.is_relative_to(root.resolve())
                    or not target.is_file() or digest(target) != item["sha256"]):
                raise ValueError("build source/MLX identity mismatch: " + item["path"])
        counts[group] = len(build["inputs"][group])
    return counts


def stat_identity(path):
    value=path.stat()
    return {key:getattr(value,key) for key in ("st_size","st_mtime_ns","st_ctime_ns","st_ino","st_dev")}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-dir",type=Path,required=True)
    parser.add_argument("--model-root",type=Path,required=True)
    parser.add_argument("--mlx-root",type=Path,default=Path(sysconfig.get_paths()["purelib"])/"mlx")
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    if platform.system()!="Darwin" or not 0<=args.seed<2**32:
        parser.error("macOS and uint32 seed required")
    output=args.output_dir.resolve()
    output.mkdir(parents=True,exist_ok=False)
    report={"schema":"tc-qwen-attention-capture-receipt-v1","status":"starting","production_qualified":False,
            "not_performance":True,"driver_report":None,"errors":[],"seed":args.seed,
            "capture_kind":"exposed-output replay from actual ordinary GPU block arguments, not unchanged internal kernel capture",
            "maximum_tensor_bytes":128*1024*1024,"runner_sha256":digest(Path(__file__))}

    def save():
        (output/"receipt.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")

    save()
    try:
        if shutil.disk_usage(output).free < (512+128)*1024*1024:
            raise ValueError("requires 640 MiB disk headroom; no other data deleted")
        native,mlx,model=args.native_dir.resolve(),args.mlx_root.resolve(),args.model_root.resolve()
        manifest_path=native/"runtime-build/runtime-build-manifest.json"
        build=json.loads(manifest_path.read_text())
        report["build_inputs_verified"]=build_match(build,mlx)
        report.update(runtime_build_id=build["runtime_build_id"],runtime_manifest_sha256=digest(manifest_path),
                      native_library_sha256=digest(native/"libturbocider.dylib"))
        sources=[model/name for name in ("diffusion_models/qwen_image_2.1_bf16.safetensors",
                                        "text_encoders/qwen3vl_8b_bf16.safetensors","processor/tokenizer.json")]
        before=[stat_identity(path) for path in sources]
        (output/"source").mkdir()
        files=[Path(__file__),Path(__file__).with_suffix(".cpp")]
        report["source_snapshots"]=[]
        for path in files:
            shutil.copy2(path,output/"source"/path.name)
            report["source_snapshots"].append({"path":"source/"+path.name,"sha256":digest(path)})
        binary=output/"capture-qwen-attention"
        command=["xcrun","clang++","-std=c++20","-O2","-Wall","-Wextra","-fvisibility=default",
            "-mmacosx-version-min=26.2","-I",str(ROOT/"native"),"-I",str(ROOT/"native/core"),
            "-I",str(ROOT/"bindings/c/include"),"-isystem",str(mlx/"include"),
            str(output/"source/capture_qwen_attention.cpp"),"-L"+str(native),"-lturbocider",
            "-L"+str(mlx/"lib"),"-lmlx","-ljaccl","-licucore","-framework","Foundation","-framework","Metal",
            "-Wl,-rpath,"+str(native),"-Wl,-rpath,"+str(mlx/"lib"),"-o",str(binary)]
        with (output/"build.stdout.log").open("x") as stdout,(output/"build.stderr.log").open("x") as stderr:
            built=subprocess.run(command,stdout=stdout,stderr=stderr,timeout=120)
        report["build_returncode"]=built.returncode
        save()
        if built.returncode:
            raise ValueError("attention client build failed; inspect local build log")
        report["binary_sha256"]=digest(binary)
        environment={key:value for key,value in os.environ.items() if not key.startswith(("TURBOCIDER_","ANE_","ANEFORGE_","DYLD_"))}
        with (output/"driver.stdout.log").open("x") as stdout,(output/"driver.stderr.log").open("x") as stderr:
            child=subprocess.run([str(binary),str(model),str(output),str(args.seed)],env=environment,
                                 stdout=stdout,stderr=stderr,timeout=150)
        report["driver_returncode"]=child.returncode
        if child.returncode:
            raise ValueError("attention replay failed; partial artifacts preserved for diagnosis")
        driver=json.loads((output/"driver.json").read_text())
        if driver["loaded_runtime_id"]!=build["runtime_build_id"] or driver["raw_capture_bytes"]>report["maximum_tensor_bytes"]:
            raise ValueError("loaded library or capture byte identity differs")
        if [record["phase"] for record in driver["records"]]!=["prefill","steady"] or driver["vae_decode"] or driver["image_export"]:
            raise ValueError("bounded two-phase trajectory differs")
        report["driver_report"]=driver
        report["artifacts"]=[{"path":record["file"],"sha256":digest(output/record["file"]),
                              "bytes":(output/record["file"]).stat().st_size} for record in driver["records"]]
        report["model_source_stats_unchanged"]=before==[stat_identity(path) for path in sources]
        report["build_inputs_verified_after"]=build_match(build,mlx)
        report["source_inputs_unchanged"]=all(digest(path)==snapshot["sha256"] for path,snapshot in zip(files,report["source_snapshots"]))
        if not report["model_source_stats_unchanged"] or not report["source_inputs_unchanged"]:
            raise ValueError("model source generation or capture source changed")
        report["status"]="captured-replay-with-controls"
    except (Exception,KeyboardInterrupt) as error:
        report["status"]="failed"
        report["errors"].append({"type":type(error).__name__,"message":str(error)})
    save()
    print(json.dumps({"status":report["status"],"errors":report["errors"],"production_qualified":False}))
    return int(report["status"]!="captured-replay-with-controls")


if __name__=="__main__":
    raise SystemExit(main())

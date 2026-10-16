#!/usr/bin/env python3
"""Opt-in held-out ordinary GPU block vs staged/copy/shared whole-block replay.

Uses a source-matched existing runtime and frozen owned private constants. No
product rebuild, default mutation, model download, VAE decode or image export.
Only owned 16MiB CPU BF16 copies may leave the observer; borrowed refs stay local.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import statistics
import struct
import subprocess
import sys
import sysconfig
import zlib

from capture_qwen_ffn import digest, source_match
from capture_qwen_holdout import PROMPT, text_sha, different_workload
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"labs/metal"))
from mlx_adoption_probe import assets


def local(root,name):
    item=Path(name);path=(root/item).resolve()
    if item.is_absolute() or ".." in item.parts or not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("bounded explicit source asset required")
    return path


def raw_bf16(path,key):
    with path.open("rb") as stream:
        size=struct.unpack("<Q",stream.read(8))[0]
        if not 2<=size<=65536:raise ValueError("bounded owned BF16 artifact header required")
        header=json.loads(stream.read(size));node=header[key]
        if node!={"dtype":"BF16","shape":[1,1024,4096],"data_offsets":[0,8388608]} or path.stat().st_size!=size+8+8388608:
            raise ValueError("complete exact BF16 artifact required")
        value=hashlib.sha256()
        for part in iter(lambda:stream.read(1<<20),b""):value.update(part)
    return value.hexdigest()


def read_archive(path):
    with path.open("rb") as stream:
        prefix=stream.read(8)
        if len(prefix)!=8:raise ValueError("short archive header")
        size=struct.unpack("<Q",prefix)[0]
        if not 2<=size<=65536:raise ValueError("archive header cap")
        header=json.loads(stream.read(size));payload=stream.read(20*1024**2+1)
    if (header.get("schema")!="tc-bf16-planes-zlib-v1" or header.get("dtype")!="BF16" or header.get("shape")!=[1,1024,4096]
            or header.get("tensor_order")!=["input","reference"] or header.get("plane_order")!="high-then-low-per-tensor"
            or header.get("uncompressed_bytes")!=16777216 or header.get("compressed_bytes")!=len(payload) or len(payload)>20*1024**2):
        raise ValueError("explicit complete lossless BF16 archive required")
    decoder=zlib.decompressobj();planes=decoder.decompress(payload,16777217)
    if len(planes)!=16777216 or not decoder.eof or decoder.unconsumed_tail or decoder.unused_data:
        raise ValueError("bounded complete single zlib stream required")
    count=4194304;digests=[]
    for offset in (0,2*count):
        raw=bytearray(count*2);raw[0::2]=planes[offset+count:offset+2*count];raw[1::2]=planes[offset:offset+count]
        digests.append(hashlib.sha256(raw).hexdigest())
    if digests!=[header["input_sha256"],header["reference_sha256"]]:raise ValueError("lossless inverse byte-plane hashes differ")
    return header


def ratios(samples,repeats):
    arms={"gpu-original","gpu-staged","copy-hybrid","shared-serial","shared-hybrid"}
    rows={i:{} for i in range(repeats)}
    for row in samples:
        if not row["numeric_pass"] or not row["guards_pass"]:raise ValueError("no ratio after whole-block/component rejection")
        if row["phase"]=="measured":rows[row["trial"]][row["arm"]]=row["wall_ns"]
    if any(set(row)!=arms for row in rows.values()):raise ValueError("all five same-input block arms required")
    result=[]
    for a,b in (("gpu-original","gpu-staged"),("gpu-original","copy-hybrid"),("gpu-original","shared-serial"),
                ("gpu-original","shared-hybrid"),("gpu-staged","shared-hybrid"),("copy-hybrid","shared-hybrid")):
        values=[row[a]/row[b] for row in rows.values()];rng=random.Random(73)
        boot=sorted(statistics.median(rng.choices(values,k=len(values))) for _ in range(2000))
        result.append({"baseline":a,"candidate":b,"paired_ratios":values,"median_paired_host_ratio":statistics.median(values),
                       "bootstrap_95pct":[boot[49],boot[1949]],"production_qualified":False})
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device",action="store_true")
    parser.add_argument("--native-dir",type=Path,required=True)
    parser.add_argument("--model-root",type=Path,required=True)
    parser.add_argument("--bank-run",type=Path,required=True)
    parser.add_argument("--baseline-capture",type=Path,required=True)
    parser.add_argument("--reuse-raw-from",type=Path)
    parser.add_argument("--lossless-archive",action="store_true",help="retain all BF16 bits using a bounded post-run byte-plane/zlib archive; raw-pair gate unchanged")
    parser.add_argument("--match-observation-from",type=Path,help="same ordinary/FFN hashes as an explicit completed-numerics/failed-export observation; not raw reuse")
    parser.add_argument("--temp-context",choices=("owned","inherited"),default="owned",
                        help="explicit SDK cache-context control; ordinary GPU request never creates a product runtime-ANE lease")
    parser.add_argument("--mlx-root",type=Path,default=Path(sysconfig.get_paths()["purelib"])/"mlx")
    parser.add_argument("--case-id",default="bicycle-s43")
    parser.add_argument("--seed",type=int,default=43)
    parser.add_argument("--prompt",default=PROMPT)
    parser.add_argument("--repeats",type=int,choices=(10,30),default=30)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    if not re.fullmatch("[a-z][a-z0-9-]{2,63}",args.case_id):parser.error("bounded case ID required")
    baseline_root=args.baseline_capture.resolve();baseline=json.loads((baseline_root/"receipt.json").read_text())
    old_request=json.loads((baseline_root/"request.json").read_text());different_workload(args.seed,args.prompt,baseline["seed"],old_request["prompt"])
    native,model,bank,mlx=(value.resolve() for value in (args.native_dir,args.model_root,args.bank_run,args.mlx_root))
    output=args.output_dir.resolve();output.mkdir(parents=True,exist_ok=False)
    report={"schema":"tc-qwen-ordinary-block-wrapper-v1","status":"starting","case_id":args.case_id,"seed":args.seed,
            "prompt_sha256":text_sha(args.prompt),"baseline_seed":baseline["seed"],"baseline_prompt_sha256":text_sha(old_request["prompt"]),
            "baseline_capture_receipt_sha256":digest(baseline_root/"receipt.json"),"device_requested":args.device,"native":None,
            "production_qualified":False,"physical_overlap_qualified":False,"runtime_internal_copy_behavior":"unknown","comparisons":None,"errors":[]}

    def save():
        (output/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")

    save()
    try:
        if args.lossless_archive and args.reuse_raw_from:raise ValueError("choose lossless retention or existing raw reuse")
        minimum=132 if args.lossless_archive else 162 if args.reuse_raw_from else 180;report["admission_minimum_free_mib"]=minimum
        report["artifact_policy"]="lossless byte-plane/zlib; actual compressed size plus unchanged 128MiB reserve" if args.lossless_archive else "existing raw reuse" if args.reuse_raw_from else "raw BF16 pair plus unchanged 128MiB reserve"
        if shutil.disk_usage(output).free<minimum*1024**2:raise ValueError(f"{minimum} MiB streamed block admission required; original 267 MiB full-capture gate unchanged")
        build_path=native/"runtime-build/runtime-build-manifest.json";build=json.loads(build_path.read_text())
        report["native_sources_verified"]=source_match(build)
        if build["runtime_build_id"]!=baseline["runtime_build_id"]:raise ValueError("existing exact runtime required; no product rebuild")
        report["runtime_build_id"]=build["runtime_build_id"];report["native_build_manifest_sha256"]=digest(build_path)
        report["native_library_sha256"]=digest(native/"libturbocider.dylib")
        if report["native_library_sha256"]!=baseline["library_sha256"]:raise ValueError("native library no longer matches source-qualified capture runtime")
        before_assets=assets(mlx);report["mlx_assets"]=before_assets
        source_files=[model/name for name in ("diffusion_models/qwen_image_2.1_bf16.safetensors","text_encoders/qwen3vl_8b_bf16.safetensors","processor/tokenizer.json")]
        before_stats=[{key:getattr(path.stat(),key) for key in ("st_size","st_mtime_ns","st_ctime_ns","st_ino","st_dev")} for path in source_files]
        bank_report=json.loads(local(bank,"report.json").read_text());bank_manifest_path=local(bank,"manifest.json");bank_manifest=json.loads(bank_manifest_path.read_text())
        graph=Path(bank_manifest["model_directory"]).resolve()
        if (bank_report.get("status")!="measured" or bank_report.get("share")!=2048 or bank_report.get("shape_mhf")!=[1024,4096,12288]
                or digest(bank_manifest_path)!=bank_report["raw_manifest_sha256"] or not graph.is_relative_to((ROOT.parent/"outputs").resolve())
                or digest(graph/"model.mil")!=bank_report["private_mil_sha256"] or digest(graph/"weights.bin")!=bank_report["private_blob_sha256"]):
            raise ValueError("explicit complete source-qualified owned private constant bank required")
        report["bank_report_sha256"]=digest(bank/"report.json");report["bank_manifest_sha256"]=digest(bank_manifest_path)
        report["private_mil_sha256"]=bank_report["private_mil_sha256"];report["private_blob_sha256"]=bank_report["private_blob_sha256"]
        config={"schema":"tc-qwen-ordinary-block-config-v1","case_id":args.case_id,"seed":args.seed,"prompt":args.prompt,
            "model_root":str(model),"output_directory":str(output),"model_directory":str(graph),"measured_groups":args.repeats,
            "expected_gate_up_sha256":bank_manifest["weights"][0]["sha256"],"expected_down_sha256":bank_manifest["weights"][1]["sha256"]}
        if args.lossless_archive:config["lossless_archive"]=True
        if args.match_observation_from:
            old_root=args.match_observation_from.resolve();old_path=local(old_root,"report.json");old=json.loads(old_path.read_text());data=old.get("native") or {}
            if (old.get("case_id")!=args.case_id or old.get("seed")!=args.seed or old.get("prompt_sha256")!=report["prompt_sha256"]
                    or data.get("errors")!=["post-execution artifact reserve insufficient; no overwrite"]
                    or len(data.get("samples",[]))!=165 or not all(s["numeric_pass"] and s["guards_pass"] for s in data["samples"])):
                raise ValueError("only explicit same-input completed-numerics/export-refused observation can be matched")
            config.update(expected_ordinary_argument_sha256=data["ordinary_argument_sha256"],expected_ordinary_block_sha256=data["ordinary_block_output_sha256"],
                          expected_input_sha256=data["exposed_ffn_input_sha256"],expected_reference_sha256=data["full_ffn_reference_sha256"])
            report["matched_observation_report_sha256"]=digest(old_path)
        if args.reuse_raw_from:
            original=args.reuse_raw_from.resolve();prior=json.loads(local(original,"report.json").read_text())
            if (prior.get("status")!="measured" or prior["case_id"]!=args.case_id or prior["seed"]!=args.seed or prior["prompt_sha256"]!=report["prompt_sha256"]):
                raise ValueError("only matching measured independent case artifacts may be reused")
            data=prior["native"];input_path=local(original,"ffn-input.safetensors");reference_path=local(original,"ffn-reference.safetensors")
            if raw_bf16(input_path,"input")!=data["exposed_ffn_input_sha256"] or raw_bf16(reference_path,"reference")!=data["full_ffn_reference_sha256"]:
                raise ValueError("explicit original BF16 artifacts changed")
            config.update(reuse_artifacts=True,reuse_input_sha256=data["exposed_ffn_input_sha256"],reuse_reference_sha256=data["full_ffn_reference_sha256"])
            report["reused_artifacts_report_sha256"]=digest(original/"report.json")
        temporary=output/"task-tmp";temporary.mkdir(mode=0o700);report["temporary_context"]=args.temp_context
        report["task_tmpdir_supplied"]=args.temp_context=="owned"
        own=Path(__file__);paths=[own,own.with_suffix(".mm"),own.with_name("qwen_shared_abi.h"),own.with_name("qwen_shared_compute.h"),
            ROOT/"labs/ffn_cpu_witness.hpp",own.with_name("capture_qwen_ffn.py"),own.with_name("capture_qwen_holdout.py"),ROOT/"labs/metal/mlx_adoption_probe.py"]
        (output/"source").mkdir();report["source_sha256"]={}
        for path in paths:shutil.copy2(path,output/"source"/path.name);report["source_sha256"][path.name]=digest(path)
        with (output/"config.json").open("x",encoding="utf-8") as stream:stream.write(json.dumps(config,indent=2)+"\n")
        report["raw_config_sha256"]=digest(output/"config.json")
        if not args.device:report["status"]="planned-only-no-device"
        else:
            report["macos_build"]=subprocess.check_output(["sw_vers","-buildVersion"],text=True).strip()
            report["sdk_version"]=subprocess.check_output(["xcrun","--sdk","macosx","--show-sdk-version"],text=True).strip()
            report["compiler_version"]=subprocess.check_output(["xcrun","clang++","--version"],text=True).splitlines()[0]
            binary=output/"qwen-block-shared"
            command=["xcrun","clang++","-std=c++20","-O2","-fobjc-arc","-Wall","-Wextra","-fvisibility=default","-mmacosx-version-min=26.2",
                "-I",str(ROOT.parent/"native"),"-I",str(ROOT.parent/"native/core"),"-I",str(ROOT.parent/"bindings/c/include"),
                "-isystem",str(mlx/"include"),str(output/"source/qwen_block_shared.mm"),"-L"+str(native),"-lturbocider","-L"+str(mlx/"lib"),
                "-lmlx","-ljaccl","-licucore","-lz","-framework","Foundation","-framework","Metal","-framework","IOSurface",
                "-Wl,-rpath,"+str(native),"-Wl,-rpath,"+str(mlx/"lib"),"-o",str(binary)]
            with (output/"build.stdout.log").open("x") as stdout,(output/"build.stderr.log").open("x") as stderr:
                child=subprocess.run(command,stdout=stdout,stderr=stderr,timeout=120)
            report["build_returncode"]=child.returncode;save()
            if child.returncode:raise ValueError("small block client build failed; inspect owned log")
            report["binary_sha256"]=digest(binary)
            environment={key:value for key,value in os.environ.items() if not key.startswith(("TURBOCIDER_","ANE_","ANEFORGE_","DYLD_","MPSGRAPH_","MLX_DISABLE_COMPILE"))}
            if args.temp_context=="owned":environment["TMPDIR"]=str(temporary)
            with (output/"native.stdout.log").open("x") as stdout,(output/"native.stderr.log").open("x") as stderr:
                child=subprocess.run([str(binary),str(output/"config.json"),str(output/"native.json")],env=environment,stdout=stdout,stderr=stderr,timeout=180)
            report["native_returncode"]=child.returncode
            if (output/"native.json").is_file():report["native"]=json.loads((output/"native.json").read_text());report["raw_native_sha256"]=digest(output/"native.json")
            report["status"]="measured" if child.returncode==0 else "native-failed"
            if child.returncode==0:
                if report["native"]["loaded_runtime_id"]!=build["runtime_build_id"]:raise ValueError("actual loaded runtime identity differs")
                report["comparisons"]=ratios(report["native"]["samples"],args.repeats)
                if args.lossless_archive:
                    archive=output/"ffn-bf16.tcbf16z";decoded=read_archive(archive)
                    if decoded!=report["native"]["bf16_archive"]:raise ValueError("native archive metadata differs from independent decode")
                    report["lossless_archive"]={"bytes":archive.stat().st_size,"sha256":digest(archive),"independent_decode_verified":True,
                        "input_sha256":decoded["input_sha256"],"reference_sha256":decoded["reference_sha256"]}
                elif not args.reuse_raw_from:
                    report["raw_artifacts"]={name:{"bytes":(output/name).stat().st_size,"sha256":digest(output/name)} for name in ("ffn-input.safetensors","ffn-reference.safetensors")}
        report["source_inputs_unchanged"]=all(digest(path)==report["source_sha256"][path.name] for path in paths)
        report["native_sources_verified_after"]=source_match(build)
        report["native_library_unchanged"]=digest(native/"libturbocider.dylib")==report["native_library_sha256"]
        report["mlx_assets_unchanged"]=before_assets==assets(mlx)
        report["original_model_stats_unchanged"]=before_stats==[{key:getattr(path.stat(),key) for key in item} for path,item in zip(source_files,before_stats)]
        report["private_bank_unchanged"]=digest(graph/"model.mil")==report["private_mil_sha256"] and digest(graph/"weights.bin")==report["private_blob_sha256"]
        if not all(report[key] for key in ("source_inputs_unchanged","native_library_unchanged","mlx_assets_unchanged","original_model_stats_unchanged","private_bank_unchanged")):
            raise ValueError("source/library/model/bank identity changed")
    except (Exception,KeyboardInterrupt) as error:
        report["status"]="failed";report["comparisons"]=None;report["errors"].append({"type":type(error).__name__,"message":str(error)})
    save();print(json.dumps({"status":report["status"],"case_id":args.case_id,"samples":len((report["native"] or {}).get("samples",[])),"errors":report["errors"]}))
    return 0 if report["status"] in ("measured","planned-only-no-device") else 1


if __name__=="__main__":raise SystemExit(main())

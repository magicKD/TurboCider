#!/usr/bin/env python3
"""Selected real ConvRot projections; synthetic inputs, not whole-model timing.

--prepare-only extracts exact source bytes and compiles without running the GPU.
--capture-only --cases dit_q --rows 128 records one projection in a
separate process and does not emit performance timings.
The timed route includes Comfy H256, projection, activation quantization (A8),
original output cast and bias. Both arms compile the same production Weights
interface; reference packs the original I8/F32 source to MLX affine Q8 and A8
retains original I8/F32 sources. No float weight bank is created.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import sysconfig

ROOT=Path(__file__).resolve().parents[2]
CASES={
    "encoder_q":("text_encoders/qwen3vl_8b_int8_convrot.safetensors",["model.layers.0.self_attn.q_proj"]),
    "encoder_gate_up":("text_encoders/qwen3vl_8b_int8_convrot.safetensors",["model.layers.0.mlp.gate_proj","model.layers.0.mlp.up_proj"]),
    "encoder_down":("text_encoders/qwen3vl_8b_int8_convrot.safetensors",["model.layers.0.mlp.down_proj"]),
    "dit_q":("diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors",["transformer_blocks.0.attn.to_q"]),
    "dit_gate_up":("diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors",["transformer_blocks.0.img_mlp.gate_up"]),
    "dit_down":("diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors",["transformer_blocks.0.img_mlp.out"]),
}

def extract(source:Path,prefixes:list[str],destination:Path)->dict:
    before=source.stat();header={};offset=0;selected=[]
    with source.open("rb") as handle:
        size=struct.unpack("<Q",handle.read(8))[0]
        if not 2<=size<=16<<20:raise ValueError("invalid source header size")
        raw=handle.read(size);index=json.loads(raw);data_base=8+size
        for i,prefix in enumerate(prefixes):
            for suffix in (".weight",".weight_scale",".comfy_quant",".bias"):
                key=prefix+suffix
                if suffix==".bias" and key not in index:continue
                tensor=index[key];start,end=tensor["data_offsets"]
                if not 0<=start<end<=before.st_size-data_base:raise ValueError("source tensor range invalid")
                if suffix==".weight" and tensor["dtype"]!="I8":raise ValueError("original I8 required")
                if suffix==".weight_scale" and tensor["dtype"]!="F32":raise ValueError("original F32 coefficient required")
                renamed=f"p{i}"+suffix
                header[renamed]={"dtype":tensor["dtype"],"shape":tensor["shape"],"data_offsets":[offset,offset+end-start]}
                selected.append((key,renamed,start,end));offset+=end-start
        encoded=json.dumps(header,separators=(",",":")).encode();encoded+=b" "*((-len(encoded))%8)
        tensors={}
        with destination.open("wb") as output:
            output.write(struct.pack("<Q",len(encoded)));output.write(encoded)
            for key,renamed,start,end in selected:
                handle.seek(data_base+start);remaining=end-start;digest=hashlib.sha256()
                while remaining:
                    block=handle.read(min(1<<20,remaining))
                    if not block:raise ValueError("truncated source tensor")
                    output.write(block);digest.update(block);remaining-=len(block)
                tensors[key]={"renamed":renamed,"sha256":digest.hexdigest(),**index[key]}
    after=source.stat()
    generation=lambda value:[value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns,value.st_ctime_ns]
    if generation(before)!=generation(after):raise ValueError("source changed during extraction")
    return {"source":str(source.resolve()),"generation":generation(after),"header_sha256":hashlib.sha256(raw).hexdigest(),"tensors":tensors}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",type=Path,default=ROOT/"models/Qwen-Image-2.1-Turbo-INT8")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--rows",type=int,nargs="+",default=[32,128,256,1024])
    parser.add_argument("--warmup",type=int,default=5);parser.add_argument("--repeats",type=int,default=20)
    parser.add_argument("--cases",nargs="+",choices=list(CASES),default=list(CASES))
    parser.add_argument("--prepare-only",action="store_true")
    parser.add_argument("--capture-only",action="store_true")
    args=parser.parse_args();args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=True)
    if args.warmup<1 or args.repeats<3 or any(row<1 for row in args.rows):parser.error("invalid run sizes")
    if args.capture_only and (args.prepare_only or len(args.cases)!=1 or len(args.rows)!=1):
        parser.error("capture-only requires exactly one case/row count and cannot be prepare-only")
    native=Path(os.environ.get("TURBOCIDER_TEST_NATIVE_DIR",ROOT/"build/native")).resolve()
    native_manifest=json.loads((native/"runtime-build/runtime-build-manifest.json").read_text())
    for source in native_manifest["inputs"]["sources"]:
        if hashlib.sha256((ROOT/source["path"]).read_bytes()).hexdigest()!=source["sha256"]:
            raise ValueError("C++ source/header ABI requires matching native tree: "+source["path"])
    mlx=Path(os.environ.get("MLX_ROOT",Path(sysconfig.get_paths()["purelib"])/"mlx"))
    compiler=subprocess.check_output(["xcrun","--find","clang++"],text=True).strip()
    sdk=os.environ.get("SDKROOT") or subprocess.check_output(["xcrun","--sdk","macosx","--show-sdk-path"],text=True).strip()
    binary=args.output/"projection-benchmark"
    subprocess.run([compiler,"-std=c++20","-O2","-Wall","-Wextra","-Werror","-Wno-unused-parameter",
        "-isysroot",sdk,"-mmacosx-version-min=26.2",'-DTC_TEST_BOUND_NATIVE_ID="'+native_manifest["runtime_build_id"]+'"',
        "-I",str(ROOT/"native/core"),"-I",str(ROOT/"bindings/c/include"),"-isystem",str(mlx/"include"),
        str(Path(__file__).with_suffix(".cpp")),"-L"+str(native),"-lturbocider","-L"+str(mlx/"lib"),"-lmlx","-ljaccl",
        "-framework","Foundation","-framework","Metal","-Wl,-rpath,"+str(native),"-Wl,-rpath,"+str(mlx/"lib"),"-o",str(binary)],check=True)
    def sha256(path):
        with path.open("rb") as handle:return hashlib.file_digest(handle,"sha256").hexdigest()
    library=native/"libturbocider.dylib"
    native_hash=sha256(library)
    manifest={"input":"synthetic BF16 independent normal(0,1), std::mt19937 seed20261106, two alternating fixed arrays; not captured model activations",
        "scope":"compiled production Weights::project; includes dense H256, projection, optional A8, original cast/bias; excludes loading/packing/compile",
        "encoder_gate_up":"two separate production projection calls with the same input, as in the encoder; MLX may share equivalent H256 expressions when compiling",
        "native_library":str(library),"native_library_sha256":native_hash,"driver_sha256":sha256(binary),
        "mlx_enable_tf32":os.environ.get("MLX_ENABLE_TF32","unset (runtime default)"),
        "rows":args.rows,"warmup":args.warmup,"repeats":args.repeats,"cases":{}}
    for name in args.cases:
        relative,prefixes=CASES[name];manifest["cases"][name]=extract(args.model/relative,prefixes,args.output/(name+".safetensors"))
    (args.output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    if args.prepare_only:
        print(json.dumps({"prepared":str(args.output),"gpu_executed":False,"cases":args.cases}));return
    if args.capture_only:
        name=args.cases[0];capture=args.output/(name+".gputrace")
        if capture.exists():raise ValueError("capture output already exists: "+str(capture))
        environment=dict(os.environ);environment["MTL_CAPTURE_ENABLED"]="1"
        record=subprocess.run([str(binary),str(args.output/(name+".safetensors")),str(len(CASES[name][1])),
            str(args.rows[0]),str(args.warmup),str(args.repeats),str(capture)],
            cwd=ROOT,env=environment,capture_output=True,text=True,timeout=300,check=True)
        value={"case":name,"capture_path":str(capture),**json.loads(record.stdout)}
        if not capture.is_dir():raise ValueError("Metal capture bundle was not produced")
        command_stream=(capture/"capture").read_bytes()
        labels=sorted(set(item.decode() for item in re.findall(rb"tc_convrot_(?:quantize|w8a8)_[A-Za-z0-9_]+",command_stream)))
        value["captured_command_stream_kernel_labels"]=labels
        value["capture_command_stream_sha256"]=hashlib.sha256(command_stream).hexdigest()
        if sha256(library)!=native_hash:raise ValueError("native library changed during capture")
        (args.output/"capture.json").write_text(json.dumps(value,indent=2)+"\n")
        gemm="tc_convrot_w8a8_f32_small" if args.rows[0]<=64 else "tc_convrot_w8a8_f32"
        if not {"tc_convrot_quantize_bf16",gemm}.issubset(labels):
            raise ValueError("required W8A8 dispatch labels were not observed in the capture command stream")
        print(json.dumps(value));return
    results=[]
    for name in args.cases:
        for rows in args.rows:
            record=subprocess.run([str(binary),str(args.output/(name+".safetensors")),str(len(CASES[name][1])),str(rows),str(args.warmup),str(args.repeats)],
                cwd=ROOT,capture_output=True,text=True,timeout=300,check=True)
            value={"case":name,**json.loads(record.stdout)};results.append(value);print(json.dumps(value),flush=True)
            (args.output/"results.json").write_text(json.dumps(results,indent=2,allow_nan=False)+"\n")
    if sha256(library)!=native_hash:raise ValueError("native library changed during the experiment")

if __name__=="__main__":main()

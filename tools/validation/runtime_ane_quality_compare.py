#!/usr/bin/env python3
"""CPU-only matched generation dumps; N1 is not media/device qualification."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

import numpy as np

from runtime_ane_common import sha256_file,validate_results,validate_fp32_channel_join,validate_fp16_bf16_values,validate_row_placement
from runtime_ane_image_compare import compare_png

LAYOUTS = {
    "z-image-turbo": (("z_conditioning", "z_latent_initial"), "z_latent_final"),
    "z-image-turbo-gguf": (("z_conditioning", "z_latent_initial"), "z_latent_final"),
    "qwen-image-2.1": (("qwen21_text", "qwen21_initial"), "qwen21_latents"),
}


def load_tensor(path):
    """Bounded single-tensor native dump, including BF16; never imports MLX."""
    path=Path(path)
    before=sha256_file(path)
    with path.open("rb") as source:
        prefix=source.read(8)
        if len(prefix)!=8:
            raise ValueError("truncated native tensor header")
        length=struct.unpack("<Q",prefix)[0]
        if not 0<length<=16<<20:
            raise ValueError("native tensor header exceeds bound")
        header=json.loads(source.read(length))
        if not isinstance(header,dict) or set(header)-{"tensor","__metadata__"}:
            raise ValueError("expected a single native diagnostic tensor")
        tensor=header.get("tensor")
        if not isinstance(tensor,dict):
            raise ValueError("native tensor descriptor missing")
        shape=tensor.get("shape")
        if (not isinstance(shape,list) or not 1<=len(shape)<=8 or
                any(type(x) is not int or x<=0 for x in shape)):
            raise ValueError("invalid native tensor geometry")
        count=math.prod(shape)
        if count>1<<25:
            raise ValueError("native diagnostic tensor exceeds element bound")
        dtype=tensor.get("dtype")
        if dtype not in ("F32","F16","BF16"):
            raise ValueError("unsupported native diagnostic dtype")
        offsets=tensor.get("data_offsets")
        item=4 if dtype=="F32" else 2
        if (not isinstance(offsets,list) or len(offsets)!=2 or
                any(type(x) is not int for x in offsets) or offsets[0]<0 or
                offsets[1]-offsets[0]!=count*item or offsets[1]>path.stat().st_size-8-length):
            raise ValueError("native tensor byte range/geometry mismatch")
        source.seek(8+length+offsets[0]);raw=source.read(count*item)
        if len(raw)!=count*item:
            raise ValueError("truncated native tensor payload")
    if dtype=="BF16":
        values=(np.frombuffer(raw,dtype="<u2").astype(np.uint32)<<16).view(np.float32).astype(np.float64)
    else:
        values=np.frombuffer(raw,dtype="<f4" if dtype=="F32" else "<f2").astype(np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("nonfinite native diagnostic tensor")
    if sha256_file(path)!=before:
        raise ValueError("native tensor changed during comparison")
    return values,dict(shape=shape,dtype=dtype,file_sha256=before,payload_sha256=hashlib.sha256(raw).hexdigest())


def pair(reference,candidate):
    a,ai=load_tensor(reference);b,bi=load_tensor(candidate)
    if ai["shape"]!=bi["shape"] or ai["dtype"]!=bi["dtype"]:
        raise ValueError("native source/candidate geometry or dtype differs")
    aa,bb,dot,delta=(float(x) for x in (np.dot(a,a),np.dot(b,b),np.dot(a,b),np.dot(a-b,a-b)))
    if not all(math.isfinite(x) for x in (aa,bb,dot,delta)) or aa<=0 or bb<=0:
        raise ValueError("invalid native comparison statistics or zero-energy tensor")
    return dict(reference=ai,candidate=bi,exact=ai["payload_sha256"]==bi["payload_sha256"],
                rel_l2=math.sqrt(delta/aa),cosine=dot/(math.sqrt(aa)*math.sqrt(bb)),
                max_abs=float(np.max(np.abs(a-b))))


def compare_generation(reference,candidate,model_id):
    if model_id not in LAYOUTS:
        raise ValueError("unsupported native generation comparison model")
    reference,candidate=Path(reference),Path(candidate)
    inputs,final=LAYOUTS[model_id]
    result=dict(schema="tc-runtime-ane-generation-quality-v1",model=model_id,
        scope="matched CPU FP64 native generation dumps; local N1 only; not perceptual/semantic/device qualification",
        qualification_passed=False,inputs={},trajectory=[],gates=dict(max_rel_l2=.03,min_cosine=.999))
    for key in inputs:
        measured=pair(reference/(key+".safetensors"),candidate/(key+".safetensors"))
        if not measured["exact"]:
            raise ValueError("conditioning/initial state does not match: "+key)
        result["inputs"][key]=measured
    if model_id in ("z-image-turbo","z-image-turbo-gguf"):
        files=[{p.name for p in folder.glob("z_latent_step_*.safetensors")} for folder in (reference,candidate)]
        if not files[0] or files[0]!=files[1]:
            raise ValueError("native Z trajectory dumps incomplete/unmatched")
        steps=sorted(int(name.removeprefix("z_latent_step_").removesuffix(".safetensors")) for name in files[0])
        if steps!=list(range(1,len(steps)+1)) or len(steps)>50:
            raise ValueError("native Z trajectory steps are not complete")
        for step in steps:
            name=f"z_latent_step_{step}.safetensors"
            result["trajectory"].append(dict(step=step,**pair(reference/name,candidate/name)))
    measured=pair(reference/(final+".safetensors"),candidate/(final+".safetensors"))
    result["final_latent"]=measured
    result["n1_final_latent_pass"]=measured["rel_l2"]<=.03 and .999<=measured["cosine"]<=1.000001
    return result


def bind_execution(reference,candidate,model_id,*,runtime_backend="private",data_path="w8a8",channel_auto=True,device_io=True,gpu_control=False,
                   fp16_bf16_values=False,row_placement="suffix"):
    if runtime_backend not in ("private","public") or data_path not in ("fp16","w8a8","convrot_w8a8","gguf_raw_w8a8"):
        raise ValueError("unsupported quality execution backend/data path")
    if any(type(flag) is not bool for flag in (channel_auto,device_io,gpu_control,fp16_bf16_values)):
        raise ValueError("quality execution policy must be explicit booleans")
    if (channel_auto and (runtime_backend!="private" or data_path not in ("w8a8","gguf_raw_w8a8"))) or (device_io and runtime_backend!="private" and data_path not in ("w8a8","convrot_w8a8","gguf_raw_w8a8")):
        raise ValueError("unsupported quality execution channel/device policy")
    if data_path=="convrot_w8a8" and model_id!="z-image-turbo":
        raise ValueError("ConvRot quality route requires explicit Z model, not ordinary GGUF/Qwen")
    if data_path=="gguf_raw_w8a8" and model_id!="z-image-turbo-gguf":
        raise ValueError("raw GGUF quality route requires explicit GGUF model")
    if gpu_control and (model_id!="z-image-turbo" or channel_auto or device_io):
        raise ValueError("GPU boundary control requires dense Z and no channel/device offload")
    if fp16_bf16_values and (runtime_backend!="private" or data_path!="fp16" or channel_auto or gpu_control):
        raise ValueError("FP16 BF16 values require explicitly requested Private FP16 rows")
    if row_placement not in ("suffix","image_prefix","image_tail") or (row_placement!="suffix" and
            (model_id!="z-image-turbo" or data_path!="fp16" or channel_auto or gpu_control)):
        raise ValueError("image-only row placement requires explicit dense Z FP16 runtime rows")
    records=[]
    for path,route in ((reference,"gpu"),(candidate,"gpu_control" if gpu_control else "runtime")):
        path=Path(path);digest=sha256_file(path)
        raw=json.loads(path.read_text())
        if type(raw.get("exit_code")) is not int or raw["exit_code"]!=0 or not isinstance(raw.get("stdout"),str):
            raise ValueError("native quality execution receipt incomplete")
        def valid_digest(value):
            return isinstance(value,str) and len(value)==64 and all(c in "0123456789abcdef" for c in value)
        if not valid_digest(raw.get("binary_sha256")):
            raise ValueError("native quality binary identity missing/invalid")
        library=raw.get("adjacent_library_sha256")
        if library is not None and (not valid_digest(library) or raw.get("artifacts_unchanged") is not True):
            raise ValueError("native quality adjacent library identity missing/changed")
        if "artifacts_unchanged" in raw and raw["artifacts_unchanged"] is not True:
            raise ValueError("native quality execution artifacts changed")
        rows=[]
        for line in raw["stdout"].splitlines():
            try:row=json.loads(line)
            except ValueError:continue
            if isinstance(row,dict) and "runtime_backend" in row:rows.append(row)
        if len(rows)!=1 or rows[0].get("model")!=model_id:
            raise ValueError("native quality execution workload/result count mismatch")
        row=rows[0]
        if any(type(row.get(key)) is not int or row[key]<=0 for key in ("width","height","steps","actual_denoise_steps")):
            raise ValueError("native quality actual geometry/steps missing/invalid")
        if row["steps"]!=row["actual_denoise_steps"]:
            raise ValueError("native quality requested and actual steps differ")
        validation_rows=rows
        if data_path=="gguf_raw_w8a8":
            imported=row.get("gguf_import") or {}
            keys=("raw_window_budget_bytes","raw_window_peak_bytes","raw_window_live_bytes","raw_window_entries",
                "raw_window_misses_session_total","raw_window_source_read_bytes_session_total")
            if (imported.get("recipe")!="gguf-mlx-compat-affine-packed-bank-v1" or
                    imported.get("ane_weight_source")!="bounded-raw-ggml-window-v1" or
                    not valid_digest(imported.get("source_sha256")) or
                    any(type(imported.get(key)) is not int or imported[key]<0 for key in keys) or
                    imported["raw_window_budget_bytes"]!=256<<20 or
                    imported["raw_window_peak_bytes"]>imported["raw_window_budget_bytes"] or
                    imported["raw_window_live_bytes"] or imported["raw_window_entries"]):
                raise ValueError("raw GGUF quality source/window identity or completed-drain evidence missing")
            if route=="gpu":
                if (row.get("runtime_backend")!="mlx_cpp_metal_gguf_cpu_direct_packed" or
                        row.get("runtime_precision")!="z-mlx-compat-affine-v1" or
                        imported["raw_window_misses_session_total"] or imported["raw_window_source_read_bytes_session_total"]):
                    raise ValueError("raw GGUF quality requires matched native-affine GPU control without raw ANE reads")
                validation_rows=[dict(row,runtime_backend="mlx_cpp_metal_gguf")]
            elif (not imported["raw_window_misses_session_total"] or not imported["raw_window_source_read_bytes_session_total"] or
                    not imported["raw_window_peak_bytes"]):
                raise ValueError("raw GGUF candidate never consumed raw source weights")
        if data_path=="convrot_w8a8":
            expected_label="mlx_cpp_metal_convrot_packed_q8" if route=="gpu" else (
                "mlx_cpp_metal_convrot+coreml_runtime_weight_experimental" if runtime_backend=="public" else
                "mlx_cpp_metal_convrot+private_ane_runtime_weight_experimental")
            checkpoint=row.get("checkpoint")
            if (row.get("runtime_backend")!=expected_label or not isinstance(checkpoint,str) or
                    not checkpoint.endswith(".safetensors") or "convrot" not in checkpoint.lower() or
                    (route=="gpu" and row.get("runtime_precision")!="int8_tensorwise_convrot_g256")):
                raise ValueError("explicit ConvRot execution lacks packed source/actual backend binding")
            # Only this explicitly validated native source label is mapped
            # to the shared validator's canonical label; retain originals.
            canonical="mlx_cpp_metal" if route=="gpu" else (
                "mlx_cpp_metal+coreml_runtime_weight" if runtime_backend=="public" else
                "mlx_cpp_metal+private_ane_runtime_weight_experimental")
            validation_rows=[dict(row,runtime_backend=canonical)]
        if gpu_control:
            expected_backend="mlx_cpp_metal_dense_split_gpu_control" if route=="gpu_control" else "mlx_cpp_metal"
            expected_graph="compiled_split_gpu_ffn_control" if route=="gpu_control" else "compiled_fused_blocks"
            if (row.get("runtime_backend")!=expected_backend or
                    row.get("gpu_graph")!=expected_graph or row.get("hybrid") or
                    row.get("execution",(row.get("plan") or {}).get("execution"))!="gpu" or
                    row.get("encoder_execution")!="gpu" or
                    row.get("encoder_runtime_backend")!="mlx_cpp_metal" or row.get("encoder_hybrid") or
                    row.get("lora_strategy")!="none" or
                    type(row.get("lora_applied_projections",0)) is not int or row.get("lora_applied_projections",0)!=0):
                raise ValueError("GPU boundary control lacks an exclusively GPU execution receipt")
            # Only this explicitly checked diagnostic label is normalized for
            # the shared GPU timing validator. Preserve the actual label below.
            validation_rows=[dict(row,runtime_backend="mlx_cpp_metal")]
        validate_results(validation_rows,"gpu" if route=="gpu_control" else route,1,model_id=model_id,runtime_backend=runtime_backend,expect_device_io=route=="runtime" and device_io,
            expected_data_path={"w8a8":"w8a8_hadamard","gguf_raw_w8a8":"w8a8_hadamard","convrot_w8a8":"w8a8_convrot","fp16":"fp16"}[data_path] if route=="runtime" else None,
            channel_auto=route=="runtime" and channel_auto)
        if route=="runtime":
            validate_row_placement(rows,row_placement)
            validate_fp16_bf16_values(rows,fp16_bf16_values)
            runtime=(row.get("hybrid") or {}).get("runtime_weight") or {}
            validate_fp32_channel_join(rows,runtime.get("fp32_channel_join_enabled",False),allow_gpu_decline=True)
        if sha256_file(path)!=digest:raise ValueError("quality execution receipt changed")
        records.append(dict(receipt_sha256=digest,binary_sha256=raw.get("binary_sha256"),
            adjacent_library_sha256=library,artifacts_unchanged=raw.get("artifacts_unchanged"),
            runtime_backend=row["runtime_backend"],runtime_precision=row.get("runtime_precision"),checkpoint=row.get("checkpoint"),
            gguf_import=row.get("gguf_import"),
            gpu_graph=row.get("gpu_graph"),hybrid=row.get("hybrid"),model=row["model"],
            width=row["width"],height=row["height"],seed=row["seed"],steps=row["steps"],actual_denoise_steps=row["actual_denoise_steps"],
            lora_strategy=row.get("lora_strategy"),lora_applied_projections=row.get("lora_applied_projections")))
    for key in ("binary_sha256","adjacent_library_sha256","model","checkpoint","width","height","seed","steps","actual_denoise_steps","lora_strategy","lora_applied_projections"):
        if records[0][key]!=records[1][key]:raise ValueError("unmatched quality execution field: "+key)
    if data_path=="gguf_raw_w8a8":
        for key in ("source_sha256","plan_digest","affine_packing_recipe","float_import_recipe"):
            if not records[0]["gguf_import"].get(key) or records[0]["gguf_import"][key]!=records[1]["gguf_import"].get(key):
                raise ValueError("unmatched raw GGUF source/import recipe: "+key)
    candidate_hybrid=records[1]["hybrid"] or {}
    executor=(candidate_hybrid.get("runtime_weight") or {}).get("executor_backend")
    calls=candidate_hybrid.get("runtime_calls_session_total",0)
    active=executor=="private_ane" and calls>0
    return dict(reference=records[0],candidate=records[1],candidate_ane_executed=active,
                candidate_coreml_executed=executor=="public_coreml" and calls>0,
                requested_backend="gpu_only" if gpu_control else runtime_backend,
                requested_data_path="bf16_gpu" if gpu_control else data_path,
                gpu_boundary_control=gpu_control,fp16_bf16_values=fp16_bf16_values,row_placement=row_placement,
                channel_auto=channel_auto,device_io=device_io,
                adjacent_runtime_identity_bound=records[0]["adjacent_library_sha256"] is not None,
                scope="native execution receipts; no performance or physical-engine qualification")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference",type=Path,required=True)
    parser.add_argument("--candidate",type=Path,required=True)
    parser.add_argument("--model-id",choices=tuple(LAYOUTS),required=True)
    parser.add_argument("--reference-png",type=Path)
    parser.add_argument("--candidate-png",type=Path)
    parser.add_argument("--reference-receipt",type=Path)
    parser.add_argument("--candidate-receipt",type=Path)
    parser.add_argument("--runtime-backend",choices=("private","public"),default="private")
    parser.add_argument("--runtime-data-path",choices=("fp16","w8a8","convrot_w8a8","gguf_raw_w8a8"),default="w8a8")
    parser.add_argument("--channel-auto",choices=("0","1"),default="1")
    parser.add_argument("--device-io",choices=("0","1"),default="1")
    parser.add_argument("--gpu-only-control",action="store_true",
        help="explicit dense Z GPU split control; also requires --channel-auto 0 --device-io 0")
    parser.add_argument("--fp16-bf16-values",action="store_true",help="explicit actual Private FP16 base-only compact value-rounding recipe")
    parser.add_argument("--row-placement",choices=("suffix","image_prefix","image_tail"),default="suffix")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists() or args.output.is_symlink():parser.error("output already exists")
    if bool(args.reference_png)!=bool(args.candidate_png):parser.error("provide both matched PNGs")
    if bool(args.reference_receipt)!=bool(args.candidate_receipt):parser.error("provide both execution receipts")
    if args.gpu_only_control and not args.reference_receipt:parser.error("GPU boundary control requires execution receipts")
    if args.fp16_bf16_values and not args.reference_receipt:parser.error("FP16 BF16 values require execution receipts")
    if args.row_placement!="suffix" and not args.reference_receipt:parser.error("row placement requires execution receipts")
    result=compare_generation(args.reference,args.candidate,args.model_id)
    if args.reference_receipt:
        result["execution"]=bind_execution(args.reference_receipt,args.candidate_receipt,args.model_id,
            runtime_backend=args.runtime_backend,data_path=args.runtime_data_path,channel_auto=args.channel_auto=="1",device_io=args.device_io=="1",
            gpu_control=args.gpu_only_control,fp16_bf16_values=args.fp16_bf16_values,row_placement=args.row_placement)
        if args.model_id in ("z-image-turbo","z-image-turbo-gguf") and len(result["trajectory"])!=result["execution"]["reference"]["actual_denoise_steps"]:
            raise ValueError("native Z dumps do not cover all executed steps")
        result["n1_with_actual_ane"]=result["n1_final_latent_pass"] and result["execution"]["candidate_ane_executed"]
        result["n1_with_actual_coreml"]=result["n1_final_latent_pass"] and result["execution"]["candidate_coreml_executed"]
    if args.reference_png:result["png"]=compare_png(args.reference_png,args.candidate_png)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x") as output:json.dump(result,output,indent=2,allow_nan=False);output.write("\n")
    print(json.dumps(dict(n1_final_latent_pass=result["n1_final_latent_pass"],**result["final_latent"]),allow_nan=False))


if __name__=="__main__":main()

#!/usr/bin/env python3
"""CPU/source/raw recheck for the complete QKV+Private block composition."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np

from flux9_bf16_framework_screen import write_json
from qwen_qkv_lora_block_audit import tensors,difference,same_quality
import qwen_qkv_private_combo_screen as screen
from runtime_ane_common import sha256_file

ROOT=Path(__file__).resolve().parents[2]


def packed_control_geometry(name,value):
    widths={"partial_reference":4096,"partial_packed":4096,"hidden_reference":10240,"hidden_packed":10240}
    if name not in widths:raise ValueError("unknown packed control consumer")
    width=widths[name];a,b=value["data_offsets"]
    if (value["dtype"]!="BF16" or value["shape"]!=[1,1024,width] or
            type(a) is not int or type(b) is not int or a<0 or b-a!=1024*width*2):
        raise ValueError("complete typed packed control geometry required")
    return value["shape"],a,b


def packed_control_tensors(path):
    # The ordinary block reader intentionally remains fixed at H4096. This
    # separate ABI admits H4096 partials and F10240 hidden, by exact key only.
    with path.open("rb") as stream:
        length=struct.unpack("<Q",stream.read(8))[0]
        if not 2<=length<=1<<20:raise ValueError("bounded packed control header required")
        header=json.loads(stream.read(length));header.pop("__metadata__",None)
        if set(header)!={"partial_reference","partial_packed","hidden_reference","hidden_packed"}:
            raise ValueError("all complete packed controls required")
        result={};regions=[]
        for name,value in header.items():
            shape,a,b=packed_control_geometry(name,value)
            stream.seek(8+length+a);payload=stream.read(b-a)
            if len(payload)!=b-a:raise ValueError("truncated packed control")
            regions.append((a,b));result[name]=np.frombuffer(payload,dtype="<u2").reshape(shape).copy()
        regions.sort()
        if regions[0][0]!=0 or any(left[1]!=right[0] for left,right in zip(regions,regions[1:])) or\
                8+length+regions[-1][1]!=path.stat().st_size:
            raise ValueError("overlapping/gapped/extra packed control payload")
        return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--library",type=Path,required=True);p.add_argument("--cohort",type=Path,action="append",required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args();args.library=args.library.resolve(strict=True)
    if args.output.exists():p.error("fresh audit output required")
    manifest=json.loads((args.library.parent/"runtime-build/runtime-build-manifest.json").read_text())
    for item in manifest["inputs"]["sources"]:
        if sha256_file(ROOT/item["path"])!=item["sha256"]:raise ValueError("native source/header differs from loaded library")
    observations=[];sources=None;constants=None;case_inputs={}
    for root in args.cohort:
        root=root.resolve(strict=True);s=json.loads((root/"summary.json").read_text());n=json.loads((root/"trial/native.json").read_text())
        if s["status"]!="complete" or s["library_sha256"]!=sha256_file(args.library) or n["loaded_runtime_id"]!=manifest["runtime_build_id"]:
            raise ValueError("complete same-library actual block required")
        snapshot=json.loads((root/"source.json").read_text())
        current_source_matches=True
        for name,digest in s["source_sha256"].items():
            if hashlib.sha256(snapshot[name].encode()).hexdigest()!=digest:
                raise ValueError("bound composition source changed")
            current_source_matches &= sha256_file(ROOT/name)==digest
        actual=screen.analyze(n)
        if actual!=s["analysis"]:raise ValueError("complete timing denominator or gates differ")
        memory=s["memory"]
        for name in ("evidence","report"):
            if sha256_file(root/memory[name])!=memory[name+"_sha256"]:raise ValueError("memory artifact changed")
        if not memory["verified"]["complete"] or memory["verified"]["command_exit_code"]!=0:raise ValueError("incomplete process window")
        refs=tensors(root/"trial/inputs-reference.safetensors");outputs=tensors(root/"trial/complete-outputs.safetensors")
        names,_,_,_=screen.recipe(n)
        feed_names={"private":"private_ffn","private-qkv":"private_qkv_ffn"}
        if n.get("schedule_recipe")=="producer-fence-v1":
            feed_names.update({"private-fenced":"private_fenced_ffn","private-qkv-fenced":"private_qkv_fenced_ffn"})
        elif n.get("schedule_recipe") in ("producer-event-v2","submission-order-v3","packed-suffix-v4"):
            feed_names.update({"private-event":"private_event_ffn","private-qkv-event":"private_qkv_event_ffn"})
            if n.get("schedule_recipe")=="submission-order-v3":
                feed_names.update({"private-gpu-first":"private_gpu_first_ffn","private-qkv-gpu-first":"private_qkv_gpu_first_ffn"})
            elif n.get("schedule_recipe")=="packed-suffix-v4":
                feed_names.update({"private-packed":"private_packed_ffn","private-qkv-packed":"private_qkv_packed_ffn"})
        if set(refs)!={"input","block_reference","ffn_input_original","ffn_input_qkv","ffn_reference_original","ffn_reference_qkv"} or\
                set(outputs)!=set(names)|set(feed_names.values()):
            raise ValueError("complete full-block and both FFN chains required")
        input_hash=hashlib.sha256(refs["input"].tobytes()).hexdigest();ref_hash=hashlib.sha256(refs["block_reference"].tobytes()).hexdigest()
        if input_hash!=n["input_sha256"] or ref_hash!=n["ordinary_block_sha256"]:raise ValueError("actual input/reference changed")
        packed_controls=None
        if n.get("schedule_recipe")=="packed-suffix-v4":
            controls=packed_control_tensors(root/"trial/packed-suffix-controls.safetensors")
            if set(controls)!={"partial_reference","hidden_reference","partial_packed","hidden_packed"}:
                raise ValueError("complete packed suffix controls required")
            packed_controls=[]
            for key,width in (("partial",4096),("hidden",10240)):
                if controls[key+"_reference"].shape!=(1,1024,width):raise ValueError("complete GPU suffix control extent differs")
                q=difference(controls[key+"_reference"],controls[key+"_packed"])
                if q["changed"]!=0:raise ValueError("packed GPU suffix control is not exact")
                packed_controls.append(dict(consumer=key,quality=q))
        raw=[]
        for name in names:
            q=difference(refs["block_reference"],outputs[name]);last=[row for row in n["samples"] if row["arm"]==name][-1]
            same_quality(q,last["quality"]);screen.quality(q,.001,name in screen.NAMES[:2]);raw.append(dict(arm=name,quality=q))
            if name.startswith("private"):
                feed=outputs[feed_names[name]]
                matched=difference(refs["ffn_reference_qkv" if "qkv" in name else "ffn_reference_original"],feed)
                chained=difference(refs["ffn_reference_original"],feed)
                same_quality(matched,last["matched_input_ffn_quality"]);same_quality(chained,last["original_chain_ffn_quality"])
                screen.quality(matched,.03);screen.quality(chained,.03);raw[-1].update(matched_ffn=matched,chained_ffn=chained)
        if packed_controls is not None:
            for candidate,control in (("private-packed","private"),("private-qkv-packed","private-qkv")):
                if not np.array_equal(outputs[candidate],outputs[control]) or not np.array_equal(outputs[feed_names[candidate]],outputs[feed_names[control]]):
                    raise ValueError("packed suffix changed complete joined block/FFN output")
        if sources is None:sources=n["actual_layer_source_sha256"]
        elif sources!=n["actual_layer_source_sha256"]:raise ValueError("base/factor source changed between composition cases")
        code=(n["private_constants_sha256"],n["private_mil_sha256"])
        if constants is None:constants=code
        elif constants!=code:raise ValueError("static Private model changed between cases")
        key=s["case"]
        if key in case_inputs and case_inputs[key]!=(input_hash,ref_hash):raise ValueError("same task input/reference changed on repetition")
        case_inputs[key]=(input_hash,ref_hash)
        loads=[json.loads(line) for line in (root/"load.jsonl").read_text().splitlines()]
        observations.append(dict(cohort=root.name,case=key,summary_sha256=sha256_file(root/"summary.json"),analysis=actual,
            current_source_matches_snapshot=bool(current_source_matches),
            raw_quality=raw,private_calls=n["successful_private_calls"],private_setup_seconds=n["private_setup_seconds"],
            packed_suffix_controls=packed_controls,packed_suffix_setup_seconds=n.get("packed_suffix_setup_seconds"),
            qkv_setup_seconds=n["qkv_pack_and_verify_seconds"],input_sha256=input_hash,reference_sha256=ref_hash,
            memory_peak=memory["verified"]["tree_peak_phys_footprint_bytes"],swap_in=memory["verified"]["swap_in_bytes"],swap_out=memory["verified"]["swap_out_bytes"],
            competing_load_samples=sum(bool(row["competing_processes"]) for row in loads),load_error=s.get("load_error")))
    result=dict(schema="tc-qwen-qkv-private-combo-audit-v1",status="complete",qualification_passed=False,
        library_sha256=sha256_file(args.library),runtime_build_id=manifest["runtime_build_id"],native_source_hashes_verified=len(manifest["inputs"]["sources"]),
        actual_layer_source_hash_count=len(sources or {}),same_static_model_and_source=True,observations=observations,
        script_sha256=sha256_file(Path(__file__)),scope="CPU/raw/source audit only; one layer, no full-image or physical-engine qualification")
    write_json(args.output,result);print(json.dumps(result,indent=2))


if __name__=="__main__":main()

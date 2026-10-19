#!/usr/bin/env python3
"""Independent three-arm full40 text-prefix/first-QKV counterfactual audit."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import tarfile

import numpy as np
from PIL import Image,ImageDraw

import analyze
import collect
import joint_audit
import text_prefix_audit
from audit import normalized,pixels,quality
from collect import ROOT,model_snapshot,sha256_file,write_json


def only_policy(a,b):
    left,right=a["summary"],b["summary"]
    if (set(left["environment"])!=set(right["environment"]) or
            {k for k in left["environment"] if left["environment"][k]!=right["environment"][k]}!={"TURBOCIDER_QWEN21_RUNTIME_TEXT_PREFIX_QKV_STEP0"} or
            any(left["assets"][k]!=right["assets"][k] for k in ("library","manifest","reference"))):
        raise ValueError("counterfactual may change only the explicit first-step QKV policy")
    for i in range(3):
        if normalized(a["root"]/f"trial/request-{i}.json")!=normalized(b["root"]/f"trial/request-{i}.json"):
            raise ValueError("counterfactual changed actual prompt/seed/geometry")
    if not np.array_equal(a["images"][0],b["images"][0]):raise ValueError("first-step policy changed an uncached miss")


def frozen(root,value):
    for i in range(3):
        if normalized(root/f"request-{i}.json")!=normalized(value["root"]/f"trial/request-{i}.json") or not np.array_equal(pixels(root/f"{i}.png"),value["images"][i]):
            raise ValueError("unselected/default native arithmetic did not restore actual frozen requests and RGBA")
    return True


def cached_gpu_pair(off,gpu,*,require_exact=False):
    left,right=off["summary"],gpu["summary"]
    if (set(left["environment"])!=set(right["environment"]) or
            {k for k in left["environment"] if left["environment"][k]!=right["environment"][k]}!={
                "TURBOCIDER_QWEN21_RUNTIME_TEXT_PREFIX_SNAPSHOT","TURBOCIDER_QWEN21_RUNTIME_TEXT_PREFIX_QKV_STEP0"} or
            any(left["assets"][k]!=right["assets"][k] for k in ("library","manifest","reference"))):
        raise ValueError("GPU cache pair may change only cache enable and declared first-step policy")
    for i in range(3):
        if normalized(off["root"]/f"trial/request-{i}.json")!=normalized(gpu["root"]/f"trial/request-{i}.json"):
            raise ValueError("GPU cache pair changed actual source requests")
    result=[quality(a,b) for a,b in zip(off["images"],gpu["images"])]
    if len(result)!=3 or not result[0]["rgba_exact"] or (require_exact and not all(q["rgba_exact"] for q in result)):
        raise ValueError("GPU cache cold-miss or selected complete-RGBA-exact gate failed")
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("off","ane","gpu","library","native-source-archive","output"):p.add_argument("--"+name,type=Path,required=True)
    for name in ("reverse-off","repeat-gpu","timeline","matrix","edit","historical-off","historical-ane","historical-edit"):
        p.add_argument("--"+name,type=Path)
    p.add_argument("--tool-source-archive",type=Path,action="append",default=[])
    p.add_argument("--additional-pair",nargs=3,action="append",default=[],metavar=("LABEL","OFF","GPU"))
    p.add_argument("--require-gpu-exact",action="store_true",help="all actual same-seed GPU-policy RGBA must equal the off control")
    args=p.parse_args()
    if args.output.exists():p.error("fresh audit directory required")
    manifest=json.loads((args.library.parent/"runtime-build/runtime-build-manifest.json").read_text())
    joint_audit.archive_contract(args.native_source_archive,manifest);historical={}
    for path in args.tool_source_archive:
        with tarfile.open(path,"r:gz") as archive:
            for item in archive.getmembers():
                if item.isfile():historical.setdefault(item.name,set()).add(archive.extractfile(item).read())
    def window(root,text,policy,mode="stock",*,edit=False,matrix=False):
        summary=json.loads((root/"summary.json").read_text());report=json.loads((root/"trial/report.json").read_text())
        count=4 if matrix else 3;scenario="text-prefix-matrix" if matrix else "repeat";case="edit" if edit else "generate"
        expected=collect.environment("runtime",mode,6144,edit,False,True,True,"descriptor","sigmoid-complement",runtime_text_prefix=text,text_prefix_qkv_step0=policy)
        if (summary["status"]!="complete" or report["status"]!="complete" or summary["route"]!="runtime" or
                summary["case"]!=case or summary["mode"]!=mode or summary["scenario"]!=scenario or len(report["runs"])!=count or
                summary["channels"]!=6144 or summary.get("runtime_text_prefix_snapshot") is not text or
                summary.get("text_prefix_qkv_step0")!=policy or report.get("text_prefix_qkv_step0")!=policy or
                summary.get("runtime_prefix_snapshot") is not edit or summary.get("resident_prefix") is not False or
                summary.get("private_qkv") is not True or summary.get("fixed_program") is not True or
                summary.get("ffn_projection")!="descriptor" or summary.get("ffn_activation")!="sigmoid-complement" or
                summary["environment"]!=expected or summary["assets"]["library"]!=sha256_file(args.library) or
                report["loaded_runtime_id"]!=manifest["runtime_build_id"] or
                summary["model_snapshot"]!=model_snapshot(ROOT/"models/Comfy-Org-Qwen-Image-2.1")):
            raise ValueError("complete same-native original-base explicit first-step recipe required")
        source=json.loads((root/"source.json").read_text())
        if set(source)!=set(summary["source_sha256"]):raise ValueError("complete profiling source snapshot required")
        for name,digest in summary["source_sha256"].items():
            raw=source[name].encode()
            if hashlib.sha256(raw).hexdigest()!=digest or not (raw==(ROOT/name).read_bytes() or raw in historical.get(name,set())):
                raise ValueError("bound current/archive profiling source differs")
        metrics=[r["metrics"] for r in report["runs"]]
        collect.validate(metrics,"runtime",mode,6144,edit,False,True,True,"descriptor","sigmoid-complement",runtime_text_prefix=text,text_prefix_matrix=matrix,text_prefix_qkv_step0=policy)
        previous_ffn=previous_qkv=previous_gpu=0;counts=[]
        for i,run in enumerate(report["runs"]):
            request=json.loads((root/f"trial/request-{i}.json").read_text());m=run["metrics"]
            if (run["index"]!=i or request.get("operation")!="image."+case or request.get("seed")!=([73,74,73,73] if matrix else [73,74,73])[i] or
                    request.get("loras") or request.get("audio") is not False or
                    (edit and (len(request.get("inputs",[]))!=1 or request.get("qwen21_reference_size")!=1024)) or
                    (not edit and request.get("inputs")) or sha256_file(Path(request["ane_manifest"]))!=summary["assets"]["manifest"]):
                raise ValueError("actual complete source requests/descriptor differ")
            f=m["hybrid"]["runtime_calls_session_total"];q=m["qwen_private_qkv"]["predictions_session_total"];g=m["qwen_private_qkv"]["gpu_blocks_session_total"]
            if not text and not edit and f-previous_ffn!=1312:raise ValueError("off unexpectedly changed actual FFN chunks")
            counts.append(dict(index=i,ffn=f-previous_ffn,qkv=q-previous_qkv,gpu_qkv_blocks=g-previous_gpu))
            previous_ffn,previous_qkv,previous_gpu=f,q,g
        images=[pixels(root/f"trial/{i}.png") for i in range(count)]
        if any(x.shape!=(512,512,4) for x in images) or np.array_equal(images[0],images[1]):raise ValueError("complete changed-seed RGBA images required")
        if not text and not edit and not np.array_equal(images[0],images[2]):raise ValueError("off failed seed restoration")
        for key in ("report","evidence"):
            if sha256_file(root/summary["memory"][key])!=summary["memory"][key+"_sha256"]:raise ValueError("owned memory evidence changed")
        if not summary["memory"]["verified"]["complete"] or summary["memory"]["verified"]["command_exit_code"]!=0:
            raise ValueError("complete successful owned model window required")
        observer=None
        if mode=="timeline":
            raw=json.loads((root/"trial/observer.json").read_text())
            if raw["source_sha256"]!=summary["source_sha256"]["tools/profiling/qwen_base/observer.mm"]:raise ValueError("actual observer source differs")
            observer=analyze.observer_contract(raw,report["runs"],True,expected_models=2)
            if observer!=summary["analysis"]["observer"] or not observer["hot_no_compile_observed"] or not observer["hot_no_model_load_observed"]:
                raise ValueError("actual fixed-model hot compile/load contract differs")
        hot=[r["request_wall_including_export"] for r in report["runs"][1:]]
        if summary["analysis"]["hot_median_seconds"]!=(None if matrix else statistics.median(hot)):
            raise ValueError("complete request denominator or heterogeneous no-pooling contract differs")
        return dict(root=root,summary=summary,images=images,receipt=dict(cohort=root.name,policy=policy,text_prefix=text,
            first_seconds=report["runs"][0]["request_wall_including_export"],hot_seconds=hot,hot_median_seconds=None if matrix else statistics.median(hot),
            actual_per_request_counts=counts,prefix_receipts=[m.get("qwen_runtime_prefix_snapshot") for m in metrics],observer=observer,
            memory=summary["memory"]["verified"],load_error=summary.get("load_error"),summary_sha256=sha256_file(root/"summary.json")))
    off=window(args.off,False,"ane");ane=window(args.ane,True,"ane");gpu=window(args.gpu,True,"gpu")
    text_prefix_audit.matched(off,ane);only_policy(ane,gpu)
    if not np.array_equal(off["images"][0],gpu["images"][0]):raise ValueError("GPU policy changed original cold miss")
    if ane["receipt"]["prefix_receipts"][0]["recipe_sha256"]==gpu["receipt"]["prefix_receipts"][0]["recipe_sha256"]:
        raise ValueError("different first-step arithmetic reused the same prefix recipe namespace")
    records=[off,ane,gpu]
    primary_quality=cached_gpu_pair(off,gpu,require_exact=args.require_gpu_exact)
    for root,base,text,policy,mode in ((args.reverse_off,off,False,"ane","stock"),(args.repeat_gpu,gpu,True,"gpu","stock"),(args.timeline,gpu,True,"gpu","timeline")):
        if root:
            value=window(root,text,policy,mode);text_prefix_audit.replay(base,value);records.append(value)
    if args.matrix:
        value=window(args.matrix,True,"gpu","timeline",matrix=True);text_prefix_audit.prompt_matrix(gpu,value);records.append(value)
    edit=window(args.edit,False,"ane",edit=True) if args.edit else None
    if edit:records.append(edit)
    additional=[];additional_sheets=[]
    for label,left_root,right_root in args.additional_pair:
        left=window(Path(left_root),False,"ane");right=window(Path(right_root),True,"gpu")
        checks=cached_gpu_pair(left,right,require_exact=args.require_gpu_exact);records.extend((left,right))
        additional.append(dict(label=label,off=left["receipt"]["cohort"],gpu=right["receipt"]["cohort"],quality=checks,
            hot_wall_reduction=1-right["receipt"]["hot_median_seconds"]/left["receipt"]["hot_median_seconds"]))
        additional_sheets.append((left,right))
    controls={}
    for label,path,value in (("off",args.historical_off,off),("legacy_ane",args.historical_ane,ane),("edit",args.historical_edit,edit)):
        if path:
            if value is None:raise ValueError("frozen control has no actual current cohort")
            controls[label]=frozen(path,value)
    args.output.mkdir(parents=True)
    sheet=Image.new("RGB",(1536,1088),"white");draw=ImageDraw.Draw(sheet)
    for j,(label,value) in enumerate((("prefix off",off),("prefix hit, ANE step0 QKV",ane),("prefix hit, GPU step0 QKV",gpu))):
        for row,i in enumerate((1,2)):
            draw.text((j*512+8,row*544+8),f"{label}, seed {(73,74,73)[i]}",fill="black")
            sheet.paste(Image.fromarray(value["images"][i][:,:,:3]),(j*512,row*544+32))
    sheet.save(args.output/"comparison.png")
    for index,(left,right) in enumerate(additional_sheets):
        sheet=Image.new("RGB",(1024,1632),"white");draw=ImageDraw.Draw(sheet)
        for j,(label,value) in enumerate((("prefix off",left),("prefix hit, GPU step0 QKV",right))):
            for i in range(3):
                draw.text((j*512+8,i*544+8),f"{label}, seed {(73,74,73)[i]}, request {i}",fill="black")
                sheet.paste(Image.fromarray(value["images"][i][:,:,:3]),(j*512,i*544+32))
        sheet.save(args.output/f"additional-pair-{index}.png")
    result=dict(schema="tc-qwen-text-prefix-first-qkv-audit-v1",status="complete",qualification_passed=False,
        native_library_sha256=sha256_file(args.library),runtime_build_id=manifest["runtime_build_id"],native_source_archive_sha256=sha256_file(args.native_source_archive),
        script_sha256=sha256_file(Path(__file__)),cohorts=[r["receipt"] for r in records],frozen_default_exact=controls,
        cold_miss_rgba_exact=True,quality_ane_vs_off=[quality(off["images"][i],ane["images"][i]) for i in range(3)],
        quality_gpu_vs_off=primary_quality,additional_pairs=additional,complete_rgba_exact_gate_requested=args.require_gpu_exact,
        quality_gpu_vs_ane=[quality(ane["images"][i],gpu["images"][i]) for i in range(3)],
        gpu_hot_wall_reduction_vs_off=1-gpu["receipt"]["hot_median_seconds"]/off["receipt"]["hot_median_seconds"],
        scope="Matched full40 original base requests. API/CB host spans do not prove physical INT8/overlap; finite/SSIM/provenance checks are not general image semantic or exclusive-device qualification.")
    write_json(args.output/"audit.json",result)
    print(json.dumps(dict(status="complete",cohorts=len(records),gpu_hot_wall_reduction_vs_off=result["gpu_hot_wall_reduction_vs_off"],gpu_quality=result["quality_gpu_vs_off"]),indent=2))


if __name__=="__main__":main()

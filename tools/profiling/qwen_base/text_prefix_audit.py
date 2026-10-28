#!/usr/bin/env python3
"""CPU-only full40 text-prefix repeat/prompt-change/default-edit regression audit."""
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
from audit import normalized,pixels,quality
from collect import ROOT,sha256_file,model_snapshot,write_json


def matched(off,on):
    a,b=off["summary"],on["summary"]
    if (set(a["environment"])!=set(b["environment"]) or
            {k for k in a["environment"] if a["environment"][k]!=b["environment"][k]}!={"TURBOCIDER_QWEN21_RUNTIME_TEXT_PREFIX_SNAPSHOT"}):
        raise ValueError("only the explicit text-prefix switch may differ")
    if any(a["assets"][k]!=b["assets"][k] for k in ("library","manifest","reference")):
        raise ValueError("same library/descriptor/reference assets required")
    if any(normalized(off["root"]/f"trial/request-{i}.json")!=normalized(on["root"]/f"trial/request-{i}.json") for i in range(3)):
        raise ValueError("actual same-prompt/seed full requests required")


def replay(base,value):
    if len(base["images"])!=len(value["images"]):raise ValueError("complete same-recipe image count required")
    for i,(a,b) in enumerate(zip(base["images"],value["images"])):
        if normalized(base["root"]/f"trial/request-{i}.json")!=normalized(value["root"]/f"trial/request-{i}.json") or not np.array_equal(a,b):
            raise ValueError("same recipe replay did not restore every request and complete PNG")


def prompt_matrix(base,value):
    root=value["root"];requests=[normalized(root/f"trial/request-{i}.json") for i in range(4)]
    if requests[0]!=normalized(base["root"]/"trial/request-0.json") or requests[1]!=normalized(base["root"]/"trial/request-1.json") or requests[3]!=requests[0]:
        raise ValueError("prompt matrix must restore original source requests")
    if set(requests[0])!=set(requests[2]) or {k for k in requests[0] if requests[0][k]!=requests[2][k]}!={"prompt"}:
        raise ValueError("only actual prompt may change at the conditioning miss")
    if (not np.array_equal(value["images"][0],base["images"][0]) or
            not np.array_equal(value["images"][1],base["images"][1]) or
            not np.array_equal(value["images"][3],base["images"][0]) or
            any(np.array_equal(value["images"][2],x) for x in base["images"])):
        raise ValueError("matrix missed actual input/output restoration or returned stale target pixels")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("off","on","reverse-off","timeline","matrix","edit","library","native-source-archive","output"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--historical-gen",type=Path);p.add_argument("--historical-edit",type=Path)
    p.add_argument("--repeat-on",type=Path)
    p.add_argument("--tool-source-archive",type=Path,action="append",default=[]);args=p.parse_args()
    if args.output.exists():p.error("fresh audit directory required")
    manifest=json.loads((args.library.parent/"runtime-build/runtime-build-manifest.json").read_text())
    joint_audit.archive_contract(args.native_source_archive,manifest)
    historical={}
    for path in args.tool_source_archive:
        with tarfile.open(path,"r:gz") as archive:
            for entry in archive.getmembers():
                if entry.isfile():historical.setdefault(entry.name,set()).add(archive.extractfile(entry).read())
    def window(root,text,mode,*,edit=False,matrix=False):
        summary=json.loads((root/"summary.json").read_text());report=json.loads((root/"trial/report.json").read_text())
        count=4 if matrix else 3;scenario="text-prefix-matrix" if matrix else "repeat";case="edit" if edit else "generate"
        expected_environment=collect.environment("runtime",mode,6144,edit,False,True,True,"descriptor","sigmoid-complement",runtime_text_prefix=text)
        if (summary["status"]!="complete" or report["status"]!="complete" or summary["case"]!=case or summary["mode"]!=mode or
                summary["route"]!="runtime" or summary["channels"]!=6144 or summary["scenario"]!=scenario or len(report["runs"])!=count or
                summary.get("runtime_text_prefix_snapshot") is not text or report.get("runtime_text_prefix_snapshot") is not text or
                summary.get("runtime_prefix_snapshot") is not edit or summary.get("resident_prefix") is not False or
                summary.get("private_qkv") is not True or summary.get("fixed_program") is not True or
                summary.get("ffn_activation")!="sigmoid-complement" or summary.get("ffn_projection")!="descriptor" or
                summary["environment"]!=expected_environment or summary["assets"]["library"]!=sha256_file(args.library) or
                report["loaded_runtime_id"]!=manifest["runtime_build_id"] or
                summary["model_snapshot"]!=model_snapshot(ROOT/"models/Comfy-Org-Qwen-Image-2.1")):
            raise ValueError("complete explicit matched text-prefix/current-base recipe required")
        source=json.loads((root/"source.json").read_text())
        if set(source)!=set(summary["source_sha256"]):raise ValueError("complete profiling source snapshot required")
        for name,digest in summary["source_sha256"].items():
            raw=source[name].encode()
            if hashlib.sha256(raw).hexdigest()!=digest or not (raw==(ROOT/name).read_bytes() or raw in historical.get(name,set())):
                raise ValueError("bound current/archive profiling source differs")
        metrics=[r["metrics"] for r in report["runs"]]
        collect.validate(metrics,"runtime",mode,6144,edit,False,True,True,"descriptor","sigmoid-complement",runtime_text_prefix=text,text_prefix_matrix=matrix)
        previous=0
        for i,row in enumerate(report["runs"]):
            request=json.loads((root/f"trial/request-{i}.json").read_text())
            if (row["index"]!=i or request.get("seed")!=([73,74,73,73] if matrix else [73,74,73])[i] or
                    request.get("operation")!="image."+case or request.get("loras") or request.get("audio") is not False or
                    (edit and (len(request.get("inputs",[]))!=1 or request.get("qwen21_reference_size")!=1024)) or
                    (not edit and request.get("inputs"))):raise ValueError("actual full source request/seed geometry changed")
            if sha256_file(Path(request["ane_manifest"]))!=summary["assets"]["manifest"]:
                raise ValueError("actual request descriptor differs from bound asset")
            calls=row["metrics"]["hybrid"]["runtime_calls_session_total"]
            if not text and not edit and calls-previous!=1312:raise ValueError("text-off generation unexpectedly changed actual FFN chunk coverage")
            previous=calls
        images=[pixels(root/f"trial/{i}.png") for i in range(count)]
        if any(x.shape!=(512,512,4) for x in images) or np.array_equal(images[0],images[1]):raise ValueError("complete changed-seed RGBA PNGs required")
        if not text and not edit and not np.array_equal(images[0],images[2]):raise ValueError("uncached generation did not restore original seed")
        for key in ("report","evidence"):
            if sha256_file(root/summary["memory"][key])!=summary["memory"][key+"_sha256"]:raise ValueError("memory evidence changed")
        if not summary["memory"]["verified"]["complete"] or summary["memory"]["verified"]["command_exit_code"]!=0:
            raise ValueError("complete successful owned model window required")
        observer=None
        if mode=="timeline":
            raw=json.loads((root/"trial/observer.json").read_text())
            if raw["source_sha256"]!=summary["source_sha256"]["tools/profiling/qwen_base/observer.mm"]:raise ValueError("observer binary source differs")
            observer=analyze.observer_contract(raw,report["runs"],True,expected_models=2)
            if observer!=summary["analysis"]["observer"] or not observer["hot_no_compile_observed"] or not observer["hot_no_model_load_observed"]:
                raise ValueError("actual stable two-model hot compile/load contract differs")
        hot=[r["request_wall_including_export"] for r in report["runs"][1:]]
        if matrix:
            if summary["analysis"]["hot_median_seconds"] is not None:raise ValueError("heterogeneous prompts cannot have pooled hot median")
        elif statistics.median(hot)!=summary["analysis"]["hot_median_seconds"]:raise ValueError("full request timing denominator differs")
        return dict(root=root,summary=summary,report=report,images=images,receipt=dict(cohort=root.name,case=case,scenario=scenario,
            first_seconds=report["runs"][0]["request_wall_including_export"],hot_seconds=hot,
            hot_median_seconds=None if matrix else statistics.median(hot),observer=observer,
            prefix_receipts=[m.get("qwen_runtime_prefix_snapshot") for m in metrics],native_deltas=analyze.native_deltas(report["runs"]),
            memory=summary["memory"]["verified"],load_error=summary.get("load_error"),summary_sha256=sha256_file(root/"summary.json")))
    off=window(args.off,False,"stock");on=window(args.on,True,"stock");matched(off,on)
    if not np.array_equal(off["images"][0],on["images"][0]):raise ValueError("cold miss changed original arithmetic")
    reverse=window(args.reverse_off,False,"stock");replay(off,reverse)
    timeline=window(args.timeline,True,"timeline");replay(on,timeline)
    matrix=window(args.matrix,True,"timeline",matrix=True);prompt_matrix(on,matrix)
    edit=window(args.edit,False,"stock",edit=True)
    historical_exact={}
    for label,path,value in (("generation",args.historical_gen,off),("edit",args.historical_edit,edit)):
        if path is None:continue
        checks=[]
        for i in range(3):
            if normalized(path/f"request-{i}.json")!=normalized(value["root"]/f"trial/request-{i}.json"):raise ValueError("frozen default actual request differs")
            checks.append(bool(np.array_equal(pixels(path/f"{i}.png"),value["images"][i])))
        if not all(checks):raise ValueError("disabled text feature changed frozen generation/edit outputs")
        historical_exact[label]=checks
    records=[off,on,reverse,timeline,edit,matrix]
    repeated=None
    if args.repeat_on:
        repeated=window(args.repeat_on,True,"stock");replay(on,repeated);records.append(repeated)
    args.output.mkdir(parents=True)
    sheet=Image.new("RGB",(1024,3*544),"white");draw=ImageDraw.Draw(sheet)
    for i in range(3):
        for j,(label,value) in enumerate((("text prefix off",off),("text prefix on",on))):
            draw.text((j*512+8,i*544+8),f"{label}, seed {(73,74,73)[i]}, request {i}",fill="black")
            sheet.paste(Image.fromarray(value["images"][i][:,:,:3]),(j*512,i*544+32))
    sheet.save(args.output/"comparison.png")
    result=dict(schema="tc-qwen-runtime-text-prefix-audit-v1",status="complete",qualification_passed=False,
        native_library_sha256=sha256_file(args.library),runtime_build_id=manifest["runtime_build_id"],
        native_source_archive_sha256=sha256_file(args.native_source_archive),script_sha256=sha256_file(Path(__file__)),
        cohorts=[v["receipt"] for v in records],cold_miss_rgba_exact=True,historical_default_rgba_exact=historical_exact,
        incremental_quality=[quality(off["images"][i],on["images"][i]) for i in range(3)],
        cached_seed73_vs_first_rgba_exact=bool(np.array_equal(on["images"][0],on["images"][2])),
        stock_hot_wall_reduction=1-on["receipt"]["hot_median_seconds"]/off["receipt"]["hot_median_seconds"],
        repeated_on_hot_seconds=repeated["receipt"]["hot_seconds"] if repeated else None,
        prompt_change_and_restore_exact=True,same_recipe_timeline_and_reverse_exact=True,
        limitations=["Non-exclusive small pilot, no statistical/thermal/general semantic qualification.",
            "First-step target-only hit changes GPU/ANE QKV coverage; cache algebra alone does not prove original pixels exact.",
            "Two fixed models with actual observed client hot compile/load zero; opaque driver internals and physical INT8/overlap unproven."])
    write_json(args.output/"audit.json",result)
    print(json.dumps(dict(status=result["status"],stock_hot_wall_reduction=result["stock_hot_wall_reduction"],historical_default_rgba_exact=historical_exact),indent=2))


if __name__=="__main__":main()

#!/usr/bin/env python3
"""Read explicit held-out block partial runs; never make failed delivery a win.

Stdlib/source/identity and recorded-data verification only. No candidate imports,
model execution, private outputs replay, cache deletion or raw prompt publication.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"labs"));sys.path.insert(0,str(Path(__file__).resolve().parent))
from block_context import verify
from curate_real_shared_ffn import tensor


def sha(path):
    result=hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda:stream.read(1<<20),b""):result.update(data)
    return result.hexdigest()


def local(root,name):
    relative=Path(name);path=(root/relative).resolve()
    if relative.is_absolute() or ".." in relative.parts or not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("explicit bounded current asset required")
    return path


def text_sha(value):return hashlib.sha256(value.encode()).hexdigest()


def curate(args):
    root=args.root.resolve();native_dir=args.native_dir.resolve();mlx=args.mlx_root.resolve()
    baseline_path=local(args.baseline_capture,"receipt.json");baseline=json.loads(baseline_path.read_text())
    baseline_request=json.loads(local(args.baseline_capture,"request.json").read_text())
    build_path=local(native_dir,"runtime-build/runtime-build-manifest.json");build=json.loads(build_path.read_text())
    repo=ROOT.parent
    if build["runtime_build_id"]!=baseline["runtime_build_id"]:raise ValueError("fixed source-qualified runtime required")
    for item in build["inputs"]["sources"]:
        if sha(local(repo,item["path"]))!=item["sha256"]:raise ValueError("runtime source differs")
    library=local(native_dir,"libturbocider.dylib")
    if sha(library)!=baseline["library_sha256"]:raise ValueError("runtime library differs")
    bank_path=local(args.bank_run,"report.json");bank=json.loads(bank_path.read_text());manifest_path=local(args.bank_run,"manifest.json");manifest=json.loads(manifest_path.read_text())
    if bank.get("status")!="measured" or sha(manifest_path)!=bank["raw_manifest_sha256"]:raise ValueError("frozen qualified prepared bank required")
    model_dir=Path(manifest["model_directory"]).resolve()
    if not model_dir.is_relative_to(root) or sha(local(model_dir,"model.mil"))!=bank["private_mil_sha256"] or sha(local(model_dir,"weights.bin"))!=bank["private_blob_sha256"]:
        raise ValueError("read-only private bank source changed")
    checkpoint=local(args.model_root,"diffusion_models/qwen_image_2.1_bf16.safetensors")
    for key,node,shape in (("gate_up.weight",manifest["weights"][0],[24576,4096]),("out.weight",manifest["weights"][1],[4096,12288])):
        verified,_=tensor(checkpoint,"transformer_blocks.0.img_mlp."+key,shape,node["sha256"])
        if verified!=node:raise ValueError("original bank source/shape/span differs")
    result={"schema":"tc-qwen-block-context-evidence-v1","curator_sha256":sha(Path(__file__)),"contract_sha256":sha(ROOT/"labs/block_context.py"),
        "production_qualified":False,"qualified_performance_ratios_available":False,"raw_tensors_available":False,
        "raw_addresses_paths_prompts_exported":False,"data_verifier_reexecuted_original_inputs_or_private_outputs":False,
        "source_identities":{"baseline_capture_receipt_sha256":sha(baseline_path),"baseline_seed":baseline["seed"],"baseline_prompt_sha256":text_sha(baseline_request["prompt"]),
            "runtime_build_id":build["runtime_build_id"],"native_build_manifest_sha256":sha(build_path),"native_source_files_verified":len(build["inputs"]["sources"]),
            "native_library_sha256":sha(library),"bank_report_sha256":sha(bank_path),"bank_manifest_sha256":sha(manifest_path),"private_mil_sha256":bank["private_mil_sha256"],
            "private_blob_sha256":bank["private_blob_sha256"],"original_ffn_weight_sha256":[value["sha256"] for value in manifest["weights"]]},
        "capture_admission_rejections":[],"block_runs":[]}
    selected=args.capture_rejected+args.block_run
    if not args.block_run or len(selected)!=len(set(selected)):raise ValueError("explicit unique selected outcomes required")
    assets_checked=set()
    for name in selected:
        launch=local(root,name+"-launch/receipt.json");terminal=json.loads(launch.read_text())
        if terminal.get("owned_process_exit_confirmed") is not True or terminal.get("returncode")!=1:raise ValueError("explicit terminal wrapper failure receipt required")
        capture=name in args.capture_rejected;path=local(root,name+("/receipt.json" if capture else "/report.json"));document=json.loads(path.read_text())
        item={"id":name,"raw_report_sha256":sha(path),"supervisor_sha256":sha(launch),"owned_wrapper_exit_confirmed":True}
        if capture:
            if (document.get("schema")!="tc-qwen-heldout-capture-v1" or document.get("status")!="failed" or document.get("records")!=[]
                    or len(document.get("errors",[]))!=1 or "267 MiB" not in document["errors"][0]["message"]
                    or document.get("owned_child_exit_confirmed") is not False or (path.parent/"captures").exists()):
                raise ValueError("full capture was refused before native child; do not invent hardware work")
            public=copy.deepcopy(document);public.pop("checkpoint_stat_before",None)
            item.update(report=public,independent_validation={"native_inference_started":False,"hardware_unsupported":False,"kind":"capture-storage-admission"})
            result["capture_admission_rejections"].append(item);continue
        proof=verify(document);config_path=local(path.parent,"config.json");config=json.loads(config_path.read_text())
        if sha(config_path)!=document["raw_config_sha256"] or document["native"]["source_config_sha256"]!=sha(config_path):raise ValueError("actual native config binding differs")
        if (config["seed"]!=document["seed"] or text_sha(config["prompt"])!=document["prompt_sha256"] or Path(config["model_root"]).resolve()!=args.model_root.resolve()
                or Path(config["model_directory"]).resolve()!=model_dir or config["expected_gate_up_sha256"]!=manifest["weights"][0]["sha256"]
                or config["expected_down_sha256"]!=manifest["weights"][1]["sha256"]):raise ValueError("actual request/model/private weight bindings differ")
        for filename,digest in document["source_sha256"].items():
            if sha(local(path.parent,"source/"+filename))!=digest:raise ValueError("per-run source differs")
        if sha(local(path.parent,"qwen-block-shared"))!=document["binary_sha256"] or sha(local(path.parent,"native.json"))!=document["raw_native_sha256"]:
            raise ValueError("native executable/report content differs")
        if json.loads(local(path.parent,"native.json").read_text())!=document["native"]:raise ValueError("wrapper/native capsule differs")
        cpp=local(path.parent,"source/qwen_block_shared.mm").read_text();start=cpp.index("for(uint32_t group=0;group<repeats+3")
        end=cpp.index('report[@"final_identity"]',start)
        item["timed_loop_and_validation_source_sha256"]=text_sha(cpp[start:end])
        assets=document["mlx_assets"];identity=text_sha(json.dumps(assets,sort_keys=True))
        if identity not in assets_checked:
            inventory=[str(p.relative_to(mlx)) for p in sorted((mlx/"include").rglob("*")) if p.is_file()]
            if inventory!=[value["path"] for value in assets["headers"]]:raise ValueError("complete MLX header inventory differs")
            for value in assets["headers"]:
                if sha(local(mlx,value["path"]))!=value["sha256"]:raise ValueError("installed MLX header differs")
            for filename,digest in assets["libraries"].items():
                if sha(local(mlx,"lib/"+filename))!=digest:raise ValueError("installed MLX library differs")
            assets_checked.add(identity)
        for filename in ("ffn-input.safetensors","ffn-reference.safetensors","ffn-bf16.tcbf16z"):
            if (path.parent/filename).exists():raise ValueError("failed export cannot advertise persisted full tensors")
        item.update(report=document,independent_validation=proof,original_raw_model_source_verified=True)
        result["block_runs"].append(item)
    encoded=json.dumps(result,ensure_ascii=False,allow_nan=False)
    if any(value in encoded for value in ("/Users/","/private/var/folders/","/home/")):raise ValueError("public capsule contains machine-local paths")
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("root","native-dir","model-root","bank-run","baseline-capture","mlx-root"):parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--capture-rejected",action="append",default=[]);parser.add_argument("--block-run",action="append",required=True)
    parser.add_argument("--output",type=Path,required=True);args=parser.parse_args();value=curate(args)
    with args.output.open("x",encoding="utf-8") as stream:stream.write(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"block_records":len(value["block_runs"]),"completed_block_calls_before_export_refusal":sum(item["independent_validation"]["candidate_timed_blocks"] for item in value["block_runs"]),
        "capture_admission_rejections":len(value["capture_admission_rejections"]),"qualified_ratio_available":False}))


if __name__=="__main__":main()

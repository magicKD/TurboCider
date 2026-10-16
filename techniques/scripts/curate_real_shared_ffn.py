#!/usr/bin/env python3
"""Read explicit real shared-FFN runs and original-source assets, never devices.

Standard library only. Recheck original BF16/fixture/prefix/program identities,
complete recorded gates/order/host dependencies, join hashes and ratio arithmetic.
Does not reexecute private outputs or independently attest reported hardware work.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "labs"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from real_shared_ffn import verify, port
from curate_qwen_private_ffn import native_prefix_hashes


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1 << 20), b""): value.update(part)
    return value.hexdigest()


def local(root, relative):
    name = Path(relative); path = (root / name).resolve()
    if name.is_absolute() or ".." in name.parts or not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("bounded explicit existing file required")
    return path


def span(path, offset, size):
    if type(offset) is not int or type(size) is not int or offset < 0 or size <= 0 or offset + size > path.stat().st_size:
        raise ValueError("invalid complete source span")
    value = hashlib.sha256()
    with path.open("rb") as stream:
        stream.seek(offset)
        while size:
            data = stream.read(min(size, 1 << 20))
            if not data: raise ValueError("short source")
            value.update(data); size -= len(data)
    return value.hexdigest()


def tensor(path, key, shape, expected):
    with path.open("rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        if not 2 <= size <= 16 * 1024**2: raise ValueError("safetensors header cap")
        data = json.loads(stream.read(size)); node = data[key]
    start, end = node["data_offsets"]
    elements = 1
    for dim in shape: elements *= dim
    if (node["dtype"] != "BF16" or node["shape"] != shape or type(start) is not int or type(end) is not int
            or start < 0 or end - start != elements * 2 or span(path, 8 + size + start, end - start) != expected):
        raise ValueError("original BF16 tensor/shape/content differs")
    return {"file":str(path.resolve()),"offset":8+size+start,"bytes":end-start,"sha256":expected,"shape":shape,"dtype":"BF16"}, data.get("__metadata__",{})


def reference(path):
    with path.open("rb") as stream:
        if stream.read(8) != b"\x93NUMPY\x01\x00": raise ValueError("original reference NPYv1 required")
        size = struct.unpack("<H", stream.read(2))[0]
        if not 1 <= size <= 65536: raise ValueError("reference header cap")
        header = ast.literal_eval(stream.read(size).decode("ascii")); offset = stream.tell()
    if header != {"descr":"<f4","fortran_order":False,"shape":(1024,4096)} or path.stat().st_size != offset + 16777216:
        raise ValueError("whole original BF16-replica reference required")
    return {"file":str(path.resolve()),"offset":offset,"bytes":16777216,"sha256":span(path,offset,16777216),"shape":[1024,4096],"dtype":"F32"}


def inspect(document):
    native = document["native"]
    if (document.get("status") != "ports-inspected-no-execution" or native.get("status") != "ports-inspected-no-execution"
            or document.get("mode") != "inspect" or native.get("samples") != [] or native.get("formal_performance_score") is not False
            or native.get("unload_succeeded") is not True or document.get("source_inputs_unchanged") is not True
            or document.get("cloned_graph_unchanged") is not True):
        raise ValueError("inspection is not a numerical/performance run")
    port(native["input_descriptor"]); port(native["output_descriptor"])
    return {"executed_ffn_calls":0,"actual_ports_read":True,"performance_qualified":False}


def curate(args):
    root, fixture_path = args.root.resolve(), args.fixture.resolve(); fixture_root = fixture_path.parent
    fixture = json.loads(fixture_path.read_text())
    if (fixture.get("shape_mhf") != [1024,4096,12288] or fixture.get("source_layer") != 0 or fixture.get("loras") != []):
        raise ValueError("complete base layer0 fixture required")
    for name, digest in fixture["files"].items():
        if sha(local(fixture_root,name)) != digest: raise ValueError("fixture asset differs")
    checkpoint, capture = args.checkpoint.resolve(), args.capture_root.resolve()
    weights = [tensor(checkpoint,"transformer_blocks.0.img_mlp.gate_up.weight",[24576,4096],fixture["original_gate_up_tensor_sha256"])[0],
               tensor(checkpoint,"transformer_blocks.0.img_mlp.out.weight",[4096,12288],fixture["original_down_tensor_sha256"])[0]]
    receipt_path = local(capture,"receipt.json")
    if sha(receipt_path) != fixture["capture_receipt_sha256"]: raise ValueError("original capture receipt differs")
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("status") != "captured" or receipt.get("loras") != []: raise ValueError("complete original base capture required")
    points = {}
    for info in fixture["points"]:
        rows = [row for row in receipt["records"] if Path(row["path"]).name == f"layer-0-step-{info['step']}.safetensors"]
        if len(rows) != 1: raise ValueError("unique original captured point required")
        path = local(capture,rows[0]["path"])
        if sha(path) != info["capture_file_sha256"]: raise ValueError("actual activation file differs")
        value, metadata = tensor(path,"input",[1,1024,4096],info["original_input_tensor_sha256"])
        if metadata.get("trajectory") != "forced-gpu" or metadata.get("capture_recipe") != "qwen21-real-ffn-bf16-v1":
            raise ValueError("original capture recipe differs")
        points[info["id"]] = {"id":info["id"],"input":value,"reference":reference(local(fixture_root,info["original_reference"]))}
    prior_root = args.prior_run.resolve(); prior_path = local(prior_root,"report.json"); prior = json.loads(prior_path.read_text())
    prefix_hashes = native_prefix_hashes(fixture_root,fixture,2048)
    if (prior.get("status") != "measured" or prior.get("share") != 2048 or prior.get("native_weight_sha256") != prefix_hashes
            or prior["original_weight_sha256"] != {key:fixture[key] for key in ("original_gate_up_tensor_sha256","original_down_tensor_sha256")}):
        raise ValueError("frozen native prefix must match original paired source")
    result = {"schema":"tc-real-qwen-shared-evidence-v1","curator_sha256":sha(Path(__file__)),"contract_sha256":sha(ROOT/"labs/real_shared_ffn.py"),
        "production_qualified":False,"physical_overlap_qualified":False,"runtime_internal_copy_behavior":"unknown",
        "raw_tensor_values_addresses_paths_exported":False,"data_verifier_reexecuted_private_outputs":False,
        "verified_sources":{"fixture_sha256":sha(fixture_path),"capture_receipt_sha256":sha(receipt_path),
            "original_weight_tensor_sha256":[item["sha256"] for item in weights],"native_prefix_sha256":prefix_hashes,
            "prior_report_sha256":sha(prior_path),"original_point_input_sha256":{key:value["input"]["sha256"] for key,value in points.items()}},
        "runs":[],"inspections":[],"admission_rejections":[]}
    selected = args.run + args.inspection + args.rejected
    if not args.run or len(selected) != len(set(selected)): raise ValueError("explicit unique selected runs required")
    checked_assets = set()
    for name in selected:
        path = local(root,name+"/report.json"); document = json.loads(path.read_text())
        launch = local(root,name+"-launch/receipt.json"); terminal = json.loads(launch.read_text())
        rejected = name in args.rejected
        if terminal.get("owned_process_exit_confirmed") is not True or terminal.get("returncode") != (1 if rejected else 0):
            raise ValueError("owned terminal child receipt required")
        row = {"id":name,"raw_report_sha256":sha(path),"supervisor_sha256":sha(launch),"child_exit_confirmed":True,"report":document}
        if rejected:
            if (document.get("status") != "failed" or document.get("native") is not None or document.get("comparisons") is not None
                    or len(document.get("errors",[])) != 1 or "256 MiB" not in document["errors"][0]["message"]
                    or (path.parent/"qwen-shared").exists() or (path.parent/"native.json").exists()):
                raise ValueError("admission refusal must not be relabelled as hardware execution/failure")
            row["independent_validation"]={"executed_ffn_calls":0,"hardware_unsupported":False,"kind":"storage-admission-no-execution"}
            result["admission_rejections"].append(row); continue
        row["independent_validation"] = inspect(document) if name in args.inspection else verify(document)
        manifest_path = local(path.parent,"manifest.json"); manifest = json.loads(manifest_path.read_text())
        if sha(manifest_path) != document["raw_manifest_sha256"] or document["native"]["manifest_sha256"] != sha(manifest_path):
            raise ValueError("actual native manifest binding differs")
        if manifest["weights"] != weights: raise ValueError("native uploaded source spans differ")
        for point in manifest["points"]:
            expected = points[point["id"]]
            if any(point[key] != expected[key] for key in ("input","reference")): raise ValueError("native input/reference must be whole actual source point")
        if name not in args.inspection and [point["id"] for point in manifest["points"]] != document["native"]["point_cycle"]:
            raise ValueError("timed point cycle differs from actual original input bindings")
        if name not in args.inspection:
            if (manifest.get("schema") != "tc-qwen-shared-manifest-v1" or manifest.get("mode") != "benchmark"
                    or manifest.get("shape_mhf") != [1024,4096,12288] or manifest.get("share") != 2048
                    or manifest.get("gates") != {"relative_l2_max":.03,"cosine_min":.999}
                    or manifest.get("measured_groups") != document["native"]["measured_groups"]):
                raise ValueError("actual native workload/timing/quality manifest differs")
            expected_sources={"qwen_shared_ffn.py","qwen_shared_ffn.mm","qwen_shared_abi.h","qwen_shared_compute.h",
                              "curate_qwen_private_ffn.py","qwen_witness.py","real_bf16_source.py","mlx_adoption_probe.py"}
            if set(document["source_sha256"]) != expected_sources:
                raise ValueError("complete runner/native/oracle/asset-helper source bindings required")
            for point in manifest["points"]:
                for kind in ("full","prefix"):
                    value=point["oracle"][kind]
                    if (value["rows"] != [0,341,682,1023] or value["cols"] != [0,1365,2730,4095]
                            or len(value["values"]) != 4 or any(len(row)!=4 for row in value["values"])):
                        raise ValueError("independent full-K/F oracle sample geometry differs")
        model = Path(manifest["model_directory"]).resolve()
        if not model.is_relative_to(root): raise ValueError("owned graph source escaped explicit campaign root")
        if (sha(local(model,"model.mil")) != document["private_mil_sha256"] or document["private_mil_sha256"] != prior["private_mil_sha256"]
                or sha(local(model,"weights.bin")) != document["private_blob_sha256"] or document["private_blob_sha256"] != prior["private_weights_blob_sha256"]
                or document["native_weight_sha256"] != prefix_hashes):
            raise ValueError("actual cloned/reused MIL/blob differs from frozen original prefix")
        if manifest.get("skip_compile"):
            prepared = json.loads(local(model.parent,"report.json").read_text()); old_manifest = local(model.parent,"manifest.json")
            if (sha(local(model.parent,"report.json")) != document["prepared_source_report_sha256"]
                    or sha(old_manifest) != document["prepared_source_manifest_sha256"] or prepared.get("status") != "measured"
                    or document["native"].get("compile_selector_called") is not False): raise ValueError("read-only load replay lineage differs")
        for filename,digest in document["source_sha256"].items():
            if sha(local(path.parent,"source/"+filename)) != digest: raise ValueError("per-run source snapshot differs")
        if (sha(local(path.parent,"qwen-shared")) != document["binary_sha256"]
                or sha(local(path.parent,"native.json")) != document["raw_native_sha256"]
                or json.loads(local(path.parent,"native.json").read_text()) != document["native"]): raise ValueError("binary/native content differs")
        if "mlx_assets" in document:
            assets = document["mlx_assets"]; token = hashlib.sha256(json.dumps(assets,sort_keys=True).encode()).hexdigest()
            if token not in checked_assets:
                inventory = [str(item.relative_to(args.mlx_root.resolve())) for item in sorted((args.mlx_root.resolve()/"include").rglob("*")) if item.is_file()]
                if inventory != [item["path"] for item in assets["headers"]]: raise ValueError("full installed-header inventory differs")
                for item in assets["headers"]:
                    if sha(local(args.mlx_root,item["path"])) != item["sha256"]: raise ValueError("installed header changed")
                for filename,digest in assets["libraries"].items():
                    if sha(local(args.mlx_root,"lib/"+filename)) != digest: raise ValueError("installed library asset changed")
                checked_assets.add(token)
        # The private FP64 vectors stay local; bind them without publishing source-derived values.
        row["private_oracle_manifest_sha256"] = hashlib.sha256(json.dumps([point.get("oracle") for point in manifest["points"]],sort_keys=True).encode()).hexdigest()
        row["original_spans_independently_verified"] = True
        result["inspections" if name in args.inspection else "runs"].append(row)
    encoded=json.dumps(result,ensure_ascii=False,allow_nan=False)
    if "/Users/" in encoded or "/private/var/folders/" in encoded or "/home/" in encoded:
        raise ValueError("capsule contains a machine-local path; no automatic permissive redaction")
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("root","fixture","checkpoint","capture-root","prior-run","mlx-root"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--run",action="append",required=True)
    parser.add_argument("--inspection",action="append",default=[])
    parser.add_argument("--rejected",action="append",default=[])
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(); value=curate(args)
    with args.output.open("x",encoding="utf-8") as stream: stream.write(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"runs":len(value["runs"]),"complete_consumers":sum(row["independent_validation"]["complete_consumers"] for row in value["runs"]),
        "exact_runtime_join_consumers":sum(row["independent_validation"]["exact_runtime_join_consumers"] for row in value["runs"]),
        "inspections":len(value["inspections"]),"admission_rejections":len(value["admission_rejections"])}))


if __name__=="__main__": main()

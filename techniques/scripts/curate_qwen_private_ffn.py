#!/usr/bin/env python3
"""Read explicit real-Qwen private-FFN reports; verify source, all-call gates and ratios.

No candidate/upstream/NumPy/MLX imports or device execution. Content hashes and
reported oracle/gates are audited, not hardware-attested or numerically rerun.
"""
import argparse
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import struct
import subprocess

COMMIT = "896c4943c5ebf811c0e815f8d1381fd1e157fa7d"
BOUNDARY = "prepared original BF16 GPU input to BF16 GPU output with explicit MLX drain; conversion/feed/restore/join included"


def sha(path):
    return span_sha(path, 0, path.stat().st_size)


def span_sha(path, offset, size, digest=None):
    value = hashlib.sha256() if digest is None else digest
    if type(offset) is not int or type(size) is not int or offset < 0 or size < 0 or offset + size > path.stat().st_size:
        raise ValueError("source byte range invalid")
    with path.open("rb") as stream:
        stream.seek(offset)
        while size:
            data = stream.read(min(size, 1 << 20))
            if not data: raise ValueError("short source")
            value.update(data); size -= len(data)
    return value.hexdigest()


def local(root, relative):
    item = Path(relative); path = (root / item).resolve()
    if item.is_absolute() or ".." in item.parts or not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("explicit existing non-escaping asset required")
    return path


def tensor_identity(path, key, shape, expected):
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8: raise ValueError("short safetensors")
        length = struct.unpack("<Q", prefix)[0]
        if not 2 <= length <= 16 * 1024**2: raise ValueError("unbounded safetensors header")
        data = json.loads(stream.read(length)); item = data[key]
    begin, end = item["data_offsets"]
    if (item["dtype"] != "BF16" or item["shape"] != list(shape) or type(begin) is not int or type(end) is not int
            or begin < 0 or end - begin != math.prod(shape) * 2):
        raise ValueError("original source shape/dtype/range differs")
    if span_sha(path, 8 + length + begin, end - begin) != expected:
        raise ValueError("original tensor content differs")
    return data.get("__metadata__", {})


def npy_span(path, shape):
    with path.open("rb") as stream:
        if stream.read(6) != b"\x93NUMPY": raise ValueError("not NPY")
        version = tuple(stream.read(2))
        if version not in ((1,0),(2,0)): raise ValueError("unknown NPY version")
        length = struct.unpack("<H" if version == (1,0) else "<I", stream.read(2 if version == (1,0) else 4))[0]
        if not 1 <= length <= 65536: raise ValueError("NPY header cap")
        header = ast.literal_eval(stream.read(length).decode("ascii")); offset = stream.tell()
    if header["descr"] != "<f2" or header["fortran_order"] is not False or header["shape"] != shape:
        raise ValueError("row-major FP16 fixture required")
    if offset + math.prod(shape) * 2 != path.stat().st_size: raise ValueError("NPY payload bounds differ")
    return offset


def native_prefix_hashes(root, fixture, share):
    gu = local(root, fixture["gate_up"]); down = local(root, fixture["down"])
    gu_offset, down_offset = npy_span(gu, (24576,4096)), npy_span(down, (4096,12288))
    value = hashlib.sha256()
    span_sha(gu, gu_offset, share * 4096 * 2, value)
    span_sha(gu, gu_offset + 12288 * 4096 * 2, share * 4096 * 2, value)
    other = hashlib.sha256()
    with down.open("rb") as stream:
        for row in range(4096):
            stream.seek(down_offset + row * 12288 * 2); data = stream.read(share * 2)
            if len(data) != share * 2: raise ValueError("short down prefix")
            other.update(data)
    return [value.hexdigest(), other.hexdigest()]


def metric_pass(value):
    return (value.get("finite") is True and type(value.get("relative_l2")) in (int,float)
            and math.isfinite(value["relative_l2"]) and 0 <= value["relative_l2"] <= .03
            and type(value.get("cosine")) in (int,float) and math.isfinite(value["cosine"]) and .999 <= value["cosine"] <= 1 + 1e-10
            and type(value.get("max_abs")) in (int,float) and math.isfinite(value["max_abs"]) and value["max_abs"] >= 0)


def verify_report(document, group):
    if (document.get("schema") != "tc-aneforge-qwen-ffn-v1" or document.get("status") != "measured"
            or document.get("shape_mhf") != [1024,4096,12288] or type(document.get("source_layer")) is not int or document.get("source_layer") != 0 or document.get("loras") != []
            or document.get("zero_copy") is not False or document.get("production_qualified") is not False
            or document.get("boundary") != BOUNDARY or document.get("gates") != {"relative_l2_max":.03,"cosine_min":.999}
            or document.get("source_inputs_unchanged") is not True or document.get("checkpoint_stat_unchanged") is not True):
        raise ValueError("complete source-stable original workload and fixed consumer/quality contract required")
    share = document["share"]; repeats = document.get("measured_groups",6)
    if type(share) is not int or share not in (512,1024,2048) or type(repeats) is not int or repeats not in (6,24):
        raise ValueError("bounded declared search space required")
    if (group not in ("discovery","confirmation") or group == "discovery" and document["point"] != "layer-0-step-1"
            or group == "confirmation" and (document["point"] not in ("layer-0-step-10","layer-0-step-19") or share != 2048)):
        raise ValueError("frozen 2048-channel held-out confirmation required")
    expected = {"gate":[0,share],"up":[12288,12288+share],"down_columns":[0,share],
                "gpu_gate":[share,12288],"gpu_up":[12288+share,24576],"gpu_down_columns":[share,12288]}
    if document["channel_slices"] != expected:
        raise ValueError("gate/up/down must be paired and the complete GPU remainder retained")
    if (document["logical_private_weight_bytes"] != 6 * 4096 * share or document["logical_per_call_gpu_host_input_bytes"] != 8388608
            or document["logical_per_call_host_gpu_partial_bytes"] != 16777216 or document["logical_gpu_weight_bytes"] != 603979776):
        raise ValueError("complete representation/copy/residency costs differ")
    if "mlx_library_assets" in document and (not document["mlx_library_assets"] or document.get("mlx_library_assets_unchanged") is not True):
        raise ValueError("binary identity drift")
    if "native_requested_device_mask" in document and (type(document["native_requested_device_mask"]) is not int or document["native_requested_device_mask"] != 4):
        raise ValueError("private policy differs")
    arms = ("gpu","serial","hybrid")
    orders = (arms,arms[::-1],(arms[1],arms[2],arms[0]),(arms[0],arms[2],arms[1]),(arms[2],arms[0],arms[1]),(arms[1],arms[0],arms[2]))
    order = [(phase,trial,arm) for phase,count in (("first",1),("warmup",2),("measured",repeats)) for trial in range(count) for arm in orders[trial%6]]
    samples = document["samples"]
    if [(r["phase"],r["trial"],r["arm"]) for r in samples] != order:
        raise ValueError("all ordered first/warmup/measured calls required")
    overlaps = []
    for sample in samples:
        if (type(sample["trial"]) is not int or type(sample["wall_ns"]) is not int or sample["wall_ns"] <= 0
                or sample["point"] != document["point"] or sample["output_dtype"] != "bfloat16"
                or sample.get("numeric_pass") is not True or sample.get("physical_overlap_qualified") is not False):
            raise ValueError("typed full-consumer timing, provenance and numeric markers required")
        keys = {"full_original","independent_full_sample"} | (set() if sample["arm"] == "gpu" else {"prefix_original","independent_prefix_sample"})
        if set(sample["metrics"]) != keys or not all(metric_pass(v) for v in sample["metrics"].values()):
            raise ValueError("every call must pass full-source and independent full/prefix gates")
        if sample["arm"] == "gpu" and (sample["metrics"]["full_original"]["relative_l2"] != 0 or sample["metrics"]["full_original"]["max_abs"] != 0):
            raise ValueError("full BF16 GPU denominator must exactly match its saved replica")
        if sample["arm"] == "gpu":
            if sample["host_events"]: raise ValueError("do not invent GPU kernel events")
            continue
        events = sample["host_events"]
        names = {"gpu-fp16-cast-host-readback","private-model-feed-execute-read-fp32","gpu-suffix-output-ready","partial-gpu-upload-fp32-join-bf16-drain"}
        if len(events) != 4 or {e["name"] for e in events} != names: raise ValueError("complete conversion/private/remainder/join boundary required")
        for e in events:
            if (type(e["start_ns"]) is not int or type(e["duration_ns"]) is not int or e["start_ns"] < 0 or e["duration_ns"] <= 0
                    or e["start_ns"] + e["duration_ns"] > sample["wall_ns"] or e["clock_domain"] != "same-process-perf-counter-host"):
                raise ValueError("invalid host clock event")
        table = {e["name"]:(e["start_ns"],e["start_ns"]+e["duration_ns"]) for e in events}
        stage, private, gpu, join = [table[n] for n in ("gpu-fp16-cast-host-readback","private-model-feed-execute-read-fp32","gpu-suffix-output-ready","partial-gpu-upload-fp32-join-bf16-drain")]
        if stage[1] > min(private[0],gpu[0]) or join[0] < max(private[1],gpu[1]) or sample["arm"] == "serial" and private[1] > gpu[0]:
            raise ValueError("producer/consumer or serial ordering violated")
        overlaps.append({"phase":sample["phase"],"trial":sample["trial"],"arm":sample["arm"],
                         "api_host_intersection_ns":max(0,min(private[1],gpu[1])-max(private[0],gpu[0])),"physical_overlap_qualified":False})
    stats = []
    for a,b in (("gpu","serial"),("gpu","hybrid"),("serial","hybrid")):
        measured = {i:{} for i in range(repeats)}
        for row in samples:
            if row["phase"] == "measured": measured[row["trial"]][row["arm"]]=row["wall_ns"]
        values = [rows[a]/rows[b] for rows in measured.values()]
        rng = random.Random(73); boot = sorted(statistics.median(rng.choices(values,k=len(values))) for _ in range(2000))
        stats.append({"baseline":a,"candidate":b,"paired_ratios":values,"median_paired_host_ratio":statistics.median(values),
                      "bootstrap_95pct":[boot[49],boot[1949]],"production_qualified":False})
    if document["comparisons"] != stats: raise ValueError("ratio or uncertainty differs")
    return {"comparisons":stats,"host_intersections":overlaps,"production_qualified":False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True); parser.add_argument("--fixture",type=Path,required=True)
    parser.add_argument("--checkpoint",type=Path,required=True); parser.add_argument("--capture-root",type=Path,required=True)
    parser.add_argument("--upstream-root",type=Path,required=True); parser.add_argument("--source-root",type=Path,required=True)
    parser.add_argument("--run",action="append",required=True,help="discovery:ID or confirmation:ID")
    parser.add_argument("--failure",action="append",default=[]); parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists() or args.output.is_symlink(): parser.error("fresh curated output required")
    commit=subprocess.check_output(["git","-C",str(args.upstream_root),"rev-parse","HEAD"],text=True).strip()
    dirty=subprocess.check_output(["git","-C",str(args.upstream_root),"status","--porcelain","--untracked-files=no"],text=True)
    if commit!=COMMIT or dirty: parser.error("fixed clean upstream required")
    fixture=json.loads(args.fixture.read_text()); fixture_root=args.fixture.resolve().parent
    for name,expected in fixture["files"].items():
        if sha(local(fixture_root,name))!=expected: parser.error("fixture changed")
    layer=fixture["source_layer"]; stem=f"transformer_blocks.{layer}.img_mlp."
    tensor_identity(args.checkpoint,stem+"gate_up.weight",(24576,4096),fixture["original_gate_up_tensor_sha256"])
    tensor_identity(args.checkpoint,stem+"out.weight",(4096,12288),fixture["original_down_tensor_sha256"])
    capture_path=local(args.capture_root,"receipt.json")
    if sha(capture_path)!=fixture["capture_receipt_sha256"]: parser.error("capture identity differs")
    capture=json.loads(capture_path.read_text())
    result={"schema":"tc-qwen-private-ffn-evidence-v1","curator_sha256":sha(Path(__file__)),"production_qualified":False,
            "upstream_commit":commit,"selected_share":2048,"raw_machine_timestamps_exported":False,
            "verification_scope":"source content, reported all-call quality/boundaries and independent ratio arithmetic; not output-tensor/hardware rerun",
            "source_checks":{"fixture_sha256":sha(args.fixture),"original_tensor_content_verified":True,"capture_receipt_verified":True},"runs":[],"failures":[]}
    selections=[(*item.split(":",1),) for item in args.run]+[("failure",name) for name in args.failure]
    if len({name for _,name in selections})!=len(selections): parser.error("duplicate selected run")
    for group,name in selections:
        path=local(args.root,name+"/report.json"); value=json.loads(path.read_text())
        launched_path=local(args.root,name+"-launch/receipt.json"); launched=json.loads(launched_path.read_text())
        if launched.get("owned_process_exit_confirmed") is not True: parser.error("owned child exit unconfirmed")
        for filename,expected in value["upstream_sha256"].items():
            if sha(local(args.upstream_root/"aneforge",filename))!=expected: parser.error("upstream source differs")
        for item in value["source_snapshots"]:
            if sha(local(path.parent,item["path"]))!=item["sha256"]: parser.error("snapshot differs")
        if sha(local(args.source_root,"aneforge/_lib/libane_e5rt_dispatch.dylib"))!=value["shim_sha256"]: parser.error("shim differs")
        if sha(local(path.parent,"ane-model/model.mil"))!=value["private_mil_sha256"] or sha(local(path.parent,"ane-model/weights.bin"))!=value["private_weights_blob_sha256"]:
            parser.error("private program changed")
        if value["native_weight_sha256"]!=native_prefix_hashes(fixture_root,fixture,value["share"]): parser.error("paired FP16 prefix content differs")
        point=next(p for p in fixture["points"] if p["id"]==value["point"])
        if (value["original_input_tensor_sha256"]!=point["original_input_tensor_sha256"] or value["original_capture_file_sha256"]!=point["capture_file_sha256"]
                or value["fixture_sha256"]!=sha(args.fixture) or value["capture_receipt_sha256"]!=fixture["capture_receipt_sha256"]): parser.error("real source link differs")
        records=[r for r in capture["records"] if Path(r["path"]).name==f'layer-{layer}-step-{point["step"]}.safetensors']
        if len(records)!=1: parser.error("unique original capture required")
        captured=local(args.capture_root,records[0]["path"])
        if sha(captured)!=point["capture_file_sha256"]: parser.error("capture file changed")
        meta=tensor_identity(captured,"input",(1,1024,4096),point["original_input_tensor_sha256"])
        if meta.get("trajectory")!="forced-gpu" or meta.get("capture_recipe")!="qwen21-real-ffn-bf16-v1": parser.error("capture recipe differs")
        validation=None
        if group=="failure":
            if value["status"]!="failed" or value["samples"] or value["comparisons"] is not None or launched["returncode"]==0: parser.error("unproven failure")
        else:
            if launched["returncode"]!=0: parser.error("incomplete measured process")
            validation=verify_report(value,group)
        public=copy.deepcopy(value); public.pop("checkpoint_stat_before",None)
        record={"id":name,"group":group,"raw_report_sha256":sha(path),"supervisor_sha256":sha(launched_path),
                "child_exit_confirmed":True,"child_returncode":launched["returncode"],"independent_data_validation":validation,"report":public}
        result["failures" if group=="failure" else "runs"].append(record)
    encoded=json.dumps(result,indent=2,allow_nan=False)+"\n"
    if "/Users/" in encoded or "/Volumes/" in encoded: parser.error("machine-local paths must remain private")
    with args.output.open("x",encoding="utf-8") as stream: stream.write(encoded)
    print(json.dumps({"runs":len(result["runs"]),"failures":len(result["failures"]),"production_qualified":False}))


if __name__=="__main__": main()

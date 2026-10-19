#!/usr/bin/env python3
"""Independent CPU saved-BF16/source/denominator recheck, no GPU work."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct
import sysconfig
import tarfile

import numpy as np

from flux9_bf16_framework_screen import write_json
import qwen_lora_swiglu_block_screen as screen
from runtime_ane_common import sha256_file

ROOT=Path(__file__).resolve().parents[2]


def arrays(path):
    with path.open("rb") as stream:
        size=struct.unpack("<Q",stream.read(8))[0]
        if not 2<=size<=1<<20:raise ValueError("bounded raw tensor header required")
        header=json.loads(stream.read(size));payload=stream.read()
    result={}
    for key,value in header.items():
        if key=="__metadata__":continue
        dtype=value["dtype"];shape=tuple(value["shape"]);a,b=value["data_offsets"]
        width={"BF16":2,"F32":4}.get(dtype)
        if width is None or not 0<=a<=b<=len(payload) or b-a!=math.prod(shape)*width:raise ValueError("complete typed raw array required")
        raw=payload[a:b];words=np.frombuffer(raw,dtype="<u2" if dtype=="BF16" else "<u4")
        mask=0x7f80 if dtype=="BF16" else 0x7f800000
        if np.any(words&mask==mask):raise ValueError("nonfinite complete saved tensor")
        result[key]=(dtype,shape,raw)
    return result


def same(a,b):
    if a!=b:raise ValueError("full saved BF16 tensor differs")


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--library",type=Path,required=True);p.add_argument("--binary",type=Path,required=True)
    p.add_argument("--native-source-archive",type=Path,required=True);p.add_argument("--cohort",type=Path,action="append",required=True)
    p.add_argument("--output",type=Path,required=True);args=p.parse_args()
    if args.output.exists():p.error("fresh audit output required")
    manifest=json.loads((args.library.parent/"runtime-build/runtime-build-manifest.json").read_text());sources=manifest["inputs"]["sources"]
    with tarfile.open(args.native_source_archive,"r:gz") as archive:
        members={member.name:member for member in archive.getmembers()}
        if set(members)!={source["path"] for source in sources}:raise ValueError("complete sealed native archive required")
        for source in sources:
            stream=archive.extractfile(members[source["path"]])
            if stream is None or hashlib.sha256(stream.read()).hexdigest()!=source["sha256"] or sha256_file(ROOT/source["path"])!=source["sha256"]:
                raise ValueError("archived/current native source differs from library")
    library_sha=sha256_file(args.library);observations=[];seen=set();by_case={};file_hashes=[];binary_sha=sha256_file(args.binary);exact_arrays=0
    for root in args.cohort:
        summary=json.loads((root/"summary.json").read_text());native=json.loads((root/"trial/native.json").read_text());key=summary["edit"],summary["order"]
        if summary["status"]!="complete" or summary["asset_sha256"]["library"]!=library_sha or native["loaded_runtime_id"]!=manifest["runtime_build_id"] or key in seen:
            raise ValueError("distinct same-library complete windows required")
        seen.add(key)
        if binary_sha!=summary["asset_sha256"]["binary"]:raise ValueError("same sealed probe binary required")
        snapshots=json.loads((root/"source.json").read_text());headers=json.loads((root/"mlx-headers.json").read_text())
        for name,digest in summary["source_sha256"].items():
            if hashlib.sha256(snapshots[name].encode()).hexdigest()!=digest or sha256_file(ROOT/name)!=digest:raise ValueError("bound screen/probe source differs")
        for name,digest in summary["mlx_header_sha256"].items():
            current=Path(sysconfig.get_paths()["purelib"])/"mlx/include/mlx/backend/metal/kernels"/name
            if hashlib.sha256(headers[name].encode()).hexdigest()!=digest or sha256_file(current)!=digest:raise ValueError("typed operator source snapshot differs")
        analysis=screen.analyze(native,summary["order"])
        if analysis!=summary["analysis"]:raise ValueError("strict full denominator recheck differs")
        # Recompute from raw samples independently of the collector analyzer.
        medians={arm:statistics.median(row["seconds"] for row in native["samples"] if row["iteration"]>=0 and row["arm"]==arm) for arm in screen.ARMS}
        primary=100*(1-medians[screen.PRIMARY]/medians[screen.CONTROL])
        if medians!=analysis["median_seconds"] or primary!=analysis["primary_reduction_percent"]:raise ValueError("direct strong baseline denominator differs")
        diagnostics=arrays(root/"trial/inputs-and-hidden.safetensors");reference=diagnostics["block_reference"]
        if reference[:2]!=("BF16",(1,1024,4096)):raise ValueError("complete BF16 block boundary required")
        if hashlib.sha256(reference[2]).hexdigest()!=native["ordinary_block_sha256"] or hashlib.sha256(diagnostics["argument-0"][2]).hexdigest()!=native["input_sha256"]:
            raise ValueError("actual input/reference identity differs")
        if diagnostics["changed_reference"]==reference:raise ValueError("fresh input did not recompute block")
        hidden=diagnostics["gpu-replay-hidden"]
        if hidden[:2]!=("BF16",(1,1024,12288)):raise ValueError("complete corrected hidden required")
        for name in ("paired-b-swiglu-b16-hidden","paired-b-swiglu-b32-hidden"):same(hidden,diagnostics[name]);exact_arrays+=1
        outputs=arrays(root/"trial/complete-block-outputs.safetensors")
        if set(outputs)!=set(screen.ARMS):raise ValueError("all final strong/control/candidate outputs required")
        for value in outputs.values():same(reference,value);exact_arrays+=1
        for arm in screen.ARMS[1:]:same(reference,arrays(root/f"trial/{arm}-initial.safetensors")["block"]);exact_arrays+=1
        prior=by_case.get(summary["edit"])
        if prior:
            if (native["actual_layer_source_sha256"]!=prior[0] or diagnostics!=prior[1]):raise ValueError("repeat window source/input/hidden/reference differs")
        else:by_case[summary["edit"]]=(native["actual_layer_source_sha256"],diagnostics)
        for kind in ("evidence","report"):
            if sha256_file(root/summary["memory"][kind])!=summary["memory"][kind+"_sha256"]:raise ValueError("memory evidence changed")
        memory=summary["memory"]["verified"]
        if not memory["complete"] or memory["command_exit_code"]!=0:raise ValueError("complete process window required")
        loads=[json.loads(line) for line in (root/"load.jsonl").read_text().splitlines()]
        observations.append(dict(cohort=root.name,edit=key[0],order=key[1],analysis=analysis,memory=memory,load_error=summary.get("load_error"),
            gpu_replay_vs_original_percent=100*(medians["gpu-replay"]/medians["gpu-original"]-1),
            competing_load_samples=sum(bool(row["competing_processes"]) for row in loads),summary_sha256=sha256_file(root/"summary.json")))
        file_hashes.extend(dict(path=str(path),sha256=sha256_file(path)) for path in sorted((root/"trial").glob("*.safetensors")))
    if seen!={(edit,order) for edit in (False,True) for order in ("forward","reverse")}:raise ValueError("both full forward/reverse workload windows required")
    if by_case[False][0]!=by_case[True][0]:raise ValueError("generation/edit original layer/factors differ")
    result=dict(schema="tc-qwen-lora-swiglu-block-audit-v1",status="complete",qualification_passed=False,library_sha256=library_sha,probe_binary_sha256=binary_sha,
        runtime_build_id=manifest["runtime_build_id"],native_source_hashes_verified=len(sources),native_source_archive_sha256=sha256_file(args.native_source_archive),
        saved_exact_array_comparisons=exact_arrays,measured_full_blocks=147*len(observations),warm_full_blocks=21*len(observations),
        preflight_full_blocks=19*len(observations),partial_hidden_taps=3*len(observations),observations=observations,raw_file_sha256=file_hashes,
        all_primary_local_signals=all(row["analysis"]["primary_local_signal"] for row in observations),script_sha256=sha256_file(Path(__file__)),
        scope="CPU re-read saved first/final full block and hidden arrays, sources and sample denominators; per-call live byte-hash equality logs validated, not raw replay of every sample or full-image/ANE/physical/exclusive qualification")
    write_json(args.output,result);print(json.dumps(result,indent=2))


if __name__=="__main__":main()

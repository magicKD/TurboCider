#!/usr/bin/env python3
"""Portable digest-bound summary of private GGUF diagnostics, not qualification."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    with path.open("rb") as source:return hashlib.file_digest(source,"sha256").hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reports",type=Path,nargs="+",required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():p.error("output already exists")
    summary={"schema":"tc-gguf-private-diagnostics-v1","scope":"selected real inputs; not full R1/model/memory/performance certification","cases":[]}
    for path in args.reports:
        report=json.loads(path.read_text());case={"id":path.parent.name,"raw_report_sha256":digest(path),
            "library_sha256":report["library_sha256"],"runs":[]}
        reference=path.parent/"packed-0-tensors"
        for row in report["runs"]:
            metrics=row.get("metrics") or {}
            record={"prefetch":row["prefetch"],"status":row["status"],"error":row.get("error"),
                "png_sha256":row.get("png_sha256"),"cancellation_triggered":row.get("cancellation_triggered",False),
                "wall_seconds":row["wall_seconds"],"timings_seconds":metrics.get("timings_seconds"),
                "quantized_execution":metrics.get("quantized_execution"),"mlx_memory":metrics.get("memory"),
                "vm_deltas":row.get("vm_deltas"),"size":row["request"]["outputs"][0]["width"],
                "seed":row["request"]["sampling"]["seed"],"steps":row["request"]["sampling"]["steps"],
                "profile":row["request"]["execution"].get("quantized_execution",{}).get("precision_profile","native-packed")}
            q=record["quantized_execution"]
            if row["status"]==0 and q:
                if q["fill_count"]!=record["steps"]*30 or q["slot_count"]!=row["prefetch"]+1:
                    raise ValueError("observed slot/fill counts inconsistent")
            name=f'p{row["prefetch"]}-{row["index"]}' if row["prefetch"]>=0 else f'packed-{row["index"]}'
            current=path.parent/(name+"-tensors")
            if reference.is_dir() and current.is_dir():
                files=list(reference.glob('*.safetensors'))
                record["reference_tensors_bytes_exact"]=bool(files) and all((current/file.name).is_file() and digest(current/file.name)==digest(file) for file in files)
            case["runs"].append(record)
        case["numerical_gates"]={path.name:json.loads(path.read_text()) for path in path.parent.glob('*-*.json') if 'events' not in path.name and path.name!='report.json'}
        summary["cases"].append(case)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x") as out:json.dump(summary,out,indent=2,allow_nan=False);out.write("\n")
    print(args.output)

if __name__=="__main__":main()

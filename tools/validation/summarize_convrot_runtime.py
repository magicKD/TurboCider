#!/usr/bin/env python3
"""Portable raw-bound ConvRot consumer diagnostics; not runtime qualification."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source,"sha256").hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reports",type=Path,nargs="+",required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists() or a.output.is_symlink():p.error("output already exists")
    result={"schema":"tc-convrot-private-diagnostics-v1","production_qualified":False,
            "scope":"actual execution/fallback/quality negatives; no performance/INT8/whole-memory qualification","cases":[]}
    for path in a.reports:
        raw=json.loads(path.read_text())
        if raw.get("schema")!="tc-convrot-runtime-consumer-screen-v1":raise ValueError("wrong raw report schema")
        case={"id":path.parent.name,"raw_report_sha256":digest(path),"library_sha256":raw["library_sha256"],
              "source_sha256":raw["source_sha256"],"manifest_sha256":raw["manifest_sha256"],"route":raw["route"],"runs":[]}
        for row in raw["runs"]:
            m=row.get("metrics") or {};h=m.get("hybrid") or {}
            case["runs"].append({"status":row["status"],"error":row.get("error"),"png_sha256":row.get("png_sha256"),
                "wall_seconds_diagnostic_only":row["wall_seconds"],"timings_seconds":m.get("timings_seconds"),
                "runtime_backend":m.get("runtime_backend"),"runtime_precision":m.get("runtime_precision"),
                "runtime_calls_session_total":h.get("runtime_calls_session_total"),"runtime_failed":h.get("runtime_failed"),
                "runtime_weight":h.get("runtime_weight"),"observed_ane_residency":h.get("observed_ane_residency"),
                "placement":"unknown","arithmetic":"unknown","vm_deltas":row.get("vm_deltas")})
        case["numerical_gates"]={p.name:json.loads(p.read_text()) for p in path.parent.glob("vs-*.json")}
        result["cases"].append(case)
    result["summarizer_sha256"]=digest(Path(__file__))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open("x") as output:json.dump(result,output,indent=2,allow_nan=False);output.write("\n")


if __name__=="__main__":main()

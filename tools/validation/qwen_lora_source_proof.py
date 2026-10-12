#!/usr/bin/env python3
"""Validate native cold/warm source verification diagnostics, not placement."""
import argparse
import json
from pathlib import Path

SCOPE="native generation-bound full SHA256; held-fd/path revalidation; bound A/B materialized before successful leased bind"


def validate_source_proof(lines,expected_bytes,requests=3):
    if type(expected_bytes) is not int or expected_bytes<=0 or type(requests) is not int or requests<3:
        raise ValueError("need positive adapter size and cold plus at least two warm requests")
    records=[]
    for line in lines:
        if "qwen_lora_source_verification" not in line:continue
        record=json.loads(line).get("qwen_lora_source_verification")
        if not isinstance(record,dict):raise ValueError("malformed source verification diagnostic")
        records.append(record)
    if len(records)!=requests:raise ValueError("missing/duplicate source verification request records")
    for index,record in enumerate(records):
        if any(type(record.get(k)) is not int for k in ("bytes_read","native_cache_hits","source_files")) or \
            record["source_files"]!=1 or record.get("scope")!=SCOPE or \
            record["bytes_read"]!=(expected_bytes if index==0 else 0) or record["native_cache_hits"]!=(0 if index==0 else 1) or \
            record.get("bound_this_request") is not (index==0):
            raise ValueError("need native cold full hash and unchanged-generation warm proof, not omitted source checks/rebinding")
    return records


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stderr",type=Path)
    parser.add_argument("--bytes",type=int,required=True)
    parser.add_argument("--requests",type=int,default=3)
    args=parser.parse_args()
    rows=validate_source_proof(args.stderr.read_text().splitlines(),args.bytes,args.requests)
    print(json.dumps(dict(status="validated_native_source_diagnostics",records=rows,qualification_passed=False),indent=2))


if __name__=="__main__":main()

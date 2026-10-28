#!/usr/bin/env python3
"""Search Markdown knowledge directly; no catalog files, devices or network."""
import argparse
import json
from pathlib import Path
import sys
from knowledge import FIELDS, INTENTS, ROOT, load, query


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="*")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--intent", choices=INTENTS, default="learn")
    parser.add_argument("--type", dest="kind", action="append")
    for field in FIELDS:
        parser.add_argument("--" + field.replace("_", "-"), dest=field, action="append")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    try:
        filters = {field: getattr(args, field) for field in ("kind", *FIELDS) if getattr(args, field)}
        result = query(load(args.root), " ".join(args.query), filters, args.limit, args.intent)
    except (ValueError, OSError) as error:
        print(json.dumps({"error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2
    if args.compact:
        for node in result["results"]:
            print(f'{node["id"]}\t{node["page_type"][0]}\t{node["title"]}\t{node["path"]}')
        for warning in result["warnings"]:
            print(warning, file=sys.stderr)
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

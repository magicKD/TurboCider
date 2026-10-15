#!/usr/bin/env python3
"""Search catalog chapters and registered local sources without executing code."""
import argparse
import json
from pathlib import Path
import sys

from knowledge import FIELDS, ROOT, load, query


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="*", help="keywords, including Chinese text; ranked OR search")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--type", dest="kind", action="append")
    for field in FIELDS:
        parser.add_argument("--" + field, action="append")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    try:
        filters = {name: getattr(args, name) for name in (*FIELDS, "kind") if getattr(args, name)}
        value = query(load(args.root), " ".join(args.query), filters, args.limit)
    except (ValueError, KeyError, OSError) as error:
        print(json.dumps({"error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2
    if args.compact:
        for node in value["results"]:
            print(f'{node["id"]}\t{node["title"]}\t{node["path"]}')
    else:
        print(json.dumps(value, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

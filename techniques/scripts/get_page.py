#!/usr/bin/env python3
"""Expand a registered local page and optional source/code assets, never execute."""
import argparse
import json
from pathlib import Path
import sys

from knowledge import ROOT, get_page, load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lookup", help="stable ID or registered relative page path")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--follow-sources", action="store_true")
    parser.add_argument("--include-code", action="store_true")
    parser.add_argument("--max-bytes", type=int, default=24000)
    args = parser.parse_args()
    try:
        value = get_page(load(args.root), args.lookup, args.follow_sources, args.include_code, args.max_bytes)
    except (ValueError, KeyError, OSError) as error:
        print(json.dumps({"error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(value, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

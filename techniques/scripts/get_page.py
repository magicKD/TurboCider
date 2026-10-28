#!/usr/bin/env python3
"""Read a knowledge page with bounded optional source/example expansion."""
import argparse
import json
from pathlib import Path
import sys
from knowledge import ROOT, get_page, load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lookup", help="stable ID, alias or relative Markdown page")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--follow-sources", action="store_true")
    parser.add_argument("--include-code", action="store_true")
    parser.add_argument("--max-bytes", type=int, default=24000)
    parser.add_argument("--offset-bytes", type=int, default=0)
    parser.add_argument("--follow-evidence", "--include-negative-results", action="store_true",
                        help="legacy flag: experiment payloads are no longer included")
    args = parser.parse_args()
    try:
        result = get_page(load(args.root), args.lookup, args.follow_sources, args.include_code,
                          args.max_bytes, args.offset_bytes)
        if args.follow_evidence:
            result["warnings"] = ["Experiment payloads are not distributed. Negative findings remain in case-study notes."]
    except (ValueError, OSError) as error:
        print(json.dumps({"error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

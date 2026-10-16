#!/usr/bin/env python3
"""Offline topic/alias/content-binding audit. No latest-version or device probe."""
import argparse
import json
from pathlib import Path
import sys

from knowledge import ROOT, load
from retrieval_contracts import context, iso_day, review


def audit(root=ROOT, observed=None, as_of=None):
    observed = context({} if observed is None else observed)
    if as_of is not None:
        iso_day(as_of)
    index = load(root)
    reviews = [review(claim, observed, as_of) for claim in index["claims"].values()]
    return {"metadata_coverage": index["metadata_coverage"], "version_reviews": reviews,
            "needs_review": [r["id"] for r in reviews if r["state"] == "needs-review"],
            "retrieval_grants_qualification": False, "errors": [],
            "scope": "registered local topics, aliases, exact historical content/context and date windows; no release discovery, model/hardware qualification or execution"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--context", default="{}", help="exact recorded context JSON, not a capability probe")
    parser.add_argument("--as-of", help="explicit YYYY-MM-DD for deterministic review")
    parser.add_argument("--strict", action="store_true", help="fail if content, date or supplied environment needs review; missing context stays explicitly incomplete")
    args = parser.parse_args()
    try:
        result = audit(args.root, json.loads(args.context), args.as_of)
    except (ValueError, KeyError, OSError) as error:
        print(json.dumps({"errors": [str(error)]}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return int(args.strict and bool(result["needs_review"]))


if __name__ == "__main__":
    raise SystemExit(main())

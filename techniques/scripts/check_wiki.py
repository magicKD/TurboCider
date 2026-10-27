#!/usr/bin/env python3
"""Lightweight documentation maintenance, not an experiment verifier."""
import argparse
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

from knowledge import ROOT, LINK, load


def without_fences(text):
    lines, fenced = [], False
    for line in text.splitlines():
        if line.lstrip().startswith(("```", "~~~")):
            fenced = not fenced
        elif not fenced:
            lines.append(line)
    return "\n".join(lines)


def audit(root=ROOT):
    root = Path(root).resolve()
    errors, warnings, incoming, links, external_local = [], [], set(), 0, 0
    forbidden = [p for p in root.rglob("*") if p.is_file() and
                 (p.suffix in (".json", ".jsonl", ".ndjson") or p.name.endswith((".json.gz", ".jsonl.gz")))]
    for path in forbidden:
        errors.append("experiment/data artifact in knowledge tree: " + path.relative_to(root).as_posix())
    pages = sorted(root.rglob("*.md"))
    for page in pages:
        relative = page.relative_to(root).as_posix()
        if not page.resolve().is_relative_to(root):
            errors.append("escaping Markdown symlink: " + relative)
            continue
        text = page.read_text(encoding="utf-8")
        if not text.startswith("# "):
            errors.append("missing title: " + relative)
        if re.search(r"/Users/[A-Za-z0-9_.-]+/|(?:sk-|ghp_|github_pat_)[A-Za-z0-9_-]{16,}", text):
            errors.append("private path or apparent credential: " + relative)
        for label, raw in LINK.findall(without_fences(text)):
            target = urlsplit(raw.strip("<>"))
            if target.scheme or target.netloc or not target.path:
                continue
            path = (page.parent / unquote(target.path)).resolve()
            if not path.is_relative_to(root):
                external_local += 1
                continue
            links += 1
            if not path.exists():
                errors.append(relative + ": missing link " + raw)
            if path.suffix == ".md":
                incoming.add(path)
    for page in pages:
        if page != root / "README.md" and page.resolve() not in incoming:
            warnings.append("no incoming learning link: " + page.relative_to(root).as_posix())
    try:
        index = load(root)
        registered = len(index["nodes"])
    except (OSError, ValueError) as error:
        errors.append(str(error))
        registered = 0
    return {"pages": len(pages), "indexed_pages": registered, "local_links": links,
            "optional_product_links": external_local, "forbidden_data_files": len(forbidden),
            "errors": errors, "warnings": warnings,
            "scope": "Documentation links/metadata and no-data policy only; not numerical, device or model verification."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    result = audit(args.root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(bool(result["errors"]))

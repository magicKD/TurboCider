#!/usr/bin/env python3
"""Offline link, navigation, portable-path and source-symbol audit for this wiki."""
import argparse
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

LINK = re.compile(r"!?\[[^\]]*\]\((<[^>]+>|[^\s)]+)(?:\s+\"[^\"]*\")?\)")


def without_fences(text):
    lines, fenced = [], False
    for line in text.splitlines():
        if line.lstrip().startswith(("```", "~~~")):
            fenced = not fenced
            lines.append("")
        else:
            lines.append("" if fenced else line)
    return "\n".join(lines)


def audit(root, include_integrations=False):
    root = root.resolve()
    errors, warnings, links, integration_links = [], [], 0, 0
    pages = sorted(root.rglob("*.md"))
    incoming = set()
    for page in pages:
        content = page.read_text(encoding="utf-8")
        if not content.startswith("# "):
            errors.append(f"{page.relative_to(root)}: missing page title")
        if re.search(r"/Users/[A-Za-z0-9_.-]+/|sk-[A-Za-z0-9_-]{12,}", content):
            errors.append(f"{page.relative_to(root)}: machine-specific path or apparent credential")
        for match in LINK.finditer(without_fences(content)):
            raw = match.group(1).strip("<>")
            target = urlsplit(raw)
            if target.scheme or target.netloc or not target.path:
                continue
            links += 1
            destination = (page.parent / unquote(target.path)).resolve()
            if not destination.is_relative_to(root):
                integration_links += 1
                if not include_integrations:
                    continue
            if not destination.exists():
                errors.append(f"{page.relative_to(root)}: missing link {raw}")
            if destination.suffix == ".md" and destination.is_relative_to(root):
                incoming.add(destination)
    for page in pages:
        if page != root / "README.md" and page not in incoming:
            errors.append(f"{page.relative_to(root)}: no incoming wiki link")
    manifest = root / "references" / "sources.json"
    checked, snapshots, skipped_integrations = 0, 0, 0
    if not manifest.exists():
        errors.append("missing references/sources.json")
    else:
        data = json.loads(manifest.read_text(encoding="utf-8"))
        ids = [source["id"] for source in data["sources"]]
        if len(set(ids)) != len(ids):
            errors.append("duplicate source IDs")
        for source in data["sources"]:
            if source.get("scope") == "integration" and not include_integrations:
                skipped_integrations += 1
                continue
            base = root if source.get("scope") == "local" else root.parent
            path = (base / source["path"]).resolve()
            if source.get("scope") == "local" and not path.is_relative_to(root):
                errors.append("local source escapes techniques: " + source["path"])
                continue
            if not path.is_file():
                (warnings if source.get("optional", False) else errors).append("source missing: " + source["path"])
                continue
            checked += 1
            if source.get("sha256"):
                snapshots += 1
                if hashlib.sha256(path.read_bytes()).hexdigest() != source["sha256"]:
                    (warnings if source.get("optional", False) else errors).append("source snapshot changed; re-review: " + source["path"])
            if source.get("symbols"):
                content = path.read_text(encoding="utf-8")
                for symbol in source["symbols"]:
                    if symbol not in content:
                        errors.append(source["path"] + ": symbol missing: " + symbol)
    return {"pages": len(pages), "local_links": links - integration_links,
            "all_relative_links": links, "optional_integration_links": integration_links,
            "source_files_checked": checked, "external_snapshots_checked": snapshots,
            "optional_integration_sources_skipped": skipped_integrations,
            "scope": "local paths/navigation/symbols; --include-integrations additionally checks optional product/sibling anchors; not semantic, URL or anchor validation",
            "errors": errors, "warnings": warnings}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--include-integrations", action="store_true", help="also check optional product/sibling source anchors")
    args = parser.parse_args()
    result = audit(args.root, args.include_integrations)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return bool(result["errors"])


if __name__ == "__main__":
    raise SystemExit(main())

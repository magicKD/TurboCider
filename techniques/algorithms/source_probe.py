#!/usr/bin/env python3
"""Collect selected public source evidence without installing/running project code.

The generated receipt contains no absolute checkout paths. Existing checkout
HEADs are never updated or reset. --download may retrieve missing blobs from
partial clones; without it lazy network fetching is disabled.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


def git(root, *args, allow_fetch=False):
    env = dict(os.environ)
    if not allow_fetch:
        env["GIT_NO_LAZY_FETCH"] = "1"
    return subprocess.run(
        ["git", "--no-optional-locks", "-c", "core.fsmonitor=false", "-C", str(root), *args],
        check=True, capture_output=True, timeout=180, env=env,
    ).stdout


def collect(spec, root, download):
    repo = spec["repository"]
    if not re.fullmatch(r"[a-z0-9-]+", spec["id"]):
        raise ValueError("invalid source id")
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo):
        raise ValueError("invalid public repository")
    folder = root / spec["id"]
    if folder.is_symlink():
        raise ValueError("symlink checkout rejected")
    if folder.exists() and not (folder / ".git").exists():
        raise ValueError("existing non-checkout preserved")
    if not folder.exists():
        if not download:
            return {"id": spec["id"], "status": "not_downloaded"}
        subprocess.run(
            ["git", "clone", "--depth", "1", "--filter=blob:none", "--no-checkout",
             "https://github.com/" + repo + ".git", str(folder)],
            check=True, capture_output=True, timeout=180,
        )
    commit = git(folder, "rev-parse", "HEAD").decode().strip()
    tree = git(folder, "ls-tree", "-r", "--name-only", "HEAD").decode().splitlines()
    patterns = spec.get("patterns", [r"(^|/)README.*\.md$"])
    selected = [p for p in tree if any(re.search(pattern, p) for pattern in patterns)]
    files = []
    for path in selected:
        blob = git(folder, "show", commit + ":" + path, allow_fetch=download)
        files.append({"path": path, "bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest(),
                      "url": f"https://github.com/{repo}/blob/{commit}/{path}"})
    return {"id": spec["id"], "repository": repo, "commit": commit,
            "status": "source_collected_not_automatically_reviewed", "files": files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; use a new receipt path")
    specs = json.loads(args.plan.read_text())["projects"]
    for spec in specs:
        if not re.fullmatch(r"[a-z0-9-]+", spec["id"]):
            parser.error("invalid source id")
    args.root.mkdir(parents=True, exist_ok=True)
    def one(spec):
        try:
            return collect(spec, args.root, args.download)
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            return {"id": spec["id"], "status": "collection_failed", "error_type": type(error).__name__}
    with ThreadPoolExecutor(max_workers=4) as pool:
        projects = list(pool.map(one, specs))
    result = {"schema": "algorithm-source-probe-v1", "network_enabled": args.download,
              "project_code_executed": False, "projects": projects}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    for item in projects:
        print(item["id"], item["status"], item.get("commit", ""), len(item.get("files", [])))
    return int(any(p["status"] == "collection_failed" for p in projects))


if __name__ == "__main__":
    raise SystemExit(main())

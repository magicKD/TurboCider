#!/usr/bin/env python3
"""Bounded, read-only GitHub source acquisition; never runs upstream code.

Snapshots and machine-local receipts belong in a new ignored outputs directory.
An acquired file is NOT marked read or experimentally verified by this tool.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fnmatch
import hashlib
import json
from pathlib import Path, PurePosixPath
import urllib.error
import urllib.request


RECIPES = {
    "maderix/ANE": ["README.md", "LICENSE*", "training/README.md",
        "training/training_dynamic/mil_dynamic.h", "training/training_dynamic/io.h",
        "training/training_dynamic/train.m", "training/training_dynamic/cpu_ops.h"],
    "MrZaKaRiA/ane": ["README.md", "LICENSE*", "*int8*.m", "gpu_ane_share.m"],
    "mechramc/Orion": ["README.md", "LICENSE*", "src/*delta*", "src/*lora*",
        "src/*train*", "src/*ane*", "docs/*"],
    "vipuldivyanshu92/ANEgpt": ["README.md", "LICENSE*", "*mil*.h", "*train*.m"],
    "johnmai-dev/ANE-LM": ["README*", "LICENSE*", "Sources/*/*ANE*.swift",
        "Sources/*/*MIL*.swift", "Sources/*/*Model*.swift"],
    "skyfallsin/apple-neural-engine-field-guide": ["README.md", "LICENSE*", "program.md"],
    "Anemll/anemll": ["README.md", "LICENSE*", "anemll/models/llama_model.py",
        "anemll/ane_converter/llama_converter.py", "anemll/utils/convert_model.py"],
    "Anemll/anemll-bench": ["README.md", "LICENSE*", "*.swift", "*.m"],
    "apple-aiml-research/ml-stable-diffusion": ["README.md", "LICENSE*",
        "python_coreml_stable_diffusion/attention.py",
        "python_coreml_stable_diffusion/torch2coreml.py",
        "swift/StableDiffusion/pipeline/StableDiffusionPipeline.swift",
        "swift/StableDiffusion/pipeline/ManagedMLModel.swift"],
    "apple/coreai-models": ["README.md", "LICENSE*", "models/*/README.md"],
    "ggml-org/whisper.cpp": ["README.md", "LICENSE*", "src/whisper.cpp",
        "models/convert-whisper-to-coreml.py", "src/coreml/whisper-encoder.mm"],
    "argmaxinc/WhisperKit": ["README.md", "LICENSE*",
        "Sources/WhisperKit/Core/Configurations.swift",
        "Sources/WhisperKit/Core/Models.swift"],
    "tlkh/asitop": ["README.md", "LICENSE*", "asitop/utils.py", "asitop/asitop.py"],
    "context-labs/mactop": ["README.md", "LICENSE*", "internal/metrics/*ane*"],
}


def read_url(url, limit):
    request = urllib.request.Request(url, headers={"User-Agent": "ANE-learning-source-audit/1"})
    with urllib.request.urlopen(request, timeout=40) as response:
        value = response.read(limit + 1)
    if len(value) > limit:
        raise ValueError("source exceeds acquisition byte cap")
    return value


def acquire(repo, root):
    destination = root / repo.replace("/", "--")
    destination.mkdir()
    record = {"repo": repo, "status": "error", "files": [], "errors": [],
              "read_status": "acquired-not-read", "execution": "not-run"}
    try:
        metadata = json.loads(read_url("https://api.github.com/repos/" + repo, 500_000))
        record["resolved_repository"] = metadata["full_name"]
        record["is_fork"] = bool(metadata.get("fork"))
        record["parent_repository"] = (metadata.get("parent") or {}).get("full_name")
        branch = metadata["default_branch"]
        head = json.loads(read_url("https://api.github.com/repos/" + repo + "/commits/" + branch, 1_000_000))
        sha = head["sha"]
        tree = json.loads(read_url("https://api.github.com/repos/" + repo + "/git/trees/" + sha + "?recursive=1", 5_000_000))
        if tree.get("truncated"):
            raise ValueError("truncated repository tree")
        record.update(commit=sha, commit_date=head["commit"]["committer"]["date"],
                      license=(metadata.get("license") or {}).get("spdx_id"),
                      source_url="https://github.com/" + repo + "/tree/" + sha)
        candidates = []
        for entry in tree["tree"]:
            path = entry["path"]
            if (entry["type"] == "blob" and entry.get("size", 1_000_001) <= 1_000_000
                    and any(fnmatch.fnmatchcase(path, pattern) for pattern in RECIPES[repo])):
                if PurePosixPath(path).is_absolute() or ".." in PurePosixPath(path).parts:
                    raise ValueError("unsafe upstream path")
                candidates.append(path)
        if len(candidates) > 45:
            raise ValueError("too many selected files; narrow recipe")
        # Tree is a locator, not proof that any particular implementation was read.
        (destination / "tree.json").write_text(json.dumps(tree, indent=2) + "\n")
        for path in sorted(candidates):
            url = "https://raw.githubusercontent.com/" + repo + "/" + sha + "/" + path
            try:
                value = read_url(url, 1_000_000)
                target = destination / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(value)
                record["files"].append({"path": path, "sha256": hashlib.sha256(value).hexdigest(),
                    "bytes": len(value), "lines": len(value.splitlines()),
                    "url": "https://github.com/" + repo + "/blob/" + sha + "/" + path})
            except (ValueError, OSError, urllib.error.URLError) as error:
                record["errors"].append({"path": path, "error": str(error)})
        record["status"] = "acquired" if not record["errors"] else "partial"
    except (KeyError, ValueError, OSError, urllib.error.URLError) as error:
        record["errors"].append({"error": str(error)})
    (destination / "receipt.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo", action="append", choices=sorted(RECIPES))
    parser.add_argument("--extend", type=Path, help="existing acquisition; fetch --file paths at its pinned commits, without API calls")
    parser.add_argument("--file", action="append", default=[], help="repo:path, explicit paths only; requires --extend")
    args = parser.parse_args()
    if args.output_dir.exists() or args.output_dir.is_symlink():
        parser.error("use a new task directory")
    # A bounded source-only run: no git checkout, installs, model assets, or exec.
    args.output_dir.mkdir(parents=True)
    if args.extend:
        known = {r["repo"]: r for r in json.loads((args.extend / "receipt.json").read_text())["repos"]}
        receipts, errors = [], []
        for item in args.file:
            repo, path = item.split(":", 1)
            if repo not in known or not known[repo].get("commit"):
                parser.error("extension requires an already pinned repo")
            if (PurePosixPath(path).is_absolute() or ".." in PurePosixPath(path).parts
                    or not path or len(args.file) > 40):
                parser.error("explicit safe path and <=40 files required")
            sha = known[repo]["commit"]
            try:
                value = read_url("https://raw.githubusercontent.com/" + repo + "/" + sha + "/" + path, 1_000_000)
            except (ValueError, OSError, urllib.error.URLError) as error:
                errors.append({"repo": repo, "path": path, "error": str(error)})
                continue
            target = args.output_dir / repo.replace("/", "--") / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(value)
            receipts.append({"repo": repo, "commit": sha, "path": path,
                "sha256": hashlib.sha256(value).hexdigest(), "lines": len(value.splitlines()),
                "read_status": "acquired-not-read", "execution": "not-run",
                "url": "https://github.com/" + repo + "/blob/" + sha + "/" + path})
        (args.output_dir / "receipt.json").write_text(json.dumps({"files": receipts, "errors": errors}, indent=2) + "\n")
        print(json.dumps({"files_acquired": len(receipts), "errors": errors}, indent=2))
        return int(bool(errors))
    if args.file:
        parser.error("--file requires --extend")
    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(lambda repo: acquire(repo, args.output_dir), args.repo or RECIPES))
    report = {"schema": "ane-source-acquisition-v1", "captured_at": datetime.now(timezone.utc).isoformat(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "repos": records}
    (args.output_dir / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"repos": [{"repo": r["repo"], "status": r["status"], "commit": r.get("commit"),
            "files": len(r["files"]), "errors": r["errors"]} for r in records]}, indent=2))
    return int(any(r["status"] != "acquired" for r in records))


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Explicit, non-overwriting source checkout helper for the research campaign.

Default is a read-only plan. --download performs only reviewed git clone/sparse
checkout operations; it never installs packages or runs a project's scripts.
Existing checkouts are inspected, not fetched, reset, cleaned or modified.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def run(command, timeout=180):
    return subprocess.run(command, text=True, capture_output=True, timeout=timeout)


def checkout(project, destination_root, download=False):
    destination_root = destination_root.resolve()
    folder = project["checkout"]
    if not isinstance(folder, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", folder):
        raise ValueError("unsafe checkout folder")
    repository = project["repository"]
    if not isinstance(repository, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", repository):
        raise ValueError("expected a GitHub owner/repository name")
    sparse_paths = project.get("sparse_paths", [])
    if (not isinstance(sparse_paths, list)
            or any(not isinstance(path, str) or not path or path.startswith(("/", "-"))
                   or ".." in Path(path).parts or Path(path).as_posix() != path for path in sparse_paths)):
        raise ValueError("unsafe sparse checkout paths")
    branch = project.get("branch")
    if branch is not None and (not isinstance(branch, str)
                               or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_./-]*", branch)
                               or ".." in branch or "//" in branch or branch.endswith(("/", ".", ".lock"))):
        raise ValueError("unsafe branch name")
    target = destination_root / folder
    if target.is_symlink() or not target.resolve().is_relative_to(destination_root.resolve()):
        raise ValueError("reference target escapes the explicit root")
    result = {"id": project["id"], "repository": project["repository"], "checkout": folder}
    existed = target.exists()
    if existed:
        if not (target / ".git").exists():
            return {**result, "status": "existing_non_git_preserved"}
    elif not download:
        return {**result, "status": "planned_not_downloaded", "sparse_paths": sparse_paths}
    else:
        destination_root.mkdir(parents=True, exist_ok=True)
        url = "https://github.com/" + repository + ".git"
        command = ["git", "clone", "--depth", "1", "--filter=blob:none", "--no-tags"]
        if sparse_paths:
            command += ["--sparse"]
        if branch:
            command += ["--branch", branch]
        command += [url, str(target)]
        started = time.monotonic()
        completed = run(command)
        if completed.returncode:
            return {**result, "status": "clone_failed", "exit_code": completed.returncode,
                    "stderr": completed.stderr[-3000:], "seconds": time.monotonic() - started}
        if sparse_paths:
            completed = run(["git", "-C", str(target), "sparse-checkout", "set", "--", *sparse_paths])
            if completed.returncode:
                return {**result, "status": "sparse_checkout_failed", "exit_code": completed.returncode,
                        "stderr": completed.stderr[-3000:]}
    # Avoid index refresh writes and configured fsmonitor commands when inspecting.
    inspect = ["git", "--no-optional-locks", "-c", "core.fsmonitor=false", "-C", str(target)]
    head = run([*inspect, "rev-parse", "HEAD"])
    dirty = run([*inspect, "status", "--short", "--untracked-files=no"])
    if head.returncode or dirty.returncode:
        return {**result, "status": "inspection_failed"}
    return {**result, "status": "existing_preserved" if existed else "cloned",
            "head": head.stdout.strip(), "tracked_worktree_dirty": bool(dirty.stdout.strip()),
            "project_code_executed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", action="append", required=True, help="roadmap project ID, repeatable")
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--download", action="store_true", help="explicit network and checkout writes; never executes project code")
    parser.add_argument("--output", type=Path, help="new receipt path, never overwritten")
    args = parser.parse_args()
    if args.output and (args.output.exists() or args.output.is_symlink()):
        parser.error("receipt already exists")
    try:
        inventory = json.loads((ROOT / "candidates/projects.json").read_text(encoding="utf-8"))
        projects = {item["id"]: item for item in inventory["projects"]}
        if len(projects) != len(inventory["projects"]):
            raise ValueError("duplicate inventory project IDs")
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.error("invalid project inventory: " + str(error))
    if len(set(args.project)) != len(args.project) or not set(args.project) <= projects.keys():
        parser.error("use unique known roadmap project IDs")
    results = []
    for identity in args.project:
        print("Inspecting " + identity + " " + projects[identity]["repository"], file=sys.stderr, flush=True)
        try:
            results.append(checkout(projects[identity], args.reference_root.resolve(), args.download))
        except subprocess.TimeoutExpired:
            results.append({"id": identity, "status": "operation_failed", "error": "git operation timed out; any partial checkout is preserved"})
        except (ValueError, KeyError, OSError) as error:
            results.append({"id": identity, "status": "operation_failed", "error": str(error)})
    receipt = {"schema": "turbocider-reference-checkouts-v1", "network_explicitly_enabled": args.download,
               "existing_projects_mutated": False, "project_code_executed": False, "results": results}
    encoded = json.dumps(receipt, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
    print(encoded, end="")
    return int(any(item["status"].endswith("failed") for item in results))


if __name__ == "__main__":
    raise SystemExit(main())

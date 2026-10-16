#!/usr/bin/env python3
"""Check local learning dependencies; optionally execute a techniques-only copy.

Isolation proves the CPU suite and local tooling do not need product/sibling
source. Hardware SDKs and optional framework packages remain declared deps.
This is a dependency/coverage gate, not a semantic or model-quality proof.
"""
import argparse
import ast
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
STAGES = {"encoder", "prefill", "decode", "dit", "vae", "pipeline"}
DESIGNS = {"gpu-dataflow", "owned-graphs", "hybrid-ffn", "packed-streaming", "residency", "composition",
           "parallel-topologies", "consumer-abi", "heterogeneous-planner"}
STDLIB = set("argparse ast collections concurrent contextlib copy dataclasses errno fractions hashlib importlib inspect io itertools json math os pathlib platform plistlib queue random re resource shutil signal statistics struct subprocess sys sysconfig tempfile threading time types typing unittest urllib".split())
OPTIONAL = {"numpy", "coremltools", "mlx"}
DEVICES = ("metal-gemm", "coreml-frozen", "coreml-runtime", "coreml-hybrid", "shared-int8")
ISOLATION_IGNORES = (
    "__pycache__", ".DS_Store", ".venv", ".cache", ".pytest_cache", ".mypy_cache", ".pyright", ".ruff_cache",
    ".aneforge_cache", ".aneforge-cache", ".build", ".swiftpm", "DerivedData", "CMakeFiles",
    "build", "dist", "outputs", "results", "artifacts", "scratch", "private",
    "*.gputrace", "*.gputrace.zip", "*.trace", "*.trace.zip",
    "*.tcbf16z",
)


def audit(root=ROOT):
    root = root.resolve()
    errors = []
    manifest = json.loads((root / "references/absorbed-designs.json").read_text())
    sources = json.loads((root / "references/sources.json").read_text())
    source_ids = {item["id"] for item in sources["sources"]}
    catalog = json.loads((root / "labs/catalog.json").read_text())
    catalog_cases = {case for card in catalog["entries"] for case in card["cases"]}
    for item in manifest["designs"] + manifest["stages"]:
        for relative in [item["chapter"], *item["code"], *item["tests"]]:
            target = (root / relative).resolve()
            if not target.is_relative_to(root) or not target.is_file():
                errors.append("missing/escaping required learning asset: " + relative)
        if not item["code"] or not item["tests"] or not item["cases"]:
            errors.append("incomplete learning item: " + item["id"])
        if not set(item["cases"]) <= catalog_cases:
            errors.append("undiscoverable learning cases: " + item["id"])
        if not set(item.get("source_ids", [])) <= source_ids:
            errors.append("unknown provenance ID: " + item["id"])
    if {item["id"] for item in manifest["stages"]} != STAGES:
        errors.append("required model-stage coverage is incomplete")
    if len(manifest["stages"]) != len(STAGES):
        errors.append("duplicate model-stage item")
    if {item["id"] for item in manifest["designs"]} != DESIGNS or len(manifest["designs"]) != len(DESIGNS):
        errors.append("required absorbed-design coverage is incomplete/duplicated")
    for card in catalog["entries"]:
        for relative in card["code"]:
            target = (root / relative).resolve()
            if not target.is_relative_to(root) or not target.is_file():
                errors.append("catalog requires non-local code: " + relative)
    code_files = [path for folder in ("labs", "scripts", "examples") for path in (root / folder).rglob("*.py")]
    for path in code_files:
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base = path.parent
                    for _ in range(node.level - 1):
                        base = base.parent
                    name = (node.module or "").replace(".", "/")
                    target = (base / (name + ".py")).resolve()
                    if not target.is_relative_to(root) or not target.is_file():
                        errors.append("relative import escapes/misses local code: " + str(path.relative_to(root)))
                    continue
                names = [(node.module or "").split(".")[0]]
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "spec_from_file_location":
                errors.append("dynamic source loading requires explicit audit: " + str(path.relative_to(root)))
            for name in names:
                local = any((base / (name + ".py")).is_file() or (base / name / "__init__.py").is_file()
                            for base in (path.parent, root / "labs"))
                if name not in STDLIB | OPTIONAL and not local:
                    errors.append("undeclared import: " + name + " in " + str(path.relative_to(root)))
    native_files = [path for path in (root / "labs").rglob("*") if path.suffix in (".mm", ".metal", ".h", ".hpp")]
    for path in native_files:
        for include in re.findall(r'^\s*#\s*(?:include|import)\s*"([^"]+)"', path.read_text(), re.M):
            target = (path.parent / include).resolve()
            if not target.is_relative_to(root) or not target.is_file():
                errors.append("native include escapes/misses local code: " + include)
    for path in root.rglob("*"):
        if path.is_symlink() and not path.resolve().is_relative_to(root):
            errors.append("symlink escapes techniques: " + str(path.relative_to(root)))
    return {"designs": len(manifest["designs"]), "model_stages": len(manifest["stages"]),
            "python_files_checked": len(code_files), "native_files_checked": len(native_files),
            "errors": errors, "scope": "local required assets, imports, native includes and coverage; isolation executes the CPU/tool dependency boundary"}


def isolate(root=ROOT, devices=(), overlay=None):
    result = audit(root)
    if result["errors"]:
        return result
    with tempfile.TemporaryDirectory(prefix="tc-techniques-only-") as directory:
        destination = Path(directory) / "techniques"
        shutil.copytree(root, destination, ignore=shutil.ignore_patterns(*ISOLATION_IGNORES))
        # Child CWD has ONLY techniques. Ignore inherited PYTHON* import paths.
        environment = {key: value for key, value in os.environ.items() if not key.startswith("PYTHON")}
        commands = [
            ["-m", "unittest", "discover", "-s", "techniques/tests"],
            ["techniques/scripts/check_wiki.py"],
            ["techniques/labs/check_catalog.py"],
            ["techniques/scripts/check_standalone.py"],
            ["techniques/labs/run.py", "all"],
        ]
        device_commands = {
            "metal-gemm": ["techniques/labs/metal/run.py", "gemm", "--m", "33", "--n", "41", "--k", "29"],
            "coreml-frozen": ["techniques/labs/ane/bench.py", "--kind", "frozen-fp16"],
            "coreml-runtime": ["techniques/labs/ane/bench.py", "--kind", "runtime-fp16"],
            "coreml-hybrid": ["techniques/labs/ane/bench.py", "--compare-gpu"],
            "shared-int8": ["techniques/labs/ane/shared_w8.py"],
        }
        for device in devices:
            command = device_commands[device][:]
            if overlay and device != "metal-gemm":
                command += ["--coreml-overlay", str(Path(overlay).resolve())]
            commands.append(command)
        runs = []
        for args in commands:
            child = subprocess.run([sys.executable, "-E", "-B", *args], cwd=directory, env=environment,
                                   text=True, capture_output=True, timeout=60)
            run = {"command": "python -E -B " + " ".join(args), "exit_code": child.returncode}
            if child.returncode:
                run.update(stdout=child.stdout[-3000:], stderr=child.stderr[-3000:])
                result["errors"].append("isolated command failed: " + run["command"])
            elif args[0] == "-m":
                run["test_summary"] = child.stderr.strip()
            else:
                value = json.loads(child.stdout)
                run["summary"] = {key: val for key, val in value.items() if key != "details"}
                if "experiments" in value.get("details", {}):
                    run["cpu_experiments"] = len(value["details"]["experiments"])
                elif "details" in value:
                    run["device_details"] = value["details"]
            runs.append(run)
        result["isolation"] = {"copied_top_level_entries": ["techniques"], "product_and_sibling_source_present": False,
                               "inherited_python_paths_ignored": True, "runs": runs}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--isolate", action="store_true")
    parser.add_argument("--device", choices=DEVICES, action="append", default=[], help="explicit optional device test in the isolated copy; run serially")
    parser.add_argument("--coreml-overlay", type=Path)
    parser.add_argument("--output", type=Path, help="new report path; never overwrite")
    args = parser.parse_args()
    if args.device and not args.isolate:
        parser.error("device checks require --isolate")
    if args.output and (args.output.exists() or args.output.is_symlink()):
        parser.error("report already exists")
    value = isolate(args.root, args.device, args.coreml_overlay) if args.isolate else audit(args.root)
    encoded = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
    print(encoded, end="")
    return bool(value["errors"])


if __name__ == "__main__":
    raise SystemExit(main())

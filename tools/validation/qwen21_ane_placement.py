"""Offline frozen/runtime graph plans; preferences are not a runtime trace.

The historical filename and frozen --blocks interface remain supported.
Runtime-weight manifests describe one shared graph, not one graph per layer.
No SDK import, model loading or prediction occurs on import or --help.
"""
import argparse
import json
from pathlib import Path
import re

from runtime_ane_common import sha256_file


SCOPE = "Core ML planned placement under CPU_AND_NE, not runtime hardware utilization"


def artifact_path(root, name):
    """Keep the inspection within the supplied manifest's artifact directory."""
    if not isinstance(name, str) or not name:
        raise ValueError("artifact path must be a nonempty relative path")
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != name:
        raise ValueError("artifact path must be canonical and relative")
    path = root
    for part in relative.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError("artifact symlinks are not supported")
    return path


def snapshot(manifest_path, blocks=None):
    """Resolve and bind inputs without importing Core ML; hash again after use.

    Runtime receipts must cover every compiled file. Frozen manifests retain
    their old ABI; bind the selected artifact trees without claiming checkpoint
    compatibility or native loader qualification. This is not an immutable lease.
    """
    if manifest_path.is_symlink():
        raise ValueError("manifest symlinks are not supported")
    root = manifest_path.resolve().parent
    before = sha256_file(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    runtime = manifest.get("backend") == "runtime_weight_fp16"
    if runtime:
        if blocks is not None:
            raise ValueError("runtime manifest has one shared graph; omit --blocks")
        selected = {"runtime": artifact_path(root, manifest.get("compiled_model"))}
    else:
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, dict) or not artifacts:
            raise ValueError("expected a frozen or runtime-weight compiled manifest")
        blocks = blocks if blocks is not None else sorted(map(int, artifacts))
        if not blocks or len(set(blocks)) != len(blocks):
            raise ValueError("select one or more unique blocks")
        selected = {}
        for block in blocks:
            if str(block) not in artifacts:
                raise ValueError(f"manifest has no block {block}")
            selected[str(block)] = artifact_path(root, artifacts[str(block)].get("int8_pc"))
    hashes = {}
    for path in selected.values():
        if path.suffix != ".mlmodelc" or not path.is_dir():
            raise ValueError("placement needs a compiled-cache manifest containing .mlmodelc artifacts")
        found = False
        for entry in sorted(path.rglob("*")):
            name = entry.relative_to(root).as_posix()
            artifact_path(root, name)
            if entry.is_file():
                hashes[name] = sha256_file(entry)
                found = True
            elif not entry.is_dir():
                raise ValueError("compiled artifact contains a nonregular entry")
        if not found:
            raise ValueError("compiled artifact is empty")
    if runtime:
        files = manifest.get("files")
        if not isinstance(files, dict) or not files:
            raise ValueError("runtime manifest is missing its files receipt")
        if any(files.get(name) != digest for name, digest in hashes.items()):
            raise ValueError("runtime compiled files receipt mismatch or incomplete coverage")
        for name, expected in files.items():
            path = artifact_path(root, name)
            if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
                raise ValueError("invalid runtime artifact SHA256")
            if not path.is_file() or sha256_file(path) != expected:
                raise ValueError("runtime artifact SHA256 mismatch")
            hashes[name] = expected
    if sha256_file(manifest_path) != before:
        raise ValueError("manifest changed during inspection")
    return {"backend": "runtime_weight_fp16" if runtime else "frozen",
            "manifest_sha256": before,
            "artifacts": {key: path.relative_to(root).as_posix() for key, path in selected.items()},
            "files": hashes}


def load_plan(path):
    import coremltools as ct
    from coremltools.models.compute_plan import MLComputePlan

    return MLComputePlan.load_from_path(str(path), compute_units=ct.ComputeUnit.CPU_AND_NE)


def operations(block, prefix):
    for index, op in enumerate(block.operations):
        location = f"{prefix}/{index}"
        yield location, op
        for nested, child in enumerate(op.blocks):
            yield from operations(child, f"{location}/block{nested}")


def inspect_plan(plan):
    program = plan.model_structure.program
    if program is None or not program.functions:
        raise ValueError("placement requires a nonempty ML Program")
    placement = []
    for name, function in program.functions.items():
        for location, op in operations(function.block, name):
            usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
            preferred = usage.preferred_compute_device if usage else None
            device = type(preferred).__name__ if preferred is not None else None
            operator = op.operator_name.rsplit(".", 1)[-1]
            category = ("constant" if operator == "const" or
                        operator.startswith("constexpr") else
                        "neural_engine" if device == "MLNeuralEngineComputeDevice" else
                        "unassigned" if device is None else "other")
            placement.append({"operator": op.operator_name, "preferred": device,
                              "category": category, "location": location,
                              "supported": [type(item).__name__ for item in
                                            usage.supported_compute_devices] if usage else []})
    if not placement:
        raise ValueError("placement requires a nonempty ML Program")
    return placement


def inspect_manifest(manifest_path, blocks=None, quiet=False):
    tools = [Path(__file__), Path(sha256_file.__code__.co_filename)]
    tool_hashes = {path.name: sha256_file(path) for path in tools}
    identity = snapshot(manifest_path, blocks)
    runtime = identity["backend"] == "runtime_weight_fp16"
    collection = "graphs" if runtime else "blocks"
    report = {"schema_version": 1, "status": "incomplete", "scope": SCOPE,
              "observed_ane_residency": "unknown", "identity": identity,
              "tool_sha256": tool_hashes, collection: {}}
    counts = {"neural_engine": 0, "other": 0, "unassigned": 0, "constant": 0}
    operator_types = {}
    for key, relative in identity["artifacts"].items():
        placement = inspect_plan(load_plan(manifest_path.resolve().parent / relative))
        report[collection][key] = placement
        for op in placement:
            counts[op["category"]] += 1
            kinds = operator_types.setdefault(op["operator"], dict.fromkeys(counts, 0))
            kinds[op["category"]] += 1
        if not quiet:
            print(json.dumps({"graph" if runtime else "block": key, "placement": placement}), flush=True)
    if snapshot(manifest_path, blocks) != identity:
        raise ValueError("manifest or artifact changed during plan inspection")
    if any(sha256_file(path) != tool_hashes[path.name] for path in tools):
        raise ValueError("inspection tool changed during plan inspection")
    report["summary"] = {"blocks": 0 if runtime else len(identity["artifacts"]),
                         "graphs": len(identity["artifacts"]), "operator_preferences": counts,
                         "operator_types": operator_types}
    report["status"] = "complete"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blocks", type=int, nargs="*",
                        help="frozen manifests only: selected blocks; default: every block")
    parser.add_argument("--quiet", action="store_true", help="print only a summary")
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        parser.error("output must not already exist")
    if not args.output.parent.is_dir():
        parser.error("output parent directory must exist")
    report = inspect_manifest(args.manifest, args.blocks, args.quiet)
    for relative in report["identity"]["artifacts"].values():
        artifact = args.manifest.resolve().parent / relative
        if args.output.resolve().is_relative_to(artifact):
            parser.error("output must be outside compiled artifacts")
    # Exclusive creation also protects against a path appearing during inspection.
    with args.output.open("x") as output:
        output.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"]), flush=True)


if __name__ == "__main__":
    main()

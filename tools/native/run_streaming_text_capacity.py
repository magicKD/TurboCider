#!/usr/bin/env python3
"""Collect real local-model text boundary evidence. No downloads or approval.

This serial runner preserves every command and artifact, checks exact native
token counts, and freezes clean source/build identity before creating a test
catalog. Use the resulting directory as acceptance/text-capacity.
"""
from __future__ import annotations

import argparse
import copy
import ctypes as C
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from build_streaming_catalog import canonical_record_digest, catalog_binding, validate_record_shape
from capture_streaming_source_identity import capture
from generate_runtime_build_identity import seal, source_inputs
from streaming_release_policy import CALIBRATED
from verify_streaming_text_capacity import SCHEMA, ARTIFACTS, required_rows, verify

ROOT = Path(__file__).resolve().parents[2]


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def collect(args):
    output, library, model = args.output.resolve(), args.library.resolve(), args.model.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = capture(ROOT)
    if source["clean"] is not True:
        raise ValueError("commit the complete implementation before collecting range evidence")
    build = json.loads((library.parent / "runtime-build/runtime-build-manifest.json").read_text())
    if build != seal(build["inputs"]) or build["inputs"]["sources"] != source_inputs(ROOT):
        raise ValueError("native build does not match current frozen inputs")
    write(output / "source-identity.json", source)
    write(output / "build-manifest.json", build)
    library_hash = sha(library)
    base = json.loads(args.request.read_text())
    plan = json.loads(args.plan.read_text())
    write(output / "plan.json", plan)
    lib = C.CDLL(str(library))
    lib.tc_string_free.argtypes = [C.c_void_p]
    lib.tc_z_image_tokenize_json.argtypes = [C.c_char_p, C.c_char_p, C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)]

    def consume(pointer):
        if not pointer.value:
            return ""
        result = C.string_at(pointer).decode()
        lib.tc_string_free(pointer)
        return result

    def count(prompt):
        result, error = C.c_void_p(), C.c_void_p()
        status = lib.tc_z_image_tokenize_json(str(model).encode(), prompt.encode(), C.byref(result), C.byref(error))
        raw, failure = consume(result), consume(error)
        if status:
            raise ValueError(failure)
        return json.loads(raw)["valid"]

    def command(name, argv):
        write(output / (name + "-command.json"), argv)
        with (output / (name + ".log")).open("x") as log:
            subprocess.run(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)

    minimum_prompt = min(("!", " ", "\n", "\t"), key=count)
    minimum = count(minimum_prompt)
    capacity = dict(policy_revision="z-image-dynamic-text-capacity-v1", minimum_rows=minimum, maximum_rows=1024)
    lengths = required_rows(capacity)
    # Include the user's ordinary prose prompt as well as structural boundaries.
    prompt = base["inputs"][0]["text"]
    prompt_rows = count(prompt)
    if not minimum <= prompt_rows <= 1024:
        raise ValueError("base prompt must fit the native text range")
    lengths = sorted(set(lengths) | {prompt_rows})
    requests = {}
    for rows in lengths:
        seed_prompt, seed_rows = (prompt, prompt_rows) if rows >= prompt_rows else (minimum_prompt, minimum)
        value = copy.deepcopy(base)
        value["inputs"][0]["text"] = seed_prompt + " snow" * (rows - seed_rows)
        value["parameters"] = {**value.get("parameters", {}), "dynamic_text": True}
        value["execution"]["streaming"] = dict(schema_version=2, enabled=True, selection="memory_tier",
                                                retention="request", target_request_memory_bytes=args.target_gib << 30)
        if count(value["inputs"][0]["text"]) != rows:
            raise ValueError("token boundary prompt construction did not match the native tokenizer")
        requests[rows] = value
        write(output / f"rows-{rows}-public-request.json", value)
    write(output / "protocol.json", dict(schema=SCHEMA, source_commit=source["commit"], rows=lengths,
          capacity=capacity, minimum_prompt=minimum_prompt, library_sha256=library_hash, script_sha256=sha(Path(__file__)),
          interval_ms=20, max_gap_ms=100, target_bytes=args.target_gib << 30, production_authorized=False))
    tool = lambda name: str(ROOT / "tools/native" / name)
    command("catalog", [args.python, tool("build_test_streaming_catalog.py"), "--library", str(library),
        "--model-id", "z-image-turbo", "--model-path", str(model), "--request", str(output / "rows-1024-public-request.json"),
        "--plan", str(output / "plan.json"), "--target-gib", str(args.target_gib), "--catalog-revision", args.catalog_revision,
        "--verify-sources", "--text-minimum-rows", str(minimum), "--output", str(output / "native-test-catalog.json")])
    catalog = json.loads((output / "native-test-catalog.json").read_text())
    record = catalog["records"][0]
    record["release"].update(channel="public-calibrated", policy_revision=CALIBRATED)
    record["performance"]["profile_id"] = "z-image-text-capacity-boundary-v1"
    record["canonical_record_digest"] = canonical_record_digest(record)
    validate_record_shape(record, allow_test_template=True)
    write(output / "test-catalog.json", catalog)
    binding = catalog_binding(record)
    write(output / "binding.json", binding)

    def ref(relative):
        path = output / relative
        return dict(path=relative, bytes=path.stat().st_size, sha256=sha(path))

    cases = []
    for rows in lengths:
        for route in ("public", "manual"):
            request_path = output / f"rows-{rows}-{route}-request.json"
            if route == "manual":
                value = copy.deepcopy(requests[rows])
                value["execution"]["streaming"] = record["plan"]["canonical_config"]
                write(request_path, value)
            argv = [args.python, tool("run_public_streaming_smoke.py" if route == "public" else "run_image_streaming_smoke.py"),
                    "--library", str(library), "--model", str(model), "--request", str(request_path),
                    "--output", str(output / f"rows-{rows}-{route}")]
            if route == "public":
                argv += ["--catalog", str(output / "test-catalog.json")]
                argv = [args.python, tool("collect_streaming_memory.py"), "--output", str(output / f"rows-{rows}-memory.jsonl"),
                        "--summary", str(output / f"rows-{rows}-memory-summary.json"), "--correlation-id", f"text-capacity-{rows}",
                        "--interval-ms", "20", "--max-gap-ms", "100", "--", *argv]
            command(f"rows-{rows}-{route}", argv)
        files = dict(public_plan="public/plan.json", public_result="public/result.json",
                     public_resolution="public/resolve-result.json", public_source="public/source-verification.json",
                     public_observation="public/observation.json", public_image="public/image.png",
                     manual_plan="manual/plan.json", manual_result="manual/result.json",
                     manual_source="manual/source-verification.json", manual_image="manual/image.png")
        case = dict(rows=rows, **{key: ref(f"rows-{rows}-{suffix}") for key, suffix in files.items()})
        case.update(memory=ref(f"rows-{rows}-memory.jsonl"), memory_summary=ref(f"rows-{rows}-memory-summary.json"))
        if set(case) != {"rows", *ARTIFACTS}:
            raise ValueError("incomplete case inventory")
        cases.append(case)
        print(f"Collected actual {rows}-token public/manual pair", flush=True)
    if capture(ROOT) != source or sha(library) != library_hash:
        raise ValueError("source or measured library changed during collection")
    evidence = dict(schema=SCHEMA, binding=binding, reviewed_commit=source["commit"],
                    source_identity=ref("source-identity.json"), build_manifest=ref("build-manifest.json"),
                    library_sha256=library_hash, catalog=ref("test-catalog.json"), target_bytes=args.target_gib << 30, cases=cases)
    write(output / "range.json", evidence)
    result = verify(output, binding, source["commit"])
    write(output / "verification.json", result)
    print("PASS independently verified text range; no production authorization", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("library", "model", "request", "plan", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--catalog-revision", required=True)
    parser.add_argument("--target-gib", type=int, choices=(8, 10, 12, 16, 20), default=16)
    parser.add_argument("--python", default=sys.executable)
    collect(parser.parse_args())

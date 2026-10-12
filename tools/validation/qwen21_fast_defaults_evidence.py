#!/usr/bin/env python3
"""Audit completed explicit/unflagged screens; no model loads or model hashing."""
import argparse
import json
from pathlib import Path

from qwen21_compare_sessions import tensor_identity
from qwen21_gguf_hybrid_screen import validate_receipts, validate_verified_imports
from runtime_ane_common import sha256_file

ROOT = Path(__file__).resolve().parents[2]


def relative(path):
    return str(path.resolve().relative_to(ROOT))


def audit_pair(explicit, unflagged, steps):
    records = []
    for directory, defaults in ((explicit, False), (unflagged, True)):
        summary = json.loads((directory / "summary.json").read_text())
        if (summary["status"] != "complete_diagnostic" or summary["steps"] != steps or
                summary["fast_defaults"] is not defaults or summary["source_registration"] != "reuse" or
                summary["dump_tensors"] is not (steps == 4)):
            raise ValueError("incomplete or mismatched defaults screen")
        trials = {trial["mode"]: trial for trial in summary["trials"]}
        if set(trials) != {"gpu", "a5120"}:
            raise ValueError("actual GPU and channel-5120 trials required")
        expected = {name: (digest, (ROOT / path).stat().st_size) for name, path, digest in (
            ("encoder", "models/unsloth-qwen21-q4-k-m/text_encoders/Qwen3-VL-8B-Instruct-Q4_K_M.gguf",
             "108e7ff92b78eefd3db4741885104acba514255c11b617d3c7b197a5f46efe89"),
            ("dit", "models/unsloth-qwen21-q4-k-m/diffusion_models/qwen-image-2.1-Q4_K_M.gguf",
             "631d532e7ca71e8d90a87c71d3699761a812039d22e3370e87498d87754660fe"))}
        clean = []
        for mode, trial in trials.items():
            rows = [json.loads(line) for line in (directory / f"{mode}.stdout.jsonl").read_text().splitlines()]
            validate_receipts(rows, trial["channels"], steps, encoder_prefill=True)
            validate_verified_imports(trial["verified_imports"], expected, "reuse", 4, True, 8)
            if any(item.get("fast_defaults_enabled") is not defaults for item in trial["verified_imports"]):
                raise ValueError("actual import-default policy differs")
            if defaults and trial["optimization_overrides"]:
                raise ValueError("unflagged screen injected optimization overrides")
            if not defaults and trial["optimization_overrides"].get("TURBOCIDER_QWEN21_GGUF_FAST_DEFAULTS") != "0":
                raise ValueError("explicit control did not disable automatic defaults")
            memory = trial["memory"]
            if (memory["verified"]["complete"] is not True or memory["verified"]["command_exit_code"] != 0 or
                    sha256_file(directory / memory["evidence"]) != memory["evidence_sha256"] or
                    sha256_file(directory / memory["report"]) != memory["report_sha256"]):
                raise ValueError("incomplete or changed process-tree memory evidence")
            clean.append(dict(mode=mode, channels=trial["channels"], timings=trial["timings"],
                              warm_medians=trial["warm_medians"], verified_imports=trial["verified_imports"],
                              optimization_overrides=trial["optimization_overrides"],
                              memory={key: memory["verified"][key] for key in (
                                  "complete", "tree_peak_phys_footprint_bytes", "compression_bytes",
                                  "decompression_bytes", "swap_in_bytes", "swap_out_bytes")},
                              png_sha256=trial["png_sha256"]))
        records.append(dict(defaults=defaults, evidence=relative(directory / "summary.json"),
                            evidence_sha256=sha256_file(directory / "summary.json"), trials=clean,
                            source_sha256={Path(name).name: digest for name, digest in summary["source_identities"].items()}))
    if records[0]["source_sha256"] != records[1]["source_sha256"]:
        raise ValueError("explicit and default screens used different sources/runtimes")
    comparisons = []
    for mode in ("gpu", "a5120"):
        for index in range(3):
            a = json.loads((explicit / f"{mode}-{index}.json").read_text())
            b = json.loads((unflagged / f"{mode}-{index}.json").read_text())
            for request in (a, b):
                request.pop("output"); request.pop("dump_tensors", None)
            if a != b:
                raise ValueError("effective requests differ")
            png = f"{mode}-{index}.png"
            if sha256_file(explicit / png) != sha256_file(unflagged / png):
                raise ValueError("default profile changed complete PNG")
            tensors = []
            if steps == 4:
                for name in ("text", "initial", "latents", "pixels"):
                    path = Path(f"{mode}-{index}-dump/qwen21_{name}.safetensors")
                    reference, candidate = tensor_identity(explicit / path), tensor_identity(unflagged / path)
                    if reference != candidate:
                        raise ValueError("default profile changed complete diagnostic tensor")
                    tensors.append(dict(name=name, dtype=reference[0], shape=reference[1],
                                        payload_sha256=reference[2], exact=True))
            comparisons.append(dict(mode=mode, request_index=index, png_exact=True, tensors=tensors))
    return dict(steps=steps, screens=records, comparisons=comparisons)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--explicit-4", type=Path, required=True)
    parser.add_argument("--unflagged-4", type=Path, required=True)
    parser.add_argument("--explicit-40", type=Path)
    parser.add_argument("--unflagged-40", type=Path)
    parser.add_argument("--build-manifest", type=Path, required=True)
    parser.add_argument("--session-log", type=Path, help="completed real prepare/cache/default-off/on lifecycle test")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or bool(args.explicit_40) != bool(args.unflagged_40):
        parser.error("fresh evidence output and complete optional 40-step pair required")
    build = json.loads(args.build_manifest.read_text())
    sources = build["inputs"]["sources"]
    for source in sources:
        if sha256_file(ROOT / source["path"]) != source["sha256"]:
            raise ValueError("native build inputs are stale: " + source["path"])
    report = dict(schema="tc-qwen21-fast-defaults-validation-v1", status="complete_diagnostic",
                  date="2026-10-11", qualification_passed=False, native_inputs_matched=len(sources),
                  build_identity=build.get("catalog_runtime_id"),
                  scope="unflagged measured fast profile; no automatic Private ANE opt-in or broad quality/memory qualification",
                  pairs=[audit_pair(args.explicit_4, args.unflagged_4, 4)])
    if args.explicit_40:
        report["pairs"].append(audit_pair(args.explicit_40, args.unflagged_40, 40))
    if args.session_log:
        logs = args.session_log.read_text()
        sessions = []
        for line in logs.splitlines():
            if line.startswith('{"schema": "tc-qwen21-encoder-prefill-session-v1"'):
                # unittest may append its progress dot to the last JSON line.
                sessions.append(json.JSONDecoder().raw_decode(line)[0])
        if len(sessions) != 1 or "\nOK\n" not in logs:
            raise ValueError("complete successful lifecycle test log required")
        session = sessions[0]
        lifecycle = {row["name"]: row for row in session["requests"]}
        cached, disabled, restored = (lifecycle[name] for name in
                                     ("unflagged-cache", "master-off-real-legacy", "unflagged-restored"))
        if (cached["prompt_cache_hit"] is not True or cached["weights"]["weights_retained"] is not True or
                disabled["prompt_cache_hit"] is not False or disabled["weights"]["enabled"] is not False or
                disabled["weights"]["retained_bytes"] != 0 or
                restored["prompt_cache_hit"] is not False or restored["weights"]["loads_session_total"] != 3 or
                restored["weights"]["weights_retained"] is not True or
                lifecycle["changed-processor-rejected"]["rejected"] is not True):
            raise ValueError("default/off/restored or changed-source lifecycle differs")
        report["session_lifecycle"] = dict(evidence=relative(args.session_log),
                                           evidence_sha256=sha256_file(args.session_log), record=session)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as target:
        json.dump(report, target, indent=2, allow_nan=False); target.write("\n")
    print(json.dumps(dict(status=report["status"], native_inputs_matched=len(sources),
                          steps=[pair["steps"] for pair in report["pairs"]])))


if __name__ == "__main__":
    main()

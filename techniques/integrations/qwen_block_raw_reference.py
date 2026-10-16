#!/usr/bin/env python3
"""Reexecute frozen FP64 sample math from retained BF16 input/reference files.

Explicit CPU-only NumPy integration; no MLX/ANE/candidate imports, downloads,
model execution or private-output replay. Each sample uses full original K/F.
Raw vectors and source paths stay local; only identities, metrics and counts emit.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "labs"))
sys.path.insert(0, str(ROOT / "labs/ane"))
sys.path.insert(0, str(ROOT / "labs/metal"))
from block_success import raw_bf16
from block_context import metric, require
from common import metrics
from qwen_witness import selected
from real_bf16_source import tensor


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1 << 20), b""):
            result.update(data)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bank-run", type=Path, required=True)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source_files = [Path(__file__).resolve(), ROOT / "labs/block_success.py", ROOT / "labs/block_context.py",
                    ROOT / "labs/common.py", ROOT / "labs/ane/qwen_witness.py", ROOT / "labs/metal/real_bf16_source.py"]
    report = {"schema": "tc-qwen-block-raw-fp64-v1", "status": "starting", "runs": [], "errors": [],
              "production_qualified": False, "whole_request_qualified": False, "actual_device_executed": False,
              "private_outputs_reexecuted": False, "full_block_outputs_reexecuted": False,
              "oracle_scope": "four actual rows x four output columns; full original K=4096 and F=12288; raw GPU FFN reference only",
              "python": platform.python_version(), "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in source_files}}

    def save():
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    save()
    try:
        require(shutil.disk_usage(output).free >= 129 * 1024**2, "128MiB reserve plus 1MiB metadata required; no cleanup")
        import numpy as np

        report["numpy"] = np.__version__
        manifest_path = args.bank_run.resolve() / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        report["bank_manifest_sha256"] = digest(manifest_path)
        weights = [v["sha256"] for v in manifest["weights"]]
        require(len(weights) == 2, "explicit paired original BF16 weights required")
        report["original_ffn_weight_sha256"] = weights
        checkpoint = args.checkpoint.resolve()
        before = checkpoint.stat()
        gu, _ = tensor(checkpoint, "transformer_blocks.0.img_mlp.gate_up.weight", (24576,4096), weights[0], np)
        down, _ = tensor(checkpoint, "transformer_blocks.0.img_mlp.out.weight", (4096,12288), weights[1], np)
        report["original_source_shape_verified"] = True
        require(len({p.resolve() for p in args.run}) == len(args.run), "unique explicit raw runs required")
        for path in args.run:
            path = path.resolve()
            parent_path = path / "report.json"
            parent = json.loads(parent_path.read_text())
            require(parent.get("status") == "measured" and parent.get("native_returncode") == 0
                    and parent.get("bank_manifest_sha256") == report["bank_manifest_sha256"], "complete matching raw source run required")
            native = parent["native"]
            raw_input, input_bytes = raw_bf16(path / "ffn-input.safetensors", "input")
            raw_reference, reference_bytes = raw_bf16(path / "ffn-reference.safetensors", "reference")
            require(raw_input["payload_sha256"] == native["exposed_ffn_input_sha256"]
                    and raw_reference["payload_sha256"] == native["full_ffn_reference_sha256"], "persisted payload differs from actual context")
            x = (np.frombuffer(input_bytes, dtype="<u2").astype(np.uint32) << 16).view(np.float32).reshape(1024,4096)
            reference = (np.frombuffer(reference_bytes, dtype="<u2").astype(np.uint32) << 16).view(np.float32).reshape(1024,4096)
            rows, cols, values, domain = selected(x, gu, down, np)
            observed = reference[np.ix_(rows, cols)]
            gate = metrics(values.tolist(), observed.tolist())
            metric(gate, .03)
            old = native["independent_full_reference_gate"]
            require(all(abs(gate[key] - old[key]) <= 1e-8 + 1e-10 * abs(old[key]) for key in ("max_abs", "relative_l2", "cosine")),
                    "independent FP64 reexecution disagrees with native reference prerequisite")
            report["runs"].append({"id": path.name, "case_id": parent["case_id"], "seed": parent["seed"],
                "prompt_sha256": parent["prompt_sha256"], "raw_parent_report_sha256": digest(parent_path),
                "input": raw_input, "reference": raw_reference, "rows": rows, "cols": cols,
                "independent_fp64_values_sha256": hashlib.sha256(np.asarray(values, dtype="<f8").tobytes()).hexdigest(),
                "reference_gate": gate, "sample_gate_domain": domain, "native_reference_metric_consistent": True,
                "candidate_private_outputs_reexecuted": False, "whole_block_reexecuted": False})
            save()
        after = checkpoint.stat()
        report["checkpoint_generation_unchanged"] = all(getattr(before, k) == getattr(after, k) for k in ("st_size", "st_mtime_ns", "st_ctime_ns", "st_dev", "st_ino"))
        report["source_inputs_unchanged"] = all(digest(p) == report["source_sha256"][str(p.relative_to(ROOT))] for p in source_files)
        require(report["checkpoint_generation_unchanged"] and report["source_inputs_unchanged"], "CPU verification source changed")
        report["status"] = "independent-reference-samples-reexecuted"
    except (Exception, KeyboardInterrupt) as error:
        report["status"] = "failed"
        report["errors"].append({"type": type(error).__name__, "message": str(error)})
    save()
    print(json.dumps({"status": report["status"], "sample_cases": len(report["runs"]), "actual_device_executed": False, "errors": report["errors"]}))
    return 0 if report["status"] == "independent-reference-samples-reexecuted" else 1


if __name__ == "__main__":
    raise SystemExit(main())

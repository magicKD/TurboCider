"""Compare every dumped denoising step, final latent, decoded tensor and PNG.

Strict equality is the default gate for these non-approximate fusion kernels.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from safetensors.numpy import load_file


def compare(a, b):
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    a, b = a.astype(np.float64), b.astype(np.float64)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("nonfinite validation output")
    difference = a - b
    return {"shape": list(a.shape), "equal": bool(np.array_equal(a, b)),
            "max_abs": float(np.abs(difference).max()),
            "rmse": float(np.sqrt(np.mean(difference**2))),
            "relative_l2": float(np.linalg.norm(difference.ravel()) /
                                 max(np.linalg.norm(a.ravel()), 1e-30))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("baseline", type=Path)
    p.add_argument("candidate", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    reports = [json.loads((d / "report.json").read_text()) for d in (a.baseline, a.candidate)]
    quality = [[x for x in r["runs"] if x["parity"]] for r in reports]
    if any(len(r) != 1 for r in quality):
        p.error("each report must contain exactly one parity request")
    requests = [r[0]["request"] for r in quality]
    comparable = [{k: v for k, v in r.items() if k not in ("output", "dump_tensors")} for r in requests]
    if comparable[0] != comparable[1]:
        p.error("parity requests differ")
    directories = [Path(r["dump_tensors"]) for r in requests]
    names = {"z_latent_initial.safetensors", "z_latent_final.safetensors", "z_decoded.safetensors"}
    names |= {f"z_latent_step_{i+1}.safetensors" for i in range(requests[0]["steps"])}
    results = {}
    for name in sorted(names):
        values = [load_file(str(d / name)) for d in directories]
        if values[0].keys() != values[1].keys():
            raise ValueError(f"tensor keys differ: {name}")
        for key in values[0]:
            results[f"{name}:{key}"] = compare(values[0][key], values[1][key])
    results["png"] = compare(*(np.asarray(Image.open(r["output"]).convert("RGB")) for r in requests))
    passed = all(r["equal"] for r in results.values())
    report = {"passed": passed, "gate": "exact equality of all steps and decoded image", "results": results}
    a.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

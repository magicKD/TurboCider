"""Compare native outputs before/after removing the Python-session switch.

Runs explicit native CLI binaries. The candidate receives invalid legacy
Python settings, which must not affect the native inference route.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    legacy = ["TURBOCIDER_LLADA_REFERENCE", "TURBOCIDER_LLADA_PYTHON",
              "TURBOCIDER_LLADA_WORKER", "TURBOCIDER_LLADA_SOURCE"]
    results = {}
    for label, binary in [("baseline", args.baseline), ("candidate", args.candidate)]:
        output = (args.output / (label + ".png")).resolve()
        request = {"model": "llada-image-turbo", "operation": "image.generate",
                   "prompt": "A red fox in fresh snow, cinematic photograph",
                   "output": str(output), "width": 256, "height": 256,
                   "frames": 1, "steps": 4, "seed": 42, "audio": False,
                   "execution": "gpu"}
        path = args.output / (label + ".json")
        path.write_text(json.dumps(request) + "\n")
        env = os.environ.copy()
        for name in legacy:
            env.pop(name, None)
        if label == "candidate":
            env.update({name: "/nonexistent/turbocider-boundary-test" for name in legacy})
            env["TURBOCIDER_LLADA_REFERENCE"] = "1"
        process = subprocess.run([str(binary.resolve(strict=True)), "generate",
                                  str(args.model.resolve(strict=True)), str(path.resolve())],
                                 cwd=args.output, env=env, text=True, capture_output=True)
        (args.output / (label + ".log")).write_text(process.stdout + "\n" + process.stderr)
        if process.returncode:
            raise RuntimeError(f"{label} failed: {process.stderr[-4000:]}")
        result = json.loads(process.stdout)
        if not result.get("runtime_backend", "").startswith("mlx_cpp"):
            raise AssertionError(f"not a native backend: {result.get('runtime_backend')}")
        results[label] = {"png_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                          "seconds": result["timings_seconds"]["request_wall"],
                          "runtime_backend": result["runtime_backend"]}
    results["exact_png"] = results["baseline"]["png_sha256"] == results["candidate"]["png_sha256"]
    (args.output / "report.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    if not results["exact_png"]:
        raise SystemExit("native boundary change altered output")


if __name__ == "__main__":
    main()

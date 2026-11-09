#!/usr/bin/env python3
"""Compare full resident VAE decoding with baseline and probe-only padding.

Uses a real saved final latent and matching decode noise; A/B/B/A process order,
one warmup and repeated GPU decodes per process. Excludes DiT and MP4 export.
"""

import argparse
import json
from pathlib import Path
import statistics
import subprocess

from ltx098_comfy_reference import MODEL_SHA256, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "trace", "baseline", "candidate", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    checkpoint, trace, baseline, candidate = (p.resolve(strict=True) for p in
                                                 (args.checkpoint, args.trace, args.baseline, args.candidate))
    output = args.output.resolve()
    if output.exists() or baseline == candidate or not 1 <= args.warmups <= 5 or not 2 <= args.repeats <= 10:
        parser.error("new output directory, distinct binaries, 1–5 warmups and 2–10 repeats required")
    latent, noise = trace / "step_7.safetensors", trace / "decode_noise.safetensors"
    if sha256(checkpoint) != MODEL_SHA256:
        raise ValueError("checkpoint SHA-256 differs from pinned source")
    identities = {"checkpoint_sha256": MODEL_SHA256, "latent_sha256": sha256(latent), "noise_sha256": sha256(noise),
                  "baseline_probe_sha256": sha256(baseline), "candidate_probe_sha256": sha256(candidate)}
    output.mkdir(parents=True)
    report = {"schema": "ltx098-vae-decode-abba-v1", "status": "running", **identities,
              "scope": "Actual trace latent and noise; resident full VAE decode only; warmup then timed repeats; "
                       "A/B/B/A fresh processes; excludes checkpoint load, DiT, MP4 export and process startup",
              "warmups": args.warmups, "repeats": args.repeats, "runs": []}
    try:
        for label, executable in (("a0", baseline), ("b0", candidate), ("b1", candidate), ("a1", baseline)):
            result_path = output / f"{label}.safetensors"
            command = [str(executable), str(checkpoint), str(trace), str(result_path), str(args.warmups), str(args.repeats)]
            run = subprocess.run(command, text=True, capture_output=True, timeout=1200)
            (output / f"{label}.stdout.txt").write_text(run.stdout)
            (output / f"{label}.stderr.txt").write_text(run.stderr)
            if run.returncode:
                raise RuntimeError(f"{label} exited {run.returncode}: {run.stderr[-4000:]}")
            measurement = json.loads(run.stdout)
            if ((measurement.get("frames"), measurement.get("height"), measurement.get("width")) != (121, 512, 512) or
                    len(measurement.get("seconds", [])) != args.repeats or any(v <= 0 for v in measurement["seconds"])):
                raise ValueError("invalid full-video decoder measurement")
            row = {"label": label, "mode": "baseline" if executable == baseline else "candidate", **measurement,
                   "pixels_safetensors_sha256": sha256(result_path)}
            report["runs"].append(row)
            (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            print(json.dumps(row), flush=True)
        if (sha256(checkpoint) != identities["checkpoint_sha256"] or sha256(latent) != identities["latent_sha256"] or
                sha256(noise) != identities["noise_sha256"] or sha256(baseline) != identities["baseline_probe_sha256"] or
                sha256(candidate) != identities["candidate_probe_sha256"]):
            raise ValueError("source or binary identity changed during measurement")
        exact = len({row["pixels_safetensors_sha256"] for row in report["runs"]}) == 1
        if not exact:
            raise ValueError("full decoded BF16 tensor files differ")
        times = {mode: [sample for row in report["runs"] if row["mode"] == mode for sample in row["seconds"]]
                 for mode in ("baseline", "candidate")}
        report.update(status="complete", identical_pixel_file_hashes=True,
                      baseline_median_seconds=statistics.median(times["baseline"]),
                      candidate_median_seconds=statistics.median(times["candidate"]),
                      speedup=statistics.median(times["baseline"]) / statistics.median(times["candidate"]))
    except BaseException as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in ("identical_pixel_file_hashes", "baseline_median_seconds", "candidate_median_seconds", "speedup")}, indent=2))


if __name__ == "__main__":
    main()

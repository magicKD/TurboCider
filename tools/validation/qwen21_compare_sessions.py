"""CPU-only warm-Session timing/PNG comparison; does not establish ANE placement."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import struct

def tensor_identity(path):
    """Exact tensor identity without importing MLX or submitting GPU work."""
    data = path.read_bytes()
    if len(data) < 8:
        raise ValueError(f"truncated safetensors: {path}")
    header_length = struct.unpack("<Q", data[:8])[0]
    if header_length > len(data) - 8:
        raise ValueError(f"invalid safetensors header: {path}")
    header = json.loads(data[8:8 + header_length])
    tensor = header["tensor"]
    begin, end = tensor["data_offsets"]
    payload = data[8 + header_length:]
    if not 0 <= begin < end <= len(payload):
        raise ValueError(f"invalid tensor offsets: {path}")
    return tensor["dtype"], tensor["shape"], hashlib.sha256(payload[begin:end]).hexdigest()


def main():
    import numpy as np
    from PIL import Image

    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-dumps", action="store_true",
                        help="Require and compare dumped text/noise tensors")
    args = parser.parse_args()
    names = sorted(p.name for p in args.baseline.glob("run-*.json"))
    assert names and names == sorted(p.name for p in args.candidate.glob("run-*.json")), "unmatched runs"
    comparisons = []
    for name in names:
        a, b = (json.loads((root / name).read_text()) for root in (args.baseline, args.candidate))
        keys = ("model", "operation", "width", "height", "steps", "seed", "text_tokens")
        assert all(a[key] == b[key] for key in keys), "request metadata differs"
        assert a["prompt_cache_hit"] and b["prompt_cache_hit"], "not a warm prompt-cached comparison"
        assert a["plan"]["execution"] == "gpu" and b["plan"]["execution"] == "gpu_ane_experimental"
        assert a["actual_denoise_steps"] == b["actual_denoise_steps"] == a["steps"]
        pixels = [np.asarray(Image.open(root / name.replace(".json", ".png")).convert("RGBA"),
                             dtype=np.float64) / 255 for root in (args.baseline, args.candidate)]
        x, y = (p[..., :3] for p in pixels)
        assert x.shape == y.shape
        dump_equal = None
        if args.verify_dumps:
            dump_names = ("qwen21_text.safetensors", "qwen21_initial.safetensors")
            tensors = []
            for root in (args.baseline, args.candidate):
                dump_dir = root / name.replace(".json", "-dump")
                assert dump_dir.is_dir(), f"missing dump directory: {dump_dir}"
                tensors.append([tensor_identity(dump_dir / dump_name) for dump_name in dump_names])
            dump_equal = tensors[0] == tensors[1]
            assert dump_equal, f"request tensors differ for {name}"
        ta, tb = (r["timings_seconds"] for r in (a, b))
        comparisons.append(dict(run=name, baseline_wall=ta["request_wall"], candidate_wall=tb["request_wall"],
                                wall_speedup=ta["request_wall"] / tb["request_wall"],
                                denoise_speedup=ta["denoise"] / tb["denoise"],
                                rgb_rmse=float(np.sqrt(np.mean((x-y)**2))),
                                rgb_correlation=float(np.corrcoef(x.ravel(), y.ravel())[0, 1]),
                                alpha_rmse=float(np.sqrt(np.mean((pixels[0][..., 3]-pixels[1][..., 3])**2))),
                                input_tensors_equal=dump_equal) )
    report = dict(scope="prepared, prompt-cached Session request wall time including decode/export and any enabled tensor-dump I/O; not cold-start or hardware-placement proof",
                  input_validation="metadata matched; optional dumped text/noise tensor equality checked" if args.verify_dumps else
                                   "metadata matched; prompt/noise tensor equality not checked by this script",
                  repeats=len(comparisons), runs=comparisons,
                  median_wall_speedup=statistics.median(r["baseline_wall"] for r in comparisons) /
                                      statistics.median(r["candidate_wall"] for r in comparisons))
    print(json.dumps(report, indent=2))
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

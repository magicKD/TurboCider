"""Real official-weight PyTorch / native Metal tiny VAE parity (dev only)."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-root", type=Path, required=True)
    p.add_argument("--models", type=Path, required=True)
    p.add_argument("--probe", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    import torch
    from safetensors.numpy import save_file, load_file
    from safetensors import safe_open
    torch.set_num_threads(4)
    spec = importlib.util.spec_from_file_location("official_taesd", a.reference_root / "taesd.py")
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    a.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    reports = []
    for model, arch, channels, scale, image_channels in [
            ("z-image-turbo", "taef1", 16, 8, 3), ("qwen-image-2.1", "taeqi2_1", 64, 16, 4)]:
        official = reference.TAESD(encoder_path=a.reference_root / f"{arch}_encoder.pth",
                                  decoder_path=a.reference_root / f"{arch}_decoder.pth").eval()
        cases = [("decode", (1, channels, 4, 6)), ("decode", (1, channels, 8, 8)),
                 ("encode", (1, image_channels, 4 * scale, 6 * scale)),
                 ("encode", (1, 3, 8 * scale, 8 * scale)),
                 ("decode", (1, channels, 512 // scale, 512 // scale)),
                 ("encode", (1, image_channels, 1024 if scale == 16 else 512, 1024 if scale == 16 else 512))]
        for index, (mode, shape) in enumerate(cases):
            x = (rng.random(shape) if mode == "encode" else rng.standard_normal(shape)).astype(np.float32)
            image = torch.from_numpy(x)
            if mode == "encode" and image_channels == 4 and shape[1] == 3:
                image = torch.cat((image, torch.ones((1, 1, *shape[2:]))), dim=1)
            with torch.no_grad():
                expected = getattr(official, "encoder" if mode == "encode" else "decoder")(image).numpy()
            inp, out = [a.output / f"{arch}-{index}-{suffix}.safetensors" for suffix in ("input", "output")]
            save_file({"input": x}, str(inp))
            subprocess.run([str(a.probe.resolve()), model, str((a.models / f"{arch}.safetensors").resolve()),
                            str(inp.resolve()), str(out.resolve()), mode], check=True)
            actual = load_file(str(out))["output"]
            error = float(np.max(np.abs(expected - actual)))
            relative = float(np.linalg.norm(expected - actual) / max(np.linalg.norm(expected), 1e-12))
            # FP32 convolution reductions differ between CPU (PyTorch) and
            # Metal, especially on full-resolution maps. Bound both global
            # relative error and worst pixel error: 1/2048 is <1/8 of an 8-bit
            # pixel level, not a visual/approximation-quality allowance.
            ok = actual.shape == expected.shape and np.isfinite(actual).all() and error < 1 / 2048 and relative < 2e-5
            reports.append(dict(model=model, mode=mode, input_shape=shape, output_shape=actual.shape,
                                max_abs=error, relative_l2=relative, passed=bool(ok)))
            if not ok:
                raise AssertionError(reports[-1])
        checkpoint = a.models / f"{arch}.safetensors"
        values = load_file(str(checkpoint))
        with safe_open(str(checkpoint), framework="pt") as f:
            metadata = f.metadata() or {}
        for name in ("missing", "dtype", "architecture", "nonfinite"):
            changed, meta = dict(values), dict(metadata)
            key = "decoder.1.weight" if "decoder.1.weight" in changed else "decoder.layers.0.weight"
            if name == "missing":
                changed.pop(key)
            elif name == "dtype":
                changed[key] = changed[key].astype(np.float16)
            elif name == "architecture":
                meta["architecture"] = "taesd"
            else:
                changed[key] = changed[key].copy()
                changed[key].flat[0] = np.nan
            malformed = a.output / f"{arch}-bad-{name}.safetensors"
            save_file(changed, str(malformed), metadata=meta)
            # The last encoder image is not a decoder latent; rejection must
            # occur in coefficient validation, before input-shape dispatch.
            result = subprocess.run([str(a.probe.resolve()), model, str(malformed.resolve()),
                                     str(inp.resolve()), str(out.resolve()), "decode"], capture_output=True, text=True)
            reason = result.stderr
            rejected = result.returncode != 0 and ("coefficient" in reason or "architecture/format" in reason or "nonfinite tiny VAE weights" in reason)
            reports.append(dict(model=model, malformed=name, passed=rejected))
            if not rejected:
                raise AssertionError((name, result.returncode, reason))
    report = {"tests": reports, "max_abs_limit": 1 / 2048, "relative_l2_limit": 2e-5,
              "passed": all(x["passed"] for x in reports)}
    (a.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

"""Offline-only official TAESD weight conversion; never imported by the App.

Download the four pinned .pth files from madebyollin/taesd at REVISION, then:
  python tools/convert/tiny_vae.py --source DIR --architecture taef1 --output OUT
Use taeqi2_1 for Qwen Image 2.1. Output contains both encoder and decoder.
"""
import argparse
import hashlib
import io
import os
from pathlib import Path
import tempfile

REVISION = "fdbae2df989db95761f482efb8977e1182fb68d2"
HASHES = {
    "taef1_encoder": "1d1ad34546aea0edd99dc2db9411d870319f428ced5065c6464582c17c070536",
    "taef1_decoder": "beae86f2eeaf0cea884dc8ffe639fb297dac8c984441bb53420f07d767785104",
    "taeqi2_1_encoder": "48d6812fa94711807075e882e5a8789eaeb24feb52a93398ff9729718ffac6bd",
    "taeqi2_1_decoder": "53c2212f4ea6c8ad06a30d4969d891f80339ff87bb3f77fe793848613d3a9042",
}


def convert(source: Path, architecture: str, output: Path):
    import torch
    from safetensors.torch import save_file
    if architecture not in ("taef1", "taeqi2_1"):
        raise ValueError("unsupported architecture")
    if output.exists():
        raise FileExistsError(output)
    weights = {}
    metadata = {"format": "turbocider-tiny-v1", "architecture": architecture,
                "latent_space": "diffusion_normalized", "source_revision": REVISION}
    for component in ("encoder", "decoder"):
        name = f"{architecture}_{component}"
        data = (source / f"{name}.pth").read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != HASHES[name]:
            raise ValueError(f"not the pinned official checkpoint: {name}")
        values = torch.load(io.BytesIO(data), weights_only=True, map_location="cpu")
        for key, value in values.items():
            value = value.float().contiguous()
            if not torch.isfinite(value).all().item():
                raise ValueError(f"nonfinite {name}:{key}")
            weights[f"{component}.{key}"] = value
        metadata[f"{component}_sha256"] = digest
    output = output.absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".tiny-vae-", dir=output.parent) as tmp:
        path = Path(tmp) / "model.safetensors"
        save_file(weights, str(path), metadata=metadata)
        os.link(path, output)  # no clobber, including concurrent converters
    return output


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--architecture", choices=("taef1", "taeqi2_1"), required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(convert(a.source, a.architecture, a.output))

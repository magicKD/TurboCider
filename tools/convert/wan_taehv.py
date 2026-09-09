"""Offline-only conversion of the qualified Wan2.1 TAEHV decoder.

Requires development torch and safetensors. The emitted file is read directly
by the native App; neither Python nor the upstream repository is needed there.
"""
import argparse
import hashlib
import os
from pathlib import Path
import tempfile

SOURCE_SHA256 = "d26151e76cdc2c9424bef988de874b33d9a53f30ef3060cd556c429c469c797e"


def convert(source: Path, destination: Path):
    source = source.resolve(strict=True)
    destination = destination.absolute()
    if destination.exists():
        raise FileExistsError(f"refusing to replace existing decoder: {destination}")
    # Read once so verification and deserialization refer to identical bytes.
    data = source.read_bytes()
    if hashlib.sha256(data).hexdigest() != SOURCE_SHA256:
        raise ValueError("not the qualified taew2_1.pth checkpoint")
    import io
    import torch
    from safetensors.torch import save_file
    state = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    weights = {key: value.float().contiguous() for key, value in state.items()
               if key.startswith("decoder.")}
    if not weights or not all(torch.isfinite(value).all().item() for value in weights.values()):
        raise ValueError("missing or nonfinite TAEHV decoder weights")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".wan-taehv-", dir=destination.parent) as raw:
        temporary = Path(raw) / "decoder.safetensors"
        save_file(weights, str(temporary), metadata={
            "architecture": "taew2_1", "source_sha256": SOURCE_SHA256})
        # Atomic publication without overwriting a concurrently created file.
        os.link(temporary, destination)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="qualified taew2_1.pth")
    parser.add_argument("output", type=Path, help="MODEL/vae/taew2_1.safetensors")
    args = parser.parse_args()
    print(convert(args.source, args.output))


if __name__ == "__main__":
    main()

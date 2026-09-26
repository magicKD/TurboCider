"""Real Comfy VAE checkpoint parity; mflux raw decoder retains alpha here.

The reference public decode drops alpha, so compare its RGB slice separately
and explicitly call its decoder for all four raw channels. This is tensor
parity, not proof of useful transparency or edited-image quality.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile


def canonical(key):
    # Comfy's original Wan-style names -> diffusers names expected by mflux.
    key = re.sub(r"^conv1\.", "quant_conv.", key)
    key = re.sub(r"^conv2\.", "post_quant_conv.", key)
    key = re.sub(r"^(encoder|decoder)\.conv1\.", r"\1.conv_in.", key)
    key = re.sub(r"^(encoder|decoder)\.head\.0\.", r"\1.norm_out.", key)
    key = re.sub(r"^(encoder|decoder)\.head\.2\.", r"\1.conv_out.", key)
    for i, dest in enumerate(("resnets.0", "attentions.0", "resnets.1")):
        key = key.replace(f".middle.{i}.", f".mid_block.{dest}.")
    for side, source, dest, resnets in (("encoder", "downsamples", "down_blocks", 2), ("decoder", "upsamples", "up_blocks", 3)):
        for block in range(5):
            for i in range(resnets + 1):
                suffix = f"resnets.{i}" if i < resnets else ("downsampler" if side == "encoder" else "upsampler")
                key = key.replace(f"{side}.{source}.{block}.{source}.{i}.", f"{side}.{dest}.{block}.{suffix}.")
    for i, dest in ((0, "norm1"), (2, "conv1"), (3, "norm2"), (6, "conv2")):
        key = key.replace(f".residual.{i}.", f".{dest}.")
    return key.replace(".shortcut.", ".conv_shortcut.")


def main():
    import mlx.core as mx

    parser = argparse.ArgumentParser()
    parser.add_argument("--mflux", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True, help="official Qwen VAE config.json (mflux constants contain transcription differences)")
    args = parser.parse_args()
    sys.path.insert(0, str(args.mflux / "src"))
    from mflux.models.qwen21.model.qwen21_vae.qwen21_vae import Qwen21VAE
    from mflux.models.qwen21.weights.qwen21_weight_mapping import Qwen21WeightMapping

    weights = {canonical(k): (mx.squeeze(v, 2) if v.ndim == 5 and v.shape[2] == 1 else v)
               for k, v in mx.load(str(args.weights)).items() if "time_conv" not in k}
    model = Qwen21VAE()
    config = json.loads(args.config.read_text())
    object.__setattr__(model, "LATENTS_MEAN", config["latents_mean"])
    object.__setattr__(model, "LATENTS_STD", config["latents_std"])
    params = []
    for target in Qwen21WeightMapping.get_vae_mapping():
        source = weights[target.from_pattern[0]]
        params.append((target.to_pattern, target.transform(source) if target.transform else source))
    model.load_weights(params, strict=True)
    mx.eval(model.parameters())
    for channels in (3, 4):
        mx.random.seed(912 + channels)
        latents = mx.random.normal((1, 64, 2, 3)).astype(mx.bfloat16)
        image = mx.random.uniform(-1, 1, shape=(1, channels, 32, 48)).astype(mx.float32)
        mean = mx.array(model.LATENTS_MEAN).reshape(1, 64, 1, 1)
        std = mx.array(model.LATENTS_STD).reshape(1, 64, 1, 1)
        decoded = model.decoder(model.post_quant_conv(latents * std + mean))
        encoded = model.encode(image)
        mx.eval(decoded, encoded)
        with tempfile.TemporaryDirectory(prefix="tc-qwen21-vae-") as directory:
            root = Path(directory)
            mx.save_safetensors(str(root / "inputs.safetensors"), {"latents": latents, "image": image})
            subprocess.run([str(args.probe.resolve()), str(args.weights.resolve()),
                            str(root / "inputs.safetensors"), str(root / "outputs.safetensors")], check=True)
            actual = mx.load(str(root / "outputs.safetensors"))
            errors = {}
            for name, expected in (("decoded", decoded), ("encoded", encoded)):
                delta = actual[name].astype(mx.float32) - expected.astype(mx.float32)
                errors[name] = {"max_abs": mx.max(mx.abs(delta)).item(),
                                "rmse": mx.sqrt(mx.mean(delta * delta)).item()}
            alpha_delta = actual["decoded"][:, 3] - decoded[:, 3]
            errors["alpha_max_abs"] = mx.max(mx.abs(alpha_delta)).item()
            print(json.dumps({"input_channels": channels, "errors": errors}), flush=True)
            assert errors["decoded"]["rmse"] < 1e-4 and errors["encoded"]["rmse"] < 1e-4, errors
            assert errors["alpha_max_abs"] < 1e-3, errors


if __name__ == "__main__":
    main()

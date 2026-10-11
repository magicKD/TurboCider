#!/usr/bin/env python3
"""Complete a verified CoW encoder view with both original BF16 FFN planes."""
import argparse
import json
from pathlib import Path
import shutil

from flux9_bf16_framework_screen import generation, write_json, check_generations
from mflux_original_layout import read_header, rewrite

ROOT = Path(__file__).resolve().parents[2]


def complete(destination):
    copied = destination / "text-physical-clone-receipt.json"
    encoder = json.loads(copied.read_text())
    if encoder["full_payload_equal"] is not True or encoder["kind"] != "qwen-text":
        raise ValueError("verified complete-payload encoder clone required")
    check_generations({encoder["source"]: encoder["source_generation"], encoder["derived"]: encoder["derived_generation"]})
    if Path(encoder["derived"]) != destination / "text_encoder/model.safetensors":
        raise ValueError("clone target is not this exact reference view")
    original = ROOT / "models/Comfy-Org-Qwen-Image-2.1"
    plans = [(original / "diffusion_models/qwen_image_2.1_bf16.safetensors", "transformer/diffusion_pytorch_model.safetensors", "qwen-dit"),
             (original / "vae/qwen_image_2.1_vae_bf16.safetensors", "vae/diffusion_pytorch_model.safetensors", "qwen-vae")]
    if shutil.disk_usage(ROOT).free < sum(source.stat().st_size for source, _, _ in plans) + (2 << 30):
        raise ValueError("derived DiT/VAE plus 2GiB reserve unavailable")
    receipt = destination / "layout-receipt.json"
    if receipt.exists():
        raise ValueError("fresh complete layout receipt required")
    report = dict(schema="tc-lossless-reference-layout-v1", family="qwen21", status="preparing", qualification_passed=False,
        files=[encoder], links=[], scope="complete source parameters: language prefix, all32 fused gate/up range splits, VAE names/singleton axes; full payload preserved")
    write_json(receipt, report)
    try:
        for source, relative, kind in plans:
            report["files"].append(rewrite(source, destination / relative, kind))
            write_json(receipt, report)
        header, _ = read_header(destination / "text_encoder/model.safetensors")
        write_json(destination / "text_encoder/model.safetensors.index.json",
                   dict(weight_map={name: "model.safetensors" for name in header if name != "__metadata__"}))
        processor = destination / "processor"
        processor.mkdir()
        source = original / "processor/tokenizer.json"
        target = processor / "tokenizer.json"
        target.symlink_to(source)
        report["links"].append(dict(path=str(target), source=str(source), generation=generation(source)))
        write_json(processor / "tokenizer_config.json", dict(tokenizer_class="Qwen2Tokenizer", model_max_length=131072,
                   eos_token="<|im_end|>", pad_token="<|endoftext|>"))
        report["status"] = "complete_layout_only"
    except BaseException as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        write_json(receipt, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", type=Path, required=True)
    args = parser.parse_args()
    report = complete(args.view.absolute())
    print(json.dumps(dict(status=report["status"], files=len(report["files"]))))


if __name__ == "__main__":
    main()

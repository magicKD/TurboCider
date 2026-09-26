"""Qwen3.5 PE vision component vs official Transformers CPU oracle, not edit acceptance."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import time

import torch
from safetensors.torch import load_file, save_file
from safetensors import safe_open
import transformers
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5VisionModel, apply_rotary_pos_emb_vision,
)


@torch.inference_mode()
def reference_thread_diagnostics(model, patches, grid, expected, thread_counts):
    """Diagnostic only: never replace the pinned baseline or its acceptance gate."""
    original_threads = torch.get_num_threads()
    records = []
    try:
        for threads in thread_counts:
            if threads <= 0:
                raise ValueError("reference thread counts must be positive")
            torch.set_num_threads(threads)
            actual = model(patches, grid).pooler_output
            a, b = actual.double(), expected.double()
            records.append(dict(
                threads=threads, exact_equal=bool(torch.equal(actual, expected)),
                finite=bool(torch.isfinite(actual).all()),
                mismatch_count=int((actual != expected).sum()),
                relative_rmse=float((a-b).square().mean().sqrt() /
                                    b.square().mean().sqrt().clamp_min(1e-30)),
                max_abs=float((a-b).abs().max())))
    finally:
        torch.set_num_threads(original_threads)
    return records


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--compensated-projection", action="store_true",
                        help="Experimental native reduction; does not change reference or acceptance")
    parser.add_argument("--native-fp32-visual", action="store_true",
                        help="Test native FP32 visual arithmetic directly from the BF16 checkpoint; requires real weights and --dtype float32")
    parser.add_argument("--reference-thread-check", type=int, nargs="+", default=[],
                        help="Compare official CPU outputs at these thread counts; diagnostic only")
    parser.add_argument("--weights", type=Path, help="Real PE shard containing model.visual.*")
    parser.add_argument("--patches-fixture", type=Path,
                        help="Validated multimodal-probe output (grids and patchesN) for real-image vision parity")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), help="Diagnostic precision; defaults to both for synthetic, BF16 for real weights")
    parser.add_argument("--reference-version", default="5.4.0", help="Require the official pinned Transformers version; override only for explicit version comparisons")
    args = parser.parse_args()
    if any(threads <= 0 for threads in args.reference_thread_check):
        parser.error("--reference-thread-check requires positive counts")
    if args.native_fp32_visual and (not args.weights or args.dtype != "float32" or args.compensated_projection):
        parser.error("--native-fp32-visual requires --weights and --dtype float32 without compensated projection")
    if args.patches_fixture and not args.weights:
        parser.error("--patches-fixture requires real --weights")
    if transformers.__version__ != args.reference_version:
        parser.error(f"Expected Transformers {args.reference_version}, found {transformers.__version__}; use the pinned PE reference environment")
    torch.set_num_threads(2)
    torch.manual_seed(427)
    reports = []
    fixtures = []
    if args.patches_fixture:
        tensors = load_file(str(args.patches_fixture))
        grids = tensors['grids']
        if grids.ndim != 2 or grids.shape[1] != 3 or not 1 <= grids.shape[0] <= 10:
            parser.error("invalid fixture grids")
        for i, (temporal, height, width) in enumerate(grids.tolist()):
            if temporal != 1 or height <= 0 or width <= 0 or height % 2 or width % 2:
                parser.error("fixture must contain still-image merge2 grids")
            values = tensors[f'patches{i}']
            if values.shape != (height*width, 3*2*16*16) or not torch.isfinite(values).all():
                parser.error("invalid fixture patches")
            fixtures.append((height, width, values))
    def relative_rmse(actual, expected):
        a, b = actual.float(), expected.float()
        return float((a-b).square().mean().sqrt()/b.square().mean().sqrt().clamp_min(1e-30))
    with tempfile.TemporaryDirectory(prefix="tc-qwen35-vision-") as directory:
        root = Path(directory)
        dtypes = (getattr(torch, args.dtype),) if args.dtype else ((torch.bfloat16,) if args.weights else (torch.float32, torch.bfloat16))
        for dtype in dtypes:
            hidden, heads, depth, patch, side, out, intermediate = (
                (1152, 16, 27, 16, 48, 4096, 4304) if args.weights else (32, 4, 3, 2, 4, 24, 64))
            config = Qwen3_5VisionConfig(hidden_size=hidden, num_heads=heads, depth=depth,
                patch_size=patch, temporal_patch_size=2, spatial_merge_size=2,
                num_position_embeddings=side*side, out_hidden_size=out, intermediate_size=intermediate,
                hidden_act="gelu_pytorch_tanh", attn_implementation="eager")
            model = Qwen3_5VisionModel(config).eval()
            # Official from_pretrained(dtype=...) casts checkpoint parameters,
            # but initializes nonpersistent rotary buffers in FP32. A blanket
            # model.to(BF16) also rounds inv_freq and is a different oracle.
            rotary_buffers = {name: value.clone() for name, value in model.rotary_pos_emb.named_buffers()}
            model.to(dtype)
            for name, value in rotary_buffers.items():
                setattr(model.rotary_pos_emb, name, value)
            assert model.rotary_pos_emb.inv_freq.dtype == torch.float32
            trace = {}
            model.patch_embed.register_forward_hook(lambda _, args, value: trace.update(patch=value.detach()))
            model.blocks[0].register_forward_pre_hook(lambda _, args: trace.update(input=args[0].detach()))
            for i, block in enumerate(model.blocks):
                block.register_forward_hook(lambda _, args, value, i=i: trace.update({f"block{i}":value.detach()}))
            block = model.blocks[0]
            for key, module in dict(norm1=block.norm1, qkv=block.attn.qkv, attention=block.attn,
                                    norm2=block.norm2, fc1=block.mlp.linear_fc1,
                                    activation=block.mlp.act_fn, fc2=block.mlp.linear_fc2).items():
                module.register_forward_hook(lambda _, args, value, key=key: trace.update({key:value.detach()}))
            if args.weights:
                with safe_open(args.weights, framework="pt", device="cpu") as shard:
                    state = {key[len("model.visual."):]: shard.get_tensor(key)
                             for key in shard.keys() if key.startswith("model.visual.")}
                model.load_state_dict(state, strict=True)
                weight_path = args.weights.resolve()
                if dtype != torch.bfloat16 and not args.native_fp32_visual:
                    weight_path = root / "converted-weights.safetensors"
                    save_file({"model.visual."+key: value.to(dtype).contiguous()
                               for key, value in state.items()}, str(weight_path))
            else:
                for name, param in model.named_parameters():
                    if name.endswith("norm.weight") or ".norm1.weight" in name or ".norm2.weight" in name:
                        param.add_(torch.randn_like(param) * .03)
                    elif name.endswith("bias"):
                        param.copy_(torch.randn_like(param) * .03)
                weight_path = root / "weights.safetensors"
                save_file({"model.visual."+key: value.contiguous()
                           for key, value in model.state_dict().items()}, str(weight_path))
            cases = fixtures or [(h, w, None) for h, w in ((2, 2), (4, 6), (6, 4), (8, 10))]
            for height, width, fixture in cases:
                patches = fixture.to(dtype) if fixture is not None else torch.randn(height*width, 3*2*patch*patch, dtype=dtype)
                expected = model(patches, torch.tensor([[1, height, width]])).pooler_output
                save_file({"patches": patches}, str(root/"input.safetensors"),
                          {key:str(value) for key, value in dict(patch=patch, hidden=hidden, heads=heads,
                              layers=depth, position_side=side, height=height, width=width,
                              compensated_projection=int(args.compensated_projection),
                              fp32_visual=int(args.native_fp32_visual)).items()})
                native_start = time.perf_counter()
                subprocess.run([str(args.probe.resolve()), str(weight_path), str(root/"input.safetensors"),
                                str(root/"output.safetensors")] + (["--gpu"] if args.gpu else []), check=True)
                native_probe_seconds = time.perf_counter() - native_start
                if args.compensated_projection:
                    with safe_open(root/"output.safetensors", framework="pt") as native_file:
                        if (native_file.metadata() or {}).get("compensated_projection") != "1":
                            raise RuntimeError("Probe did not confirm compensated projection; rebuild the probe")
                if args.native_fp32_visual:
                    with safe_open(root/"output.safetensors", framework="pt") as native_file:
                        if (native_file.metadata() or {}).get("fp32_visual") != "1":
                            raise RuntimeError("Probe did not confirm native FP32 visual mode; rebuild the probe")
                outputs = load_file(str(root/"output.safetensors"))
                result = outputs["merged"]
                assert result.dtype == expected.dtype and result.shape == expected.shape
                # Large real-image features can make FP32 norm/dot reductions
                # produce cosine > 1. Use FP64 metrics, not a clamped score.
                a, b = result.double(), expected.double()
                rrms = float((a-b).square().mean().sqrt()/b.square().mean().sqrt())
                cosine = float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0))
                record = dict(dtype=str(dtype), grid=[height,width], relative_rmse=rrms,
                              cosine=cosine, max_abs=float((a-b).abs().max()),
                              native_probe_seconds=native_probe_seconds)
                record["trace_relative_rmse"] = {
                    key:float((outputs[key].float()-value.float()).square().mean().sqrt()/value.float().square().mean().sqrt())
                    for key, value in trace.items()}
                if args.weights and hasattr(model, "fast_pos_embed_interpolate"):
                    grid = torch.tensor([[1, height, width]], dtype=torch.long)
                    expected_pos = model.fast_pos_embed_interpolate(grid)
                    rope = model.rot_pos_emb(grid).reshape(height * width, -1)
                    expected_cos = torch.cat((rope, rope), dim=-1).cos()
                    expected_sin = torch.cat((rope, rope), dim=-1).sin()
                    record["position_relative_rmse"] = float((outputs["pos"].float()-expected_pos.float()).square().mean().sqrt()/expected_pos.float().square().mean().sqrt())
                    record["rope_cos_relative_rmse"] = float((outputs["rope_cos"].float()-expected_cos.float()).square().mean().sqrt()/expected_cos.float().square().mean().sqrt())
                    record["rope_sin_relative_rmse"] = float((outputs["rope_sin"].float()-expected_sin.float()).square().mean().sqrt()/expected_sin.float().square().mean().sqrt())
                # Isolate primitive differences using identical native inputs,
                # instead of confusing earlier accumulated error with a local bug.
                first = model.blocks[0]
                residual = outputs["input"] + outputs["attention"]
                local_expected = dict(
                    norm1=first.norm1(outputs["input"]),
                    qkv=first.attn.qkv(outputs["norm1"]),
                    norm2=first.norm2(residual),
                    fc1=first.mlp.linear_fc1(outputs["norm2"]),
                    activation=first.mlp.act_fn(outputs["fc1"]),
                    fc2=first.mlp.linear_fc2(outputs["activation"]),
                    block0=residual + outputs["fc2"])
                if "attn_raw_score" in outputs:
                    packed = outputs["qkv"].reshape(height*width, 3, heads, -1).permute(1, 0, 2, 3)
                    grid = torch.tensor([[1, height, width]], dtype=torch.long)
                    rope = model.rot_pos_emb(grid).reshape(height*width, -1)
                    angles = torch.cat((rope, rope), dim=-1)
                    q, k = apply_rotary_pos_emb_vision(packed[0], packed[1], angles.cos(), angles.sin())
                    local_expected.update(
                        attn_q=q.transpose(0, 1).unsqueeze(0),
                        attn_k=k.transpose(0, 1).unsqueeze(0),
                        attn_v=packed[2].transpose(0, 1).unsqueeze(0),
                        attn_raw_score=torch.matmul(outputs["attn_q"], outputs["attn_k"].transpose(-1,-2)),
                        attn_score=outputs["attn_raw_score"] * first.attn.scaling,
                        attn_probability=torch.softmax(outputs["attn_score"].float(), dim=-1).to(dtype),
                        attn_output=torch.matmul(outputs["attn_probability"], outputs["attn_v"]),
                        attention=first.attn.proj(outputs["attn_output"].transpose(1, 2).reshape(height*width, hidden)))
                for variant in ("qkv_bias_after_round", "qkv_native_addmm", "qkv_native_separate"):
                    if variant in outputs:
                        local_expected[variant] = local_expected["qkv"]
                record["local_op_relative_rmse"] = {
                    key:float((outputs[key].float()-value.float()).square().mean().sqrt()/value.float().square().mean().sqrt())
                    for key, value in local_expected.items()}
                # BF16 products are exactly representable in FP32 but long
                # reductions can cross a BF16 rounding midpoint. Compare both
                # implementations with a FP64-accumulated diagnostic oracle;
                # it does not replace the pinned model acceptance reference.
                record["linear_accumulation_diagnostics"] = {}
                for key, value, module in (
                    ("qkv", outputs["norm1"], first.attn.qkv),
                    ("fc1", outputs["norm2"], first.mlp.linear_fc1),
                    ("fc2", outputs["activation"], first.mlp.linear_fc2),
                ):
                    exact = torch.nn.functional.linear(value.double(), module.weight.double(),
                        module.bias.double()).to(dtype)
                    fp32 = torch.nn.functional.linear(value.float(), module.weight.float(),
                        module.bias.float()).to(dtype)
                    record["linear_accumulation_diagnostics"][key] = dict(
                        native_vs_fp64=relative_rmse(outputs[key], exact),
                        reference_vs_fp64=relative_rmse(local_expected[key], exact),
                        torch_fp32_vs_fp64=relative_rmse(fp32, exact),
                        native_mismatch_count=int((outputs[key] != exact).sum()),
                        reference_mismatch_count=int((local_expected[key] != exact).sum()))
                # Replay every official block from the corresponding native
                # input. This separates local arithmetic differences from
                # upstream error amplification, including late tower blocks.
                grid = torch.tensor([[1, height, width]], dtype=torch.long)
                rope = model.rot_pos_emb(grid).reshape(height*width, -1)
                angles = torch.cat((rope, rope), dim=-1)
                cu_seqlens = torch.tensor([0, height*width], dtype=torch.int32)
                record["local_block_relative_rmse"] = {}
                for i, block in enumerate(model.blocks):
                    native_input = outputs["input" if i == 0 else f"block{i-1}"]
                    replay = block(native_input, cu_seqlens=cu_seqlens,
                                   position_embeddings=(angles.cos(), angles.sin()))
                    record["local_block_relative_rmse"][f"block{i}"] = relative_rmse(outputs[f"block{i}"], replay)
                record["local_merger_relative_rmse"] = relative_rmse(result, model.merger(outputs[f"block{depth-1}"]))
                if args.reference_thread_check:
                    record["reference_thread_diagnostics"] = reference_thread_diagnostics(
                        model, patches, grid, expected, args.reference_thread_check)
                record["passed"] = rrms < (.02 if dtype == torch.bfloat16 else .0001) and cosine > .999
                reports.append(record)
                print(json.dumps(record), flush=True)
                args.output.write_text(json.dumps(dict(passed=False, cases=reports, real_weights=bool(args.weights),
                    transformers_version=transformers.__version__, torch_version=torch.__version__), indent=2)+"\n")
    passed = all(record["passed"] for record in reports)
    args.output.write_text(json.dumps(dict(passed=passed, cases=reports, real_weights=bool(args.weights),
        compensated_projection=args.compensated_projection,
        native_fp32_visual=args.native_fp32_visual,
        patches_fixture=str(args.patches_fixture) if args.patches_fixture else None,
        device="gpu" if args.gpu else "cpu", transformers_version=transformers.__version__, torch_version=torch.__version__,
        reference_threads=2, reference_thread_check=args.reference_thread_check, seed=427,
        scope="vision component only; no image processor or edit PE"), indent=2)+"\n")
    if not passed:
        raise SystemExit("PE vision parity failed; see per-grid diagnostics (thresholds unchanged)")


if __name__ == "__main__":
    main()

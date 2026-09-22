"""CPU-only official Qwen3.5 delta oracle; development tool, never inference dependency."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile

import torch
from safetensors.torch import load_file, save_file
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    torch_recurrent_gated_delta_rule, torch_chunk_gated_delta_rule,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", action="store_true", help="Test native GPU, retaining the CPU Torch oracle")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    reports = []
    with tempfile.TemporaryDirectory(prefix="tc-qwen35-delta-") as temp:
        root = Path(temp)
        def run(data, normalize=True, expect_success=True, chunk_size=64):
            save_file({k: v.contiguous() for k, v in data.items()}, str(root / "input.safetensors"),
                      {"normalize": str(normalize).lower(), "chunk_size":str(chunk_size)})
            process = subprocess.run([str(args.probe.resolve()), str(root / "input.safetensors"),
                                      str(root / "output.safetensors")] + (["--gpu"] if args.gpu else []), capture_output=True, text=True)
            if not expect_success:
                assert process.returncode != 0, "Malformed tensor input was accepted"
                return
            assert process.returncode == 0, process.stderr
            return load_file(str(root / "output.safetensors"))

        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            for length, heads, kd, vd in ((1, 3, 8, 5), (7, 3, 8, 5), (64, 2, 8, 4),
                                        (65, 2, 8, 4), (129, 2, 8, 4), (3, 32, 128, 128),
                                        (130, 32, 128, 128)):
                for normalize in (False, True):
                    data = {"q": torch.randn(2, length, heads, kd, dtype=dtype) * 0.2,
                            "k": torch.randn(2, length, heads, kd, dtype=dtype) * 0.2,
                            "v": torch.randn(2, length, heads, vd, dtype=dtype),
                            "g": -torch.rand(2, length, heads),
                            "beta": torch.rand(2, length, heads).to(dtype)}
                    if length != 1:
                        data["initial"] = torch.randn(2, heads, kd, vd) * 0.1
                    expected, state = torch_recurrent_gated_delta_rule(
                        data["q"], data["k"], data["v"], data["g"], data["beta"],
                        initial_state=data.get("initial"), output_final_state=True,
                        use_qk_l2norm_in_kernel=normalize)
                    result = run(data, normalize)
                    tolerance = 2e-5 if dtype == torch.float32 else (0.004 if dtype == torch.bfloat16 else 0.0005)
                    torch.testing.assert_close(result["output"], expected, atol=tolerance, rtol=tolerance)
                    torch.testing.assert_close(result["state"], state, atol=2e-5, rtol=2e-5)
                    if length > 1:
                        # Last-token evaluation has a different vectorization
                        # shape. FP32 reduction rounding need not be bit-identical.
                        torch.testing.assert_close(result["split_output"], result["output"], atol=tolerance, rtol=tolerance)
                        torch.testing.assert_close(result["split_state"], result["state"], atol=2e-6, rtol=2e-6)
                    # Official prefill takes the chunked algorithm; check the
                    # recurrent primitive against that independent path too.
                    chunk_output, chunk_state = torch_chunk_gated_delta_rule(
                        data["q"], data["k"], data["v"], data["g"], data["beta"],
                        initial_state=data.get("initial"), output_final_state=True,
                        use_qk_l2norm_in_kernel=normalize)
                    torch.testing.assert_close(result["output"], chunk_output, atol=tolerance, rtol=tolerance)
                    torch.testing.assert_close(result["state"], chunk_state, atol=3e-5, rtol=3e-5)
                    torch.testing.assert_close(result["chunk_output"], chunk_output, atol=tolerance, rtol=tolerance)
                    torch.testing.assert_close(result["chunk_state"], chunk_state, atol=3e-5, rtol=3e-5)
                    if length > 1:
                        torch.testing.assert_close(result["chunk_split_output"], chunk_output, atol=tolerance, rtol=tolerance)
                        torch.testing.assert_close(result["chunk_split_state"], chunk_state, atol=3e-5, rtol=3e-5)
                    reports.append(dict(dtype=str(dtype), length=length, heads=heads, key_dim=kd, value_dim=vd,
                                        normalize=normalize,
                                        output_max_abs=float((result["output"].float() - expected.float()).abs().max()),
                                        state_max_abs=float((result["state"] - state).abs().max()),
                                        chunk_output_max_abs=float((result["chunk_output"].float() - chunk_output.float()).abs().max()),
                                        chunk_state_max_abs=float((result["chunk_state"] - chunk_state).abs().max())))
        # Bad state/gate/key geometry must fail before creating a recurrence.
        run({**data, "initial": torch.zeros(1)}, expect_success=False)
        run({**data, "beta": torch.zeros(1)}, expect_success=False)
        run({**data, "k": data["k"][:, :, :, :-1]}, expect_success=False)
        for size in (1, 7, 16, 128):
            result = run(data, chunk_size=size)
            torch.testing.assert_close(result["chunk_output"], result["output"], atol=tolerance, rtol=tolerance)
            torch.testing.assert_close(result["chunk_state"], result["state"], atol=3e-5, rtol=3e-5)
        for size in (0, 129):
            run(data, chunk_size=size, expect_success=False)
    report = dict(scope="component parity only; not full PE model inference or performance",
                  native_device="gpu" if args.gpu else "cpu",
                  cases=reports, malformed_rejected=3, invalid_chunk_sizes_rejected=2,
                  extra_chunk_sizes=[1, 7, 16, 128], passed=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"PASS {len(reports)} Qwen35 delta cases, recurrent/chunked oracle, cached continuation, cancellation, 3 invalid shapes, 4 extra chunk sizes, 2 invalid chunk sizes")


if __name__ == "__main__":
    main()

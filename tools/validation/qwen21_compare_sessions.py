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


def select_runs(baseline, candidate, named_run=None, prefix_hits_only=False):
    """Select like-for-like resident runs, rejecting an incomplete hit history."""
    baseline_names = sorted(p.name for p in baseline.glob("run-*.json"))
    candidate_names = sorted(p.name for p in candidate.glob("run-*.json"))
    assert not (named_run and prefix_hits_only), "select either one named run or matched prefix hits"
    if named_run:
        assert named_run in baseline_names and named_run in candidate_names, "named run missing"
        return [named_run]
    assert baseline_names and baseline_names == candidate_names, "unmatched runs"
    if not prefix_hits_only:
        return baseline_names
    assert len(baseline_names) >= 2 and baseline_names == [
        f"run-{index}.json" for index in range(len(baseline_names))
    ], "prefix-hit comparison needs a contiguous miss-then-hit run sequence"
    for root in (baseline, candidate):
        miss = json.loads((root / baseline_names[0]).read_text())
        assert "resident prefix KV miss" in miss["acceleration_selection"], (
            "prefix-hit comparison requires a preceding resident prefix miss")
        for name in baseline_names[1:]:
            hit = json.loads((root / name).read_text())
            assert "resident prefix KV hit" in hit["acceleration_selection"], (
                "mixed prefix miss/hit comparison")
            assert hit["prompt_cache_hit"], "prefix hit unexpectedly re-encoded conditioning"
            assert all(hit[key] == miss[key] for key in (
                "model", "operation", "width", "height", "steps", "text_tokens", "reference_tokens"
            )), "prefix hit changed request geometry"
    return baseline_names[1:]


def validate_db_cache_pair(baseline, candidate, before_baseline=None, before_candidate=None):
    """Check a DBCache run's saved work against its actual hybrid predictions."""
    assert baseline.get("qwen21_dbcache") is None, "DBCache baseline is already cached"
    cache = candidate.get("qwen21_dbcache")
    assert cache and cache["front_blocks"] == 8 and cache["back_blocks"] == 0
    assert cache["warmup_steps"] == 8 and 0 < cache["threshold"] <= 0.5
    assert 1 <= cache.get("max_consecutive", 2) <= 8, "invalid consecutive skip bound"
    skipped = cache["cached_steps"]
    assert 0 <= skipped < candidate["steps"] - cache["warmup_steps"], (
        "DBCache skip count exceeds its eligible decode steps")
    assert cache["saved_middle_blocks"] == skipped * 24, "DBCache saved-block count disagrees"
    if before_baseline is not None:
        assert before_candidate is not None and baseline["hybrid"]["checkpoint_sha256_verified"]
        assert candidate["hybrid"]["checkpoint_sha256_verified"]
        rectangular = "qwen21_rectangular_decode_w8a8_tiled_diagnostic"
        a_labels = baseline["plan"]["algorithm_approximations"]
        b_labels = candidate["plan"]["algorithm_approximations"]
        assert (rectangular in a_labels) == (rectangular in b_labels), (
            "DBCache comparison changed rectangular hybrid route")
        tiles = 2 if rectangular in b_labels else 1
        if tiles == 2:
            assert sorted((baseline["width"], baseline["height"])) == [512, 768]
        assert baseline["hybrid"]["runtime_failures_session_total"] == \
               candidate["hybrid"]["runtime_failures_session_total"] == 0
        baseline_calls = baseline["hybrid"]["runtime_calls_session_total"] - \
            before_baseline["hybrid"]["runtime_calls_session_total"]
        candidate_calls = candidate["hybrid"]["runtime_calls_session_total"] - \
            before_candidate["hybrid"]["runtime_calls_session_total"]
        assert baseline_calls - candidate_calls == cache["saved_middle_blocks"] * tiles, (
            "DBCache reported saved layers do not match per-request Core ML predictions")


def validate_full_reference_hybrid_pair(baseline, candidate, previous_candidate):
    """Check a full-size-reference GPU→W8A8 comparison, with or without DBCache."""
    full_ref = "qwen21_w8a8_full_reference_diagnostic"
    assert baseline["plan"]["execution"] == "gpu" and \
        candidate["plan"]["execution"] == "gpu_ane_experimental"
    assert baseline["operation"] == candidate["operation"] == "image.edit"
    assert baseline["plan"]["qwen21_reference_size"] == \
        candidate["plan"]["qwen21_reference_size"] == 1024
    assert baseline["plan"]["lora_count"] == candidate["plan"]["lora_count"] == 0
    assert not baseline["plan"]["qwen21_w8a8"] and candidate["plan"]["qwen21_w8a8"]
    assert full_ref not in baseline["plan"]["algorithm_approximations"] and \
        full_ref in candidate["plan"]["algorithm_approximations"]
    assert baseline["reference_tokens"] in (4096, 8192, 12288) and \
        candidate["reference_tokens"] == baseline["reference_tokens"]
    hybrid = candidate["hybrid"]
    assert hybrid["checkpoint_sha256_verified"] and not hybrid["runtime_failed"] and \
        hybrid["runtime_failures_session_total"] == 0 and hybrid["block_count"] == 32 and \
        hybrid["bucket"] == 1024 and hybrid["ane_mlp_range"] == [0, 6144]
    a_cache, b_cache = baseline.get("qwen21_dbcache"), candidate.get("qwen21_dbcache")
    assert bool(a_cache) == bool(b_cache), "cross-route comparison changed DBCache mode"
    assert ("qwen21_decode_dbcache_diagnostic" in baseline["plan"]["algorithm_approximations"]) == \
        ("qwen21_decode_dbcache_diagnostic" in candidate["plan"]["algorithm_approximations"]), (
            "cross-route comparison changed DBCache opt-in")
    saved = 0
    if b_cache:
        assert (a_cache["threshold"], a_cache["max_consecutive"]) == \
            (b_cache["threshold"], b_cache["max_consecutive"]), (
                "cross-route comparison changed DBCache policy")
        assert b_cache["front_blocks"] == 8 and b_cache["back_blocks"] == 0
        assert 0 <= b_cache["cached_steps"] < candidate["steps"] - b_cache["warmup_steps"]
        assert b_cache["saved_middle_blocks"] == b_cache["cached_steps"] * 24
        saved = b_cache["saved_middle_blocks"]
    calls = hybrid["runtime_calls_session_total"] - \
        previous_candidate["hybrid"]["runtime_calls_session_total"]
    assert calls == (candidate["steps"] - 1) * 32 - saved, (
        "full-reference hybrid per-request Core ML prediction count mismatch")


def expected_target_only_saved_hybrid_calls(baseline, candidate):
    """Full-size references use GPU prefill; only tiled prefill saves Core ML calls."""
    tiled = "qwen21_tiled_prefill_last16_w8a8_diagnostic"
    a_labels = baseline["plan"]["algorithm_approximations"]
    b_labels = candidate["plan"]["algorithm_approximations"]
    if tiled in a_labels or tiled in b_labels:
        assert tiled in a_labels and tiled in b_labels, "target-only changed hybrid prefill route"
        rows = baseline["text_tokens"] + baseline["reference_tokens"] + \
            (baseline["width"] // 16) * (baseline["height"] // 16)
        return (rows + 1023) // 1024 - 1
    assert baseline["plan"]["qwen21_reference_size"] == \
        candidate["plan"]["qwen21_reference_size"] == 1024
    assert baseline["reference_tokens"] > 0, "GPU prefill target-only needs full-size references"
    return 0


def main():
    import numpy as np
    from PIL import Image

    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-dumps", action="store_true",
                        help="Require and compare dumped text/noise tensors")
    parser.add_argument("--run", choices=("run-0.json", "run-1.json"),
                        help="Compare a named warm run when one probe has extra repeats")
    parser.add_argument("--prefix-hits-only", action="store_true",
                        help="Compare only repeated resident prefix-KV hits from matched run sequences")
    parser.add_argument("--candidate-execution", default="gpu_ane_experimental",
                        choices=("gpu_ane_experimental", "gpu"),
                        help="Use gpu for explicit GPU-only approximate candidates")
    parser.add_argument("--baseline-execution", default="gpu",
                        choices=("gpu_ane_experimental", "gpu"),
                        help="Set to gpu_ane_experimental for a paired hybrid GPU suffix comparison")
    parser.add_argument("--candidate-gpu-w8a16", action="store_true",
                        help="Require BF16→W8A16 GPU suffix with the same W8A8 Core ML route")
    parser.add_argument("--candidate-approximation", default="qwen21_gpu_reuse_final_ffn",
        choices=("qwen21_gpu_reuse_final_ffn", "qwen21_metal_qk_norm_rope",
                                 "qwen21_metal_fused_qkv_diagnostic",
                                 "qwen21_reference_local_attention",
                                 "qwen21_last_reference_local_attention",
                                 "qwen21_last16_reference_local_attention_diagnostic",
                                 "qwen21_gpu_reuse_penultimate_even_ffn",
                                 "qwen21_viggle_v021_r256_6step_distillation",
                                 "qwen21_viggle_lora_fp16_matmuls",
                                 "qwen21_hybrid_reuse_final_ffn_diagnostic",
                                 "qwen21_hybrid_reuse_final_last16_ffn_diagnostic",
                                 "qwen21_hybrid_reuse_penultimate_even_ffn_diagnostic",
                                 "qwen21_prefill_last_target_only_diagnostic",
                                 "qwen21_rectangular_decode_w8a8_tiled_diagnostic",
                                 "qwen21_w8a8_full_reference_diagnostic",
                                 "qwen21_runtime_lora_base_ane_suffix_only_diagnostic",
                                 "qwen21_tiled_prefix_target_only_diagnostic",
                                 "qwen21_tiled_prefill_last20_w8a8_diagnostic",
                                 "qwen21_decode_dbcache_diagnostic"),
                        help="Expected additional approximation label for GPU or hybrid candidates")
    parser.add_argument("--additional-candidate-approximation", action="append", default=[],
                        help="Require another opt-in approximation on the candidate, absent from the baseline")
    args = parser.parse_args()
    names = select_runs(args.baseline, args.candidate, args.run, args.prefix_hits_only)
    comparisons = []
    for name in names:
        a, b = (json.loads((root / name).read_text()) for root in (args.baseline, args.candidate))
        keys = ("model", "operation", "width", "height", "steps", "seed", "text_tokens",
                "reference_tokens")
        assert all(a[key] == b[key] for key in keys), "request metadata differs"
        assert a["prompt_cache_hit"] and b["prompt_cache_hit"], "not a warm prompt-cached comparison"
        assert a["plan"]["execution"] == args.baseline_execution and b["plan"]["execution"] == args.candidate_execution
        if args.candidate_approximation == "qwen21_decode_dbcache_diagnostic":
            assert args.baseline_execution == args.candidate_execution, (
                "DBCache should be compared within one execution route")
            before_a = before_b = None
            if args.candidate_execution == "gpu_ane_experimental":
                prior = "warmup.json" if name == "run-0.json" else "run-0.json"
                before_a, before_b = (json.loads((root / prior).read_text())
                                      for root in (args.baseline, args.candidate))
                assert a["hybrid"]["bucket"] == b["hybrid"]["bucket"] and \
                    a["hybrid"]["ane_mlp_range"] == b["hybrid"]["ane_mlp_range"], (
                    "DBCache hybrid comparison changed model geometry")
            validate_db_cache_pair(a, b, before_a, before_b)
        if args.prefix_hits_only:
            assert "resident prefix KV hit" in a["acceleration_selection"] and \
                   "resident prefix KV hit" in b["acceleration_selection"], "mixed prefix miss/hit comparison"
            if args.candidate_execution == "gpu_ane_experimental":
                assert b["hybrid"]["checkpoint_sha256_verified"] and \
                       b["hybrid"]["runtime_failures_session_total"] == 0 and \
                       b["hybrid"]["block_count"] == 32, "hybrid artifact or runtime is not verified"
        if args.candidate_gpu_w8a16:
            assert args.baseline_execution == args.candidate_execution == "gpu_ane_experimental"
            assert not a["plan"]["qwen21_gpu_w8a16"] and b["plan"]["qwen21_gpu_w8a16"]
            assert a["plan"]["qwen21_w8a8"] and b["plan"]["qwen21_w8a8"]
            assert a["hybrid"]["bucket"] == b["hybrid"]["bucket"]
            assert a["hybrid"]["ane_mlp_range"] == b["hybrid"]["ane_mlp_range"]
            assert a["hybrid"]["checkpoint_sha256_verified"] and b["hybrid"]["checkpoint_sha256_verified"]
            assert a["hybrid"]["runtime_calls_session_total"] == b["hybrid"]["runtime_calls_session_total"]
            assert a["hybrid"]["runtime_failures_session_total"] == b["hybrid"]["runtime_failures_session_total"] == 0
            assert a["hybrid"]["block_count"] == b["hybrid"]["block_count"] == 32
        if args.candidate_approximation == "qwen21_rectangular_decode_w8a8_tiled_diagnostic":
            assert args.baseline_execution == "gpu" and args.candidate_execution == "gpu_ane_experimental"
            assert sorted((b["width"], b["height"])) == [512, 768]
            assert not a["plan"]["qwen21_w8a8"] and b["plan"]["qwen21_w8a8"]
            assert b["hybrid"]["checkpoint_sha256_verified"] and not b["hybrid"]["runtime_failed"]
            assert b["hybrid"]["bucket"] == 1024 and b["hybrid"]["ane_mlp_range"] == [0, 6144]
            assert ("qwen21_decode_dbcache_diagnostic" in a["plan"]["algorithm_approximations"]) == \
                   ("qwen21_decode_dbcache_diagnostic" in b["plan"]["algorithm_approximations"]), (
                "rectangular comparison changed DBCache setting")
            assert bool(a.get("qwen21_dbcache")) == bool(b.get("qwen21_dbcache")), (
                "rectangular comparison changed active DBCache")
            skipped = 0
            if b.get("qwen21_dbcache"):
                cache = b["qwen21_dbcache"]
                assert (cache["threshold"], cache["max_consecutive"]) == (
                    a["qwen21_dbcache"]["threshold"], a["qwen21_dbcache"]["max_consecutive"]), (
                    "rectangular comparison changed DBCache policy")
                assert cache["front_blocks"] == 8 and cache["back_blocks"] == 0
                assert cache["saved_middle_blocks"] == cache["cached_steps"] * 24
                skipped = cache["saved_middle_blocks"]
            prior = "warmup.json" if name == "run-0.json" else "run-0.json"
            previous = json.loads((args.candidate / prior).read_text())
            assert b["hybrid"]["runtime_calls_session_total"] - \
                previous["hybrid"]["runtime_calls_session_total"] == \
                    ((b["steps"] - 1) * 32 - skipped) * 2
        if args.candidate_approximation == "qwen21_w8a8_full_reference_diagnostic":
            assert args.baseline_execution == "gpu" and args.candidate_execution == "gpu_ane_experimental"
            prior = "warmup.json" if name == "run-0.json" else "run-0.json"
            validate_full_reference_hybrid_pair(a, b, json.loads((args.candidate / prior).read_text()))
        if args.candidate_approximation == "qwen21_runtime_lora_base_ane_suffix_only_diagnostic":
            assert args.baseline_execution == "gpu" and args.candidate_execution == "gpu_ane_experimental"
            assert a["lora_applied_projections"] == b["lora_applied_projections"] == 227
            assert a["plan"]["lora_count"] == b["plan"]["lora_count"] == 1
            assert b["hybrid"]["checkpoint_sha256_verified"] and not b["hybrid"]["runtime_failed"]
            assert b["hybrid"]["bucket"] == 1024 and b["hybrid"]["ane_mlp_range"] == [0, 6144]
            prior = "warmup.json" if name == "run-0.json" else "run-0.json"
            previous = json.loads((args.candidate / prior).read_text())
            assert b["hybrid"]["runtime_calls_session_total"] - \
                previous["hybrid"]["runtime_calls_session_total"] == (b["steps"] - 1) * 32
        hybrid_new_approximation = (args.baseline_execution == args.candidate_execution ==
                                    "gpu_ane_experimental" and args.candidate_approximation in
                                    ("qwen21_metal_qk_norm_rope",
                                     "qwen21_metal_fused_qkv_diagnostic",
                                     "qwen21_last16_reference_local_attention_diagnostic",
                                     "qwen21_hybrid_reuse_final_ffn_diagnostic",
                                     "qwen21_hybrid_reuse_final_last16_ffn_diagnostic",
                                     "qwen21_hybrid_reuse_penultimate_even_ffn_diagnostic",
                                     "qwen21_prefill_last_target_only_diagnostic",
                                     "qwen21_tiled_prefix_target_only_diagnostic",
                                     "qwen21_tiled_prefill_last20_w8a8_diagnostic"))
        if hybrid_new_approximation:
            assert a["plan"]["qwen21_w8a8"] and b["plan"]["qwen21_w8a8"]
            assert a["hybrid"]["bucket"] == b["hybrid"]["bucket"]
            assert a["hybrid"]["ane_mlp_range"] == b["hybrid"]["ane_mlp_range"]
            assert a["hybrid"]["checkpoint_sha256_verified"] and b["hybrid"]["checkpoint_sha256_verified"]
            if args.candidate_approximation in ("qwen21_hybrid_reuse_final_ffn_diagnostic",
                                                 "qwen21_hybrid_reuse_final_last16_ffn_diagnostic",
                                                 "qwen21_hybrid_reuse_penultimate_even_ffn_diagnostic",
                                                 "qwen21_prefill_last_target_only_diagnostic",
                                                 "qwen21_tiled_prefix_target_only_diagnostic",
                                                 "qwen21_tiled_prefill_last20_w8a8_diagnostic"):
                prior = "warmup.json" if name == "run-0.json" else "run-0.json"
                previous_a, previous_b = (json.loads((root / prior).read_text())
                                            for root in (args.baseline, args.candidate))
                baseline_calls = (a["hybrid"]["runtime_calls_session_total"] -
                                  previous_a["hybrid"]["runtime_calls_session_total"])
                candidate_calls = (b["hybrid"]["runtime_calls_session_total"] -
                                   previous_b["hybrid"]["runtime_calls_session_total"])
                if args.candidate_approximation == "qwen21_hybrid_reuse_final_ffn_diagnostic":
                    assert baseline_calls - candidate_calls == 32, "hybrid final FFN reuse call delta mismatch"
                elif args.candidate_approximation == "qwen21_hybrid_reuse_final_last16_ffn_diagnostic":
                    assert baseline_calls - candidate_calls == 16, "hybrid last-16 final FFN reuse call delta mismatch"
                elif args.candidate_approximation == "qwen21_hybrid_reuse_penultimate_even_ffn_diagnostic":
                    assert baseline_calls - candidate_calls == 16, "hybrid penultimate FFN reuse call delta mismatch"
                elif args.candidate_approximation == "qwen21_prefill_last_target_only_diagnostic":
                    assert baseline_calls - candidate_calls == expected_target_only_saved_hybrid_calls(a, b), (
                        "target-only last-block prediction count mismatch")
                elif args.candidate_approximation == "qwen21_tiled_prefix_target_only_diagnostic":
                    assert "resident prefix KV hit" in a["acceleration_selection"]
                    assert "resident prefix KV hit" in b["acceleration_selection"]
                    assert "qwen21_tiled_prefill_prefix_kv_diagnostic" in a["plan"]["algorithm_approximations"]
                    assert "qwen21_tiled_prefill_last16_w8a8_diagnostic" in a["plan"]["algorithm_approximations"]
                    remainder = (a["text_tokens"] + a["reference_tokens"]) % 1024
                    saved = 15 if "qwen21_prefill_last_target_only_diagnostic" in a["plan"]["algorithm_approximations"] else 16
                    assert baseline_calls - candidate_calls == (saved if remainder else 0), (
                        "tiled prefix target-only prediction count mismatch")
                else:
                    assert "qwen21_tiled_prefill_last16_w8a8_diagnostic" in a["plan"]["algorithm_approximations"]
                    first_step_rows = (a["text_tokens"] + a["reference_tokens"] +
                                       (a["width"] // 16) * (a["height"] // 16))
                    assert candidate_calls - baseline_calls == 4 * ((first_step_rows + 1023) // 1024), (
                        "last-20 versus last-16 prefill prediction count mismatch")
            else:
                # Sessions can use different warmup step counts. Compare the
                # measured request's calls, not cumulative preparation calls.
                prior = "warmup.json" if name == "run-0.json" else "run-0.json"
                previous_a, previous_b = (json.loads((root / prior).read_text())
                                            for root in (args.baseline, args.candidate))
                assert (a["hybrid"]["runtime_calls_session_total"] -
                        previous_a["hybrid"]["runtime_calls_session_total"] ==
                        b["hybrid"]["runtime_calls_session_total"] -
                        previous_b["hybrid"]["runtime_calls_session_total"]), (
                    "hybrid candidate changed per-request Core ML prediction count")
            assert a["hybrid"]["runtime_failures_session_total"] == b["hybrid"]["runtime_failures_session_total"] == 0
        if args.candidate_execution == "gpu" or hybrid_new_approximation or \
                args.candidate_approximation == "qwen21_decode_dbcache_diagnostic" or \
                args.candidate_approximation in ("qwen21_rectangular_decode_w8a8_tiled_diagnostic",
                                                 "qwen21_w8a8_full_reference_diagnostic",
                                                 "qwen21_runtime_lora_base_ane_suffix_only_diagnostic"):
            for approximation in [args.candidate_approximation, *args.additional_candidate_approximation]:
                assert approximation not in a["plan"]["algorithm_approximations"]
                assert approximation in b["plan"]["algorithm_approximations"]
        assert a["actual_denoise_steps"] == b["actual_denoise_steps"] == a["steps"]
        pixels = [np.asarray(Image.open(root / name.replace(".json", ".png")).convert("RGBA"),
                             dtype=np.float64) / 255 for root in (args.baseline, args.candidate)]
        x, y = (p[..., :3] for p in pixels)
        assert x.shape == y.shape
        dump_equal = None
        if args.verify_dumps:
            dump_names = ["qwen21_text.safetensors", "qwen21_initial.safetensors"]
            reference_names = [sorted(p.name for p in (root / name.replace(".json", "-dump")).glob(
                "qwen21_reference_*.safetensors")) for root in (args.baseline, args.candidate)]
            assert reference_names[0] == reference_names[1], "reference dump files differ"
            assert not a["reference_tokens"] or reference_names[0], "missing reference tensors"
            dump_names.extend(reference_names[0])
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
                  prefix_hits_only=args.prefix_hits_only,
                  input_validation="metadata matched; optional dumped text/noise/reference tensor equality checked" if args.verify_dumps else
                                   "metadata matched; prompt/noise tensor equality not checked by this script",
                  repeats=len(comparisons), runs=comparisons,
                  median_wall_speedup=statistics.median(r["baseline_wall"] for r in comparisons) /
                                      statistics.median(r["candidate_wall"] for r in comparisons))
    print(json.dumps(report, indent=2))
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

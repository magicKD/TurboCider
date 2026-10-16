#!/usr/bin/env python3
"""Run bounded learning experiments. Device benchmarks have separate explicit CLIs."""
import argparse
import inspect
import json
from pathlib import Path

from common import emit, report
import caches
import media
import operators
import policies
import quantization
import systems
import upscaling
import partition_axes
import contracts
import model_stages
import heterogeneous
import search_contracts
import pmu
import silu_domain
import shared_backing
import mlx_adoption
import real_shared_ffn
import block_context

CASES = {
    "quant-formats": quantization.formats,
    "w8a8-reference": quantization.w8a8,
    "w4a8-reference": quantization.w4a8,
    "smoothquant": quantization.smoothquant,
    "precision-boundaries": quantization.precision,
    "online-attention": operators.attention,
    "norm-rope": operators.norm_rope,
    "global-stat-tiles": operators.global_stats,
    "causal-halo": operators.causal_convolution,
    "overlap-dag": systems.overlap,
    "slot-ownership": systems.lifetimes,
    "bounded-io": systems.streaming_io,
    "startup-amortization": systems.startup,
    "bucket-selection": systems.bucket_selection,
    "technique-composition": policies.composition,
    "share-calibration": policies.calibration,
    "cache-identity": caches.cache_identity,
    "residual-reuse": caches.residual_reuse,
    "token-merge": caches.token_merge,
    "media-backpressure": media.bounded_frames,
    "profiling-scopes": media.profiling_scopes,
    "upscale-tiles": upscaling.tiled_upscale,
    "partition-axes": partition_axes.partition_axes,
    "owned-command-graph": contracts.owned_graph,
    "dual-bank-handoff": contracts.dual_bank_handoff,
    "packed-weight-store": contracts.packed_weights,
    "residency-policy": contracts.residency_policy,
    "hybrid-policy": contracts.hybrid_policy,
    "stage-encoder": model_stages.encoder_buckets,
    "stage-llm": model_stages.llm_prefill,
    "stage-decode": model_stages.llm_decode,
    "stage-dit": model_stages.dit_conditioning,
    "stage-vae": model_stages.vae_output_tiles,
    "parallel-branches": heterogeneous.parallel_branches,
    "attention-head-shards": heterogeneous.attention_head_shards,
    "consumer-strides": heterogeneous.consumer_strides,
    "segmented-ffn": heterogeneous.segmented_ffn,
    "optional-devices": heterogeneous.optional_devices,
    "resource-hazards": heterogeneous.resource_hazards,
    "precision-islands": heterogeneous.precision_islands,
    "search-qualification": search_contracts.qualification,
    "pmu-provenance": pmu.provenance,
    "silu-domain": silu_domain.domain,
    "backing-proof": shared_backing.backing_proof,
    "mlx-adoption-contract": mlx_adoption.adoption_contract,
    "real-shared-join": real_shared_ffn.join_contract,
    "block-evidence-axes": block_context.status_contract,
}


def run_case(name, seed=17, options=None):
    function = CASES[name]
    options = dict(options or {})
    if not set(options) <= set(inspect.signature(function).parameters) - {"seed"}:
        raise ValueError("unsupported options for " + name)
    if name == "w8a8-reference":
        if any(type(value) is not int or not 1 <= value <= 64 for value in options.values()):
            raise ValueError("CPU W8A8 reference caps M/H/F at 64; use device CLI for bigger probes")
    result = function(seed=seed, **options)
    result["implementation"] = str(Path(inspect.getsourcefile(function)).relative_to(Path(__file__).parent))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=["list", "all", *CASES])
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--options", default="{}", help="JSON object, e.g. W8A8 rows/hidden/width (bounded)")
    parser.add_argument("--output", type=Path, help="new JSON file, never overwritten")
    args = parser.parse_args()
    if args.case == "list":
        emit({"cases": list(CASES), "default_backend": "CPU reference/simulation only"}, args.output)
        return
    options = json.loads(args.options)
    if not isinstance(options, dict) or (args.case == "all" and options):
        parser.error("options must be an object; all uses the documented small defaults")
    if args.case == "all":
        value = report("cpu-suite", {"experiments": [run_case(name, args.seed) for name in CASES]},
                       scope="mixed reference math, simulators and tiny actual CPU IO; not device measurements")
    else:
        value = run_case(args.case, args.seed, options)
    emit(value, args.output)


if __name__ == "__main__":
    main()

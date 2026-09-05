# Attention acceleration design

## Current fastest accepted block

The accepted design keeps fused SDPA on Metal and sends the independent fused
MLP branch to ANE. Both read the same normalized hidden state and overlap:

```text
                         +--> Metal QKV -> RoPE -> fused SDPA -> to_out(attn) --+
normalized hidden ------|                                                   +--> add
                         +--> ANE MLP-in -> SwiGLU -> to_out(mlp) ------------+
```

This is faster because neither branch waits for an intermediate produced by the
other device. It also returns only an `M x 3072` ANE partial.

## Why QKV-only ANE is rejected

Moving QKV to ANE makes GPU attention wait for ANE and returns three hidden-size
tensors. With sequence 1045, this is about 19 MB per block per step; at sequence
4117 it is about 76 MB. It removes the overlap that currently produces the
measured speedup.

## Full ANE attention result

A full fixed-shape Core ML attention block would be credible only if Core ML
lowered SDPA to a fast fused implementation. A naively materialized FP16 score
tensor is about 52 MB at sequence 1045 and about 814 MB at sequence 4117 for 24
heads. The public iOS18 SDPA operator was tested against MLX fused SDPA on this
machine:

| Shape `(B,H,S,D)` | MLX/Metal median | Core ML CPU+ANE median | Core ML / MLX |
|---|---:|---:|---:|
| `(1,24,1088,128)` | 2.41 ms | 49.86 ms | 20.7x slower |
| `(1,24,4160,128)` | 31.94 ms | 569.93 ms | 17.8x slower |

The 4160 Core ML result also failed a sampled FP32 reference gate (maximum
absolute error 0.236 across query positions spanning the sliced sequence).
Consequently the full-ANE candidate is rejected for the current runtime. A
future Core ML/macOS release must pass all of these gates before reconsideration:

1. Lower median block latency than Metal attention while Metal computes MLP.
2. End-to-end image latency improvement after conversion, RoPE, output backing,
   synchronization, and residual add.
3. No worse p95/p99 residency spikes than the existing 1024 ANE MLP path.
4. Image agreement at least as good as the accepted INT8 MLP path.

Core ML Tools 8.3 on this machine exposes the iOS18
`scaled_dot_product_attention` MIL operation and a query-slicing graph pass for
long sequences. The repository includes a public-API feasibility benchmark:

```bash
PYTHONPATH=/path/to/mac_local_ai/.deps/coreml python3 \
  scripts/benchmark_coreml_attention.py --sequence 1088 --heads 24 --head-dim 128
```

These probes include Core ML's Python provider path and do not prove ANE
attribution. That provider overhead cannot explain a 17.8-20.7x gap by itself,
and the numerical failure is independently disqualifying. A future production
attempt still needs a multi-input native bridge with caller-owned Q/K/V/output
buffers plus an Instruments trace.

## GPU/ANE head split candidate

Heads are algebraically separable until the output projection. A future
head-sharded Core ML model can compute QKV, RoPE, SDPA and its `to_out` partial
for a subset of heads while Metal computes the remaining heads. This preserves
concurrency, but it competes with the already profitable ANE MLP branch. The
runtime needs a measured per-chip choice among:

- Metal attention + ANE MLP (current production path);
- ANE attention + Metal MLP;
- GPU/ANE attention head split + Metal MLP;
- phase-alternating attention/MLP placement if ANE scheduling can be pipelined.

## Implementation gate

The public enum already exposes `ane-experimental` and
`head-split-experimental`, but both fail closed. A future attention artifact
manifest must declare its operator graph, bucket, head range, precision, output
layout, benchmark chip, latency distribution, and quality result. Only an
accepted registry entry may make `attention=auto` choose it.

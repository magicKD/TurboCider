# Request-local invariant context reuse: opt-in, small measured benefit

> **Invalid for performance analysis:** pre-reboot degraded system state.
> Exact-parity observations may be retained; timing conclusions are superseded
> by `z-image-clean-performance-2026-09-23.json`.

Dependency inspection: `z_context_block` takes caption activations, weights
and caption RoPE only, not sigma or image noise. Previously caption embedding
and two context-refiner blocks were repeated each diffusion step. Added
`TURBOCIDER_Z_CACHE_CONTEXT=1` (benchmark `--cache-context`) to reuse the exact
final caption tensor within one request. The first-step computation remains
inside diffusion timing. No cross-request cache; prompt, LoRA, shape and
weight changes cannot reuse a previous request's tensor. Quantized, hybrid
and weight-streaming paths are excluded. Default remains off.

This is exact invariant-work elimination, not timestep approximation, new
quantization or fewer image denoising layers. Image/noise-dependent refinement
and the main thirty layers still run each step. RoPE was not cached here.

Full native-only build SHA256:
`6940a8dfedb5bc71c18d154b041bfbf583ea1de4c33c26b782aca4366922d7ec`.
Same-library ABBA,512x512,8steps,BF16,pureGPU,greenhouse seed123; four warm
samples per arm. Cold and separate parity runs excluded, no concurrent GPU
experiment. Actual loaded runtime fingerprints match.

- Default samples:14.789954,14.753983,14.730934875,14.908000750 s.
- Reuse samples:14.671949375,14.604932125,14.688995083,14.716957083 s.
- Diffusion medians14.771968500 ->14.680472229 s (1.0062325x).
- Request wall medians20.046720688 ->19.956767645 s (1.0045074x).

All four parity requests have identical eleven tensor files and PNG; the
completed-run auditor passed.14 CPU benchmark/audit tests passed, including
candidate environment isolation and rejecting ignored worker flags.
`git diff --check` passed. No application package built.

The small~0.09-second benefit on one short prompt does not qualify automatic
enablement or explain the historical regression. Remains opt-in pending
more prompts/shapes and repeated qualification. No1024 measurement of this
candidate and no claim toward512<10s. Earlier cumulative speedup qualifications
belong to their recorded prior library; they are not relabeled as this build.

Raw reports: `/private/tmp/z-image-context-cache-512-abba`.
Exec session91953 completed exit0.

# Qwen Image 2.1: layer streaming and hybrid inference on M5 Pro / 24 GiB

This adaptation is enabled only when the native Metal device name is exactly
`Apple M5 Pro` and physical memory is exactly `24 * 1024^3` bytes. Other chips,
other memory capacities, and unavailable device information retain the upstream
execution and memory estimates. The hardware check is independent of request
JSON and user profiles. Existing ANE routes on other devices remain available;
this change does not qualify them for the new low-memory strategy.

## Scope and controls

| Route | Qualified request |
| --- | --- |
| BF16 GPU | Qwen Image 2.1, text-to-image, canvas at most 512×512, `component_staged`, no reference image, LoRA, prompt enhancement, or approximation |
| Experimental hybrid | Same device, 512×512 text-to-image, at least two steps, `component_staged`, explicit `gpu_ane`, `qwen21_w8a8=true`, `allow_approximation=true`, `auto`/`base_fused` MLP mode, checkpoint-matched 32-block W8A8 manifest with 6144/12288 MLP channels; no reference image, LoRA, prompt enhancement, GPU W8A16 suffix, DiT cache, or diagnostic reuse cache |

The App exposes **Qwen 低内存 ANE（实验）** under acceleration settings. The
control and request submission both check hardware eligibility. A saved draft
from this machine cannot activate this experiment on another device; the user
can turn it off and keep the existing GPU or ordinary ANE route. The flag survives
draft persistence and history reuse. Experimental App submissions select
`base_fused` explicitly so a reused request cannot carry a different hybrid graph. GPU remains the default.

The App's staged-release residency label describes the lifetime of complete
components. Within that mode, eligible requests also stream individual layers.
No separate public streaming selector or preset is introduced.

## Memory and execution

1. Evaluate language-encoder layers individually and discard consumed weights.
2. Open DiT weights lazily through a pinned source descriptor. Read, evaluate,
   and release each layer on each sampling pass. Retain owned prefix KV rows,
   not views that hold the full target K/V arrays alive. Drop compiled functions
   that captured consumed layer weights before reloading the next pass.
3. For hybrid decoding, use the existing Core ML W8A8 prefix and build a compact
   **4.5 GiB BF16 GPU suffix cache**, one layer at a time. Subsequent steps reload
   attention and other weights without rereading the full FFN tensors.
4. During qualified hybrid denoising, temporarily request a **6 GiB MLX wired
   limit**. Preserve a higher existing limit. Synchronize, release the suffix
   cache, and restore the prior limit before VAE decoding and on cancellation or
   failure. No persistent system memory or power setting is changed.
5. Load the VAE after releasing the DiT working set. Release image weights at
   request completion. Request-owned caches do not leak across generation jobs.

The GPU estimate is 18.5 GiB at 512×512; the hybrid estimate is 19.5 GiB. The
existing additional 4 GiB system reserve remains in effect. These conservative
estimates are admission heuristics, not hard caps or whole-machine measurements.
Both the planner and executor consult the same hardware-gated route predicates.
Only the qualified route receives the suffix cache or temporary wired policy.

The `dev-verify` runtime-weight FFN/QKV modes keep their existing additional
2 GiB estimate. LoRA fused/gate-up/suffix graphs and their diagnostics do not
receive the new low-memory estimate. Upstream staged cleanup and LoRA state
reset continue to apply to all devices. The new layer-streaming/cache/wired
strategy remains restricted to the qualified hardware and base graph. The newer
DiT cache presets keep upstream estimates and cannot combine with this low-memory
hybrid experiment. Explicit per-request cache `off` still overrides diagnostic
environment defaults. Ordinary LoRA sampling and edit-prefix snapshots retain
their upstream behavior.

The shared fd-backed reader also needs a correctness fix: duplicated descriptors
share a kernel cursor. Each reader now owns a logical cursor and uses positional
reads, so repeated or concurrent header reads cannot corrupt one another. This
fix applies to all users of that reader; it does not enable an optimization on
additional devices.

The held checkpoint identity is revalidated between sampling passes. Reusing a
prepared Core ML session after the checkpoint changes forces a rebuild and SHA
verification, preventing mixed GPU and Core ML sources.

## Recorded local validation

Hardware: Apple M5 Pro, 24 GiB, macOS 26.4.1, MLX 0.32.0. Tests used external power
with the lid open. Sleep-interrupted measurements were excluded.

512×512, 40 steps, fox prompt and seed 42:

| Execution context | BF16 GPU | Hybrid | Elapsed reduction |
| --- | ---: | ---: | ---: |
| Native repeated requests | 70.478 / 75.155 s | 56.550 / 54.067 s | 24.0% using means |
| Installed App, text-cache hit | 86.185 s | 76.691 s | 11.0% |
| Uncalibrated teapot prompt, seed 17, native | 74.239 s | 57.748 s | 22.2% |

The App's first hybrid image took 111.942 s: text encoding 31.143 s, Core ML setup
18.001 s, denoising 60.132 s. It is not comparable to the warm GPU sample.
App warm denoising decreased from 84.488 s to 67.084 s (20.6%). App/native timing
differences are not fully explained; these few samples do not guarantee a speedup
under other loads. The installed code sections matched the tested native build.

A naive hybrid implementation that rebuilt the GPU suffix every step took
99.643 s and was rejected. A compact cache without temporary wired residency
also had unstable 74–77 s repeats. The final cache/residency combination avoids
that measured regression without skipping denoising steps or reusing FFN outputs.

Final hybrid MLX peak was 5.67 GiB and measured native process footprint peaked
at approximately 6.53–6.56 GiB. MLX accounting excludes other Core ML/OS memory;
neither value measures whole-machine use. After completion or tested cancellation,
MLX active memory returned to approximately 165 KiB.

All full hybrid runs made 1248 runtime Core ML calls, with no failures. The
32-block compute plan preferred the Neural Engine for its 384 nonconstant
operations, but actual runtime ANE residency remains **unknown**. The App and
result JSON retain this distinction.

## Quality and safety of reuse

- GPU streaming output matched the existing BF16 baseline byte for byte.
- The final hybrid cache and wired variants, including both App images, matched
  the initial W8A8 hybrid image byte for byte. Caching adds no numerical change.
- W8A8 itself is an approximation: fox RGB SSIM 0.979449 / PSNR 34.947 dB against
  BF16; held-out teapot SSIM 0.882878 / PSNR 23.699 dB. The teapot's handle, body,
  and spout visibly changed. This is not lossless acceleration or broad quality
  qualification, hence the explicit experimental opt-in.
- Cancellation during Core ML loading and after suffix-cache construction,
  retry/repeat, GPU fallback, and nonzero prior wired-limit restoration were
  exercised. A changed checkpoint after prepare was rejected before inference.

The 32-block artifact is local and not bundled in this PR. Users need to export
and compile a partition matching their checkpoint and the geometry above. Model
weights, compiled artifacts, personal paths, raw images, and local machine
configuration are excluded from the change.

## PR build verification

After integrating `dev-verify` at `975aa91`, a clean build with the exact hardware gate passed
43 focused Qwen tests (two optional installed-adapter audits skipped), 39 Qwen
request-contract tests, the device-profile contract, CPU policy/boundary checks,
the source-lease fixture, App workflow checks, and ANE library checks. App reuse
checks cover the newer hybrid modes and explicit `base_fused` resubmission.
The device matrix uses synthetic hardware descriptions; physical inference was
run on the qualified machine only. It covers M5/M5 Max, M4 Pro/Max, other M5 Pro
memory capacities, off-by-one byte counts, unknown devices, and malformed App
hardware descriptions.

The `dev-verify` integration build's full 512×512 / 40-step `base_fused` hybrid
run completed in 56.591 s with 1248 Core ML runtime calls, 5.67 GiB MLX peak,
and approximately 161 KiB MLX active after completion. Its PNG matched the previously qualified hybrid byte for byte.
This is a functional/parity acceptance run, not another paired speed comparison.
The gated build also passed cancellation during load and denoising, retry,
repeat, and GPU switch checks. Every exit restored the pre-existing 256 MiB
wired limit, and post-request MLX active memory remained below 2 MiB.

## Reproduction

Build using the repository's managed dependencies and native build workflow.
Focused checks cover the CPU hardware matrix, native request contracts, streamed
BF16 parity and cancellation, repeated source-lease reads, App hardware admission,
draft/history persistence, and Qwen/Z/FLUX manifest registration.

An explicit hybrid request uses:

```json
{
  "model": "qwen-image-2.1",
  "operation": "image.generate",
  "prompt": "A red fox standing in fresh snow, soft morning light.",
  "width": 512,
  "height": 512,
  "steps": 40,
  "seed": 42,
  "frames": 1,
  "audio": false,
  "execution": "gpu_ane",
  "residency": "component_staged",
  "qwen21_w8a8": true,
  "hybrid_mlp_mode": "base_fused",
  "qwen21_dit_cache": "off",
  "allow_approximation": true,
  "ane_manifest": "/path/to/matching-compiled-manifest.json"
}
```

Run `tools/native/benchmark_native.py` with the built native library, local model
root, this request, and explicit `--width 512 --height 512 --steps 40 --seed 42`.
For the BF16 control, use `execution=gpu` and remove `ane_manifest`,
`qwen21_w8a8`, and `allow_approximation`. Compare full request time, stage times,
image quality, and memory separately; do not infer actual ANE use from selection.

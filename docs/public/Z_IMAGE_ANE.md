# Z-Image Turbo GPU/ANE acceleration (September 25, 2026)

The fastest **measured 512×512, 8-step** path on this checkout is an explicit,
compiled, checkpoint-bound W8A8 Core ML manifest with **1,024 image tokens and
5,120 FFN channels** on the ANE-preferred branch. The other 5,120 image-token
channels and the *entire* caption FFN stay BF16 on GPU. All 32 FFN blocks run
both branches in parallel on each of eight denoising steps: 256 Core ML calls
per image. Attention, modulation, normalization, VAE and text encoding are not
quantized by this profile. The split is over the 10,240-dimensional SwiGLU
**intermediate channels**, not the 3,840-dimensional input embedding.

The source manifest must mark `export_identity.image_only_token_rows=1024` and
be compiled for the actual BF16 transformer checkpoint. The optimized GPU
QKV and gate/FFN-input normalization kernels automatically apply only to the
qualified short-row M4 Max path (at most 1,056 total tokens); longer captions
retain the regular GPU preparation. The named profile is **explicit**:
`execution=auto` keeps the existing measured 4,096-channel policy and must not
silently select this 5,120-channel image-only experiment. Other valid ANE
models (including W8A16 and routed W8A8 partitions) remain available through
their own manifests and the same generic GPU/ANE request contract. No artifact
filename, user cache path or ANE channel count is baked into the CLI.

## Measured performance and limits

On one M4 Max with 64 GB unified memory, resident BF16 Comfy Z-Image Turbo,
fox prompt/seed 42, 512² and eight steps, a same-library fresh-process
GPU/hybrid/hybrid/GPU crossover (one excluded cold plus one warm request per
process) with the **regular `build/native` library** measured
**6.8256 → 5.1466 seconds warm denoising (1.326×)** and
**7.0688 → 5.3934 seconds warm request wall (1.311×)**. Both arms used library
SHA256 `68baf7174f43fb8b684a785e5f61ec900f807f83caba108896959080dcc1719c`.
The two GPU PNGs matched each other; the two hybrid PNGs also matched each
other and the CLI smoke output. Both hybrid processes completed 256 Core ML
FFN calls per request (512 including each excluded cold request), with no
reported output copies or runtime failures. The short-prompt system-health
anchor passed. Raw local reports:
`outputs/z-w8a8-final-build-abba-20260925/`. An earlier **isolated** library,
SHA256 `c024767fe6eb8620bcb0600ac37fc72f0f3daf0d85c50fefd6b4f72da0f62653`,
measured 7.0803 → 5.3887 seconds warm wall (1.314×); do not mix libraries
when computing a speedup.
These numbers do **not** describe cold load/compile, other chips, long prompts
or arbitrary step counts. The CLI smoke test with this library reproduced the
warm benchmark's hybrid PNG SHA256
`7109f36e7b110c70125f5e9a39d54aa2c2bb9cf05ba5e46d443854625a6ab33a`;
that **cold** invocation took 5.9671 seconds of denoising and 18.9363 seconds
request wall including load. Do not present it as a warm speed result.

For comparison, an **earlier isolated build** ran a same-library, matched
fox/42 crossover of the previous 4,096-channel W8A16 hybrid (5.603 s warm
denoise), a routed 4,608-channel W8A8/BF16-GPU hybrid (5.453 s), and the
present image-only 5,120-channel W8A8/BF16-GPU route (5.270 s). The old
W8A16 hybrid was therefore about 6.3% slower than this candidate in that
comparison. These three values are from the *same earlier library*, not the
regular-build run above, and are not universal ANE-model rankings. A simpler
same-width W8A8 single-layer graph ran about 11.6% faster than W8A16 in
Core ML probes but had severe image-row error; the more accurate grouped A8
graph actually ran about 6.7% slower per prediction. The whole-image gain
comes from a balanced GPU/ANE partition and BF16 GPU complement, not a
general rule that INT8 is faster in every compiled graph.

Several paired prompts (fox, transparent lighthouse, teapot, alley, violinist,
colored objects, a watch and typography) retained usable prompt-level
structure when inspected by eye, while details differ from BF16. This is not
an acceptance test over arbitrary prompts. The Core ML plan prefers ANE and
all FFN calls complete, but a physical trace of INT8 ANE arithmetic has not
been obtained. Longer captions can make the full BF16 GPU caption FFN critical:
the 512-caption-row performance screen was roughly tied with GPU and is not
a qualified speedup.

A **single-block-only**, native GPU→Core ML bridge probe for 1,536 captured
rows measured 25.10 ms full BF16 GPU, 17.11 ms W8A8 **row** split and
17.43 ms current channel split on held-out watercolor; a second real capture
measured 25.13 / 17.12 / 17.46 ms respectively. Both models returned their
outputs in shared MLX backing without reported copies. On the second capture,
the row split's image-token relative L2 was 0.0850 versus 0.0634 for the
channel split, so a small speed lead does not automatically pass image quality.
This small row advantage does not include all-layer residency, preceding GPU
attention, full-image performance or generated-image quality. Full-width
row-split ANE weights are larger; row artifacts have **not** replaced the
32-block 5,120-channel candidate. The repository-local research log
`docs/status/z-image-w8a8-research.md` retains the other prompt screens,
hashes and negative optimization probes.

## Use the fastest tested manifest from the CLI

The new `--ane-manifest` option works for `plan`, `generate` and `batch` and
accepts **any** compatible compiled ANE manifest. It transiently sets
`gpu_ane` and `allow_approximation=true` for each request; it never edits the
JSON file and rejects a conflicting manifest already specified there. For
the fastest tested Z-Image route, supply the locally compiled manifest for
the 1,024-image-row / 5,120-channel W8A8 profile:

```sh
MANIFEST=models/coreml/z_image_w8a8_imageonly_hidden_c4_a5120_b1024_diverse8_compiled/manifest-HASH.json
MODEL=models/Comfy-Org-z_image_turbo
build/native/turbocider plan examples/requests/z-image-turbo-512.json --ane-manifest "$MANIFEST"
build/native/turbocider generate "$MODEL" examples/requests/z-image-turbo-512.json --ane-manifest "$MANIFEST"
```

Use a **unique output path** in the request, and use a CLI and native library
built together from this checkout (`make build`; `make package` for `dist/cli`).
Replace `manifest-HASH.json` with the actual compiled manifest filename and
adjust the local model directory as needed. The sample manifest path is
intentionally not auto-discovered: a similarly
named but different checkpoint, Core ML build or shape is unsafe. No automatic
fallback to GPU occurs on an explicit incompatible manifest. To use a different
ANE accelerated model, pass its compiled manifest to the same option with its
own request JSON, or specify `gpu_ane`, `ane_manifest` and
`allow_approximation=true` directly in the request. Schema 1 and schema 2
requests both remain supported; model-specific compatibility checks and
checkpoint provenance remain in the native runtime.

## How the current path is wired

`tools/coreml/export_z_image.py` exports the checkpoint-bound per-block FFN
branches; `tools/coreml/compile_z_image_manifest.py` compiles and records
their identities. The native Core ML backend validates the selected manifest
and loads its 32 branches. The Z-Image executor in
`native/models/z_image/z_image.cpp` checks the checkpoint and geometry,
prepares the GPU BF16 complement, launches both FFN branches, and joins their
results before continuing denoising. `apps/cli/main.mm` only supplies an
explicit manifest and approximation consent to the shared request API; it
does not perform model conversion or select a profile by its filename. This
also preserves existing model-specific ANE manifests rather than requiring a
Z-Image-only CLI path.

To compare actual warm latency, send repeated requests to a resident engine
(for example `batch`); `plan` and a single cold `generate` do not measure the
published warm-request speedup.

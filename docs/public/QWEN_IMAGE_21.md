# Qwen-Image-2.1 in TurboCider

TurboCider runs Comfy-Org/Qwen-Image-2.1 natively through C++/MLX/Metal. Python
is used only for offline validators and development diagnostics; inference is
performed by the native runtime.

## Supported App/CLI paths

- Text-to-image and image editing with the BF16 Comfy checkpoint.
- RGBA generation, transparent subject extraction, and transparent-layer edits.
- One to ten ordered image references. References are encoded as visual
  conditions; a separate black/white mask is an additional visual reference,
  not hard pixel-preserving inpainting.
- Ellipse/brush annotation references, request-owned prefix KV caching, staged
  or resident GPU execution, and native App job persistence/reuse.
- Experimental PE-T2I and explicitly opt-in PE-I2I FP32 visual conditioning.

The App exposes Qwen-specific examples, ten-reference import limits, RGBA
handling, mask/annotation authoring, and the PE-I2I warning. Use the GPU route
for normal operation. The experimental GPU+ANE route requires an explicit
manifest and approximation opt-in; it is not hardware-placement or image-quality
proof.

## Explicit 512² GPU/Core ML experiments

The default remains BF16 GPU, with references resized to approximately 1024
pixels. There are two opt-in mixed routes: a 32-layer FP16 Core ML FFN prefix
for 512² text-to-image, and a W8A8 Core ML prefix for 512² text-to-image or
1–3-reference editing. The measured, faster W8A8 editing candidate keeps a
BF16 GPU FFN suffix, resizes each reference to approximately 256 pixels, and
runs layers 3, 5 and 7 entirely on GPU. Its 29/32 decode FFN coverage is
90.625%. These reference dimensions change the conditioning input and can lose
details; this mode is **not** the default or a production image-quality gate.
The resident Session now caches the text/visual conditioning and reference VAE
latents for an unchanged prompt, reference resize and ordered reference file
contents (checked by SHA-256 on every request). Denoising prefix KV remains
request-owned. Replacing a reference in place invalidates the cache; staged or
one-shot requests cannot claim this resident hit. Under the matched 512²/40-step
two-reference, seed-17 workload, two cached GPU requests took 44.733/44.777 s
and W8A8 mixed requests took 37.159/37.176 s, about **1.204×** request-wall
speedup. At 5 steps the corresponding cached figures were 6.688/6.727 s and
6.012/5.999 s, about **1.117×**. These numbers differ from the earlier
re-encode-on-every-request result (40-step 1.130×, 5-step 0.835×), and apply
only to repeated identical editing conditions after Session preparation. They
do not establish a benefit for a new prompt/reference, cold process, other
seeds, or 1024² requests; GPU stays the default.
Separate 40-step cached resident samples with one and three 256px references
measured about **1.209×** and **1.198×** request-wall speedup respectively;
all three reference subjects remained visible in the tested three-image output.
Each reference count still has only one prompt/seed and requires broader visual
validation. Earlier three-reference tests with the default 1024px reference
resize were slower on the mixed route; these are different inputs.

An additional explicitly selected **6144-channel W8A8** partition was
measured on the same 512²/40-step prepared, condition-cached 1/2/3-reference
workloads. Its request-wall speedups over matched GPU were **1.357× / 1.348× /
1.338×**, compared with **1.209× / 1.204× / 1.198×** for the established
4096 partition. For the same two-reference input at 5 steps, a new
condition-cached comparison measured GPU 6.706/6.707 s, 4096 W8A8
6.003/5.996 s, and 6144 W8A8 5.560/5.543 s (6144: **1.208×** request-wall
vs GPU). The earlier 5-step 0.835× test had `prompt_cache_hit=false`;
it does not describe this repeated-input cache-hit scenario.

To opt into the faster candidate, point the example's `ane_manifest` at a
checkpoint-matched 1024-row, 32-block, 6144-channel compiled manifest; keep
`qwen21_gpu_full_ffn_blocks: [3, 5, 7]` and the BF16 GPU suffix. No global
manifest path is bundled: a local model must first be explicitly exported and
compiled from the same checkpoint. The 4096 manifest remains available if
composition fidelity matters: the 6144 two-reference result has visibly
shifted composition (RGB correlation 0.928 to GPU, versus 0.996 for 4096).
Single- and three-reference outputs also require broader quality checks.
Neither opt-in establishes physical ANE occupancy; GPU remains the default.

For the explicit W8A8 editing candidate, use schema 1 fields like these with a
compiled 1024-row, 32-layer, checkpoint-matched, per-tensor W8A8 manifest:

```json
{
  "model": "qwen-image-2.1",
  "operation": "image.edit",
  "prompt": "Compose these two referenced objects in one scene.",
  "output": "results/edit.png",
  "width": 512, "height": 512, "steps": 40, "seed": 17,
  "frames": 1, "audio": false,
  "execution": "gpu_ane", "allow_approximation": true,
  "ane_manifest": "path/to/compiled/manifest-HASH.json",
  "qwen21_w8a8": true,
  "qwen21_reference_size": 256,
  "qwen21_gpu_full_ffn_blocks": [3, 5, 7],
  "inputs": [
    {"kind": "image", "role": "reference", "path": "path/to/first.png"},
    {"kind": "image", "role": "reference", "path": "path/to/second.png"}
  ]
}
```

Run `build/native/turbocider plan request.json` to check the policy and
`build/native/turbocider generate path/to/model request.json` to verify the
manifest and generate an image. Schema 2 puts the W8A8 and fallback flags
under `execution`, and `qwen21_reference_size` under `parameters`. W8A16 GPU
suffix is an additional explicit research option (`qwen21_gpu_w8a16: true`),
but tested slower than the BF16 suffix; do not enable it for speed by default.
The 1024² W8A8 text-to-image results remain diagnostic-only and are not
accepted by the public CLI hybrid policy. See the
[512²](../status/qwen21-w8a8-ane-diagnostics-2026-09-25.md) and
[1024²](../status/qwen21-w8a8-ane-1024-diagnostics.md) reports for timing
scope, visual differences, and remaining validation work.

## 512² validation snapshot

The maintained regression scope is 512². Evidence includes native GPU text
generation. App model selection and prompt examples default to 512×512;
reference encoding geometry is independent of that output canvas. Coverage includes
RGBA export, prefix-cache/session lifecycle, one-step progress
events (`0/steps` through `steps/steps`), and App transport/workflow tests.
The matched 512² GPU comparison against mflux is recorded in
`results/qwen21/matched-refresh-mflux-512-report.json`; native cached denoise
was measured at approximately 1.068× the mflux denoise speed in that matched
workload, with latent relative RMSE 0.007073 and RGB correlation 0.999890.

Additional completed samples exercise 1024² edits, transparent extraction,
annotation/mask edits, three-reference composition, and the ten-reference
runtime boundary. They are representative evidence, not a blanket perceptual
quality gate.

## Correctness and limitations

Run `make test-qwen21` after building the native/App targets. The target runs
focused report, mask-analysis, native contract, and App workflow checks without
starting inference. Full reports and experiment history are in
`docs/status/qwen21-convergence-2026-09-22.md` and the local
`notes/2026-09-21-qwen-image-21.md`. NumPy is an offline-test dependency;
Pillow is only needed for the image-report CLIs, not these unit tests.

Known limitations are intentional and explicit: BF16 Qwen3.5 visual parity
does not pass the strict reference gate, PE-I2I remains experimental FP32,
edit preservation is not pixel-locked, JPEG decoder parity has small byte
differences, and actual ANE occupancy cannot be proven without privileged
hardware tracing. The 2K path is exposed by the model contract but is outside
the maintained test scope; do not use its incomplete experiment as evidence
for production quality or performance.

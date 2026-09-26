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
1–3-reference editing. The W8A8 editing candidates keep a BF16 GPU FFN
suffix and resize each reference to approximately 256 pixels. The
speed-first 6144-channel candidate uses all 32/32 FFN layers in parallel;
the alternative runs layers 3, 5 and 7 entirely on GPU and covers 29/32
decode FFNs (90.625%). These reference dimensions change the input and can lose
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

An explicitly selected **6144-channel W8A8 with GPU fallback 3/5/7** was
measured on the same 512²/40-step prepared, condition-cached 1/2/3-reference
workloads. Its request-wall speedups over matched GPU were **1.357× / 1.348× /
1.338×**, compared with **1.209× / 1.204× / 1.198×** for the established
4096 partition. For the same two-reference input at 5 steps, a new
condition-cached comparison measured GPU 6.706/6.707 s, 4096 W8A8
6.003/5.996 s, and 6144 W8A8 5.560/5.543 s (6144: **1.208×** request-wall
vs GPU). The earlier 5-step 0.835× test had `prompt_cache_hit=false`;
it does not describe this repeated-input cache-hit scenario.

The faster explicit 6144-channel **full 32-layer edit** achieved about
**1.410× / 1.400× / 1.387×** request-wall speedup for 1/2/3 references at
40 steps; two-reference 5-step cached requests achieved **1.234×**. To use
it, point `ane_manifest` at a checkpoint-matched 1024-row, 32-block,
6144-channel compiled manifest and leave `qwen21_gpu_full_ffn_blocks` empty.
No global manifest path is bundled: a local model must first be exported and
compiled from the same checkpoint. Retain `[3, 5, 7]` with either the 4096
or 6144 manifest when reference details matter: on this three-reference
sample the full-coverage route altered the left teapot's spout/handle more
than the fallback. On the two-reference sample, however, full coverage was
closer to GPU (RGB correlation 0.944 versus 0.928 for fallback). Neither
route establishes physical ANE occupancy or general image fidelity; GPU
remains the default. See the
[full-coverage edit comparison](../status/qwen21-w8a8-full32-edit.md)
for timings and visual caveats.

For explicit **512² text-to-image** with a checkpoint-matched 6144-channel
W8A8 manifest, use `operation: "image.generate"`, no `inputs`,
`qwen21_w8a8: true`, `execution: "gpu_ane"`, and
`allow_approximation: true`. Leave `qwen21_gpu_full_ffn_blocks` empty for the
32/32-layer route. A single fox prompt/seed in a prepared, prompt-cached
resident Session measured GPU **5.999/5.997 s** versus W8A8
**5.042/4.820 s** at 5 steps (about **1.216×** request-wall), and
**43.184/43.223 s** versus **30.468/30.489 s** at 40 steps
(about **1.418×**). The 40-step fox remains recognizable, with differences
around its legs and tail; an independent glass-bottle-and-cactus prompt/seed
also gave visually similar results in a diagnostic generator. These are
limited samples, not a default quality or cold-start speed guarantee; the
[text-to-image evidence](../status/qwen21-w8a8-6144-t2i.md) distinguishes
per-step timing from complete resident and first requests.

For the speed-first explicit W8A8 edit candidate, use schema 1 fields like
these with a compiled 1024-row, 32-layer, checkpoint-matched, per-tensor
**6144-channel** W8A8 manifest:

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
  "qwen21_gpu_full_ffn_blocks": [],
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
For the conservative W8A8 editing option, set `qwen21_gpu_full_ffn_blocks`
to `[3, 5, 7]`; a 4096-channel edit manifest **requires** this fallback at
runtime even though `plan` does not load or inspect the manifest.
The 1024² W8A8 text-to-image results remain diagnostic-only. A local
`TURBOCIDER_QWEN21_1024_W8A8_DIAGNOSTIC=1` opt-in now permits explicit
1024² W8A8 text-to-image with a checkpoint-matched 4096-row/4096-channel
compiled manifest for resident Session benchmarking, but does not qualify
1024² editing or change the default GPU policy. The measured warm-request
speedups are 1.114× at 5 steps and 1.159× at 40 steps on one prompt/seed;
physical ANE placement and broad image quality remain unverified. See the
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

# Qwen Image 2.1: late-block reference-local attention for full-size edits

An **opt-in, lossy** prefill route now accepts
`TURBOCIDER_QWEN21_REF_LOCAL_ATTENTION=3` for a 512² edit with two or three
original-size 1024px references, explicit GPU or W8A8 GPU/Core ML,
`allow_approximation=true` and originally no LoRA. The first 16 transformer blocks
retain all prior-reference K/V dependencies. Only in blocks 16–31, later
reference queries use earlier *text* and their own reference K/V; they skip
earlier reference-image K/V. Target queries still attend to the complete
prefix, and all K/V needed for decode are cached. This changes conditioning
and **is not** a mathematically exact attention kernel improvement. Existing
mode `=1` (all later references in all blocks) and `=2` (last reference in
all blocks) remain unchanged. The one-reference request has no later
reference to optimize, and is intentionally not admitted for this mode.

For the W8A8 route, use the existing
`TURBOCIDER_QWEN21_FULL_REF_W8A8_DIAGNOSTIC=1` and the same checkpoint-bound
32-layer, 1024-row/6144-channel manifest. Every decode layer continues to
split its FFN between a W8A8 Core ML prefix and a BF16 GPU suffix: **128
Core ML calls per five-step request**, unchanged by local attention. This
does not offload the long first-step FFN and does not prove physical ANE
execution. Hybrid W8A16 suffix and GPU fallback blocks are not admitted.

## Matched warm measurements

M4 Max, BF16 checkpoint, identical prompts/seeds/ordered reference files
within each pair, independently prepared resident Sessions and cached
conditioning. The verifier checked actual byte-equal text, initial noise
and every reference latent tensor. Warm request wall includes VAE decode,
PNG and matched tensor dumps but excludes model/manifest preparation.

| 512² edit | Exact GPU | Late-local GPU | Same-input ratio |
| --- | ---: | ---: | ---: |
| Two original-size references, 5 steps, seed 17 | 18.080 s | 17.764 s | 1.018× |
| Three original-size references, 5 steps, seed 17 | 25.240 / 25.203 s | 24.167 / 24.137 s | median **1.044×** |
| Three original-size references, 5 steps, seed 29 | 25.183 s | 24.136 s | 1.043× |
| Three original-size references, 40 steps, seed 17 | 81.855 s | 80.969 s | 1.011×, one run |

The seed-17 three-reference pairs used equal **five-step** warmups; seed
29 likewise. The two-reference and 40-step screens used equal two-step
warmups. These are sequential pairs, not ABBA distributions. The
five-step normalized RGB correlation to GPU was 0.99983 for two references,
0.99896/0.99905 for three references at seeds 17/29; at 40 steps it was
0.99872. Those are diagnostic pixel metrics, not subject-fidelity scores.
Visual inspection found both pots and the orange sticker recognizable in
the three-reference outputs, with minor sticker-eye, pot and lighting
differences; the original five-step GPU images already ghost the sticker.
This is a plausible research speed/quality tradeoff for these fixtures, not
an automatic preset or guarantee for other reference styles. A five-step
single-reference edit and text-to-image are unaffected or unqualified;
do not claim those workloads accelerate. A later six-step Viggle LoRA
diagnostic also admits 2–3 references under the same `=3` locality;
hybrid additionally requires the separate base-ANE LoRA diagnostic flag.
Its timings and visual checks are recorded in
[LoRA/base ANE 诊断](qwen21-viggle-base-ane-reuse-2026-09-27.md).

Existing target-only final prefill block
(`TURBOCIDER_QWEN21_PREFILL_LAST_TARGET_ONLY_DIAGNOSTIC=1`) combined with
the new GPU mode gave **23.818/23.841 s** on the seed-17 three-reference
five-step request, or 1.058× versus exact GPU. Both candidate PNGs were
byte-identical to local mode alone, so this separately gated shortcut
added no measured image loss on this fixture. Combining the two switches
with fused Q/K norm-RoPE instead gave **23.420/23.270 s** (about 1.080×
versus exact GPU), but visibly blurred/doubled sticker features. Keep the
Q/K fusion **off** in a quality-focused full-reference edit; the extra
~0.5 s does not justify that change on the inspected sample.

| W8A8 + BF16 GPU suffix, 5 steps | Decode-only hybrid | Late-local hybrid | Late-local + target-only |
| --- | ---: | ---: | ---: |
| Two full-size refs, seed 17, two-step warmup | 16.975 s | 16.643 s | not run |
| Three full-size refs, seed 17, five-step warmup | 24.226 s | 23.140 s (verified) | **22.798/22.803 s** |

An earlier 23.185/23.196 s late-local hybrid pair used a **two-step**
warmup, so it is not included in the matched-full-warmup row. Its PNG
matched the later verified late-local run, but the pilot exited 1 during
the probe's subsequent route-switch check, as explained below.

The three-reference combined route is about **1.063×** faster than its
decode-only hybrid control, and about **1.106×** faster than the matched
25.240/25.203 s exact GPU control, but it uses W8A8 and reference-local
attention: this is **not** a kernel-only GPU gain, an ANE-residency
measurement, or a general 1.1× reference-edit guarantee. Its PNG is
byte-identical to late-local hybrid without target-only; RGB correlation
against the decode-only hybrid was 0.99854. The hybrid two-reference
late-local pair had correlation 0.99945. Both retain the requested objects
on visual inspection, though the base model itself is imperfect on
sticker details. Core ML prediction count was 128 with zero reported
runtime failures in the verified three-reference run.

The first hybrid pilot exported valid measured PNGs but **exited 1 in the
probe's new switch-back check**: that check accidentally reused a hybrid
request after preparing GPU. We corrected the probe to clear the W8A8
options when switching, rebuilt it and reran an independent five-step
three-reference hybrid Session. Its cancellation, plain-GPU switch and
unload all passed; the switched GPU PNG matched the historical exact-GPU
baseline byte-for-byte. The final combined hybrid probe passed the same
lifecycle. `make test-qwen21` (27 + 7 + 22 contracts), native library/probe
build, CLI `plan` label and `git diff --check` passed. Model-level quality
for other prompts/seeds, the physical device executing Core ML and 40-step
hybrid performance remain unproven.
An actual cold `turbocider generate` completed with all three flags, reported
GPU/Core ML execution and five denoise steps, and wrote a PNG byte-identical
to the resident combined candidate. Its **41.839 s** request wall includes
model/manifest loading and encoding, and must not be compared to the
22.80 s warmed request.

To reproduce the **explicit** three-reference quality-focused mixed route,
use a request with 512² output, five steps, full-size references, W8A8
manifest, `execution=gpu_ane` and approximation opt-in:

```sh
TURBOCIDER_QWEN21_FULL_REF_W8A8_DIAGNOSTIC=1 \
TURBOCIDER_QWEN21_REF_LOCAL_ATTENTION=3 \
TURBOCIDER_QWEN21_PREFILL_LAST_TARGET_ONLY_DIAGNOSTIC=1 \
build/native/turbocider plan EDIT_REQUEST.json
```

Then use the same switches with `generate MODEL_DIRECTORY EDIT_REQUEST.json`.
The `plan` labels the locality and target-only approximations. For the
lower-loss original hybrid, leave both switches off. All PNGs, tensor
dumps, comparison JSONs and model artifacts remain ignored under
`results/qwen21/session-fullref-edit{2,3}-last16-local-*20260927*/`,
`results/qwen21/session-fullref-edit{2,3}-hybrid-last16-local-*20260927*/`
and similarly named `*-vs-control-20260927*.json` receipts.

# Qwen Image 2.1: target-only final prefill block (opt-in screen)

The final transformer block's first-step output is consumed only for the
1,024 output-image tokens. Text and reference *K/V* are still required for
target attention and subsequent decode steps, but the final block's text and
reference attention outputs, output projection, and FFN outputs were formerly
computed and discarded. The explicit diagnostic
`TURBOCIDER_QWEN21_PREFILL_LAST_TARGET_ONLY_DIAGNOSTIC=1` keeps the full Q/K/V
and prefix-KV bank, computes this block's attention and FFN only on target
queries, and reuses the existing 1,024-row W8A8 graph once instead of 3–5
times on eligible hybrid edits. On the unfused pure-GPU path it additionally
projects and rotates Q only for the output-image rows. With fused QKV or
fused Q/K RoPE, it retains the existing full Q projection and slices queries
after projection; there is no assumption that a fused kernel accepts shorter
Q than K/V. Every earlier block and every decode step are unchanged.

The plan gate requires explicit approximation consent, 512² output and no
LoRA. Supported requests are plain GPU text-to-image or 1–3-reference edits
(512px or original 1024px reference encoding), or resized-512 edits with the
existing 6,144-channel W8A8 manifest and `TILED_PREFILL_W8A8_DIAGNOSTIC=16`.
It is **off by default**. Prefix-KV cache keys separate this mode from the
unmodified path. The Session probe turns all experimental switches off
before its final GPU-switch check.

## Matched full-size three-reference GPU screen

M4 Max, 512² output, 3 ordered 1024px references, 13,442 first-step rows,
seed 17, five steps. Independently prepared resident Sessions each used a
**matching five-step warmup** followed by two cached requests with identical
text/noise/reference tensor dumps. Request wall includes VAE, PNG and dumps,
but excludes cold preparation. No fused QKV, Core ML, LoRA, cross-request
prefix-KV reuse or final-step FFN approximation. Both arms passed the
cancellation, GPU-switch and unload probe.

| Path | Request 0 | Request 1 | Output vs GPU control |
| --- | ---: | ---: | --- |
| Original final block | 25.240 s | 25.203 s | — |
| Target-only final block + short Q | 24.796 s | 24.814 s | PNG byte-identical |

Median wall ratio **1.017×**; equality of the dumped request tensors was
verified. This is a small but repeatable saving on these two requests, not a
multi-seed performance guarantee. Before the short-Q change, the same
target-only mode took 24.846/24.894 s, also producing a byte-identical PNG.
The additional ~0.05 s of apparent benefit from shortening Q is too small
to isolate confidently from run-to-run variability. Both new PNG SHA-256
values were `fb9a24d697f2156ecef092d5e7b49638a2c8e7c0b4bd8fe310d1792c4c5f8560`,
identical to the older exact-GPU output. Thus this GPU optimization does
**not** introduce a visible quality difference for this tested request;
neither version fully solves the underlying full-reference edit quality.
Receipts: `results/qwen21/session-fullref-edit3-targetblock-gpu-control-20260927-b/`,
`results/qwen21/session-fullref-edit3-targetblock-shortq-gpu-20260927-c/`,
and `results/qwen21/session-fullref-edit3-targetblock-shortq-gpu-20260927-c-vs-control.json`.

The separate 512² five-step fox text-to-image smoke test was **not faster**:
matched full-step-warmup GPU control 6.002 s versus short-Q target-only
6.022 s (one request). The PNG and dumped text/noise inputs were identical,
and the Session lifecycle passed. This switch should therefore not be
advertised as a text-to-image acceleration. Receipt:
`results/qwen21/session-gpu-fox-t2i-targetblock-shortq-512-5-comparison.json`.

## Existing resized-reference hybrid screen

With explicit `qwen21_reference_size=512`, W8A8/6,144-channel split, final
16 first-step layers tiled, fused QKV and final-step last-16 FFN reuse, the
target-only block removes 2/3/4 Core ML calls per one/two/three-reference
five-step edit respectively. The saved three-reference seed-17 matched
controls measured 7.991/8.001 s without this switch and 7.928/7.919 s with
it (~1.009×); input dump payloads and PNGs were byte-identical between these
hybrid arms. Two-reference timings were 6.735/6.759 → 6.725/6.746 s; one
reference was 5.559/5.590 → 5.550/5.775 s, so **do not claim a one-image
speedup**. Compared with the *same resized inputs*, the existing pure-GPU
three-reference 10.114/10.137 s control versus the target-only hybrid
7.928/7.919 s is roughly 1.28× on this five-step edit. Most of this gain
predates the new target-only block; it does **not** transfer to unresized
references or prove physical ANE residency. The 512px resizing and W8A8/
final-step reuse remain separate lossy quality tradeoffs. The inspected
three-subject composition retains two pots and the orange dragon; the
one-reference matte-finish instruction remains unsatisfied even on GPU.
Evidence: `results/qwen21/session-ref512-edit{1,2,3}-w6144-fusedqkv-last16-finalreuse16-targetblock-20260926-*/`.

For full-size references, prior all-32-layer W8A8 prefill tiling did improve
the five-step prefill but visibly changed the sticker and slowed a 40-step
request; that candidate was removed. The current target-only switch does
**not** enable W8A8 tiling for full-size references. Reusing a 1,024-row ANE
layer in a sequence loop is technically working for resized-512 references,
but doing it blindly over 13,442 rows is not a qualified quality/speed win.
This is also consistent with the `../splash` sequence-row split probe:
single-layer benefits are not sufficient evidence for a full-model win.

Run the explicit GPU route with the normal request JSON and the environment
switch, or combine this switch with the three previously documented hybrid
research switches for eligible resized edits. The CLI `plan` reports the
diagnostic label; a formal `turbocider generate` on the three-full-reference
request succeeded and produced the same exact-GPU PNG. Its **cold** request
wall was 36.054 s including text/reference encoding and model loading, and
must not be compared against the resident warm figures above. Do not
silently make it the default fastest mode for other prompt/seed/device/LoRA
workloads. Local verification: native library rebuild, `make test-qwen21`,
`git diff --check`, byte-equal PNG and tensor-dump comparison, and Session
lifecycle probes passed. The hybrid test was run before the independent
short-Q-only pure-GPU refinement; that refinement is inactive for fused QKV.

An additional fused-QKV kernel screen skipped the final prefill block's
unused prefix-Q tiles while retaining full K/V. At 4,226 rows the isolated
QKV probe measured ~28.73 → 21.61 ms, with relative L2 against the complete
reference below `5.3e-6` on consumed Q rows plus all K/V. On the real
three-reference fast hybrid edit, however, warm request walls were
7.928/7.919 s without skipping and 7.895/7.925 s with it; the PNG SHA was
identical. The opposing repeat directions provide no defensible end-to-end
gain, and the skipped Q output rows would be left unwritten. The extra
kernel specialization and probe changes were reverted. Raw candidate
receipts remain ignored in
`results/qwen21/session-ref512-edit3-fusedqkv-skipq-targetblock-20260927/`.
The rebuilt library passed the contract suite and the post-rollback Session
probe: 188 Core ML calls per measured request, zero runtime failures or
output-copy bytes, 7.945 s warm wall, cancellation/GPU-switch/unload exit 0,
and a PNG byte-identical to the saved pre-screen target-only hybrid.

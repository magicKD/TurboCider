# Runtime fixed-two request comparison — 2026-10-02

Fixed two-chunk scheduling does not demonstrate a speedup over the normal GPU
route on this laptop. GPU remains the default; the diagnostic is not promoted
to an App preset. No further chunk tuning is justified by this screen.

One serial pair used the same local Qwen Image 2.1, pinned Viggle v0.2.1 r128,
strength 1, six steps, seed 42, 512² edit, standard 1024 reference, DiT cache off
and component-staged residency. Each route opened a fresh engine/process.
No additional model was downloaded. The default GPU route retained its prefix
snapshot optimization. Runtime naturally excludes that route and used only
staged diagnostic, early preparation and fixed chunk count 2. Neither engine
had a previous prefix-bank hit. This is a product-route comparison, not an
isolated chunk-count experiment; OS/cache warmth and order were uncontrolled.

| Span | Normal GPU | Fixed-two Runtime |
|---|---:|---:|
| Whole request | 37.611 s | 38.144 s |
| Denoising | 26.858 s | 27.213 s |
| First step | 13.246 s | 13.453 s |
| Remaining steps | 2.715–2.747 s | 2.712–2.794 s |

Runtime was 1.42% slower in this pair (GPU/Runtime request ratio 0.9860).
That small difference is not a stable performance estimate. Removing adaptive
GPU probe blocks did not produce an end-to-end benefit: all 192 FFN blocks used
fixed hybrid scheduling, with 384 Core ML predictions and no fallback. Both
untimed-hybrid and asynchronous-hybrid block counters were zero. This fixed
mode retains synchronous measurement/join costs. It is dynamic-weight FP16,
not a qualified W8A8 backend or a physical ANE utilization trace.

The first step remained about five times the subsequent steps in both routes.
This paired screen does not isolate compilation from full-reference prefill,
first-use kernels, allocation or staging. It specifically does not show that
ANE setup or adaptive GPU probes caused the first-step slowdown.

Both decoded outputs preserve the cobalt-blue teapot, geometry, framing and
background on visual inspection. RGB MAE is 0.36178/255, PSNR 51.01468 dB and
maximum component difference 20/255. RGBA MAE is 0.27166/255. This is one image,
not general workflow-quality qualification.

Private Runtime leases existed during denoising and were gone before VAE
export, after generation and after engine destruction. The continuation's
owned exported/private fixtures (313,912 logical bytes) were removed. The
initial harness attempt's fixture was removed too. MLX active memory after
Runtime generation was below 16 MiB. Source image, adapter and native library
identity were preserved. These checks do not clear system-managed Core ML or
GPU driver caches.

## Preserved harness failure

The GPU inference succeeded (return code 0), but the first test incorrectly
expected `hybrid: null`; the normal GPU result uses `hybrid: {}`. The assertion
failed after generation. Original logs and the memory verifier's
`command_failed` result are preserved. The corrected independent GPU
post-validation checks its actual empty metadata, Metal backend, full phase
sequence, original source hashes and empty private leases. GPU inference was
not repeated. Only the remaining Runtime request was executed afterward.

Consequently there is no qualified paired process-memory comparison in this
stage. The GPU memory trace is diagnostic evidence only; Runtime's independent
sampler verification passed. The generation timing/quality comparison remains
backed by successful native results and the explicit post-validation.

Evidence: `outputs/runtime-fixed-two-20261002/verification-report.json`,
`continuation-report.json`, original `acceptance-report.json`, per-route phase
logs, independent sampling reports and `pixel-comparison.json`.
Tested library: the frozen shared-preparation native build at `41d2e29`.

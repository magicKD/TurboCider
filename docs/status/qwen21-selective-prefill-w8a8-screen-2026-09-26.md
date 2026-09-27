# Qwen Image 2.1: selective first-step W8A8 FFN screen (rejected)

The production 512² hybrid offloads the 32 decode FFNs but keeps the first
full-prefix step BF16 GPU. On an M4 Max, a temporary opt-in candidate instead
offloaded only the **last 8 or 16 of 32 first-step FFNs** through the existing
1,024-row, 6,144-channel checkpoint-matched W8A8 Core ML graph. Each first-step
block processed 13,442 rows as 14 tiles, including a padded last tile. The
output of each Core ML call was copied into an independent MLX allocation
before another prediction reused the model's output backing. Decode steps
remained the unchanged full 32-layer W8A8 hybrid.

Each comparison used three ordered 1024px references, a 512² output, seed 17,
the same prompt and schedule, an independently prepared resident Session,
two-step warmup, and a cached request. The verifier checked identical text,
noise and each reference latent tensor byte-for-byte, matching compiled model
identity and the additional first-step predictions. Wall includes VAE, PNG
and equally enabled tensor dumping; it excludes model preparation. Only **one
run per configuration** was taken, not a controlled ABBA speed distribution.

| Steps | Decode-only control | First-step last 8 | First-step last 16 |
| ---: | ---: | ---: | ---: |
| 5 | 24.156 s | 23.324 s (1.036×) | 22.793 s (1.060×) |
| 40 | 71.600 s | 79.706 s (0.898×) | not tested |

The 8-layer candidate required 112 extra Core ML predictions per request;
the 16-layer candidate required 224. All routes passed the resident-probe
cancellation, GPU-switch and unload checks with exit 0. The 5-step RGB
correlations to decode-only control were 0.99437 and 0.98443, respectively.
Both five-step baseline and candidates have sticker ghosting; visual inspection
found changed face/edges at 8 layers and noticeably more distorted facial
details at 16 layers. The 40-step 8-layer comparison had RGB correlation
0.99234 and retained the two teapots and dragon sticker on visual inspection,
but its wall time regressed by approximately 11.3%. Pixel correlation is not
an edit-fidelity acceptance test; neither short-step candidate was promoted.

The longer request's slower decode has not been attributed to one specific
cause. Extra predictions and roughly 0.8 GiB of first-step BF16 tile-result
copying at 8 layers may alter allocation, scheduling or memory pressure;
Core ML prediction API times overlap GPU work and are not additive. The
`cpuAndNeuralEngine` policy and prediction count do not prove physical ANE
residency. A previous full-32-layer screen saved more in the first step but
regressed at 40 steps and changed the reference image more substantially.

The selective first-step runtime, option, reporting label and candidate-only
contract checks were removed; the ordinary decode-only W8A8 path and precise
attention segment diagnostic remain. Raw PNG/JSON/tensor dumps and comparison
receipts are ignored under `results/qwen21/session-fullref-edit3-prefill-last*`
and `results/qwen21/session-fullref-edit3-prefill-last*-comparison-20260926.json`.
These are local evidence, not shipped artifacts. Next work should target the
*exact* long-prefix GPU attention and QKV/FFN kernels, then measure full edit
requests at 5/6 and 40 steps with more seeds and semantic reference checks.

After removal, the native library and matching Session probe rebuilt,
`make test-qwen21` and `git diff --check` passed. A fresh same-input,
five-step decode-only hybrid request took 24.181 s, performed 128 Core ML
calls, passed cancellation/GPU-switch/unload with exit 0, and exported a PNG
byte-identical (SHA-256 `fecbea96e8d5285d6a266c33468bf796c220cec16755ae0023bed48ed4d4f248`)
to the pre-experiment decode-only control. Its receipt is ignored under
`results/qwen21/session-fullref-edit3-postselective-rollback-5-20260926/`.

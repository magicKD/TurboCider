# Qwen Image 2.1: full-size three-reference resident edit, steady-prefix comparison

This is a **five-step, 512² output, three original-size 1024px-reference**
research comparison, not the resized-512 tile-loop experiment. Both routes
use `TURBOCIDER_QWEN21_REF_LOCAL_ATTENTION=3` (last 16 blocks) and
`TURBOCIDER_QWEN21_PREFILL_LAST_TARGET_ONLY_DIAGNOSTIC=1`, plus
`TURBOCIDER_QWEN21_RESIDENT_PREFIX_KV=1`. The hybrid also uses
`TURBOCIDER_QWEN21_FULL_REF_W8A8_DIAGNOSTIC=1` and the checkpoint-bound
6144-channel/1024-row manifest. On a miss the full 13,442-row first-step
FFN stays on GPU; each subsequent decode runs 32 FFN channel partitions
through the existing Core ML graphs. **This is not the first-step
sequence-tile loop**: full-reference tiling of all 32 blocks and first-eight
blocks was separately screened and rejected because of visible sticker
degradation, despite reduced prefill time. See
[full tiled screen](qwen21-fullref-tiled-prefill-screen-2026-09-26.md) and
[first-eight screen](qwen21-fullref-first8-tiled-prefill-screen-2026-09-27.md).

On the M4 Max, each independently prepared resident Session had a matching
five-step warmup that did *not* fill the prefix bank. Both ran four requests
with the same seed 17, prompt and ordered reference files. The verifier
compared the actual text, noise and three reference-latent tensor payloads
byte-for-byte for each paired hit. Wall includes denoise, VAE and export plus
matched tensor dumps, excludes prepare/encoding cache miss. Timing order was
sequential per route, **not interleaved ABBA**, so variability remains.

| Request | GPU wall | Hybrid wall | GPU/hybrid | Cache status |
| --- | ---: | ---: | ---: | --- |
| run 0 | 23.857 s | 22.825 s | 1.045× | both prefix misses |
| run 1 | 8.732 s | 7.890 s | 1.107× | both prefix hits |
| run 2 | 8.825 s | 7.791 s | 1.133× | both prefix hits |
| run 3 | 8.688 s | 8.179 s | 1.062× | both prefix hits |

The **three-hit median ratio is 1.107×**; do not compare an 8-second hit to
a 23-second miss. An earlier first hit under a different run sequence
measured GPU 8.75 s versus hybrid 8.91 s, reversing the advantage. Neither
the single first hit nor these three later hits establishes a universal
steady-state gain. On the three-hit runs the hybrid issued 128 Core ML
predictions per request with zero reported failures; compute policy
`cpuAndNeuralEngine` does **not** prove actual ANE hardware residency.
The cached first step takes about 1.62 s for either route: only the four
decode steps remain available for additional FFN-offload gains. The full
miss's ~23-second wall remains a GPU long-prefix-attention / GPU FFN
problem, not a place where faster 1024-row decode W8A8 alone helps.

The hit PNG is byte-identical to this route's miss PNG for the same seed;
changing seed on a hit also matches its uncached same-route oracle. Across
routes the hit PNGs differ (normalized RGB correlation 0.99772, RMSE
0.02288); visual inspection shows both teapots and the orange sticker in
the intended layout, with slight sticker-eye and teapot shading differences.
The GPU control already has imperfect sticker detail; a correlation score
alone cannot judge subject identity. This is a plausible **opt-in, visibly
lossy** research tradeoff on one prompt, not proof for 1–2 references,
different subjects or seeds. Active MLX allocation on a hit was ~20.45 GiB
GPU versus ~24.96 GiB hybrid, excluding Core ML and OS allocations;
pressure/parallel requests remain untested.

The private reference-file mutation probe passed: overwriting only a copy
in the ignored probe directory caused a conditioning and prefix miss; the
unchanged repeat hit and produced a byte-identical PNG. Cancellation forced
a miss on retry with identical PNG; a switch back to GPU and unload passed.
The changed-reference request included re-encoding and took 34.75 s, so
its wall cannot be compared to a cached-input miss. Probe exit code 0.

Comparison evidence is under the ignored
`results/qwen21/session-fullref-edit3-{gpu,hybrid}-last16-prefix-steps-20260927/`,
`results/qwen21/session-fullref-edit3-last16-prefix-steady-hits-vs-gpu-20260927.json`,
and `results/qwen21/session-fullref-edit3-hybrid-last16-prefix-mutation-20260927/`.
The offline verifier now refuses hit-only comparisons with a missing initial
miss, a later miss, changed geometry or mismatched/gapped run sequences.
Keep the faster/resized 512px loop route separate from this full-size
reference comparison: resizing itself loses source detail.

# Qwen Image 2.1: fewer Core ML calls on repeated 512px edits

The existing opt-in resident prefix-KV + tiled-first-step route reproduces
the entire W8A8 1,024-row tile boundary on each hit: if the cached prefix
ends partway into a tile, its last FFN-input rows are combined with new
target rows and then the remaining target rows go through a second call.
This takes two predictions for each of the 15 eligible first-step FFNs
(block 31 already uses target-only), even though only the target output is
consumed. The new *additional explicit diagnostic*
`TURBOCIDER_QWEN21_TILED_PREFIX_TARGET_ONLY_DIAGNOSTIC=1` instead feeds
the 1,024 target rows to the **same W8A8 Core ML graph once per layer**
on a prefix hit. All subsequent 32-layer hybrid decode FFNs, the GPU suffix,
full text/reference attention and the step approximations are unchanged.
On a miss the original full tiled prefill remains unchanged. With an
aligned prefix, both options already use one call per layer.

The checkpoint-matched Qwen21 Core ML exporter builds channel-wise 1×1
convolutions and explicit input/hidden INT8 Q/DQ with **fixed calibration
scales** (`tools/coreml/export_qwen3.py`). These operations have no
cross-token reduction. Splitting a tile should therefore preserve its
target-row outputs; still, compiled Core ML numerical behavior must be
tested, rather than presumed identical for every graph or device. This
switch requires the prior resident five-step, 512², 1–3 resized-512 W8A8
edit + last-16 tiling + prefix-KV diagnostic and explicit approximation
opt-in. Its own cache identity prevents switching modes without a miss.
It cannot be combined with LoRA or a W8A16 GPU suffix. Unflagged behavior
and the default GPU route are unchanged.

After configuring the seven existing research flags in the
[tiled-prefix setup](qwen21-tiled-prefix-kv-edit-2026-09-27.md), additionally
set `TURBOCIDER_QWEN21_TILED_PREFIX_TARGET_ONLY_DIAGNOSTIC=1` before
`build/native/turbocider plan request.json`. The plan must include
`qwen21_tiled_prefix_target_only_diagnostic`. To reproduce a hit, issue the
same request twice through one long-lived resident Session; separate CLI
`generate` processes do not share a KV bank.

## Matched M4 Max evidence

Independent resident Sessions, same five-step warmup (without populating
the prefix), seed 42 for one pot or seed 17 for two/three subjects, same
prompt and ordered 512px references. Both arms dumped **identical** input
text, initial noise and reference latent tensors; the verifier compared
the tensor payloads. Wall includes VAE/PNG/dumps but excludes load/prepare.
Candidate calls are 112 per matched prefix hit, versus 127 in the prior
tail-reconstruction path. These are single hit pairs, not interleaved ABBA
timing distributions:

| 512px references | Reconstruct tile | Target-only hit | Pairwise ratio | Output |
| ---: | ---: | ---: | ---: | --- |
| 1 | 4.482 s | **4.185 s** | 1.071× | PNG identical |
| 2 | 5.093 s | **4.637 s** | 1.098× | PNG identical |
| 3 | 5.215 s | **4.673 s** | 1.116× | PNG and latent tensor identical |

The three-reference candidate repeated at **4.706 s**, giving the same
PNG; the candidate's uncached first request took **7.727 s** and made
172 Core ML predictions, with its PNG identical to the previous unflagged
miss. A separate previous three-reference exact-tail run took 4.887 s;
comparing that with 4.673 s suggests only a ~4.6% gain under that faster
condition. Treat the single paired ratios as exploratory, not guaranteed
speedups. Across the three fixture counts, different-seed candidate hits
also matched the uncached same-route oracle and the saved strict-tail
candidate hashes. The initial output quality is therefore **unchanged**
on these viewed samples: two teapots and the dragon sticker remain visible
in the three-reference case; the blue pot's transparency and the single
pot's matte-red instruction were imperfect even before this switch.

The candidate no longer retains 15 partially filled FFN-input tiles.
For this three-reference prompt the prefix ends 130 rows into a tile,
so the saved BF16 tail inputs themselves occupy only about 16 MiB.
Measured MLX active allocation was ~22.75 GB in an earlier exact-tail
process versus ~22.23 GB in the candidate; the larger observed difference
cannot be attributed to tail storage alone, and separate-process allocator
counters are not a complete Core ML/OS memory comparison.
The Core ML compute-unit policy is `cpuAndNeuralEngine`, but these
receipts do not prove physical ANE placement.

`make test-qwen21`, the real resident Session probes, the CLI `plan` label,
the matched-input comparison and `git diff --check` cover the current
gate, call counts, images and normal lifecycle. The 2-reference private
reference-overwrite/cancel/retry check is recorded separately in
`results/qwen21/session-ref512-edit2-prefix-target-only-mutation-20260927/`.
The first-stage CLI `generate` cannot have a cross-request hit; do not
compare it to warm repeat times. Other prompts, reference styles and
devices still need visual and performance coverage before any automatic
selection. All PNGs, tensor dumps and receipts stay in ignored
`results/qwen21/session-ref512-edit{1,2,3}-prefix-target-only-20260927/`.

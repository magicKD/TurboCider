# Qwen Image 2.1: repeated 512px edits with tiled W8A8 and resident prefix KV

The first denoise step of a 512² edit with 1–3 explicitly resized-512
references has approximately 2,091/3,157/4,226 sequence rows (actual text
length varies). The explicit last-16-layer prefill experiment runs each FFN
over successive 1,024-row tiles through the **same** calibrated W8A8 Core ML
layer graph, with a BF16 GPU suffix; it does not partition those reference
images between devices or make the GPU attention cheaper. Later decode steps
have 1,024 target rows and reuse those same 32 W8A8 models. The final-step
last-16 and penultimate even-block FFN reuse flags deliberately approximate
the diffusion trajectory. On the three-reference five-step path, these
choices make 172 predictions rather than the 128 decode-only predictions;
the tiled first step saves enough BF16 GPU work to offset the extra calls.

This differs from `../splash`'s ANE/GPU *row split*, where ANE computes
complete FFNs for one group of tokens while GPU computes complete FFNs for
the other group concurrently. Splash's recorded 512-row microbenchmark was
slower than its nearby channel-split control; its 1,024-row microbenchmark
was faster than pure GPU but did not establish an end-to-end win over channel
splitting. A Qwen21 row split would additionally need a full-width ANE
artifact and GPU full-FFN weights with a measured fork/join and unchanged
attention semantics; these experiments do not prove it faster than the
existing 6,144-channel route. A sequence tile loop is **serial per layer**;
its benefit here is offloading expensive long-sequence FFN rows, not overlap
between different tiles.

An additional **explicit, experimental** switch,
`TURBOCIDER_QWEN21_TILED_PREFILL_PREFIX_KV_DIAGNOSTIC=1`, allows the existing
resident prefix-KV bank to coexist with last-16 prefill tiling. It requires
`TURBOCIDER_QWEN21_RESIDENT_PREFIX_KV=1`, `residency=resident`, 512² output,
five steps, 1–3 ordered references resized to 512px, full W8A8 layer
coverage, BF16 GPU complement, approximation opt-in, and no LoRA. On a
cache miss each eligible layer saves an owned FFN input for the last,
partially filled prefix tile. On a hit its 1,024 target rows are combined
with this saved tail to preserve the original W8A8 tile/quantization
boundary; the target's remainder goes through the same model again. When
the prefix is exactly tile-aligned, the target goes straight through one
full tile without a tail lookup. The final prefill target-only diagnostic
still avoids recomputing prefix rows in layer 31.

This changes *repeated requests in the same resident Session* with unchanged
prompt, ordered reference pixels, checkpoint and runtime settings. A new
reference or prompt is a miss; a standalone CLI `generate` has no subsequent
request to hit. The flag never enables itself for `auto`, does not extend to
the six-step Viggle LoRA, and does not change the unflagged faster/lower-loss
hybrid. `cpuAndNeuralEngine` is a Core ML compute-unit policy, **not proof**
that every prediction actually executed on the ANE.

## M4 Max measurements and image inspection

One process per reference count, prepared resident Session, matched five-step
warmup without filling the prefix bank, seed 42 (one reference) or seed 17
(two/three references). Identical prompt, ordered inputs and initial noise
within each workload. Wall includes denoise, VAE and PNG export, excludes
model load/prepare. The miss and hit are sequential measurements, not a
statistically controlled ABBA study.

| 512px references | First miss | Repeated hit | Core ML calls miss → hit |
| ---: | ---: | ---: | ---: |
| 1 | 5.493 s | 4.482 s | 142 → 127 |
| 2 | 6.533 s | 5.093 s | 157 → 127 |
| 3 | 7.749 s | 5.215 s | 172 → 127 |
| 3, final rebuild | 7.706 s | **4.887 s** | 172 → 127 |

The final three-reference miss/hit PNG SHA-256 matches the flag-off fast
hybrid PNG, and the hit for another seed matches an uncached same-route
oracle byte-for-byte. The two-reference probe additionally replaced a
*private copy* of a reference: the next request missed, and the repeated
request hit. Cancellation/retry, GPU-route switch and unload completed.
Thus caching did not add any image loss on those fixtures. Compared with
the same-input 10.114 s pure-GPU warm *uncached* control, a 7.706 s miss
is ~1.31× faster; the matched repeated-prefix pure-GPU measurement from
the earlier screen (6.456 s) versus the new 4.887 s hit suggests ~1.32×,
but was not an interleaved A/B comparison. **Do not compare 4.887 s hit to
10.114 s uncached GPU as if they represented the same workflow.**

Inspected three-reference output preserves the beige pot left, blue pot
right and recognizable orange dragon sticker in the middle. The blue pot
is less transparent than the source reference, as in the GPU control; the
one-reference requested matte finish also fails in both routes. This is an
acceptable *research* loss on these samples, not general quality validation
for edits, generation, other seeds or image styles. The explicit 512px
reference resize is itself a separate loss of input detail versus 1024px.

The prefix K/V estimate is 512 KiB per prefix token: for three references
and 130 text tokens, ~1.56 GiB, plus up to ~128 MiB for 16 saved partial
FFN-input tiles. The runtime limits the single bank to 8 GiB and 1/8 of
physical memory with a reserve; measured MLX active allocation after the
three-reference request was ~22.75 GB (peak ~27.90 GB). The unflagged
fast-hybrid control measured ~19.75 GB active / ~27.38 GB peak under its
separate run. MLX accounting excludes Core ML/OS/file-cache allocations.
Repeated requests under memory pressure have not been qualified.

## Reproduction and safeguards

Provide a `residency=resident`, `execution=gpu_ane`, `qwen21_w8a8=true`,
`qwen21_reference_size=512`, `allow_approximation=true`, five-step edit
request, a calibrated full-32-layer 1,024-row / 6,144-channel manifest, and
1–3 reference files. Enable the previously screened fast-hybrid flags and
the new prefix flag in the **same long-lived Session**:

```sh
export TURBOCIDER_QWEN21_METAL_FUSED_QKV_DIAGNOSTIC=1
export TURBOCIDER_QWEN21_TILED_PREFILL_W8A8_DIAGNOSTIC=16
export TURBOCIDER_QWEN21_HYBRID_REUSE_FINAL_LAST16_FFN_DIAGNOSTIC=1
export TURBOCIDER_QWEN21_PREFILL_LAST_TARGET_ONLY_DIAGNOSTIC=1
export TURBOCIDER_QWEN21_HYBRID_REUSE_PENULTIMATE_EVEN_FFN_DIAGNOSTIC=1
export TURBOCIDER_QWEN21_RESIDENT_PREFIX_KV=1
export TURBOCIDER_QWEN21_TILED_PREFILL_PREFIX_KV_DIAGNOSTIC=1
build/native/turbocider plan request.json
```

The CLI plan was checked for the explicit
`qwen21_tiled_prefill_prefix_kv_diagnostic` label and rejects staged
residency/incompatible requests. A single CLI generation may use the
flagged **miss** path, but cannot realize its hit acceleration. The
five-step/full-reference fidelity choice, new prompts, other SoCs and
actual ANE hardware occupancy need further matched profiling. The final
library and matching Session probe passed the three-reference hit/miss,
different-seed uncached oracle and lifecycle checks; `make test-qwen21`
(27 + 7 + 21 contracts) and `git diff --check` passed. Evidence stays in
ignored `results/qwen21/session-ref512-edit{1,2,3}-hybrid-half-prefix-tail-20260927-a/`
and `results/qwen21/session-ref512-edit3-prefix-tail-final-20260927/`.

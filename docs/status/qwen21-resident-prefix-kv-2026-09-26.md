# Qwen Image 2.1: resident prefix-KV reuse experiment (2026-09-26)

The explicit `TURBOCIDER_QWEN21_RESIDENT_PREFIX_KV=1` experiment keeps one
32-layer DiT prefix-KV bank across requests in the same resident Session.
The default remains request-owned KV. It is restricted to 512², two or more
steps, **no LoRA** and no prompt enhancement. The original screen also
excluded GPU final-step FFN approximation; the guarded combination is
verified separately in the follow-up below. The LoRA restriction was added after the failed cross-seed
oracle described below; a flag set on a LoRA request leaves normal request-
owned KV behavior unchanged.
The prefix consists of the same text and **ordered reference-image** latents
(full-size or explicitly resized); different seeds can use the same bank.
The bank is eligible only if
its estimated BF16 K/V footprint is at most 8 GiB, at most 1/8 of physical
memory, and fits alongside the active MLX allocation with an 8 GiB reserve.
There is no cross-process/disk cache, eviction daemon or change to the
normal first-request execution. Eligibility does not guarantee absence of
system memory pressure; the experiment is not a production default.

The cache compares text tensor identity, each reference latent tensor identity
and geometry, actual first sigma, steps, output geometry, LoRA identity and
runtime precision, route, manifest and GPU RoPE mode. Reference files are
SHA-256 checked by the existing resident condition cache, so overwriting a
file invalidates both conditioning and prefix KV. Changing weights/LoRA,
staging/unloading, or canceling/failing a request destroys the bank. FFN
step-approximation state is cleared for every new request. On a cross-request
hit the first denoise step still uses the **complete BF16 GPU FFN**; only its
prefix attention computation is skipped. Hybrid steps 2..N still use the
same `(steps-1)×32` Core ML calls as the uncached route. A cache-hit marker
in `acceleration_selection` is diagnostic, not proof of physical ANE
occupancy.

## First same-input checks, M4 Max / 512² / five steps

Same ordered full-size references and seed within each row. Independent
processes prepare, run a two-step warmup (which does **not** fill the
five-step cache), then issue two five-step requests. Wall includes denoising,
VAE decoding and PNG export, excludes prepare/warmup. Tensor dumping was
enabled for single- and three-reference runs in both tested routes.

| References | Route | First miss wall | Second hit wall | Active MLX allocation at hit |
| ---: | --- | ---: | ---: | ---: |
| 1 | BF16 GPU | 11.636 s | 6.830 s | 17.62 GB |
| 1 | W8A8 + BF16 GPU | 10.432 s | 5.669 s | 22.46 GB |
| 2 | BF16 GPU | 18.097 s | 7.778 s | 19.79 GB |
| 3 | BF16 GPU | 25.220 s | 8.706 s | 21.96 GB |
| 3 | W8A8 + BF16 GPU | 24.201 s | 8.369 s | 26.80 GB |

The two-reference GPU probe also overwrote a **private copy** of the first
reference file with the second reference's bytes. It reported a cache miss
and a different PNG, then a hit and matching PNG on repetition; cancellation
followed by retry produced a miss and the changed-condition PNG. The final
corrected probe completed successfully. It did not modify the source
reference files. No two-reference W8A8 hit timing has yet been taken.

The same single-reference GPU request with the experiment **off** measured
11.639 / 11.638 s in a separate warmed process. Its two PNGs, both cached
GPU PNGs, and the previously saved ordinary GPU PNG are byte-identical.
The final rebuilt library also ran the **flag-off** single-reference W8A8
hybrid request in 10.383 s, with process exit 0, unchanged 128 Core ML calls,
zero output-copy bytes and byte-identical PNG against both experimental
hit/miss outputs. This verifies the normal one-request hybrid behavior did
not change in that case.
The single-reference hybrid hit/miss PNGs also match the pre-existing hybrid
PNG byte-for-byte. In the three-reference probes each route's first and
second PNGs are byte-identical; a different seed reused the prefix and
matched that seed's uncached same-route PNG byte-for-byte. Each hybrid
request retained the expected 128 FFN Core ML calls and zero output copies.
One-/three-reference warm hit GPU→hybrid ratios are about 1.20× / 1.04×:
prefix reuse greatly improves **both** routes on repeated conditioning, but
does not by itself make hybrid three-reference attention faster. The first
three-reference request still improves only about 1.04× via GPU/ANE FFN.

The 31-token text-only five-step GPU request is a negative control:
5.985 s (miss) versus 5.928 s (hit), with byte-identical PNGs including
the older baseline image. Its W8A8 hybrid route measured 4.710/4.668 s,
again with byte-identical per-route PNGs; short text-only prefixes offer
little cross-request saving. No image-generation-quality claim follows
from these five-step fox results, which were visibly soft in earlier tests.

**LoRA safety finding:** An initial six-step Viggle v0.2.1 r256 run with
the cross-request bank enabled reused the prefix after its same-step warmup.
The same-seed PNGs matched each other, but a different seed's cached PNG
did **not** match the uncached same-route oracle byte-for-byte. That test
failed: RGB RMSE was 1.75/255, MAE 0.87/255 and correlation 0.99959, with
some pixels differing by up to 49/255. The experiment now explicitly excludes
LoRA requests. This is not a claim that the Viggle output was visually bad; the source of the
numerical mismatch and any safe adapter-aware reuse need separate work.
After adding the guard, the six-step Viggle session probe with the flag still
set passed its resident LoRA→base→LoRA lifecycle and produced two normal
request-owned-KV requests (8.276/8.246 s), neither labeled a prefix-cache
hit/miss. Thus this experiment does not change Viggle's current GPU route.

The final rebuilt library and a probe recompiled from its matching Session
header passed the single-reference hybrid hit/miss, cross-seed GPU oracle,
cancel/retry, Core ML call-count, image-export and unload lifecycle checks
with process exit 0 (10.395/5.630 s). An intermediate test process had
completed its requests but faulted on Session destruction: its probe was
compiled against an earlier C++ Session member layout than the rebuilt
dynamic library. Its crash report identified HybridMLP/MLX destruction;
after recompiling the probe against the matching header, that exit fault
did not reproduce. Do not use the intermediate mismatched-binary receipt
as a passing lifecycle result.

The 5-step three-reference images have ghosting in both ordinary GPU and
W8A8 paths. Byte parity proves this cache does not make that sample worse;
it does **not** prove the underlying edit meets the user's visual-quality
target. The per-route Core ML configuration also does not prove physical
ANE residency. This one-device, one-seed initial comparison lacks repeated
ABBA runs, resource-pressure and idle-time testing, and cannot justify
promoting the option to a default or claiming this speedup for a fresh
prompt/reference set. Cold requests still pay preparation and hybrid setup.

## Follow-up GPU hotspot screen for the next optimization

Using the existing explicit **block-0-only**, synchronized operator timer on
the same three-full-reference / five-step GPU request (two-step warmup, then
one measured request), the measured first step had approximately 91 ms QKV,
41 ms Q/K norm+RoPE, 151 ms attention, 32 ms attention output projection,
and 194+99 ms FFN gate/up and down. Among measured decode steps, attention
was usually approximately 20 ms (one 39 ms outlier); gate/up plus down
approximately 23–24 ms, and QKV approximately 8 ms. Request wall was
25.252 s. This instrumentation forces synchronization and disables the
normal compiled graph for block 0, so these are **not** production per-layer
times and must not be multiplied by 32. They nevertheless identify both the
large first-step full-sequence FFN and the long-prefix decode attention as
separate candidate hotspots. Cross-request KV reuse avoids much of the
first for a repeated condition; improving the GPU attention/QKV path or
validating a first-step FFN offload is still required for fresh edits and
for hybrid to exceed GPU substantially on three references.

## Reproduce and outstanding tests

The native `qwen21-session-probe` uses its existing `--request=...` mode and
`--dump-tensors` or `--test-edit-cache`. Set the flag above only for the
candidate; compare against the same request with the flag absent. The probe
asserts hit/miss progression, PNG parity, cross-seed parity against an
uncached oracle, unchanged Core ML call counts, cancellation/retry invalidation,
and, with `--test-edit-cache`, reference-file overwrite invalidation. Raw
reports and PNGs are in ignored `results/qwen21/session-prefix-kv-*-20260926/`;
no model, tensor dump, generated image or compiled artifact is committed.

At this original screening stage, two-reference W8A8 editing and 40-step
workloads were still untested; the follow-up below adds those cases. Still
needed: memory/idle pressure, more seeds and prompts, LoRA root-cause analysis, and clean
ABBA repeats of both routes before considering automatic selection. Native
build and `make test-qwen21` pass; the latter does not exercise actual model
inference. The experimental path is never enabled merely by CLI `auto`.

## Follow-up: 1–3 resized-512 references and hybrid hit correctness

The earlier full-size-reference check did not cover the explicit 512px
reference-resize mode. On the same M4 Max with 512² output and seed 42
(single red teapot) or 17 (ordered two teapots/three subjects), each
independent resident process prepared its route, warmed it with two steps,
then ran two **five-step requests** with the same prompt and reference
files. The first request necessarily misses the five-step prefix bank; the
second hits. Request wall includes VAE/PNG and, except for the two-reference
hybrid mutation run, equal text/noise/reference tensor dumps, but excludes
prepare/load. These are sequential first-versus-second requests, not
interleaved statistical samples. The resized inputs are identical *within*
each workload; comparisons with normal full-size references are not a
same-input GPU acceleration claim.

| Number of 512px references | GPU miss → hit | 6144-channel W8A8/BF16-GPU miss → hit |
| ---: | ---: | ---: |
| 1 | 7.268 → 6.157 s | 6.078 → 4.898 s |
| 2 | 8.660 → 6.387 s | 7.405 → 5.154 s (private-file mutation test) |
| 3 | 10.156 → 6.612 s | 8.916 → 5.481 s |
| 3, 40 steps | 52.309 → 48.812 s | 39.553 → 36.128 s |

The first three-reference 512px W8A8 attempt was **not** successful: on
the second request the cached prefix made `Transformer::forward` enter its
decode-shaped path immediately, so installing the W8A8 callback before
step zero offloaded the first target FFN and violated the required
`(steps−1)×32` prediction count. The resulting request exited before PNG
export. The fix installs the callback at actual denoise step one, keeping
step zero's FFN full BF16 GPU even on a prefix hit; later steps still use
all 32 W8A8 Core ML layers. This is necessary for output parity, not an
additional quantization shortcut.

After rebuilding both the library and its matching native probe, the
corrected three-reference five-/40-step hybrid requests passed: each 5-step
request made exactly 128 W8A8 calls, each 40-step request 1248, with no
reported runtime failures. Miss and hit PNGs, and the earlier no-prefix
same-route PNGs, were **byte-identical** within each step count. A different
seed's cached PNG matched its same-route uncached GPU/ANE oracle byte-for-byte.
The two-reference hybrid probe overwrote **its own private copy** of a
reference with different bytes: it correctly missed and generated a new
PNG, hit on repetition, and missed after cancellation/retry. Source
references were untouched. Single-reference GPU and hybrid, two-reference
GPU, and three-reference GPU probes also passed their hit/miss, cross-seed,
cancellation and unload lifecycles.
The three-reference 40-step GPU request also produced byte-identical PNGs
for the miss, hit and earlier no-prefix GPU baseline; its different-seed
cached PNG matched the uncached oracle. At 40 steps the repeated-input
GPU/ANE hit is ~1.35× faster than the same-input exact-GPU hit, whereas
the GPU/ANE miss is ~1.32× faster than the GPU miss. Neither comparison
counts model preparation or establishes physical ANE execution.

The separate approximate pure-GPU mode (final FFN reuse, half penultimate
reuse, and fused Metal Q/K norm-RoPE) previously excluded cross-request
prefix KV. A second guarded candidate permits their *explicit* combination:
the prefix cache identity now includes both FFN-reuse flags and fused-Q/K
mode, and per-request FFN step cache is reset. For one, two and three
resized-512 references, the five-step first-miss → repeated-hit walls were
**6.092 → 5.014 s**, **7.446 → 5.235 s**, and **8.889 → 5.459 s**,
respectively. All three same-seed hit PNGs match their earlier uncached
approximate-GPU outputs exactly; in the three-reference run a different
seed's cached PNG also matches its uncached same-mode oracle. This is not the exact
BF16 output: separate FFN-step reuse and fused Q/K rounding remain
approximate. The prefix KV is exact relative to its selected route.
The combined two-reference candidate also passed a separate private-file
overwrite test: changing the first reference's bytes invalidated the prefix,
the next identical request hit the rebuilt bank, and cancellation cleared it.
The original source references were not modified.

The extra bank has a cost: in the three-reference five-step GPU example,
reported active MLX allocation rose from approximately **13.88 GiB**
(ordinary cached-condition request, no cross-request bank) to **15.95 GiB**
on the bank hit; hybrid rose from **18.39 to 20.46 GiB**. Core ML/OS/file
caches are excluded from those MLX figures. Admission still limits the
estimated bank to 8 GiB, at most one-eighth of physical memory, and
reserves additional space. No device-memory-pressure/idle residency study
has been completed, so repeated-edit speed should not be enabled by default.
The `cpuAndNeuralEngine` policy is not proof of physical ANE execution.

The corrected 512px input probes and JSON/PNG/tensor receipts live in
ignored `results/qwen21/session-ref512-edit{1,2,3}-{gpu,w6144}-prefix*`.
`make test-qwen21` and `git diff --check` pass; actual runtime probe call
counts and byte equality are stronger evidence for this bug than plan-only
tests. A final rebuild after adding the incompatible-mode gate repeated the
three-reference hybrid five-step hit/miss lifecycle at 8.923/5.449 s;
the prefix-hit and current no-prefix control PNG remained byte-identical to
the pre-fix decode-only W8A8 baseline. Other prompts, Viggle LoRA and broader
memory conditions still need checks.

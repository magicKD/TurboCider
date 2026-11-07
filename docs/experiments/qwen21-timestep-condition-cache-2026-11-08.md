# Turbo8 time conditions: retire 194 MiB after the last reader

The pure GPU Qwen Image 2.1 Turbo streaming candidate prepares the eight
original time conditions before the first DiT pass, then retires the four
time-only BF16 matrices. All time projections keep their original M=2
geometry, BF16 boundaries and separate SiLU expressions. This is a
lifetime optimization; it does not turn these BF16 embeddings into INT8
weights or change the original ConvRot W8A8 arithmetic.

## Ownership and arithmetic

The four original matrices occupy 203,423,744 B (194 MiB): timestep
linear_1, timestep linear_2, modulation.1 and norm_out.linear. Eight full
M2 modulation/norm-out pairs occupy 655,360 B (640 KiB). The source
matrices leave both the pager's resident arrays and its bound Weights
object after synchronization and held-fd source revalidation. Cached
layer eviction never reloads these fixed fields. External owners are
counted separately; removal from the bank alone is not a physical-memory
claim.

Prepared values are evaluated independent leaves with request-local
owner, verified source identity, exact F32 authored sigma bits and dtype
bindings. Another Transformer, a partial retirement, a rebound source,
changed bias, another arithmetic route or sigma cannot consume them.
Geometry/reset affects prefix state but preserves the externally owned
time frames. Preparation does not batch the original eight M2 operations
into M16 and does not merge their two independent SiLU producers.

Initial admission, requested weight budget and cumulative source reads
remain unchanged. Idle MLX allocator cache is drained after retirement.
The original branch keeps modulation at step start and norm-out after
the DiT blocks when the feature is disabled.

## Component and real-trajectory qualification

Apple M5 Pro, 24 GiB, macOS 26.5.1, MLX 0.32.0, TF32 disabled. The
isolated native/CLI build is
`tc-runtime-build-v1-295e70f98d04595b4fee526c55fa18b3cb491fca43a30fb849978c5c1572f7ea`,
with library SHA256
`294847cfc80f6d5e929ac6417d14f81b110d18296dc901b4378100416abd2569`.

The fixed-release test passed external-alias capacity accounting,
asynchronous last-reader drain, rejection atomicity, source/FD checks,
prefix1/2 and eviction without resurrection. The existing pager test
passed. A synthetic actual-hidden4096 test compared every BF16 byte of
all sixteen original M2 outputs, verified 640 KiB independent backing and
expired source owners. A complete small Transformer also passed ten
forwards, eight Euler updates, reset/geometry and invalid-reentry tests.
Six existing Qwen modules subsequently passed seven tests without skips.

Three initial test-fixture failures remain preserved: the CPU policy
driver omitted common.cpp, the ordinary image fixture inherited
audio=true, and fixture updates misused whole-bank bind_arrays as an
append operation. Only tests were corrected; the native build above did
not change. The first failure performed no GPU work. The third completed
the full M2 comparisons before its invalid-binding fixture failed.

Two actual 512px Turbo8 DiT trajectories then ran against the same new
DLL, with minimum optional weight residency. All eight actual velocity
and Euler latent payload pairs (131,072 B each) matched every byte; both
final latents matched the older authoritative fixture. Each arm completed
1536 W8A8 projections/A8 dispatches and 256 drained refills/readers. All
four candidate source owners expired and active allocation fell exactly
203,423,744 B. The 4 GiB reserve and 512 MiB growth guard were retained
at every step. Independent sampling observed zero swap-in/out traffic.

## Complete CLI requests

The separate complete-image correctness requests retained original VAE
decode and PNG export. Both off/on requests matched the full baseline
PNG, text, noise, final-latent and pixel payload bytes. The first campaign
completed its off arm but timed out waiting 60 seconds for normal macOS
pressure before starting on. It remains failed. After an observed
recovery, only the unlaunched candidate ran in a new directory; a named
CPU composition binds both original raw reports. Neither diagnostic
timing is performance evidence.

Eight further fresh-process CLI requests completed in fixed
off/on/on/off/on/off/off/on order, without tensor dumps. Every arm used
cached source-verification receipts, prefix0 weights, the same original
source generation and loaded library, 252 encoder / 1536 DiT W8A8
projections, the same read workload and an identical baseline PNG:

| Control | Complete request seconds | Pooled median |
| --- | --- | ---: |
| Off | 9.801351, 9.470787, 9.481017, 7.718481 | 9.475902 s |
| On | 7.722369, 7.373490, 7.496423, 7.984121 | 7.609396 s |

The observed median request reduction is **19.697% (1.2453x)** in this
small occupied-host campaign. ABBA and BAAB control/candidate window
ratios are 1.27665 and 1.11104. Keep the final pair's reversal
(7.718481 off / 7.984121 on) and all samples: these observations do not
establish an invariant speedup or identify the source of timing drift.

All four on arms directly reported eight compact frames and an actual
194 MiB active-allocation drop. Their preparation took about 26–27 ms,
included in denoise/request timing. All eight independent sampling
receipts completed with zero swap-in/out. The full-request MLX peak was
5,821,290,854 B for every arm: releasing time matrices during DiT does
not establish a lower whole-request peak when another phase dominates.
Active bytes, sampled footprint/RSS, logical source bytes and overlapping
host read/wait spans are distinct measurements.

## Evidence and adoption boundary

Local raw evidence is under
`outputs/qwen-convrot-gpu-fusion-20261107/time-condition-cache/`, notably
`trajectory/attempt-1794083458478964000/`, the preserved initial
`correctness-attempt-1794083142750296000/`,
`correctness-candidate-continuation-1794083688734784000/`, and
`performance-attempt-1794083730652014000/`. Component/build evidence is
in the isolated worktree's `outputs/convrot-time-cache-validation/`.

The initial qualification above used explicit
`TURBOCIDER_QWEN21_STREAM_TIMESTEP_CACHE=0/1` with a default-off build.
A subsequent candidate-only real trajectory used a 1280 MiB weight budget,
retained two layers for three passes, and explicitly evicted them after
step 2. All sixteen full velocity/Euler payloads and the older final
fixture matched, with 6 cached layer uses and 250 drained refills. Actual
time-source release remained 194 MiB; eviction released 436,569,952 active
bytes. All guards passed, peak sampled footprint was 1,322,042,640 B and
swap-in/out were zero. This is correctness/resource evidence, not a new
performance comparison. Evidence is `prefix2/attempt-1794084734055500000/`.

The automatic-selector follow-up enables only the qualified GPU Turbo8
dual original ConvRot request domain. M4/Float-A, base schedules, Runtime
ANE and other routes keep their previous behavior. Explicit `0` preserves
the original path; explicit `1` strictly rejects incompatible requests,
including a nonstream request at the central Session entry.

The new native build is
`tc-runtime-build-v1-568c0c336d02400ca962f6ff74cc979c9bfec792fe8970b77b20721e2a0af721`,
with original library SHA256
`5db5c5de2a58ccaf2c5823741f6cfc0087ba5d4e4169dd0cfa2d9e87849557bf`.
The automatic policy and newly linked actual Metal component tests passed.
A fresh CLI request with the timestep option absent recorded automatic
preparation/retirement and matched all four complete tensor payloads and
the final PNG. A real packaged CLI nonstream negative request with
explicit `1` rejected at the central entry without producing a PNG.

The desktop App and library helper compiled. Their unchanged Swift/C ABI
inputs (71 files) and four built target hashes were checked before reuse
with the new native library. The separate candidate App packaged and
passed ad-hoc signature verification. An App-state acceptance executable
in a signed copy then exercised fresh model registration, bundled Tiny
discovery, the ordinary Studio request, generation and history/cleanup.
No Qwen Time/W8 diagnostic flag was present. Its actual runtime was the
new build above and dyld loaded only the copied candidate library.
Previews were exactly steps 2/4/6/8; final decoding used the original VAE
and the complete PNG matched the baseline bytes. Time retirement and
252/1536 W8A8 counters were observed. Peak sampled footprint was
4,075,292,064 B, with zero swap and maximum sample gap 110.1 ms.

The original outer App checker failed after successful generation on a
receipt key (`tiny_vae` belongs to the saved request). Its failed report
is retained. The separate CPU-only `validation-addendum.json` completed
all remaining checks without another GPU run.

An additional default-build ABBA performance attempt was **not qualified**:
its first disabled arm returned a system-memory-pressure rejection and
produced no PNG or complete timing record. No candidate arm ran. This
does not replace the earlier complete eight-arm opt-in campaign or
establish another speed claim. Admission guards remain unchanged.

Follow-up evidence is in `automatic-validation/`: full default correctness
`correctness-attempt-1794085315663872000/`, interrupted timing
`performance-attempt-1794085486991836000/`, App acceptance
`tiny-preview-attempt-1794086011598516000/`, and central-entry rejection
`negative-resident-1794086269042062000/`. Actual GUI clicking remains
unverified because the Mac desktop is locked; App-state acceptance is a
separate scope.

# Qwen 2.1 streamed VAE decoder weight residency

This experiment reduces unnecessary source materialization in the original
VAE phase of GPU streamed generation. It does not change ConvRot weights,
W8A8, denoiser math, VAE coefficients, convolution precision or normalization.
The independent candidate is on `perf/qwen21-vae-decode-residency`; its
strict0/1 request-frozen switch `TURBOCIDER_QWEN21_STREAM_VAE_DECODE_ONLY`
is initially off. Shipping App artifacts have not changed.

## Mechanism and unchanged validation

The original executor loads and materializes the entire VAE bank, including
the encoder and quant-conv tensors that generation never consumes. VAE
construction also keeps aliases of those arrays until decoding finishes.
Clearing only the outer Weights would not remove those aliases.

Before materialization, the candidate performs the same full-table VAE
constructor validation in explicit `Original, fused_norm=false` mode. That
path checks shapes/canonical duplicates and creates lazy views; it does not
enter the optional FP16 eval/finite-check branch. Its temporary scope ends
before filtering. Only original keys mapping to `decoder.*` or
`post_quant_conv.*` survive; original time-conv exclusions remain. Counting
and validation finish before any erase, with cancellation checks. Retained
array IDs, dtype and saved metadata are unchanged, followed by the original
materialization and decoder. Eligibility excludes references, preparation,
resident/component-staged and ANE routes; Tiny preview may coexist, while
Tiny final decoding cannot claim this route.

The actual675509688B original VAE contains238 tensors. Reading only its
28880B prefix/header, with unchanged file generation and0 payload reads,
gave the expected retained128 tensors/506140968 logical bytes and
dropped110 tensors/169339840 logical bytes (about161.5MiB). These are
logical source counts, not a measured RSS or physical SSD saving.

## Fresh native and CPU tests

Runtime `d36cffe0…04d3662` built in an isolated directory. Its685 native
source inputs match before/build/after/test snapshots. Toolchain, MLX and
feature inputs match the prior9af runtime apart from the intended sources;
the actual CAPI getter and dladdr select the fresh library. A preceding
controller incorrectly pinned a missing Xcode directory; that pre-compiler
failure is preserved, and only the controller's toolchain path was changed
to the actual Command Line Tools installation for the successful child.

The permanent CPU fixture uses a counting Reader and the actual Original
VAE/helper. Three malformed unused metadata cases preserve the original
error and leave the bank unmodified. Legacy/canonical aliases, source IDs,
dtype, saved affine metadata, external aliases, weak Reader lifetimes and
cancellation passed. Preflight reads0 payload bytes; materialization reads
only retained sources. No model or GPU evaluation was used by this fixture.
Build/test evidence is in the candidate worktree:
`outputs/vae-decode-residency-validation/clt-path-fix-attempt-1794098461811646000/`.

## Complete image correctness passed

Two fresh processes from the same new library ran the standard512×512
Turbo8/seed73 request, original dual INT8 ConvRot W8A8, GU fusion0,
Time-cache1 and minimum optional weight residency. Both complete PNGs and
all text/initial/final-latent/F32-pixel payloads match the original references.
Actual enabled diagnostics report exactly the128/110 tensor and logical-byte
counts above, with full metadata preflight. Both preserve252 encoder and
1536 DiT projections, prefix0 and the original source/fill/fence receipts.

The correctness runs observed process-tree footprints4040771240B and
3781690904B, but system compression differed (9932701696B/4305174528B)
and decompression differed (3522396160B/1927135232B). Both samplers completed
with0 new swap-in/out. These two observations are not paired performance
or attributable memory-saving evidence. The separate dump-free ABBA/BAAB
complete-command campaign is still the performance gate. Evidence under
the primary root:
`outputs/qwen-convrot-gpu-fusion-20261107/vae-decode-residency-whole-image/correctness-attempt-1794098779050085000/`.

## Four-GiB observer result and separate diagnostic scope

The first dump-free paired campaign stopped during its initial control0
request: observed process-tree footprint4462397168B exceeded the external
4294967296B observer limit. The command group was terminated; no complete
CLI timing/result or candidate arm was published. New swap-in/out were0.
The report remains failed/unqualified; the cap was not changed during
that campaign. Evidence:
`vae-decode-residency-whole-image/performance-attempt-1794099014302937000/`.

A separately declared candidate-only resource check used the same original
4GiB observer cap without tensor dumps. It completed with exact full PNG,
the expected actual VAE residency metrics and all original GPU W8A8/Time/
source/cache receipts. The complete native command took10.327329s; observed
footprint was3991176752B, with0 new swap-in/out. System compression was
5181014016B and decompression1772322816B. This is a resource/functionality
observation, not a paired speed or attributable RSS-reduction result.
Evidence:
`vae-decode-residency-resource-check/resource-check-attempt-1794099354095266000/`.

To observe both existing routes to completion, a new comparative diagnostic
uses a common external5GiB process-tree cap. Its only controller code change
is that observer constant. The native4GiB system reserve,512MiB per-step
growth guard, normal-pressure requirement, zero-new-swap policy, source
validation, actual VAE counts and full PNG/tensor comparisons remain the
same. The old4GiB paired refusal is not reclassified as passing. Readiness
waits and the distinct cap are recorded; no further cap increase is part
of this diagnostic. Its fresh control0/control1 correctness campaign also
passed all complete PNG/tensor checks before dump-free timing:
`vae-decode-residency-5gib-diagnostic/correctness-attempt-1794099637367658000/`.

## Eight-run result: deterministic allocator saving, speed unqualified

The separate5GiB-observer ABBA/BAAB diagnostic completed all eight commands
and samplers. All complete PNGs, actual counts, original source/cache
receipts and zero-new-swap checks passed. The native library and code were
identical across the two controls, with GU0 throughout.

| Index | Decode-only | Complete CLI (s) | Denoise (s) | VAE (s) |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0 | 8.448456291 | 6.272441209 | 1.095600875 |
| 1 | 1 | 9.107062250 | 7.060582375 | 1.040953000 |
| 2 | 1 | 7.912884000 | 5.843080208 | 1.056138542 |
| 3 | 0 | 7.941273709 | 5.879787500 | 1.057683917 |
| 4 | 1 | 7.546100291 | 5.456830375 | 1.072681250 |
| 5 | 0 | 8.366964791 | 6.341768208 | 1.043075417 |
| 6 | 0 | 9.665546458 | 7.550616583 | 1.098315334 |
| 7 | 1 | 7.656810750 | 5.585052542 | 1.080244584 |

Every control0 reported MLX allocator peak5821290854B; every control1
reported5651813798B. The repeated allocator peak reduction is169477056B
(about161.6MiB), consistent with avoiding169339840B of unused logical
weights plus allocation overhead. Final active bytes were identical at
4784686B. This qualifies the allocator/source-residency improvement for
this request; it does not measure the entire system or driver allocation.
Observed physical footprints were3832153768–4160423688B for control0 and
3495020032–3990603216B for control1. Their variable compression and working
sets do not support assigning the entire physical difference to the patch.

Whole-command ABBA medians were8.194865s/8.509973s (0.962972×); BAAB was
9.016256s/7.601456s (1.186122×). Thus whole-request speed remains
unqualified: the two windows disagree and much of the variation precedes
the VAE selection in denoise. Pooled VAE medians were1.076642s/1.064410s,
which is only a scoped observation, not a stable full-request gain.
The source/lifetime change can be assessed as a memory improvement without
relabeling these timings as an acceleration result. Any default-path build
and App acceptance still require their own validation. Evidence:
`vae-decode-residency-5gib-diagnostic/performance-attempt-1794099767357958000/`.

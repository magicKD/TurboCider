# Global-statistics row/file decoder candidate

## Status and scope (2026-10-14)

The native `VAE::decode_spatial_stream` candidate lowers high-resolution
decoder residency while preserving complete per-frame GroupNorm statistics,
causal temporal taps, original coefficients, initial upsampling phases and all
spatial attention keys. Requests and the candidate App can explicitly select
`seedvr2_vae.spatial="file_rows"`; omission/older drafts retain `"full"`. This is
not an automatic/default route or full tiling of every VAE operator. Two actual
seven-request App workflows and public-engine cancellation/quota/recovery pass;
balanced warmed full-request timings,
memory admission, low-memory hardware and longer complete model clips remain
separate requirements.

Request-local `spatial_rows` accepts 8-aligned values in 8...128 (default 64),
and `spatial_disk_budget_bytes` is a positive live-file limit up to 8 GiB
(default 8 GiB), not a RAM limit. The public route is restricted to canvases
at most 1024 on each axis and `singleton2d`/`conv3d`; combinations with
`mixed3d`/`tiled3d` fail explicitly. The App presents a manual picker and IO/
numerical-scope warnings; selecting it replaces an incompatible convolution
recipe with `singleton2d`. Generic memory advice does not automatically select
file spilling. The route currently schedules decoding only, not encoding.

Eligible high-resolution norm/conv stages run in row bands (64 by default).
Global FP32 statistics use the same 4096-element centered reduction as the
existing Metal GroupNorm. The affine/SiLU rounding recipe is unchanged and
uses those full-frame statistics; it is never tile-local GroupNorm. Each band
has the convolution's spatial halo, ordinary outer zero padding and full width.
Convolution tile shape/reduction changes mean pixel results are **not bitwise
identical** to the default, despite unchanged coefficients/context. Quality must
be independently qualified; no threshold has been loosened.

For videos, eligible causal history frames are stored losslessly in private
FP16 row files and only the required halo bands are read back. Threshold defaults
to 256 MiB per input frame. Smaller stages and all-key attention keep their
existing GPU path. The current/residual activations and assembled layer output
can still be complete frames; no hard whole-VAE memory bound is claimed.

## Measured decoder-only observations

M4 Max / 64 GiB / macOS 26.6.2 / MLX 0.32.0, original VAE SHA
`20678548f420d98d26f11442d3528f8b8c94e57ee046ef93dbb7633da8612ca1`.
Both routes use frame history and 1-GiB allocator cache. Timings include
host quality scans and, for the latest video screen, bounded raw frame export.
These are single component observations, **not** warmed ABBA/full requests.

| Decoder workload | Default / row-file seconds | Default / row-file active peak (bytes) | Default / row-file sampled process footprint (bytes) |
| --- | ---: | ---: | ---: |
| 128 / t9, forced 4-MiB threshold, rows16 | .706 / 3.050 | 714,279,766 / 598,510,422 | 1,340,703,824 / 962,921,528 |
| 1024 / image t1, 256-MiB threshold, rows64 | 1.329 / 1.308 | 9,670,740,818 / 3,181,661,286 | 9,774,318,576 / 4,206,871,440 |
| 1024 / restored-video causal prefix t5, rows64, batched IO | 10.847 / 35.855 | 13,121,981,262 / 5,639,867,218 | 14,464,500,768 / 6,690,424,008 |
| 1024 / complete restored-video t25, rows64 | 56.074 / 189.350 | 13,196,954,446 / 5,647,731,538 | 14,661,682,424 / 6,824,477,992 |

The t5 video screen deliberately selects the **first five** frames of an offline
saved t25 latent/output witness. It does not truncate an App request/source video
or qualify a complete t25/t121 restoration. All five requested frames are emitted
and checked. Active allocation and sampled physical footprint are different
metrics; do not add them or infer a hard process/device bound from either.
The single-frame route creates no history files. Its improvement is primarily
avoiding full-frame normalized activations/FP32 convolution workspaces.

Video deliberately trades storage/time for residency: the t5 run writes
**16,106,127,360 bytes** and reads **25,413,287,936 bytes**; peak live files are
**6,979,321,856 bytes** (6.5 GiB), owned host read windows peak **69,206,016 bytes**.
The initial per-row IO observation was 37.180 s; batching writes into up-to-4-MiB
operations observes 35.855 s with unchanged output/error/counters. That comparison
is exploratory, not a balanced optimization speed claim. The complete t25
observation writes **80,530,636,800 bytes** and reads **152,479,727,616 bytes**,
with the same 6.5-GiB file peak and 69,206,016-byte host-window peak. All 25
frames / 78,643,200 RGB values are checked; it is not a full App request or a
16-GiB qualification. The whole request's
other phases can become the memory bottleneck once decode is reduced.

## Encoder-head follow-up (2026-10-15)

The separate native candidate now offers `encoder_spatial="full"|"file_rows"`,
independent of the existing decoder `spatial` field. Omission/older drafts keep
`full`. Row height/live-file budget are shared settings, not a whole-process RAM
limit. The same <=1024 canvas and singleton2d/conv3d restrictions apply.

Only the high-resolution singleton encoder head is scheduled. Each source
frame's four residual norm/conv operators retain complete global statistics,
original halos/taps and FP16 coefficient/bias cast boundaries. Before the head
frames are assembled, spatial scheduling is disabled for later multi-frame
stages, temporal downsamplers and all-key attention, even under a low diagnostic
threshold. It is not independent crops or full tiling of all encoder operators.
Eligible video histories spill losslessly; single images create no history files.
The original initial5/subsequent4 partition and final-source padding remain intact.

Single complete-encoder observations (not warmed full requests):

| Encoder workload | Full / row-file seconds | Full / row-file active peak (bytes) | Full / row-file sampled process peak (bytes) |
| --- | ---: | ---: | ---: |
| 1024 image | .665 / .545 | 5,425,578,590 / 2,931,575,906 | 5,643,240,480 / 3,428,975,576 |
| Complete real1024/t25 input | 20.665 / 69.627 | 8,001,784,646 / 5,854,301,002 | 8,850,081,064 / 6,646,351,144 |

The actual25-frame route executes1600 row convolutions and100 complete-frame
normalizations, writes25 GiB, reads49.40625 GiB, peaks at2.25 GiB live files and
33 MiB owned read windows. All live counters/cleanup failures retire to zero.
These numbers include buffered source tensors and are not a complete App budget.
Encoder and decoder caches have separate execution lifetimes; do not add their
individual file/process peaks as if simultaneously measured.

An independent image encoder ABBA uses four processes, one cold plus six newly
computed warm encodes each, same original weights/frame history/cache1. Warm
medians are **.659063/.561925 s**, about14.7% lower encoder latency, with active
maxima **5,425,578,594/2,931,575,910 bytes**. This is encoder-only improvement,
not an automatic default or full-request speed claim. The video observation is
slower and remains a manual storage/time tradeoff; video ABBA is still open.

All source frames/latent values are compared against native encoding. Image/
video relative L2 is .000622676/.000513736 and cosine .999999806/.999999868;
outputs are **not bitwise equal** to the full route. Independent FP32 MPS encoder
references use original FP16 coefficients promoted to FP32, the same normalized
source files, original source partitions and no automatic fallback. All floating
dispatch inputs/outputs and parameters are audited; the image/video references
check2041/27066 floating values and emit F32 `[1,16,1,128,128]` /
`[1,16,7,128,128]`. Their audited durations are not speed benchmarks.

| Original-FP32 whole-latent gate | Native relative L2 / cosine | Row-file relative L2 / cosine |
| --- | ---: | ---: |
| 1024 image | .005067376 / .999987161 | .005067904 / .999987158 |
| Complete1024/t25 | .003270006 / .999994669 | .003269107 / .999994671 |

Both routes pass unchanged .05/.999 gates. The FP32 reference includes native's
FP16 input/scaling arithmetic difference; it is not an FP16 report relabelled.
Complete source streaming uses the same encoder scheduler. Buffered/streamed
latent words match for source counts1/2/4/5/6/9, including padding and causal
history; stats/files/windows drain. Cancellation after the first full head chunk
with active histories returns the typed cancellation, quota failure is explicit,
the source tensor remains unchanged, and the same VAE subsequently succeeds.
An initial quota test caught the wrong C++ exception category; its terminal
failure is preserved, and the corrected test passes rather than being relabelled.

Schema1/2/native plans and Swift drafts carry the new selector. The candidate
App has separate encoder/decoder pickers and large-IO/numerical warnings; its
memory-advice button does not automatically select spilling. New numerical
recipes append `+vae-encoder-head-globalstats-file-rows-v1` before any decoder
suffix, and separate encoder counters report actual use/retirement. App and
full-request quality/lifecycle qualification pass for the workflows below; admission and
121-frame limits are unchanged. The previously packaged App is not replaced.

### Combined App/engine/package qualification

Two seven-request candidate App workflows complete, covering full/default and
encoder+decoder file rows, staged/cache1, source/sink streaming,512/1024 real
images/complete25-frame videos, persistent history and resident/staged/resident
switches. Default exports are byte exact to the previous full control, including
every25-frame video pixel. Combined outputs pass unchanged per-frame .05/.999
gates and exact PNG alpha: at1024 image L2/cosine is .000758410/.999999712;
H264 exported video is .014964521/.999888031, including lossy codec effects.
512 outputs are pixel exact. These are App workflow tests, not GUI clicking or
broad temporal/perceptual qualification.

Single full/combined request observations, **not warmed ABBA**:

| App workload | Full / combined seconds | Full / combined active peak (bytes) | Full / combined sampled process peak (bytes) |
| --- | ---: | ---: | ---: |
| 1024 image | 4.349 / 4.026 | 9,678,605,142 / 8,770,522,486 | 9,793,029,464 / 9,715,914,968 |
| Complete1024/t25 | 90.043 / 272.910 | 13,215,304,526 / 12,235,734,666 | 14,474,429,856 / 13,309,974,960 |

The earlier decoder-only row App already peaked at12,235,730,826 active bytes.
Adding encoder rows does **not** materially reduce this staged request's overall
peak; remaining phases/retained banks dominate, but this observation does not
locate them by phase. Do not convert encoder component gains into a whole-Mac
budget or16-GiB admission claim. Video is intentionally much slower and adds
substantial cumulative storage traffic. Defaults and conservative admission stay
unchanged; manual choices remain independent.

Actual public engine tests cancel after the initial encoder head chunk
(status2, no DiT/export event), reject64-MiB scratch quota during encoding
(status1, explicit budget error), then recover on the same engine with1024 image
(status0). Source and private existing destination hashes are unchanged; all
scratch/media staging drains and unload MLX active is zero.13 contract tests,
eight observer guards, encoder reference/ownership guards, and release guards
pass. The normal build explicitly rejects the unqualified fused-upscale selector;
packaging additionally rejects its compile flag or actual shader bytes.

Independent locally signed package:
`dist/seedvr2-encoder-spatial-20261015/TurboCider.app`, minimum macOS26.2. Its
actual packaged library passes all13 contracts, and packaged CLI generates a
real1024 encoder+decoder-row image with64/144 tiles, four/eight global norms and
clean retirement. The PNG is byte exact to the tested App workflow (SHA256
`89a5e66d72631c1dfa61bc20356173dff674e676dcfd41e3392e3f3dc25ab808`).
This image smoke is not a packaged-video or GUI-click test. Original user App
and previous packages stay intact.

Encoder records under `outputs/`:

- `seedvr2-spatial-encode-image1024-v1-20261015/report.json`
- `seedvr2-spatial-encode-real-video1024-t25-v1-20261015/report.json`
- `seedvr2-spatial-encode-image1024-abba-v1-20261015/report.json`
- `seedvr2-spatial-encode-{image1024,real-video1024-t25}-fp32-oracle-v1-20261015/report.json`
- `seedvr2-spatial-encode-{image1024,real-video1024-t25}-fp32-quality-v1-20261015.json`
- `seedvr2-app-encoder-decoder-file-rows-staged-e2e-v1-20261015/generation-report.json`
- `seedvr2-app-encoder-spatial-default-control-e2e-v1-20261015/generation-report.json`
- `seedvr2-app-encoder-spatial-default-and-quality-v2-20261015.json`
- `seedvr2-encoder-spatial-engine-lifecycle-v1-20261015/report.json`
- `seedvr2-encoder-spatial-packaged-image-{request,result}-v1-20261015.json`

The encode screen/cohort/reference/comparison tools expose `--help`; build
isolated probes with `tools/native/build_seedvr2_spatial_probes.sh`. Scope guards
reject unfinished/F16/different-source FP32 reports and incomplete head/frame/
cleanup observations. Low-memory hardware, combined App performance, broader
encoder stage scheduling and genuinely longer-context DiT remain open.

## Numerical evidence and precision correction

The unchanged whole-pixel gate is finite values, cosine >= .999 and relative
L2 <= .05. Against saved default FP16 decoder pixels, relative L2 is about
.0002205 at 128/t9, .0002085 at 1024/t1 and **.000259314** at restored 1024/t5.
The latest t5 output has 3,707,890 unequal FP16 words out of 15,728,640; this is
not relabelled bitwise equality. Default decoding remains exactly equal to its
saved witness, and all requested frames are counted explicitly.

Inspection found that the earlier `seedvr2_oracle.py` restore `dtype=fp32`
describes the **DiT**; it explicitly keeps the VAE at FP16. Its saved pixel
datatype is F16. Those reports remain intact and must not be treated as an
all-FP32 VAE witness. Future reports now make `vae_compute_dtype=fp16` explicit.
The first new screen correctly rejects that F16 witness instead of accepting
its report label; its failed preflight directory is preserved.

A separate `seedvr2_vae_fp32_decode_oracle.py` run uses the pinned local Comfy
VAE implementation on MPS, exact original FP16 coefficients promoted to FP32,
and exactly the same native latent prefix. It starts no server and changes no
Comfy/user files. Parameters and every floating dispatch input/output are audited
as FP32: 5,435 dispatch calls / 12,504 floating values checked, output F32
`[1,3,5,1024,1024]`. MPS fallback is prohibited. Its 44.696-s audited execution is
**not** a performance benchmark.

Against this genuine same-latent FP32 VAE reference:

| Decoder route | Pixel relative L2 | Pixel cosine |
| --- | ---: | ---: |
| Default | .0004891896946 | .9999998830215 |
| Row/file prototype | .0004892422334 | .9999998829584 |

Both pass the unchanged gate. This isolates VAE math on the same latent, not
DiT/noise/entire-restoration equivalence. No production/default promotion follows
from a five-frame pixel gate alone.

The complete t25 same-latent audit subsequently succeeds with FP32 MPS
arithmetic and the wrapper's CPU output-buffer storage: all parameters and
62,316 floating dispatch values / 27,000 dispatch calls are checked as FP32;
the result is F32 `[1,3,25,1024,1024]`. The separate full25 comparison checks
every emitted pixel and reports default/row-file relative L2
**.0005223236456 / .0005224389209**, cosine
**.9999998685028 / .9999998684521**, passing the unchanged gate. The reference's
217.558-s audited decode is not a comparable performance benchmark. This
retains all causal context; only final output storage changes to CPU.

The earlier full25 FP32 reference attempt exits 137 without usable output.
`termination.json` preserves that failure; its cause is not established and
it is not called an OOM or relabelled as a passed gate. The successful distinct
CPU-output attempt has its own directory/report and output identity.

## Ownership, integrity and cleanup

`SpatialFiles` uses mode-0700 `mkdtemp` scratch, relative numeric frame names,
exclusive no-follow file creation, pinned inode/device/extent and uncached IO.
It never recursively deletes a caller directory, source/cache, unknown child or
replaced inode. Files are scoped to one decode/cache; retirement preserves a
still-consumed old causal frame. Initial duplicate history slots share the same
identical file; subsequent singleton updates retain only the correct old tail.

Writes are contiguous/coverage-checked and sealing requires every row. Each row
has a SHA-256 digest checked on load. FP16 words, NaNs and signed zeros roundtrip
without conversion. Frame rows load into page-aligned, bounded owned windows
through the supported MLX allocator buffer API; their allocation owners stay
alive until GPU consumers retire. Maximum read request is 128 MiB. File lifetime
and GPU-window lifetime are separate. Model coefficients are never modified.

Default live file budget is capped at 8 GiB, with an additional 256-MiB available
disk reserve. Quota, incomplete frame, overlapping write, wrong extent, checksum
and cancellation failures reject; they do not silently fall back or approximate.
No storage availability/SSD wear/time guarantee is made, and cumulative IO must
be considered before selecting this for longer clips.

Tests pass lossless payloads, halo reads, checksum corruption, quota, typed
cancellation, loaded-window ownership after file retirement, unknown-child
preservation and empty scratch cleanup. Ten normalization cases, including
constant/near-constant, multiple frames and strided inputs, match old affine
outputs byte for byte. A spatial decoder cancellation/sink-failure screen
drains file/window counters and then recovers on the same VAE. Cold/warm cache,
full native engine/App lifecycle were not covered by these initial component
tests; the follow-up below covers the specified 1024 workflows. Broader media
and low-memory hardware remain open.

### Public engine and App follow-up

The candidate's full/file-row controls each complete seven real App job-store
requests: 512/1024 images, complete 512/1024 t25 videos at source24 FPS, and
resident/staged/resident switches. Request/draft/history persistence, selected
recipes, original checkpoint identity, model unload, streamed frame counts and
zero live files/windows/cleanup failures are asserted. The 1024 image executes
144 tiles / eight global norms with no history files; the video executes 3504
tiles / 200 global norms and the measured lossless history IO above. Below the
eligibility threshold, the 512 outputs remain pixel exact to the full route.
These are App workflow integration tests, not automated clicking of the GUI.

The independent exported-media screen checks every PNG RGBA value and every
one of the 25 decoded video frames, dimensions/FPS, unchanged .05-relative-L2 /
.999-cosine thresholds on each frame, and exact PNG alpha separately. At1024,
image relative L2 is .000351064, cosine .999999938; exported H264 video relative
L2 is .014699141, cosine .999891968 (worst frame .020319083 / .999793576).
The latter includes lossy codec effects; it is not raw FP32 VAE error or byte
equality, and per-frame numerical thresholds alone are not a broad temporal/
perceptual quality qualification. The separate same-latent all-FP32 reference
above remains the VAE arithmetic evidence.

Single full-request observations, both staged/cache1, streaming source/sink and
frame history, are **not warmed ABBA**:

| App workload | Full / file-row seconds | Full / file-row active peak (bytes) | Full / file-row sampled process peak (bytes) |
| --- | ---: | ---: | ---: |
| 1024 image | 4.435 / 4.338 | 9,678,605,142 / 8,770,518,646 | 9,791,735,152 / 9,757,825,096 |
| 1024/t25 video | 89.902 / 221.645 | 13,148,195,662 / 12,235,730,826 | 14,476,445,112 / 13,240,293,640 |

The full-request gain is much smaller than the decoder-only reduction; these
request-level samples do not identify the remaining peak by phase. File rows
remain a manual, slower option, without an automatic/16-GiB/hard-budget claim.
The conservative admission estimate has not been lowered.

The actual public C ABI cancels at the first decoded/exported frame (typed
status2, no later export), rejects a 64-MiB live-file quota (status1, explicit
budget error), then succeeds on a1024 image using the same engine. Source hashes
and a preexisting private destination sentinel remain unchanged; owned scratch
and media staging disappear after each request. Unload reports zero MLX active
bytes. The current user App/process is not replaced by these tests.

### Independent package

The full candidate App and CLI are locally ad-hoc signed and verified at
`dist/seedvr2-spatial-rows-20261014/TurboCider.app` (minimum macOS 26.2, inherited
from the installed MLX binary). The public-only/non-instrumented release
guard passes. All 12 contract tests pass against the packaged dynamic library,
not just the build directory. Its packaged CLI completes a real 1024 file-row
image request, executes 144 tiles/eight global norms, drains all file/window
counters and produces a PNG byte-identical to the App workflow's file-row
output (SHA256 `8d297dce6fdb47ffba74cca49832e4823cb9282488aec07f19957ceea3adc225`).
The signing/loader-path changes alter the packaged library's file identity;
the separate smoke receipt records the actual packaged execution. GUI clicking
and a full packaged-video rerun are not claimed by this image smoke test.

## Artifacts and next gates

Generated artifacts stay under `outputs/`, not in source commits:

- `seedvr2-spatial-decode-128-{baseline,filetiles}-v1-20261014/report.json`
- `seedvr2-spatial-decode-image1024-{baseline,filetiles}-v1-20261014/report.json`
- `seedvr2-spatial-restored-video1024-t5-v1-20261014/`: rejected old F16 “fp32”
  witness during host preflight; not a passed/complete screen.
- `seedvr2-spatial-restored-video1024-t5-v2-20261014/report.json`: initial IO.
- `seedvr2-spatial-restored-video1024-t5-batched-io-v3-20261014/report.json`: latest
  same-input original-FP32 gate, numerical/export/source hashes and observations.
- `seedvr2-spatial-video1024-t5-fp32-vae-oracle-v1-20261014/report.json`
- `seedvr2-spatial-video1024-t5-genuine-fp32-quality-v1-20261014.json`
- `seedvr2-spatial-video1024-t5-genuine-fp32-quality-v2-20261014.json`: latest t5
  batched-IO exports against the same audited reference.
- `seedvr2-spatial-restored-video1024-t25-v1-20261014/report.json`: complete25
  decoder observations and export hashes, not full-request performance.
- `seedvr2-spatial-video1024-t25-fp32-vae-oracle-v1-20261014/termination.json`:
  failed reference attempt (exit 137, cause unknown).
- `seedvr2-spatial-video1024-t25-fp32-vae-cpu-output-oracle-v2-20261014/report.json`:
  successful complete25 all-FP32 VAE reference with CPU output storage.
- `seedvr2-spatial-video1024-t25-genuine-fp32-quality-v1-20261014.json`:
  independent complete25 same-latent comparison; both decoder routes pass.
- `seedvr2-app-{file-rows,spatial-full-control}-staged-e2e-v1-20261014/generation-report.json`:
  two complete seven-request App workflows and single-request observations.
- `seedvr2-app-spatial-export-quality-v1-20261014.json`: every exported frame,
  unchanged per-frame numerical gate, exact alpha and receipt identities.
- `seedvr2-spatial-engine-lifecycle-v1-20261014/report.json`: actual cancellation,
  quota rejection, preserved destination/source, cleanup and same-engine recovery.
- `seedvr2-spatial-packaged-image-smoke-v1-20261014.json`: packaged image request.
- `seedvr2-spatial-packaged-image-smoke-result-v1-20261014.json`: actual packaged
  execution/event receipt; PNG is byte exact to the separately tested App image.

Build isolated probes with `bash tools/native/build_seedvr2_spatial_probes.sh`
and a fresh `TURBOCIDER_BUILD_OUTPUT_DIR`. The screen/FP32-oracle/comparison tools
expose `--help`. CPU oracle guards reject old FP16 reports, missing dtype audits,
unfinished screens and different latent identities; reference thresholds are
unchanged. Existing canonical App/native binaries and request defaults are
untouched; public configuration is packaged in the independent candidate.

Next: broader App media/t121/temporal-perceptual qualification, balanced
full-request performance and conservative admission. Extend scheduling
to additional operators/resolution/encoder phases and bounded full-context DiT
for genuinely longer videos. Genuine 16-GiB operation remains unproven; reducing
a decoder component on a 64-GiB host does not complete the entire objective.

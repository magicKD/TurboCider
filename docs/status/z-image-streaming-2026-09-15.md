# Z-Image Turbo BF16 explicit streaming experiment

Implemented an opt-in `residency: "streamed"` path for the Comfy BF16 checkpoint.
The client exposes it in **高级参数 → 模型驻留 → 流式加载（实验）**, with
6/8/10/12 GiB denoiser planning budgets. The default remains `resident`.
The currently installed BF16 checkpoint works without conversion or duplicate
weight files. Diffusers directories/shards, quantized checkpoints, LoRA and
GPU+ANE are rejected by this first streaming implementation.

## Execution and memory ownership

- Read and validate only the safetensors header initially. Reject malformed
  shapes/dtypes, overlaps, truncated ranges and mismatched layer layouts.
- Reuse the H3/LTX shared block-residency planner. Keep embeddings, the two
  noise refiners, two context refiners, output projections, and a budgeted
  prefix of the 30 main Transformer layers resident.
- Allocate two sets of MLX-owned shared BF16 buffers once. A background reader
  fills the next layer directly with chunked `pread` while the GPU evaluates
  the current layer. No MLX calls run on the reader thread.
- `F_NOCACHE` avoids deliberately retaining a second full weight working set
  in the file cache. Reported read bytes describe application reads, not
  measured physical SSD traffic.
- Evaluate each streamed block output before its buffers can be overwritten.
  Reuse the same fused GPU block implementation as resident inference; no
  quantization, weight approximation or sampling changes are introduced.
- On a changed prompt, release the streamed image working set before loading
  Qwen3. Keep it reusable for an unchanged prompt. On error/cancellation, drain
  GPU work and join the outstanding reader before resetting buffers.
- During streaming only, restrict the MLX allocator cache to the request's
  existing cache setting (512 MiB by default). Restore the previous global
  setting when the request exits, including errors and cancellation. Resident
  execution keeps its existing cache policy. Reset MLX peak telemetry at each
  request; older Z-Image results recorded a lifetime peak.

The budget is a denoiser working-set estimate, including reserves, not an OS
memory limit. It does not promise that the whole process will fit that budget.
For example the BF16 Qwen3 text stage alone peaks around 7.7 GiB here, even
when selecting a 6 GiB denoiser budget. OS/other-app pressure still matters.
`pinned_blocks` means retained arrays, not wired physical pages: this change
does not call `set_wired_limit` and cannot guarantee the absence of OS paging.

## Local measurements

24 GiB Apple Silicon host; native MLX runtime; same engine session; no other
model loaded in the TurboCider GUI. The computer remained an interactive
workstation rather than an isolated benchmark machine.

Initial experiment: 512×512, 8 steps, seed 42, one fixed fox prompt. At this
stage both modes used a 512 MiB allocator cache; the PR cleanup subsequently
scoped that override to streaming. Preserve these numbers as the original
experiment, rather than a benchmark of the final resident cache behavior.

| Run | API wall time | Denoise | MLX active peak | Stream read wait |
|---|---:|---:|---:|---:|
| Resident, first request | 19.32 s | 9.23 s | 14.93 GiB | — |
| Streamed, switch into 10 GiB mode | 11.60 s | 9.53 s | 9.87 GiB | 2.69 s |
| Streamed, same prompt and buffers | 10.50 s | 9.73 s | 9.87 GiB | 2.78 s |
| Resident, switch back, cached prompt | 11.02 s | 8.00 s | 14.93 GiB | — |

The streaming plan retained 13 main layers and streamed 17 through two slots.
The repeated request made 136 refills, requesting 49.21 GB (45.83 GiB) of layer
data. File-read time was 6.58 s, with 2.78 s spent waiting on unfinished reads;
these are overlapping measurements and must not be added to wall time.

All four PNGs and the post-cancellation retry are byte-identical (SHA-256
`1177f0a9492b183fdada99dbe701ec7dc83b0bc6876c6d560ff2f569f7525e9c`).
This verifies final PNG parity for these inputs, not a comprehensive latent
numerical-equivalence or model-quality suite.

MLX active peak dropped 33.9%. Warm sampling itself was slower in streaming
mode, while end-to-end times were similar in this sequence. The cold resident
request included text encoding and incurred 7.63 GiB of **system-wide**
swap-out counter growth; streaming runs had zero system-wide swap-out growth.
These counters cannot attribute individual swapped pages to TurboCider, and
the first/warm timings do not establish a universal speedup.

The two streaming requests still recorded system-wide swap-in growth of
0.264 and 0.065 GiB respectively. Zero swap-out is not evidence of zero
swap-in or zero page-fault stalls. The initial resident request's median/max
step times were 0.82/3.51 s; repeated streaming's were 1.21/1.32 s. Eight steps
per request and differing warm-up states do not establish a tail-latency win.

A second suite used 256×256, 2 steps, seed 314159 and a 6 GiB budget. It retained
two layers and streamed 28. Resident/streamed PNGs matched, including after a
prompt change. Same-prompt streaming took 2.94 s with a 4.45 GiB MLX peak;
fresh text encoding raised request peak to about 7.69 GiB. Prepare-only,
cancellation during sampling and retry also passed.

The rebuilt, packaged macOS client also generated the same byte-identical PNG
twice with streaming selected and a 10 GiB budget. The repeat request completed
in 16.0 s in the UI, reused the text encoding and stream buffers, and retained
the 9.87 GiB MLX peak. This was a separate interactive run with other GPU
activity, not part of the paired timing comparison above.

The first client request waited in model inspection before native execution.
Thread samples showed blocked file opens; macOS TCC logs reported that the
rebuilt ad-hoc signature no longer matched the old Desktop-folder grant.
The file access subsequently completed without changing privacy settings;
native generation took 21.27 s, but the UI total included the long preflight
wait. The second request entered sampling directly. Do not attribute that
first UI wait to weight streaming or use its elapsed time as inference timing.

## PR cleanup and verification (2026-09-16)

The PR is based directly on `dev`; the separate client-download changes are
not included. The cache override now applies only to streaming and restores
the previous setting on every exit. Native peak reporting remains per request.
Client layer-count telemetry uses checked integer conversion.

A follow-up overnight run passed image parity and cache restoration, but its
native and Python durations disagree by large intervals. Preserve that suite
as `pr-cleanup` with invalid comparison timing; do not use it for speed claims.

The final `pr-final` suite inhibited idle sleep and used the same 512×512,
8-step, seed-42 workload. Both clocks agreed:

| Run | API wall time | Denoise | MLX active peak | Stream read wait |
|---|---:|---:|---:|---:|
| Resident, first request | 14.21 s | 6.47 s | 14.93 GiB | — |
| Streamed, switch into 10 GiB mode | 10.45 s | 8.27 s | 9.87 GiB | 2.97 s |
| Streamed, same prompt and buffers | 8.99 s | 8.32 s | 9.87 GiB | 3.02 s |
| Resident, switch back, cached prompt | 8.18 s | 5.81 s | 14.93 GiB | — |

All generated images matched within each prompt group, including cancellation
retry and a changed prompt. Cache restoration passed after every generate,
prepare and cancelled request. These runs confirm the memory reduction with
the final cache scope; warm resident sampling remains faster in this sample.

Validation includes native compilation, the macOS App build, 70 request
contract tests (3 fixture-dependent skips), real-GPU streaming buffer tests,
Studio behavior and RunInsights tests. Full `make test` reaches the unchanged
VDN solve test and fails its `5e-4` maximum absolute error threshold, also
reported before this change in PR #2. It is not a fully green repository suite.

Portable measurements, step timings, VM deltas and image hashes are preserved
in [the checked-in JSON record](z-image-streaming-2026-09-15.json). It omits
machine-specific paths and unrelated runtime-plan fields. Raw local reports
and images remain available separately:

- `artifacts/z-image-streaming-20260915/paired-512/report.json`
- `artifacts/z-image-streaming-20260915/low-budget-256/report.json`
- `artifacts/z-image-streaming-20260915/client-first.json`
- `artifacts/z-image-streaming-20260915/client-reused.json`
- `artifacts/z-image-streaming-20260915/pr-cleanup/report.json`
- `artifacts/z-image-streaming-20260915/pr-final/report.json`

These raw generated artifacts are local and are not checked into source control.

## Verification boundary

The current evidence covers bounded buffers, exact PNG parity for the tested
inputs, preparation, cancellation/retry, prompt changes and cache restoration.
It does not establish that asynchronous reads eliminate swap-related stalls.
To isolate that claim, compare resident, synchronous streaming and asynchronous
streaming under matched memory pressure and warm-up conditions; the synchronous
control is not yet implemented. Record per-layer read, wait and GPU execution
intervals, and compare first-step, median, P95 and maximum step latency across
repeated runs. Correlate stalls with paging activity; whole-system counters
alone cannot attribute them to this process.

## Reproduction

```sh
TURBOCIDER_TEST_GPU=1 .venv/bin/python3 tests/native/test_z_image_weight_stream.py
.venv/bin/python3 tests/native/test_contract.py
.venv/bin/python3 tests/repository/test_cpp_boundaries.py
.venv/bin/python3 tools/native/benchmark_z_image_streaming.py \
  --model /absolute/path/to/Comfy-Z-Image-installation \
  --output /tmp/z-stream-validation \
  --modes resident,streamed,streamed,resident --budget-gib 10 --lifecycle \
  --check-cache-scope
```

Run on a host with Metal access. `examples/requests/z-image-turbo-streamed.json`
provides the request format. Result `block_residency` telemetry reports retained
and streamed layers, fixed slot allocations/refills, read bytes, accumulated
read duration and uncovered read wait. System swap deltas in the benchmark
remain explicitly system-wide.

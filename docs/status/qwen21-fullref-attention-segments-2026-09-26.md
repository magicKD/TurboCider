# Qwen Image 2.1: exact long-reference attention segment screen

On the M4 Max, 512² output, three ordered full-size 1024px references,
seed 17 and five steps, the unchanged BF16 GPU first step handles 12,288
reference tokens plus 130 text and 1,024 output-image tokens. The opt-in
`TURBOCIDER_QWEN21_PROFILE_PREFILL_SEGMENTS=1` times each first-step
attention segment **in block 0 only**. It materializes Q/K/V once, forces
one GPU completion barrier per attention segment, and disables compilation
for that single block while timing. It prints JSON to stderr with kind,
query/key rows and seconds. Normal requests do none of this. The flag is
accepted only on explicit 512² BF16 GPU edits with 1–3 full-size references,
without LoRA, reference-local attention or resident prefix-KV reuse. It
cannot be combined with the existing GPU block/operator profilers.

In one independently prepared resident Session, after a two-step warmup,
the matched prompt-cached request measured:

| Block-0 first-step segment | Query × visible K/V rows | Synchronized attention |
| --- | ---: | ---: |
| Text before reference 1 | 8 × 8 | 2.48 ms |
| Reference 1 | 4096 × 4104 | 21.47 ms |
| Text before reference 2 | 6 × 4110 | 0.55 ms |
| Reference 2 | 4096 × 8206 | 42.10 ms |
| Text before reference 3 | 6 × 8212 | 0.79 ms |
| Reference 3 | 4096 × 12308 | 65.63 ms |
| Remaining text | 110 × 12418 | 3.09 ms |
| Output image | 1024 × 13442 | 17.82 ms |

The three exact reference attentions total about 129 ms of the ~154 ms
eight-segment block-0 sum; their cost increases with visible K/V rows.
This diagnoses a useful next target, **not** a 32-block speedup estimate:
forced per-segment barriers, block-0 decompilation, different layer inputs
and other GPU work prevent extrapolating these numbers to the uninstrumented
18-second first step. The request wall was 25.150 seconds with 24.512 seconds
denoise. Its PNG SHA-256
`fb9a24d697f2156ecef092d5e7b49638a2c8e7c0b4bd8fe310d1792c4c5f8560`
matches the saved uninstrumented same-input pure-GPU baseline exactly.
The probe's cancellation/GPU-switch/unload lifecycle passed. Ignored
request/PNG/dump/JSON receipts are in
`results/qwen21/session-fullref-edit3-segment-profile-gpu-20260926/`.

An isolated, **reverted** FP16 attention candidate cast the already
materialized BF16 Q/K/V inputs to FP16 for each image segment, executed the
same SDPA, then cast the result back to BF16. It did not replace the real
BF16 path used to produce the PNG. On a second same-input, same-device
profiled request the reference/image segment pairs were:

| Segment | BF16 attention | FP16 casts + attention + output cast | Output relative L2 |
| --- | ---: | ---: | ---: |
| Reference 1 | 20.87 ms | 23.21 ms | 0.001416 |
| Reference 2 | 41.65 ms | 45.44 ms | 0.001372 |
| Reference 3 | 63.87 ms | 68.40 ms | 0.001369 |
| Output image | 17.73 ms | 22.12 ms | 0.001315 |

All four FP16 cast-and-attend branches are slower at these measured
boundaries. The request took 25.390 seconds but ran **both** BF16 and FP16
attention in block 0; this is not a candidate end-to-end speed comparison.
It exported a BF16 PNG byte-identical to the uninstrumented baseline and
passed the probe lifecycle. Its ignored receipts are in
`results/qwen21/session-fullref-edit3-fp16-attention-screen-20260926/`.
Only the exact segment profiler is retained; the slower FP16 screening code
was removed. This result does not exclude a separately exported native
FP16 model with different casting and quality costs, which was not tested.

After removing the FP16 screen, the native library and matching probe
rebuilt and `make test-qwen21` passed. With both profiler flags **off** a
fresh matched five-step three-reference GPU request took 25.201 seconds,
passed cancellation/GPU-switch/unload and exported a PNG byte-identical
to the historical uninstrumented GPU baseline. The ignored control is
`results/qwen21/session-fullref-edit3-segment-profile-off-gpu-20260926/`.

Next: benchmark an exact Metal attention/KV layout candidate at the same
segment geometry, measuring GPU completion **and** full 5/40-step edit
requests. Retain all prefix-to-reference dependencies: the prior
reference-local shortcut visibly changes sticker/teapot fidelity.

# Qwen Image 2.1: fused Q/K norm-RoPE in W8A8 hybrid (2026-09-26)

The existing opt-in `TURBOCIDER_QWEN21_METAL_QK_NORM_ROPE=1` Metal kernel
now also accepts explicit **512² W8A8 Core ML + BF16 GPU** requests with
`allow_approximation=true`. The default route and the 32-layer ANE FFN
partition are unchanged. Hybrid W8A16 GPU suffix, unquantized Core ML,
1024² requests and simultaneous paired-RoPE mode remain disallowed for this
combination. The kernel fuses Q/K RMSNorm, RoPE and layout on the GPU; it
does not accelerate the Core ML FFN or eliminate long-reference attention.
Floating-point operation order changes, so this is an approximation.

One M4 Max, checkpoint-matched 6,144-channel / 1,024-row / 32-block W8A8
manifest, three ordered **full-size 1024px** references, 512² output,
seed 17. Each arm used a separate resident Session, a two-step warmup,
cached prompt/reference conditioning and the same request tensors.
`TURBOCIDER_QWEN21_FULL_REF_W8A8_DIAGNOSTIC=1` allowed this full-reference
edit. Each request wall includes VAE decoding, PNG export and equal tensor
dumping, but excludes model preparation. The two 5-step samples are not an
ABBA interleaving; the 40-step samples have only one run per arm.

| Steps | Original hybrid wall | Fused-QK hybrid wall | Ratio | RGB correlation / RMSE ([0,1]) |
| ---: | ---: | ---: | ---: | ---: |
| 5 | 24.262 / 24.151 s | 23.682 / 23.729 s | median **1.021×** | 0.99637 / 0.02886 |
| 40 | 71.450 s | 71.306 s | **1.002×**, indistinguishable from noise | 0.98556 / 0.05369 |

The verifier checked text, initial noise and each of the three reference
latent tensors byte-for-byte. It also checked the same ANE row bucket,
partition and successful runtime-call totals for both arms: 128 per 5-step
request and 1,248 per 40-step request, with zero reported output copies.
The probe passed cancellation and GPU-switch lifecycle checks and exited 0.
`CPU_AND_NE` is a Core ML selection policy, not proof of physical ANE MACs.

Visual inspection: both five-step images retain the two teapots and central
sticker, but both show sticker ghosting; eye/outline detail changes with
fusion. At 40 steps the two teapots, spouts/handles, and orange sticker remain
recognizable in the same composition; the sticker face and small contours
change. This is not a strict likeness/identity pass or a broad image-quality
gate. Keep the fusion explicit; in particular **do not promote it as a
40-step optimization**. Even if the short-step 2% direction replicates,
it does not close the three-reference gap to 1.3× over GPU.

The 5/40-step matched reports are ignored under
`results/qwen21/session-fullref-edit3-hybrid-fusedqk-{20260926,40-20260926}-comparison.json`;
the underlying candidate/control run directories use the same stem with
`-candidate-` or `-control-`. `tools/validation/qwen21_compare_sessions.py`
now accepts identical `gpu_ane_experimental` arms for this particular fused
candidate and asserts identical Core ML metadata and approximation labels.
The native build and `make test-qwen21` passed after initial integration;
the small final admission restriction was rebuilt and retested separately.
No input image, model, compiled Core ML cache or generated output is committed.

Next: collect no-instrumentation ABBA runs for short steps if this 2% is
worth promoting; prioritize first-step long-prefix GPU attention and
attention/KV memory traffic before tuning Q/K microkernels further.

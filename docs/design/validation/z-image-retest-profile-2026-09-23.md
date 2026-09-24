# Z-Image 512 retest and operator profile — 2026-09-23

> **Invalid for performance analysis.** This was recorded before the
> 2026-09-23 reboot in the degraded system state. Retain only as diagnostic
> history; use `z-image-clean-performance-2026-09-23.json` for timings.

This follow-up uses the same M4 Max 40-core GPU, 64 GiB machine and the
greenhouse prompt / seed 123 workload used by the 2026-09-22 qualification.
The machine had not rebooted: `kern.boottime` remained September 18, 2026.
No Z-Image benchmark process was left running before the control.

## Current performance did not recover spontaneously

Current default BF16, pure GPU, resident, 512x512, 8 steps, two warm samples:

- diffusion: 14.745974459 and 14.738976834 seconds;
- diffusion median: 14.7424756465 seconds;
- request wall median: 20.0162096045 seconds;
- VAE median: 5.015558 seconds.

Raw report: `/private/tmp/z-image-retest-20260923`. The loaded native, MLX and
JACCL hashes match the pre-control reports. Killing the user-level aerial
wallpaper and VideoToolbox decoder processes did not form a clean control
because launch services immediately restarted both. The subsequent warm sample
was 14.708028584 seconds diffusion / 4.924030333 seconds VAE, normal run noise
rather than historical recovery.

## Detailed 512 block attribution

Added an opt-in `TURBOCIDER_Z_PROFILE_MODE=gpu_detail` diagnostic. It bypasses
the compiled full-block graph and inserts explicit completion boundaries after
each operation, so its 17.929-second instrumented diffusion time is not a
production timing. It preserves production operations and is useful only for
relative attribution. The 240 main blocks in the warm request measured:

| Phase | Share | Median per block |
| --- | ---: | ---: |
| QKV projection | 20.8% | 15.4859 ms |
| gate projection + fused SwiGLU | 18.0% | 10.6898 ms |
| FFN down projection | 13.9% | 7.8530 ms |
| FFN up projection | 11.4% | 6.7709 ms |
| attention residual + feed norm | 9.6% | 6.7411 ms |
| attention output projection | 9.4% | 5.9160 ms |
| Q/K norm + RoPE + layout | 4.4% | 1.5974 ms |
| final gate + norm | 4.3% | 1.9339 ms |
| fused SDPA | 3.7% | 2.3393 ms |
| attention input norm/modulation | 2.5% | 1.0184 ms |
| modulation projection/split | 1.9% | 0.9798 ms |

Raw trace: `/private/tmp/z-image-gpu-detail-512-v2-20260923.jsonl`.
This shows that SDPA is not the main remaining bottleneck; dense QKV/FFN
projections dominate.

## Rejected projection candidates

All comparisons below use one warm sample per process in ABBA order, with the
production default as control. They are screens, not promotion qualifications.

- MPP QKV-only: 14.836996229 -> 15.157495667 seconds, 0.9789x. Rejected.
- MPP small FFN-down 16x256: 14.735920938 -> 14.790967292 seconds,
  0.9963x. Rejected.
- FP16 QKV weights/compute with immediate BF16 output boundary:
  14.772470688 -> 16.253954209 seconds, 0.9089x. Rejected.
- FP16 FFN-down produced a nonfinite latent in the first denoise step, including
  in the one-step isolation check. Rejected before any speed or quality claim.

The rejected model switches were removed. Shape filtering added to the GEMM
probe remains because it reduces diagnostic runtime without changing kernels.

## Retained MLX runtime comparison

The machine contains multiple different `libmlx.dylib` hashes reporting the
same broad 0.32-era version. An ABI-local minimal BF16 GEMM diagnostic, compiled
against each library's matching headers, found that the September 5 library was
faster for isolated FFN shapes. That did not translate to the model.

The complete current source was independently built against the retained
September 5 MLX library (`d24c7a9b...`) and tested on the same 512 workload:

- retained MLX diffusion median: 14.7709637915 seconds;
- current MLX diffusion median: 14.7424756465 seconds;
- all eleven parity tensor files and the final PNG were byte-identical.

Raw retained-runtime report: `/private/tmp/z-image-sep05-mlx-512-20260923`.
Therefore an MLX library rollback does not restore the historical model speed.

## Current conclusion

The current optimized BF16 path remains approximately 14.7 seconds for
512x512 / 8 steps on this machine. The historical same-machine 1024x1024
workload was rerun exactly with the final clean default library: 1024x1024,
9 steps, fox prompt, seed 42, pure GPU, resident BF16 and compiled full blocks.
The warm request measured **75.466043834 seconds diffusion**, **20.533000792
seconds VAE** and **97.071632917 seconds request wall**. The September 7
historical values were 35.437059625 seconds diffusion, 0.857119583 seconds VAE
and 36.359820875 seconds request wall. The historical absolute performance did
not recover.

A final same-library 256-thread approximate norm ABBA also failed to improve:
14.8064731455 seconds default versus 14.8739464165 seconds candidate (0.9955x).
The candidate PNG hash differed from the default. No approximate norm setting
was promoted.

The simultaneous historical/current DiT and VAE gaps still point to a
machine/runtime-state issue broader than a single DiT kernel.
AC power, low-power mode, thermal/performance warnings and the 40-core GPU
identity are normal. The prepared fresh-boot matched-binary control remains the
next clean discriminator; no reboot or driver reset was performed here.

Raw final controls:

- `/private/tmp/z-image-clean-final-512-20260923`
- `/private/tmp/z-image-historical-match-1024-20260923`
- `/private/tmp/z-image-norm256-current-512-screen-20260923`

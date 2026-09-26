# Z-Image Turbo Metal status — 2026-09-23

This is the authoritative performance status. The pre-reboot September 22
timings are invalid for performance analysis; see
`docs/design/validation/z-image-performance-ledger-2026-09-23.json`.

On the qualified Apple M4 Max, 512×512 / 8-step / pure-GPU resident BF16 now
measures 6.7677 seconds median diffusion and 7.0066 seconds median request wall.
Against the same-binary fully unfused control, this is 1.0810× diffusion and
1.0782× wall speedup. Four warm samples per arm were measured in ABBA order;
all denoising tensors, decoded tensor and PNG are byte-identical.

The promoted exact path uses a healthy-state-selected 32×128 MPP SwiGLU tile,
shape-specific projection tiles, a fused QKV projection/norm/RoPE/layout
epilogue, exact virtual gate+norm fusion and request-local invariant context
reuse. Small-shape defaults are restricted to the qualified M4 Max and at most
1056 transformer rows. Every addition retains an environment disable switch.

The final 1024×1024 / 9-step historical-comparison workload measures 34.2020
seconds diffusion, 0.8710 seconds VAE and 35.1408 seconds request wall. The
September 7 healthy reference was 35.4371 / 0.8571 / 36.3598 seconds, so the
current system is operating normally.

The clean 512×512 / 8-step hybrid retest must use a device-specific split.
The historical M4 Pro-oriented `b1056/a8192` candidate is slower on M4 Max
(9.3154 seconds warm wall), while `b1056/a4096` measures 5.8039 seconds versus
7.0045 seconds for optimized pure GPU, a 1.2068× warm speedup. Cold hybrid
requests remain slower (about 11.8 versus 7.6–7.8 seconds) because they include
manifest validation and Core ML model loading. The INT8 route is deterministic
but fails the current aligned-RGB quality gate (correlation 0.9611, MAE
5.397/255), so it remains explicit and is not added to automatic selection.

Full evidence: `docs/design/validation/z-image-clean-performance-2026-09-23.json`.

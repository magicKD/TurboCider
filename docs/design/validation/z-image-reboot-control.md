# Completed fresh-boot control — 2026-09-23

The machine rebooted at 2026-09-23 17:13:16 local time. The first control used
the exact pre-reboot native library SHA256
`ca0012f771341dfe2eef896d89d9b79a68d9ec5ac07b7e04f20737b86f799e67`.

For 512×512 / 8 steps / greenhouse prompt / seed 123, the same binary recovered
from 14.7425 to 7.1501 seconds median diffusion, from about 5.02 to 0.2291
seconds VAE and from about 20.02 to 7.4063 seconds request wall. This confirms
that the pre-reboot performance state was contaminated and invalidates kernel
speedup conclusions derived from its absolute or relative timings.

The historical 1024×1024 / 9-step control also recovered: 34.5441 seconds
diffusion, 0.9004 seconds VAE and 35.5121 seconds wall versus the September 7
healthy reference of 35.4371 / 0.8571 / 36.3598 seconds.

Subsequent optimized results are recorded in
`z-image-clean-performance-2026-09-23.json`.

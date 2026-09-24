# Virtual256 versus production-default: completed ABBA

Started and completed September 22, 2026. Results below supersede the initial
pending status; the filename is retained for existing references.

## Verified results

All four processes exited successfully. Four warm diffusion samples per arm:

- Production default: 72.172924333, 72.440992, 72.133994250, 72.140963333 s.
- Virtual256: 67.089056708, 66.998053792, 66.991981625, 66.921083959 s.

Median diffusion 72.156943833 -> 66.995017709 s: **1.0770494x** speedup.
Median measured request wall 93.636248063 -> 88.465845646 s: 1.0584452x.
VAE medians 20.285543209 and 20.299540021 s, respectively.

Actual loaded native/MLX/JACCL paths and SHA256 records match across all four
processes. Control has no kernel overrides; candidate has only the virtual256
override. All 12 PNGs, including cold outputs, have the identical SHA256
`a8a502441a0628374d3d9d2880ccb7d760ad3776de9094b421926f063cf4ee99`.
This run does not dump intermediate latents; previous same-library exact
latent/decoded-tensor parity is recorded separately in the virtual-norm report.

This establishes a reproducible benefit on this workload in the current
machine state, not recovery of historical performance, benefit at512, or
qualification of all prompt lengths/devices. Default dispatch remains unchanged.

## Original launch metadata

Command (from repository root):

```sh
.venv/bin/python3 tools/native/benchmark_z_image_metal.py \
  --model models/Comfy-Org-z_image_turbo \
  --output /private/tmp/z-image-virtual256-default-abba-1024-20260922 \
  --size 1024 --steps 8 --seed 123 --runs 2 \
  --prompt 'A sunlit greenhouse filled with orchids and ferns, photographic detail.' \
  --control-default --virtual-norm-threads 256
```

Four fresh processes in baseline/fused/fused/baseline order. Here baseline
explicitly means **production defaults**, not the historical unfused control.
Only the candidate receives `TURBOCIDER_Z_VIRTUAL_NORM_THREADS=256`.
Two warm requests per process; each first request is excluded. No new parity
request is included; prior exact parity evidence is separate. No native rebuild
or default dispatch change is made during measurement.

First cold control: denoise 72.081896208 s, VAE 20.221108167 s,
request wall 95.477895 s. Environment overrides empty; loaded native SHA256
742ad2df8ece7903033e242b9ecf1b13dc7250fc1dc3992f2940cb5a7a4a8db2.

Launcher session ID 71775 completed with exit0; original parent PID55032 and
first worker PID55034. The authoritative output is the per-process `report.json` and parent
`summary.json` in the output directory above. Never count a partial schedule
as a completed ABBA qualification.

Benchmark now supports `--control-default`; seven CPU unittest methods passed,
including mocked verification of baseline/candidate environment isolation and
rejection of ignored worker-mode flags. `git diff --check` passed.

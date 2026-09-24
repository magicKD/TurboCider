# Shape-scoped virtual256 promotion: verified rebuild

## Completed checks

Rebuilt production SHA256:
`9f3dd08f7243b4f1128cbd8f057a5eae6f30458bb5f6fda089405b1bbd02cedb`.
All3 GPU unittest methods passed (141.879 s), including4096-row virtual256
coverage. Seven CPU benchmark tests passed.

512/8 default smoke completed: warm diffusion14.778990083 and14.836028125 s;
request-wall metrics19.939767875 and19.994287792 s. These two warm samples
are a dispatch/rebuild smoke check, not a fresh ABBA. All11 parity dump files
are byte-identical to the prior512 default, and the parity PNG SHA256 matches:
`077252d22e0cd8e4259d2dc4ace16fcc7169146668097d39697dafdeab1dfcd4`.
Environment overrides empty; system eligibility true does not mean512 selects
the large-shape kernel.

Session73422 completed with exit0.1024 default warm diffusion67.021032750 s,
wall88.544417500 s; same-library fallback diffusion72.236026792 s,
wall93.422530500 s. The fallback environment contains only
`TURBOCIDER_Z_DISABLE_VIRTUAL_NORM=1`. These are single warm rebuild/dispatch
smoke samples, not a new ABBA or an independent robust speedup qualification.

All11 dump files are byte-identical between rebuilt1024 default/fallback and
the prior explicit256 run. Parity PNGs also match SHA256
`a8a502441a0628374d3d9d2880ccb7d760ad3776de9094b421926f063cf4ee99`.
The source default is promoted for the gated shapes/device/dtype; native
library built and tested, no app packaging. `git diff --check` passed.

## Scope and verification plan

Pre-promotion ABBA: `z-image-virtual256-abba-pending-2026-09-22.md` records
four warm samples per arm, 1.0770494x diffusion benefit at1024/8, matching
runtime fingerprints and all12 PNGs identical.

Implemented automatic selection only on the qualified macOS26+/Apple9/
Apple M4 Max device, BF16 `[1,4096,3840]` and `[1,4128,3840]` tensors.
Other shapes (including512 image shapes) retain the prior default. The4096
image-refiner portion can still select this kernel for longer prompts while
unqualified combined image/text shapes retain the original kernel.

`TURBOCIDER_Z_DISABLE_VIRTUAL_NORM=1` disables automatic selection.
Explicit `TURBOCIDER_Z_VIRTUAL_NORM_THREADS` overrides automatic selection
(including its disable flag). Explicit scalar/reduction geometry, disabled
fusion and disabled vector norm retain their existing behavior. System JSON
`z_image_virtual_norm_default` reports device eligibility, not effective
per-tensor selection. It must not be interpreted as proof that every request
ran the virtual kernel.

Native-only full build completed; no app package built. Seven CPU benchmark
tests passed, including validation of false-looking presence-switch values.
Expanded GPU coverage includes4096 rows for256 threads in both gate modes
and all three dtypes.

Sequential verification completed in exec session73422:

1. Complete GPU kernel unittest suite.
2.512 default: cold + two warm + separate parity request.
3.1024 default: cold + one warm + separate parity request.
4.1024 with automatic virtual norm disabled: same schedule as3.

Output root: `/private/tmp/z-image-virtual-default-promotion-20260922`.
No concurrent GPU work. New512 dump directory was compared against prior512
default; new1024 default/fallback were compared against prior explicit256
dumps. Do not count cold or dump
requests as timings. The new whole-model runs are rebuild/dispatch smoke
checks, not a replacement for the pre-promotion ABBA.

The512-under10s target and historical same-machine absolute-runtime
regression remain unresolved. No new quantization was introduced.

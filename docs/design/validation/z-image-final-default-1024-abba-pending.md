# Final-library1024 cumulative qualification: completed

Session49867 completed exit0. Final audit passed. Diffusion median
131.603520563 ->66.997461417 s (1.9643061x), request wall152.984028229
->88.243298896 s (1.7336617x). Two warm samples per arm. All four parity
requests have eleven identical tensor files and identical PNG; actual loaded
runtime fingerprints match. See `z-image-final-default-1024-abba.json` for
final results and limitations. The pending progress notes below are historical.

Latest verified progress: first A process completed (warm diffusion
131.568994542 s); first B process completed its cold request (67.168067333 s).
Do not compare this warm/cold pair as a qualified speedup. Session49867 remains
live; no duplicate run was launched.

Added `tools/validation/z_image_benchmark_audit.py`, a read-only standard-library
auditor for complete ABBA reports. Verifies actual steps, BF16/pureGPU plan,
workload consistency, actual library fingerprints, per-arm environment,
unfused control switches, exact required tensor files and PNG byte equality,
and warm-only sample accounting. It deliberately does not parse tensors for
finiteness or assess perceptual quality.14 CPU benchmark/audit tests passed.
It successfully revalidated the completed512 ABBA (1.5857466x).
Run on this1024 summary only after the live parent completes; partial schedules
are explicitly rejected. No model code or loaded library changed this turn.

The remainder records original launch metadata, superseded by results above.

Native SHA256:
`349630000153d6b294712f325c1e90b48fb8f11f5cf9048352d1d0c03f84b818`.
Same library as the completed512 cumulative ABBA (1.5857466x diffusion).
No source edits, rebuild or other GPU experiment during this run.

Command:

```sh
.venv/bin/python3 tools/native/benchmark_z_image_metal.py \
  --model models/Comfy-Org-z_image_turbo \
  --output /private/tmp/z-image-final-default-1024-abba \
  --size 1024 --steps 8 --seed 123 --runs 1 --parity \
  --prompt 'A sunlit greenhouse filled with orchids and ferns, photographic detail.'
```

ABBA, four fresh processes. Each process: one excluded cold request, one
timed warm request, one excluded tensor-dump parity request. Two warm samples
per arm at completion. Control disables fused QKV, fused modulation and MPP
SwiGLU; candidate uses current default including qualified virtual256 norms.
BF16, pureGPU, no quantization, same weights. This is not a replay of the
historical September7 binary and cannot establish recovery from its regression.

Live exec session49867 was confirmed by polling; original launcher PID57756,
first worker PID57758. Revalidate that handle/process before claiming a wait,
completion, or restarting. Authoritative reports evolve under the output
directory above. Do not use partial summaries as a completed ABBA result.

After completion verify actual loaded runtime fingerprints, environments,
matching requests, all eleven required dump files plus PNG across four
parity requests, and exclusion of cold/parity samples. Report diffusion and
wall separately. The >=1.5x1024 claim remains unqualified by this run until
those checks complete. The512<10s and historical absolute-runtime issues
are independent and still unresolved.

# LoRA production boundary — 2026-09-09

H3 and LTX production sessions now accept only offline-premerged checkpoints
with matching provenance manifests. Missing or mismatched artifacts fail closed;
sessions no longer create a Python-backed runtime merge cache or consult
`TURBOCIDER_H3_LORA_BASE` / `TURBOCIDER_LTX_LORA_BASE`.

The native Python launcher (`native/platform/apple/lora_cache.mm`) and its
runtime interface have been removed. Model descriptors report
`runtime_lora=false`, `lora_mode=premerged-manifest`; `disk_premerge` remains
the supported request strategy. Existing adapter identity, strength, shape and
checkpoint validation paths remain in place. No inference kernels or arithmetic
were changed in this boundary cleanup.

The merge/cache Python tools remain available in `tools/native` for offline
development and release preparation. The App packaging script no longer copies
LoRA merge/cache scripts into App resources. CLI offline preparation scripts
are still distributed separately in `dist/cli`.

This is not a claim that every App feature is Python-free: Core ML export
still invokes a Python toolchain and its scripts are currently packaged.
Loading/executing prepared Core ML models is a separate native operation.
Separating export UI/API from production inference remains necessary.

This change also does not establish full model quality or speed parity.
Wan UMT5 numerical differences, end-to-end reference comparisons, cold/warm
performance measurements, and a complete fresh-machine packaging check remain
open; see `wan-native-session-2026-09-09.md`.

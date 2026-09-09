# Native component boundaries

TurboCider is being migrated to native, session-owned inference components.
Python remains permitted for development or offline conversion. The migration
now separates H3/LTX disk LoRA preparation and Core ML export into offline
Python tools; production App/CLI do not start those tools. The production Wan
entry uses its native Pipeline. Broader quality/performance and distribution
qualification remain distinct from this architecture cleanup.

LLaDA's production adapter now delegates only to its native implementation.
The environment-selected Python diagnostic branch has been removed; its oracle
is retained under `tools/validation/llada/` and is never packaged.

## Components

- `native/components/text/qwen3.*`: shared Qwen3 conditioning for FLUX Klein
  and Z-Image. The caller selects hidden-state taps and residual precision;
  sharing a backbone does not imply sharing cached prompt outputs.
- `native/components/text/umt5.*`: BF16 bidirectional UMT5 encoder with
  per-layer relative attention bias. No tokenizer or model discovery inside
  the tensor component. Unlike Qwen attention, its scores are not divided by
  the square root of the head dimension.
- `native/core/unigram_tokenizer.hpp` and the Apple adapter: UMT5's explicit
  Unigram/Metaspace tokenizer.json contract. Unsupported normalization and
  special-token rules are rejected. It does not implement arbitrary
  SentencePiece models and must not replace the existing Qwen BPE adapter.
- `native/components/weights/affine.*`: manifest-declared affine matrices.
  Wan's group-64 layout is separate from the supported GGUF tensor formats;
  adding this component does not broaden GGUF acceptance.
- `native/components/diffusion/wan.*`: patch geometry, timestep embedding
  and QAD re-noising arithmetic. QAD uses nearest lookup in the shifted
  training schedule, not a direct shift of the requested timestep.
- `native/models/wan/checkpoint.*` and `dit.*`: native Wan 1.3B checkpoint
  adapter and DiT, composed by `wan::Pipeline` and the Apple `WanSession` adapter.

Weights remain owned by callers. Inference components do not spawn Python,
download assets, select a working directory or infer external toolchains.
The native session owns loading, cancellation, request caches,
Core ML resource lifetimes, decoding and media export.

## Verification tools

`tools/validation/unigram_parity.py` compares the native tokenizer probe with
the supplied tokenizer.json using multilingual, special-token, whitespace,
unknown-character and truncation cases. It needs the development `tokenizers`
package, not a production Python installation.

`tools/validation/wan_dit_parity.py` compares native DiT stages and DMD updates
against an explicitly selected local reference repository. Initial real-weight
fixtures pass exact dtype and tensor comparison, including 480 latent tokens
and 512 text tokens. This is not an end-to-end video quality/performance gate.

`tools/validation/umt5_parity.py` reports the native encoder's differences from
Transformers BF16/MPS. The first real-weight fixture has exact embedding and
position buckets and zero padding, but non-exact encoder outputs; its successful
exit proves only those structural/finite-value gates, not numerical equivalence.
After matching the FP32 GELU scalar boundary, the first report reaches final
block RMSE about 4.98 and maximum absolute difference 320 in BF16 tensors, while
the final normalized conditioning output cosine is about 0.99983. The remaining
attention/projection/FFN precision boundary is not bit-exact. The qualified
832x480x81 native-vs-reference video gate passes, but this does not establish
parity for every prompt, size, or hardware target.

## Remaining production migration

Native three-axis RoPE generation and TAEHV decoding now have standalone parity
probes. The native Wan Pipeline composes these with UMT5 and DiT, and accepts an
explicit hybrid manifest for 832x480x81 only. Hybrid and whole-DiT compilation
are mutually exclusive. `wan2.1-1.3b-qad` is now registered with the native
session and no executable Python profile. Its model package must provide
`vae/taew2_1.safetensors`, produced by the offline `tools/convert/wan_taehv.py`.

UMT5 numerical investigation, end-to-end Core ML Wan qualification, full
session/LoRA lifecycle and production packaging remain to be qualified.
The historical Python worker is now development-only. The native session
explicitly rejects unsupported output sizes, audio and decoder configurations;
historical worker quality/performance measurements are not native evidence.
Production disk-merge/export launchers were removed; see
`../status/production-tool-boundary-2026-09-09.md` for the source and package audit.

Final acceptance requires end-to-end image/video comparisons, cold/warm timing,
prompt changes, cancellation/recovery, and a copied packaged App exercised
without Python or repository-relative paths. Small-fixture parity and successful
compilation do not establish those guarantees.

Production-entry migration and lifecycle/package evidence are recorded in
`../status/wan-native-session-2026-09-09.md`.

### Wan hybrid diagnostic (2026-09-09)

`tools/native/wan_hybrid_probe.cpp` exercises the real 30-block checkpoint and
fixed 32760-row INT8 Core ML artifacts using zero and seeded random inputs.
It reports error against the native unsplit affine FFN, rejects nonfinite
outputs, and checks repeated predictions. It does not impose an arbitrary
quality threshold or claim that quantized and unsplit outputs are equivalent.

The initial run completed 60 comparisons / 120 predictions, with repeat
equality and zero output copied bytes. Zero-input RMSE ranged from 0.00110004
to 0.00255829; random-input RMSE ranged from 0.00428351 to 0.00628427.
These are synthetic FFN measurements, not a video quality acceptance gate.
All 30 inspected compiled MIL graphs include `fc2_bias` in the final `y`
convolution, so the GPU suffix deliberately adds no output bias.

`wan-generate-probe ... --hybrid MANIFEST` selects the integrated hybrid path;
the default remains native MLX. The legacy artifact schema still needs a
qualified Wan-named export/migration path before production naming is complete.

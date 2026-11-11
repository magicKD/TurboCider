# Z-Image INT8 GPU streaming on Apple M5 / 16 GiB

The exact Apple M5 / 16 GiB profile enables the existing Comfy INT8 ConvRot
streaming reader for explicit GPU requests. It uses one-layer prefetch and no
ANE suffix, multi-layer lookahead, or other M5 Pro optimizations. App sampling
budgets are 6 and 8 GiB, with 8 GiB selected when entering streaming from an
unqualified saved budget. The native request and reader reject larger budgets
on this profile. Other hardware keeps its previous policy.

## Real-model validation

On a 16 GiB MacBook Air with Apple M5, macOS 26.4 and MLX 0.32.0, the source
model was the installed Comfy INT8 ConvRot DiT, MLX affine Q4 Qwen3-4B text
encoder, original VAE and tokenizer. Requests used the same Chinese prompt,
seed 42, 512×512, nine steps, GPU and no LoRA. The native benchmark ran without
the App UI. Raw reports are in
`results/benchmarks/z-image-int8-m5-16-streaming-2026-10-28/`.

| Route and request | Wall | Denoise | Process physical footprint | System swap-out | Streamed blocks | Application weight read |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 6 GiB crossover: resident first | 18.391 s | 13.697 s | 11.33 GiB | 3.26 GiB | — | — |
| 6 GiB crossover: streamed first | 16.091 s | 14.340 s | 3.28 GiB | 0 | 23 / 30 | 39.59 GB |
| 6 GiB crossover: streamed repeat | 15.740 s | 14.829 s | 3.28 GiB | 0 | 23 / 30 | 37.56 GB |
| 6 GiB crossover: resident after switch | 18.135 s | 14.714 s | 11.33 GiB | 1.04 GiB | — | — |
| 8 GiB: streamed first | 16.992 s | — | 5.31 GiB | 0 | 13 / 30 | 25.08 GB |
| 8 GiB: streamed repeat 1 | 15.527 s | — | 5.19 GiB | 0 | 13 / 30 | 21.23 GB |
| 8 GiB: streamed repeat 2 | 15.900 s | — | 5.19 GiB | 0 | 13 / 30 | 21.23 GB |

All seven PNGs had the same SHA-256 digest. Every request completed nine
denoising steps with `runtime_precision=int8_tensorwise_convrot_g256`.
Repeated prompts hit the text cache. The 6 GiB route used two refill slots,
retained seven blocks and reported under 17 ms of load wait on first use and
under 1 ms on its repeat. The 8 GiB route also used two slots, retained 17
blocks and read fewer weight bytes. The 8 GiB warm median was 15.713 s; its
lower read volume and zero observed swap-out motivate the App default. The
6 GiB option has a smaller footprint for a busier machine.

A final-library 8 GiB smoke run with the same request completed in 18.310 s,
reported `int8_tensorwise_convrot_g256`, and produced the same PNG SHA-256.
Its system-wide swap-out delta was 0.94 GiB; this counter cannot attribute
paging to the benchmark process. The signed App bundle passed `codesign`
verification.

These are few local samples. System VM counters include other processes, and
the reported footprint is the benchmark process rather than the whole machine.
The 6 GiB crossover switches residency in one engine, so its resident samples
do not establish a long-session resident median. Earlier same-machine
resident sessions showed 20.8–47.9 s warm requests with heavy paging. The
streaming path trades more planned weight I/O for lower retained memory; its
benefit depends on disk and other applications' memory use.

## Admission and regression checks

The device profile grants only `z_image_int8_streaming`. Reader admission
still requires the actual Metal GPU name and physical memory. Hybrid suffix
and multi-layer prefetch remain exclusive to the qualified M5 Pro / 24 GiB
profile. App and native budget validation select 6 or 8 GiB on M5 / 16 GiB.
The device profile and request contract tests passed. The Metal GPU weight
stream suite passed all five applicable cases; five ANE suffix or unqualified
device cases were skipped on this M5. The final-library smoke run used the
installed real model.

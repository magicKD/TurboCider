# Verified Performance and Correctness

Measurements below were collected on the current Apple Silicon host through September 5, 2026. They are regression evidence, not universal hardware promises.

| Path | Direct engine | TurboCider | Correctness |
|---|---:|---:|---|
| H3 GPU, 256×256, 22 frames, 4 steps | 29.309 s process | 29.714 s process | Byte-identical MP4 and decoded video/audio |
| LTX GPU complete worker, 704×480×97 | 97.573 s | 96.952 s engine | MP4, decoded video/audio, and BF16 artifacts byte-identical |
| LTX GPU+ANE complete worker, same request | 105.311 s | 103.245 s engine | MP4, decoded video/audio, and BF16 artifacts byte-identical |
| FLUX.2 GPU persistent, 512×512, 4 steps | 2.216 s engine | 2.219 s engine | Direct/TurboCider decoded RGB byte-identical |
| FLUX.2 GPU+ANE a6144 split, compiled-cache warm | 1.570 s engine | 1.576 s engine | 80 ANE calls; decoded RGB byte-identical |
| FastMetal GPU, 832×480, 81 frames, 3 steps | 72.929 s engine | 72.937 s engine | Latent and MP4 byte-identical |
| FastMetal GPU+ANE, same request | 69.970 s engine | 69.970 s engine | 90 ANE calls; latent and MP4 byte-identical |

For the reproducible FLUX.2 benchmark prompt (`A cider press in an orchard`),
the current a6144 INT8-ANE split versus BF16 GPU measured RGB MAE 3.9712/255,
PSNR 30.8424 dB, and cosine similarity 0.998482. LTX media finalization
validates that the muxed MP4 retains all 97 frames.

FastMetal splits every INT8 FFN block at 4,096/4,864 intermediate channels.
The retained hybrid implementation measured 66.08–66.19 seconds for denoise
versus 69.56 seconds for compiled pure GPU, a 1.051–1.053× hot-path speedup.
Prompt-to-video improved from 72.93 to 69.97 seconds. GPU and hybrid are
different numerical paths: hybrid versus GPU measured latent cosine
0.998748, relative L2 0.050263, decoded-video SSIM 0.967460, and PSNR
37.851 dB. Within either backend, direct-engine and TurboCider outputs were
byte-identical.

An attempted fusion of each residual join into the next block's compiled
attention prefix preserved the hybrid latent SHA-256 but did not improve
runtime: denoise was 66.21 seconds and peak MLX memory rose from 4,287,770,538
to 4,687,261,610 bytes. That experiment was rejected; the production path
keeps separately compiled pre-FFN, GPU suffix, and residual-join graphs.

The final LTX end-to-end app-path recheck, including dynamic text conditioning,
connector preparation, full generation, video encode, audio decode, and mux,
completed in 112.72 seconds. Independent `ffprobe` validation reported 97 H.264
frames at 704×448/24 fps plus 48 kHz stereo AAC; the longer audio no longer
truncates the video stream.

The current stable FLUX.2 split cache contains 20 INT8 Core ML branches. Source
packages and compiled assets each occupy approximately 1.13 GB. A
`prepare-model --cache` invocation now recognizes the selected compiled
manifest as a cache hit and does not try to compile a compiled manifest again.

The reproducible JSON benchmark runner measured the current FLUX.2 paths again:

| Benchmark | Direct engine | TurboCider engine | Ratio | Decoded RGB |
|---|---:|---:|---:|---|
| 4B GPU persistent, 512×512, 4 steps | 2.216 s | 2.219 s | 1.0010× | identical |
| 4B GPU+ANE a6144 compiled-cache warm | 1.570 s | 1.576 s | 1.0033× | identical |

The a6144 route assigns SwiGLU channels `[0,6144)` to ANE and evaluates the
remaining 3,072 MLP channels together with attention on GPU. It reduces the
TurboCider engine time from 2.2187 to 1.5756 seconds, a 1.408× speedup and
28.98% latency reduction. The measured request made 80 Core ML calls with an
ANE-call p50 of 10.53 ms. Its metadata reports `CPUAndNeuralEngine`, the a6144
compiled manifest, and 100% steady-state input/output backing reuse.

The previous full-ANE MLP route was load-imbalanced on the M4 Max. A compiled
GPU attention+MLP block measured about 20.30 ms, while GPU attention alone was
7.51 ms and the full ANE MLP took 19.33--19.54 ms. Each block therefore waited
on ANE, limiting the complete request to roughly a 4% improvement over compiled
GPU. With the 6,144/3,072 split, the ANE prefix measured about 10.59 ms and GPU
attention plus the suffix about 11.88 ms, so the two sides overlap much more
evenly.

For the pure-GPU one-shot command, total process time was 1.046× direct. That
includes Python process, request validation, job persistence, and log handling;
the generation engine itself differed by about 0.12% in that sample.

The unified H3 rerun measured 29.31 seconds direct and 29.71 seconds through
TurboCider (1.0138× total-process ratio). Both outputs contained 22 H.264 frames
at 24 fps and stereo AAC at 32 kHz; decoded video pixels and decoded PCM audio
were exactly identical. The LTX parity artifacts were also re-hashed: direct
and TurboCider `video_latent.bf16`, `audio_latent.bf16`, and
`video_pixels.bf16` were byte-identical in all three comparisons.

## LTX two-layer benchmark

LTX is measured at two distinct boundaries so checkpoint setup, prompt
conditioning, media encoding, and the native decoded-pixel hot path are not
mixed into one misleading ratio.

The fixed-conditioning native comparison runs the original 8+3 schedule,
704×480 request (704×448 decoded), 97 frames, seed 42, and MLX media backend.
Each target gets one warmup followed by one measured process:

| Native path | Decoded-pixel engine | Fresh process | Versus GPU |
|---|---:|---:|---:|
| Dense GPU | 59.73 s | 74.82 s | 1.000× |
| Dense GPU+ANE | 49.96 s | 73.25 s | 1.196× engine speedup |
| Fast GPU+ANE, r256 K/V + Stage-2 Sol | 47.19 s | 75.22 s | 1.266× engine speedup |

The dense GPU+ANE result uses the complete 48-block Video MLP split and Stage-1
text-K/V placement while retaining dense attention and all 1,024 conditioning
rows. Relative to GPU, the final video latent had cosine 0.97007, audio latent
0.99971, and decoded BF16 pixels 0.99056. All tensors were finite. Fresh-process
speedup is smaller because Core ML/session and checkpoint setup are outside the
reported `two_stage_elapsed_seconds` hot path.

The reproduced fast candidate completed Stage 1 in 18.317 seconds, Stage 2 in
23.732 seconds, and VAE decode in 2.861 seconds, for 47.189 seconds from fixed
conditioning to decoded pixels. This is within 0.78% of the earlier 46.824-second
paired mean. It enabled all 48 ANE MLP branches, r256 ANE K/V in both stages,
full-GPU MLP release, and Stage-2 Sol. This is an experimental preview route:
the 256-row context pruning and Sol attention are approximate and require a
broader visual-quality suite before becoming the quality default.

The second layer invokes the same prompt-to-media worker directly and through a
TurboCider job. Each side again receives one warmup and one measured run:

| Complete workload | Direct worker | TurboCider worker | Total-process ratio | Correctness |
|---|---:|---:|---:|---|
| GPU | 97.57 s | 96.95 s | 0.9934× | MP4, decoded RGB/PCM, and all BF16 artifacts identical |
| Dense GPU+ANE | 105.31 s | 103.24 s | 0.9804× | MP4, decoded RGB/PCM, and all BF16 artifacts identical |

Both final files contained exactly 97 H.264 frames at 24 fps and 48 kHz stereo
AAC. The hybrid TurboCider run selected `ltx25.gpu_ane.dense.480p` through a
real match to `apple-m4-max-40gpu-64gb`; it was not a forced device override.
These worker totals include dynamic Gemma conditioning, the native connector,
fresh native-engine setup, generation, H.264 encode, Audio VAE finalization,
and AAC mux. They therefore must not be compared directly with the 49.96/59.73
second native hot-path row.

TurboCider previously failed to reproduce the 46--47 second candidate because
the adapter replaced the fast plan's r256 K/V directory with the default r1024
directory. The plan now maps `LTX_ANE_KV_DIR` to a distinct
`ane_kv_r256_directory`; dry-run validation shows `LTX_TEXT_ROWS_LIMIT=256`,
K/V enabled in both stages, and Stage-2 Sol enabled. The historical ~103-second
TurboCider number is the complete dense prompt-to-media quality workload, not
the same timing boundary as the 46--47 second fixed-conditioning preview path.

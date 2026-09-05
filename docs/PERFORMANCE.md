# Verified Performance and Correctness

Measurements below were collected on the current Apple Silicon host on September 4, 2026. They are regression evidence, not universal hardware promises.

| Path | Direct engine | TurboCider | Correctness |
|---|---:|---:|---|
| H3 GPU, 256×256, 22 frames, 4 steps | 29.309 s process | 29.714 s process | Byte-identical MP4 and decoded video/audio |
| LTX GPU complete worker, 704×480×97 | 97.573 s | 96.952 s engine | MP4, decoded video/audio, and BF16 artifacts byte-identical |
| LTX GPU+ANE complete worker, same request | 105.311 s | 103.245 s engine | MP4, decoded video/audio, and BF16 artifacts byte-identical |
| FLUX.2 GPU, 512×512, 4 steps | 2.383 s engine | 2.386 s engine | Direct/TurboCider decoded RGB byte-identical |
| FLUX.2 GPU+ANE, compiled-cache warm | 2.296 s engine | 2.303 s engine | 80 ANE calls; decoded RGB byte-identical |
| FastMetal GPU, 832×480, 81 frames, 3 steps | 72.929 s engine | 72.937 s engine | Latent and MP4 byte-identical |
| FastMetal GPU+ANE, same request | 69.970 s engine | 69.970 s engine | 90 ANE calls; latent and MP4 byte-identical |

For the reproducible FLUX.2 benchmark prompt (`A cider press in an orchard`),
INT8-ANE versus BF16 GPU measured RGB MAE 4.3128/255, PSNR 30.3296 dB, and
cosine similarity 0.9982865. An earlier dynamic-text development sample
measured MAE 1.8409/255, PSNR 34.7989 dB, and cosine 0.998702; that sample is
retained as diagnostic evidence but is not the formal benchmark baseline. LTX
media finalization validates that the muxed MP4 retains all 97 frames.

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

The stable FLUX.2 cache was generated from 20 existing INT8 packages in 1.85
seconds of orchestration (0.70 seconds summed Core ML compile time) and occupies
approximately 1.70 GB. A second `prepare-model --cache` invocation reported a
cache hit without recompilation. The first and second compiled-cache outputs,
and the prior source-manifest output for the same prompt/seed, had RGB MAE 0
and identical decoded pixels.

The reproducible JSON benchmark runner measured the current FLUX.2 paths again:

| Benchmark | Direct engine | TurboCider engine | Ratio | Decoded RGB |
|---|---:|---:|---:|---|
| 4B GPU, 512×512, 4 steps | 2.39 s | 2.39 s | 1.0012× | identical |
| 4B GPU+ANE compiled-cache warm | 2.30 s | 2.30 s | 1.0034× | identical |

After adding explicit 4B/9B model-variant handling, the 4B regression remained
stable. Direct and TurboCider decoded RGB were identical within both GPU and
GPU+ANE backends; the warm direct hybrid request completed in 2.2983 seconds
versus the previous 2.2956-second baseline, with 80 ANE calls and no session
reload. The cross-backend GPU-versus-hybrid values remain the formal MAE
4.3128, PSNR 30.3296 dB, and cosine 0.9982865 reported above.

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

The GPU+ANE result uses the complete 48-block Video MLP split and Stage-1
text-K/V placement while retaining dense attention and all 1,024 conditioning
rows. Relative to GPU, the final video latent had cosine 0.97007, audio latent
0.99971, and decoded BF16 pixels 0.99056. All tensors were finite. Fresh-process
speedup is smaller because Core ML/session and checkpoint setup are outside the
reported `two_stage_elapsed_seconds` hot path.

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

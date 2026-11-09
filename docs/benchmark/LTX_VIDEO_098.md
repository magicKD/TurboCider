# LTX-Video 0.9.8 2B: local GPU and Core ML experiments

These are bounded local measurements, not a general performance or quality guarantee. The model route remains experimental. ANE execution is not enabled in the production LTX 0.9.8 route.

## Multiscale workflow and quality follow-up

The original eight-step, single-resolution App path below is not the same as Lightricks' `ltxv-2b-0.9.8-distilled.yaml` multiscale pipeline. The new source also implements its seven-step first pass, 2× learned spatial latent upsampler, per-channel AdaIN statistics matching, and three-step second pass. CFG is 1, STG is 0, and decoder timestep/noise are 0.05/0.025. The first-frame conditioning scheduler follows Lightricks' capped per-token timestep; the retained eight-step path uses Comfy's inpainting-style scheduler. The App can keep the upscaled output or resize it back to the base canvas; it defaults to keeping the upscaled dimensions. This is a source/build validation, not proof that an existing installed App has been updated.

For a 512×512 base and 121 frames, the official 0.6666666 ratio and 32-pixel alignment give a 320×320 first pass and 640×640 second pass. On the same M4 Max / 64 GiB machine, fresh native diagnostic probes with seed 42 and first-frame strength 1 took 18.06 s for a fox and 17.23 s for a portrait with 640×640 output. The corresponding eight-step, 512×512 single-scale runs took 15.84 s and 14.91 s. These single runs include export and are not medians or matched-resolution performance claims. The measured MLX allocator peak for the multiscale probe was about 20.1 GiB, excluding other process/system allocations.

For a controlled upsampler component check, the **same saved first-pass latent** was passed to the native and ComfyUI BF16 upsamplers with the same checkpoint and VAE statistics. The initial native implementation had relative L2 error 0.0440 on the lighthouse latent; the opt-in 0.9.8 FP32-intermediate normalization/activation path reduced it to 0.00987. The pre-existing upsampler mode remained byte-identical on the tested latent. These are component checks, not proof of full-video parity.

An isolated ComfyUI MPS fox reference used the same prompt, first image, seed, 320→640 geometry, seven and three sigma intervals, strength 1 and the same model files. Stock Comfy nodes perform inference; diagnostic nodes provide the second RNG draw and Lightricks' AdaIN. The Comfy and native decoder noise streams remain different. Re-encoding all 121 Comfy PNG frames with the native video encoder gave mean correlation 0.9073, mean MAE 13.89/255 and maximum motion-energy relative difference 0.2458; the existing strict aligned-RGB gate **failed**. Initial noise matched exactly; the first-frame latent relative L2 was 0.0238 and first-pass output relative L2 was 0.6405, so the discrepancy is present before the upsampler and is not explained by the MP4 encoder alone. This diagnostic does not establish a single cause.

Visual inspection of sampled frames found that the fox's fur is softened, and its face and legs change across time in both the native and Comfy multiscale examples. In this one input/seed the native eight-step result retained the fox's face more consistently; it is not a general ranking of workflows. The portrait example blinked, but eye detail and glasses reflections changed over time. Retaining 640 versus resizing to 512 affects output pixels and apparent sharpness, not the underlying trajectory or anatomy. Different prompts, seeds, inputs and models require additional evaluation.

## Full GPU requests

On November 9, 2026, an Apple M4 Max with 64 GiB unified memory and macOS 26.6.2 was tested with MLX 0.32.0 and an isolated ComfyUI 0.39.0 / PyTorch 2.12.1 MPS reference. Each pair uses the same LTX 0.9.8 distilled checkpoint, T5-XXL weights, image, prompt, seed, 512×512 canvas, 121 frames, 24 fps, strength, eight Euler steps and full VAE decode. TurboCider uses BF16 DiT/VAE and BF16 T5 weights with FP32 T5 activations; the scheduler is FP32. The measured checkpoint SHA-256 is `76aa8c4786af752fa6f951947129d5290c3c6c0b2fadcadea6b5e114ae2cad8f`.

`tools/validation/ltx098_abba.py` runs fresh processes in native / Comfy / Comfy / native order. Native time includes CLI process startup and MP4 export; Comfy time starts after its isolated server is ready and includes workflow execution, export and media retrieval. The first native request also includes first-use startup costs. These are not warm-session medians.

| Input / strength | Seed | TurboCider GPU | ComfyUI MPS | Matched speedup |
|---|---:|---:|---:|---:|
| Lighthouse / 1.0 | 42 | 16.90 s | 36.78 s | 2.18× |
| Lighthouse / 1.0 | 43 | 15.06 s | 36.81 s | 2.44× |
| Tram / 1.0 | 42 | 15.03 s | 36.88 s | 2.45× |
| Tram / 1.0 | 43 | 15.03 s | 36.86 s | 2.45× |
| 448×320 tram crop / 0.75 | 42 | 15.10 s | 36.87 s | 2.44× |
| 448×320 tram crop / 0.75 | 43 | 15.03 s | 36.86 s | 2.45× |

After the exact GPU VAE-padding change below, a fresh A/B/B/A comparison of the newly packaged CLI and the same isolated ComfyUI setup measured lighthouse seed 42 at 14.99 s versus 36.93 s, and seed 43 at 14.98 s versus 36.84 s. These are approximately 2.46× matched speedups under this test's timing boundaries; the cross-encoder strict quality gate still failed its motion threshold for both runs.

The largest recorded MLX allocator peak in these earlier packaged-GPU tests was approximately 20.1 GiB. This excludes other process/system allocations and is not a hard memory limit. The later decoder-padding candidate's separate measurements are recorded below.

In an earlier same-process lighthouse batch, seed 42 took 15.02 s and seed 43 took 14.09 s. A separate batch with the new package took 16.85 s and 14.05 s, respectively; its second request also reported both text and cross-attention cache hits. The first observation includes startup costs. These are single warm observations, not warm medians.

The installed desktop runtime was also exercised through the App's actual `StudioState`, first-frame importer, acceleration resolution, request construction and embedded `NativeJobStore`, with an isolated test state directory. The App's default first-frame strength was 0.75; its complete 512×512×121, 24 fps, eight-step GPU request produced a validated MP4 in 15.23 s and persisted a succeeded job record. This test does not automate or render a desktop window.

## Quality scope and codec effect

The existing aligned-RGB gate requires mean correlation ≥0.99, minimum frame correlation ≥0.95, mean cosine ≥0.995, mean MAE ≤5/255 and maximum motion-energy relative difference ≤0.15. Direct comparisons between the native AVFoundation H.264 and ComfyUI's H.264 outputs failed that strict gate for all six runs. Importantly, re-encoding the *same* Comfy RGB frames with native H.264 also failed the motion threshold, so a cross-encoder failure alone does not isolate model quality.

For a more controlled comparison, reference PNG frames were encoded with the native encoder, and compared with the native video encoded by the same implementation. The model runs retain their normal, different decoder noise streams. The candidate probe and packaged CLI videos were checked to have identical decoded frames for these four cases.

| Input / seed | Mean correlation | Mean MAE / 255 | Max motion difference | Strict gate |
|---|---:|---:|---:|---|
| Lighthouse / 42 | 0.9976 | 2.50 | 0.123 | Pass |
| Lighthouse / 43 | 0.9963 | 2.75 | 0.037 | Pass |
| Tram / 42 | 0.9743 | 6.52 | 0.042 | Fail |
| Tram / 43 | 0.9215 | 13.90 | 0.195 | Fail |

Thus the speed requirement was observed in these six requests, but broad visual parity is **not** established. On the tram example, first-frame latent relative L2 error was about 0.008; small differences grew through the denoising trajectory. FP32 T5 reduced valid-token relative L2 against captured Comfy conditioning from 0.0286 to 0.0000068 for that prompt. For seed 42 the final latent relative L2 fell from about 0.407 in an earlier candidate to 0.192 in the current candidate. These figures do not establish perceptual equivalence or generalize to other prompts.

`ltx098_codec_diagnostics.py`, `ltx098_video_comparison.py`, `ltx098_euler_comparison.py` and `ltx098_trace_comparison.py` preserve the separate codec, raw-frame and trajectory diagnostics. Diagnostic workflows save extra data and their wall times are not performance baselines.

## Exact GPU VAE padding optimization

The 0.9.8 decoder now combines temporal edge replication and one-pixel spatial reflection into one BF16 Metal gather instead of three intermediate concatenations. This is a bit-preserving data movement operation, not a lower-precision or approximate decoder. Unsupported dtype, shape, channel count, negative-stride layout or indexing range uses the original MLX implementation. The guard covers the supported model geometry; it is not a performance guarantee for other devices or sizes.

In the isolated GPU padding screen, boundary, transposed and sliced inputs were byte-identical, including fixtures with signed zero, infinities and NaN bit patterns; negative strides and a non-multiple-of-eight channel count selected the original fallback. A saved 2×2 latent fixture also produced identical complete BF16 VAE outputs for one frame at 64×64, nine frames at 64×64 and nine frames at 64×96. Repeated synthetic latents also matched for 121-frame 128×1024 and 1024×128 outputs. One-frame decoding is a decoder diagnostic, not a supported public video request. On saved full-size lighthouse and tram latents with their original decode noise, the four baseline/candidate full-decoder output files were byte-identical. Across two full-decoder ABBA screens, the resident decode median fell from about 3.10 s to 3.03 s; the isolated MLX peak fell from approximately 6.50 GiB to 4.56 GiB.

For a direct-source, complete GPU image-to-video probe, lighthouse and tram were each measured A/B/B/A in four processes, with two sequential requests per process. All 121 decoded RGB frames matched exactly for each paired video. On the lighthouse warm requests the baseline/candidate medians were 14.102/14.036 s; on the tram they were 14.111/14.037 s. Measured cold-request differences varied with model loading and did not consistently favor the candidate. The largest process-reported MLX allocator peak fell from about 20.1 GiB to 18.2 GiB in these particular probes; these numbers exclude other allocations. The new packaged CLI independently reported approximately 18.2 GiB in the lighthouse batch, whose decoded frames matched the earlier package for both seeds. These are small samples on one machine, and neither isolated decode timing nor warm-probe timing is a fresh-process App/Comfy comparison.

`ltx098_vae_pad_screen.py`, `ltx098_vae_geometry_screen.py`, `ltx098_vae_decode_abba.py`, `ltx098_vae_full_abba.py` and `ltx098_package_exact.py` retain the separate component, geometry, full-decoder, full-video and packaged-frame checks. Their saved reports include input and binary hashes. The baseline build uses a private source overlay for the original padding and does not change the production decoder.

## Acceleration screens

- Cross-attention text K/V reuse was bit-identical in an eight-step full-video trace and in an A→A→B→A prompt-invalidation test. A 4096-token transformer-only ABBA screen estimated roughly 0.010 seconds saved over eight steps after cache preparation; this is a small component result, not a claimed end-to-end speedup.
- Increasing the MLX block evaluation interval from 1 to 4 gave bit-identical output but only about 0.4% improvement in an earlier transformer screen. It was not adopted as a performance claim.
- Packing the three self-attention Q/K/V projections gave bit-identical synthetic-input output but measured 7.20 ms versus 7.12 ms for separate GPU projections. It was not adopted.
- The existing weighted and unweighted BF16 RMSNorm matched PyTorch MPS bit-for-bit on the synthetic screen. A float-weight fused alternative introduced differences and was not adopted.
- An FP32-intermediate SiLU matched the synthetic MPS activation exactly and improved first-frame latent error, but its full-video effect was mixed: the seed-42 lighthouse final latent error improved from about 0.087 to 0.079 while the tram worsened from 0.192 to 0.237. It was not adopted.
- Fusing the BF16-to-RGB8 conversion matched all 65,280 finite BF16 bit patterns and real lighthouse/tram pixels. Component timing fell from 8.36 ms to 2.31 ms, but a separate full-video ABBA screen did not consistently improve matched requests: warm medians were 14.031 s baseline and 14.023 s candidate, with one candidate slower, and cold medians were 14.983 s and 15.022 s. The difference was within observed request variation, so this candidate was not adopted.
- An explicit tanh-GELU Core ML runtime-weight graph passed the existing strict self-test. With real block-0/14/27 matrices, synthetic activations, **no biases**, 4096 rows and a 1024-row CPU_AND_NE split, the component screen repeatedly measured about 1.23–1.24× versus full GPU including staging. Other splits were slower. CPU_AND_NE is a scheduling policy, not proof of physical ANE placement; there is no complete-bias, full-trajectory or end-to-end qualification, so this path is not enabled for the model.
- A separate experimental graph with both real biases and a captured real block-0 activation slice passed a signed-weight/bias self-test. CPU_AND_NE output differed from the BF16 GPU result by relative L2 ≈0.00665 on that slice. Its Python/Core ML 256-row prediction median was 12.54 ms versus 2.75 ms for the GPU calculation; these standalone timings include different interface overheads, do not overlap the branches, and do not prove ANE placement. They are not an end-to-end acceleration result.
- An independent native IOSurface screen then used the same bias-inclusive graph with captured first-step activations and real weights/biases from blocks 0, 14 and 27, in order 0→14→27→27→14→0. With 1024 of 4096 rows assigned to CPU_AND_NE, six component runs including weight staging were 1.231–1.234× faster than full GPU; maximum full-output relative L2 was 0.00350 (Core ML tail 0.00676). The GPU head was bit-identical. This still does not establish physical ANE placement, eight-step quality, or end-to-end performance.
- Core ML's `MLComputePlan` reported the Neural Engine as its preferred device for all 36 nonconstant operations in that biased graph, including its eight matmuls. A compute plan is an anticipated preference, not a runtime hardware trace; physical ANE occupancy remains unknown.
- A probe-only eight-step GPU/Core ML experiment exercised every FFN using real bias and a 768-row split. For lighthouse seeds 42/43, fresh-process ABBA measured GPU 15.13/15.44 s versus hybrid 15.04/15.04 s and both same-encoder video gates passed. The same experiment on the tram measured GPU 15.13/15.44 s versus hybrid 15.02/15.08 s, but **both** same-encoder video gates failed (mean MAE 8.73/10.63 per 255). It is not a qualified or enabled model route.
- Restricting the same 768-row split to the last two steps made both tram seed gates pass, but an earlier fresh-process median was slightly slower than GPU. A later two-request-per-process ABBA screen measured warm tram medians of 14.103 s GPU and 14.019 s hybrid, with all four paired video gates passing; the second request hit both prompt and cross-attention caches.
- Applying the split only to the final three steps passed the same-encoder gates for tram seeds 42/43 and, in a separate resident ABBA screen, lighthouse seeds 42/43. Lighthouse warm-request medians were 14.114 s GPU versus 13.974 s hybrid, but cold medians were 15.206 s versus 15.354 s and one matched cold hybrid request was slower. These limited, approximate gains do not justify enabling the hybrid path, and `CPU_AND_NE` still does not establish physical ANE execution. The hook is confined to a private build overlay used by the diagnostic probe.

Local machine-specific outputs and weights are intentionally not committed. The named validation scripts record binary, input and checkpoint identities alongside raw results when rerun.

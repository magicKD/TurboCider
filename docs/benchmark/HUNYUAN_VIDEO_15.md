# HunyuanVideo 1.5: in-progress ComfyUI reference

This is not an implemented TurboCider executor and does not establish a
TurboCider-versus-ComfyUI speed comparison.

## Model and workflow

The reference uses the full `480p_i2v_step_distilled_fp16` Transformer from
Comfy-Org's HunyuanVideo 1.5 repackaging, Qwen2.5-VL-7B, ByT5, SigCLIP
vision and the full FP16 VAE. All five downloaded files were SHA-256 checked.
The upstream guide distinguishes this checkpoint from CFG-only distillation;
the step-distilled I2V recommendation is CFG 1, flow shift 7, and 8 or 12 steps.
The tested stock ComfyUI graph uses Euler/simple, 8 steps and those settings.

Primary references:

- https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/ComfyUI/README.md
- https://huggingface.co/Comfy-Org/Hunyuan-Video-1.5

## M4 Max reference attempt, 2026-11-09

On an M4 Max with 64 GiB unified memory, ComfyUI 0.39.0 and PyTorch 2.12.1
used MPS with `--gpu-only`, BF16 model computation and no MPS CPU fallback.
The input was the same red-fox image and prompt as the Wan2.2 comparison,
seed 42, 512×512, 97 frames, 24 fps.

- The default PyTorch SDPA route aborted during the first step while requesting
  a 44,527,752,256-byte private Metal buffer. It produced no video.
- Stock ComfyUI sub-quadratic attention completed all eight denoising steps
  in about **339 s**. In the saved-latent retry, the elapsed time from prompt
  submission through successfully saving the final latent was **358.12 s**.
  The same process was later terminated with SIGKILL while decoding; the
  precise OS-level reason was not established.
- A fresh ComfyUI process loading only the full VAE decoded that exact saved
  latent and exported the MP4 in **285.24 s** request wall time. It used
  256-pixel spatial tiles, 64-pixel overlap and `temporal_size=4096`, as the
  upstream guide recommends avoiding temporal tiling artifacts. The resulting
  MP4 was verified as 512×512, 97 frames and 24 fps.

Adding the separately measured stages gives roughly **643 s**, excluding
server startup between processes; it is not a single-request measurement,
not a warm median, and not an accelerator trace. Both processes are needed
for this recovered result, so it must not be presented as an ordinary
same-process ComfyUI baseline or as a matched TurboCider comparison.

The five sampled frames show a recognizable fox and coherent snowy background;
the facial features remain fairly clear in this one example, while the ears,
fur and legs change over time. This is not a general quality ranking and no
spatial-tile seam assessment across a broader image set has been completed.

The recovered video, contact sheet, latent, logs and per-run reports remain
under the local `outputs/hunyuan15-comfy-512x97-*` experiment directories.

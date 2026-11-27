# RelayVSR 2× image and video super-resolution

RelayVSR 2× INT8 runs in the native macOS Studio, CLI, and SDK. It writes
a PNG with source alpha or an H.264 MP4 with the original frame timestamps.
For video, enable **保留原音轨** to copy compatible audio into the MP4.

## Use the local weights

Open **模型 → 超分 → RelayVSR 2× INT8** to download the two pinned weights
or register an existing folder. Both **创作 → 超分** and the sidebar **超分**
open the same workspace. Choose **图片** or **视频**, select the model in
the right sidebar, and add a source. In video mode, choose or drag in a
local video, or open a history preview and click **用作超分源**. The current
source appears beside **开始视频超分**. Drag the result's divider to compare
the original on the left with the restored media on the right. Video
comparison uses one set of playback and scrubbing controls for both sides.

You can register a folder symlink. TurboCider expects two files under the selected folder.

```text
diffusion_models/relayvsr_x2_lightx2v_int8_convrot.safetensors
vae/relayvsr_flash_vae_x2_fp32.safetensors
```

These are the Comfy-Org RelayVSR 2× ConvRot and Flash checkpoints at revision
`202c35eb0a5bb64b7017d5a09ea9acea87b7a081`. From adjacent TurboCider and
EZVideoRepair checkouts, you can create the local link from the TurboCider root.

```sh
mkdir -p models
ln -s ../../EZVideoRepair/ModelAssets/RelayVSR-X2 models/RelayVSR-X2
```

Replace the source and output paths in the matching example before running the CLI.

```sh
build/native/turbocider generate models/RelayVSR-X2 examples/requests/relayvsr-2x-image-upscale.json
build/native/turbocider generate models/RelayVSR-X2 examples/requests/relayvsr-2x-video-upscale.json
```

The model ID is `relayvsr-2x`. Omitting dimensions, frame count, and FPS
uses the source geometry and timing. Output axes are always doubled.
The source can be up to 2048 pixels per axis. For inputs larger than the
model canvas, inference aspect-fits within 640×512 (landscape) or 512×640
(portrait), then resizes the result to the exact 2× raster. Audio stream copy
requires an MP4-compatible source audio format.

## Build and runtime

Supply arm64 LibTorch 2.14.1 (headers and native dylibs) through
`TURBOCIDER_RELAY_TORCH_ROOT` or `.deps/relayvsr-torch`, then run `make build`.
An existing macOS PyTorch 2.14.1 wheel installation supplies the SDK in its
`site-packages/torch` directory. The App and CLI use the native
libraries directly. The package includes the needed dylibs and licenses.

On compatible Apple GPUs with macOS 26.4 or later, the runtime uses fused
W8A8 ConvRot and TensorOps Flash attention. Other Apple Silicon GPUs use the
MPS W8A16 and Metal Flash paths. It streams Wan blocks from the selected SSD
files with one GPU weight block and up to one CPU read-ahead slot. The video
path encodes frames as they finish and keeps at most two 16-frame input windows.
The result includes the selected precision, blocks fetched, and input-window
and weight-slot peaks.

## Local comparison

On November 27, 2026, an Apple M5 Pro with 24 GB unified memory and macOS
26.5.1 restored the same silent 512×512, 24 fps, 97-frame clip with each
implementation. Both used the same local checkpoint, SSD streaming, TensorOps,
and 1024×1024 H.264 output. The four runs used independent processes in
EZVideoRepair, TurboCider, TurboCider, EZVideoRepair order.

| Native export | EZVideoRepair `d2e03fb` | TurboCider |
|---|---:|---:|
| Median wall time | 30.54 s | 23.63 s |
| Median peak process footprint | 3.47 GiB | 3.78 GiB |

TurboCider reduced native export time by 22.6%. Decoded YUV pixels, frame
timestamps and frame durations matched across all four exports. Both
TurboCider runs prefetched 261 out of 270 Wan blocks and kept one GPU weight
block live. An additional 17-frame variable-rate clip verified exact frame
timing and source AAC passthrough. A 127×95 RGBA still verified 254×190 PNG
output and alpha preservation.

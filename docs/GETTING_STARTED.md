# Getting started / 快速开始

This guide covers the native macOS App, CLI and local job API. It does not
require downloading a model if you already have compatible local weights.
本指南介绍原生 App、CLI 和本地任务 API；已有兼容模型时可以直接使用。

## 1. Build / 构建

Use an Apple silicon Mac and a working Xcode toolchain. Install Python 3.11
for managed dependencies and offline conversion tools. From the repository:

```sh
make setup
make package
make test
dist/cli/turbocider doctor
```

`setup` installs the pinned development dependencies, not model weights.
`package` builds the engine, App, CLI and Swift test runners. Output:

- `dist/TurboCider.app`: locally signed native App.
- `dist/cli/`: CLI plus required libraries and helper resources. Keep this
  directory together when moving the executable.

The minimum macOS version follows the linked MLX library's deployment target.
Use `doctor` to inspect this Mac. If Xcode selection is broken but Command
Line Tools is installed, select it explicitly:

```sh
DEVELOPER_DIR=/Library/Developer/CommandLineTools \
SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk make package
```

已经配置好 MLX 时可传 `MLX_ROOT=/path/to/mlx`。不要把缺少 GPU 设备的
受限运行环境当成模型失败；原生生成需要可访问 Metal 的 macOS 会话。

## 2. Register a model / 选择模型

Open `dist/TurboCider.app` and select **模型 / Models** in the sidebar. Search
by name, model ID or registered path. Filter image, video or registered models.
Select a row to inspect it; browsing does not change your current draft.
Choose **选择文件夹…**, then **用于创作**.

新用户可以点击 **下载模型…**，默认来源为 ModelScope，也可切换 Hugging Face。
先预览文件与所需空间，再开始下载；可取消后重试，已校验的完整文件会复用。
LTX 的默认配方会在同一次安装中补齐另一个仓库的 tokenizer，并显示两处来源。
**管理目录** 可设置 App / CLI 共用目录，或导入包含 `modelPaths` 的 JSON。
Z-Image 下载页面可选择现有兼容文本组件，清单会跳过这些文件。
参见[模型库配置](MODEL_LIBRARY.md)。

模型库只展示当前执行器可用的操作。Z-Image 仅接受文字；FLUX 可接受图片；
LTX 当前为文字生成无音轨视频，即使上游具有图片输入能力也不会在 App 中开放。
“已登记”表示保存了路径。点击 **检查安装** 可先检查缺失组件、分片和文件完整性；
原生加载还会验证具体张量。CLI 对应 `turbocider library inspect MODEL_ID DIRECTORY`。

Local folders and symlinks are supported. For Z-Image:

- A complete Diffusers directory works directly.
- A Comfy directory may contain
  `models/diffusion_models/z_image_turbo_bf16.safetensors` and
  `models/vae/ae.safetensors`.
- If text components are absent, the App offers a second folder picker for
  a compatible FLUX.2 Klein 4B `text_encoder` and `tokenizer`. It creates a
  managed linked installation; it does not duplicate or modify those weights.

Keep linked source directories available. Moving or deleting a source breaks
its links. Never assume two encoders are interchangeable merely because their
folder names match.

## 3. Create / 创作

Select an available operation and enter a prompt. Set size, seed and steps;
video models additionally expose frames and frame rate. The defaults follow
the selected model. Add reference images only for operations that accept them.

Start with **GPU**. Add LoRA files as needed, toggle each adapter and set its
strength (default `1.0`). An unchecked LoRA preserves its configuration but
is not used. Press Command-Return to generate. Progress reports the current
phase and completed steps; decode/export follow sampling.

相同提示词换 seed 时，常驻会话可以复用文本编码。模型、文本编码参数或
输入身份变化时会重新编码。单次 CLI 命令通常会退出进程；批量生成需要
使用 `batch` 或常驻任务服务，才能复用同一模型会话。

The model-library session banner distinguishes an open session from loaded
weights. Staged execution can release completed components while keeping the
session available. **释放内存** closes the session without deleting models.
GPU telemetry is system-wide; ANE route selection is not a utilization meter.

任务详情和模型页会显示文本缓存命中、MLX 内存快照与 Core ML 会话累计调用。
设置页可清理内存、检查可重建张量缓存，并选择保留时间。
可通过 **登记输出张量目录…** 将指定的 FLUX / Z-Image dump 目录加入清理范围，
不会递归扫描研发 outputs，也不删除媒体或模型。
自动清理默认关闭；详见[缓存与内存](CACHES.md)。

## 4. CLI and video / 命令行与视频

Inspect a request before running it:

```sh
dist/cli/turbocider models
dist/cli/turbocider plan examples/requests/generate.json
dist/cli/turbocider generate /absolute/FLUX.2-klein-4B examples/requests/generate.json
```

For Z-Image use `examples/requests/z-image-turbo.json`; for LTX video use
`examples/requests/ltx-video.json`. Edit the output path, model-specific
parameters and optional configuration first. Existing examples may use an
absolute `/tmp` output path; use a unique path to preserve earlier results.

```sh
dist/cli/turbocider plan examples/requests/ltx-video.json
dist/cli/turbocider generate /absolute/LTX-2.5 examples/requests/ltx-video.json
```

H3 and FastMetal have separate layout and dependency requirements; see the
[model-specific reference](USAGE.md). A successful plan validates the request,
not the presence of all model artifacts.

For repeated seeds, create two request JSON files with the same prompt and
model, different seeds and different output paths:

```sh
dist/cli/turbocider batch /absolute/FLUX.2-klein-4B first.json second.json
```

stdout contains final JSON results; stderr contains progress events. Ctrl-C
requests cancellation at a safe boundary.

## 5. Prepare ANE / 准备 ANE

In the model library, choose the active model and expand **加速与编译缓存**.
The tools export supported FFN partitions from local weights, then compile
and cache Core ML models. Offline export uses the configured Python environment
and `coremltools`; inference uses the native runtime.

Artifacts must match the base checkpoint, input capacity, LoRA identity and
strength. Reuse a compiled manifest when available. First Core ML loading or
device preparation can still take time even when source compilation is cached.
For Z-Image at 512×512, the enumerated 1056–1536 row range supports up to512
encoded text tokens. Larger image resolutions need suitable larger partitions.

不要手动修改 manifest 的尺寸来假装扩大容量。实际 Core ML 图的输入约束也
必须匹配。错误的分区会明确报错，不会静默裁剪提示词。

The CLI uses the same resource operations:

```sh
dist/cli/turbocider coreml coreml-request.json
dist/cli/turbocider compile-coreml /absolute/block.mlpackage /absolute/cache
```

For a concrete512×512 Z-Image base-model example, edit the model directory in
[`export-z-image.json`](../examples/coreml/export-z-image.json), then run from
the repository root:

```sh
dist/cli/turbocider coreml examples/coreml/export-z-image.json
```

The referenced [flexible512 profile](../profiles/z-image-flex-512.example.json)
uses the M4 Pro48GB-validated8192-channel partition. It is an offline build
configuration; it does not enable ANE for every request or certify other chips.
For the validated LoRA route, copy the profile, set `ane_mlp_width` to6144 and
add the actual `loras` path/role/strength to the export request. In the App,
the same active LoRA settings are included automatically.

Copy the returned `source_manifest` path into
[`compile-z-image.json`](../examples/coreml/compile-z-image.json), choose a cache
directory, and run it with the same `coreml` command. Repeating this compilation
reuses matching partitions. Select the returned compiled `manifest` in the
App's **选择已编译分区 manifest…**, then enable ANE. App and CLI can use the same
absolute artifact paths; source conversion and compilation are separate from
loading/specialization on first use. Existing caches should be selected before
exporting another copy. The App exposes the same two actions in the active
model's preparation panel.

See [the resource reference](USAGE.md) and
[flexible-input validation](status/z-image-flexible-ane-2026-09-08.md) for
export configuration and capacity details. Hybrid acceleration remains
model- and device-dependent; compare warmed runs before choosing a policy.

## 6. Local task API / 本地任务 API

App 侧边栏的 **本地 API** 页面可启动和停止服务，并显示 Socket 路径、
执行中的任务、历史任务数量和日志入口。启动会先释放 App 的嵌入式会话；
服务运行时，通过 API 提交生成请求。停止后可回到创作页使用嵌入式引擎。
退出 App 或 App 异常退出会停止它启动的服务；服务会取消执行中的任务。

Start the persistent Unix socket service in a terminal:

```sh
mkdir -p "$HOME/Library/Application Support/TurboCiderService"
dist/cli/turbocider serve /tmp/turbocider.sock "$HOME/Library/Application Support/TurboCiderService"
```

In another terminal, create `rpc.json` containing `{"action":"models"}`:

```sh
dist/cli/turbocider rpc /tmp/turbocider.sock rpc.json
```

Each connection sends one newline-terminated JSON object. Supported actions
include `models`, `doctor`, `service_status`, `plan`, `submit`, `status`, `jobs` and `cancel`.
The socket is restricted to the current user. This is currently a Unix socket
API, not an OpenAI-compatible HTTP endpoint. See [RPC fields](USAGE.md#常驻后台服务).

For an end-to-end Python client and CLI stop behavior, see [Local API](LOCAL_API.md).

## Troubleshooting / 常见问题

| Symptom | Next step |
|---|---|
| Missing model / broken link | Re-register the source folder; verify shared components still exist |
| FFN geometry mismatch | Use an actually exported partition with sufficient image + text capacity |
| ANE fails after changing LoRA strength | Prepare artifacts bound to that exact adapter and strength |
| First generation is slow | Separate compile/load time from warmed generation; inspect result timings |
| GPU busy | Another TurboCider process may hold the GPU lock; finish or stop that job |
| Memory remains after generation | Check residency; use release-memory when the session is idle |
| Unsupported video input or audio | Inspect `executor_operations`; upstream capability is not runtime support |

## Development records

Development notes are retained under `docs/status/` and `docs/design/`.
Their historical observations are not promises for every device. Start with
[performance methodology](PERFORMANCE.md) for the current published numbers.

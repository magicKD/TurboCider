# TurboCider 独立运行与外部依赖边界

更新时间：2026-09-06

## 结论

TurboCider 已经是一个可以独立发布和启动的 native application/package，
但还不是一个“下载一个仓库、零外部资源即可运行全部六个模型”的完全封闭系统。

更准确的定义是：

- **核心源码独立**：H3 和 LTX 的运行时源码已经 vendored 到
  `native/models/h3_runtime/`、`native/models/ltx_runtime/`，构建 TurboCider
  不需要在编译时读取兄弟 `h3.c` 或 `ltx-mac` 工作树。
- **发行目录可脱离源码启动**：`dist/cli/` 携带 native dylib、MLX/JACCL、Metal
  shader、LTX helper 和必要脚本；复制到临时目录后，`models` 和 `doctor` 不依赖
  TurboCider 源码路径即可运行。
- **模型推理仍依赖用户提供的模型资产**：权重、tokenizer、VAE、LoRA 和特定
  Core ML artifact 不进入 Git 或默认 App 包。
- **全部模型能力尚非完全独立**：FastMetal、H3 媒体 I/O、H3/LTX LoRA 首次
  merge 仍有外部依赖。

## 依赖矩阵

| 范围 | 当前状态 | 外部依赖 |
|---|---|---|
| TurboCider core、C ABI、CLI、service、Swift App | 可独立运行 | macOS 系统框架、Apple Silicon、已构建 native dylib |
| FLUX.2 Klein 4B/9B native | 基本独立 | 用户模型目录；发行包内的 MLX dylib/metallib |
| Z-Image Turbo native | 源码独立、真实出图待验收 | 用户 ComfyUI Qwen3/DiT/VAE/tokenizer 目录；发行包内 MLX dylib/metallib |
| LTX video-only native | 基本独立 | 用户 LTX checkpoint/Gemma/upsampler/VAE；发行包内 MLX dylib 和 helper |
| H3 native denoise | 源码独立 | 用户 H3 模型目录、ANE/Core ML artifact（如启用） |
| H3 输入/MP4 输出 | 非完全独立 | `ffmpeg` 与 `ffprobe`；可通过 `H3_FFMPEG`、`H3_FFPROBE` 指定路径 |
| H3/LTX runtime LoRA cache | 非完全独立 | Python；`h3.c/tools/merge_h3_lora.py` 或 `merge_ltx_refiner.py`；缓存目录 |
| FastMetal 1.3B QAD | 明确是外部 worker 集成 | 用户 Python、MLX Python、PyTorch、FastVideo/TAEHV、entrypoint、profile、模型和 ANE artifacts |
| 编译 TurboCider | 非零依赖 | Xcode/Command Line Tools、macOS SDK、MLX C++ headers/libs (`MLX_ROOT`) |

## 为什么说源码已经独立

`tools/native/build.sh` 的正式源文件清单只读取 TurboCider 自己的
`native/`、`apps/`、`services/` 和 `bindings/`。H3/LTX runtime 的 README
记录了上游 provenance，但不会在运行时通过相对路径加载兄弟仓库。

发行包通过 `@loader_path` 和 `@executable_path` 解析 dylib；LTX shader 与
Video VAE helper 也从 dylib/发行目录定位。因而删除源码工作区后，已构建的
`dist/cli/` 仍能执行：

```sh
cp -R dist/cli /private/tmp/turbocider-portable
cd /private/tmp
env -i PATH=/usr/bin:/bin:/usr/sbin:/sbin \
  /private/tmp/turbocider-portable/turbocider models
env -i PATH=/usr/bin:/bin:/usr/sbin:/sbin \
  /private/tmp/turbocider-portable/turbocider doctor
```

本轮两条命令均已通过。受限环境中的 `doctor` 报告 GPU unavailable 是硬件权限
结果，不是源码缺失。

## 尚未完全独立的关键点

### 1. 模型权重不打包

这是有意的发行边界，而不是遗漏。用户必须单独提供模型目录；TurboCider 只负责
校验 manifest、身份、shape、LoRA 和资源路径。

### 2. H3 仍调用 FFmpeg

H3 runtime 的 `h3_ffmpeg.c` 通过 `posix_spawnp` 查找 `ffmpeg`/`ffprobe`，用于
图片、视频、音频输入以及 MP4 输出。当前 package 没有携带这两个二进制，因此
H3 完整媒体请求不能宣称“零外部运行时依赖”。

### 3. LoRA cache 仍是外部 Python merge

`lora_runtime_cache.py` 是缓存/身份封装，不包含实际的 H3/LTX safetensors、
ConvRot 和量化 merge 算法。cache miss 会调用 `h3.c/tools` 的 merge 脚本；
因此没有旁置的 merge 工具时，H3/LTX 独立 LoRA 请求会 fail closed。已经准备好
并带 provenance 的 merged checkpoint 不需要在推理时重新 merge。

### 4. FastMetal 是受控外部集成

FastMetal 的 native module 负责 profile 校验、worker 生命周期、取消、缓存和
协议，不重复实现 FastVideo/TAEHV。它必须由 profile 提供 Python executable、
engine root、entrypoint、worker 和可选 ANE bridge；没有这些资源不会伪造结果，
而是拒绝执行。

## 最终判断

如果“独立运行”指：

> 不依赖 `h3.c`/`ltx-mac` 源码工作树，复制发行目录后 CLI/App 能启动。

答案是：**已经达到**。

如果“独立运行”指：

> 不安装任何外部工具、不提供模型、不提供 Python/FastVideo、不提供 FFmpeg，
> 六个注册模型全部可以生成结果。

答案是：**尚未达到**。

推荐把当前系统称为“可独立发布的 native host + 分层可选运行时”，而不是
“完全封闭的 all-in-one inference appliance”。

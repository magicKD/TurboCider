<p align="center">
  <img src="assets/branding/logo.svg" width="760" alt="TurboCider — More from your Mac." />
</p>

<p align="center"><strong>原生推理，释放 Apple Silicon 的更多算力。</strong></p>
<p align="center">SwiftUI 工作室 · 原生命令行 · C / Swift SDK · 本地 API</p>
<p align="center"><strong>中文</strong> · <a href="README.md">English</a> · <a href="docs/GETTING_STARTED.md">快速开始</a> · <a href="docs/PERFORMANCE.md">性能测试</a></p>

TurboCider 是面向 Apple Silicon Mac 的本地多模态推理引擎。它把统一内存架构变成实际的优化空间：**CPU 调度原生计算，GPU 与 ANE 并行处理选定的模型分区**，减少设备边界的复制和等待，让计算密集型图像推理用上 GPU 之外的算力。

用原生 macOS App 创作图片和视频，用 CLI 批量运行，或把同一套运行时接入你的应用。模型和生成内容保留在本机。

## 为什么选择 TurboCider

- **硬件级异构并行。** Metal / MLX 执行 GPU 计算，Core ML 承接适合 ANE 的 FFN 分区；CPU 负责调度、准备和同步。已验证的分区路径使用共享输出缓冲，避免额外的输出复制。
- **可核查的提速。** 在 M4 Max 64 GB 的 FLUX.2 Klein 4B、512×512、4 步测试中，GPU + ANE 达到 **1.39× 吞吐，生成耗时降低约 28%**。相同测试输出的像素余弦相似度为 **0.999840**。速度和质量一起报告。
- **为重复创作优化。** 常驻模型、提示词编码缓存、GPU 图编译与可复用的 Core ML 编译缓存，减少更换 seed 后的重复工作。Z-Image 可变长度 ANE 分区在 512×512 下覆盖最多 512 个编码后的文本 tokens。
- **一套引擎，多种入口。** SwiftUI App、CLI、本地 Unix socket 任务 API，以及 C / Swift SDK 共用模型能力和请求契约。
- **按模型能力创作。** 文生图、图片修改、参考图编辑与视频生成按执行器开放。独立 LoRA、强度、随机种子和 GPU / ANE 选择都有明确控制。
- **模型与缓存集中管理。** App / CLI 共用模型目录，支持外部路径登记、ModelScope / Hugging Face 下载预览及兼容文本组件复用；App 可启停本地 API，并管理可重建的文本张量缓存。

ANE 路径使用经过验证的 INT8 分区，属于高保真近似，不是逐位无损。不同模型、芯片和输入的收益不同；默认使用 GPU。公开 Core ML 接口不能保证每个算子都实际驻留 ANE，也不提供可靠的本任务 ANE 占用百分比。我们公开边界和测量方法，不以单项测试宣称通用 SOTA。

## 实测表现

以下为命名工作负载的 warm request wall 中位数；不含首次模型加载和编译。吞吐倍数 = 基线耗时 / 优化耗时。

| 模型 / 设备 / 工作负载 | TurboCider GPU | GPU + ANE | 吞吐倍数 |
|---|---:|---:|---:|
| FLUX.2 Klein 4B · M4 Max 64 GB · 512² · 4 步 | 2.266 s | **1.628 s** | **1.39×** |
| Z-Image Turbo · M4 Max 64 GB · 1024² · 9 步 | 36.369 s | **30.001 s** | **1.21×** |
| Z-Image Turbo · M4 Pro 48 GB · 512² · 512 tokens · 9 步 | 23.964 s | **20.425 s** | **1.17×** |

[性能方法、质量指标和完整来源](docs/PERFORMANCE.md)。512-token 的可变长度分区不等于任意图片分辨率。

## 开始使用

需要 Apple Silicon、可用的 macOS SDK / Swift / Clang 和兼容的 MLX 0.32.x。推荐完整 Xcode；已有 Command Line Tools 可通过 `DEVELOPER_DIR`、`SDKROOT` 选择。Python 3.11 用于依赖安装、离线转换及特定模型工具，FLUX / Z-Image / H3 / LTX 的原生推理不依赖 Python 模型运行时。

```sh
make setup
make package
make test
open dist/TurboCider.app
```

在 App 的模型库选择模型文件夹，再进入创作。已有权重可以继续放在原来的目录；Z-Image 可关联 FLUX.2 Klein 4B 的兼容文本组件。首次先使用 GPU，成功生成后再按设备配置 ANE。发行目录中的 App 当前为本机 ad-hoc 签名，尚未完成 Developer ID 公证。

CLI 示例：

```sh
dist/cli/turbocider doctor
dist/cli/turbocider models
dist/cli/turbocider plan examples/requests/z-image-turbo.json
dist/cli/turbocider generate /absolute/path/to/z-image-turbo examples/requests/z-image-turbo.json
```

相对输出路径按当前工作目录解析；示例可能使用 `/tmp` 下的绝对路径。完整图像、视频、LoRA、ANE 准备和服务步骤见[快速开始](docs/GETTING_STARTED.md)与[使用参考](docs/USAGE.md)。

## 模型能力

`turbocider models` 返回当前执行器实际开放的操作；下表不代表上游模型的全部能力。

| 模型 | 输出 | 输入 | 说明 |
|---|---|---|---|
| FLUX.2 Klein 4B / 9B | 图像 | 文字、单图修改、参考图编辑 | 4B 已有混合加速验收配置；9B 性能覆盖尚不完整 |
| Z-Image Turbo | 图像 | 文字 | Comfy / Diffusers 目录；内存中应用 LoRA；可变长度 ANE 输入 |
| MiniMax H3 Turbo | 视频 | 文字、关键帧、参考素材 | 原生 Metal / MPS；ANE 需要匹配的模型产物；需要媒体工具 |
| LTX 2.5 Distilled | 视频 | 文字，仅无音轨视频 | 图片输入与音频尚未开放；分阶段驻留 |
| FastMetal 1.3B QAD | 视频 | 固定形状执行器，详见模型注册信息 | 显式配置的持久 Python worker；仅支持已验证的预合并 LoRA |

权重遵循上游许可证。仓库不包含大型模型文件和生成内容。

## 项目结构

| 目录 | 职责 |
|---|---|
| `native/` | 推理核心、模型执行器、Metal / MLX / Core ML 后端 |
| `apps/` | SwiftUI App 与原生命令行 |
| `services/` | 常驻本地任务服务、模型与缓存管理工具 |
| `bindings/` | C ABI 与 Swift SDK |
| `profiles/`、`examples/` | 硬件策略与请求示例 |
| `tools/`、`tests/` | 安装、打包、转换、回归测试与性能测试工具 |
| `assets/branding/` | 矢量标识与可复现的 App 图标 |
| `docs/` | 使用指南、设计与保留的开发记录 |
| `experimental/video/` | 冻结的迁移快照，不参与发行构建 |

[参与贡献](CONTRIBUTING.md) · [第三方声明](native/THIRD_PARTY_NOTICES.md) · [当前验收记录](docs/status/README.md)

[模型库](docs/MODEL_LIBRARY.md) · [本地 API](docs/LOCAL_API.md) · [缓存与内存](docs/CACHES.md)

TurboCider 原创代码采用 [MIT 许可证](LICENSE)。第三方代码保留[原有许可声明](native/THIRD_PARTY_NOTICES.md)，模型权重遵循各自的上游条款。

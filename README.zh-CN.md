<p align="center">
  <img src="assets/branding/logo.svg" width="760" alt="TurboCider — More from your Mac." />
</p>

<p align="center"><strong>原生推理，释放 Apple Silicon 的更多算力。</strong></p>
<p align="center">SwiftUI 工作室 · 原生命令行 · C / Swift SDK · 本地 API</p>
<p align="center"><strong>中文</strong> · <a href="README.md">English</a> · <a href="docs/public/GETTING_STARTED.md">快速开始</a> · <a href="docs/public/PERFORMANCE.md">性能测试</a></p>

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

## 性能亮点

**同一台 Mac，用上更多算力：M4 Max FLUX 的 GPU → 混合加速达到 1.39×，
Z-Image 相对记录中的 ComfyUI 为 1.34×；历史 M4 Pro FLUX 快照最高达到 1.64×。**

以下挑选已有记录中的 **warm request 中位数**，不是最快的单次运行：

| 工作负载 | 基线 → TurboCider | 加速比 |
|---|---|---:|
| FLUX 4B · M4 Pro 48 GiB · 512² · 4 步 | 原生 GPU 4.758 s → GPU + ANE **2.896 s** | **1.64×**¹ |
| FLUX 4B · M4 Max 64 GB · 512² · 4 步 | 原生 GPU 2.266 s → GPU + ANE **1.628 s** | **1.39×** |
| Z-Image Turbo · M4 Max 64 GB · 1024² · 9 步 | Stock ComfyUI GPU 40.110 s → GPU + ANE **30.001 s** | **1.34×**² |

M4 Max FLUX 的生成耗时降低 **28.2%**，对应 GPU／混合输出 PNG 余弦相似度
**0.999840**。Z-Image 原生 GPU 单独为 36.369 s，混合路线在此基础上再提速 **1.21×**。

¹ 来自 2026-09-05 两个测量阶段的历史比值，不是同一构建的配对复测；
旧混合产物只记录了路径／大小来源信息。
² 来自 2026-09-07 的工作流记录：原生 seed 42、每路线 2 次 warm；
ComfyUI seed 43–45、3 次 warm，计时接口不同，并非同 seed 的严格框架排名。

混合路线采用近似 INT8 分区；warm 耗时不含首次准备和编译。
**本次文档更新没有重跑实验**，精选历史结果不代表所有设备或最新构建的保证。
历史原生混合路线也并未快于原始引擎的混合路线。

[性能与保真度（英文）](docs/public/PERFORMANCE.md) ·
[公开样本、测量条件与比较边界（英文）](docs/public/BENCHMARKS.md)

## 开始使用

需要 Apple Silicon、可用的 macOS SDK / Swift / Clang 和兼容的 MLX 0.32.x。推荐完整 Xcode；已有 Command Line Tools 可通过 `DEVELOPER_DIR`、`SDKROOT` 选择。Python 3.11 用于依赖安装、离线转换及特定模型工具，FLUX / Z-Image / H3 / LTX 的原生推理不依赖 Python 模型运行时。

```sh
make setup
make package
make test
open dist/TurboCider.app
```

在 App 先选择图片或视频创作，自动匹配兼容执行器，再登记所需模型文件夹。已有权重可以继续放在原来的目录；Z-Image 可关联 FLUX.2 Klein 4B 的兼容文本组件。首次先使用 GPU，成功生成后再按设备配置 ANE。发行目录中的 App 当前为本机 ad-hoc 签名，尚未完成 Developer ID 公证。

CLI 示例：

```sh
dist/cli/turbocider doctor
dist/cli/turbocider models
dist/cli/turbocider plan examples/requests/z-image-turbo.json
dist/cli/turbocider generate /absolute/path/to/z-image-turbo examples/requests/z-image-turbo.json
```

相对输出路径按当前工作目录解析；示例可能使用 `/tmp` 下的绝对路径。完整图像、视频、LoRA、ANE 准备和服务步骤见[快速开始](docs/public/GETTING_STARTED.md)与[使用参考](docs/public/USAGE.md)。

## 模型能力

`turbocider models` 返回当前执行器实际开放的操作；下表不代表上游模型的全部能力。

| 模型 | 输出 | 输入 | 说明 |
|---|---|---|---|
| FLUX.2 Klein 4B / 9B | 图像 | 文字、单图修改、参考图编辑 | 4B 已有混合加速验收配置；9B 性能覆盖尚不完整 |
| Z-Image Turbo | 图像 | 文字 | Comfy / Diffusers 目录；内存中应用 LoRA；可变长度 ANE 输入 |
| MiniMax H3 Turbo | 视频 | 文字、关键帧、参考素材 | 原生 Metal / MPS；ANE 需要匹配的模型产物；需要媒体工具 |
| LTX 2.5 Distilled | 视频 | 文字，仅无音轨视频 | 图片输入与音频尚未开放；分阶段驻留 |
| Wan 2.1 1.3B QAD | 视频 | 原生 MLX / Core ML，832×480、3 步 | 离线转换的 TAEHV；只接受验证过的预融合 LoRA |

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
| `assets/branding/` | Turbo Drop 标识与可复现的 App 图标 |
| `docs/public/` | 独立英文使用指南与公开性能证据 |

[公开文档（英文）](docs/public/README.md) · [性能证据（英文）](docs/public/BENCHMARKS.md)

[模型库](docs/public/MODEL_LIBRARY.md) · [本地 API](docs/public/LOCAL_API.md) · [缓存与内存](docs/public/CACHES.md)

TurboCider 原创代码采用 MIT 许可证；第三方代码和模型权重遵循各自条款，请保留源码或发行包附带的许可证及声明。

# TurboCider

Apple Silicon 原生多模态推理系统。纯 C++/C/Objective-C++/Metal native runtime、SwiftUI App、CLI、C/Swift SDK 和本地任务服务共用一套实现；模型专用的 Apple bridge 保持在 platform/API 层，通用 runtime 不依赖 Foundation。Wan 已接入原生 Pipeline；H3/LTX 磁盘 LoRA 融合等运行时 Python 路径仍在清理，暂不能宣称所有用户功能都无 Python 依赖。

当前注册并提供八个模型模块：FLUX.2 Klein 4B/9B、MiniMax H3 Turbo、Wan 2.1 1.3B QAD、LTX 2.5 Distilled、Z-Image Turbo、Z-Image Turbo GGUF 和 LLaDA-Image-Turbo。FLUX/H3/LTX/Z-Image/LLaDA 的正式路径不依赖外部模型源码仓库；Wan 使用原生 UMT5、DiT、TAEHV 与 Core ML 分片；完整视频质量和性能验收仍在进行。LTX 当前公开 video-only 文生视频；Z-Image 已接入 BF16、ConvRot、GGUF 和独立 LoRA 路线，M4 Max 64 GB 的 base a4096 GPU+ANE 路线已通过重复 warm 端到端门槛；优化后的纯 GPU 路径也快于 stock ComfyUI GPU。LLaDA 当前正式公开自包含的原生文生图，图像编辑和 LoRA 尚未作为发行能力开放。

当前完成度、真实性能和未完成项以 [2026-09-08 当前状态](docs/status/current-status-2026-09-08.md) 为准；历史实现边界见 [实现状态](docs/status/implementation-status-2026-09-06.md)，提交前检查见 [版本准备度](docs/status/release-readiness-2026-09-06.md)。

## 构建与运行

需要 Apple Silicon、完整 Xcode、CPython（开发、离线转换及尚未原生化的 LoRA 工具）以及与已绑定 dylib 匹配的 MLX C++ 0.32.x。发行包的最低 macOS 版本取决于 MLX dylib 的 deployment target；可用 `TURBOCIDER_DEPLOYMENT_TARGET` 显式覆盖。

```sh
make setup
make package
make test
build/native/turbocider doctor
build/native/turbocider generate /path/to/FLUX.2-klein-4B examples/requests/generate.json
build/native/turbocider generate /path/to/Comfy-Org-z_image_turbo examples/requests/z-image-turbo.json
```

App：`dist/TurboCider.app`。独立 CLI：`dist/cli/turbocider`。本机 ad-hoc 签名，尚未公证发行。

## 代码结构

| 目录 | 职责 |
|---|---|
| `native/` | 推理核心、后端、模型模块、图像处理 |
| `apps/` | macOS App 与 CLI |
| `services/` | 原生本地持久任务服务 |
| `bindings/` | 公共 C ABI 与 Swift SDK |
| `profiles/`、`examples/` | 设备策略与请求示例 |
| `tests/`、`tools/` | 契约/集成测试、构建和性能验证 |
| `experimental/video/` | 已冻结的早期 H3/LTX 迁移快照，不进入构建 |
| `docs/` | 当前状态、使用、设计、验收及历史资料 |

旧 Python 控制层、旧 Swift App、重复引擎适配器及其构建入口已经退役；恢复位置记录在项目重构文档中。

- [使用、SDK 与服务](docs/USAGE.md)
- [当前实现、性能与未完成项](docs/status/implementation-status-2026-09-06.md)
- [独立运行与外部依赖边界](docs/status/independence-and-dependencies-2026-09-06.md)
- [版本提交准备度](docs/status/release-readiness-2026-09-06.md)
- [项目级重构与旧代码退役](docs/design/project-restructure.md)
- [当前实现与模块职责](docs/design/rewrite-implementation-status.md)
- [FLUX 性能对比](docs/design/flux-performance-comparison.md)：保留 FLUX 专项历史对照；跨模型的最新性能和完成度以 [实现状态](docs/status/implementation-status-2026-09-06.md) 为准。
- [GPU/ANE 并行化方案与 SVG 图](docs/design/parallel-acceleration.md)：按模型、芯片和 LoRA 门禁说明 fork/join 执行路径。
- [H3/LTX 接入与验收](docs/design/video-model-acceptance.md)

GGUF 发行路径仅使用 native MLX（Q8_0/Q4_0/Q4_1/F16/BF16/F32），暂不支持 mixed K-quants 或 GGUF streaming。App 不包含 stable-diffusion.cpp；对照工具隔离在 `tools/validation/sd_cpp/`。当前边界见 [native-only 整理说明](docs/status/native-gguf-boundary-2026-09-09.md)。

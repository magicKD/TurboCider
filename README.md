# TurboCider

Apple Silicon 原生多模态推理系统。C/C++/Objective-C++/Metal 推理库、SwiftUI App、CLI、C/Swift SDK 和本地任务服务共用一套实现。

当前注册并提供五个模型模块：FLUX.2 Klein 4B/9B、MiniMax H3 Turbo、FastMetal 1.3B QAD 和 LTX 2.5 Distilled。FLUX/H3/LTX 的正式路径不依赖 Python 模型运行时；FastMetal 有意保留显式配置的持久 Python/MLX worker，以复用上游 FastVideo/TAEHV。LTX 当前公开 video-only 文生视频，I2V、音频和默认 GPU+ANE 仍按能力门禁。

当前完成度、真实性能和未完成项以 [2026-09-06 实现状态](docs/status/implementation-status-2026-09-06.md) 为准；提交前检查见 [版本准备度](docs/status/release-readiness-2026-09-06.md)。

## 构建与运行

需要 Apple Silicon、完整 Xcode，以及兼容的 MLX C++ 0.32.x。历史 FLUX 性能证据使用 0.32.0；当前本机构建也验证了 0.32.2。发行包的实际最低 macOS 版本取决于所绑定 MLX dylib 的 deployment target。

```sh
export MLX_ROOT=/path/to/site-packages/mlx
make package
make test
build/native/turbocider doctor
build/native/turbocider generate /path/to/FLUX.2-klein-4B examples/requests/generate.json
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
- [H3/LTX 接入与验收](docs/design/video-model-acceptance.md)

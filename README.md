# TurboCider

Apple Silicon 原生多模态推理系统。C++/Metal 推理库、SwiftUI App、CLI、C/Swift SDK 和本地任务服务共用一套实现，原生推理不依赖 Python；离线 Core ML 转换使用本系统托管的 Python 工具链。

当前交付聚焦 **FLUX.2-klein-4B**：文生图、图生图、多参考图编辑、GPU 默认执行、可配置 GPU/Core ML 分区、缓存与预热、取消与历史记录。H3/LTX 为后续接入设计和验收契约，默认不可执行，不下载其模型。

## 构建与运行

需要 Apple Silicon、macOS 26.2+、完整 Xcode 和 CPython 3.11。固定依赖由 TurboCider 自己管理。

```sh
make setup
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
| `experimental/video/` | 未进入发行目标的 H3/LTX 迁移草稿 |
| `docs/` | 使用、设计、验收及历史资料 |

旧 Python 控制层、旧 Swift App、重复引擎适配器及其构建入口已经退役；恢复位置记录在项目重构文档中。

- [使用、SDK 与服务](docs/USAGE.md)
- [项目级重构与旧代码退役](docs/design/project-restructure.md)
- [当前实现与模块职责](docs/design/rewrite-implementation-status.md)
- [FLUX 性能对比](docs/design/flux-performance-comparison.md)：本机512×512/4步，GPU暖中位数快约21.7%，同分区混合路径慢约0.9%。
- [H3/LTX 接入与验收](docs/design/video-model-acceptance.md)

# TurboCider 原生推理核心

- `core/`：纯 C++ 请求、事件、错误与分词器接口。
- `runtime/`：纯 C++ 模型会话、执行计划、结果、驻留策略与执行互斥。
- `models/`：C++ 模型注册；`flux2/` 包含实际神经网络和流水线。
- `backends/`：MLX C++、Core ML 薄适配及分区缓存。
- `platform/apple/`：设备、配置、JSON、Unicode 的系统适配。
- `media/`：Apple 图像编解码。
- `api/`：稳定 C ABI 与 JSON/事件兼容层。

构建：仓库根目录运行 `make build`。完整边界、内存策略与验收方法见 [原生 C++ 引擎设计](../docs/design/native-cpp-engine.md)。H3/LTX 的历史草稿位于 `experimental/video/`，未进入正式构建。

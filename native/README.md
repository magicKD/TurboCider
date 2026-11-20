# TurboCider 原生推理核心

- `core/`：纯 C++ 请求、事件、错误与分词器接口。
- `runtime/`：纯 C++ 模型会话、执行计划、结果、驻留策略与执行互斥。
- `models/`：模型注册、各模型的执行器及推理流水线。
- `backends/`：MLX C++、Core ML 薄适配及分区缓存。
- `components/`：模型共用的文本、权重、扩散和 VAE 组件。
- `platform/apple/`：设备、平台配置及 Apple framework 适配。
- `media/`：图像、视频和音频处理。
- `api/`：稳定 C ABI 与 JSON/事件兼容层。
- `workflows/`：供 CLI 和 App 使用的工作流目录。
- `third_party/`、`licenses/`：保留的上游代码与许可。

从仓库根目录运行 `make build`。当前可执行模型、操作和输入输出限制见
[请求能力表](../docs/public/USAGE.md#model-capabilities)，维护历史见 [状态入口](../docs/status/README.md)。
与用户请求相关的限制以 [使用文档](../docs/public/USAGE.md) 和当前 CLI 的 `capabilities`、`plan` 为准。

Qwen/Z-Image 的冻结图与 runtime-weight GPU/ANE 分层、LoRA 正确计算顺序、
optional 和清理边界见 [后端维护说明](backends/README.md)；性能只维护在
[GPU/ANE 状态入口](../docs/status/acceleration.md)，不在代码目录复制跑分。

LTX 的请求操作、端点图片、音频和驻留策略见 [LTX 用法](../docs/public/USAGE.md#ltx-video)。
历史实测与适用边界归入 [LTX 实验索引](../docs/experiments/ltx/README.md)。

LTX 中 C 负责调度、Transformer block 和 connector，Objective-C/Metal 负责 Apple 适配与计算核，
C++/MLX 负责上采样、VAE 和音频。共享源的同步规则见 [LTX runtime 说明](models/ltx_runtime/README.md)。

`tc_ltx_audio_preflight_json` 检查原始音频组件的来源和能力。构建中的 `ltx-audio-vae-decode`、
`ltx-vocoder-decode` 和 `ltx-bwe-decode` 用于分阶段对照音频 VAE、16 kHz vocoder 和 48 kHz BWE，
`ltx-audio-mux` 用于检查 WAV/AAC 封装。

服务层还提供一个显式实验开关 `TURBOCIDER_LTX_RESIDENT_CANDIDATE=1`，允许
`residency=resident` 的 video-only 请求复用同一 C/Metal Transformer Session；VAE 仍在
独立 MLX helper 中解码。该路径只用于诊断，因为 64 GB unified memory 下保留 Transformer
会让 VAE decode 从约 3 秒退化到约 10--15 秒，当前默认仍是
`component_staged + exec finalizer`。conditioning cache 则由服务自动落盘，并按模型根目录、
checkpoint、Gemma/tokenizer 文件身份和 prompt 绑定，可跨服务重启复用。

完整边界、内存策略与验收方法见 [原生 C++ 引擎设计](../docs/design/native-cpp-engine.md)。
模型专用 Objective-C++ 适配器位于 `platform/apple` 或 `api`，`core`、`runtime` 和通用 `models` 保持 C++ 边界。

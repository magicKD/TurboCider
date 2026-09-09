# TurboCider 原生推理核心

- `core/`：纯 C++ 请求、事件、错误与分词器接口。
- `runtime/`：纯 C++ 模型会话、执行计划、结果、驻留策略与执行互斥。
- `models/`：C++ 模型注册；`flux2/` 包含实际神经网络和流水线。
- `backends/`：MLX C++、Core ML 薄适配及分区缓存。
- `platform/apple/`：设备、配置、JSON、Unicode 的系统适配。
- `media/`：Apple 图像编解码。
- `api/`：稳定 C ABI 与 JSON/事件兼容层。

使用方法见 [使用文档](../docs/USAGE.md)，当前完成度见 [实现状态](../docs/status/implementation-status-2026-09-06.md)，构建从仓库根目录执行 `tools/native/build.sh`。FLUX、H3、LTX video-only native session，以及 manifest-gated 原生 Wan session 已进入正式目标；LTX 音频和 I2V 仍按 operation 做能力门禁。完整验收边界见 [视频模型验收](../docs/design/video-model-acceptance.md)。

LTX 的 `video.generate`、`audio=false` 路径已经可以通过公共 native engine 创建；默认 residency 为 `component_staged`，使用动态 Gemma/connected conditioning、双阶段 Transformer，以及 ltx-mac 已验证的 C++/MLX clean-exec Video VAE finalizer。CLI 默认在用户 Caches 目录持久化 identity-bound conditioning。I2V 和音频仍然是 capability-gated candidate，不会因为模型描述为可执行而被静默放行。`tc_ltx_audio_preflight_json` 只做只读 provenance/能力检查，不执行外部 Python finalizer。

LTX 热路径不是 Objective-C 重写：C 负责 schedule/blocks/connector，Objective-C/Metal 只承担 Apple framework bridge 与 kernels，C++/MLX 负责 upsampler/VAE/audio。共享源的同步规则见 [`models/ltx_runtime/README.md`](models/ltx_runtime/README.md)。

Audio VAE latent→mel、16 kHz base vocoder 与 48 kHz BWE 已作为独立 MLX/Metal runtime 接入构建；`build/native/ltx-audio-vae-decode`、`build/native/ltx-vocoder-decode`、`build/native/ltx-bwe-decode` 提供阶段性 parity 入口，`build/native/ltx-audio-mux` 验证 WAV/AAC 封装。它们已连入 LTX Session，但音频请求仍需 provenance 和端到端 parity 门禁。

服务层还提供一个显式实验开关 `TURBOCIDER_LTX_RESIDENT_CANDIDATE=1`，允许
`residency=resident` 的 video-only 请求复用同一 C/Metal Transformer Session；VAE 仍在
独立 MLX helper 中解码。该路径只用于诊断，因为 64 GB unified memory 下保留 Transformer
会让 VAE decode 从约 3 秒退化到约 10--15 秒，当前默认仍是
`component_staged + exec finalizer`。conditioning cache 则由服务自动落盘，并按模型根目录、
checkpoint、Gemma/tokenizer 文件身份和 prompt 绑定，可跨服务重启复用。

构建：仓库根目录运行 `make build`。完整边界、内存策略与验收方法见 [原生 C++ 引擎设计](../docs/design/native-cpp-engine.md)。模型专用 Objective-C++ 适配器位于 `platform/apple` 或 `api`，`core`、`runtime` 和通用 `models` 保持 C++ 边界。

# TurboCider 当前实现状态

更新时间：2026-09-06（合并验收补充）
验证硬件：Apple M4 Max，64 GB unified memory

本文是本轮代码、真实运行记录和性能结论的统一入口。它区分“代码已经接入”“真实路径已经运行”和“已经通过正式质量/性能门禁”，不把 smoke、manifest wiring 或候选能力写成完整验收。

## 一句话结论

TurboCider 已经从单一 FLUX native 纵切扩展为六个注册模型模块：FLUX.2 Klein 4B/9B、FastMetal 1.3B QAD、MiniMax H3 Turbo、LTX 2.5 Distilled 和 Z-Image Turbo。`main` 的 C++ core/runtime 已通过 merge commit `2cee456` 进入 `dev`；各模型执行器、C ABI、Swift binding 与 App 使用同一模块注册和请求契约。H3、FastMetal、FLUX 4B 的核心路径已经接近 direct 实现；FLUX 4B 在 M4 Max 上新增正确的 `[0,6144)` ANE MLP 前缀 + GPU 后缀并行路径，单组 warm 观察为 1.409×，输出对 GPU 的 latent cosine 0.999660、PNG correlation 0.998391。FLUX 9B 和 LTX video-only 已在合并后的 C++ runtime 上完成真实生成。Z-Image 已完成 Apple M4 Max 真实 1024×1024、9-step 出图、Swift App embedded-session smoke，以及 base/官方独立 LoRA 的 ComfyUI 同噪声最终 latent/PNG parity；支持 ComfyUI split-files 和 Tongyi diffusers 目录入口，LoRA 只在内存中融合。Z-Image 的 32 分区 Core ML 导出和 `output_scale` ABI 已接入，GPU+ANE 只保留为显式 M4 Max 候选，自动模式不会启用尚未通过性能门禁的路径。LTX 包含模型建立、warmup、Core ML attach 和 VAE 生命周期的端到端请求还没有稳定快于 mac-ltx。独立 LoRA 请求已经进入 C/Swift/App 接口；H3/LTX 的审计过的 merge 实现现已随 TurboCider 源码和发行包提供，不再从兄弟仓库发现脚本，但首次使用仍会生成可清理的 merged runtime artifact，尚未完成纯内存融合。

## 代码结构与实现边界

```text
apps/                       SwiftUI App、CLI
bindings/                   C ABI、Swift SDK
services/                   Unix socket 持久任务服务
native/core/                纯 C++ Request/Recipe/common contracts
native/runtime/             纯 C++ plan、execution、residency、session
native/backends/            C++ MLX 与 Apple Core ML/artifact cache
native/models/flux2/*.cpp   FLUX 4B/9B 文本、DiT、VAE、LoRA
native/models/h3_runtime/  H3 C/Metal/ANE vendor runtime
native/models/ltx_runtime/  LTX C/Metal/ANE/MLX runtime
native/models/z_image/      Z-Image C++/MLX Qwen3、S3-DiT、VAE runtime
native/platform/apple/      ObjC++ Apple bridge 与 H3/LTX/FastMetal Session
native/api/                 C ABI
native/media/               图像、视频、音频和原子导出
profiles/                   显式设备/ANE/worker 配置
examples/requests/          可审阅的请求样例
tests/                      契约、服务、生命周期和 parity 测试
tools/native/               构建、基准、LoRA 准备和 oracle 工具
docs/status/                当前状态和版本提交准备度
docs/design/                长期设计、验收契约和历史架构
```

实现刻意保持混合语言：C/C++/MLX 承担模型数学和媒体子图，Objective-C++/Metal/Core ML 承担 Apple backend 与异构执行，TurboCider 自己主要提供统一 C ABI、Session、请求计划、缓存、取消、服务和媒体封装。LTX 共享热路径的同步规则见 [LTX runtime README](../../native/models/ltx_runtime/README.md)。

当前没有把完整 LTX 48-block denoiser 重写成一个 Objective-C monolithic graph。已有的 compile/fusion 是分层的：Metal pipeline cache、MPSGraph shape cache、fused INT8 MLP/QKV/attention，以及显式 Core ML artifact 的系统 compile/cache。完整 48-block、双流 AV、11 次 update 的静态单图尚未实现；两个额外边界 fusion 的实测收益只有约 0.08% 和 0.01%，因此不是当前首要瓶颈。

## 模型完成度

| 模型 | 当前可用能力 | 真实证据 | 尚未完成 |
|---|---|---|---|
| FLUX.2 Klein 4B | 文生图、图生图、多参考编辑、GPU/GPU+ANE、独立 LoRA；M4 Max 使用 6144 ANE 前缀 + GPU 后缀 | 6144 路径 warm 1.409×；latent cosine 0.999660、PNG correlation 0.998391；自动/显式输出一致 | 交错 AB/BA p50/p95、更广尺寸/机器矩阵、LoRA 专用 ANE artifact |
| FLUX.2 Klein 9B | GPU-only native 路径、文生图/图生图/编辑契约、App 选择 | 256×256、4-step 真实生成，request wall 4.257 s，MLX peak 18.90 GB | 正式 reference parity、标准尺寸 warm/ABBA、GPU+ANE |
| FastMetal 1.3B QAD | 持久 MLX/TAEHV worker、GPU/ANE split、取消重建、prompt cache | base GPU/GPU+ANE latent 与 direct 逐元素一致 | 多机器中位数、独立 LoRA runtime bake、LoRA ANE artifact |
| MiniMax H3 Turbo | 文生视频、首尾帧、reference、音视频、streamed/resident/component-staged、manifest LoRA | 512×512、22 帧、4-step 输出 byte-exact | 更广输入/尺寸、resident/ANE 多轮矩阵 |
| LTX 2.5 Distilled | video-only 文生视频、动态 Gemma、8+3、upsample、clean-exec Video VAE、conditioning cache、App 视频请求 | 704×448、97 帧既有真实生成；本轮另完成 704×448、9 帧 smoke | I2V 数值 parity、音频 Session parity、默认 GPU+ANE、端到端稳定快于 mac-ltx |
| Z-Image Turbo | ComfyUI split-files 与 Tongyi diffusers 目录、固定 shift=3.0/离散 sigma table 的 9-step flow schedule、FP32 Euler 状态、App/CLI/plan、独立 LoRA 内存融合、共享初始噪声/逐步 latent dump、32 分区 Core ML 导出 | Apple M4 Max 真实 1024×1024 base/LoRA 出图；官方 238-pair LoRA 全部内存融合；对 ComfyUI 最终 latent cosine 0.998929、PNG correlation 0.999156；App smoke 通过；output scaling 单 block finite | 逐 step oracle、warm 多轮性能、GPU+ANE 正式 parity/1.3×门禁 |

## 性能与准确性证据

以下数字只比较相同 workload、steps、shape、conditioning 和计时边界。

| 路径 | direct | TurboCider | 判断 |
|---|---:|---:|---|
| H3 512×512、22 帧、4-step process wall | 42.621 s | 42.963 s | 慢约 0.80%，MP4 byte-exact，已过 5% 回归门槛 |
| FastMetal GPU engine wall | 72.929 s | 72.937 s | 差约 0.01%，latent/视频一致 |
| FastMetal GPU+ANE engine wall | 69.970 s | 69.970 s | 计算热路径基本相同，输出一致 |
| FLUX 4B warm GPU engine wall | 2.216 s | 2.219 s | 慢约 0.1%，decoded visual 一致 |
| FLUX 4B warm GPU+ANE engine wall | 2.296 s | 2.303 s | 慢约 0.3%，decoded visual 一致 |
| FLUX 4B M4 Max 6144 前缀 warm | 2.3479 s GPU | 1.6659 s GPU+ANE | 单组观察快 1.409×；尚非正式 AB/BA 中位数 |
| Z-Image M4 Max 1024×1024、9-step warm | 45.17 s GPU | 47.52 s GPU+ANE | 混合路径慢约 5.2%，因此自动模式保持 GPU |
| 旧 LTX worker full chain | 97.573 s | 96.952 s | TurboCider 快约 0.64%，共享 artifact 一致 |

LTX 当前 public CLI 的最新重建二进制实测为：完整请求 86.284 s，conditioning cache 命中；pre-finalizer 81.918 s，clean-process Video VAE 3.410 s。此前相同 cache-hit 口径为 81.37 s。mac-ltx 的 59.483 s 是已准备 conditioning、已加载模型的 decoded-pixel 热路径，不包含同样的 cold/model setup，因此不能直接当作完整请求对比。扣除 TurboCider 的约 22.75 s model setup 后，热链路约 58.61 s，基本持平并略快，但波动不足以宣称稳定更快。

本轮 Z-Image Core ML backing 复核将 32 个串行 block 的 FP16 output backing 收敛为一个 session-wide buffer；真实 M4 Max 4096-channel、1024×1024、9-step 双请求完成，`output_copy_bytes_session_total=0`，两次 PNG SHA-256 相同。首次 request wall 约 44.69 s，第二次 cache-hit/warm request wall 约 37.51 s，第二次 denoise 约 36.59 s、VAE 约 0.85 s；此前同配置曾观察到约 4.5 s 的 VAE 退化。相对约 45.19 s 的 warm GPU 请求和约 44.28 s 的 GPU denoise，当前单组观察约为 1.20× 端到端、1.21× denoise，仍未达到目标 1.3×；因此 `auto` 保持 GPU，GPU+ANE 仍是显式实验路径。

LTX conditioning cache 已落盘并可跨 service 重启复用：cold 106.925 s，cache-hit 81.649 s，节省 23.64%；97 帧解码后的 framemd5 一致。resident candidate 曾出现 115–121 s，原因是保留 denoiser residency 使 VAE 从约 2.8–3.0 s 退化到约 14–15 s，因此默认仍使用 `component_staged + exec finalizer`。

LTX GPU+ANE 当前只作为候选实验路径：同一动态 Gemma 请求约 125.90 s，dense 约 119.08 s；最终 video latent cosine 约 0.945，RGB cosine 约 0.978。它尚未通过 exact parity 或端到端性能门禁，不能默认启用。

Z-Image 7680-channel ANE 前缀的重复输出可复现，GPU 与 GPU+ANE 的 PNG correlation 为 0.998571、cosine 为 0.999819，但 32 blocks × 9 steps 会产生 288 次同步 Core ML 调用，GPU 后缀尚未真正隐藏这些边界开销。该路径保留给显式实验和继续优化，App/API 的 `auto` 不会选择它。

## LoRA 状态

### 已完成

- FLUX 4B/9B 接受独立 `.safetensors`，支持常见 A/B、up/down 命名和多个 adapter；load-time bake；LoRA 会正确使 base conditioning/transformer/ANE identity 失效。
- H3/LTX 请求接受独立 adapter；首次使用调用 TurboCider 自带的审计 merge 工具，生成内容寻址、可清理的 runtime cache；cache hit 不重复 merge。发行 CLI/App 也携带 `merge_h3_lora.py` 和 `merge_ltx_refiner.py`，不再依赖兄弟源码仓库。
- `turbocider prepare-lora` 已提供只读检查和 provenance manifest 入口。

### 尚未完成

- H3/LTX native loader 仍消费 merged artifact；首次 cache miss 会临时占用接近 Transformer 的派生空间。当前已经消除兄弟仓库路径依赖，但还没有把 LoRA A/B 逐层挂到 native GPU/ANE linear 上。
- 最终目标“磁盘只保留 base + 独立 LoRA，逐层加载时在内存 merge/requantize”尚未实现。
- FastMetal 目前仍要求 provenance-verified premerged MLX checkpoint，独立 LoRA runtime bake 和真实 LoRA parity/性能尚未完成。
- Z-Image adapter 保持为独立 `.safetensors`；运行时只在内存中的 packed QKV/FFN/out projection 上应用 delta，不生成第二份 checkpoint。官方 151 MB distill patch LoRA 的 238 个 A/B 对全部被应用；同噪声对 ComfyUI 的最终 latent cosine 为 0.998929，PNG correlation 为 0.999156。完整记录见 `docs/design/validation/z-image-comfy-lora-parity-2026-09-06.json`。

## 测试、构建与运行状态

已验证：

```text
make test
Python/bin/python -m pytest -q <无参数单元与 fixture 测试集合>
`make test` 当前为 42 项 contract 加 9 项 repository/boundary 检查通过（含 Z-Image plan/LoRA/fail-closed、FLUX 6144 补算和 M4 Max profile 契约）；`make test-app` 通过，剪贴板服务在受限会话中按条件跳过。
```

`tests/native` 同时包含需要 `--model/--manifest/--output` 的真实验收程序，因此不能把整个 `tests/` 当作无参数 pytest collection；这些入口应按文档单独运行。

构建脚本会显式编译 TurboCider core/runtime、H3 runtime、LTX runtime、LTX media helpers、CLI 和 Swift runners；不自动下载模型或依赖。deployment target 会从实际绑定的 MLX dylib 推导并同时应用到 native/Swift 产物。本轮最终完整构建无编译或链接 warning，App 与 CLI package ad-hoc 签名通过。真实 Apple Silicon 环境可以启动 CLI/service，并已完成 FLUX 9B、LTX video-only 与 Z-Image Swift App embedded-session 的真实 smoke；本轮 Z-Image App request wall 为 45.632 s。受限沙箱没有 Metal/pasteboard service 时按测试条件跳过相应系统能力，不记为产品启动失败。

## 尚未达到的目标

1. LTX 包含 warmup/模型建立/compile 的端到端 wall 稳定超过 mac-ltx。
2. LTX GPU+ANE 达到可接受的 video latent/RGB parity，并且端到端快于 dense GPU。
3. LTX I2V、音频 VAE/vocoder/BWE/AAC 的完整 Session parity 和公开 executor。
4. H3/LTX 纯内存 LoRA merge，避免第一次使用生成完整 merged Transformer 副本；现阶段只完成了本地 vendored merge 工具和可清理 cache。
5. FastMetal 独立 LoRA runtime bake、LoRA 质量/性能和匹配 ANE artifacts。
6. FLUX 9B 标准尺寸的 parity、warm/resident 和 AB/BA 性能矩阵。
7. 多机器、多 macOS/Apple Silicon 版本和 Developer ID 公证包认证。
8. Z-Image 同噪声最终 latent/PNG 与独立 LoRA 图片 parity 已完成；仍需逐 step oracle、1024×1024 warm 性能矩阵与 GPU+ANE 1.3× 目标。

## 当前版本定位

当前代码适合提交为“多模型 native integration milestone / 开发验收版本”，不适合标记为“所有模型生产就绪”或“LTX 已稳定超越 mac-ltx”。提交时应保留 fail-closed 门禁和候选标签，不要为了让模型列表看起来完整而放开 I2V、音频或未经验证的 GPU+ANE。

更细的原始 wall、stage hash 和媒体对比保留在本机 `outputs/` 与 `notes/`；这些目录不作为版本入口。需要进入版本的证据应先去除机器路径，再整理为 `docs/design/validation/` 下的小型摘要。

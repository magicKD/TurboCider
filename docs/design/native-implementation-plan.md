# TurboCider 本机原生重构实施设计

日期：2026-09-05。本文细化前一份 rearchitecture 提案。实际完成状态以同目录的验证记录为准，不把设计、编译通过和真实模型验收混为一谈。

## 本机约束

- 只使用 `/Users/kd/Documents/project/mac_image_generation/models/FLUX.2-klein-4B` 的既有模型文件；本轮不下载任何模型，包括 H3/LTX。
- 现有 macOS 26.6；默认 Command Line Tools 与完整 Xcode 都是较旧的 16.x 工具链。Xcode launcher 存在 CoreDevice/Mercury 符号加载问题。构建脚本直接指定已有 compiler 和 SDK，避免更改系统 xcode-select 或修补系统。
- 可复用的原生 MLX 0.32.0 在 mflux 虚拟环境中的 `mlx/include` 与 `mlx/lib`。这是本地依赖来源，发布时需复制 dylib/metallib 并重写相对加载路径；推理不启动 Python。
- 不使用私有 ANE API。已有 Core ML artifact 已接入 native hybrid 并真实运行；显式标为 INT8 近似实验路线，不能凭 cpuAndNeuralEngine 就宣称实际驻留 ANE，也不进入 auto。

## 参考版本

| 仓库 | 当前 HEAD |
|---|---|
| TurboCider | 0035171b0906d608f570072f62bcea8481fdac3d |
| h3.c-fork | fceeb88122c44e42785956b320f0753c339978ef |
| ltx-mac | f1a47138ea18355df8f3df5d8d2d19cd012dd2ad |
| flux2-engine | 98aa6f1710a44e220a4b2829a250fa9fdbc3daec |
| mflux（数学参考） | 12fd27ea7015c6c872ced51b56b313306a543dd2 |

这些 HEAD 是本轮开始时读取的版本。验证工具需同时记录工作区脏状态与关键文件身份，不能把 HEAD 当作所有工作区文件的精确快照。

## 第一阶段代码结构

```text
native/
  include/turbocider/turbocider.h   C ABI 1，opaque Engine，JSON schema 1
  include/module.modulemap        Swift 导入边界
  core/
    runtime.hpp                   内部 Tensor/Weights/Request/Recipe/Event
    json.mm                       类型与未知字段检查
    tokenizer.mm                  原生 NFC + Unicode regex + byte-level BPE
    api.mm                        错误/生命周期/取消/事件/设备诊断
  models/
    recipes.mm                    三模型阶段依赖与形状/步数约束
    flux.mm                       常驻会话、请求状态、缓存和生成流程
    flux_text.mm                  Qwen3 9/18/27 层 conditioning
    flux_transformer.mm           5 dual + 20 single blocks，4轴 RoPE
    flux_vae.mm                   batchnorm 去归一化、unpatchify、VAE decode
  backends/mlx.mm                 原生 Metal 运算后端及 custom Metal Euler
  backends/coreml.{hpp,mm}        公开 Core ML 分区、固定桶和同步/复制边界
  media/image.mm                  ImageIO 原生 PNG，临时文件原子提交
  cli/main.mm                    原生 doctor/models/plan/generate
  swift/                         Swift SDK/任务管理/SwiftUI 客户端
  tests/                         无权重检查、接口检查及真实 FLUX 对照
 tools/native/                   构建、打包、离线参考导出和验证工具
```

采用 `native/` 独立纵切，避免在正在验证原生数值期间破坏旧 Python 服务。它是新架构的实现入口；旧 `src/turbocider` 和旧 Swift App 在原生验收前仅作回归与迁移参考。将新 App 改成默认产品，需要它通过相同任务/错误/输出验证后再切换。

## 共同抽象如何提取

### 与模型无关

1. **Request / PreparedPlan**：声明意图与实际可执行计划分离。未知字段、错误 dtype/shape、不可用设备路线在生成前报错。
2. **Recipe / Stage**：模型阶段和真实数据依赖。检测缺失依赖、重复节点、非法依赖顺序。采样循环以模板表示，避免展开巨图。目前 Recipe 用于 preflight 和能力报告；FLUX 按显式模型流程执行，尚非解释任意 Recipe 的通用执行器。
3. **Weights**：safetensors 分片枚举、重复名称检查、惰性加载与组件释放。模型模块保留权重命名和结构验证。
4. **Tensor / backend operations**：第一阶段 Tensor 是内部 MLX C++ array 类型；不能据此宣称已拥有独立 Metal allocator 或任意后端零拷贝。算子入口集中到后端，为替换 kernel 留边界。
5. **Execution / Event / Cancellation**：阻止 session 并发写同一 workspace；MLX 存在全局配置，第一阶段所有嵌入调用采用进程级 GPU 准入。MLX 0.32 默认流绑定线程；Swift 串行队列并不保证固定线程，因此使用 new_thread_unsafe_stream 创建跨线程流，并以该锁保证串行访问。回调线程、有效期和取消安全点写入 C ABI。
6. **Media / Result**：原生图像存储、实际尺寸与性能元数据，结果与文件导出分离演进。

### 保留在模型模块

- FLUX 的 token 打包、4轴 RoPE、双/单流、经验 sigma shift、VAE BN。
- H3 的 FL2VA/Ref2VA、first/last 与有序 references 互斥、Turbo LoRA provenance 和音视频 flow shift。
- LTX 的 connector、音视频双流、8+3 schedule、中间 latent upsample 与音视频同步。

不共享这些模型专属规则。`generate()` 的统一形式不意味着数学图可以互换。

### 后续从 MLX 子图演进为独立 Runtime

第一阶段 native MLX backend 可以完整执行真实模型，并去掉 Python、mflux patch 和子进程数学实现。下一阶段抽出 StorageDescriptor（设备/布局/stride/owner）、CompletionToken、BufferLease、WorkspacePlan，逐子图替换 MLX。MLX 的内部缓存计入不透明预算；不能让自有 Metal 与 MLX 同时认为自己拥有同一 backing 的写入权。

优先迁移 fused norm/quantize、GEMM epilogue、RoPE/projection、attention、VAE conv；每次替换保留参考路径和真实张量测试。首版自定义 Euler Metal kernel 经 MLX 的 custom kernel 接口接入，所以仍由 MLX 管理其依赖与分配，避免错误的手工跨队列同步。

## FLUX 本机完整路径

1. 读取本地配置与 tokenizer；拒绝结构不匹配的 9B 或其他变体。
2. 原生构造 Qwen3 单条 user chat template，关闭 thinking；NFC、Unicode pretokenizer、byte BPE，显式限制 512 tokens，不静默截断。
3. BF16 Qwen3，提取第 9/18/27 个 hidden states。后续 28...36 层对这些输出无贡献，可精确省略；并非省略被消费的模型层。
4. 释放文本权重，保留 conditioning。相同 prompt+padding 策略可复用缓存；不同 prompt 先释放驻留 DiT，再编码文本。
5. 使用现有原始 BF16 transformer 权重；实现 5 个 dual block + 20 个 single block，不减层、不稀疏化。
6. 保持与参考一致的 RNG、noise layout、sigma shift、BF16 step 舍入。自定义 kernel 必须与参考 Euler 比较。
7. 去 BN normalization、unpatchify、post_quant_conv、VAE decode，原生写 PNG。
8. Swift App 通过 C ABI 调用同一核心，显示阶段/步数、支持取消、保存任务记录和预览最终图像。无模型下载功能默认触发。

第一阶段正式支持 `text_to_image`。img2img/多参考编辑需要增加 encoder、输入角色与 token 规则后逐项验收，不能从旧引擎能力自动继承。动态文本长度和固定 512 padding 是不同 recipe 输入设置，测试与结果必须记录。

## H3/LTX 的代码迁移与缺模型验收边界

| 工作 | 无权重可执行检查 | 有权重后必须完成 |
|---|---|---|
| H3 模型模块 | recipe DAG、Turbo 4-step 配置、shape/输入规则、公共 C 头编译、源引擎无权重测试 | 9/18 等规则不适用于 H3；逐 encoder/DiT/refiner/VAE 与原实现对照，完整视频+音频、reference 顺序、常驻/流式权重 |
| LTX 模型模块 | connector→8步→upsample→3步→双解码→mux 的依赖；frames=8n+1；请求/实际尺寸区分；源引擎编译/无权重检查 | 真实 Gemma/connector、AV block、双阶段 timestep 和 first-frame mask、upsampler、video/audio VAE/vocoder、最终音视频 |
| 共同 runtime | 错误码、取消、图依赖、缓存失效、配置校验、buffer 生命周期基础检查 | 持续生成内存压力、真实 checkpoint peak、不同 shape 交替和异常回收 |

当前 recipe/contract 迁移不是完整 H3/LTX 数学 executor 的迁移。源码编译通过仅证明接口和构建兼容；如果尚未将原引擎调用重构至共同后端，文档必须标成待实现。无权重时 executor 返回明确 unavailable，不返回伪造媒体或“测试通过”。

真实权重到位后必须固定 checkpoint hash、H3 adapter/flow shift、LTX 8+3 recipe。逐算子 → block → 单步 latent → 多步 latent → decode → 完整媒体依次验收，数值误差阈值按各张量动态范围预先制定，禁止只看像素平均误差。视频至少覆盖人物/运动/长 prompt/首帧/多模态输入，检查闪烁、音轨内容、幅值和同步。

## 原生依赖与发行

| 依赖 | 第一阶段用途 | 长期处理 |
|---|---|---|
| C++20/Objective-C++ / Apple SDK | 核心、C ABI、Foundation JSON/tokenizer、媒体 | 必需、显式 SDK 路径 |
| MLX C++ 0.32.0 + libjaccl + mlx.metallib | 本机 Metal 计算 | 版本固定，发布目录复制；可渐进替换热点 |
| Metal | MLX GPU + 自定义 Euler | 必需；新 TensorOps 需更高 SDK 后再加能力路径 |
| Swift/SwiftUI/AppKit | SDK、任务生命周期、原生 App | 必需；不依赖 Python daemon |
| Core ML | 已实现 FLUX single-block MLP 的公开固定桶子图 | INT8 实验路线；实际 ANE residency 未确认，不进入 auto |
| Python/MLX Python/transformers | 仅离线参考导出、数值比较 | 开发验证工具，不在发行推理链路 |
| ImageIO/CoreGraphics | PNG | 发布必需 |

模型不进入 App bundle。模型路径用户选择或通过显式环境配置设置；不能把开发机绝对路径写成不可修改的产品常量。SDK 错误返回包含具体原因，所有 returned char* 由统一 free 释放。

## 测试分层与接受标准

- **无权重**：三模型 DAG 与错误配置；BF16 custom kernel 对照；ABI 生命周期与错误输入；请求未知字段必须失败。
- **Tokenizer**：英语、中文、数字、emoji、Unicode combining marks、换行、special token；与本地 Qwen2TokenizerFast 完整 ids 一致。
- **真实张量**：固定相同 tokenizer IDs、initial latent、step sigmas；对照 conditioning、每步 noise/latent、decoded NHWC pixels；报告 max/mean/RMSE、nonfinite 和 shape，保存结果而不更新 goldens。
- **完整生成**：至少小图与 512 标准图，固定与动态文本，重复 prompt 缓存与不同 prompt 切换；生成图可以解码、尺寸正确、非空白，人工查看内容与参考差异。
- **App**：真实模型路径选择→生成→阶段事件→结果图；取消后 UI 可恢复；错误模型路径不会启动任务；重启后结果记录可读取。
- **隔离**：原生可执行文件的依赖清单没有 libpython；脱离 mflux 源码 cwd 仍能生成；不触发网络或模型下载。
- **性能**：报告当前硬件/OS、cold/warm、样本数、全部请求墙钟、文本/DiT/VAE、MLX峰值和其统计范围。第一阶段先实现正确性，不预先宣称比旧引擎更快。

## 暂不等同于第一阶段完成项

独立内存池、通用优化编译器、所有形状 ANE、自定义高性能 GEMM/attention 全覆盖、H3/LTX 全功能数学 executor、App Store 发行和完整 HTTP/XPC daemon 都是后续明确模块，不能以骨架或单测代替。当前纵切与最终目标的差距应随每次验收记录更新。

## 本轮已经落地的精度约束

- Qwen/DiT SiLU 使用与参考相同的融合语义，避免 BF16 中间舍入改变后续结果。
- Euler 的 dt 和 noise 乘积先按 BF16 舍入，再与 latent 相加；自定义 Metal kernel 对多个非平凡 dt 和非整线程组元素数逐元素比较。
- VAE BN 的 eps 必须显式成为 BF16 scalar；C++ 默认 float scalar 会提升为 FP32，造成从 unpack 开始的数值差异。以逐阶段 fixture 定位后修复，最终两组真实测例均包含像素逐元素一致。
- 单个 prompt 的 conditioning cache 只在固定 model root 的 session 内复用。暂不支持原地替换权重文件；更换文件须销毁并重建 session。下一阶段以 content hash 加入缓存身份。

## 下一阶段的具体迁移顺序

1. 将 `tc_engine` 内硬编码的 Flux 成员改为 `ModelSession`/factory 注册表；PreparedPlan 绑定 executor 与 checkpoint 身份，unsupported 在 create/prepare 阶段返回。首轮保持同一真实 FLUX fixture 不变。
2. 引入 `StorageDescriptor`、`CompletionToken`、`WorkspacePlan` 和组件驻留预算。先接管一个自有 Metal 子图的输入/输出与同步，再逐项替换 MLX 热点；不得让 opaque MLX allocator 与自有 allocator 重复管理内存。
3. 按 `video-model-acceptance.md` 的迁移表分别引入 H3/LTX 的原生计算组件。兼容子图需显式声明其所有权、内存预算、输入输出和同步，不能只包 CLI 子进程作为统一后端。
4. 核心进程提取为 `turbociderd`：统一设备准入、模型驻留、作业表、artifact store。App 使用 XPC client，CLI/SDK 支持 embedded 与 daemon 两种模式；GPU 资源不得由多个进程各自无条件占满。
5. 增加默认不监听外网的 HTTP 作业 API：prepare/submit/cancel/events/artifacts；协议与 SDK 共享 schema，长任务采用 job handle。启动/关闭/崩溃恢复与进程断连独立验收。
6. 将已测 shape+checkpoint+OS+硬件+精度+artifact 身份写入 tuning database。只有质量、端到端延迟和完整内存统计通过的路线才允许 auto 选择；冷加载代价必须计入短会话策略。

以上是尚未完成的工程工作，不在本轮验收中伪报通过。新架构最终将模型数学、计算后端、资源管理和产品入口分离；本轮真实 FLUX 纵切为这些接口提供可重复的数值基线。

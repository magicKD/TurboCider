# H3 / LTX 迁移契约与延期真实模型验收

> 本文保留 2026-09-05 制定的验收契约和迁移背景，其中部分“尚未接入”的状态描述已经过时。当前实现与真实运行结论以 [2026-09-06 实现状态](../status/implementation-status-2026-09-06.md) 为准；尚未完成的项目继续使用本文门禁。

2026-09-05。本文不把参考源码可编译视为新 executor 已实现。

## 新框架已经覆盖的部分

`native/models/recipes.mm` 将 H3/LTX 阶段依赖、LTX 8+3 schedule、请求与解码尺寸，以及 executor readiness 纳入与 FLUX 相同的计划接口。`tests/native/test_contract.py` 验证它们；App 展示等待模型验收，不允许返回伪造的成功结果。

共享层来自真实 FLUX 纵切：C ABI、请求/计划、事件/取消、权重组件生命周期、原生张量后端、Swift SDK 与持久化任务。默认发行目标不启用 H3/LTX executor。迁移草稿已保留：H3 C 库会话/native AV I/O，LTX typed denoiser 与 Gemma 计算原型；它们不构成完整、已验收的模型实现。按最新范围，先交付 FLUX，视频模型随后依据本文迭代。

## H3 源码迁移表

| 参考资产 | 目标归属 | 重构要求 |
|---|---|---|
| h3.h / h3.c | models/h3/session | 将 h3_ctx 生命周期对应到 ModelSession；request activation 与常驻 weights 分离；不再由 CLI env 决定数学图 |
| h3_host / h3_dit_schedule | models/h3/recipe | checkpoint/LoRA identity、video/audio flow shift、4-step 与原始 schedule 显式区分 |
| h3_gpu / h3_metal | backends/metal + compatibility subgraph | 分阶段把 h3_gpu_tensor 纳入 Storage/CompletionToken；兼容期间原 h3_gpu 保持整块所有权，不能假装与 MLX 共用 allocator |
| h3_weights / safetensors | core/weights + H3 mapping | 提取映射/分片/流式预取逻辑，保留只读权重、关闭句柄、在途读写生命周期 |
| text/vision/multimodal encoder | models/h3/conditioning | 文本与图像/音频引用顺序、token 预算、预处理身份进入缓存 key |
| h3_dit / token refiner | models/h3/transformer | 不将 H3 块映射成 FLUX single block；保留双模态条件和 refiner |
| video/audio VAE / TAEH3 | models/h3/decode + media | 正式解码与预览解码区分，原生媒体时间戳/色彩与音量对照 |
| h3_coreml | backends/coreml + H3 partition manifests | 经公开 Core ML 路径重新验收；私有 _ANE* 不进入发行目标 |

### H3 无权重检查记录

2026-09-05 对 19 个 H3 主要源文件执行 `-fsyntax-only`：18 通过，`h3_metal.m` 一项失败。原因是当前 SDK 15.1 没有 `MTLGPUFamilyMetal4` 声明，而源文件仅做 runtime `@available(macOS 26.0, *)` 判断。运行时 availability 不能替代编译期 SDK 检查。

处理策略：完整 Metal 4 目标使用包含该声明的 Xcode/SDK；保留旧 SDK 构建时必须加编译期 guard，并明确禁用该能力分支。不要把 Metal3 枚举替代成 Metal4，也不要猜测私有数值。本轮未修改参考仓库来掩盖这个问题。

## LTX 源码迁移表

| 参考资产 | 目标归属 | 重构要求 |
|---|---|---|
| ltx.h / ltx.c / rng | models/ltx/config、sampler | 原始 schedule、RNG 和帧/latent 关系形成可单测模型语义 |
| tools/bench_block.c 正式生成流程 | models/ltx/pipeline | 从 benchmark main/环境变量拆出 library pipeline；benchmark 只负责测量与固定 fixture |
| ltx_conditioning / connector | models/ltx/conditioning | 原生 Gemma 接口、connector、learned registers、完整文本 mask |
| ltx_transformer_io / GPU blocks | models/ltx/transformer | Video/Audio 自注意力、文本注意力、跨流依赖明确表达 |
| MLX C++ upsampler / video VAE | backends/mlx subgraphs + models/ltx/decode | 首阶段允许整子图 MLX；显示隐式内存预算；后续热点迁自有 Metal |
| audio finalizer / vocoder / mux | models/ltx/audio + media | 去掉运行时 Python 子进程，保证采样率、声道、LUFS、时长、帧对齐 |
| ltx_ane_mlp / kv | Core ML manifests + LTX partition policy | rows 1001/4004、text 1024、FPS、权重映射与误差不能遗漏 |

### LTX 无权重检查记录

对 15 个 C/Objective-C/MLX C++ 源文件执行 `-fsyntax-only`，全部通过，包括主要 GPU/权重/conditioning、公开 Core ML MLP/KV、MLX upsampler/VAE。未执行任何 LTX checkpoint 读取、输出质量或性能测量。

## 有模型后的分层退出条件

### 共同数值门禁

1. 从各参考实现导出真实 tensor fixture；源 checkpoint、adapter、config、tokenizer、预处理和工具 commit/hash 固定。
2. 同 dtype/布局/算法路径尽量逐元素一致；不一致时定位到首个分歧算子，解释融合、舍入或求和顺序，不能直接放宽终图阈值。
3. 跨精度/ANE 路径预先固定绝对误差、相对 RMSE 与输出质量阈值。所有 tensor shape 一致、无 NaN/Inf。
4. 保存每阶段输入/输出与最终媒体；同一 prompt/seed 并不自动证明相同初始 latent 或 schedule。
5. 质量通过后才比较性能；跳层、减步、上下文裁剪、近似 attention 作为另一条算法路线验收。

### H3 必测矩阵

- 指定原始/指定 Turbo checkpoint 分开测；Turbo LoRA revision、merge scale、视频/音频 flow shift 验证，不混用 544p v0.1 和 768p v1.1。
- text-to-video+audio、first frame、last frame、first+last、有序 reference image/video/audio、带嵌入音轨 reference video。
- first/last 与 Ref2VA 互斥；reference 重排、删除后 conditioning/cache 必须改变。
- 完整层数 GPU 基线；TAEH3 preview 与正式 VAE decode 各自标识；视听同步与音频幅值检查。
- 相同 prompt 暖会话、不同 prompt reprepare、不同尺寸、memory streaming、取消后下一个任务。
- 首个建议 fixture 是小合法 canvas/短视频；基线确定后加入真实 480p 工作负载，再测试高分辨率。不得照抄其他机器的 64 GB 支持声明。

### LTX 必测矩阵

- 97 frames / 24 FPS / 704×448 的 two-stage pipeline；新契约要求宽高 64 倍数，704×480 明确拒绝，禁止静默改成448。若未来增加 crop/pad，必须增加独立输出策略字段并验收。
- Stage1 原始 8-step ancestral、BF16 latent ×2 upsample、Stage2 原始 3-step deterministic，保留所有对应 RNG 更新。
- 单 first-frame I2V：两个 stage 的 clean prefix、per-token timestep、mask 与 strength 语义。
- Gemma conditioning、connector、1024 context registers、视频 self/text attention、音频 self/text attention、跨流连接分别对照。
- video VAE、audio VAE/vocoder、final MP4/AAC：帧数、FPS、RGB范围/色彩、PCM长度、采样率、声道与同步。
- 当前 M4 Pro 48 GB 必须先做峰值准入，不能直接沿用 M4 Max 64 GB 实验配方。

### 引擎/产品验收

- 一个任务从 Swift App/SDK/CLI 发起都得到同一结果语义；模型不可用时错误发生在输出前。
- repeated warm、换 prompt、换 shape、模型切换不持续增长内存；Core ML footprint 另计。
- 取消在安全边界完成；失败任务不留下成功媒体记录；重启恢复 interrupted 状态。
- 冷编译/加载、文本编码、DiT、upsample、VAE、audio、封装与完整请求墙钟分开记录。
- 无 Python、ComfyUI、FFmpeg 子进程的正式发行链路完成，才标记该模型原生迁移完成。

## 明确不成立的验收结论

- 34 文件静态检查不能证明两个视频模型能生成。
- recipe 图通过不能证明 executor 已完成。
- 有旧引擎二进制不能证明统一 runtime 接管了张量和内存。
- FLUX 的逐张量一致不能外推到 H3/LTX。


## 后续代码接入分工与里程碑

### M1：锁定模型身份和预处理

每个 ModelModule 新增版本化 ModelManifest：checkpoint 分片哈希、tensor mapping 版本、tokenizer/encoder 配置、adapter/LoRA 合并比例、精度、预处理、shape bucket、模型许可来源。权重目录不可在活跃 Session 内热改；通过新 Session 明确切换。Core ML cache key 必须包含 partition 图/权重哈希、dtype/layout、shape、转换器版本、OS build 与硬件能力。

H3 FL2VA 和 Ref2VA 分开注册变体/能力。只有对应 transformer 的 Turbo 合并 provenance 通过，才使用该变体的4步 schedule；不能拿 FL2VA 的 Turbo manifest 对未合并 Ref2VA 执行4步并声称正确。v0.1 544p 与 v1.1 768p 的 merge scale、视频 flow shift 和音频 flow shift 独立核对。

LTX tokenizer 必须读取实际 Gemma4 tokenizer 的模型格式与 chat/padding 模板，不能复用 Qwen BPE 当成兼容。全部49层 hidden-state 的层选择/flatten 排序、缩放、mask、video/audio projection 与 connector learned registers 分别验收。

### M2：每个模型提供独立 Session

H3：`ModelSession → h3_ctx → text/reference encoders → av denoise → video/audio VAE → native mux`。会话配置只从已解析请求/profile 生成，不读取任意进程环境变量；C callback 不允许 C++ 异常越过 C 栈。磁盘流式权重使用双缓冲，每块完成后才允许复写，取消时 drain 在途 I/O/GPU。

LTX：`ModelSession → Gemma4 → hidden projection → connector → stage1 → latent upsample → stage2 → video/audio decode → mux`。直接传递张量/布局/owner/completion；不能通过 benchmark CLI、临时 BF16 文件和 Python audio finalizer 绕过 runtime。stage1/stage2 各自维护 RNG、sigma、first-frame mask/per-token timestep，任何“通用 sampler”必须能表达这些语义再提取。

现有 `experimental/video/ltx/vendor/ltx_native.h`/`ltx_blocks.c` 是 typed C API 草稿，脱离旧 benchmark main，但仍需内存/异常/数值复核。TurboCider 内 旧Git提交中的 `engines/ltx-mac/tools/bench_block.c` 包含首帧逐 token 条件，外部 ltx-mac 的对应版本缺少它；迁移以带此修正的版本为对照，记录源哈希。LTX Gemma 小型 BF16 随机模型的5个 hidden tensors 已与 ltx-2-mlx 参考逐值一致；这仅验证部分算子，不涵盖真实 Gemma checkpoint、投影、tokenizer、audio 或整条视频。

### M3：后端划分、缓存和调度

GPU 原生路径先完成数值基线。H3/LTX 自有 Metal 张量与 MLX 子图暂时各自拥有 allocator，以清晰的子图边界交换；跨框架资源桥接必须说明 storageMode、stride、dtype、对齐和 completion，禁止只凭指针宣称零拷贝。

GPU/ANE 同步策略属于模型 partition policy。H3 视频/音频模块依赖、LTX AV cross-attention 的先后关系不能照搬 FLUX 同输入 attention/MLP fork-join。每个候选切分先测转换/同步/拷贝成本，再以端到端墙钟决定启用，配置按机器与模型 identity 生效。

`prepare` 编译/加载与 `warmup` 独立计时，非活动 GPU 作业空闲时才预热。取消编译与取消生成分开；失败 artifact 不提交 ready。模型大于工作集时选择 component_staged 或验证过的 block streaming，预测/记录读盘与重复加载；内存不足不能依赖系统 swap 勉强完成后宣称加速。

### M4：数值、资源、产品门禁

| 层级 | 无权重时 | 权重到位后必须完成 |
|---|---|---|
| contract | shape、role、variant、schedule、非法输入与 unavailable | 相同请求由 App/SDK/CLI/service 得到一致语义 |
| math | synthetic 算子、shape/stride、mask、RNG、缓存失效 | 每阶段真实张量对照、第一分歧定位；GPU先过，再验量化路线 |
| media | synthetic RGB/PCM、时间戳、方向/色彩、原子导出 | 完整视频帧数、音频长度/采样率、视听同步与首尾条件 |
| resources | 错误/取消路径、handle释放、in-flight 生命周期 | 多轮warm/换尺寸/换prompt/跨模型、峰值、长跑、offload/cancel恢复 |
| speed | 不填写推理性能数字 | 同机同权重/切分/步数/精度，AB/BA重复，首请求/预热/暖p50和p95分别报告 |

性能门槛建议与 FLUX 本轮一致：预先固定5%暖中位数回归上限，超过则定位框架开销；不能减少层数/步数/文本或音频质量通过门禁。所有模型 ready 开关需要以上真实门禁通过，缺模型期间保持 false。当前未下载任何 H3/LTX 权重。

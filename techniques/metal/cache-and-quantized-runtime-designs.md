# Cache 与量化 runtime 深读：Candle、MLX-LM 和 paged serving 的差异

<!-- knowledge
id: technique-metal-cache-and-quantized-runtime-designs
type: method
stage: prefill, decode
operator: quantization, attention, scheduling
backend: metal, mlx, cpu-reference
device: cpu, gpu
symptom: measurement, memory-bound, lifetime
semantics: contract-only
boundary: producer-consumer
api: public, reference-only
-->

[GPU 导航](README.md) · [来源与选读区段](sources.md#deepdives-sources-20261016) · [vLLM Metal](vllm-metal-paged-runtime.md)

2026-10-16，固定 Candle `12bb5d1f6350f4ea9827d9086f213feec1f0af6a`、MLX-LM `93e6d4a6447b3080861985947f0820e76fcb957d`。本页继续追上一轮较浅的包装层；未构建 Candle/MLX-LM 完整模型，以下是源码机制，不是跨框架速度排名。

后续2026-10-17已实际完成同一Candle的[packed/view/copy验证](case-studies/candle-packed-views-and-copy.md)与[原始Qwen2完整图/queue/capture](case-studies/candle-qwen2-queues-and-capture.md)。这些新证据不反向改写本页历史源码快照；特别保留origin失败、CPU公开解码成本和未取得物理GPU时间的边界。

## Candle：quantized storage 支持不等于每个 op 支持

[QMetalStorage](https://github.com/huggingface/candle/blob/12bb5d1f6350f4ea9827d9086f213feec1f0af6a/candle-core/src/quantized/metal.rs)保持 GGML dtype 与 compressed buffer。forward 不是先调用公开 `dequantize()` 展开整个权重：M=1 路线使用 dtype 专用 GEMV，其余路线调用 packed GEMM。公开 `dequantize()` 在这个读本中却是 blit/readback→CPU block decode→创建 FP32 Metal storage，且要 `flush_and_wait_current`。若在每次 inference 前调用，会引入完全不同的同步/准备成本。

quantize/imatrix 路线也明确在 CPU；load_quantized 则可直接上传已有 block bytes。应分开“加载时一次性工作”和“forward 热路径”，不要因看到一个 CPU decoder 就断言整个量化 inference 在 CPU，反之也不要把公开解码 API 当成 GPU 内 tile 解码。

`fwd_mv` 对 rank2/3 的 batch 行做展开，逐行发 GEMV；`fwd` 的 matrix 路线要求 FP32 activation，按 4D layout/byte-stride 绑定，输出也是 FP32。它与 MLX 常见 FP16 affine group64 的接口/数学不同。不同 storage bit 数不能掩盖 activation、accumulator、output 与转换差异。

`candle-metal-kernels/src/kernels/quantized.rs` 的 matrix dispatch 使用 32×64 输出覆盖、128 threads 和 8192-byte TG workspace；M/V 路线另有 dtype 专用线程参数。Q8_1/Q8K 的 dtype 枚举/存储存在，但所读 matrix/get-rows 分支明确拒绝部分组合。可迁移知识是**按 op 建立能力表**，而不是只列“支持 Q8”。

## Candle：省 hazard tracking 需要自己保留依赖

`metal/commands.rs` 管理复用 encoder、按 compute-count 提交、residency 与跨 encoder 的输出 fence。读本的默认 count 为 50，并可配置；它是提交粒度，不是 GPU 最优工作量常数。untracked hazard 模式不能假定跨 encoder/command-buffer 边界会自动刷新缓存，因此 first dispatch 前需要相应 wait。

`flush_and_wait_current` 专为 CPU readback 保留特定当前 command buffer 的完成关系，避免另一个线程取走 in-flight 列表后，本线程“等待空列表”却读取未完成输出。RAII/Mutex 维护的是 owner/编码并发边界，不能把语言内存安全等同于设备可见性。

后续对照应分别量 host 编码数、GPU queue/span、readback 和最终消费者；不能仅调高提交 count 得到局部吞吐后忽略交互时延/峰值或取消 drain。

## MLX-LM：连续 KV、批处理和 rotating cache 不是同一对象

[models/cache.py](https://github.com/ml-explore/mlx-lm/blob/93e6d4a6447b3080861985947f0820e76fcb957d/mlx_lm/models/cache.py)的基础 KVCache 以 256-token step 扩容，保留 offset、按有效前缀返回 view；QuantizedKVCache 还保存 code/scale/bias tuples，append 时量化新 K/V。capacity、有效长度、逻辑 dtype 与实际 bytes 分别追踪。

BatchKVCache 用 left-padding/offset/_idx 区分每条请求；merge 可以构造矩形 batch 并搬移有效前缀，finalize 还处理 right padding。Rotating 类另外维护环形位置、保留区、时序重排与窗口 mask。截掉/旋转历史可能是模型 sliding-window 语义，也可能是有损策略；不能只因占用少就宣称 full-context 等价。

这些对象不是 vLLM scheduler 的物理 paged block 表，不能直接套用相同 prefix sharing、CoW 或碎片化假设。基础场景中连续 cache 简单、成熟；多请求复用与换入换出则要比较完整管理成本，而不是用“paged”这个名词推断性能。

## MLX-LM：异步生成的关键是 item() 放在哪里

[generate.py](https://github.com/ml-explore/mlx-lm/blob/93e6d4a6447b3080861985947f0820e76fcb957d/mlx_lm/generate.py)先分块 prefill，显式 eval cache state；decode 先构建/提交 `next_y`，然后才把前一 token `y.item()` 交给 host。依赖仍在 device graph 中，CPU 可提前构图，不必每层/每算子同步。

不是“永远不要 item()”：流式用户最终需要 token 值，关键是不要把 host round-trip 放在下一 graph 的前置条件上。vLLM Metal 把同类机制推广到 request membership、placeholder 与 serial 管理；门槛更复杂，不能把单生成器循环原样塞进多请求引擎。

## 三类方案如何公平比较

### 从实现迁移到自己的 runtime

先选择一个目的：降低 prefill 峰值、摊薄多请求权重读取、减少 KV 复制，或让 CPU 准备与 GPU 执行衔接；不要同时改变全部策略。

1. 分块 prefill 要保留完整 token/position/KV 语义，按真实块形状核对路由和精度；块越小并非越快。用[分块消费者案例](case-studies/mlx-prefill-chunking-and-numerical-routes.md)学习验证，不复制固定 chunk 大小。
2. batch 的逻辑有效长度、物理 padding、capacity 和 index 分开维护；filter/extend/extract 后验证每条请求的完整 cache 与输出，而非只比总 shape。参考 [ragged KV](case-studies/mlx-lm-ragged-cache-and-padding.md)与[连续 batching 交接](case-studies/mlx-lm-continuous-batching-and-cache-handoff.md)。
3. 异步生成先明确定义返回 token、已写入 KV 的 token 与 pending lookahead 的关系；恢复/交接时保存真实 producer 历史。少一次 `.item()` 等待的收益不能靠读取未完成输出取得。

最小对照保持同权重/请求集合/输出需求，分别记录吞吐、单请求延迟/公平性、准备/扩容、峰值与取消 drain；加入错误历史、不同长度、cache 更新和请求退出负控。容量优化、调度收益和模型质量分别评价。

| 维度 | 必须固定/记录 | 典型错误 |
| --- | --- | --- |
| 权重 | packed 配方、scale/zero dtype、原始权重身份 | 同 Q4 名字直接互换格式 |
| activation/output | source dtype、cast、accumulator、restore | FP32 Candle 岛与 ready FP16 MLX 岛混比 |
| cache | actual lengths、capacity、GQA heads、量化、window、prefix 命中 | 将矩形最大长度当每条请求长度 |
| 提交 | 每步 encode/commit/wait、async、collector | 编译/IO/CPU decode 混入某个 kernel |
| 生命周期 | view/base owner、donation、resize、CoW、cancel/drain | 共享内存就省略完成边界 |

本机 [原生 MPS / MLX 实验](case-studies/mps-vs-mlx-20261016.md)只比较 dense GEMM，不能给上表的 quantized/runtime 策略排名。可运行的原创控制为 [packed 格式](../examples/reference/quantization.py)、[cache identity](../examples/reference/caches.py)、paged 地址/预算，case `quant-formats` / `cache-identity` / `paged-runtime-contracts`。

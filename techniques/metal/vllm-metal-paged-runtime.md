# vLLM Metal 深读：paged KV、分区 attention 与异步 decode

<!-- knowledge
id: technique-metal-vllm-metal-paged-runtime
type: implementation
stage: prefill, decode
operator: attention, layout, scheduling, quantization
backend: metal, mlx, cpu-reference
device: cpu, gpu
symptom: measurement, memory-bound, lifetime
semantics: contract-only
boundary: producer-consumer
api: public, reference-only
-->

[GPU 导航](README.md) · [第二轮来源](sources.md#deepdives-sources-20261016) · [持续研究账本](../MAINTENANCE.md)

2026-10-16，固定 `vllm-project/vllm-metal@468b0c8d6a3f2afe617311d76ed25ee5e3b0d05e`。本页选读实际 Python owner/layout、C++ dispatch 和 Metal shader，不只复述服务 README。**未安装/运行 vLLM 服务，也未运行上游全部测试；CPU 学习模型不等于该引擎本机验收。**

## 1. 从调度器到物理 cache，而不是再造一套页分配器

调用链可沿以下文件阅读，具体范围和 SHA 在来源登记：

```text
vLLM scheduler / resolved KVCacheConfig
  → attention/caches/storage.py : KVCacheStorage / CacheViews
  → caches/kv_cache.py : MetalPagedKVCache.from_upstream
  → block_tables.py : scheduler page → kernel subpage
  → metal/paged_ops.cpp : primitive / buffers / strides / dispatch
  → pagedattention*.metal : per-token / split-KV / tiled / NAX
```

[storage.py](https://github.com/vllm-project/vllm-metal/blob/468b0c8d6a3f2afe617311d76ed25ee5e3b0d05e/vllm_metal/attention/caches/storage.py)使用上游分配和 `create_kv_cache_views` 的 shape/stride/offset，不分配独立的“MLX shadow KV”。按 layer region 导入一份大 backing 的子区间，使全池不必落入单个 Metal buffer。cache 的逻辑页号仍由调度器拥有。

这里的共享不是免同步：`CacheViews.__setitem__` 用 `mx.depends` 将原生写入挂到 region owner；兄弟 view 需要跟随最新写依赖。普通 MLX view 上做 index-assign 不一定改到外部 backing，因此页 copy/zero/scatter 走原生写回。copy 环路会先 snapshot 所有源页，再写任一目的页；顺序赋值会把 `[A,B]` 的交换错成 `[A,A]`。

`tensor_bridge.py` 也明确 MPS→MLX 前同步；DLPack view 保留 stride/owner，但 MLX buffer donation 可能写回源 tensor。`copy=False` 是别名/生命周期合同，不是“原 tensor 可以继续任意改动”。负 stride 的 Torch 导入会被提前拒绝，不能把失败交给底层进程 abort。

### 空池不清零：优化的是触页，不是模型算术

`allocate_kv_cache_lazily` 对无需 zeroing 的统一精度 attention 池使用 `torch.empty`；前提是所有有效 slot 写后再读、长度外 slot 完全 mask。Mamba state、mixed-precision 重解释等 `needs_kv_cache_zeroing` 场景仍按上游路径清零。统一内存里给多 GiB 池做全量 memset 会实际触页；去掉它可能降低首步驻留，但不表示 cache 容量上限变小，更不允许读取未初始化数据。

## 2. Scheduler page 与 kernel page 可以不同

`block_tables.py` 从 `(32,16,8)` 选能整除 scheduler page 的最大 block。大页 `b` 扩展为 `b*r ... b*r+r-1`，`r = scheduler_size/kernel_size`。

例：1056-token scheduler page → 32-token kernel page，`r=33`。不能只扩表而保留旧页 pitch！`bind_paged_attn_strides` 在原 cache 的第二维大于 kernel block 时，改用 `block_size * key_cache.strides()[1]` 作为 subpage pitch，同时保留 head stride。无需物化重排整个 KV；一处 stride 错误却可能产生有限但错误的 attention。

本地 CPU 控制枚举 528/784/1056 三种页、所有 token 地址，比较翻译前后的物理线性地址。它验证地址恒等式，未调用上游 kernel 或 Metal driver。

## 3. 四条不同 attention 数据流

| 路线 | 解决什么问题 | 已读实现要点 | 不能扩大成什么 |
| --- | --- | --- | --- |
| per-token / split-KV | 少 Q、长 KV 的并行不足 | 沿 KV 分区，再合并 `(max,sum,partial)` | decode 与 prefill 用同一 tile 最优 |
| GQA decode | 相关 Q heads 的 KV locality 与占用 | 每 SIMD group 一 Q head，按页取数，4-token 独立 dot，寄存器 online softmax | 一个 threadgroup 真正只读一次 KV；每 group 仍发自己的 loads |
| tiled prefill | 多 Q 行复用 KV，避免全 score | SIMD 8×8 MMA；Q/O 驻寄存器；mask 与 row reduction 在 fragment 内 | `simd_sum` 可直接替代按行 shuffle |
| NAX prefill | GPU 内 Neural Accelerator 矩阵路线 | MPP cooperative inputs，device→register fragments，无 TG staging | ANE；或与旧核 bitwise 等价 |

[tiled shader](https://github.com/vllm-project/vllm-metal/blob/468b0c8d6a3f2afe617311d76ed25ee5e3b0d05e/vllm_metal/metal/kernels_v2/pagedattention_tiled.metal)的 8×8 fragment 一行分布于四个 lane，row reduction 的 XOR mask 为 1/8；全 32-lane `simd_sum` 会把不同 query 行混到一起。paged K/V 经页表取数，不能假设 dense `async_copy` 可直接用于跨物理页 tile。TG padding 是上游设计假设，本轮没有 bank counter 复现。

[NAX shader](https://github.com/vllm-project/vllm-metal/blob/468b0c8d6a3f2afe617311d76ed25ee5e3b0d05e/vllm_metal/metal/kernels_v2/pagedattention_nax.metal)使用 `matmul2d(16,32,16)` 和 `relaxed_precision=true`；注释明确该模式影响 cooperative layout，且会截断 PV 的 FP32 P operand。它为了寄存器压力在每个 KV tile 重读 Q，**并非所有更快 attention 都必须把 Q 永久驻留**。源文件注释的 26.2 构建要求、安装文档 SDK 要求、运行 hardware gate 与 Apple API 演进是不同证据，不能混作一条普适最低版本保证。

## 4. GQA 分区器：完整工作量、实际长度与 scratch 分开

`paged_ops.cpp` 的 admission table 只列五组 `(Qheads,KVheads,D,block)`，不是按模型名字路由。对 `P=512,256` 依次检查：

`Qheads * sum_i floor(L_i/P) >= 33 * GPU_cores`。

33 是该读本的经验 SIMD-group 预算，不是 Apple 硬件常数。只把完整分区计入收益门槛；真正 producer/reducer 仍按 `ceil(L_i/P)` 执行尾部。CPU 一次扫描产生 immutable length plan，多层复用，避免每层 GPU readback 或重扫列表。

40-core、32 Q heads 的单路阈值可**推导**为 P256 从 10752、P512 从 21504 开始。对于 `[1,10752]`，应选 P256；若错误地用 `B*max(L)`，会误选 P512。上游还有特定 10-core M3 的短上下文与 D256 preference；不移植成所有芯片定律。

功能/资源 admission 还有 dtype、普通 decode 行、verification、TurboQuant/sinks/softcap/window、core count 与 TG memory 条件。临时数组按矩形 `B*Q*ceil(allocation_max/P)*(2D+8)` bytes 计算，读本上限 512 MiB。actual length 可以足够短，但过大的 allocation bound 仍会导致 fallback；不得只按实际元素之和估 scratch。

本页 CPU model只重建所选 work/scratch/M3 policy，不声称穷尽所有原生 guard，也不用于产品路由。

## 5. 混合批次：不要让一行 decode 占整块 prefill tile

`dispatch_paged_attention_v2_online` 可将 leading decode rows 与 prefill suffix 分开。decode prefix 使用自己的 context bound/length plan，不允许很长的 prefill 给短 decode 虚增工作量或 scratch。suffix 的 Q-block offset 必须与 `cu_seqlens_q` 的 packed 序列映射一致。

speculative verification 是另一类：多个短 Q 的 window kernel 复用 KV，仍可 split-KV；不能因 `total_Q > requests` 就误判为普通长 prefill。上游注释记录某些 core-count 下的分区边界会暴露 bitwise 差异，体现测试必须覆盖多 SKU 与路由两侧，而不只测一台 Mac 的输出。

## 6. 压缩 KV prefill：有界物化与页去重，而非永远直接 packed

原生 tiled/NAX dispatcher 不直接接收 TurboQuant，不表示整个 engine 永远不能用这些核。`turboquant_prefill.py` 在 CPU scheduler 元数据上筛选足够多 query 的请求，按实际引用的 kernel subpage 去重，预留 gather/restore/索引成本；超预算的 history 留在 compressed fallback。不能只算一条最长请求的 FP16 K/V 大小，也不能为一个 batch 分两路后漏算 Q gather 和输出恢复。

`turboquant_materialize.py` 的自定义 Metal kernel 根据原始 stride，联合 gather 与解码，寄存器中做 V 的 inverse FWHT，直接写 FP16/BF16 K/V，不构造 context-sized 全局 FP32 中间数组。只缓存页索引/计划，不缓存可能过期的解码 K/V；新写入后读最新 writer handle。收益需要足够 Q/GQA 复用摊薄物化，短 suffix 或部分 prefix-hit 可以更适合直接 compressed kernel。当前门槛与 dtype/head/format 校准属于该读本，不作跨芯片定律。

这种“压缩驻留 + prefill 临时 FP16 + decode 直接低比特”的分阶段策略，比全模型永久展开更细。`sdpa.py` 末尾实际使用 `mx.eval(out)` 再 `mx.synchronize()`，随后释放 K/V 对，避免多个 layer 的 lazy 工作同时保留物化副本而突破单层预留。它以更强完成边界换取 workspace 上界，不能将预算控制当成免费优化；`eval` 的结果就绪与 completion-handler 不再持有输入也不是同一边界。本轮只读源码，尚未运行这条路线。

## 7. One-step-ahead decode：保留 lazy token，延后 host 同步

[decode_pipeline.py](https://github.com/vllm-project/vllm-metal/blob/468b0c8d6a3f2afe617311d76ed25ee5e3b0d05e/vllm_metal/v1/decode_pipeline.py)把 sampled token 作为 lazy device value 传给下一步，CPU 在前一步 GPU 工作未完成时构建下一 graph；engine 最后再取前一步输出。它不让有依赖的两步数学无条件并行，主要隐藏 CPU 构图/编码与 host token round-trip。

关键安全点是：最多一份 pending + 一份已 resolve 未消费的结果；serial 防重复/乱序；提交前完成 value-independent bookkeeping；真正 token 用 `-1` placeholder 暂占；请求被驱逐后通过直接 owner 引用回填，不重插入 runner 字典。new/prefill/preempt/resume/structured/MM/spec/不支持 executor 等情况会先 barrier 再改状态。membership 缩小可 gather/reorder，缺 pending row 必须回退。

这些机制适合借鉴到重复 denoise、GPU→采样→下一步的流水，但需要自己的状态合同。不能把这个读本仅支持的 executor 条件推广到任意多进程服务。

## 8. Profiling 与采用建议

dispatch diagnostics 是 opt-in、process-wide 的最后一次记录，不是每请求 trace；selector query 也不是实际 execution 证明。kernel 数值、路由覆盖、TTFT/TPOT 和 HTTP serving throughput 应分别保留。benchmark 文档特意要求相同进程拓扑、token 历史、cache dtype/长度、warmup 与桌面 GPU 负载；不借用其上游倍率作为本机成绩。

`docs/profiling.md` 将服务 profiler 接口接到所选 backend 的 frame-capture：MLX capture 或 PyTorch MPS capture，而不是因为配置字段名叫 `torch` 就获得 PyTorch CUDA 风格 trace。捕获可能包含巨大的 KV backing；Xcode 的 Profile 会重放/重新分配资源，不能把一次普通 capture 当成无风险的逐 kernel benchmark。诊断时缩小 pool/请求窗口只是控制 trace 的实验设置，不能拿改变工作量后的速度与正式请求混比。本轮没有运行服务 capture，也没有改 developer directory 或系统权限。

优先可迁移设计：共享 allocation 的 ownership、layout 保真、页表/stride 联动、ragged-aware 计划、按 route 限定功能、immutable per-forward metadata、lazy 采样流水和可观测 fallback。后续应实际测试 quantized KV、prefix CoW、GDN state 与 serving；当前没有默认接入 TurboCider。

可执行参考：CPU 合同、测试。

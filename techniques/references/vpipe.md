# vpipe：低内存不是一个kernel，而是协作合同

<!-- knowledge
id: technique-references-vpipe
type: method
-->

不用原仓库也可学习：[packed / typed refill 设计](../designs/packed-streaming.md)、[阶段与驻留设计](../designs/residency-and-phases.md)和[独立实现](../examples/reference/contracts.py)包含核心算法、状态与反例。下列外部文件及产品接缝仅用于可选溯源。

[总目录](../README.md) · [Streaming](../memory/streaming.md)

## 核查来源与版本

本地`../references/vpipe`只读，HEAD `c2989a76841888e4d8d208fad4f476c487710962`；2026-10-15核查，不声称上游最新。[上游](https://github.com/tgo-app-dev/vpipe)，Apache-2.0。本轮重点读`docs/MODEL-MEMORY.md`、`generative-models/shared/{block-residency,streamed-refill}.h`与相关WeightSet/模型组织。

## 最值得迁移的合同

| 机制 | 核心问题 | 迁移建议 |
| --- | --- | --- |
| shared immutable WeightSet | 同checkpoint重复加载、原/derived冗余保留 | source identity去重；read-consume不进长期cache |
| declare→decide两pass | 并发初始化/遍历顺序使内存计划不一致 | 所有模型先声明，再一起决定条件释放 |
| phase intervals | conditioning/latent跨阶段存活 | 用真实生存区间max，不只给weight标单phase |
| scratch/payload申报 | 小VAE权重、大激活/输出被漏记 | 模型与媒体输出都入预算 |
| reusable typed refill | 每block allocation+mmap faults抖动 | pread到owner buffer，odd tensors局部慢路径 |
| block residency feedback | free/filecache算术自我放大 | fixedsubset、hysteresis/ratchet、实际resident观测 |
| parking/release | idle模型不把内存还同伴 | 只声明确实可释放的bytes；真实release确认 |

## 三个特别容易踩的坑

### 1. 顺序循环不要LRU

容量C<N时，`0…N-1`每轮扫描的LRU可能零命中。固定subset能让每轮确实少读C个block；增长不逐出其他block的座位，evict后降低ceiling，避免反复growth/evict。策略不直接拥有weights，模型负责真正释放。

### 2. “available”可能是自己的file cache

streaming读刚让file-backed计数增大，再把它看成新余量保留更多weights，会自我强化。vpipe当前合同加入实际weight residency反馈，而不只更换free公式；reserve已分配部分还要扣回，避免activation被保护两遍。

这不是建议用户随意wire整个模型。wired pool/内存cap/parking是另有权限与全局协作的策略，TurboCider应按现有设备准入和恢复合同实现，不在wiki脚本中自动改系统设置。

### 3. dtype / offset错了也可能“能跑”

Refill显式区分raw bytes与BF16 consumer：F16→BF16不能靠同宽reinterpret，也不能重复转换已有BF16 scales。`unservable`是单tensor慢路径，`failed`则目标可能partial-written必须无效。

`F_NOCACHE`和no-copy还依赖fileoffset/tensorlayout对齐；通用safetensors未必满足。Splash自有alignedpack与vpipe的Copied/pread偏好适用于不同来源格式，不互相证明谁总更快。

## 迁移到TurboCider的实际接缝

- [block_residency](../../native/runtime/block_residency.h)：工作集prefix/slots policy。
- [slot pool](../../native/runtime/streaming/slot_pool.cpp)、[context/StageExecutor](../../native/runtime/streaming/context.hpp)：layout/generation/last reader。
- [SourceLease](../../native/runtime/streaming/source_lease.hpp)、[GGUF bank](../../native/runtime/streaming/gguf_packed_bank.cpp)、[bounded read-ahead](../../native/runtime/streaming/read_ahead_worker.hpp)：完整来源、有限IO、错误传播。
- [model memory accounting](../../native/runtime/memory_accounting.cpp)：同一账本，不再造一套只计GPU的预算。

已有 [量化/streaming对照](../../docs/design/quantized-streaming-vpipe-comparison.md)与 [streaming设计](../../docs/design/streaming/README.md)包含不同阶段实测/待实施项。读代码和current receipt确认实际范围，不把设计卡当release资格。

## 对比速度的边界

vpipe README的H3表使用960×544/124帧/6steps、不同M4/M5机器且有散热说明；不能与TurboCider M4Max四步直接相除。可复用的是实验设计、完整phasefloor和数据组织，不是宣传倍率。

新模型先迁移source/tensor-table/声明/typedread，再做fixedsubset/lookahead，最后全请求低RAM实机。SSD带宽或64GiB模拟不能替代16/24GiB完整媒体资格。

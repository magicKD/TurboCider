# 技术组合不是叠开关：capability、effect 与逐阶段 schedule

<!-- knowledge
id: technique-designs-technique-composition
type: method
stage: dit, pipeline
operator: scheduling
backend: cpu-reference
device: cpu, gpu, ane
symptom: lifetime, stale-cache, precision-loss
semantics: contract-only
boundary: producer-consumer, request
api: reference-only
-->

[设计目录](README.md) · [本地 checker](../examples/reference/policies.py) · [模型阶段](../models/stage-map.md)

## 从 Sana 吸收的分层

Model adapter 申报可用接缝：attention、FFN、block loop、step loop、conditioning、token-set 等；技术只声明需要哪些 capability、读写哪些 seam、在哪个 phase 生效。一次性的权重转换/patch 在 build/load 运行；residual cache、skip、timestep-dependent precision 在 runtime hooks 运行。两者不能混成每一步都重新量化权重的隐式开销。

本地 `compose`接收小字典，显式列出 `id/requires/phase/reads/writes/steps`；先拒绝模型缺 capability，再检查 overlapping schedules，最后按 `(phase,id)`确定性排序。它不导入 Sana，不依赖 CUDA 或其类型系统。

## 三条具体规则

1. `attention_backend/ffn_precision/token_set/step_output`是 exclusive seams；同一 step 不能有两个 owner。互斥 steps 可以共享同一 seam。
2. 同 phase 的写→读有顺序歧义，应把生产者和消费者拆成明确 phases，或定义一个联合技术，而不是依赖列表遍历巧合。
3. disabled schedule 是空 steps，执行效果必须等于 identity，包含不偷偷保留旧 cache、不消耗 RNG、不改 dtype 或 position。

例如 channel-FFN 占 FFN precision，residual-cache 占 step output，结构上可能兼容；另一个 quantizer 同时占 FFN precision 则拒绝。若 merge 改 token-set，attention、RoPE、cache identity 的消费者都必须认识新 token geometry，不能仅让两个开关“都能打开”。

## Schedule 与数值不能被结构检查代替

实际参数应是 `(stage,step,layer,shape)`的函数：同一模型首尾 denoising step 可用不同精度；encoder 只运行一次；prefill 与 decode rows 不同；视频不同 stage 的 frame layout 不同。教学 checker 只检查一个显式有界 step horizon；stage 要由 adapter 先独立实例化，并分别检查。它不证明未知未来 step 或所有动态状态。

有损组合要测 full baseline、A、B、A+B 四组：量化误差可能放大 residual-reuse 误差，token merge 也可能改变缓存阈值。给定结构无冲突，不等于有限值、质量、资源 lifetime 或性能已通过。

### 例：少步采样与等价 kernel 的组合

把六步/五步采样与原始/融合 kernel 组成四组时，等价工作只在同一采样档内成立：分别检查原始输入、最终潜变量和完整输出；跨采样档只共享匹配输入，另评画质取舍。即使每个档内 kernel 输出逐位一致，也不能推导五步与六步等价，更不能把历史两项加速比例相乘。组合收益应从同轮完整请求直接计算，并固定 prefix/cache、encoder、VAE 和导出边界。

准入与执行也要分开：为复用旧 shape/source 检查而构造的局部请求副本，不得改变实际 sampler。逻辑覆盖应按真实 schedule 计算，例如首步 prefill、余步 decode 时为 `(actual_steps - 1) * layers`，不能沿用旧档的固定计数。开关和步数必须在每次请求显式恢复，缓存 identity 应包含 sampler recipe；迁移到新步数、shape 或有状态组合时，重新验证换档失效、同 seed 恢复与新增 owner 成本。结构兼容和覆盖计数本身不证明物理 kernel 数量或性能资格。

前缀命中还可能改变首步的阶段：若首步也成为target-only decode，应按 `(actual_steps - actual_prefill_steps) * layers`核对覆盖，实际prefill可以为0。采样更新次数不能简单从decode计数加1推算。encoder conditioning、reference encoding和DiT K/V可能是不同cache；操作切换可以保留其中一种、淘汰另一种。应分别根据源码owner/key和实际回执冻结命中预期，而不是要求所有cache同步hit/miss。[Qwen direct与前缀组合](../../docs/status/qwen512-qkv-prefix-composition-2026-10-18.md)给出此类边界及强前缀基线的有限案例，不代表其他schedule可直接扩大准入。

## 本地实现如何嵌入新模型

先写 adapter 的精确操作边界与 capability 表，再把技术变为纯配置/策略。通过 `compose`才建立执行 plan；执行时记录有效 schedule、source、step count、precision、cache hit 与 fallback。Cache 的实际 reset/异常路径由 [caches.py](../examples/reference/caches.py)展示，候选 share 的内存准入与拒绝外推由 [select_share](../examples/reference/policies.py)展示。

正反例包括 capability 缺失、两个 precision owner、同 phase 隐藏依赖、互斥 schedule 和 off identity。重新探索条件：新 stage、hook、layout、量化策略、approximation 组合或搜索 horizon。设计来源与独立实现边界见 [吸收账本](README.md)。

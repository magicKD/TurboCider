# 技术组合不是叠开关：capability、effect 与逐阶段 schedule

[设计目录](README.md) · [本地 checker](../labs/policies.py) · [模型阶段](../models/stage-map.md)

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

## 本地实现如何嵌入新模型

先写 adapter 的精确操作边界与 capability 表，再把技术变为纯配置/策略。通过 `compose`才建立执行 plan；执行时记录有效 schedule、source、step count、precision、cache hit 与 fallback。Cache 的实际 reset/异常路径由 [caches.py](../labs/caches.py)展示，候选 share 的内存准入与拒绝外推由 [select_share](../labs/policies.py)展示。

```sh
python3 -B techniques/labs/run.py technique-composition
python3 -B techniques/labs/run.py residual-reuse
python3 -B techniques/labs/run.py share-calibration
```

正反例包括 capability 缺失、两个 precision owner、同 phase 隐藏依赖、互斥 schedule 和 off identity。重新探索条件：新 stage、hook、layout、量化策略、approximation 组合或搜索 horizon。设计来源与独立实现边界见 [吸收账本](../references/absorbed-designs.json)。

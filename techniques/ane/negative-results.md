# ANE 负结果与来源纠错：如何拒绝一个看起来合理的加速方案

[ANE 目录](README.md) · [来源审计](source-audit.md) · [实验](experiments-20261016.md)

每条记录区分源码分析、上游报告和本机观察；新的 SoC/OS、编译入口、dtype、shape、权重、mask、layout 或消费者变化后可重开。负结果不是永久硬件禁令。

| 尝试/说法 | 证据与结论 | 回退或重开条件 |
| --- | --- | --- |
| “W8 解码为 FP16，所以 M4 没有 INT8 计算” | 某些旧博客/README 将局部实现泛化；Apple 明确支持 M4 W8A8 整数路径 | 区分 W8A16/W8A8，检查图与实际执行，见 [INT8](w8a8-int8.md) |
| “LUT4 小4倍，所以算4倍快” | 本轮 package 接近理想4×，延迟并未对应缩短；LUT4 随机多层误差很大 | mixed-bit/分组/训练式压缩，完整质量预算，见 [压缩](weight-compression.md) |
| “CPU_AND_NE 就是 ANE-only” | 本轮小 Conv 和动态训练 matmul 的计划首选 CPU；CPU_ONLY 小图结果接近 | 更大图/融合/布局，重新读计划与测量；不伪造物理归因 |
| “只换 computeUnits 不影响精度” | [既有同 bundle 对照](../review/qwen-ffn-tiling-20261015.md)曾出现 CPU_AND_NE sigmoid 数值拒绝，CPU_ONLY 通过 | 缩到单算子、保持 oracle；不将所有 Core ML 判坏 |
| “原生 SiLU 一定更可靠” | [ANEForge 单算子缩减](../review/aneforge-20261016.md)观察到约4%误差，exp 改写改善；只限定该版本/图 | 真实输入域、稳定 `exp(-abs(x))`、新编译入口重测；不任意放宽门槛 |
| “把 SD split-einsum-v2 直接用于任意 attention” | 已读代码未覆盖非512整除 tail，V2 无 mask 加法；是适用域风险，不是本轮硬件复现 | 长度/mask gate，保留 causal/全mask负控，见 [SD](stable-diffusion.md) |
| “runtime Conv/tile 在所有 ANE 都不支持” | Field Guide 主要是 M3 Max 的私有路径结果；其他项目/入口可能不同 | 保存最小图/错误/版本，在新设备/入口重测 |
| “低 tok/s 证明 ANE 算力太少” | Field Guide 发现 dispatch+weight streaming、融合和 W-lane batching 更关键；其拟合不是规格 | 降必要 dispatch、合批，分别测单流延迟/总吞吐 |
| “PMU 工具 exit0，所以有硬件 counter” | 本库原 CLI 缺模型也 exit0；81次有效推理未拿到 stats | stats availability 单独 gate；不可用保留 null，见 [profiling](profiling.md) |
| “共享内存去 copy 后就大幅快” | [真实五臂 FFN](../review/qwen-shared-ffn-20261016.md)的同client copy/shared 只约1.1%，首轮反向 | 相同client/图/输入/消费者，另计 dtype、pack、join、内部复制未知 |
| “Adam 不可能在 ANE，因为参数不能改” | 这是常量参数路径的限制；ANEForge 输出更新后 state 并 alias 回 input | 独立验 optimizer 数学/epsilon/moments/overflow，见 [autograd](training-autograd.md) |
| “loss下降就说明训练在 ANE 且正确” | 本轮公共手写反向 loss下降，但 plan 首选 CPU；无物理 ANE 训练证据 | 梯度 oracle、参数 generation、设备证据、GPU trainer 与收敛对照 |
| “MLCompute 曾公开支持 ANE training graph” | SDK26.5 `MLCDeviceTypeANE` 明确仅 inference，和某 iPhone 项目 README 矛盾 | 使用官方接口说明；私有训练单列，不继承宣传结论 |

## Field Guide 的方法比固定结论更可迁移

固定 [README](https://github.com/skyfallsin/apple-neural-engine-field-guide/blob/83e73eb94051dc67431829d0f3d3eb67db0b8e9b/README.md)、[tests 导读](https://github.com/skyfallsin/apple-neural-engine-field-guide/blob/83e73eb94051dc67431829d0f3d3eb67db0b8e9b/tests/README.md)与 [results.tsv](https://github.com/skyfallsin/apple-neural-engine-field-guide/blob/83e73eb94051dc67431829d0f3d3eb67db0b8e9b/results.tsv)构成“探针→保留/丢弃→引擎”的链。README 的性能表还有表头/列对齐问题，读者应回到原测量行确认 precision、prompt/generation 和 aggregate 含义；不机械抓表做跨机器排名。

程序编译成功但首 eval 失败、一次 tile 后本进程后续失败、RMSNorm 精度不够等，都应在子进程隔离，保留失败且停止复用可疑资源。不要为了继续搜索屏蔽错误、清系统 cache、关闭安全保护。

## 历史知识的正确位置

[hollance/neural-engine](https://github.com/hollance/neural-engine/tree/beef0000ab38df4807acef6dc331ddc97525b897)的价值是提出布局、算子回退、精度与观测的问题地图；作者自己声明是经验、不完整且可能有错。历史 unsupported-layer 清单不应覆盖今天的 compiler，旧“不能直接编程”也不能否认后来研究接口或 Core AI。保留日期、版本和重开条件，比删除旧结论更有用。

## 失败记录模板

保存原图/系数/输入身份 → 明确失败阶段 → 最小 witness → 全输出与错误 → 匹配 baseline → 影响范围 → fallback → revisit_when。不要用“driver error”解释所有问题；真实 finite 但错 stride、旧 output generation、Q/DQ 饱和、CPU fallback 和未拿到 counter 必须分成不同状态。

## 本地学习入口与复查

- [search_contracts.py](../labs/search_contracts.py)：原创CPU合同/显式实验客户端，不是上游完整实现。
- [silu_domain.py](../labs/silu_domain.py)：原创CPU合同/显式实验客户端，不是上游完整实现。

CPU入口：`python3 -B techniques/labs/run.py search-qualification`。只运行参考数学/合同，不自动触发Core ML、私有API或网络；实际设备命令与证据边界见[实验报告](experiments-20261016.md)。

SoC/OS/SDK、源码/编译入口、权重/LoRA、输入域/shape/layout/精度或实际消费者改变时重开；不继承旧数值、性能或物理设备资格。

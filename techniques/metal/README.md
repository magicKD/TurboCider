# Apple Silicon GPU / Metal 推理加速研究导航

[百科总目录](../README.md) · [路线图覆盖与核查](research-audit.md) · [固定版本来源清单](research-sources.json)

核查日期：2026-10-16。这里整理 GPU 的实现与诊断；独立 ANE 的图编译见 [ane](../ane/coreml-graphs.md)，跨设备交接见 [gpu-ane](../gpu-ane/shared-memory.md)。不把 M5 GPU 内 Neural Accelerators / NAX 叫作 ANE。用户需求前段提到 `techniques/ane`、结尾指定 `techniques/metal`；本专题按最终指定放在这里，交叉链接既有 ANE 内容，不复制两套。

## 先读什么

| 你要回答的问题 | 阅读入口 | 可迁移的核心 |
| --- | --- | --- |
| 引擎怎么选，为什么不能只按 tok/s 排名？ | [推理引擎地图](inference-engines.md) | 模型覆盖、部署 ABI、prefill/decode、权重格式与设备路线 |
| M5 之后怎么利用矩阵加速？ | [TensorOps / NAX / Metal 4](tensorops-and-m5.md) | 能力分层、cooperative tensor、量化格式与执行 API |
| 社区微架构数字怎样用？ | [微基准与硬件特性](empirical-hardware.md) | 历史观测、规格、推断与当前设备重测分开 |
| GPU 忙不忙，慢在 CPU 还是 GPU？ | [Profiling 操作手册](profiling-playbook.md) | host / queue / kernel / counter / 完整消费者分开测 |
| Q4 为什么不一定比 FP16 快？ | [GEMM/GEMV 与量化数据流](gemm-and-quantization.md) | small-M、tile 内 decode、split-K、驻留展开的成本 |
| Flash Attention、稀疏 attention 有何不同？ | [Attention 实现对照](attention-implementations.md) | online softmax、KV 复用、实际 dispatch 与质量边界 |
| 图像/视频模型如何落到 GPU？ | [扩散与媒体引擎](diffusion-and-media.md) | denoise 图、mask、量化、LoRA、VAE 和阶段驻留 |
| 哪些 PR 值得逐行学？ | [优化 PR 阅读账本](optimization-prs.md) | patch、状态、收益条件、失败边界与可迁移实验 |
| 本机到底验证了什么？ | [合成组件对照](experiments-20261016.md) | 两个独立进程/seed、全输出 CPU FP64 参考、ABBA 原始样本 |

已有技术卡继续使用：[kernel 调优](kernel-tuning.md)、[融合与布局](fusion-and-layout.md)、[command graph](command-graphs.md)、[attention 合同](attention.md)。新页负责跨项目综合与学习路径；真实 Qwen 的完整 FFN、原始 BF16、mask/stride 负例仍引用既有报告，不用小合成实验覆盖它们。

## 五条主路线，先明确加速的是哪一种成本

1. **少做内存往返**：producer-consumer 布局传递、融合 norm/activation、online softmax。数学基本不变，浮点舍入仍可能变化。
2. **提高矩阵数据复用和设备占用**：分开 GEMV/GEMM、按 shape 选 tile、split-K、cooperative tensor。新增 scratch/reduction/尾部成本必须算进去。
3. **少搬权重或 KV**：Q4/Q8、量化 KV、按阶段驻留。文件压缩、GPU 带宽下降与整数 MAC 加速不是同一个结论。
4. **少付 host/编译/提交开销**：lazy evaluation、`mx.compile`、pipeline cache、ICB/Metal 4 提交。热态收益要与首调用、重新编译、地址生命周期一起记录。
5. **减少模型计算量**：稀疏、cache、推测解码等。必须单列数学/分布变化，不能与同语义 kernel 优化合并宣传。

以上是本文的综合分类，不是 Apple 的硬件规格分类。每条都需要反例：布局错误、全空 mask、不同 source、非有限值、cold path、低内存或完整请求回退，见各专题。

## 对 TurboCider 的采用顺序

先建立原始 BF16/当前最快 GPU 完整消费者基线；确认是否是权重/attention/launch/内存瓶颈。优先复用成熟 MLX/GGML/MFA 实现和布局；只有 profiler 指向的热区才值得另写 kernel。TensorOps/NAX 作为独立受能力保护的候选，保留旧 GPU 路径。量化或稀疏候选再加完整层、扩散步、LoRA、图像/视频质量门禁；最后才比较 GPU/ANE 并行。

本专题没有修改生产代码、默认 backend、依赖或系统电源设置。没有下载模型，没有清理用户缓存；本轮可用空间约 45 GiB，小型探针不需要扩盘。

## 证据怎么读

`S` = 所列源码区段已读；`D` = 官方/项目文档已读；`P` = PR patch 已读；`R` = 本机已运行。它们彼此独立：`S/P` 不是性能资格，`R` 也不代表已完整审计引擎。来源清单固定 commit、逐文件 SHA256 或 PR diff SHA256；活动文档没有虚构 Git commit。不同版本/芯片的上游数字只作学习线索。

更新时先跑 [核查页](research-audit.md) 的命令。对新 SDK、新 GPU、wheel 或 PR 修订，重新取证；禁止将 2026-10-16 的 PR 状态写成永久事实。

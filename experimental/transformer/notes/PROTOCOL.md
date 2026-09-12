# 实验协议（2026-09-10）

目标：实测 M4 Pro 上固定形状、batch=1、causal pre-norm RMSNorm/SwiGLU Transformer 1–3 个独立 block 的异构并行收益。随机权重用于系统研究，不宣称语言任务质量或完整生成模型收益。

主机：M4 Pro，14 CPU cores (10P+4E)、20 GPU cores、48 GB UMA；macOS 26.6 (25G72)。历史 M4 Max 数字不能合并。

## 预先约定

- S=64/256/1024，H=256/1024/2048，F=4H；head dimension 默认 64，另测试 32/128。先小规模筛选，再扩展有前景和边界形状。
- L=1/2/3 独立权重、独立缓冲；逐层输出实际成为下一层输入。
- FP16 权重/激活；GPU MPS/MPSGraph SDPA，CPU BNNS，public Core ML CPUAndNeuralEngine。该配置排除 GPU，但本身不证明所有算子都在 ANE。
- 所有 plan 从同一 seed、同一完整矩阵提取 shard；GPU 对照重建同一矩阵。不会因切分比例改变模型。
- TP 拆 FFN 成对 up/gate 中间通道；对比 materialized、segmented-reduce、supergraph 完整 FFN shard。
- SP 指 FFN token-row split，完整 attention 保留全局上下文。不能误称完整 context parallel。
- head parallel 必须包括各 head 的 QK、causal softmax、PV；仅 QKV 投影切分单独标注。
- 内部 AB/BA 交替计时，保存逐次 wall samples；筛选与确认分开，确认至少 3 次 fresh-process、每次 >=30 对。候选最大值是有限矩阵的观测最优，不是硬件上界。
- wall latency 包含 CPU/GPU/ANE 同步、转换、join；compile/load 与热循环分开。p50、p95、配对速度比 bootstrap CI；结果报告失败和负收益。
- 正确性：finite、cosine、relative L2；建议 relative L2 <= 0.01，cosine >= 0.999；额外独立 FP32 参考验证，避免仅 GPU/异构共同 bug。
- public/private 比较必须同 shape/weights/math/I/O 边界；算子结果不得外推为完整 block。
- 不并发运行不同 benchmark；记录系统状态限制，不把未测功耗称为能效。
- 仅清理本次 manifest 中拥有的 packages、compiled models 和临时缓存；保留源代码、原始 JSON、图表、报告，不删除用户其他模型缓存。

## 实现审计记录

原 mac_local_ai synthetic stack 共享权重及输出；本实验改用独立 TransformerBlock 实例，且每层种子 1000+layer。CPU shard 原实现内生生成权重；增加外部 CPU shard 权重加载，以保持计划之间同一模型。发现 ANE=0 时 concat 无条件解引用 ANE 指针，已在本实验 adapter 中修正；零维 GEMM 直接跳过。原文件保留为 baseline.mm，修改逻辑集中在 adapt_baseline.py。

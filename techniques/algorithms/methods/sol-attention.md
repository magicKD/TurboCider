# Sol-Attn：在线路由、低分块补偿与真实 Metal 路径

<!-- knowledge
id: technique-algorithms-methods-sol-attention
type: method
stage: dit
operator: attention, sampling
backend: mlx, metal
device: gpu
symptom: precision-loss, launch-bound
semantics: approximate
representation: bf16
boundary: request, producer-consumer
api: reference-only
-->

配套：[机制 witness 与证据合同实验](../../examples/README.md)。这是支持性小实验，不是本方法的完整模型复现。

[算法入口](../README.md) · [PISA](piecewise-attention.md) · [Sana 集成](sana-sol-engine.md)

来源：Sana `sol_attn/mps`、interface 与文档 [S01](../sources.md#s01)，ComfyUI-SolAttn-MPS [S11](../sources.md#s11)，本库 LTX 反例 [S29](../sources.md#s29)。分类：`approximate-train-free`。本轮没有运行第三方整模型视频基准。

第三轮新增[固定原始Metal forward的七点组件实测](../case-studies/sol-mps-component-study.md)：双dense基线、四份真实Qwen早层输入、两份合成长序列和一份尾块控制，672次计时。另有[调试route shader的D128分配反例](../case-studies/sol-debug-bounds.md)；修正仅用于非计时参考，不修改原forward或上游。

## 不是简单 top-k 删除

先按连续 block 汇总 Q/K centroid、V sum；以 centroid 统计估计路由分数/阈值。重要 block 执行完整 QK/PV；不选的 block 用 proxy score 与 V summary 近似其 contribution，并进入同一 online-softmax 归一化。阈值系数 tau 控制 exact 工作量；通常更大 tau 选更少 exact 块，但 actual density 还受邻居/sink/dense-query 保护影响。

Sana Metal `preprocess.py::_routing_thresholds` 有 full covariance 与对角方差近似两种阈值统计。这里参数 `thresh_type="exact"` 的 exact 指**阈值统计**，不是整个 attention 变成 dense exact。tau=0 也不是天然 all-exact；验证 full-exact 必须直接强制全部有效 blocks 或用可证明的边界。

## 关键补偿公式

对未精算块 b，令 `μ_b=mean(K_b)`、`s_b=sum(V_b)`、`n_b=valid_tokens`，零阶近似分子贡献为 `exp(q·μ_b*scale)*s_b`，分母为 `n_b*exp(q·μ_b*scale)`。块长度必须同时进入分母；V 已经取 sum，不能再乘 n。exact 与 approx 集合必须互斥，且使用相同 max/exp2/scale。

这保留远端 softmax mass，但假设块内 keys 相近；高方差、相反 logits 的 centroid 可能误导。近似补偿不是数值精确恢复。尾块不能按固定64计权；padding/text sink 必须按 valid token 范围处理。

## Metal 源码怎么读

`mps/metal.py` 内嵌 MSL 包含 `sol_reduce_summaries`、`sol_route_tile`、`sol_accumulate_tile`：summary 合并 Q/K/V 遍历，tiled simdgroup matrix 做 QK/PV，FP32 online softmax 缩放历史分子/分母。路由放进 attention pass，避免物化全局大 connectivity map；调试 route mask 入口不应混进正常性能路径。

读到的 MPS 实现处理 half/bfloat16、head64/128 等特化，公开文档入口仍有较窄的统一 BTHD/BF16/dim128 合同；这是**底层 kernel 与公开 dispatch 范围不同**，不是文档允许任意 shape。Sana interface、Comfy patch 各自检查 layout、dtype、mask 与 fallback，实际支持要取三者交集。

代码依赖 `torch.mps.compile_shader` 和 PyTorch 随包 shader header，包括 `PrefillAttention.h`，是明显版本敏感点。Comfy 作者记录某些 torch2.12 包缺 header，回退 SDPA；应同时输出 `requested_backend`、`actual_backend`、sparse calls 与 fallback reason，性能验证使用严格失败模式。

## 上游 Apple 结果与本库负结果

Comfy 作者报告 M3 Ultra 60 GPU、BF16、56 heads/dim128、tau1.3 的 15,485 tokens attention 为 308.36→59.53 ms（5.18×）；对应 480p/124帧四步完整生成为 278.55→194.18 秒（1.435×）。这是作者受限场景，不是本机复现；抽查视频结构与完整定量质量矩阵也不是同一资格。[S11](../sources.md#s11)

本库 M4 Max 的 LTX-2.5 已有反例：720p direct top-k32 晚层 core 约4.05×，完整 Stage-2 仅1.238×、请求1.106×且质量门槛失败；all-exact sparse 路径反而慢。各策略配对与多数 pilot 的限制保留在 [原报告](../../../docs/design/ltx-stage2-sparse-results.md)。这些是借鉴 Metal-friendly 策略，不冒充完整 Sol-Attn 论文复现。[S29](../sources.md#s29)

## 下一步迁移方案

先 BF16/QKV 原语义强制 exact，做尾块、text-query dense、exact KV sink、scale/log2、stride、empty-row 控制；再扫 tau 和 full/diag covariance，记录每 head actual density。分别计 summary、routing、core、投影/FFN、完整 stage 和请求。

exact sink 在 packed 多模态模型中应由真实 spans 生成，不硬编码“前512都是文字”；跨 attention mask 不受支持时显式拒绝或 dense fallback。保留第一步、敏感 layers 与音视频条件，独立 prompt/seed 检查画质和时序后才做 cache/PTQ 组合。

论文：[Sol-Attn, arXiv:2607.24027](https://arxiv.org/abs/2607.24027)。

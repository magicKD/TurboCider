# W8A8：M4 整数计算能力与具体图路径不是一回事

<!-- knowledge
id: technique-ane-w8a8-int8
type: method
operator: quantization
backend: coreml, private-ane, cpu-reference
device: cpu, ane
symptom: precision-loss, memory-bound
semantics: approximate
representation: fp16, q8, w8a8
boundary: producer-consumer
api: reference-only
-->

[ANE 目录](README.md) · [能力地图](capability-atlas.md) · [本机实验](experiments-20261016.md)

## 官方结论与逆向观察如何同时成立

[Apple 量化性能文档](https://apple.github.io/coremltools/docs-guides/source/opt-quantization-perf.html)明确区分 weight-only 与 weight+activation quantization：以 A17 Pro/M4 为例，ANE 的 INT8×INT8 吞吐较前代提高，W8A8 有机会获得更多延迟收益；较旧设备也能因更低数据搬运量受益。这份资料不是“整数能力首次出现于M4”或“所有旧芯片只能FP16”的证明，更新SKU同样应复查。[WWDC24 压缩讲解](https://developer.apple.com/videos/play/wwdc2024/10159/)介绍相应量化与表示能力。

因此，“我测到 W8 解码为 FP16”并不否定另一种 W8A8 lowering 的整数路径。反过来，官方支持整数计算也不证明某个 private MIL 或某次 Core ML prediction 实际使用整数 MAC。审查结论至少拆成：**硬件可支持 → 图表达可识别 → 编译计划 → 正确输出 → 实际性能 → 物理算术证据**。

## 三条常被混写的实现

| 路线 | 典型表达 | 主要作用与验收 |
| --- | --- | --- |
| W8A16 常量压缩 | INT8 blob + `constexpr_*dequantize`，浮点 activation | 减文件/权重流量；对 FP 源权重与解码权重分别验数值 |
| 静态校准 W8A8 | representative activation → scale；输入/中间 Q/DQ + W8 | 给 compiler 形成整数区域的机会；观察分割、Q/DQ 融合、饱和和质量 |
| runtime codes/scales | CPU/GPU 动态生成 codes、scale，图内解码或低比特接口 | 降 staging/接口流量；包含 pack/rotation/restore 才能评价完整收益 |

公共 API 的实际示例是本轮 compression_survey.py：先 `linear_quantize_activations`，再 `linear_quantize_weights`，保存转换后算子计数并编译。8.3.0 已有非 experimental API；不照抄已弃用的 experimental 名称。校准输入与确认输入种子分离，但这些合成数据不替代模型校准集。

2026-10-17的[校准调用链深读](calibration-order-and-qdq-patterns.md)进一步验证固定8.3内部仍复用experimental目录的helper：历史range被最后样本覆盖，公共group参数未转发，临时观测CPU_ONLY；自动pattern不含本例matmul，API成功后可能没有任何activation Q/DQ。旧survey的原始输出/时间不覆盖，也不能把“给了四个校准样本”自动解释为完整合并历史极值。新证据中的union是显式局部控制，不是对旧实验回填修复。

[后续版本核查](coreml-versioned-calibration-and-ownership.md)区分9.0/main：9.0已转发group但尚未累计range；main源函数修正range，未据此运行新模型。9.0/main还有SDK26 build条件下的Python INT8输入branch，这只是输入admission，不是旧wheel、内部W8A8 lowering或整数MAC资格；main原生owner/GIL改动亦未成为已发布9.0的保证。

ccv 的 runtime i8 图、Splash 的 normalized W8A8 和本库 Public shared W8A8，承担的是不同的接口、量化 basis 和数据寿命；即使都含 INT8，也不能合并为一种实现，见 [异构对照](hybrid-runtime-comparison.md)与 [既有量化数学](quantization.md)。

## maderix 的 INT8 benchmark：学习表达，补上正确性合同

固定 [ane_int8_bench.m](https://github.com/maderix/ANE/blob/d91c9845c0784dec7753048954fc6d0e8411fe29/ane_int8_bench.m)构造深层 Conv 链，W8 通过 `constexpr_affine_dequantize`，层间插 Q/DQ。上游 README 报告 M4 某些配置约 1.85–1.88×；**本轮没有复跑其原 benchmark，不采用其 TOPS 为本机证据**。

已读代码还有重要局限：FP16 与 INT8 两臂分别生成随机权重，未形成同源配对；创建 IO 后所读路径未显式填充代表输入；warm/timed evaluate 的 BOOL 返回未逐次验收，未核对全输出数值。固定 scale 和深层无校验链也可能饱和或产生不可用结果。因此它适合提出“Q/DQ 是否改变 lowering”的性能假设，不足以作为“相同有效模型快 1.88×”的独立证明。作者对 SRAM 带宽的解释也不替代 counter。

## 可靠对照该怎样做

先固定 FP16 系数与输入，用独立 FP64 reference；同一权重量化得到 W8、W8A8，另做解码后纯浮点控制。校准记录 clipping/zero-point/scale 粒度，确认集检查 outlier、零行与不同层/step。每次输出都 finite、shape 相同且过门槛，失败不参与速度比。

同时保留 CPU_ONLY、CPU_AND_NE 的计算计划和实测；小图可能首选 CPU。正式性能不启用 profiler；诊断另采计划/trace。整数 MAC 没有独立证据时保留 `integer_mac_observed=false`，仍可以诚实报告“该 W8A8 图的完整接口更快/更慢”。

本轮大 Conv 链两种子只观察到约 1.1× 的 FP16→W8A8 热边界变化，小图反而慢；这与 M4 有整数能力并不矛盾，因为图规模、分区、IO 和固定开销不同。具体数值、误差、计划及局限全部见 [实验报告](experiments-20261016.md)。

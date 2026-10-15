# 实现实验室：从机制到可复现对照

[总目录](../README.md) · [完整 SOP](../guides/sop.md) · [技术卡模板](../templates/technique-card.md)

本目录把主题手册连接到可读实现、可运行控制和迁移练习，而不另造产品推理框架。`run.py`不导入MLX/CoreML；设备实验必须显式调用各自入口。新代码独立实现；生产能力仍复用仓库runtime。

核心 Python、Metal 与 native Core ML 样本只依赖本目录源码。按部件学习看 [阶段地图](../models/stage-map.md)，外部项目精髓看 [设计样本](../designs/README.md)；[旧生产夹具](../integrations/README.md)保留为明确可选接入，不是必需依赖。

## 三层实验，不混淆证据

| 层 | 入口 | 能证明什么 |
| --- | --- | --- |
| CPU reference / simulator | [run.py](run.py)，Python3.9+标准库 | 数学、索引、状态机、控制器、少量真实文件/队列IO；不是GPU/ANE速度 |
| 实际 Metal | [metal/run.py](metal/run.py) → [bench.mm](metal/bench.mm) / [MSL](metal/kernels.metal) | 小型真实GPU算子、完整输出和尾guard、host/device计时；不是全模型收益 |
| 实际 Public Core ML / MLX | [ane/bench.py](ane/bench.py) | export/compile/load/first/hot predict；可选真实GPU suffix与CoreML prefix；物理ANE放置仍未知 |

例子使用合成权重/输入，**不下载或加载用户模型**。独立编译产物/临时模型由各工具私有目录管理，报告只写新路径；不更换App、runtime库或系统参数。设备访问受沙箱/工具链权限限制时先报告，不伪装CPU fallback为设备测量。

## 按主题学习

| 主题 | 可运行实验 | 深入说明 |
| --- | --- | --- |
| Q4/Q8物理格式、range、RNE | `quant-formats` / `precision-boundaries` | [量化实现](quantization.md) |
| 完整W8A8/W4A8 FFN / LUT / Hadamard / scales | `w8a8-reference` / `w4a8-reference` / `smoothquant` | [量化实现](quantization.md)、[ANE设备](ane.md) |
| vector/SGMMA/fusion/norm-RoPE | Metal `vector/gemm/fusion/norm-rope` | [GPU实现](gpu.md) |
| online softmax / structured sparse | `online-attention`、Metal `attention` | [GPU实现](gpu.md) |
| graph batching / 两队列 | Metal `graph/overlap` | [GPU实现](gpu.md)；后者是GPU/GPU，不冒称ANE |
| 合法partition / LoRA | [CPU FFN示例](../examples/split_ffn_reference.py)、CoreML `--compare-gpu` | [调度实现](scheduling.md)、[ANE设备](ane.md) |
| M/F/H输入/H输出/KV序列与带宽 | `partition-axes` | [维度、merge与UMA](../gpu-ane/axes-and-bandwidth.md) |
| overlap / bank lifetime / calibration | `overlap-dag` / `slot-ownership` / `share-calibration` / `hybrid-policy` | [调度实现](scheduling.md) |
| cold start / bucket | `startup-amortization` / `bucket-selection`，CoreML lifecycle | [ANE设备](ane.md) |
| boundedpread / metadata / phase预算 | `bounded-io`、[inspect_model.py](inspect_model.py)、[interval_peak](../scripts/interval_peak.py) | [IO实现](memory-io.md) |
| condition cache / residual reuse / ToMe | `cache-identity` / `residual-reuse` / `token-merge` | [组合实现](cache-and-composition.md) |
| capability/phase/seam | `technique-composition` | [组合实现](cache-and-composition.md) |
| exact context / causal halo / upscale / frame sink | `global-stat-tiles` / `causal-halo` / `upscale-tiles` / `media-backpressure` | [媒体实现](media.md) |
| profile/嵌套计时/功率 | `profiling-scopes`、[summarizer](summarize.py) | [结果解释](profiling.md) |
| owned graph / 双 W bank | `owned-command-graph` / `dual-bank-handoff` | [owned graph](../designs/owned-graphs.md)、[hybrid FFN](../designs/hybrid-ffn.md) |
| immutable pack / typed refill / 固定驻留 | `packed-weight-store` / `residency-policy` | [存储设计](../designs/packed-streaming.md)、[阶段驻留](../designs/residency-and-phases.md) |
| Encoder / LLM prefill / decode / DiT / VAE | `stage-encoder` / `stage-llm` / `stage-decode` / `stage-dit` / `stage-vae` | [本地实现](model_stages.py)、[阶段方案](../models/stage-map.md) |
| 不同parallel拓扑 / heads | `parallel-branches` / `attention-head-shards` | [拓扑](../designs/parallel-topologies.md) |
| consumer stride / segments / identity | `consumer-strides` / `segmented-ffn` / `resource-hazards` | [消费者ABI](../designs/consumer-abi.md) |
| 可选三设备 / 精度岛 | `optional-devices` / `precision-islands` | [planner](../designs/heterogeneous-planner.md)、[源码项目对照](../references/gpu-ane-projects.md) |

## 一次跑完所有轻量机制实验

```sh
python3 -B techniques/labs/run.py list
python3 -B techniques/labs/run.py all --output outputs/my-lab-cpu.json
python3 -B techniques/labs/run.py w8a8-reference --options '{"rows":7,"hidden":32,"width":64}'
python3 -B -m unittest discover -s techniques/tests -v
```

40个case共享少量参考函数，但各自有不同的baseline/candidate/negative control；不是给所有章节重复挂同一FFN toy。`--options`只接受该case真实函数参数，CPU矩阵实验有规模上限；大shape转实际设备probe或修改受控实验，不暗示CPU上限是硬件限制。

## 小型真实设备实验

```sh
python3 -B techniques/labs/metal/run.py gemm --m 33 --n 41 --k 29 --output outputs/my-gemm.json
python3 -B techniques/labs/metal/run.py graph --output outputs/my-graph.json
.venv/bin/python -B techniques/labs/ane/bench.py --kind frozen-fp16 --compute-units cpu-ne --output outputs/my-coreml.json
.venv/bin/python -B techniques/labs/ane/bench.py --compare-gpu --output outputs/my-hybrid.json
```

Metal需要Apple GPU与macOS/CLT；CoreML需要兼容arm64 Python/NumPy/coremltools，可选MLX。本地 native INT8 matmul features另需CoreML9/macOS26/SDK26；旧环境返回明确unsupported，不能据此否定W8A8路线。

每次硬件benchmark串行运行，关掉本任务其他build/tests/profile，不擅自停止用户服务。低延迟小probe的百分比受主机调度影响，不自动推广默认。

## 对新模型如何使用

先找章末的“实现参考与实验”，再复制 [技术卡](../templates/technique-card.md)写实际shape/encoding/拓扑。跑机制控制 → 用真实source/activation替换toy → 完整block → 完整请求与质量/内存/生命周期。已有负结果给出失败条件和`revisit_when`，可以改变边界、布局、精度、bucket、硬件或consumer重新探索；安全与来源检查不因探索而关闭。

机器可读映射在 [catalog.json](catalog.json)，由 [check_catalog.py](check_catalog.py)核对章节、脚本、case与控制说明。目录索引完整不等于模型已经快；实际本轮执行范围见 [自包含验收](../review/self-contained-validation.md)；旧硬件结果在[历史实验记录](../review/labs-validation.md)。

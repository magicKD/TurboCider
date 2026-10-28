# 私有 ANE、图搜索与动态权重：能跑之外还要验证什么

<!-- knowledge
id: technique-gpu-ane-ane-private-and-search
type: method
stage: pipeline
operator: scheduling
backend: coreml, private-ane
device: ane, gpu
symptom: lifetime
semantics: contract-only
boundary: gpu-ane, request
api: public, private
-->

[专题导航](README.md) · [公共图/量化](ane-graphs-and-quantization.md) · [项目部署](hybrid-projects-and-stages.md)

私有路径用于研究，可能随 OS 变化；本专题不要求关闭 SIP/AMFI、不提权修改系统文件、不把上游安装命令自动执行。可用性失败就记录并回退公共框架/优化 GPU。

## 三个层次

| 层次 | 项目读本 | 学习内容 | 不代表 |
| --- | --- | --- | --- |
| runtime/编译器探针 | [maderix/ANE `d91c984…`](https://github.com/maderix/ANE/tree/d91c9845c0784dec7753048954fc6d0e8411fe29)、同内容的 MrZaKaRiA fork | MIL、`_ANEInMemoryModel`、IOSurface、compile/load/evaluate、动态权重 | 研究程序能执行不等于有完整产品恢复/安全合同 |
| 图 IR 与优化 | [ANEForge `896c494…`](https://github.com/sbryngelson/ANEForge/tree/896c4943c5ebf811c0e815f8d1381fd1e157fa7d) | fused e5rt program、lowering、route rewrite、缓存、数值门槛与计时搜索 | 不把每个 compile path 都叫直接 `_ANEClient` |
| 模型部署与负结果 | [ANE-LM](https://github.com/johnmai-dev/ANE-LM)、[Field Guide `83e73eb…`](https://github.com/skyfallsin/apple-neural-engine-field-guide/tree/83e73eb94051dc67431829d0f3d3eb67db0b8e9b) | chunked FFN、fused residual、CPU 支持算子、batching、program lifecycle | LLM 小模型或 M3 Max 的数据不等于 DiT/新 SoC 性能 |

相同 commit/内容的 fork 不算两份独立硬件验证。具体源码哈希和所读范围见 [来源页](../MAINTENANCE.md)。

## ANEForge 的可迁移方法

本轮读 `_optimize.py` 1–165 和 `_compile.py` 1–120。优化器为 DAG 建结构 key，按 op/shape/连边/属性与 input shapes 识别候选；NumPy 权重数组只记 shape/dtype，不含实际值。它适合复用性能探索，而不应作为 checkpoint/LoRA 数值资格。

attention query-tile 候选改变 query 的分段数，保持每个 query 覆盖完整 K/V，减少单图压力；native/decomposed route、常量折叠、INT8 又是不同候选。先合法化，再编译和测量，不能每次随机改整个 graph。

`_compile.py` 为图发出 fused MIL+blob，经 e5rt 执行；lowering 中还有针对特定 compiler/family 的 workaround。注释中的 M5 saturation/round/softplus 观察是限定路径的经验，不推成 ANE 全平台数学规格。跨芯片 compile check 也不是跨芯片执行/数值证据。

至少拆三种 cache：编译资产（graph+常量内容+toolchain）、性能搜索（SoC/OS/shape/layout/concurrent context）、质量（checkpoint/LoRA/recipe/验证集）。原子 rename 保护 cache 文件完整性，但不保证 key 语义完整或多进程测量互斥。

本库已有 [ANEForge 真机研究](../ane/case-studies/graph-rewrite-search-cost.md)包含 13 组比较和负控。其合成 host-fed 收益不能直接与 Qwen shared surface 或 GPU complement 拼成未运行的整体链路。

## 动态权重为什么与训练研究相关

在 `maderix/ANE` 的 `training_dynamic/mil_dynamic.h::gen_dyn_matmul`，activation 与 weight 被放进同一输入的不同 spatial 区间；图内 slice/reshape/transpose 再 matmul。`io.h::io_write_dyn` 则按真实 channel/spatial pitch 写 activation 和 W。这说明一种**避免每次权重变化都重新编译常量图**的表达，不是免费权重更新。

其代价包括更大的输入、每次写权重、图内 layout、可能不同的 compiler 路线。`gen_dyn_matmul` 的数学正确性、输出 layout 与当前 runtime 数值都应先验证；然后再比较 constant graph、runtime inputs 和 reload/delta 方式的完整成本。

[Orion `4dcc1c2…`](https://github.com/mechramc/Orion/tree/4dcc1c2a626c9ee1f35576bd0b290d489db17d08) README 描述 delta reload/LoRA 输入方案，属于另一种探索；本轮只读文档，不为“无重新编译”“生产质量”或训练加速倍率背书，也不把训练吞吐当推理收益。

## 负结果比“全放 ANE”更有价值

Field Guide 明确限定 M3 Max/macOS26.3.1，保留 chunking/fusion、W-lane batching，以及 runtime-weight conv、decode attention 等不盈利方向。其 const-weight matvec 的固定开销/带宽拟合是作者该路径的模型，不能写入 TurboCider 全芯片配置。

W-lane batching 的关键是把独立样本放进同一次执行，摊薄 dispatch/权重成本；提高 aggregate throughput 不意味着单用户 latency 同比例降低。多进程 program load 失败也不等于 ANE 算力已饱和。

## 安全且有用的探针顺序

1. 自己生成的小图：compile/load、shape/stride、finite、独立 CPU reference。
2. 固定 source 的 constant/runtime、FP16/W8 变体；失败图最小化并保留错误。
3. 完整 GPU→ANE→GPU consumer，含 copy、restore、ownership 和 stale 负控。
4. 优化 GPU 分母、serial/hybrid、冷/热/驻留；固定候选后独立输入确认。
5. 真实模型质量、取消/失败回收与长时运行后，才讨论产品采用。

任何阶段失败都不发布 partial output；先 drain 自己提交的工作再复用/释放。private compiler 暴露的内部端口格式以实际探针为准，别复制旧 `[1,C,1,S]` 假设。

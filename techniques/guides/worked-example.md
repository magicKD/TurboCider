# 桌面演练：给一个新SwiGLU图像模型接入GPU/ANE

[总目录](../README.md) · [新模型指南](new-model.md) · [检查清单](../templates/new-model-checklist.md)

这是接入设计的完整走查，不是一个已实现/测快的新模型profile。假设只有一个新模型的权重/API，依照本页应能决定改哪里、先测什么、何时拒绝候选；可运行部分是本wiki的CPU代数和预算工具。

## 已知事实

假设模型是sequential pre-norm Transformer：attention residual后进入SwiGLU，`H=4096,F=12288`，gate/up各`[F,H]`、down`[H,F]`，原BF16；首步有3144行，后续1024行；支持runtimeLoRA。这里的数字是演练设定，不能自动继承同形状Qwen的资格。

先写原始GPU完整FFN与完整block，确认tokenizer/position/scheduler/VAE。若新模型实际是parallelMLP、F16、不同activation或全局norm，必须改事实表和候选，不能继续套用。

## 1. 先证明拆分数学

运行CPUoracle：

```sh
python3 techniques/examples/split_ffn_reference.py
```

它验证同一逻辑通道切gate/up、down对应列、LoRA在正确位置和不均匀tail。真实模型另用其真实weight/activation oracle；不能用toy输出替代BF16/W8A8质量。

设计候选可以先给ANE4096、GPU8192（不是推荐比例）；down原physicalpitch仍是12288，不可写8192当stride。比较row split时又是另一条路线，不把channels和rows同一个整数混用。

## 2. 选图与来源

先PublicCoreML或已授权的researchbackend。Frozenartifact需checkpoint/shape/分区绑定；runtime模板需准确kind/H/F/rows/tile/LoRAhiddenABI。3144行需容纳它的fixedbucket，1024行选小bucket或普通GPU，padding不改变真实rows。

当前PublicW8row合同、Privatechannel合同、FP16runtime合同不同；先看 [GraphShape/Executor](../../native/backends/ane_runtime.hpp)与 [图合同](../ane/coreml-graphs.md)，不能把相同维度当成模板可互换。source第一次验证后保留generation/FD/owner，不把derivedW当原文件。

## 3. 实现模型适配层

在新`native/models/<model>/`提供模型数学与回调，复用 [HybridFfn](../../native/backends/ane_ffn.hpp)：

| 实际接缝 | 由模型提供 / 负责 |
| --- | --- |
| `FfnWeight` | 原base值、真实encoding/scales/offsets/group/physicalshape；无mergedadapter |
| `Gpu` | 已优化的完整GPUFFN，不换一个慢分母 |
| `ChannelGpu` | 给定logicalfirst/count的basepartial+correctedhidden，不在这里重复downLoRA |
| `Adapter::gate_up` / `channel_gate_up` | 原alpha/strength/rounding下激活前修正 |
| `Adapter::down_and_add` | 完整correctedhidden的一次downLoRA与原baseadd边界 |
| `NextWeights` | 不变的未来layer来源；保活到producer/consumerdrain |

这些是真实接口名，但表格不是完整可编译模型代码。sourceowner在`SourceScope`前建立，使异常析构先drain再释放source。模型module/Session登记、capability/error/actualreceipt沿现有registry接入，不另写私有ANEclient。

## 4. 保留真实依赖与fallback

先`plan_block(layer,rows)`决定完整GPU还是split。未选split走普通完整GPUblock；不要仍支付stage/bridgefence。选split时按合同stageweights，让attention完成后得到FFN共同输入，再GPU/ANE分工、join、残差，下一层只读完整发布输出。

不能把当前attention和依赖它的FFN画并行。测量模式按接口在完整block前清上游依赖，输出eval后`observe_block`；稳态不加多余fence。latechunk/finite/source错误全FFNGPU重算并记录，不拼半份旧输出。

## 5. 先规划内存再分配

写真实phaseclaims：encoder与conditioning、DiTtrunk/weights/activation、两Wbank/A8/restore/LoRAhidden、VAEhistory/workspace、writer和frameworkallowance。view用完整backing，跨phase区间延长。

```sh
python3 techniques/scripts/interval_peak.py techniques/examples/memory-plan.json
```

示例只是合成serializedbudget。新模型需用实际tensorbytes替换，并接原runtimeadmission；如果不fit，先componentrelease/typedstreaming，不能靠忽略CoreML或systemreserve让plan变绿。

## 6. 设计最小实验，不先开大网格

按首步/后续分别做：完整GPU、same-boundaryzero-ANE消融、一个固定share的完整FFN；再GPU-alone/ANE-alone/concurrent，含stage/LoRA/restore。候选通过才测完整block与整请求。多share校准的建议必须经独立实际executor试验。

测试非法pitch/shape/encoding、末tail、base→A→B→base、首/末chunk失败、IOerror、cancel/retry、返回结果后重用scratch、timeout/latecompletion。首步好而后续亏就分phase路由；不要求所有步骤都启ANE。

## 7. 将结论交给他人

填 [实验记录](../templates/experiment.md)，保留rawreceipts后可按normalizedschema运行：

```sh
python3 techniques/scripts/compare_runs.py techniques/examples/paired-runs.json
```

此处示例依然synthetic；真实模型记录需source/actualcall/quality/load/memory证据。最终交付不是一个speedup数字，而是精确device/workload支持卡、便携请求、matchedGPU结果、质量/生命周期、失败及默认理由。

## 走查结果

本wiki对新模型需要的九个问题都有明确入口：模型事实、强GPU、合法切分、图/来源、adapter、交接owner、内存、测量、产品资格。它可以指导实施和设计评审；在新模型真实权重运行与媒体验收前，不宣称这个演练已经产生硬件加速。

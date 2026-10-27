# Orion 优化pass与profiler：精度合同、估计预算和指标来源

<!-- knowledge
id: technique-ane-orion-rewrites-and-profiler-metric-provenance
type: method
stage: encoder, decode, pipeline
operator: layout, scheduling
backend: coreml, cpu-reference
device: cpu, ane
symptom: precision-loss, measurement, storage-pressure
semantics: contract-only
representation: fp16, fp32
boundary: producer-consumer, request, weight-runtime
api: public, private, reference-only
-->

[ANE目录](README.md) · [LoRA原调用链](orion-lora-inputs-and-scalar-serialization.md) · [42文件来源](sources.md#orion-sources-20261017) · [验证记录（过程原件已移出）](../MAINTENANCE.md)

固定Orion `4dcc1c2a`、2026-10-17批次。原C/ObjC compiler passes和profiler均实际编译执行；公开模型仅用本库合成输入及接口sidecar，不执行私有runtime或原CLI/训练。这里回答“优化是否保留原函数”和“一个看似硬件指标实际由什么产生”，不做速度排行榜。

## 1. 原pipeline到底执行哪些pass

[pipeline.c](https://github.com/mechramc/Orion/blob/4dcc1c2a626c9ee1f35576bd0b290d489db17d08/compiler/pipeline.c)按identity→cast→DCE迭代至fixpoint，最多20轮。conv-bias融合虽然在header和include里出现，body已明确禁用；不能从header注释就说该优化已生效。ANE wrapper另调用SRAM和uniform-output helper，验证仍须调用方读取结果；警告/返回bool不是自动修复tensor或退出。

每进程7组真实before/after IR：LoRA从17 live节点变16，cast消费者从7变5。两个进程重复确认相同fixture，不称14种独立算法。源结构validator检查DAG、引用、名字、banned concat和live output，**不检查完整dtype/shape/numeric域**。rank=0的LoRA图仍能通过结构验证并发射带inf scale/零维度的文本；本库明确不为它生成sidecar、分配输入或预测。不是说实际compiler接受了无效模型。

## 2. round-trip cast何时不是identity

[原pass_cast](https://github.com/mechramc/Orion/blob/4dcc1c2a626c9ee1f35576bd0b290d489db17d08/compiler/pass_cast.c)发现`cast(cast(X,A→B),B→C)`且A==C便绕过这两节点，没有检查中间B是否丢失信息。头文件以“ANE内部FP16”解释动机，但这不是所有声明dtype/公共CPU消费者的语义证明。

本批构造合法小IR并真正公开执行**优化前后完整消费者**，输入不是恰好FP16可表示的捷径：

| 路径及后续消费者 | before | 原pipeline after | 两进程×两policy |
| --- | --- | --- | --- |
| FP32→FP16→FP32，再`(z−1)*4096` | 保留显式round结果 | 中间round被删，读原FP32 x | 完整Y max差1.97216796875，4份after拒绝 |
| FP32→INT32→FP32，再`(z−1)*16` | 当前输入截断为1，Y=0 | 截断被删，Y最大8 | 4份after拒绝 |
| FP16→FP32→FP16，再`(z−1)*16` | finite输入的widen/narrow正控制 | 同一结果 | 4对before/after完整字节相同 |

第一条首个x约1.0001时，before消费者为0、after约.409668；不是只比较一个很小的round误差。显式cast、常量、sub和mul均由独立stdlib interpreter重放，实际公开结果符合各自发射后的函数，但8份after不满足**原before函数**的固定.02/.01目标。

原LoRA frontend本来全程FP16、没有这些cast，因此其normal路径通过与这个组件反例不矛盾。此处证明泛化pass缺少有限精度前提，**不证明默认GPT-2/Stories私有执行已发生此故障**；本批也未调用原private ABI。若编译器只允许特定dtype域，应显式拒绝/绑定资格；不要靠“目标名字是ANE”忽略可观测量化语义。

public plan的cast例operation记录可从8减到4，但数值失败就没有采用资格。少几个source/plan op不能直接解释为更快或更少硬件dispatch；本批没测性能，也没给失败arm配speed ratio。

## 3. SRAM和ANE validator是source policy，不是硬件探测

[pass_sram.h/c](https://github.com/mechramc/Orion/blob/4dcc1c2a626c9ee1f35576bd0b290d489db17d08/compiler/pass_sram.c)使用32×1024² byte常量，逐个live非const/input节点估算`shape乘积*dtype_bytes`，只在单节点大于预算时报warning。不算同时活跃数据、buffer alias、编译融合、权重/中间副本或物理驻留。

原函数在shape-only控制上得到：

- 一个32MiB输出：不warning。
- 增长到32MiB+4096 byte：warning。
- 两个各32MiB的live输出：仍不warning，因为没有节点单独超预算。

没有为这些预算图分配tensor buffer或执行预测。两个identity输出还可能alias，**不能反过来将64MiB逻辑和当作真实peak**。它证明检查公式/边界，不测得M4 ANE SRAM容量；header的“32MB for M4”保留为项目假设，不升级成Apple规格。

[ANE validator](https://github.com/mechramc/Orion/blob/4dcc1c2a626c9ee1f35576bd0b290d489db17d08/compiler/pass_ane_validate.c)对小compute节点应用49,152-byte门槛，并检查部分ordering/uniformity/banned-op。它与结构validator是不同结果，本批小图前者拒绝、后者接受，而public CPU预测可运行。这不否认特定private IOSurface分配经验；文档讲allocation，检查器讲每node tensor，不能不分入口互相替代。其“11约束”也不是把文档中的全部17条运行时规则自动验证完毕。

## 4. 原profiler实际计算的不是ANE PMU

[core/profiler.m](https://github.com/mechramc/Orion/blob/4dcc1c2a626c9ee1f35576bd0b290d489db17d08/core/profiler.m)接受调用方的eval_ms/total_ms、decode_ms和total_flops，然后计算：

```text
ANE utilization = sum(caller eval_ms) / sum(caller total_ms)
TFLOPS          = caller total_flops / caller total_time_seconds / 1e12
tokens/s        = 1000 * retained_decode_sample_count / sum(retained_decode_ms)
```

没有读取ANE counter或发现真实每op FLOPs。源码`total_tokens`参数未使用，最多保留4096个decode样本；p50/p90取sorted索引，不是插值median。本库独立native client链接原profiler，**只喂人工数字、不运行模型/私有runtime**，两进程共12个控制：

| 人工输入 | 原函数实际输出 | 应怎样解释 |
| --- | --- | --- |
| decode=[1,2,3,4]ms，eval10/total15ms，caller30G FLOPs | p50=3、p90=4、400 tok/s、比值2/3、2 TFLOPS | 验公式，不是测得设备2 TFLOPS |
| 只把total_tokens从1000改成1 | 上述速率/percentile相同 | 计数来自retained samples，非该参数 |
| 4096个1ms后再加两个100ms | 仍1000 tok/s、p50/p90=1 | 后两个样本被丢弃；全记录速率约953.9 tok/s，不同分母 |
| 两个零时长 | `+inf` tok/s | 保留无效/未知区间，不是无限推理速度 |
| eval20/total10ms | utilization=2，即200% | 未验证的overlap/不一致spans，不是ANE占用200% |
| 无记录finish | rate/ratio默认0 | 未观测，不代表真实耗时/忙碌为零 |

`rss_bytes`是原函数真实读取的**当前本native CPU进程RSS**，不是peak、相对rss_start增量、每model工作集或leak结论。人工时长和真实RSS必须分开标注，不能将整个结果对象统称实测硬件profile。

selected `infer.m`调用`record_decode`，finish后另把实测prefill wall写入返回struct。core helper本身的prefill字段一直默认0，所以既不能说CLI没测prefill，也不能把helper的0当测得零成本。固定tree的tracked `.m/.h`检索中`record_eval`只出现声明/定义，未发现调用；没有喂这个分母的utilization缺失/0不能解释成ANE idle。

## 5. CLI报告还要再审核一层

这里只读CLI，不运行模型/原benchmark、写baseline或训练：

- `bench_training`计时一次train_step后，固定按40%/60%分到forward和bwd+dW；这两列是**估计拆分**。Adam与reload另计wall，不能把40/60列当独立量到的硬件阶段。
- `bench_inference`的tok/s分母是decode函数调用区间总和，不含prefill/load/tokenize，sampling也在该区间外；不是整个request端到端吞吐。`mode_str`在timed fallback前决定，后来decode失败转CPU仍可能打印原标签，标签不能作设备归因。
- `baseline_compare`对所有指标统一以“上升>15%”警告，未区分延迟与吞吐方向；tok/s升高可能被警告，降低则未必。这里只证明所读比较规则，不冒充本机发生了真实性能退化。

这些源码很适合学习阶段拆分、回归记录和动态参数路线，但报告字段名、百分比、日志标签和实际硬件时间必须保持独立证据。建议每字段带measured/derived/assumed/unknown、边界、单位、sample count/drop count、真实backend尝试历史和资格；硬件profiler仍见[总览](profiling.md)。

## 6. 复算与边界

独立IR/指标数学、原profiler native client、collector、及portable tests均本地自包含。默认CPU入口不编译或运行上游：

raw审计使用`curate_ane_orion.py --root outputs/ane-orion-20261017 --references ../references --output-dir NEW`，由supervisor约束到300s和新output。它只用标准库读固定Git、模型源文件和完整自有数组，不load/predict；自身调用边界不与设备实验混计。

两项原观察器失败与后续原MIL字节确认保留在同批证据；前次source manifest的vpipe URL在发布前按实际remote校正为`tgo-app-dev/vpipe`，不改变数值或重跑设备。未改上游pass、profiler、产品路由或系统设置。commit/工具链/SoC/OS、dtype/cast域、source预算与输入shape、事件计时边界/采样策略或consumer改变均重验；实际private行为、动态训练、完整checkpoint质量与物理ANE profiling仍待研究。

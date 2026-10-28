# SpargeAttention入口合同：mask、causal、dtype与CDF不能只看签名

<!-- knowledge
id: technique-algorithms-case-studies-sparge-call-contracts
type: case-study
stage: prefill, dit
operator: attention, quantization, layout
backend: cpu-reference, metal, mlx
device: cpu, gpu
symptom: wrong-layout, precision-loss, measurement
semantics: contract-only, approximate
representation: bf16, fp16, q8
boundary: producer-consumer
api: reference-only
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

[算法入口](../README.md) · [Sage/Sparge机制](../methods/quantized-attention.md) · [数值消融](attention-precision-study.md) · [实验](../../examples/README.md)

2026-10-16第五轮。固定`thu-ml/SpargeAttn@ae5b629ebb41e41f86b3ea2ab5a3283f13ac151a`，不是远端最新状态或所有版本的结论。[S14](../sources.md#s14)登记Python控制、量化/LUT、C++ binding与sm80 kernel接缝。没有安装CUDA扩展、运行上游attention、修改上游或发外部issue/PR。

## 实际核对了什么

sparge_dispatch_probe.py核验六个完整源文件hash，再提取推荐`spas_sage2_attn_meansim_topk_cuda`、旧版`spas_sage_attn_meansim_topk_cuda`和CDF控制函数的原AST。只去除compiler decorator，函数body不改。

实际执行CPU Torch的dtype/layout控制、变量读取与参数传递。**替身**包括CUDA set-device、架构选择（人为选sm80）、pool/selector/quant、sparse kernel以及CDF prefix-fill；未加载Triton/CUDA模块。kernel只写零sentinel，不能将其finite输出当作attention正确性证明。

因此本轮记录的是“入口把什么传到了哪里”，不是“Mac模拟出了原CUDA内核”。八个dispatch和五个CDF观察完整保留在。

## 八个入口观察

| 调用 | 观察 | 能得出的结论 |
| --- | --- | --- |
| 推荐入口默认值 | selector/kernel桩各一次 | 正向控制能到达指定dispatch路径，不代表实际内核执行 |
| 提供显式attention mask | 未拒绝，mask未进入selector/kernel调用 | 该参数签名不是任意mask支持合同 |
| dropout_p=0.2 | 未拒绝，无对应下游参数 | 不能按通用SDPA dropout语义调用此固定入口 |
| 推荐入口is_causal=True | selector收到true，sm80 kernel flag仍为0 | block plan的causal与token级causal flag脱节 |
| 旧入口is_causal=True | selector收到true，kernel flag为1 | 对照证明不能把两个入口视为完全相同 |
| smooth_k=False | selector前发生UnboundLocalError，变量km | 未赋值变量路径实际可到达；不是正常禁用平滑 |
| output_dtype=float32 | 实际allocation仍为float16 | 当前dtype由输入cast和empty_like决定，不能只读请求参数 |
| 非法layout | 立即AssertionError，无下游调用 | 有效拒绝控制；上述其它问题不是桩总是放行 |

测试输入为HND/FP32，源入口先转FP16；未穷尽BF16、NHD、sm89/sm90、GQA或所有wrapper。显式mask/dropout的忽略由所选body与下游实参共同界定；没有用零sentinel制造输出相等的“证据”。

## causal链为何要读到C++

`core.py`推荐入口向sm80扩展传入的flag为false；`attn_cuda.h`将该位置声明为`is_causal`，`pybind.cpp`绑定到相同函数，`.cu`用它做`DISPATCH_CAUSAL`，`.cuh`据此选`MaskMode::kCausal`或`kNone`，决定tile内部的causal mask。

即使block map已删掉明显位于未来的块，同一个保留tile内仍可包含未来token。原创标准库case中，一个全保留的2-key block让第0行从causal结果0变为5，证明block mask不能替代逐token三角语义。**这不是本轮执行CUDA得到的输出偏差**；本轮确证的是所查参数链，实际模型影响需另测。

对非因果视觉DiT，默认`is_causal=False`可能完全符合模型。风险集中在自动替换通用SDPA、packed text/visual或其它带mask消费者，不能因此宣称所有生成模型默认已出错。已有[Sol组件](sol-mps-component-study.md)采用单独text-causal替换，正是不同消费者语义需显式处理的例子。

## CDF阈值不等于保留mass下界

原`get_block_map_meansim`由CPU Torch计算CDF与`searchsorted(..., right=True)`，将返回index直接当prefix个数；所读Triton fill只对0个特判为1，不普遍加1。本轮pooling桩固定为概率0.5/0.3/0.2，prefix-fill桩按该源循环建map：

| 阈值 | searchsorted返回个数 | 实际保留mass | ≥阈值？ |
| --- | --- | --- | --- |
| 0.4 | 0，fill保底1 | 0.5 | 是 |
| 0.79 | 1 | 0.5 | 否 |
| 0.8 | 2 | 0.8 | 是 |
| 0.9 | 2 | 0.8 | 否 |
| 1.0 | 3 | 1.0 | 是 |

这里关闭额外sink/low-sim追加，才能隔离CDF逻辑；真实fallback可能保留更多块。这反例只否定“该阈值天然保证保留mass不少于阈值”的解释，不否定所有CDF路由或所有输入。

若移植时要求标准top-p下界，应另明确越过阈值的prefix定义、等号、尾部、最少块与fallback，并做对照；本轮没有修改上游计数，也没有把修正提案冒充原实现结果。

## Apple adapter的准入条件

先枚举实际mask、causal、dropout、dtype、GQA与valid spans，再选择后端；不支持时显式拒绝或记录真实dense fallback。plain visual attention、causal text与joint multimodal不共用未经检查的替换开关。

同时输出requested/actual dtype/backend、sparse调用次数和fallback原因。`smooth_k`禁用异常若被外层吞掉，应记录为后端未engage，而不是稀疏成功。重新审读其它commit后再更新这些结论，不能把固定版本问题永久贴到项目名上。

第二条仅运行原创标量合同。默认tests/curator不运行上游body、CUDA或GPU；它们校验已保留的参数、异常、prefix和来源，不授予完整模型资格。

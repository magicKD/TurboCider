# MLX实际attention、wide GEMV与split-K：把源码预测接到本机dispatch

<!-- knowledge
id: technique-metal-case-studies-mlx-attention-routes-and-wide-gemv-observed
type: case-study
stage: prefill, decode, pipeline
operator: attention, gemm, norm, rope, layout, scheduling
backend: metal, cpu-reference
device: cpu, gpu
symptom: measurement, precision-loss, launch-bound, memory-bound
semantics: approximate, contract-only
representation: bf16, fp32
boundary: cpu-gpu, producer-consumer, request
api: public, reference-only
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

> 固定版本实现案例，供理解机制与复核边界；学习先读[方法导航](../README.md)，数据保留规则见[证据说明](../../MAINTENANCE.md)。

[GPU导航](../README.md) · [PSO关联合同](mlx-dispatch-pso-and-function-association.md) · [此前GQA/score精度](mlx-sdpa-gqa-fallback-and-score-precision.md) · [此前wide/head归因](mlx-small-m-wide-gemv-and-head-attribution.md) · 完整汇总 · [来源](../sources.md#mlx-dispatch-sources-20261017) · [验证（过程原件已移出）](../../MAINTENANCE.md)

2026-10-17，第二十四轮。此前只凭shape/源码guard不能确认安装wheel实际执行哪个variant。本轮从公开创建/绑定/编码/完成API取得实际函数名及grid，将选定Qwen路径接到原生PSO。这里的“观测到”指真实**direct dispatch编码与已完成父CB关联**，不是反汇编、单个kernel GPU耗时或每层IR标记。

## 1. 同一个fast SDPA API，三条实际路线

固定原Qwen24层、D64、14Q/2KV、GQA7、无quantized KV。每个cell在两文章、两个order replica中重复，下面的每次Model调用计数一致：

| 调用 | 实际函数族/数量 | 独立softmax | 完整direct编码数 BF16 / FP32 |
| --- | --- | --- | --- |
| fresh prefill7 | 无full/vector SDPA入口；QK/PV等显式图 | 24 | 651 / 771 |
| fresh prefill9 | `steel_attention_...`，24 | 0 | 459 / 579 |
| fresh prefill32 | `steel_attention_...`，24 | 0 | 579 / 579 |
| 任一prefill后单query decode | `sdpa_vector_...`，24 | 0 | 411 / 411 |

7-token BF16的实际`block_softmax_precise_bfloat16`、scale/mask/index/copy以及两个matmul族，和[前轮typed fallback数学](mlx-sdpa-gqa-fallback-and-score-precision.md)一致。FP32也走同一类图，不因dtype变宽就绕开query×GQA guard，但score不再落BF16。9/32则出现full入口；单query decode的nomask只适用于本轮scalar有效KV，不授权ragged padding下去掉mask。

来自[固定Metal host源码](https://github.com/ml-explore/mlx/blob/64ea011cb65f14d9ce2737e60db9a4ae91ed7441/mlx/backend/metal/scaled_dot_product_attention.cpp)的准入仍是软件合同：query≤8还需query×GQA≤32。新记录补了实际wheel的名字/geometry，不把32升级成硬件普适query极限，也不把本例指认为vector shader算错。

每次49个RMS入口、48个RoPE及24个compiled-elementwise入口与固定模型结构相符。名字汇总不是逐层IR所有权：公共API没有层号，不能只按重复序号硬填layer0..23；部分copy/cast也不能在没有buffer语义的情况下全部叫cache更新。

## 2. small-M wide真实出现，但不是所有小M/dtype

[wide配置源码](https://github.com/ml-explore/mlx/blob/64ea011cb65f14d9ce2737e60db9a4ae91ed7441/mlx/backend/metal/matmul.cpp)的FP16/BF16、M2..15、K/offset/stride/architecture guard此前已读。现在实际BF16 prefill7可见：

```text
gemv_wide_bfloat16_nv4_kl16_nc0_axpby0
gemv_wide_bfloat16_nv4_kl16_nc0_axpby1
```

prefill9的同类名字为nv5/kl16；Model中两者分别有193/169条wide编码，7的额外24条包括fallback QK使用的另一wide形态。不能把193全部叫词表head。

存在一个更具体的词表宽候选：BF16 prefill7/9各有一条wide编码的grid=`[1,37984,1]`、group=`[32,2,1]`。151936/4=37984，与源码每组4个输出行、vocab-wide grid_x1相符；7/9分别nv4/nv5。两个文章×两长度×两rep共8条这样的候选。

这比仅看shape更强，但未观察buffer/IR节点身份，仍称源码一致的词表宽候选。切到M1 decode后是regular GEMV；BF16 prefill32和本轮所有FP32调用都没有wide入口，符合已读dtype/M条件。不是“MLX所有matmul使用wide”，也不是已测wide开关收益。

## 3. FP32的120个split-K为什么是240条编码

FP32每次prefill实际出现120条`steel_gemm_splitk_nt_float32_float32_...`以及120条归并入口；归并分为72条`..._axbpy`和48条普通accum，名字按原库拼写保留。另有49条非split-K的主线GEMM；7-token attention fallback再增加其他QK/PV路径。

因此“289条GEMM族记录”不能等同289个高层nn.Linear。部分投影把K拆成并行partial，随后归并/应用bias；中间FP32 partial及额外提交是方案代价。相关源码是[`steel_gemm_splitk_axpby`与选择器](https://github.com/ml-explore/mlx/blob/64ea011cb65f14d9ce2737e60db9a4ae91ed7441/mlx/backend/metal/matmul.cpp)，并不因名字有splitK就天然更快。

本轮没有对其分配scratch/DRAM字节或做matched split-K禁用A/B，不能把FP32比BF16多的编码数归为单一性能原因。dtype、reduction、输出存储与whole-model预算也不同；原FP3272通过/BF1672拒绝保留，无跨dtype速度排名。

## 4. 资源属性必须跟具体PSO匹配

下面是本轮实际入口而非一张通用硬件规格表；full名字还区分9的非对齐和32的对齐variant。

| PSO族 | grid / group | 该PSO max threads | static TG bytes BF16 / FP32 |
| --- | --- | --- | --- |
| full D64 | [1,14,1] / [32,4,1] | 128 | 9728 / 17920 |
| 此短KV、单query vector | [14,1,1] / [1024,1,1] | 1024 | 不据此推总内存或occupancy |
| 词表宽wide候选 | [1,37984,1] / [32,2,1] | 记录在PSO表 | 静态值不含寄存器/stack/dynamic绑定 |

full的max128是编译PSO约束，不是M4设备总上限；也不能拿vector的1024上限授权任意full shader使用1024线程。当前单query vector的实际group并不是“32×GQA×query”的机械布局；不同vector/多query/分区variant要分别读source和API记录。

grid/threadgroups乘积只是API请求的逻辑线程计数，含尾部/guard，不等于活跃SIMD lane、有效FLOPs或GPU利用率。static TG也不是总动态TG、寄存器压力或spill。没有counter/ISA时，不给这些值补硬件故事。

## 5. 从观测到优化，还缺哪些验证

24次full入口替换显式图，解释了launch/中间score路径为什么可能变化；wide的多向量复用和split-K的partial并行也都有真实入口。但7与9包含不同真实token数，不是同工作量性能A/B；少192条编码不能叫相应比例加速。

本轮实际Model scope只有CB级GPUStartTime/EndTime。不能把CB耗时按dispatch数量均分，更不能因head候选所在CB最长就称head是瓶颈。C API元数据、source匹配、完整输出、原数学门槛和最终任务/性能采用继续分账；所有qualified performance为null。

CPU实现/tests覆盖全部外键、最近绑定、grid语义、PSO上限及错误关联负控；完整汇总和两份metadata trace保留每个捕获记录，不包含权重/真实输出向量。ordinary tests不执行GPU。

复查条件：模型/实际prefix、dtype、mask/GQA、M/N/K/layout/offset、compiler/source/wheel/SDK/OS、PSO工厂与encoder/dispatch形式或consumer变化。后续优先给选定pipeline接可信counter/采样边界、同完成语义的真实长请求和更多引擎；不将本机函数名或资源属性迁移成M5/NAX/ANE、全Metal后端或生产默认资格。

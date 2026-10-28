# MLC/Metal 原始量化 FFN：融合、低 batch 调度与 host 边界

<!-- knowledge
id: technique-metal-case-studies-mlc-quantized-ffn-and-lowbatch-boundaries
type: case-study
stage: ffn, prefill, decode, pipeline
operator: gemm, gemv, norm, swiglu, quantization, layout, scheduling
backend: metal, cpu-reference
device: cpu, gpu
symptom: compile-bound, launch-bound, precision-loss, wrong-layout, lifetime, measurement
semantics: approximate, numerically-equivalent, contract-only
representation: q4, packed, fp16, fp32
boundary: weight-runtime, producer-consumer, cpu-gpu
api: public, private, reference-only
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

> 固定版本实现案例，供理解机制与复核边界；学习先读[方法导航](../README.md)，数据保留规则见[证据说明](../../MAINTENANCE.md)。

[GPU导航](../README.md) · [前轮TVM AOT](tvm-metal-aot-and-runtime-only.md) · [固定来源](../sources.md#mlc-quant-sources-20261017) · 完整输出与失败 · [验证快照（过程原件已移出）](../../MAINTENANCE.md)

2026-10-17，第十二轮。已执行原始MLC GPU quantizer、`quantize_model`、`GroupQuantizeLinear`、QWen2 FFN/RMSNorm、注册的完整MLC compiler pipeline和导出后的Relax VM。冻结的**584个完整输出中578通过、6个高幅度codebook FFN输出失败**，全部独立CPU回放；另保留原生trap与ASan诊断。没有用手写量化器替代原实现，没有完整模型、attention、KV/serving、M5/NAX或性能默认资格。

## 1. 固定源码与真正执行的入口

MLC固定`8978ea9aec626d64096aaa92fbc89bda2e3ea802`，实际TVM fork固定`201fa0001c81431ccce05419b442c47779ea917c`；FFI/构建沿用前轮固定产物，逐进程复核加载库SHA。manifest有64个MLC导入/选读文件、7个本轮TVM文件和commit blob一致性；**导入/hash不是全文审计**。关键深读：

- [group_quantization.py](https://github.com/mlc-ai/mlc-llm/blob/8978ea9aec626d64096aaa92fbc89bda2e3ea802/python/mlc_llm/quantization/group_quantization.py)、[utils.py](https://github.com/mlc-ai/mlc-llm/blob/8978ea9aec626d64096aaa92fbc89bda2e3ea802/python/mlc_llm/quantization/utils.py)：量化、打包/解码、mutator、参数映射、Metal编译及cache。
- [qwen2_model.py](https://github.com/mlc-ai/mlc-llm/blob/8978ea9aec626d64096aaa92fbc89bda2e3ea802/python/mlc_llm/model/qwen2/qwen2_model.py)：实际MLP、post-attention RMSNorm和单shard residual。
- [pipeline.py](https://github.com/mlc-ai/mlc-llm/blob/8978ea9aec626d64096aaa92fbc89bda2e3ea802/python/mlc_llm/compiler_pass/pipeline.py)、[融合pass](https://github.com/mlc-ai/mlc-llm/blob/8978ea9aec626d64096aaa92fbc89bda2e3ea802/python/mlc_llm/compiler_pass/fuse_dequantize_matmul_ewise.py)、[LowBatch特化](https://github.com/mlc-ai/mlc-llm/blob/8978ea9aec626d64096aaa92fbc89bda2e3ea802/python/mlc_llm/compiler_pass/low_batch_specialization.py)：模式约束、分支、后续memory/VM lowering。
- [TVM LowBatchGEMV](https://github.com/mlc-ai/relax/blob/201fa0001c81431ccce05419b442c47779ea917c/python/tvm/s_tir/dlight/gpu/low_batch_gemv.py)：拒绝不支持pattern时，部分修改过的schedule是否真的丢弃。

主机仍为M4 Max、macOS26.6.2/25G83、SDK26.5、clang21。新任务venv为Python3.13.12、NumPy2.5.2、ml_dtypes0.5.3等，仅额外安装`tqdm4.67.1`；未安装Torch/Transformers/服务依赖，未改前轮环境和上游源。

普通`import mlc_llm`因缺`libmlc_llm_module.dylib`失败。官方`SKIP_LOADING_MLCLLM_SO=1`仍导入serve/data，因原生类型`mlc.serve.Data`未注册失败。没有往它搜索到的全局目录安装，也没有伪造该类型。

本轮显式建立root/model/quantization/compiler_pass/interface五个**仅含导入元数据的namespace**，绕过包聚合入口、加载原始功能模块。没有替代算子、假Mixtral/KV类型或native注册，没有构建MLC服务库。它是研究导入适配，不是受支持的MLCEngine启动方式。早期缺`batch_size`编译参数的任务错误也保留；补齐后运行的仍是原始完整pipeline。

## 2. packed format不能只写“int4”

对原始FP16权重`W[N,K]`，Q4 group32每word8码、中心7，量化器逻辑码0..14；Q3 group40每word10码、中心3、逻辑码0..6，uint32最高2bit不使用。decoder能读所有字段值，不能将可解码的15/7误称为该quantizer正常生成的最大码。

以下K65/N97包含尾组，bytes含FP16 scale，不含对象/allocator：

| 格式/布局 | q_weight | q_scale | 总bytes / 原始FP16 |
| --- | --- | --- | --- |
|Q4 KN|12×97|3×97|5238 /12610|
|Q4 NK|97×12|97×3|5238 /12610|
|Q3 NK|97×8|97×2|3492 /12610|

KN量化轴0，NK轴1；转换W时KN实际转置packed/scales，不只交换layout字符串。padding码写0，**非零scale下码0解码为负中心值，不是零权重**；消费者必须用原始逻辑K裁尾，不能将物理存储覆盖长度当K。

`quantize_model`实际替换两个`nn.Linear`，建立`weight→q_weight/q_scale`及原始量化函数映射。参数按`export_tvm`返回顺序传入，不自行排序。量化函数cache key含shape/dtype/device/axis/transpose、不含值；两个seed及codebook/zero/tiny都通过同一shape缓存传新权重。

## 3. 模型层、完整pipeline与独立参考

```text
原始GPU quantize_weight → uint32权重 + FP16组scale
                                     ↓ 原始param mapping
X → 原始RMSNorm → 原始QWen2MLP gate/up → split → SiLU×up → down → +X
                                     ↓
export_tvm → 完整MLC passes → DLight → MSL/C-host dylib → reload → Relax VM
```

FFN小岛复用原始DecoderLayer的post-attention RMSNorm和MLP，单shard下用相同residual表达式组合；**未调用attention，也没有替身attention冒充完整DecoderLayer**。H64/F96及尾部H65/F97，动态`X[1,M,H]`覆盖M1/2/8/9/17；另有Q3/NK的K65/N97原始quantized linear。权重为确定seed合成值，不是Qwen checkpoint。

完整pipeline包含transpose融合、Legalize/FuseTIR、原始dequant-matmul融合、LowBatch及memory/VM lowering。附加sampling函数也生成代码，但本轮未执行这些入口。IR阶段、MSL和dylib保留hash；“生成多少kernel”不等于执行覆盖。

参考用**独立unsigned位解码**读取实际GPU生成的packed/scales，按FP16重建权重后计算消费者，不把原float权重当唯一oracle。FFN用FP64 dense运算，在norm输出、gate/up投影、SiLU、乘积和down输出处显式FP16舍入。这是分阶段参考，不是物理累加器trace或全部表达式的逐位模拟。

预置逐元素门槛：linear误差≤`0.015+0.015×|ref|`；FFN≤`0.02+0.02×|ref|`。另记max abs/RMS/relative L2，不在失败后改用更宽L2线。原权重→量化权重损失、量化权重→消费者误差、模型质量分账。

## 4. 消融失败：host代码索引了Metal句柄

直接关闭`FuseDequantizeMatmulEwise`、保留LowBatch时，展开基线在独立/双模块、普通/尾部配置均原生SIGTRAP。最早的任务手写pass子集也失败；改为**完整原始pipeline仅关闭此pass**仍失败，不能把两者归因合并，也不能拿失败基线计时。

LowBatch外归约schedule先做padding/rfactor/线程绑定；没有dequantize block时，内部helper仅将局部`sch`设None并return，上层`apply`却仍返回部分修改的schedule。生成host.cpp出现直接`half*` padding复制/归约；这些指针实际承载Metal `MTLBuffer`对象句柄，不是CPU可索引的张量地址。

证据链：无dequantize的实际TIR → 未完整GPU调度的pad/reduction → C-host直接half索引 → allocator损坏/trap。对**生成host库**额外开ASan，独立replay报告`heap-use-after-free`、READ2，PC位于`__tvm_ffi_matmul`区间。它不是GPU shader ASan，也不能因原始trap最终落在Metal创建function时就归咎Metal编译器。

LLDB诊断未越过启动等待，已停止自己的目标/debugserver并确认退出；未改系统debug权限。ASan默认symbolizer也曾等待；停止自有诊断后用`symbolize=0`得到终结日志。失败/诊断receipt均保留，没有遗留实验进程。

前轮`using half=__fp16`只修类型拼写，**不能修复host访问GPU句柄**。当前客户端有限扫描生成host，发现half指针索引即拒绝后续执行，另一次新进程验证了拒绝。它不是通用C/Metal验证器或上游修复；没有改LowBatch源码、生产fallback。

## 5. 恢复对照：少两次dispatch不等于通用加速

配对实验在**两臂共同关闭LowBatchGemvSpecialize**，其余完整pipeline相同；只有materialized臂再关闭dequant-matmul融合。比较原始DLight Matmul：一臂GPU融合读取packed/scales，另一臂每次在GPU先展开**同一份packed/scales**再乘，不是预备好的FP16常驻权重基线。

两个seed、同组驻留输入/packed/scales、五个M；每shape/臂先调用验算一次，再跑4个ABBA/BAAB block，全输出读回。上传/D2H在`call+device.sync`窗外；first仍含MSL编译/PSO等，不能叫冷驱动测量。自己GPU任务串行，不宣称全机持续独占。

Q4 FFN融合/展开实际软件dispatch为4/6，Q3 linear为1/2；两臂Matmul的MSL都有SIMD-group MMA。默认完整FFN动态图另生成`M≤2→bucket2`、`M≤8→bucket4`、其余Matmul并跨分支验算；不能将动态M1直接称固定shape GEMV。

多数Q4尾部热态host完成中位数约0.15–0.17ms；NK seed17/M1两臂却约0.55/0.39ms，另有seed/M翻转。Q3微小倍率约0.99–1.04，不是稳定引擎排名。完整样本在JSON；FFN因下节stress失败，qualified speed ratio均null，**不删除失败再发布总加速资格**。

pipeline的`memory_usage[forward]`对尾部FFN约22784/46150 bytes，是给定M上界、含复用的IR规划估算，不是进程峰值、驱动驻留、DRAM流量或硬件计数；不能只加两份FP16权重bytes当峰值差。

## 6. 冻结结果与六个数值失败

| cohort | 常规完整输出 | edge完整输出 | 原定容差失败 |
| --- | --- | --- | --- |
|完整默认FFN H64/F96，KN|10|3|codebook 1|
|完整默认FFN H65/F97，KN|10|3|codebook 1|
|共同禁LowBatch的尾部FFN配对，KN|180|6|codebook 2|
|同一配对，NK|180|6|codebook 2|
|共同禁LowBatch的Q3 linear配对，NK|180|6|0|
|合计|560|24|6|

584个完整输出：560个常规（含480个配对热样本）全通过；24个edge有18通过、6失败。常规FFN最大误差约0.011，Q3 linear约0.00467。**578通过不是全部通过**；四个FFN进程因stress以非零码退出，Q3通过。独立CPU归档回放重新解码、重建全参考、重算584个门禁，六个失败仍失败。

codebook用可精确表示的scale/码，每组含最大幅值；五个cohort的GPU words/scales/重建权重与独立编码逐位一致。但送入完整FFN后，H64最大误差约0.113、H65约0.050，均有元素超原线；尾部fusion/materialized该输出甚至逐位相同。**精确codec parity不保证完整层通过，去掉dequant融合也未消除此误差**。source有FP16运算，但未做中间tap将全部误差归因某个物理阶段；保留本预算失败，不宣称普遍上游算术bug。

316个额外控制含280个stale-seed拒绝、32个真实改packed位lane/scale×2的消费者拒绝、4个Q3最高两bit扰动不变性正控，不加进584。wrong-lane/scale保留同一物理shape，排除只靠shape检查的假安全。

## 7. 零组与tiny：有限不等于无精度损失

实测全零组scale0、逻辑码0；最小FP16次正规数`2^-24`的组scale也下溢0，逻辑码却为Q4的14/Q3的6。两者重建有限零，经过真实linear/FFN消费者，本fixture没有NaN输出。

不能只见除零就断言零组消费者有bug，也不能因输出有限说tiny被精确保留：原值已丢失，码的观察受当前MSL/fastMath/设备语义影响。其他dtype/组幅值/SDK/消费者需重测。原始上游quantization测试的NumPy参考含`max_abs≥1e-4`保护，源码量化器无此下界；该CPU/Torch/LLVM测试集合本轮**未运行**，不是Metal zero/tiny的逐位gold。

## 8. 复现与后续

CPU学习：labs/mlc_quant_contracts.py，独立layout/裁尾/位码/scale消费者；离线测试读冻结证据，不导入TVM/MLC或启动GPU。全raw CPU回放由归档器可选NumPy模式执行；portable指标/hash不是再次执行设备。

可选客户端：mlc_quantized_model_probe.py。沿用前轮固定TVM构建、新任务venv，fresh目录及owned supervisor；换机器需重建、重验身份：

正式raw前缀为`outputs/metal-mlc-quant-20261017-final-{full64,fulltail,pair-kn,pair-nk,q3}`，各有`-launch`receipt；其他smoke/失败/ASan/host拒绝单列history。上游源保持clean，无模型下载、缓存清理、产品依赖/默认/系统设置变更。

后续：缩减LowBatch不支持pattern的schedule拒绝合同；真实中间tap检验FFN数值预算；再考虑LLVM-host、原生runtime-only VM、真实权重/attention/请求。TVM/FFI/MLC源或库、host适配、dtype/packing/layout、M分支、幅度、compiler/SDK/OS/SoC变化均触发复查。**本轮不关闭更广GPU研究，也不继承前轮TIR加载器的native-only资格。**

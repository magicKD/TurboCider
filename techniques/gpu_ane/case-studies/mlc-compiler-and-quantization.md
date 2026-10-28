# MLC-LLM：图融合、DLight Metal lowering 与量化语义

<!-- knowledge
id: technique-gpu-ane-case-studies-mlc-compiler-and-quantization
type: case-study
stage: ffn, prefill, decode
operator: gemm, gemv, quantization
backend: metal, mlx
device: cpu, gpu
symptom: measurement, precision-loss, launch-bound
semantics: contract-only
boundary: producer-consumer
api: public, reference-only
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

> 实现案例：按机制学习，不按研究批次采用。文中的次数、环境、命令和结果是固定历史实验；旧采集脚本依赖当时的输出布局，不是当前一键复现入口。可运行的离线学习见[方法实验](../examples/README.md)，原始来源和数值证据保留在文内链接。

[专题导航](../README.md) · [GEMM/量化](../gpu-gemm-quantization.md) · [第三轮证据](../../MAINTENANCE.md)

固定MLC [`8978ea9aec626d64096aaa92fbc89bda2e3ea802`](https://github.com/mlc-ai/mlc-llm/tree/8978ea9aec626d64096aaa92fbc89bda2e3ea802)，并读取其gitlink指定的TVM/Relax [`201fa0001c81431ccce05419b442c47779ea917c`](https://github.com/mlc-ai/relax/tree/201fa0001c81431ccce05419b442c47779ea917c)。不是用TVM最新主线猜测MLC行为；未安装/运行编译器或模型。

## 1. 编译型引擎的优化跨越多个层次

```text
模型/量化表达
  → Relax图改写、layout/epilogue处理
  → Legalize/FuseOps/FuseTIR
  → dequantize+matmul、dequantize+take等TIR融合
  → low-batch specialization + DLight schedule
  → VM、静态内存计划、后端代码与模型库
```

pipeline显式包含每阶段debug IR dump入口。遇到性能问题时，应核查想要的pattern是否匹配、哪些函数已scheduled、最终intrinsic和内存计划；不能只看模型导出成功。

`FuseDequantizeMatmulEwise`匹配特定call_tir结构、辅助tensor数量、函数名提示和静态shape表达，再执行FuseTIR。它是一组具体规则，不是“所有量化图都自动最优融合”的保证。避免物化完整dequantized matrix只是候选目的，最终要检查生成IR/代码。

## 2. 小batch不是“大GEMM缩小M”

`LowBatchGemvSpecialize`尝试bucket2/4的low-batch GEMV schedule，按运行时符号维度≤2、≤8选择不同body，较大值走matmul schedule；只有满足结构条件才改写。它不是给所有函数建立独立LLM实时planner。

这是把常见动态shape的有限分支编进产物：既避免为每个小M逐个重新编译，又不强迫decode用大tile。实验应扫bucket边缘、真正的dynamic shape、pack group与尾部，不能只测M1和一个大M。

## 3. 真正读到的 Metal kernel 路线

高层MLC使用`dl.gpu.Matmul/GEMV/Reduction/Fallback`。在固定TVM的Matmul规则中，Metal尝试`MetalMatmul`；所读实现采用8×8×8 SIMD-group matrix intrinsic，shared staging、16/64/32方向的层级padding和128线程布局。

TIR intrinsic再对应Metal simdgroup load/store/multiply-accumulate。这个固定default路径不是MPP cooperative TensorOps；升级SDK不会自动把已有schedule改成M5 MXU。该观察限定于所读pass，不声称整个TVM没有其他实验路线。

Metal tensorization失败时，所读规则捕获异常后可继续generic schedule；因此编译成功不能证明走到SIMD-group快路径，更不能证明NAX。保留fallback有价值，但性能诊断要看实际生成物，而不是仅看API层叫Matmul。

MPP官方对dense GEMM的direct-device建议也不能反向证明该旧shared-staging路径无效；权重解码、芯片、shape和编译器不同，必须在相同source与consumer上对照。

## 4. Group quantization 的具体数学

`GroupQuantize`区分storage/model/quant dtype、group size及KN/NK布局。uint32可装`floor(32/qbits)`个code；量化轴由weight layout决定，不是固定沿某一个逻辑维度。

所读int4表达可概括为：

```text
q_max = 2^(bits-1)-1 = 7
scale[group] = max(abs(weight[group])) / q_max
encoder code ≈ clip(round(weight/scale + q_max), 0, 2*q_max)
decoder weight = (unpack(code) - q_max) * scale[group]
```

因此常规encoder产生0–14，偏移是7，不是GGUF某些Q4格式的8，也不是MLX的任意affine scale/bias。bit pattern 15在这个decoder里仍有明确数值8×scale，但并非此encoder通常产生的码。不能把“都叫Q4”当ABI兼容。

group tail、pack axis、scale dtype、输出transpose及shard对齐都需绑定原source。全零group的scale可能为0，所读表达未显式加epsilon；量化舍入/NaN转整数在具体编译栈中的行为应另验，不据一段公式宣布全平台错误或稳定。

默认Metal低比特路径可能先解码为浮点再做SIMD-group运算；模型文件4bit不等于矩阵MAC使用4bit整数。融合减少解码materialization与字节流量，仍可能受metadata、寄存器、tile复用和框架开销限制。

## 5. KV cache 后端与部署不能混用宣传

MLC的KV creation pass先生成TIR-based paged cache候选；FlashInfer分支显式要求CUDA及其支持的dtype/RoPE条件。Metal不会因项目有FlashInfer选项就获得CUDA kernel，仍应追自己的TIR/Metal路线。

分页参数、sliding-window、head维度、layer partition和cache dtype进入编译/运行接口。主机库、model library与参数资产版本也必须匹配；只固定MLC而漏掉TVM gitlink，可能连IR API和调度规则都不相同。

这是一条GPU编译/执行路线，不是ANE后端。迁移到GPU+ANE时可借鉴图分区、静态内存、有限shape specialization和精确量化语义，但需要另建跨框架ownership/完成边界，不能把Relax图自动等同于CoreML图。

## 6. 最有价值的对照

同一encoded权重比较：独立decoder、完整dequantize+GEMM、fused TIR、GPU原生强核。记录生成IR、实际kernel、cold compile/load、warm、额外workspace与完整FFN。再看模型prefill/decode及请求，而不以导出成功或一个TIR函数吞吐排名。

本轮的CPU见证只验证packed解释、bucket和代数/状态，不运行TVM schedule，不证明生成Metal代码的数值或速度。具体源文件/行范围在[第三轮清单](../sources.md#compiler-packets)，未读模块不借此标为已审计。

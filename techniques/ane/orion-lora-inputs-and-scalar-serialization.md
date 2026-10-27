# Orion LoRA-as-input：原frontend、热切换与标量序列化边界

<!-- knowledge
id: technique-ane-orion-lora-inputs-and-scalar-serialization
type: method
stage: ffn, prefill, decode
operator: conv, gemm, layout, scheduling
backend: coreml, cpu-reference
device: cpu, ane
symptom: precision-loss, stale-cache, wrong-layout, measurement
semantics: numerically-equivalent, approximate, contract-only
representation: fp16, fp32, packed
boundary: weight-runtime, producer-consumer
api: public, private, reference-only
-->

[ANE目录](README.md) · [42文件来源](sources.md#orion-sources-20261017) · 全量数组/原IR/逐次审计 · [验证记录（过程原件已移出）](../MAINTENANCE.md) · [原优化pass及profiler](orion-rewrites-and-profiler-metric-provenance.md)

2026-10-17批次，固定[Orion `4dcc1c2a`](https://github.com/mechramc/Orion/tree/4dcc1c2a626c9ee1f35576bd0b290d489db17d08)，M4 Max / macOS26.6.2 build25G83、SDK26.5、既有CoreML8.3.0/NumPy2.0.2。新sparse clone只用于阅读；不构建完整CLI、下载checkpoint或运行原私有ANE runtime。两seed进程共148次公开预测，104份原目标通过、44份有意/缩减负例保留；其中LoRA124份、cast消费者24份，分开分类，不是148份都过原语义。

本页重点是原LoRA frontend如何把参数变化放到输入边界，以及**源码IR中的FP16常量、发射出的十进制文本、实际消费者**为什么仍要分别核对。私有delta reload/patch的文件和状态操作未执行，不能把本批结果算成原训练速度，背景见[delta与LoRA概览](training-delta-and-lora.md)。

## 可复用的方法：将 LoRA 做成明确的运行时输入合同

1. 固定 base 权重、A/B 轴方向、rank 和 scale；使 adapter 切换成为输入变化，而不是隐含更改图常量。
2. 同一 owner 下测试多代 adapter、错误 shape 和真实下游，检查旧结果是否受复用影响。
3. 同时核对数学目标、发射的 scalar 文本和实际 dtype 舍入；区分原 MIL 与自行补充的执行载体。

适用于探索不重编译的 adapter 切换；下文公开 sidecar 不代替原私有 runtime 或完整训练验收。

## 1. 明确四个轴，才有可热切换的adapter ABI

[原`lora.c`](https://github.com/mechramc/Orion/blob/4dcc1c2a626c9ee1f35576bd0b290d489db17d08/compiler/frontends/lora.c)创建三个输入和一个常量base Conv。以本例K=64、N=48、S=17、rank=3/8为例：

```text
x       [1,K,1,S]    → X_rows [S,K]
lora_A  [1,K,1,R]    → A      [K,R]
lora_B  [1,N,1,R]    → B      [N,R]
W_base  [N,K,1,1]    → W      [N,K]

Y_rows = X_rows Wᵀ + scale · (X_rows A) Bᵀ
scale  = alpha / rank
Y      [1,N,1,S]
```

这里B存成`[N,R]`，不能直接按其他库常见的`[R,N]`喂字节。源码把x/A/B经`[0,2,3,1]`置成matmul布局，第一段对A设transpose_y=true，第二段消费已经转好的`[R,N]`，最后`[0,3,1,2]`恢复BC1S。base权重按`[out,in]`做Conv1×1，无bias。

优化目的：base保持常量图，不因换adapter反复传全矩阵；A/B作为runtime input，在固定rank/shape下复用同一图。它不减少LoRA两次matmul和数据传输，也不自动完成LoRA训练。源码“transpose免费”的注释不是本机copy、带宽或完整消费者成本证据，本批没有采这些物理指标。

用非方阵64→48、ragged S17和R3控制，可避免在768→768、R16这种对称示例里掩盖轴/packing错误。rank改变属于不同artifact，本批R3/R8是两个模型，不是假装同一个compiled图支持任意rank。

## 2. 实际执行层次：原MIL不改，公开sidecar由本库补齐

原封未动地编译graph/builder、frontend、优化pass、topo/validator及ObjC codegen，native client只链接Foundation；另独立编译原profiler。**没有链接或调用`core/ane_runtime.m`、reload/patch、IOSurface或原test/demo**。编译单元、header、license与CLI阅读范围见42文件账本，不是42个文件都运行。

首次source packet漏掉pipeline include的`pass_conv_bias.h`，在本库编译阶段失败，补入原header后通过。随后尝试把裸原MIL目录作为public模型加载，18次均因缺`coremldata.bin`在预测前拒绝。这是本库接口假设不完整，不是原私有路径故障。

最终在新owned目录补入**本库生成的ModelDescription sidecar**，保持原MIL和BLOBFILE内容不改。sidecar的83-byte外层preamble来自固定[vpipe观察格式](https://github.com/tgo-app-dev/vpipe/blob/c2989a76841888e4d8d208fad4f476c487710962/apple-silicon/coreml/ane-emitter.cc#L95)，内部是CoreML protobuf特征描述；它不是公开稳定的`.mlmodelc`二进制ABI或产品部署建议。独立decoder核对所有实际名称、dtype和shape，不能只改扩展名当“已经compile”。

原native emitter另在无sidecar/无加载的新目录重跑，14份before/after MIL逐字节等于主试源文件，确认公开包装没有悄悄重写图。base blob的64-byte file header、64-byte chunk metadata、offset64指向metadata和完整FP16 payload也由标准库重新核对。

codegen的buildInfo写死`coremltools-version=9.0`，**本机调用者实际仍是8.3.0**。这个banner不是编译器/wheel版本证明；来源记录绑定实际Python loader body和环境，不据该字符串宣称装了9.0。内部系统编译器与loaded image亦未独立attest。

## 3. 一次owner下换A/B：正确路径和错误caller分别验收

每seed、每个R3/R8模型、每个CPU_ONLY/CPU_AND_NE owner，做两代不同X；每代序列为：

```text
base → adapter-A → adapter-B → base → adapter-A
     → 故意只换B、留下旧A → 故意按错A的物理顺序喂入
```

每次输出全部48×17元素保存。普通五次控制共80份均通过原目标；重回base和A时完整字节分别复现，不仅检查“换adapter之后与旧结果不一样”。相同shape但A未更新或packing错误的32份输出仍finite，**符合实际错误feed的函数，却不符合本代请求的完整A/B目标**；这是明确注入的caller/staging负控，不是发现CoreML自己会随机stale或race。

每个owner显式load一次，后续仅调用predict；验证MIL、base blob和sidecar源文件不变。这证明当前调用协议无需显式重发图/重载来换这些固定shape输入，**不是硬件/内部compiler计数为0的测量**。内部编译缓存、异步换adapter原子性、多线程、取消/drain和私有session都没有覆盖。

public Python bridge可能把输入dict的FP16 ndarray改成FP32容器；每次完整检查logical值仍相等，输出也必须无损还原到声明dtype才收录。不把host容器dtype、端口dtype、内部算术或物理ANE位宽混为一谈。

原私有[test_lora](https://github.com/mechramc/Orion/blob/4dcc1c2a626c9ee1f35576bd0b290d489db17d08/tests/test_lora.m)另依赖IOSurface统一allocation、按声明shape在flat前缀紧密packing，以及按参数名排序的surface数组。本批用**命名的公开feature**，没有验证这些private ABI经验；源测试的“非零/改变输出”控制也不能替代这里独立完整数学目标。

## 4. 一个ULP来自文本发射，而不只是FP16本身

`orion_gb_const_scalar`保存float32的alpha/rank，IR节点dtype标为FP16；[原codegen](https://github.com/mechramc/Orion/blob/4dcc1c2a626c9ee1f35576bd0b290d489db17d08/compiler/codegen.m#L47)用`%g`写十进制，仅默认6个有效数字。rank3/alpha8的普通图仍通过本批门槛，但临界控制显示两次舍入不是总等价：

| 层次 | rank1、alpha=`1.0004879f` |
| --- | ---: |
| 原float32 scalar | 1.0004879236221313 |
| 按IR dtype直接round成FP16 | 1.0 |
| 原codegen实际文本 | `fp16(1.00049)` |
| 文本解析后的FP16 | 1.0009765625 |

控制中W=0，只有一个X/A特征为1，B全64。原typed-IR目标为64，配置float32实数目标约64.031227，真实公开输出却为**64.0625**，与实际序列化scalar的参考完全相符。两seed×两policy共4份active输出对typed目标差.0625、对配置目标差约.03127，均超过预先固定maxabs .02；两次zero/reset控制则通过。

本批采用`maxabs≤.02且relativeL2≤.01`的完整输出gate，不以“它很接近”覆盖拒绝，也不把相对实际错误scalar的低执行误差拿来通过原目标。建议artifact验收直接比较**序列化后typed scalar的值/bit**，发射使用足够round-trip精度或明确编码；不是仅对代码中的float变量做单元测试。本批未修改上游格式器/默认，也未核查其他版本是否修复。

## 5. 完整数值、适用域与复现

148份公开输出均finite，含124份LoRA和24份[cast消费者](orion-rewrites-and-profiler-metric-provenance.md)，共逐元素检查102,720个Y值；重复输出按dtype/shape/完整hash去重存储，但每次调用记录、输入代际和资格仍保留。104份满足原目标，32份caller负例、4份scalar序列化负例、8份cast重写负例分别拒绝；**全部148份均符合实际feed/实际序列化图的执行参考**。source-execution最大abs约.002561，不能转成原目标全部通过。

每份LoRA同时用独立scalar FP64闭式公式和完整原IR interpreter计算；重建全部W/X/A/B配方，并分别用配置scale、typed-IR scale、实际文字scale分析。不是用CoreML输出或只算几个值作oracle。两种独立参考的完整结果一致到1e-12。

这些小图的已知plan preferred均为CPU，unknown仍保持unknown；CPU_AND_NE只是许可。没有speedup、GPU基线、能耗、物理ANE、实际adapter checkpoint质量或训练资格。原attention frontend只读未运行；本批无base bias、residual、非线性join或量化base，不把一个线性正例推广为完整LLM/DiT。

本地实现：原生emit client、显式collector、、、离线测试。默认CPU入口不导入框架、上游或raw数据：

实际collector需macOS、已有固定依赖、`--reference ../references/Orion-lora-research-20261017 --seed 17 --output-dir NEW`，另加`--device`才公开加载/预测；用supervisor和全新owned目录、2GiB reserve。不会安装依赖、执行上游setup/test、清cache、改变系统策略或修改private目录。源码/MIL/模型/binary/原日志留ignored outputs，公开capsule为原创合成数据与身份。

重开：版本/SoC/OS/SDK/compiler、alpha/rank/scalar格式、dtype/shape/packing、原base/adapter/bias/量化/consumer、更新原子性/队列或完整质量预算变化。继续问题是实际支持的checkpoint、完整消费者与优化GPU分母，而不是从这批CPU首选小图推断通用ANE收益。

# Candle Metal packed matmul：contiguous、copy与非零起始偏移

<!-- knowledge
id: technique-metal-case-studies-candle-packed-views-and-copy
type: case-study
stage: ffn, prefill, decode
operator: gemm, gemv, quantization, layout
backend: metal, cpu-reference
device: cpu, gpu
symptom: wrong-layout, lifetime, precision-loss, measurement, compile-bound
semantics: numerically-equivalent, approximate, contract-only
representation: q4, fp16, fp32, packed
boundary: weight-runtime, producer-consumer, cpu-gpu
api: public, reference-only
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

> 固定版本实现案例，供理解机制与复核边界；学习先读[方法导航](../README.md)，数据保留规则见[证据说明](../../MAINTENANCE.md)。

[GPU导航](../README.md) · [前轮源码导读](../cache-and-quantized-runtime-designs.md) · [完整Qwen2/提交/capture](candle-qwen2-queues-and-capture.md) · [来源](../sources.md#candle-sources-20261017) · [验证（过程原件已移出）](../../MAINTENANCE.md)

2026-10-17，第十四轮。实际构建并执行固定Candle原始Rust/Metal实现。**contiguous view不等于零起始偏移，当前Metal `copy()`也没有提供独立buffer。** 本cohort的packed GEMM丢失非零view origin，240个完整linear输出失败；普通contiguous/copy未修复，force-contiguous/owning/dense对照通过。不是任意版本的永久bug结论，未修改上游或生产默认。

## 1. 真正构建了什么

固定`huggingface/candle@12bb5d1f6350f4ea9827d9086f213feec1f0af6a`，workspace版本0.11.0，不冒充已安装正式release。原始core/nn/transformers/metal-kernels启用Metal、默认不开debug labels；27份选读/调用相关文件逐commit blob核对，文件身份不等于全部源码审计。

环境原无Rust；按官方分发manifest取证后，将Rust1.99.1、rustup1.29.1、Cargo/cache/target都放入ignored任务目录，使用`--no-modify-path`及私有`CARGO_HOME/RUSTUP_HOME`。原始仓库没有本轮所需的Cargo.lock，因此另冻结了客户端lock、161个resolved package身份和源码/二进制SHA；后续构建用`--locked`，不能只固定Candle commit却任凭依赖更新。

一次尝试从旧稀疏promisor引用本地clone失败，日志的corruption措辞不证明原仓损坏；改在任务目录从上游固定SHA取完整所需子树，旧参考工作树不改。初次依赖/Release构建约1分钟，后续客户端增量构建约1秒，均不是推理时间。未全局安装Rust或改变shell配置。

## 2. Q4_0与MLC/MLX Q4不是同一个载体

[原始Q4定义/codec](https://github.com/huggingface/candle/blob/12bb5d1f6350f4ea9827d9086f213feec1f0af6a/candle-core/src/quantized/k_quants.rs)每32个值18 bytes：有符号FP16 scale与16个byte。每byte的低nibble属于前16个值、高nibble属于后16个值，中心是8；不是相邻两值，也不是MLC中心7的uint32位槽。

本轮调用原`QTensor::quantize_onto`，从CPU FP32权重执行原CPU quantizer，再建立Metal compressed storage。独立标准库decoder读取实际packed bytes；四个native进程的public-dequantized FP32权重均逐位匹配。编码误差、GPU计算误差、模型质量分账，不用原float权重替代encoded-weight gold。

原[QMetalStorage](https://github.com/huggingface/candle/blob/12bb5d1f6350f4ea9827d9086f213feec1f0af6a/candle-core/src/quantized/metal.rs)的公开dequantize仍是blit/readback、CPU block decode、建立FP32 Metal storage；它不是GPU tile materializer。热packed forward不调用此API。resident dense对照只在准备时展开一次，不能将它与“每次CPU展开”的成本混淆。

## 3. FP32接口并不保证所有中间权重都为FP32

原始[dispatch](https://github.com/huggingface/candle/blob/12bb5d1f6350f4ea9827d9086f213feec1f0af6a/candle-metal-kernels/src/kernels/quantized.rs)在M1走GEMV，matrix走32×64输出覆盖、128 threads、8192-byte TG workspace。[shader模板](https://github.com/huggingface/candle/blob/12bb5d1f6350f4ea9827d9086f213feec1f0af6a/candle-metal-kernels/src/metal_src/quantized.metal)的Q4 matrix specialization使用half权重fragment、float activation/accumulator对象，输出float；不能从`_f32`函数名推断无half重建，更不能把源码类型写成物理ISA证据。

本轮K64/N65与K96/N97覆盖N尾部；K仍必须满足32元素block约束，没有任意K尾部资格。activation/output均FP32；不将这组数字与MLX常见FP16 affine模型直接排名。

## 4. 原始view origin在哪里丢失

原fwd检查`layout.is_contiguous()`，但matrix分支另建`Layout::contiguous(src_shape)`，将这个新layout的`start_offset`传给kernel，结果恒0；M1的fwd_mv却使用原`layout.start_offset()`。这不是“所有stride都不支持”，而是连续布局与非零origin两个条件被混同。

输入从`base[M+2,K].narrow(0,2,M)`取得，逻辑shape与stride均正确，起始偏移为`2K`个元素。所有控制保持相同逻辑X/packed权重：

| 输入/路径 | 实际起始偏移 | 与原view同buffer | M1 | M>1 |
| --- | --- | --- | --- | --- |
|packed原view|2K|是|通过|失败|
|packed `.contiguous()`|2K|是|通过|失败|
|packed `.copy()`|2K|是|通过|失败|
|packed `.force_contiguous()`|0|否|通过|通过|
|packed重新拥有logical X|0|否|通过|通过|
|resident dense读取原view|2K|是|通过|通过|

[`Tensor::contiguous`](https://github.com/huggingface/candle/blob/12bb5d1f6350f4ea9827d9086f213feec1f0af6a/candle-core/src/tensor.rs)对已经连续的view返回clone；force-contiguous才复制logical values并重置layout。[MetalStorage::try_clone](https://github.com/huggingface/candle/blob/12bb5d1f6350f4ea9827d9086f213feec1f0af6a/candle-core/src/metal_backend/mod.rs)在这个版本直接clone storage，所以`Tensor::copy`虽然建立新Tensor/锁对象，底层Buffer仍相同。客户端比较真实Buffer引用身份，而非只看Tensor id；没有发布原始地址。

240个失败全部通过**错误的base零偏移prefix**参考，而不通过预先单独保存的logical X参考；GPU input-view readback也与logical X逐字节相同。因此错误不是CPU oracle也读错了view、旧权重缓存或未写完尾部。M1、force、owning、dense与两个seed/提交阈值形成反证链。

这是固定路径的实际失败，不等于所有Candle模型都会触发。上游producer若已经物化为零origin，会掩盖此条件；本轮[原始Qwen2图](candle-qwen2-queues-and-capture.md)就是独立消费者验证，不能由一个linear失败宣布整个引擎不可用。

## 5. 全量数值与有效性能窗口

两个seed×提交阈值1/50，共4个linear进程。每个304个完整输出：144个view控制、32个warmup、128个ABBA/BAAB hot。共1216个，其中976通过、240失败；失败不丢弃，也不发布错误路径的加速比。标准门槛预置为`0.005+0.005×|FP64 encoded-weight ref|`，实际正确arm误差约1e-4量级，不在失败后放宽。

512个hot样本只比较零origin owning X下的原QMatMul packed与同encoded权重预展开的原QMatMul dense。每block切换驻留输入/权重variant，上传、公开dequantize、读回和文件写入在`forward+device.synchronize`计时窗外；所有结果独立完整验算。

K64/N65的M1约packed129–134µs、dense119–123µs；M33两者约129–140µs，M64约559–602µs/571–585µs。其他shape/seed有翻转，不能把小差距当稳定矩阵吞吐结论。Q4逻辑权重18/32 bytes/值，对照FP32为4 bytes/值，但没有测物理DRAM/RSS/能耗。小矩阵权重更小，不保证完成延迟更低。

## 6. 复现与迁移

CPU学习：labs/candle_contracts.py，case `candle-packed-view-contracts`；离线测试不导入Candle/启动Rust或GPU。可选驱动、原生客户端、独立归档器与lock分离。

正式raw为`outputs/metal-candle-final2-20261017-linear-*`，先前smoke/第一批未补capture primer的数据不混入最终计数。force-contiguous是本边界的明确GPU复制正控，不是免费修复；若接入真实消费者必须计入复制/内存成本。源码、库/lock/Rust/SDK/OS/SoC、dtype/format、shape/origin/stride/owner、producer或queue行为变化均复查；未改产品fallback/默认，没有最新版本bug/fix状态声明。

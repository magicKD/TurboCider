# TVM/Metal 真实编译与 AOT：原始 DLight、纯 runtime 消费者和计数边界

<!-- knowledge
id: technique-metal-case-studies-tvm-metal-aot-and-runtime-only
type: case-study
stage: ffn, pipeline
operator: gemm, gemv, layout, scheduling
backend: metal, cpu-reference
device: cpu, gpu
symptom: compile-bound, launch-bound, measurement, wrong-layout, precision-loss, lifetime
semantics: numerically-equivalent, contract-only
representation: fp16, fp32
boundary: producer-consumer, cpu-gpu
api: public, private, reference-only
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

> 固定版本实现案例，供理解机制与复核边界；学习先读[方法导航](../README.md)，数据保留规则见[证据说明](../../MAINTENANCE.md)。

[GPU导航](../README.md) · [前轮MLC/TVM源码导读](../mlc-tvm-metal-compiler.md) · [固定来源](../sources.md#tvm-aot-sources-20261016) · [验证快照（过程原件已移出）](../../MAINTENANCE.md)

实验2026-10-16，复核2026-10-17，第十一轮。已实际完成固定TVM fork的源码构建、原始DLight调度、TIR→MSL→C-host dylib导出、加载及原生消费者执行。**纯runtime消费者不加载Python或libtvm_compiler，仍需在首用时编译打包的MSL并创建pipeline**。这不是完整MLC模型/Relax VM验收，也不是新跑了M5/NAX。

## 1. 固定的不只是TVM版本字符串

使用MLC-LLM固定版本的实际TVM gitlink：`mlc-ai/relax@201fa0001c81431ccce05419b442c47779ea917c`；其 `apache/tvm-ffi@df463f9bd269867fc7f4d578d62d80cc2a11d004` 及dlpack/libbacktrace gitlink也固定。33个选读文件、构建库与客户端SHA见来源JSON。Python源版本标签分别回退为`0.28.dev0`/`0.0.0.dev0`，**标签不是已安装官方release或ABI身份**。

在独立任务目录构建原始C++，Release、Metal ON、LLVM/Z3/GTest/RPC OFF；C++构建约293秒，含准备步骤的owned launch约300秒。保留源码只读，不升级产品环境。Python3.13.12及固定NumPy/Cython/pytest等放在任务venv；无editable install。Cython从固定FFI源码编译，导入目录链接同一份已构建FFI。逐进程通过dyld镜像枚举/hash排除重复或外来TVM/FFI库。

TVM根工作流禁止editable，FFI子仓推荐editable，两者在本任务中用独立导入目录协调；Blackwell全量TIRX测试工作流不适用这台Mac，没有假称通过。不是安装任意TVM wheel后把版本名贴到MLC gitlink上。

主机SoC为Apple M4 Max，macOS26.6.2/25G83、SDK26.5、clang21。这个fork的Metal `GetAttr(kDeviceName)`直接返回，所以engine device-name属性实际为null；证据没有伪造名称，也没有把null当GPU不可用。Metal设备exist、实际Metal模块/dispatch与完整输出另行验证。target的32KiB shared-memory配置不是本轮测出的硬件上限。

## 2. 真正执行的消费者和原始选路

```text
A[M,K], B[N,K] → C=A×Bᵀ → ReLU(C) × gain + R[M,N] → D
                                      ↑ trailing float32 scalar
```

两种seed17/29；四个shape：FP16的64×64×64、尾部17×65×33；FP32的33×47×65和M1/N129/K127。输入、权重、residual与gain形成两个独立状态A/B，输入都在计时前上传并完成同步；热测交替绑定这些驻留buffer，不把CPU上传算进forward timer，也不称为零copy。

本轮手动调用固定源码的GEMV/Matmul规则，不冒充完整MLC compiler_pass pipeline。这里的`default`是该fixture的原始规则选择；`generic`仅在任务进程里使 `MetalMatmul.apply`抛出一个诊断异常，让原始 `Matmul.apply` 的通用fallback继续执行。没有改upstream文件、MSL或数学门禁。M=1两臂同样命中原始GEMV，FP32矩阵两臂最终MSL也相同，作为空臂。

[`matmul.py`](https://github.com/mlc-ai/relax/blob/201fa0001c81431ccce05419b442c47779ea917c/python/tvm/s_tir/dlight/gpu/matmul.py) 的Metal规则用16×64输出覆盖、BK32、8×8 fragment、128线程与shared staging；尾部通过pad/reindex、加载/写回保护处理。原始FP16规则确实完成tensorize，最终MSL含 `simdgroup_multiply_accumulate`；不是因函数名里有Metal就宣布优化命中。FP32尝试该f16 intrinsic时产生dtype不匹配的ScheduleError，随后原始逻辑回退，完整FP32结果通过；这不是“硬件不支持FP32”的证明。

这里的A/B/C intrinsic类型均是float16，不能从其他引擎的习惯写成FP32 accumulator保证。实际计算顺序与舍入由生成程序决定，验收使用原始完整参数的独立FP64参考及容差，不要求跨schedule逐位相同。

### 一次真实的“编译成功但优化没命中”

早期探针禁用了TVM后端自动注册，只加载Metal。该fork的tensor-intrin模块导入又间接需要 `ptx_legacy` namespace，导致AttributeError被Matmul的fallback捕获，编译仍成功、却没有SIMD-group矩阵代码。恢复fork默认注册后命中。本轮保留这个**任务配置错误**，不归因为Metal能力或上游数值bug。优化验证必须同时检查调度trace、异常路径和最终MSL。

## 3. C-host导出不是无条件开箱可用

LLVM关闭时，原始build选择C-host，并以本机clang打包。FP32小kernel可直接导出；FP16的生成host wrapper包含`half*`，默认C++编译命令没有定义half，实际导出失败。没有删除这个失败，也没有改成只测FP32来掩盖。

本轮显式加入任务兼容头：`using half = __fp16`并静态检查sizeof=2；**只解决host wrapper的指针类型拼写**，不改tensor dtype metadata、GPU MSL或算术。原始源码与dylib构建不变。正式FP16/AOT结果以该适配为前提，不代表stock LLVM-host路径、所有CPU half算术或所有平台已验证。

这个限制也在未改的上游Metal测试中复现：

| 测试环境 | 实际结果 | 没有被改写的边界 |
| --- | --- | --- |
| C-host默认、排除需离线工具的一项 |23通过、2失败|inf/nan与erf的FP16host导出缺half定义|
| 同一组原测试，仅加入host typedef编译include |25通过、1项仍未选中|不修改IR/MSL/测试断言，不等于整个TVM套件|
| 单独运行离线编译的trailing-POD测试 |1失败、25未选中|xcrun缺metal/metallib；callback返回None，表层是None→string TypeError|

原始失败与适配成功都有JUnit/hash和完整日志。后一个失败不是GPU执行失败，也不证明MSL不能运行：本轮使用callback显式返回`(source, "metal")`，实际运行时编译成功。没有安装完整Xcode或擅自下载系统组件。

## 4. 哪部分AOT了，哪部分仍在首用发生

[`BuildMetal`](https://github.com/mlc-ai/relax/blob/201fa0001c81431ccce05419b442c47779ea917c/src/backend/metal/codegen/codegen_metal.cc)将每个kernel的payload与format打包。裸str/bytes callback返回按历史合同声明为metallib；显式`(payload, format)`可声明MSL。**内容看起来像源代码，不会让裸str自动变成metal格式**。多kernel必须格式一致。本轮混合格式、非法格式4个负控均在dispatch前被拒绝，没有把错误payload送入设备编译器。

| 阶段 | 实际完成的事 | 不代表 |
| --- | --- | --- |
| schedule/codegen |原始规则→带padding/epilogue的MSL|kernel已经通过数值或性能资格|
| host export |C wrapper编译为dylib，嵌入真实MSL payload|device binary/metallib已经生成|
| load |反序列化函数metadata和MSL字节|所有PSO已创建|
| first forward |原始runtime调用newLibraryWithSource、创建PSO并执行|测得的是单纯kernel busy或冷驱动编译时间|
| warm forward+sync |复用该module/device/function的pipeline|完整MLC请求、权重准备或迁移都免费|

导出的每份dylib中均检查到了精确生成的完整MSL字节。reload后 `inspect_source()` 却是空字符串：[`metal_module.mm`](https://github.com/mlc-ai/relax/blob/201fa0001c81431ccce05419b442c47779ea917c/src/backend/metal/runtime/metal_module.mm)不序列化debug source map，但仍序列化payload map。**空inspect不等于丢了kernel或需要重新跑TVM编译器**。

独立C++消费者只加载`libtvm_runtime`、`libtvm_ffi`、`libtvm_runtime_metal`，不链接/加载libtvm_compiler，也不依赖Python；从同一导出库执行64个完整输出，全部通过。它验证了真正的runtime-only AOT交接，而不是在同一个Python进程保留原编译对象后假装reload。PSO与MSL首用成本仍在，且没有清驱动cache，不能把所有first都叫“冷编译”。

## 5. 全输出数值、ABI与负控

正式共336个完整输出：256个热态、16个first、64个独立原生reload输出，全部对完整FP64消费者计算。FP16预设relative L2≤0.02且max abs≤0.03；FP32为1e-4/1e-3。本矩阵热样本最大L2约0.00112，未因失败放宽线；这些不是图像/模型质量标准。

32个生成host ABI负控使用相同元素数的错误shape、相同字节数的错误dtype，全部在dispatch前拒绝。256个旧状态输出/新状态参考负控也被拒绝。gain在正/负值间改变，真实验证了一个trailing float32 scalar的eight-byte slot；没有将其扩大为float3、任意strided外部tensor或全DLPack交接资格。

每个case/seed热测8个block，ABBA/BAAB交替并切换A/B状态；原始样本、完整检查、IR/trace/MSL/host库hash保留。warmup不混入正式统计。先通过first全输出，再开始性能采样。

## 6. 观察到的时延，不当成引擎排名

以下是resident输入下`forward+device.sync`中位数µs，均不含随后D2H校验；每格为seed17/29。两臂使用同一框架、同一数学消费者与dtype。

| case |原始规则default|任务forced-generic|
| --- | --- | --- |
|FP16 64³|147.2 /157.4|161.0 /171.9|
|FP16 17×65×33|190.6 /174.4|173.9 /180.7|
|FP32 33×47×65（MSL相同空臂）|194.8 /213.7|205.8 /204.4|
|FP32 M1/N129/K127（同GEMV空臂）|169.9 /176.2|185.0 /178.2|

FP16常规形状约9%差距，小tail在两个seed翻转；连MSL相同的空臂也有相当漂移。不能据此断言SIMD-group在所有shape更快，或把小幅收益自动归因于矩阵吞吐。这里没有GPU busy/counter/IR-spill归因、全局最优schedule证明、跨引擎模型排名或能耗结果。

## 7. 软件计数器为何会少报blit

实际调用原始 `metal.GetProfileCounters`。热forward的dispatch=1；加上D2H后gpu_to_cpu从0变1、sync增加，但blits仍为0。原生消费者4步的计数为dispatch4、cpu_to_gpu12、gpu_to_cpu4、blits12。

这不是零copy。源码 `AllocDataSpace` 默认private：H2D通过shared staging后走pending encoder；D2H通过单独command buffer/blit/wait，但该路径不增加pending-blit字段。**按本轮private-buffer源码路径可推得至少12次上传+4次读回copy encoder，instrumented blits却只记12**。这是源码与软件计数的推论，不是假称trace直接观测了16个事件。

这些字段不是完整CB/encoder清单、PMU、DRAM bytes或GPU时间。`runtime.timer.metal`又以GPU时钟sample包围一个运行跨度，不能仅因“GPU clock”就等同于kernel忙时。本轮正式计时使用host完成边界，没有把该timer当物理kernel计时。

## 8. 复现入口与剩余范围

本地CPU：labs/tvm_aot_contracts.py、离线测试；归档器：curate_tvm_aot.py。标准库XML只解析自己的JUnit，已显式登记；核心不导入TVM或启动GPU。

可选客户端：构建/运行驱动、原生消费者、C-host兼容头、上游测试适配器。完整命令与版本在owned receipts/脚本参数中，阶段入口如下，均使用新输出目录：

最终原始前缀为`outputs/metal-tvm-aot-20261016-final17/final29`，各有`-launch`receipt；早期smoke、注册错误、half导出失败及成功小样均独立归档，不混入336。portable capsule保存完整指标与hash；没有原始数组时离线测试不是再次执行336个设备输出。

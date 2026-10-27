# MLC/Metal 导出图的原生 Relax VM：不带 Python 的执行与失败恢复

<!-- knowledge
id: technique-metal-case-studies-mlc-runtime-only-relax-vm
type: case-study
stage: ffn, pipeline
operator: gemm, gemv, layout, scheduling, quantization
backend: metal, cpu-reference
device: cpu, gpu
symptom: compile-bound, wrong-layout, lifetime, precision-loss, measurement
semantics: bitwise, numerically-equivalent, contract-only
representation: fp16, q4, packed
boundary: weight-runtime, producer-consumer, cpu-gpu
api: public, private, reference-only
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

> 固定版本实现案例，供理解机制与复核边界；学习先读[方法导航](../README.md)，数据保留规则见[证据说明](../../MAINTENANCE.md)。

[GPU导航](../README.md) · [前轮量化FFN](mlc-quantized-ffn-and-lowbatch-boundaries.md) · [本轮精度观察](mlc-ffn-stage-observation-and-precision.md) · [来源](../sources.md#mlc-vm-stages-sources-20261017) · [验证（过程原件已移出）](../../MAINTENANCE.md)

2026-10-17，第十三轮部署部分。**8个独立原生进程完成112个完整输出，全部与原Python VM逐位相同；其中6个仍未通过原数值预算。** 24个VM ABI负控在dispatch前拒绝，8个拒绝后的完整消费者恢复输出也相同。部署复现正确，不等于被复现的数值失败消失。

## 1. 比前轮TIR加载器多验证了什么

[第十一轮](tvm-metal-aot-and-runtime-only.md)原生客户端只调用固定TIR函数；它不能直接证明Relax VM的字节码、shape函数、参数、allocator与异常状态已正确交接。本轮读取**上一轮已导出并运行过的同一批模型dylib**，不重新量化、不重建它们、不改上游runtime。

固定身份仍为MLC `8978ea9aec626d64096aaa92fbc89bda2e3ea802`、TVM fork `201fa0001c81431ccce05419b442c47779ea917c`、FFI `df463f9bd269867fc7f4d578d62d80cc2a11d004`。本机M4 Max、macOS26.6.2/25G83、SDK26.5、clang21；逐进程比较实际加载库SHA，不按Python版本标签猜ABI兼容。

关键源码入口：

- [Python VM初始化](https://github.com/mlc-ai/relax/blob/201fa0001c81431ccce05419b442c47779ea917c/python/tvm/runtime/vm.py)：加载executable，按device/id/allocator三元组初始化，补CPU shape设备。
- [VM executable](https://github.com/mlc-ai/relax/blob/201fa0001c81431ccce05419b442c47779ea917c/src/runtime/vm/executable.cc)与[VM实现](https://github.com/mlc-ai/relax/blob/201fa0001c81431ccce05419b442c47779ea917c/src/runtime/vm/vm.cc)：`vm_load_executable`、`vm_initialization`、函数查找和异常处理。
- [FFI packed Function](https://github.com/apache/tvm-ffi/blob/df463f9bd269867fc7f4d578d62d80cc2a11d004/include/tvm/ffi/function.h)与[动态库加载](https://github.com/apache/tvm-ffi/blob/df463f9bd269867fc7f4d578d62d80cc2a11d004/src/ffi/extra/library_module_dynamic_lib.cc)：引用生命周期、packed参数和局部动态加载。它们是固定版本合同，不是任意TVM/MLC版本永久ABI保证。

## 2. 实际原生调用链

```text
显式加载Metal runtime plugin
  → LoadFromFile(原模型dylib)
  → vm_load_executable()
  → vm_initialization(Metal0 pooled, CPU0 pooled)
  → GetFunction("forward") + 有序Tensor参数
  → 原VM字节码/shape函数/Metal kernel
  → 完成同步 → 全量读回 → 外部CPU检查
```

本fixture初始化实参为`[8,0,2, 1,0,2]`：两个device/id/allocator三元组。CPU被用于shape等host运行工作，**不等于发生了矩阵CPU fallback**；本轮还检查返回Tensor的Metal device、dtype/shape，以及实际Metal dispatch。

新C++客户端只链接`libtvm_runtime`和`libtvm_ffi`，显式`dlopen`同构建的`libtvm_runtime_metal`。每个原生消费者的TVM相关镜像恰为这三份，不包含`libtvm_compiler`/`libpython`；直接Mach-O依赖也已检查。进程当然仍依赖系统C++/Metal/Foundation等库，不是“总共只加载三个库”。Python只在进程外准备/监督/验算，不参与消费者的执行。

这不需要MLC serving库，是因为选定的FFN/linear图没有调用MLC serving对象或KV外部函数；**不能由此推断完整MLCEngine、tokenizer、attention/KV模型都不需要MLC runtime**。

## 3. 参数、设备与owner不能依赖隐式约定

输入用有界的任务manifest描述参数数量、dtype、rank和shape；实际二进制文件逐项核对旧packed/scales/Norm参数的字节身份。该manifest是本实验载体，不是MLC的新通用模型格式。

权重仍是uint32 packed加FP16 scales，KN/NK/Q3保持原物理布局，按原`export_tvm`参数顺序构造`AnyView`数组。所有Tensor由C++容器持有；packed views不拥有参数，不能让bank先于调用销毁。每个variant拥有独立bank，调用后同步完成再读结果；没有测试在GPU未完成时覆写同一buffer。

CPU→GPU上传和GPU→CPU读回都是真实copy。本轮既非零copy交接，也非内存最优的单bank部署。五份variant bank用于确定性fresh参数、codebook/zero/tiny控制，不能拿它的进程驻留当真实模型峰值。

## 4. 112个输出的组成

| 原模型cohort | 原生臂/进程 | 完整输出（含恢复） | 原数值门槛失败 |
| --- | --- | --- | --- |
|Q4/KN FFN H64/F96，默认pipeline|1|14|1|
|Q4/KN FFN H65/F97，默认pipeline|1|14|1|
|Q4/KN尾部FFN，共同禁LowBatch的融合/展开|2|28|2|
|Q4/NK同一尾部FFN融合/展开|2|28|2|
|Q3/NK linear融合/展开|2|28|0|
|合计|8|112|6|

每个进程13个正式forward：两个seed×M1/2/8/9/17，再加M9的codebook/zero/tiny；最后额外一次错误恢复forward。全部输出与对应的旧Python结果逐位相同。独立CPU归档又重建实际encoded权重和完整dense参考，重算112个输出（106通过、6失败），不只比较最终hash。

原生程序exit0表示“执行及本地协议完成”，不是数值预算全通过。六个高幅度FFN失败仍在证据中，**runtime-only身份与数值资格是独立轴**。

## 5. 拒绝与恢复均实际执行

每个进程完成正常调用后，对同一VM执行错误X shape、错误X dtype和缺一个函数参数。三种错误均由VM/生成函数的真实检查拒绝，而不只是manifest parser提前返回；软件dispatch=0。这里没有扩大成所有q_weight/scale/任意strided tensor的自动安全验证。

随后重新绑定第一组正确输入和参数，运行完整消费者，8个恢复输出均逐位一致。它比“捕获到exception”多证明一次可用性，但仍不覆盖设备fault、取消、并发、跨stream异常或长时间OOM恢复。

## 6. AOT范围与剩余边界

本轮dylib身份沿用前轮，仍携带MSL和VM字节码。原生部署不用TVM编译器/Python，**不消除runtime源编译和PSO首用**，也未清驱动cache来制造“冷编译”。保存load/init和call+sync诊断值，不作Python/C++性能排名。

前轮materialized+LowBatch的host句柄访问问题没有被“原生部署”修复；本轮仅运行已检查安全的默认融合图，以及两臂共同禁LowBatch的对照产物。也没有重跑那个已知unsafe生成路径。LLVM-host、离线metallib、完整模型/请求、M5/NAX和物理GPU时间仍未验证。

## 7. 复现与采用顺序

CPU学习：labs/mlc_vm_stage_contracts.py，case `mlc-native-vm-contracts`；离线测试只检查标准库合同/冻结证据，不启动GPU。可选原生客户端、准备/重放驱动和独立归档器分离。

正式raw前缀`outputs/metal-mlc-vm-20261017-final-*`，每项有外层和`native-launch`的owned receipt。构建只新增任务客户端，不改TVM/FFI/MLC、全局环境和产品默认；磁盘仍足够，没有清模型或cache。

迁移时先固定source/binary/参数与VM ABI，再验证数值、错误后恢复，最后做完整请求的成本测量。任一library/source、编译器/SDK/OS/SoC、packed格式、参数顺序、owner/stream、图或shape变化均复查。它是确定图的部署证据，不是整个GPU研究完成声明。

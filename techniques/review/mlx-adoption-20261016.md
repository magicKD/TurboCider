# MLX实际adoption：view生命周期、Copy alias与过期结果

[总目录](../README.md) · [机制/复现](../gpu-ane/shared-surface.md) · [固定源码](../sources/repos/mlx-external-memory.md) · [完整证据](mlx-adoption-evidence-20261016.json)

2026-10-16，Apple M4 Max/64GiB、macOS26.6.2 **build25G83**、SDK26.5、Apple clang21.0.0、C++20，已安装MLX0.32.0/实际architecture applegpu_g16s。标准库wrapper记录383个installed headers与libmlx/libjaccl/metallib SHA并前后核对；源码阅读固定v0.32.0。没有下载/安装/重建MLX、私有ANE调用、模型/产品依赖或安全变更。

## 配方：整个region导入，逻辑view另建

| 合成case | 逻辑FP16 shape | row pitch元素 | physical payload bytes | IOSurface allocation bytes |
| --- | --- | ---: | ---: | ---: |
| padded | `[32,33]` | 64 | 4096 | 32768 |
| port-sized | `[1024,4096]` | 4096 | 8388608 | 8404992 |

页大小16384。两个模式均先导入整块page-aligned allocation（包括尾guard），再`as_strided`出逻辑view。raw-pointer路径检查deleter未被立即调用、真实buffer未copy；strict路径只接受`allocator::make_buffer`非null，用自有deleter调用`allocator::release`后通知owner。没有把任意CPU指针伪装成MLX Buffer。

实际比较Metal buffer.contents、MLX allocator raw_ptr与IOSurface base、buffer长度、storage mode、设备身份和offset；只发布相等bool、不发布地址/registry ID。所测MLX buffer实际为shared/untracked；每次外部GPU producer前先排空MLX stream、producer成功完成后才构造新MLX consumer并排空。host串行事件只证明此依赖合同，不是异步并行或物理device时钟。

外部Metal producer写每个逻辑FP16值，consumer用FP16 `2*x+0.25`再恢复FP32，全量exact gate。period37只是公开合成配方，8MiB case并非真实Qwen输入；native没有裁小或重复几行冒充原始模型。独立标准库verifier可利用声明周期流式计算完整digest，digest仍覆盖全部4194304个元素。

## 四份完整协议和一份保留的客户端失败

先行padded/seed17记录在首代alias与完整输出通过后，以`explicit copy did not create distinct backing`拒绝了错误的snapshot假设：`mx::copy`仍共享buffer。raw source/binary/报告与退出1保留、mapping quarantine到该child退出；这不是硬件不支持，也不能将部分成功冒充完整生命周期通过。

更正控制后，padded和port-sized各用seed17/29在四个新进程确认，同一份新native源码。每run两import模式×generation0/1/2，另每模式一个same-dtype Copy新consumer，共**32次完整正向consumer检查**；主代际还有24个实际GPU input全量hash。失败记录另保留1个partial全量consumer，不加到32个完整协议样本。

所有正向input/consumer hash与独立dyadic oracle相同，tail guard完整。flat parent离开scope后deleter仍0；所有consumer/controls结束、最后view释放后deleter恰好1次且原指针相同。只验证正常完成下的活owner/释放顺序，没有做真正UAF，也没有把错误/取消分支当已实测。

## 三种“保留结果”行为，不能混用

| 在generation0求值后，外部GPU写generation1 | 实际backing/结果 | 对generation1的完整检查 |
| --- | --- | --- |
| live strided view的全新consumer | 同backing，读取新值 | exact通过 |
| same-dtype `mx::copy`的全新consumer | 仍alias，同样读取新值 | exact通过 |
| 旧的已求值consumer array | 旧consumer输出未自动失效 | 全部元素错误，最大abs4 |
| FP16→FP32 materialized snapshot的新consumer | snapshot独立backing，仍是旧值 | 全部元素错误，最大abs4 |

八项cached-result检查与八项materialized-snapshot负控均按预期拒绝；前者重新eval既有已求值结果不等于新GPU kernel。另四项padded紧密stride实际GPU负控有1023/1056错误、其中496 nonfinite；contiguous port没有伪造“不正确packed stride”。共20项负控观察不计性能。它们与上一轮private output中“全finite但stride错误”的负控互补。

## 复核、范围与下一步

[独立合同/witness](../labs/mlx_adoption.py)、[native probe](../labs/metal/mlx_adoption_probe.mm)、[wrapper](../labs/metal/mlx_adoption_probe.py)、[curator](../scripts/curate_mlx_adoption.py)和[离线反例](../tests/test_mlx_adoption.py)都在techniques。curator只读明确原件、fixed clean source与调用方指定的installed assets，复核完整hash/顺序/owner/alias与失败；不导入MLX或重跑设备，不是硬件attestation。

本批比上一轮多采OS build和installed header/library身份，但没有对dyld实际loaded image做独立hash attestation、没有系统/逐arm峰值、cold/warm优化GPU分母或性能分数。源码tag与同版本wheel不是可复现构建证明。外部adoption不消除MLX运算自身的输出分配/转换，更不是runtime内部全零copy。

这里验证的是完成外部写入后构造全新、未显式`mx::compile`的consumer；没有授予compiled callable、闭包常量/结果缓存或原位重用graph的generation正确性。真实FFN应交替使用已有不同timestep的完整实际输入及其独立reference，不能因为重复同一个输入一直数值相同，就误把旧输出缓存当新计算。

**尚未跑过**真实二维private FFN→MLX restore/join完整路径；没有把独立shared toy、8MiB adoption与旧copy-bridge约1.12×拼成新模型收益。下一步按真实图descriptor做GPU pack/cast，保留完整BF16 input/output、paired prefix/GPU remainder、固定full/prefix/FP64 gates，才比较GPU-only/copy/shared serial/hybrid；完整block/request/LoRA、内存/冷启动/长时和失败取消继续独立验证。磁盘本轮约0.3GiB，未复制权重、删旧数据或修改产品默认。

后续记录：同日[真实共享FFN五臂](qwen-shared-ffn-20261016.md)已单独实测，包括actual port和264个compiled join位级见证。本页“尚未”描述当时adoption批次，不覆盖后续新证据，也不给本页合成输入换成真实模型标签。

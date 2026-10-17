# 编译与提交相关 PR：索引、cache 生命周期与参数依赖

<!-- knowledge
id: technique-metal-compiler-optimization-prs
type: method
stage: pipeline
operator: layout, scheduling
backend: metal, mlx, cpu-reference
device: cpu, gpu
symptom: compile-bound, launch-bound, wrong-layout, lifetime
semantics: contract-only
boundary: producer-consumer
api: public, reference-only
-->

[GPU 导航](README.md) · [本轮固定 diff](sources.md#compiler-sources-20261016) · [TinyJit](tinygrad-metal-jit.md) · [MLX compile](mlx-compile-cache-and-state.md)

2026-10-16 核对四份 PR 的 REST 元数据、head/base/merge 信息、diff SHA256，选读其关键区段。**closed 与 merged 再次分开**；不把两个关闭未合并的 tinygrad patch 当成当前主线已有修复，也不以本机普通 GEMM/Norm 通过代替其回归测试。

这两份 tinygrad 的 API 返回里 `merged_at=null`，却都有非空 `merge_commit_sha`；故登记将原 API 值单独保留，实际合并字段不据此填充。仅凭某个 SHA 非空或 `state=closed` 都不能宣布已合并，本轮也没有对这些 API 指针的具体生成原因作猜测。

| PR | 核查状态 | 解决的问题 | 可迁移技巧 |
| --- | --- | --- | --- |
| [MLX #3720](https://github.com/ml-explore/mlx/pull/3720) | 2026-07-07 merged | compiled negative-stride 的 contiguity/index 类型 | 连续范围与正向行连续分开；负 stride 也会要求 signed large index |
| [MLX #4248](https://github.com/ml-explore/mlx/pull/4248) | 2026-08-17 merged | compiled function 跨线程销毁时 cache erasing | 保留原 cache 身份及 entry 生命周期，不只给查找加一把锁 |
| [tinygrad #18648](https://github.com/tinygrad/tinygrad/pull/18648) | closed，未合并 | 任意 buffer/scalar 参数顺序贯穿多后端 | 明确 signature/slot 顺序、宽度、更新和 native launch 的一致性 |
| [tinygrad #18129](https://github.com/tinygrad/tinygrad/pull/18129) | closed，未合并 | PROGRAM 必须有 ProgramInfo/输入输出依赖信息 | 指令可运行不等于 scheduler 拥有正确依赖图 |

## #3720：不是把所有 view 都 materialize

该方案修复 compiled route 内的 contiguity 判定与有符号索引，而不是无条件把负 stride 复制为正 stride。1D 反向 view 也需要对应 large kernel；不能因元素数很少就认为 uint 索引足够。作者提供多 rank/mixed-stride 回归，当前本地实验只覆盖正 stride transpose，未跑该整套矩阵。

学习方法：对同一逻辑 view 分别跟踪 flags、stride、base offset、index dtype 与实际 kernel variant；准备/复制/输出恢复必须进入完整消费者。存在上游修复不等于任意安装 wheel 都相同，版本号本身也不是可复现构建证据。

## #4248：释放 cache 的线程不一定是创建线程

Python object 的销毁可能发生在另一个线程，若只访问“当前线程的 cache”就可能清理错误对象。patch 保留原 cache 句柄、entry shared ownership 与 erasing 同步；作者明确保留有关创建线程/函数使用期间销毁的假设。

不能将它简化为“compile 现在完全线程安全”。服务 reload、stream、request owner 与 interpreter teardown 都是独立场景。本轮没有复现竞态、没有给当前系统制造解释器关闭/并发压力，更没有将源码中新的锁授予旧 wheel。

## #18648：所有参数布局必须贯通

patch 试图用 signature 保持标量与 buffer 在任意顺序下的一致绑定，覆盖 codegen、host program、不同设备 launcher 和更新接口。一个简单“buffers 全在前、scalars 全在后”的 kernel 通过，不能证明交错顺序、64-bit scalar、image/pointer 别名与重放更新正确。

当前状态没有合并信息；关闭原因本轮未充分核对，不推测为收益不足、设计被否定或问题不存在。它可作为 ABI 学习样本，不作主线资格。另见 TVM POD-slot CPU 控制：是独立接口模型，不执行这个 patch。

## #18129：在一个设备通过可能隐藏依赖缺失

作者指出 HCQ 用 ProgramInfo 的 ins/outs 构造依赖，缺失信息在其他多 queue 后端可暴露错误，而 Metal/CPU 某些测试可能因没有独立 copy queue 而通过。这是该 PR 描述的测试范围，不是对全部 Apple 内存模型的保证。

本轮 Tinygrad collector 初次从错误层级找 PROGRAM/元数据的客户端失败，与此 PR 的框架问题不是一回事；不把同名对象合并为同一根因。后续验证应同时覆盖编译 metadata、实际任务依赖、输出与 owner，不只检查 source/binary 文件存在。

可执行学习控制：compiler_contracts.py、[owned graph](../examples/reference/contracts.py)、search qualification；均为原创 CPU 模型，不运行这四个 PR。换版本/参数顺序/stream/consumer 后重核。

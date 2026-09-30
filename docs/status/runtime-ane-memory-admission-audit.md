# Runtime ANE：内存准入缺口与下一步验证

更新：此页为接入前审查快照。请求级准入和真实模型复测见
[接续实施记录](runtime-ane-memory-admission-2026-09-29.md)；完整内存资格
仍未完成，下面的“当前实现”只指本页审查时的旧构建。

2026-09-29，1024² Q/K六配置对照运行期间的只读代码审查。
本页不是新性能实验，也没有实现或验收内存压力控制；运行库仍为55e5。
设计依据为工作区 `notes/runtime_ane_backend.md` 第33–35、72、82节，
参考 `references/vpipe/generative-models/shared/ane-ffn.cc`，参考仓库未改动。
设计稿第19节建议的LoRA权重合并已被用户后续要求覆盖：仍必须保持base-only
权重槽和运行时激活修正，不能通过合并adapter规避开销。

## 当前实现与证据

| 检查点 | 当前代码 | 结论 |
| --- | --- | --- |
| 建图前准入 | Qwen `pipeline.cpp`、Z `z_image.cpp` 构造 `HybridFfn` 前 | `min(2 GiB, physical - min(physical, MLX active + 4 GiB))`；不是当前系统可用内存 |
| 图内预算 | `ane_runtime.mm::RuntimeGraph` | 分配前检查形状估算，分配后检查实际IOSurface字节；预算不足经 `MemoryBudgetError` 回GPU |
| resident重用 | 两模型相同manifest身份直接 `begin_request` | 不重新评估系统压力；启动时足够不代表后续请求仍足够 |
| host输出与LoRA scratch | `ane_ffn.cpp::HybridFfn::run` | `output_`、`hidden_` 按全部ANE rows扩展，而图预算只依赖chunk行数 |
| 降级释放 | `HybridFfn::degrade` | 标记failed和原因；不会立即释放graph或scratch，不能直接等同于内存回收 |
| 现有预算回归 | `tests/native/ane_ffn_test.cpp` 的 `no_budget` | 覆盖极低建图预算及GPU正确结果，不覆盖压力在resident请求间上升 |
| 严格内存模式 | `native/runtime/memory_policy.cpp` | 明确拒绝GPU/ANE；不能把已有GPU-only内存认证算作ANE认证 |

所以已有64 GB机器的进程树采样和小预算回退，不足以证明设计要求的
低内存自动停用、无换页退化或完整driver/wired计账。这里确认的是缺少
保护与验收，不是声称已经观察到ANE造成OOM或换页。

## Chunk预算不等于整段host预算

当前SwiGLU v2图的估算为：

```text
weight_bytes = 6 * hidden * width
estimate = 2 * weight_bytes + chunk * (6 * width + 8 * hidden)
           + 64 MiB + chunk * width * 16
```

从当前代码公式计算（不是实测内存峰值）：

| 图形状 | 图估算 | 一chunk的output+hidden scratch | 一chunk的BF16 gate/up修正 |
| --- | ---: | ---: | ---: |
| Qwen H4096/F12288/c320 | 732.5 MiB | 10 MiB | 15 MiB |
| Qwen H4096/F12288/c1792 | 1158 MiB | 56 MiB | 84 MiB |
| Z H3840/F10240/c352 | 599.9375 MiB | 9.453125 MiB | 13.75 MiB |

scratch两项只讨论有adapter时；base不需要host hidden。host scratch和
修正随实际ANE chunks增加，发布给MLX的拥有所有权副本又有独立生命周期。
这些不是可直接相加的进程峰值，也不能假定图估算中的安全余量必然覆盖
任意长序列。`vector::resize`缩小size也不保证释放之前的capacity。

VPIPE的 `runtime_bytes` 把host项写为一chunk，并在首次成功predict后估计
external wired、向pool记账、析构时归还。它也明确系统wired会受其他进程
影响，并作上下界截断。不能把这个估算复制为TurboCider的精确ANE归因；
尤其当前host输出组织和LoRA修正生命周期与其不同。

## 计时结束后的实施要求

1. 先独立验证当前1024²对照，保留其库/源码/工具身份，另建候选。
2. 共用准入决策，区分系统可用余量、进程footprint、图/slot估计、host
   scratch与GPU保留空间。已有 `ProcessMemoryObservation` 可提供输入，
   但其系统available是VM计数估算，不是压力或driver上限的完整证明。
3. 在owner-thread安全点检查新请求及形状变化；低内存停用可选ANE，
   drain所有借用输入/异步工作后再释放资源。保留遥测和GPU正确回退，
   不在worker callback中释放自己，不引入每层后台采样线程。
4. 添加可注入观察值的host测试：未知观测、预算边界、整数溢出、
   resident压力上升、多chunk host增量、adapter切换、释放后的GPU结果。
   合法artifact不足预算应回GPU；非法manifest仍须报错，不能借降级吞掉。
5. 再跑微图/失败恢复和两模型匹配对照，验证正常内存时不降低现有性能，
   以及停用时确实释放自身资源。低内存资格需要专门、可控的实验，不能
   通过占满共享机器内存或影响他人任务来制造压力。

这不放开现有严格内存模式，也不以单个guard替代统一planner或完整内存
验收。当前[匹配内存证据](runtime-ane-matched-memory.md)与
[1024²在跑对照](runtime-ane-qwen-qk-1024.md)仍保留各自的限制。

## 独立草稿，不是产品实现

计时期间在忽略目录 `outputs/runtime-ane/memory-admission-candidate/`
准备了 `ane_memory.hpp`、`ane_memory_test.cpp` 和说明。只包含纯host准入
算术、scratch高水位/溢出计算及测试用例，尚未编译、执行或接入产品。
已有resident字节只计入可选tier逻辑额度，不再次扣除观测到的系统余量；
scratch计划也不冒充MLX修正/拥有所有权副本/Core ML的完整上界。
后续必须先完成上述会计和生命周期集成要求，不能直接复制后宣称内存安全。
草稿不在campaign绑定输入中；25项绑定SHA核对不变，未并行运行测试。

## SDK核对：不可直接复用当前available汇总

本机Command Line Tools的macOS SDK `usr/include/mach/vm_statistics.h`
在 `vm_statistics64` 定义处明确说明，`free_count` 已经包含speculative页。
当前 `native/runtime/memory_accounting.cpp::observe_process_memory` 却将
free、inactive、speculative、purgeable相加，所以至少speculative重复计数。
不能直接将该汇总接入ANE准入，也不能仅因数值小于物理内存就视为可信。
这次没有修改共享内存观测或改变既有GPU-only内存策略。

独立草稿改为明确要求raw `free_count * page_size`，并增加页数乘法的
零页大小/溢出测试用例（仍未运行）。它刻意不预支inactive/purgeable回收，
可能更保守；需要验证正常机器上不会不合理关闭已有收益的ANE路线。
free-only同样只是瞬时观测，不构成无压力、无换页或分配成功保证。

同一SDK的 `usr/include/os/proc.h` 将 `os_proc_available_memory` 标为
macOS不可用，不能用这个名字替代实现。`dispatch/source.h` 提供系统内存
压力事件，但如何接入owner-thread安全释放仍需实现与验证；本轮未新增
监控线程、dispatch source或产品平台依赖。

随后在同一隔离目录补充了 `ane_memory_observation.{hpp,cpp}` 与独立smoke
测试源：owner-thread调用Mach/sysctl，检查footprint字段版本、释放host port，
只取raw free页；非Apple返回unknown。它们仍未编译、执行或进入产品构建。
没有将草稿算作系统压力处理或统一planner已经完成。

草稿复查还区分了scratch净增长与扩容峰值：例如Qwen LoRA从一个c1792
chunk扩为三个，最终56→168 MiB、净增112 MiB；若旧buffer在新分配完成
前保留，需要新分配168 MiB，payload瞬时上界为224 MiB，而不是168 MiB。
已增加对应的未运行测试用例；该上界只针对精确容量替换，不能直接套在
容量增长规则不透明的 `std::vector::resize` 上，也不含MLX/Core ML内存。

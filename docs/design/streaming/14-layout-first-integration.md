# 14 · 布局优先：端到端整合与设计取舍

[目录](README.md) · [参数唯一规范](02-configuration.md) · [模型施工细则](15-adapter-implementation-plan.md) · [性能工具实施](16-performance-toolchain-plan.md)

日期：2026-09-16。本文是后续实施设计，不是新模型已可执行的声明。现有代码事实见 [13](13-implementation-progress.md)。
01–12 中的参数、容量算法、安全协议和性能阈值继续有效；本文补充整合边界、当前实现缺口和决策理由，不另造一套 schema。
框架级对象/身份/API事务与所有权合同进一步收敛于 [17](17-layout-first-framework.md)；
对应当前源码的逐文件实施、完整LTX接线与本机验收清单见 [18](18-code-change-matrix.md)。

## 1. 最终框架应该是什么

结论：采用“显式布局 + 模型描述 + 通用执行器 + 可选预算守卫”，而不是一个按内存大小自动选择 loader 的大类。
用户主要选择 K（槽数）、G（每组块数）、P（常驻前缀）；高级参数 D/Q 控制预取和读入并发。
不要求普通用户手算全部五个参数，可以显式选择完整 preset；运行时必须拿到确定的五元组。

布局优先有三层含义：

1. **策略先固定**：相同请求在不同剩余内存下得到同一布局；内存不足可以拒绝，但不能静默改参数。
2. **分配先于执行**：从 metadata 得到真实 slot bundle、scratch 和合法重用边界；不能边读边发现还需加内存。
3. **执行只消费计划**：运行时决定“现在能不能推进”，不再决定“下一层要用几个槽”。

“只设 slot_num 就够了”只适合作为 UI：同样 K，不同 P/G、activation、格式和组件驻留会有不同峰值。
保留预算作为正交 add-on，是为了让精确布局易于复现，同时让需要 Y 上限的用户得到明确的拒绝和资源解释。

## 2. 三个入口，一套核心

```text
未启用新配置 ──────────────────────────> 原有 planner/session/loader

显式 manual / 显式 preset ─> 合并与校验 ─> metadata describe
                                               |
                                   确定性布局编译 + 资源预览
                                               |
                                    模型执行资格 preflight
                                               |
                           可选 guard：同布局 admission/运行守卫
                                               |
                                 request orchestration（串行组件）
                                               |
                            stage executor（固定池、持久 I/O workers）
                                               |
                              model adapter ─> 原有数值 kernel

离线推荐工具 ─> 输出完整配置文件 ─> 用户接受后走 manual 入口
```

第三条入口不是隐藏自动模式。推荐失败不影响原请求，生成建议也不改 profile。
新框架 resident 是合法的未来执行形态，但当前 StageExecutor 只执行单 class streamed，不能拿 resident 编译成功当执行成功。

## 3. 层级、对象与所有权

| 层 | 对象/代码位置 | 生命周期与职责 | 禁止依赖 |
|---|---|---|---|
| 配置 | `streaming_contracts.hpp`、`config.cpp`、platform parser | request intent；保存 presence/provenance | GPU、系统内存探测 |
| 模型描述 | adapter describe；拟拆 `descriptor.*` | checkpoint/shape metadata；只读 | thread pool、真实权重 materialize |
| 计划 | `layout.*`；拟扩展 resource/pass/read plans | resolve 后 immutable | GPU completion、剩余 RAM 驱动改参 |
| 请求编排 | 拟议 `StreamingExecutionContext` / session bridge | 组件依赖、取消、资格、可选 guard、最终结果 | 逐 GEMM 数值实现 |
| 阶段执行 | 现有 `StageExecutor`、`io_executor.*`、`slot_pool.*` | pool 创建至最后 pass；单 owner | profile、重新调参、全系统 swap 控制 |
| 模型数据面 | `ModelSlotAdapter` / `tc_stream_adapter_v1` | buffer、read ranges、派生值、kernel/fence | 自建 pager、第二套调度器 |
| 平台数据面 | Metal/MLX bridge | 文件读取、allocation、completion | 解释用户 K/P 或签发资格 |

当前 `context.*` 实现的是 StageExecutor，**不是完整 request orchestration**。不要靠重命名把缺少的 text/VAE/跨 stage 生命周期算作已完成。
`c_bridge.cpp` 已是 C 到同一个 C++ executor 的桥，不应在 LTX/H3 再实现 C 版 pump。

资源 owner 必须分清：

- request/session 保存 checkpoint fd、descriptor、adapter state、终态错误；直到全部 worker/回调退出才释放。
- adapter 拥有真实 backing；SlotSafetyTracker 只拥有使用状态，不直接替模型释放 MTLBuffer。
- worker 仅借用被授予的写入 spans 和 scratch；GPU callback 仅借用稳定 mailbox 控制块。
- pass inputs 借用的 workspace、rope、conditioning 必须活到 pass drain；不能让 sampler 提前释放栈上绑定对象。
- model session 被 quarantine 时连同 adapter、mailbox、fd 和 worker 可见状态一起保留，不只留一个裸 pool 指针。

## 4. 布局不是文件切片：编译产物需要四张表

初版 `FieldSpec` 只有名称、storage ID、bytes、alignment；当前已扩展真实materialization/source/派生关系与pass identity，见13第9节。
session/API的immutable construction view接线和完整非slot资源仍待闭环。
F1/F3 应补齐以下关系，尽量使用索引而不是热路径字符串查找：

| 表 | 关键字段（内部草案） | 消费者 |
|---|---|---|
| source table | file identity、offset、length、dtype/packing、校验信息 | bounded reader |
| backing table | storage ID、field ID、容量、alignment、storage mode、owner | allocator / 可选 ledger |
| binding table | group/block/field → slot span、shape/stride、转换策略 | fill / kernel binding |
| access table | pass/step 模板、block 顺序、external reads、safe commit/last-use | executor / verifier |

source bytes、decoded bytes、allocated capacity 分开报告：压缩/量化权重、F32→BF16 表、padding 都会让三者不同。
同一个 source 可产生两个独立 backing；只有实际共用 storage 才能去重，不能按 checkpoint tensor 名字去掉实际副本。
binding 描述必须包含 tail group 的有效 block/field 数；slot 中上次残留的数据不能被本次 kernel 读取。

实施上先在 C++ descriptor 扩展真实字段，生成 immutable model construction view。
现有 C bridge 的 `slot_capacity_bytes` 是容量校验值，**不是完整 buffer 布局**；LTX 可通过 adapter.user 持有 construction view。
construction view 必须从同一编译结果投影，核对 digest/field totals，不允许 adapter 再自行分组或重新算 P/K。
首版不必把所有描述字段塞进公共 ABI；需要跨 ABI 传更多字段时另建带 version/size 的 view，不静默扩现有结构体。

## 5. 单请求完整事务

| 阶段 | 成功条件 | 失败后的行为 |
|---|---|---|
| Parse/Merge | schema、presence、旧字段冲突通过 | 不触碰旧模型 session |
| Describe/Compile | 真实文件、格式、shape、访问闭包可描述 | 输出确切不支持原因；不分配 slot |
| Eligibility | exact tuple 有证据，功能分支受支持 | 保持 `streaming_layout_not_certified` 或细分原因 |
| Optional admission | whole-request upper 与 Y/X/S 合法；baseline 可接受 | 拒绝同布局；不能减槽后继续 |
| Construct | prefix、slot、control、scratch 可完整创建 | 按已创建资源逆序清理，未创建资源不释放 |
| Execute | pass 单调、读写权限/fence 合法、实际布局一致 | sticky failure → stop dispatch → join/drain |
| Terminal | 最后输出和必要 cleanup 完成 | 无法证明安全则 quarantine，绝不报告成功 |

prepare 与 generate 的访问计划不同：load-only 不能伪装成完成了 denoise/VAE。
是否清理旧 retained session 是 admission 阶段显式操作；不支持的布局不得先卸载旧缓存再拒绝。
取消标志可以跨线程设置，但 destroy/finish/ledger/最终错误归并仅由 owner 完成。

## 6. 持久 multi-slot 的实际生命周期

现有 `begin → run_pass(pass, step) × S → finish` 是正确的接入基础：**持久的是池和线程，不是尾块的内容缓存**。

```text
request start
  describe + compile
  create prefix + K slot bundles + Q worker scratch
  executor.begin                    （一次）
    bind step inputs -> run_pass(0, step0) -> drain pass readers
    bind step inputs -> run_pass(1, step1) -> drain pass readers
    ...
    LTX upsample：池保留；workspace 交接单独核算
    ...
    bind step inputs -> run_pass(S-1, last_step) -> drain
  executor.finish                   （join / drain / destroy，一次）
  VAE / export / request terminal
```

pool 一次创建是“同一兼容 resource domain”的承诺，不是所有组件或 layout class 永远只分配一次。
多 class 先按 09 的 barrier 切换池；首发 executor 暂不支持，必须在路由阶段拒绝。
中间 pass 完成不调用 finish，不重建 Q 个线程，不把局部 diffusion step 当新的 request generation。
run_pass 的 pass 从0连续递增；step 是模型语义坐标，可不同于 pass，不能用 `run()` 的默认 pass=step 方便写法接 LTX 全流程。

### 6.1 Offload 到底是什么

- 不可变权重：reader 结束后丢弃逻辑内容、下次覆盖 slot；一般不需要 GPU→CPU 再写 SSD。
- 每步派生表：明确下一步重算或重读；复用原地址也必须等最后读者。
- 必须保留的状态/activation：不能丢弃；首发放独立资源并计峰值，不偷用 slot 做有状态交换。
- 统一内存机器上的 CPU 副本不等于释放物理内存；首发避免另建全量 host 权重缓存。

### 6.2 K/G/P/D/Q 调参顺序

先选合法最小 P 和 G=1，验证 K=1 正确性；再验证 K=2 的 overlap；K=3 用于检查尾延迟/调度抖动收益。
这只是探索顺序，不是给所有机器的生产推荐。先保留 Q=1 的可解释基线，再独立测试 Q 增加后的读盘与计算竞争。
P 增加可减少每 pass 重读，但增加常驻和首次加载；G 增大可降低调度/小读次数，却提高首组等待、容量和边界约束。
每次只改变一类因素；不能用 P 增多带来的提速证明 executor 更轻。

## 7. 当前基础必须补的工程缺口

以下是源码审阅发现的接入前检查项，不代表已复现线上故障：

| 缺口 | 当前事实 | 设计要求 / 验收 |
|---|---|---|
| deadline 不是硬停止 | `run_pass` 有 stall timeout；`shutdown_and_join` 仍阻塞 join，C drain 无 deadline | 区分 progress timeout 与 cleanup timeout；阻塞 fill/driver 用隔离 worker 生命周期，不能 detach 后 free；FLT-02 |
| request cancellation | C cancel 置位；owner 到检查点才传播到 I/O shutdown | 定义 cancel-to-stop-dispatch 和 cancel-to-safe-terminal 两个指标；长 prefix/encode 单独报告 |
| pass 内全 drain | 当前 run_pass 尾部调用 adapter drain | 首版允许 pass barrier；不能在每个 group 用全设备 wait 冒充 fence；PERF-01 |
| per-group 控制成本 | mailbox 用 mutex，pump 读时钟并有5ms超时等待 | 先测 wake/lock/CPU，5ms是最长轮询等待而非固定 sleep；不要未测就上 lock-free |
| C failure 语义 | 错误文本由 adapter/per-slot buffer 和 bridge 汇总 | 测旧错误残留、第二次调用、非 std exception、owner 误用；主错误不可被 cleanup 覆盖 |
| quarantine 销毁 | C destroy 失败保留 handle；C++不安全析构会 terminate | service 必须保留整个 owner/session，禁回 idle；所有异常出口都有 test |
| C bridge 能力 | 单 class streamed、固定 callback 表，无真实模型 ranges | 添加 adapter construction view 和 exact resolve；不是另写 scheduler |
| 性能证据 | host/C bridge/synthetic GPU 有部分证据 | 当前 lifecycle 变更需重新跑 sanitizer/Metal；真实模型 P0/P1 仍缺 |

## 8. 配置文件和机器档位

沿用 [02](02-configuration.md) 的 JSON/profile v2。不要新增 `slot_num`、`memory_tuning`、`gpu_slots` 等同义 runtime 字段。
统一单位：byte 字段整数；UI 可展示 GiB，但转换只在输入边界一次完成。buffer_percent 是预留比例，B=Y×(1−x/100)，按既有校验舍入。

建议文件分工（目录为后续建议，尚未实现）：

```text
configs/streaming/presets/<model>/<layout-name>.json       完整可合并的配置，不含认证自签字段
docs/design/validation/<campaign>/preset-evidence.json    文件hash→layout→测量与资格范围
用户硬件 profile v2                                     显式复制/采用测过的 preset 值
```

不要给当前 parser 添加尚不支持的 `extends`/`preset_ref`。第一阶段工具输出展开后的合法 request/profile 文件即可。
选择器可按 RAM 档位筛选候选，再按 shape、GPU、SSD、格式展示证据；RAM 不能单独决定最佳 K/P。
“16/24/32/64 GiB”只是 UI 分类，不硬编码到 compiler；没有该硬件证据时标 experimental，不推断必定可跑。
layout 报告列 prefix/pool/scratch/control 和 non-slot unknown；尚无完整 upper 就不能推荐为“保证低于Y”。

用户路径建议：

1. 默认用户什么都不配，完整保持当前行为。
2. 高级用户指定完整 layout，plan 解释资源和支持状态，再显式运行。
3. 普通低内存用户选一个有证据的 preset，可附加 guard；若失败，展示资源缺口和未执行建议。
4. 高内存用户继续默认，或显式测试新 resident；绝不为了“统一管理”强行通过 slot 执行器。

## 9. 代码整合顺序与完成门

本表是 [11](11-work-packages.md) 的落地导航，不引入第二套 F 编号。

| 原任务 | 下一最小交付 | 必须先有 | 不得提前声称 |
|---|---|---|---|
| F1/F2 | persistent bridge 的异常、取消、多pass回归；descriptor真实ranges | 当前 host 基础 | 模型可执行 |
| F3a/b | 复用已实现LTX metadata/固定slot，补generic真实identity与construction投影 | field级构造计划 | 全进程上限已保证 |
| F3c/d | 内部两stage已tiny验证；补正常session exact路由、status-aware owner与完整请求 | pass输入/所有权合同 | 默认性能已通过 |
| F6-layout | LTX质量、P0/P1、normal-target、失败清理证据 | 支持tuple固定 | 任意K/shape均认证 |
| F4 | H3双槽映射与闭包 | LTX验证的通用协议 | H3单槽自然支持 |
| F5/F6-bounded | 全请求资源闭包、same-layout guard、L2/L3/P2 | 真实模型adapter | 无guard也保证零swap |
| F8 | Flux/Z component，再独立group reader | backend引用释放证据 | 统一API即已统一执行 |
| F9a | 工具原型可提前；正式preset最后签核 | 可解释计划/独立证据 | 仿真推荐可自授执行资格 |

首个可交付垂直切片：一个明确的 LTX checkpoint/format/shape/K/G/P/D/Q tuple，真实 GPU 运行完整请求，输出同质量，默认 P0 和同布局 P1 通过。
先做到这一点再扩矩阵，比同时给四个模型挂空 adapter 更有价值；默认路径始终保留。

## 10. 架构评审通过条件

- 任意模块都能回答“谁决定布局、谁拥有 backing、谁能写 slot、谁证明最后 reader 完成”。
- 编译输出不依赖运行时空闲内存，低预算只拒绝同布局。
- adapter 没有私有线程池/隐藏缓存/预算重选；新模型主要新增 descriptor、reader/binding、completion 三类代码。
- 默认路由不创建新对象/线程、不新增每层同步；性能是否合格仍以 [12](12-acceptance-playbook.md) 为准。
- 文档/结果明确区分已实现、合成验证、真实模型验证、性能通过和 bounded认证。

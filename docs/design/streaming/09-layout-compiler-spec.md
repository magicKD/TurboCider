# 09 · Layout compiler：数据模型与可实现算法

[目录](README.md) · [配置规范](02-configuration.md) · [执行器细化](10-executor-implementation.md) · [实施任务](11-work-packages.md)

状态：2026-09-16 目标规格；部分类型/分组容量算法已实现，完整metadata与资源编译仍待完成，范围见 [13](13-implementation-progress.md)。
最新compiler已有v2 canonical：源artifact/ranges、destination materialization/派生关系、workload和pass模板进入identity；
source读取、未对齐content、对齐capacity分开计数。完整request资源/资格仍未实现，不能把`materializations_complete`当认证。
02 定义用户参数，本文定义其确定性编译结果。
目标是“一个配置、一份计划、一条执行路径”，不是再增加一套预算驱动的调参器。

## 1. 模块边界与依赖方向

```text
request/profile --parse/merge--> StreamingIntent
                                      |
model metadata --describe--> Descriptor+----> LayoutCompiler ----> ResolvedPlan
                                                                    |
                                           execution registry ------+-- eligibility
                                           optional BudgetGuard ----+-- admission
                                                                    |
                                                  StreamingExecutor + ModelAdapter
```

依赖规则：

- `native/core/streaming_contracts.hpp` 只含值类型；不包含 Metal、MLX、session 或文件系统实现。
- `native/runtime/streaming/descriptor.*` 验 descriptor；`layout.*` 只做纯计算，不创建线程、不读系统内存。
- platform/session 负责把真实 checkpoint metadata 转成 descriptor；编译器不知道某模型文件名如何拼接。
- executor 消费只读 plan，不能调用 config merge、推荐器或旧 budget→prefix 函数。
- memory guard 消费资源计划；不反向修改计划，也不控制数值 kernel 选择。
- Python 工具调用同一 native compiler；不得复制分组/容量算法。

编译器对相同规范化输入必须逐字节产生相同 canonical plan，不依赖当前空闲 RAM、线程完成顺序或文件缓存。

## 2. 三份对象，禁止混用

| 对象 | 创建时机 | 可变性 | 必须包含 |
|---|---|---|---|
| `StreamingIntent` | parse/merge | resolve 前可合并 | presence、字段来源、K/G/P/D/Q、retention |
| `ResolvedStreamingPlan` | metadata preflight | 发布后 immutable | groups、pool layouts、pass templates、资源生命周期、digest |
| `StreamingRunState` | execution | owner 唯一写者 | tickets、slot state、fences、取消、actual counters |

系统 observation、wall time、实际 completion、baseline 不写回 resolved plan。否则同一布局会产生不同 digest，难以验收。
`requested` 保存提交意图；`resolved` 展示继承的完整配置；`actual` 由执行事实累计，不是拷贝 resolved 冒充。

## 3. Descriptor 必需类型（内部草案）

```cpp
struct TensorStorageSpec {
    StorageKey unique_storage;
    LayoutClassId layout_class;
    uint64_t logical_bytes, backing_bytes, alignment;
    Span<FileRange> source_ranges;
    FormatId format;
    Mutability mutability;  // immutable_weight / per_step_derived / stateful
};
struct BlockSpec {
    BlockId id;
    Span<TensorStorageSpec> tensors;
    Span<StorageKey> external_reads;
    BoundaryContract boundary; // 该边界是否可结束 encode/reuse
};
struct StageSpec {
    StageId id;
    Span<BlockSpec> blocks;
    Span<PassSpec> passes;
    SupportedLayoutSet supported;
    Span<ResourceLifetimeSpec> non_slot_resources;
};
```

这些 `Span/Id` 是概念类型，不要求先引入新依赖库。实际可用预分配 vector 加索引；跨 C ABI 用 POD 数组和 count。
每个 borrowed view 的 owner/lifetime 必须写在头文件；执行期间不得引用 parser 临时字符串或已关闭的 metadata mapping。

### 3.1 Stage、pass、step 不能用一个序号替代

- config stage 是稳定的资源域，例如 `denoiser`；它不一定等于现有 recipe 的一个 stage。
- execution pass 描述一次有序访问；LTX 的 `av_stage1/av_stage2` 可共享 denoiser backing，但有不同 shape/workspace。
- step 使用稳定的全局执行坐标；模型局部 sigma index 另列，不在框架中推断。
- `resource_domain_id + pass_id + step_id + group_id` 唯一定位工作项；旧 recipe ID 通过显式映射保留。
- descriptor 描述前缀每个 pass 仍参与计算，P 只改变 residency，不删除 P 个 block 的计算。

H3 当前存在 active-block/跨 block fusion 逻辑。首发 adapter 只接受已冻结的访问模板；动态跳块/自适应 cache
若不能证明外部读集合与使用终点，明确 unsupported，不把它伪装成普通顺序循环。

### 3.2 Layout class 不只是 dtype 或总字节

至少包含：tensor field 语义、shape/stride、dtype/packing、alignment、scale 布局、storage mode、绑定方式。
允许 padding 的兼容关系必须由 adapter 显式提供，compiler 不能仅凭 `sizeof` 相同判定兼容。
两块共享一份不可变 tensor 时只计一次 storage；只要 refill 会覆盖它，就不能把相同地址当合法共享。

## 4. 编译流程与错误位置

| 步骤 | 动作 | 必须拒绝的情况 |
|---|---|---|
| C0 Normalize | 合并 presence/provenance，固定 stage 顺序 | 未知字段、legacy 双权威、streamed 参数缺失 |
| C1 Describe validation | 核对 IDs、ranges、aliases、pass graph | 重复 ID、越界 offset+length、循环依赖、未定义引用 |
| C2 Boundary analysis | 标记可切分位置和跨块读取 | 所选 G/K 无法满足 fusion/dependency closure |
| C3 Group | prefix 剥离后，连续兼容最多 G 块 | 不可流式 block、非法分组边界、P 超范围 |
| C4 Assign | 固定 group→pool→slot 映射 | K 大于该 pool generation 可服务的 groups |
| C5 Size | 逐 field 算 capacity、staging、control | 对齐/加法/乘法溢出、reader destination 不适配 |
| C6 Schedule | pass 模板、release 边界、prefetch 表 | read-before-ready、未提交 buffer 的循环依赖 |
| C7 Resources | unique storage live intervals 和上界 | bounded-required unknown 标为不可 admission，不能填 0 |
| C8 Seal | canonical serialization/digests/report | 超过控制平面大小限制，identity 不完整 |

一个失败至少报告 `code/config_path/stage/block_or_group/expected/actual`。涉及 bytes 时附 required/available。
失败的 plan 可给诊断，但不得留下半初始化 executor。

## 5. 分组与多 layout class 的确定性

1. 从逻辑 block P 开始；按原顺序最多取 G 个。
2. layout-class 边界提前结束 group；最后 group 可以不足 G。
3. 若有 fusion/跨块依赖且上述边界不安全，首版拒绝该 G；不自动加大 G 或降低 fusion 优化。
4. 同一连续兼容区间形成 pool generation；首版只允许一个活动 streaming pool。
5. 跨 class：drain old → destroy old → create new，不预先创建另一套权重池；pass 再进入该 class 时重新建立。
6. 每个 pool generation 内使用局部 ordinal modulo K；全局 group ID 仍唯一。
7. 任一区间 group 数小于指定 K 时拒绝，不能偷偷减槽。需要支持小尾段时另定义并认证一种 layout revision。

因此 G/K 是显式且可解释的约束；并非所有模型都支持任意组合。首发 LTX/H3 同构候选先验证单 class；
Flux 多 class 首期只 plan-only。padding 成单类池属于以后显式的 adapter revision，不能运行时临时决定。

## 6. 槽容量必须逐 storage field 求上界

同一个 slot 先后服务 group A、B，A 的两个独立字段是 `(320,64)` MiB，B 是 `(64,320)` MiB。
若 field backing 地址/绑定固定，该 slot 需要 `max(320,64)+max(64,320)=640` MiB，
不是 `max(384,384)=384` MiB。只有明确支持重排 offsets 的 packed arena 才能按各 group footprint 最大值分配。

```text
fixed-field bundle:
    capacity(slot, field) = checked_align(max(required(group, field)), alignment(field))
    capacity(slot) = sum(unique backing fields) + mandatory binding overhead
packed arena (adapter explicitly supports relocation):
    capacity(slot) = max(aligned footprint(group))
    plan stores per-group offsets + bindings + validation constraints
```

tail group 没有的字段不得被 kernel 读取；未使用空间可以不清零，但必须证明无越界读取。
slot backing 固定容量并不意味着 worker 可以任意写其全部 bytes，read plan 只授予当次必要 spans。

## 7. 可手算 golden fixture

见 [compiler-golden.json](examples/compiler-golden.json)，它是合成测试输入，不是模型配置：

- N=10；每块 128 MiB；P=2；G=3；K=2；D=1；Q=1；两个相同 pass。
- prefix=[0,1]；groups=[2,3,4],[5,6,7],[8,9]；slot assignment=[0,1,0]。
- 两个 slot capacity 都为 384 MiB；prefix=256 MiB；weight pool+prefix=1024 MiB。
- 不做跨 pass content cache 时：每 pass suffix 读 1024 MiB，两 pass 读 2048 MiB；prefix 另读 256 MiB 一次。
- 上述读量假设fixture是无转换的一对一复制。当前只有bytes而没有materialization的合成descriptor返回source读取量unknown，
  不能自动把destination bytes填成source；手算expected不替代真实source描述。
- pool 创建一次、backing 跨 pass 保留、总 fill=6；这是 slot bundle 数量，不等于底层 MTLBuffer 个数。
- 整请求峰值仍需额外计算 activation/VAE/framework；fixture 不声称 1 GiB 能运行一个真实模型。

另外必须有：N−P=1/K=2 拒绝、P=N streamed 拒绝、class 尾段少于 K 拒绝、零/溢出/异构 field 最大值例子。

## 8. 计划大小和运行时复杂度

- descriptor/range/group 表按 metadata 大小存储；执行不展开 `steps × blocks × tensor fields` 的重复计划。
- 使用 immutable pass template + 有界 loop cursor；逻辑事件序号用 checked 64-bit arithmetic。
- runtime 活动记录 O(K + Q + queue_count×K)，不随完成的 steps 增长；telemetry 另有明确上限。
- canonical report 可流式序列化；不为输出 JSON 同时复制所有 tensor 名称/ranges 数次。
- 每个 metadata 文件大小、tensor 数、range 数、stage/pass 数和 report bytes 必须设显式实现上限；
  F1 提交实际数值和 max-limit fixture，超过返回 `plan_too_large`，禁止默默截断。
- 计划、metadata、workers、error mailbox 本身计入 control-memory；先检查 metadata 上限，再加载完整索引。

streaming模板cursor不等于现有memory schedule已经支持压缩模板。F5需明确bridge方案：
对当前显式schedule在受控大小内预展开并完整计入control upper，超过上限拒绝；或另PR实现兼容的versioned模板cursor。
不得一边保留无界逐step展开，一边报告O(K)为全框架内存。上述O(K)仅指活动执行状态，不含不可变计划和guard schedule。

## 9. Identity、cache 与文件一致性

分离三个 digest：

1. `layout_digest`：规范化布局+descriptor structural identity+schedule/adapter revision，不含 Y、时间和用户注释。
2. `admission_digest`：layout digest+required-site/envelope revision+memory policy；baseline observation 单独记录。
3. `evidence_identity`：模型内容 identity、device/backend/build、workload、layout、quality policy、测试工具版本。

改变 Y 但不改变布局，layout digest 必须相同；改变 G/K/P/D/Q、格式、pass 或实际 reader revision 则变化。
request-only plan cache 可在 F1 之后另加；首发不要增加全局 cache 以免破坏默认行为或留下跨请求资源。
持有用于 describe/fill 的 fd 与稳定文件 identity，拒绝 truncate/replacement；mtime/size 校验不是完整内容完整性证明。
可信离线 digest 可避免每次全文件 hash；本轮不引入跨设备共享的“自动认证 cache”。

## 10. Compiler Definition of Done

- 所有 C0–C8 正反例覆盖；JSON key 排序变化不改变 canonical digest。
- 纯 compiler 测试不访问 GPU、swap、系统 available 或全权重文件。
- 相同 layout 的 guard on/off 生成同一布局和容量；不足只影响 admission。
- property tests 独立验证每块执行次数/顺序、group coverage、slot capacity、bytes overflow、alias 唯一计数。
- verifier 用独立简单参考实现验证 fixture，不能只让 compiler 序列化自身再反序列化就判正确。
- F1 仅 plan-only；上述完成不授予真实执行资格。

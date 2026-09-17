# 25 · Public Preset 代码实施规格

[目录](README.md) · [产品与配置](23-public-memory-tier-presets.md) · [候选探索](24-memory-tier-exploration-and-acceptance.md) · [实施验收](26-public-preset-acceptance-and-release.md)

日期：2026-09-17。原始设计源码基线：`7308db3`，runtime 基线 `052265f`；控制面提交：`63b73d9`。
状态：**部分控制面已提交，engine/App/模型执行与档位认证仍待实施**。本文不改变 manual v1、slot 安全协议或 P0–P4 定义。
当前代码检查和总体追踪见[27](27-public-streaming-delivery-blueprint.md)；实施级对象/API设计见
[28](28-public-runtime-code-design.md)，逐PR和实机验收计划见[29](29-public-implementation-and-acceptance-plan.md)。

## 0. 当前工作树检查点和立即阻断项

### 0.1 已提交并完成编译/host 回归的部分

| 实现 | 真实文件 | 当前能力 |
|---|---|---|
| selector contract | `native/core/streaming_contracts.hpp` | schema v2、memory_tier/preset、五个target、disabled语义 |
| request/profile | `contracts.hpp`、`request.mm`、`profile.mm`、`streaming_config.mm` | manual/selector互斥、完整selector替换、provenance和legacy冲突 |
| plan report | `plan.cpp`、`results.mm` | active selector只返回plan-only和resolution-required |
| catalog skeleton | `runtime/streaming/preset_catalog.*` | target margin、release/revoke/device/workload/calibration过滤和确定排序 |
| production registry | 同上 | 故意为空，revision=`tc-streaming-catalog-empty-v1` |
| public options ABI | `turbocider.h`、`c_api.mm` | metadata-only列出五档；当前没有artifact exact identity |
| Swift skeleton | `TurboCiderNative.swift` | v2 selector/request/options类型、plan/options/generate overload |
| host tests | `test_streaming_contract.py`、`streaming_preset_resolver_test.cpp` | parser/plan/options空表/resolver基础 |

本次审阅实际运行：

```sh
tools/native/build.sh
make test-streaming-contract
make test-streaming-host
```

native 与 Swift/App 全量编译成功；contract 12项和四模型public fail-closed gate通过；host的3535 layouts、14组
K/D/Q、multi-class/fault/cleanup、resolver及四模型descriptor通过。这些结果不包含GPU public执行、App UI、整请求内存或新性能数据。

### 0.2 已修复的第一道安全闸门

原实现的 `tc_engine_generate` 只检查 manual `request.streaming.active()`，`preparation_call` 也只拒绝 manual
streaming。当前工作树已在 resolver/authority 尚未接入期间补上 active selector 的统一早拒绝，避免继续落入普通
`session->generate/prepare`。

当前实现等价于：

```cpp
require(!(request.streaming_selector && request.streaming_selector->active()),
        "streaming_preset_resolution_required: unresolved public selector");
```

该 gate 位于全局GPU execution lock、跨进程DeviceLease、GPU configure、session prepare/generate和pool/worker创建之前。
contract 已覆盖普通 public engine 与 private candidate 的 generate/prepare。实现真正 resolver 后，只能把它替换成显式
`resolve → authority → generate_resolved` 分支，绝不能直接删除。

### 0.3 当前 skeleton 不能直接升级为 production 的原因

1. `StreamingPresetRecord` 还没有完整 artifact/variant/token/kernel/reader/component/build/evidence identity。
2. `tc_streaming_options_json` 目前只使用 request 和 device，返回 tentative；没有 engine root、verified source 或 token exact match。
3. production catalog 为空，没有任何 full-request calibration 或 reviewed record。
4. 没有 `tc_engine_resolve_streaming_json`、`StreamingAuthority`、`ResolvedRequestExecution`、`generate_resolved`。
5. Swift v2 是可编译骨架，尚未完成 v1 所有语义的 round-trip/golden，也未接 `NativeJob` versioned envelope。
6. App 没有 `StreamingChoice`、异步 options 状态、picker、unavailable reason 或 result 展示。
7. request/profile 已有 raw duplicate-key scanner；但 target 仍通过 NSNumber/double 后验判断，数值精确的指数形式会被接受，
   尚未满足冻结的“无指数十进制整数”lexical contract。catalog builder 也还没有同等级 scanner。
8. current selector syntax允许一致的GPU+ANE请求，但public v1按用户范围应只发布GPU-only record；无record时需早拒绝。

因此当前正确产品行为是：off/default保持原样，options显示无public记录，active selector执行fail-closed。

## 1. 先固定架构：一条控制面、一套执行器

用户选内存目标，后台选择已审核 exact preset。目标不在运行时反推任意 P/G/K/D/Q，也不进入 block hot path。

```text
App/CLI/service 用户意图
  ├─ off / legacy → 原有请求与执行路径
  ├─ private manual v1 → 原有精确布局与测试 gate
  └─ public selector v2
       → profile 合并 / 请求语义检查
       → artifact、token/workload、device、container identity
       → catalog 过滤与确定性排序
       → canonical manual config + compile_layout
       → 内部 authority + 不可变执行 snapshot
       → 模型原 kernel + 原 StageExecutor / pager / reader fences
```

分工不得倒置：App 不维护“12 GiB=P14”表；resolver 不创建 GPU 权重；adapter 不读取 target 选 slot；executor 不认识模型名。
不新增 `AutoMemoryScheduler`。`MemoryGuard` 的纳管/硬上限能力仍独立，catalog 的经验内存目标不授予 guard 资格。

### 1.1 本次源码核对发现的实际接缝

| 接缝 | 当前事实 | 必须采取的改造 |
|---|---|---|
| `native/core/contracts.hpp::Request` | 已有 manual `streaming`、`streaming_requested` | selector 意图单独存储，不把 v2 塞入 v1 stages |
| `request.mm → resolve_profile → validate_streaming_config` | 先合并再验证；profile 会覆盖部分其他 execution 字段 | 保持旧合同；新增 selector 原子替换，合并后查 legacy 冲突 |
| `plan.cpp::make_plan` | 不知道 engine root；active streaming 的旧 heuristic estimate 被清空 | resolver 不能直接藏进此纯请求函数里读模型/授予权限 |
| `c_api.mm::tc_engine_generate` | 默认不额外 make_plan；public 被 bool gate 拒绝 | 只在 opt-in 新分支创建 preflight snapshot |
| `ModelSession::generate` | 接收 Request，模型内部制定 plan | 新增明确的已解析入口，避免重复 compile/重新选 layout |
| Swift `NativeRequest` | schema v1，execution 是字符串 | additive v2 payload；off 保留原序列化 |
| `NativeJob.request` | 固定为 v1；输出管理/历史复用也依赖它 | 任务 envelope 迁移，不仅添加几个 preset 展示字段 |
| `FluxExactAdapter` | P0、两个 retained class pool；single 首组触发 concatenate | prefix 改造需移动边界逻辑，不能只解除 P0 校验 |
| `MlxWeightPager::State` | 保存 descriptor/stage/layout 的裸引用 | snapshot 必须比 pager、worker、回调活得更久 |

## 2. 类型合同：意图、解析结果、权限分离

下列 C++ 是接口草案，省略 include/错误包装，不可直接当 patch 编译。

### 2.1 保留 manual v1，不污染配置类型

建议新增 `native/core/streaming_selector_contracts.hpp`：

```cpp
struct MemoryTierIntent { uint64_t target_request_memory_bytes; };
struct PresetIntent {
    std::string preset_id, catalog_revision;
    uint32_t preset_revision;
    uint64_t target_request_memory_bytes;
    std::optional<std::string> expected_resolution_digest;
};
struct DisabledStreamingIntent {};
using PublicStreamingIntent = std::variant<
    DisabledStreamingIntent, MemoryTierIntent, PresetIntent>;

struct SelectorEnvelope {
    uint32_t schema_version = 2;
    PublicStreamingIntent selection;
    std::string retention = "request";
    std::string origin; // request/profile; not authority
};
```

`Request` 拟增加 `std::optional<SelectorEnvelope> public_streaming_requested` 与 profile 合并后的 effective intent。
现有 `streaming` 仍只保存 manual semantics；public 分支成功解析后填 canonical config。原始 selector 保留用于报告。

不在公开 Request 中保存 `StreamingAuthority`，不序列化指针、capability record 或可反序列化的 `authorized=true`。
只有 native resolver 可以构造权限对象；字符串 digest 仅用于绑定/诊断，不是密码或授权票据。

### 2.2 Workload、calibration 和 performance 分开建 key

`WorkloadKey` 至少包括：model/variant、operation、artifact/content revision、dtype/quantization、backend/kernel/reader revision、
W/H/frames/fps/steps/batch、每个 encoder 的 valid/padded/compute token rows、dynamic_text、audio、conditioning/refiner/upsample、
VAE tile policy、LoRA 与 approximation identity。首版禁止的功能仍写入 key 的禁用值，不能省略成 wildcard。

- `ExecutionKey`：决定同一计算/布局是否安全正确；不含输出文件路径或随机 seed。
- `CalibrationKey`：ExecutionKey + execution container + device/物理内存类 + runtime/allocator/cache-state policy。
- `PerformanceKey`：CalibrationKey + SSD/storage class + timing protocol revision。
- `RequestDigest`：完整语义请求，包括 seed、prompt/input identity；用于任务冻结与复放。

首版只匹配审核过的 exact shape 和明确 token 条件。`512²` 不能自动覆盖任意等面积长宽比；相同 token 数也不自动覆盖
不同 conditioning 分支。是否允许一段 token 范围必须由 record 显式声明并有边界/内部样本证据。

### 2.3 不可变 snapshot 和单次规划

建议新增 `native/runtime/streaming/preset_resolver.hpp/.cpp`：

```cpp
struct ResolvedSelection {
    PresetKey preset;
    StreamingConfig canonical_manual;
    std::string component_policy_revision, execution_container;
    std::string calibration_id, performance_profile_id;
    std::string request_digest, resolution_digest;
    std::optional<uint64_t> calibrated_request_bytes;
};

class StreamingAuthority { // internal, private constructor
    friend class PublicPresetResolver;
    // Binds artifact, workload, layout, component plan, release/channel policy.
};

struct ResolvedRequestExecution {
    ExecutionPlan model_plan;
    std::shared_ptr<const ModelSourceSnapshot> source;
    std::shared_ptr<const streaming::Descriptor> descriptor;
    std::shared_ptr<const streaming::Layout> layout;
    ResolvedSelection selection;
    std::shared_ptr<const StreamingAuthority> authority;
};
```

`ModelSourceSnapshot` 是建议的跨 adapter 包装，不要求抹平已有 LTX/H3/Flux/Z 专用 snapshot 类型；可以持有 type-erased
verified source lease 和 model-specific metadata。不能为了统一类型再次扫描、转换或复制整份权重。

执行时生成一个带 `request_generation` 的 request owner；compiled snapshot 可以只读共享，但每次执行的 slots、tickets、
cancel flag 和 mailbox 绝不能跨 request generation 复用。P/K 或 source 改变即失效。

`M` 用独立 `calibrated_request_bytes` 报告，保留现有 `memory_estimate_bytes` 的 unknown 语义；不能将经验 M 填入
已有 guard 当成 required resource upper。layout digest、calibration digest 和 authority digest 是三个不同字段。

## 3. Parser、profile 与数值合同

### 3.1 唯一线协议

沿用文档 23 的 `execution.streaming`，**不新增 `execution.streaming.selector` 嵌套层**。
外层 request schema v2、streaming schema v2、catalog schema v1 是三个独立版本。

| 输入 | 处理 |
|---|---|
| 无 streaming | 原路径；不实例化 selector |
| streaming v1/manual | 原 presence overlay 与 validation 不变 |
| streaming v2 + enabled=false | 只允许 schema_version/enabled；原子关闭 inherited selector |
| streaming v2 + selection=memory_tier | 必须 enabled=true、retention=request、target；不允许 stages/preset 字段 |
| streaming v2 + selection=preset | 必须 target/ID/revision/catalog；可选 expected_resolution_digest；不允许 stages |
| 未知 schema/selection/field | 明确拒绝；不解释为 off 或默认值 |

所有 v2 selector 是完整对象，不采用 optional 字段逐项继承。profile 内完整 selector 与 request 内完整 selector 都使用同一 parser。
manual v1→v2 或 v2→manual 时整体替换 streaming 分支；两个 manual v1 仍使用现有 stage-aware overlay。
原 profile 中显式 legacy residency/budget 即使 App 没输出，合并后仍可能冲突：展示冲突，不静默抹掉用户 profile。

### 3.2 bytes 使用 UInt64，但首版 wire 范围有界

建议 wire 接受 `1..9007199254740991`（`2^53−1`）的无小数十进制 JSON integer；内部用 uint64。
8/10/12/16/20 GiB 都在安全范围内。拒绝 bool、负数、零、小数、指数写法、字符串、null 和越界；不支持任意大 UInt64 的
隐式浮点舍入。将来需要超此范围必须版本化 decimal-string 字段，不能静默扩解析行为。

当前 `streaming_config.mm::integer` 是 uint32，只可复用字段存在性/known-key 检查，不可用来解析 target。
所有加法和 `ceil(0.10*T)` 用 checked integer arithmetic；冻结 `H(T)=max(536870912,(T+9)/10)` 时也检查 `T+9` 溢出。
不以 `double` 比较边界、不先把 bytes 除成浮点 GiB 再分档。

### 3.3 重复 key 与输入限制

NSDictionary 解析后的 known-key 检查无法恢复已合并的重复 key。当前 request/profile 入口已经在解析前调用
`reject_duplicate_json_keys`，并覆盖普通重复键与 Unicode 转义后的同名键；新增 selector 必须继续复用这条入口，
不能退回只依赖 NSDictionary。catalog source/builder 仍需实现同等级检查。

另一项未完成规则是 lexical integer：当前 `exact_byte_count` 经 NSNumber/double 判断数值是否为整数，
`1.2884901888e10` 会被归一为合法 12 GiB。若冻结协议要求拒绝指数写法，raw scanner需同时记录目标字段的token类别，
只接受无符号十进制 digits，并在转换前做长度/范围检查。
建议扩展现有 `KeyScanner` 或在同一次 raw 扫描中记录 selector target 的 token 形态，不新增第二遍全JSON扫描。
输入长度/nesting上限在JSON解析前统一检查；off/default请求不应因为 lexical selector 规则增加额外遍历。

冻结初值：新 query/resolve 请求 ≤1 MiB、nesting≤32、preset/catalog ID≤128 UTF-8 bytes、digest≤128 bytes；reject NUL、
非法 UTF-8 和非对象根。catalog 构建器也拒绝 duplicate keys、重复 ID/revision、循环引用、未知 required 字段。
这些是控制面限额，不替换各模型已有 prompt/input 限额。

## 4. Native resolver 的执行顺序

### 4.1 options、resolve、generate 的区别

| 操作 | 允许工作 | 禁止工作 |
|---|---|---|
| options | 已嵌入 catalog 过滤，metadata identity 查询；token 未知可 tentative | 权重 materialization、开 GPU pool、跑 benchmark |
| engine resolve | 验证真实 artifact/workload、轻量 tokenizer、compile exact layout | 开 fill worker、加载 tensor、执行 text encoder |
| generate | 重新核验撤回/source/request、获取资源、执行冻结计划 | 换 preset、改变 P/K/shape、用压力探测结果重新排序 |

`tc_plan_json` 没有 model root，不能生成执行 authority。它对 v2 selector 可以返回 syntactic/recipe validation
和 `resolution_required=true`，其成功不表示 generate 已授权。App 新路径不再把旧 plan-only 成功当作 preset 准入。

### 4.2 两阶段 token 匹配

1. UI 编辑时按 model/shape/steps/device/container 过滤，token 未知仅返回 tentative 候选，不承诺可生成。
2. 正式 resolve 在 native tokenizer 上得到各 encoder 的 valid/padded/compute rows；不执行 encoder forward。
3. tokenizer 无独立低内存入口的模型需先拆出 count/prepare-tokens 接口，否则该模型暂不发布，而非先加载 encoder。
4. exact request digest 绑定 prompt/input 语义和 tokenizer revision；执行时数据未变可复用 token 结果，变更则拒绝 stale。
5. prompt 文本/完整 tokens 不写入可提交的 catalog/evidence；本地任务仍按现有隐私策略保存。

### 4.3 过滤与排序伪代码

```text
validate merged selector and request conflicts
acquire trusted source identity + bounded metadata snapshot
derive exact workload and verified execution container
filter catalog by model/artifact/execution/workload
filter release policy + revocation + compatibility
filter calibration completeness and target fit
if empty: emit typed rejection (no fallback)
sort using frozen rank; deterministic memory/read/ID tie-break
canonical manual = selected preset layout
model_plan = make_plan(request with canonical manual)
descriptor = adapter.describe(source, model_plan)       # metadata only
layout = compile_layout(canonical manual, descriptor)
check layout/component/reader identities against selected record
mint internal authority; freeze ResolvedRequestExecution
```

不要为每个被拒绝的 candidate compile 一次 descriptor；先索引过滤，最终候选复用同一个已验证 metadata snapshot。
resolver 的 unit tests 只用 fake identity/device，不需要 MLX/Metal。record 不完整、identity 不一致就是拒绝，不能 UNKNOWN→PASS。

### 4.4 source lease、缓存和 TOCTOU

受信模型安装的 content digest、文件快照和 reader open-file 检查职责不同：digest 表示校验过的内容，stat/index 表示当前
可见快照，打开 fd 保证引用对象；stat 相同不能证明任意 in-place 篡改未发生。不得把现有 snapshot fingerprint 标成 content hash。

首版 public 要求只读/受管理且已验证的安装，记录 provenance；未知/可变 root 返回 `artifact_verification_required`。
完整 hash 是独立、用户可见的安装验证动作，不由每次 query 隐式触发。普通文件变化使 lease/cache 失效；执行前/读边界复核
已有 file identity。若产品未来承诺抵抗恶意同时写文件，需另做不可变快照或读时内容验证，不能宣称现有 stat 检查已解决。

metadata cache key：verified artifact digest + snapshot identity + descriptor/tokenizer/runtime revision；限定 entry 数和 bytes，
不跨卸载保留 GPU arrays。cache miss 可以重建相同 plan，不允许因 cache miss 重选另一个 preset。

### 4.5 resolution digest 不是 bearer token

engine resolve 返回 exact `selection=preset` 请求片段和 `resolution_digest`。App 保存 ID/revision/catalog/target；可把 digest
放入 `expected_resolution_digest`，generate 比较后再执行。native cache 只是优化，不是权限来源。
进程重启时重新验证相同 ID/revision 的 snapshot；若 digest 不一致，返回 `streaming_resolution_stale`，用户重新确认。

v1 默认参数不变；新 public snapshot 不随 catalog 在执行中更新。撤回在下一次执行前检查；排队任务也要重查。
首版 catalog 随 build 嵌入，不引入自动联网更新。以后做远端 catalog 必须另设计签名、回滚防护和更新事务。

## 5. Public C ABI 与 session 接线

### 5.1 additive API 和内存所有权

建议在已有 C header 中增加两个接口，不另建重复 public header：

```c
int tc_streaming_options_json(const char *query_json,
                             char **result_json, char **error);
int tc_engine_resolve_streaming_json(tc_engine *engine,
                                    const char *request_json,
                                    char **result_json, char **error);
```

保留当前 generate/Swift 所依赖的 status 约定（0成功、2取消、其他失败）；不要本轮擅自重新定义全库 status 3/4。
两个新接口可把 `error` 内容定义为 JSON `{schema_version,code,message,retryable}`；仍由 `tc_string_free` 释放。
generate 的旧 error string 保持兼容，新 streaming 错误以稳定 code 前缀包装；Swift 新 error decoder 同时支持 JSON 与旧字符串。

调用开始把提供的 `*result_json/*error` 设 NULL；成功只返回 result，失败只返回 error；无 C++/ObjC exception 穿过 C ABI。
输入只在调用期有效，native 不保留 char*；Swift 无论成功/失败都 consume/free 两个指针。

### 5.2 Query payload 与线程规则

options 的拟议 payload 为 `{schema_version:1, model, request, execution_container}`；`request` 是 v2 请求结构。
options 可在没有 engine 时返回 catalog-level/tentative 结果；模型路径如需提供只能作本地 metadata hint，不能由其声明 trusted。
实际 container 由宿主入口策略验证，请求字符串不能把 embedded App 伪装成更低基线的 cli_worker。

- options 无 engine lock、无 GPU lease、无 process-global MLX policy 变更；使用只读 catalog snapshot。
- engine resolve 使用现有 engine try-lock；busy 时立即返回 `engine_busy`，不阻塞生成线程，不重置正在运行的 cancel flag。
- resolve 不调用 `configure_streams` 或拿全局 GPU 执行锁；需要有限 tokenizer 工作时在该 engine 的串行队列执行。
- 暂停/取消 UI query 以 draft revision 丢弃旧响应；不调用 `tc_engine_cancel` 去取消另一生成任务。
- 首版 resolve 限定为短时 metadata 操作；需长时间内容校验时返回 verification-required，走单独安装工作流。
- 所有 callback 期间禁止递归操作同一 engine；engine free 仍要求无活动调用。

### 5.3 Session 的新增入口

避免把预编译 snapshot 挂在非拥有裸指针成员后立即释放。建议在 `native/runtime/session.hpp::ModelSession` 增加：

```cpp
virtual RunResult generate_resolved(
    std::shared_ptr<const ResolvedRequestExecution> execution,
    const Event &, std::atomic<bool> &);
```

默认实现抛出 `streaming_public_adapter_unsupported`，不能退回普通 generate。只有本轮四模型显式 override。
原 `generate(Request,...)` 保持默认路径，仍由模型内部 plan；新入口使用已验证 `model_plan`、descriptor、layout 和 authority，
不再调用 resolver/compile_layout。两个入口可以复用模型私有函数，但不能对 off 增加动态 layout 推断或扫描。

H3 的 session-level private bool 仍保留开发路径；public override 验证 authority，不通过把 `allow_experimental_streaming_` 永久置 true。
C API source/model 不匹配、guard 未获独立资格时，在 configure/alloc 前拒绝。

### 5.4 执行与释放顺序

```text
try engine + GPU lock
resolve/verify frozen snapshot → recheck source and revocation
validate public adapter + optional independent guard eligibility
explicit mode-transition cleanup (if needed)
create request owner → configure required backend → create adapter/pools
execute → stop dispatch → join fills → drain all readers/callbacks
release pool/prefix/source → terminal report
```

资源 owner 持有 snapshot、adapter/pager、executor/mailbox 直到安全 drain。取消只改变 request cancel 状态；worker 不碰 Swift。
drain 失败时 quarantine 必须保留这些对象和源 lease，后续请求拒绝；不能用栈上 authority 比异步 callback 先析构。
现有 unsafe executor destruction 可能终止进程，public 发布必须测试实际 session 隔离策略，不声称一定能无损重试。

第一次 implementation 可以继续拒绝 streaming prepare；App 新模式跳过 prepare，也不能先 `load()` 全驻留权重再 generate。
已有 resident cache → streaming 的清理是显式模式切换成本：计入全请求 wall/peak，不能在 benchmark 的计时窗口外藏掉。

## 6. App、Swift 和历史任务：按完整 payload 迁移

### 6.1 不改旧 NativeRequest 编码

`bindings/swift/TurboCiderNative.swift` 拟新增明确 `CodingKeys` 的 `NativeRequestV2/ExecutionV2/StreamingSelectorV2`，不要依赖
Swift enum 的默认关联值 JSON 形状。`StreamingSelectorV2` 自定义 encode/decode 必须生成文档 23 的 snake_case wire schema。

```swift
enum NativeRequestPayload: Sendable {
    case legacy(NativeRequest)
    case v2(NativeRequestV2)

    func encoded() throws -> Data { /* explicit wire encoding */ }
    var modelID: String { /* common semantic accessor */ }
    var outputPath: String { /* never read from current draft */ }
    var inputAssets: [NativeInput] { /* preserve job inputs */ }
}
```

保留 `generate(_ request: NativeRequest,...)`，内部复用既有执行函数；新增 payload overload，仅 v2 分支先 resolve/validate。
不能让旧调用者的默认请求先转换成 v2，再声称“效果一样”。semantic/JSON golden 验收两个 serializer 分别覆盖。

v2 field mapping 必须覆盖 inputs/outputs/sampling/execution/parameters、audio/fps、LTX flags、compile/dynamic_text、noise 和
profile 等原 draft 可表达的语义。新 preset 不支持的字段由 validation 拒绝，不能因 v2 struct 没字段就悄悄丢失。
schema v1/v2 共有 ID、output/input accessor 供 App 用，不给 UI 暴露 native layout mutable 属性。

### 6.2 StudioDraft 的开关与 profile

新增 `StreamingChoice { mode: off|memoryTier, targetBytes: UInt64? }`，model-scoped 保存目标偏好。
旧 draft 缺字段 → off；新字段损坏 → 告知配置问题，不擅自猜档位。streaming off 时完全沿用现有 `request(output:)`。

开启后构造 v2 请求并省略 legacy residency/budget/offload；旧 draft 对应值只保存不发送，关闭时恢复原值。
不要删除 `profilePath` 来掩盖 profile 中的 legacy 冲突。UI 应提示用户显式选择兼容 profile 或关闭该 profile；用户未选择前
保持拒绝状态。独立 memory guard 也不被开关暗改，缺 bounded record 时明确提示不支持组合。

### 6.3 NativeJob 存储格式

现有 `NativeJob.request: NativeRequest` 是迁移重点：仅添加 `presetID` 不能保存 v2 输入和 execution 语义。
推荐自定义 `NativeJob.Codable`，内存使用 `requestPayload: NativeRequestPayload`：

| 持久化格式 | decode | encode / replay |
|---|---|---|
| 历史对象含 `request`、无 job_schema_version | decode 原 NativeRequest → legacy | 保留原 v1 语义，不自动升级 preset |
| 新 off 任务 | legacy payload | 可继续输出原 request 格式，避免无关迁移 |
| 新 streaming 任务 | `job_schema_version=2` + `request_payload` + `streaming_submission` | 输出明确 v2 wire request，不合成假的 legacy request |

`streaming_submission` 包括原始目标、resolve 返回的 exact selector、resolution digest、catalog revision、execution container、
证据展示摘要。执行 authority 不持久化；重启后重新验证相同 snapshot 语义。

需要逐一修改现有 `request.output`、`request.inputs`、`request.model` 访问者：输出删除安全检查、历史参数复用、预览、
routeSummary、LTXWorker.accepts、generate 和 sessionState。输出管理必须继续限制在 App managed outputs，不因 envelope 改造失效。

decode→encode→decode golden 必须保留老 request 的显式 false/0/absent；有新任务的 jobs 文件不得让当前版本静默丢字段。
建议写文件前保留可恢复旧备份并继续 atomic replace；不能启动时破坏无法识别的文件。降级到不支持 v2 的 App 的行为需在发布说明声明。

### 6.4 提交事务

当前 `NativeJobStore.generate` 是单作业，没有完整自动排队执行器。本轮不趁机引入新队列；“冻结任务”也适用于以后排队：

1. 在任何 await 前设置 busy，复制不可变 draft/payload；不要等待 query 时允许第二次提交。
2. 开关 off → 现有 plan/acquire/generate 顺序不变。
3. 开关 on → acquire metadata-only public engine（或明确 worker resolver），exact resolve，显示最终目标/状态。
4. 校验 tentative 与 exact 是否兼容；无可用档立即失败，不加载权重。
5. 持久化新任务及冻结 exact request 后才启动生成；写入失败不启动 GPU。
6. generate 重新验证冻结 ID/digest，记录 actual resolution；取消、失败、quarantine 均写终态。
7. 历史“完全重放”使用 exact selector；“复用参数重新推荐”是显式新提交，可重新选 preset。

若跨进程部署，worker 自己用 public API resolve/verify，不接收 App 发送的 authority 指针。App 可以传 exact request/digest；
worker 返回实际 container/device/source identity。两边 snapshot 不一致则拒绝，不能由 App 冒充 worker 的资格。

### 6.5 Query/UI 性能与显示

新增 `apps/macos/StreamingOptions.swift` 管理只读 query：开关未开启不发请求；开启后以约 300 ms debounce 合并输入变化，
结果附 `draft_revision`，旧 task 返回必须丢弃。不因为 UI task cancel 调用正在生成的 engine.cancel。

query cache key 包括 model/source snapshot、workload、container、catalog/runtime revision；target 改变可在已验证 options 内重过滤。
model/profile/LoRA/backend/shape/prompt 改变使 resolution 失效；临时内存压力只使准入失效，不改 exact selection。

显示 target、预计整请求值/范围、证据 scope、适用设备、实验标签；没有本设备速度证据则不显示预计秒数。
`RunInsights` 不将 `MLX peak=null` 显示成 0；现有 `ResourceMonitor` 仍叫 App RSS，不冒充 worker/tree footprint。
取消失败后 UI 必须区分“可重试”和“需重建 session”，不能统一设置“会话就绪”。

## 7. 四模型具体施工单

### 7.1 共同接入要求

不新增另一套 `PublicPresetAdapter` 执行接口。复用现有 model descriptor/snapshot + `ModelSlotAdapter`，只补 metadata 能力查询
和 `generate_resolved` 接线。每个模型要完成下列闭环：

1. exact snapshot 校验含 source/workload/backend/component policy。
2. 复用 frozen compiled layout，不调用 legacy budget→prefix heuristic。
3. 保持同一 kernel、计算顺序、reader revision 和质量，不借 preset 开启 approximation。
4. 全请求 live order 覆盖 text→denoise/refiner→VAE/audio/export→cleanup，而非只看 denoiser。
5. 真实 metrics 输出 actual P/G/K/D/Q、pool 数、prefix/source/fill bytes、authority origin。
6. 模式切换、取消、I/O 错误、drain/quarantine 独立测试，详见 26。

### 7.2 LTX 2.5 distilled

修改接缝：`native/platform/apple/ltx_session.mm`、`native/models/ltx_runtime/ltx_streaming_descriptor.cpp`、
`ltx_streaming_plan.cpp`、现有 C bridge/slot hooks。候选锚点为 P8/G1/K3/D2/Q3，详见 24 第 4 节。

- 在 existing exact metadata resolution 外层增加 public snapshot 消费入口，private/manual 仍保留原行为。
- 初轮 G1；prefix 上界按当前 compiler 的 `48−P >= K` 条件检查，P≥1；不是 `P<=48−1` 即总是有效。
- K2/K3 是不同布局，D 必须合法且固定，不能 target 低时先分配 K3 再偷偷减成 K2。
- 两阶段、conditioning、upsample 和 audio/VAE 的源身份/释放边界全部入 component policy；AV 阶段不得拆丢某支权重。
- request retention 不能复用 legacy retained cache 的隐式状态；cold/warm 命名与实际 cache 生命周期对齐。
- `LTXWorker.accepts` 当前依赖 legacy component_staged，新 public route 使用明确 container policy，不能伪造 residency 触发。
- 首批 full card 是 512×320×33、11 steps；97 frames 单独校准。不得将 tiny/P1 证据升级为任意视频长度支持。

### 7.3 Z-Image Turbo

修改接缝：`native/models/z_image/z_image.cpp`、`weight_stream.hpp`、
`native/platform/apple/z_image_streaming_descriptor.mm`、现有 Z session。

- 首轮固定 K2/G1/D0/Q1、claim-overlap，按文档 24 的 P candidates 探索；P29 对 30-block/K2 不合法。
- shape 与 valid/padded/compute text rows 都由原 tokenizer/模型几何规则产生，不再实现 App 版本。
- legacy streamed sampling budget 保持原路径；new preset 不调用该预算派生器。
- VAE 前释放 DiT 的策略若改变，要新增 component-policy revision，重新测 full request，不沿用 P1 的 retention identity。
- 原始/分片 BF16 也需校验 source layout 和 reader identity，GGUF 不自动获得资格。
- 首批卡包含 512² 和 1024²、9 steps；当前 1-step P1 仅证明冻结 tuple 的 executor 开销。

### 7.4 H3 Turbo original BF16

修改接缝：`native/platform/apple/h3_session.mm`、`native/models/h3_runtime/h3_streaming_descriptor.cpp`、
`h3_streaming_policy.c` 及实际 source/pass hooks。

- 只做 `minimax-h3-turbo` original BF16；不因模型名字前缀相似接普通 H3、VDN、FastH3/INT6。
- 保留 K2/G1/D1/Q1、`carry_first_group` 和当前大矩阵 pread 路径；不顺手改成 8 MiB 切读。
- 所有 P 变体单独检查 suffix group/carry，特别是奇数 suffix；每 pass 仍每个 suffix group 一次 fill，不宣称省掉一次 I/O。
- authority 消费由 session 显式支持；private 环境变量不是 App 权限。
- full card：512×512×22、4 steps；frames 必须满足当前模型 `5+17n` 且 22..362。39/73 frames 分开确认。
- full pipeline 包括文本、DiT、VAE 和终态视频；denoiser probe 只作定位。bootstrap INCONCLUSIVE 保留原状态，工程接受
  可以记录 release exception，但不能输出 statistics PASS 或自动升 stable。

### 7.5 Flux 9B：先保 P0，再做 prefix

修改接缝：`native/models/flux2/flux_streaming.hpp`、`flux_transformer.cpp`、`pipeline.cpp`、
`native/platform/apple/flux_streaming_descriptor.mm`、`native/runtime/streaming/mlx_weight_pager.*`。

当前 descriptor execution 校验与 direct benchmark executor 均限制 P0。`min_prefix=0` 只是下界，不意味着执行已支持 P>0。
第一批 public 校准可以只用现有 P0/G1/K2/D1/Q2、retain_all；Q 变化主要影响 I/O concurrency，不保证形成不同内存档。

#### 7.5.1 Prefix pager 接口

不引入未定义的第二份 PrefixLayout。使用已有 `StageDescriptor.blocks[0..layout.prefix)` 与编译器的 prefix byte/read closure：

```cpp
void MlxWeightPager::load_prefix(const std::atomic<bool> *cancel);
const Weights &MlxWeightPager::prefix_weights(uint32_t block_id) const;
```

实现要点：

1. `State` 新增按 descriptor block ID 索引的 `PrefixBlock {arrays,pointers,bound_weights}`；与 stage `resident_fields` 分开。
2. owner 在 setup 中逐 prefix block 分配 BF16 arrays、读 source、预建 binding；cancel/error 时只清理已完成的对象。
3. 不把全部权重先加载进一个 Weights 再从中取 prefix；真实分配只覆盖 resident+prefix+suffix pools。
4. prefix arrays 不接受 refill ticket，worker 不得写；GPU reader 在最终 drain 前可能依赖它们。
5. 新增 prefix source-read/materialized bytes、array counts；与 `prefix_source_read_bytes/prefix_bytes` 逐项核对。
6. pass 内 prefix_weights 查询不构造 map/vector、不新增 MLX backing；模型运算的 activation 分配仍与 framework allocation 分开。
7. P0 完全跳过 prefix allocation；保留原 frozen P0 direct baseline，不为测试“方便”改其定义。

#### 7.5.2 统一 block 执行与 concatenate 边界

全局 block index：dual `0..7`，single `8..31`。把当前 `prepare_group(block==8)` 内 concatenate 抽成边界 helper，
prefix 与 suffix 共用，但每 pass 只发生一次：

```text
begin_pass: next_block=0; single_started=false
execute_one(block, weights):
    require block == next_block
    if block == 8:
        require !single_started
        image = concatenate(context, image)  # 同原顺序与维度
        single_started = true
    if block < 8: run existing dual kernel + original eval boundary
    else: require single_started; run existing single kernel + original eval
    next_block++

encode_prefix(pass): execute_one for block in [0,P)
encode_group(group): execute_one for each compiled suffix block (G1 first)
end_pass: require next_block==32 && single_started
```

P8 时 prefix 只执行 dual，concatenate 在 suffix 首个 single 发生；P12 时在 prefix 内 block8 发生，suffix 首个 block12
不得重复拼接。已有框架可以在 encode_prefix 前启动首批 suffix fill，继续利用 overlap，不增加全设备同步。
此处 `encode_prefix` 是每 pass 执行 prefix 计算，**不是每请求只执行一次**；prefix 权重则只加载一次。

#### 7.5.3 池消除与有效边界

| P | prefix | suffix pool | 首轮资格 |
|---:|---|---|---|
| 0/2/4/6 | 无/部分 dual | dual K2 + single K2 | 分别实测 |
| 7 | 7 dual | dual 仅1组，不足K2 | 拒绝，不减槽 |
| 8 | 全 dual | 仅 single K2 | 需新 pager/adapter 测试 |
| 12 | 全 dual +4 single | 仅 single K2 | 需新测试 |
| 30 | 全 dual +22 single | single 恰2组 | 边界单测，不作为首轮推荐 |
| 31/32 | suffix不足/全resident | 不符合当前 streamed K2 | 拒绝；不偷偷变 resident |

compiler 应只为实际 suffix class 建 pool；P8 不得保留空 dual pool。adapter 当前 `active_pool_<2` 等假设需改为已编译 pool ID
集合，不能把 class enum、pool ID、vector index 混用。pool bytes 随 P 不一定单调：消除 dual pool 会产生台阶，必须测实际总峰值。

`retain_all` 新 prefix variants 可追求稳态 framework backing/thread 0/0；`serial` 双池切换需独立 adapter 支持，
可能每 pass 有 allocation，不可把它的策略收益写成现有同布局 P1 PASS。当前 direct executor 只支持旧 P0 tuple：新 P 的 P1
若无对应 reference，要明确 NOT_RUN；不能与 P0 direct 比较后叫 same-layout。

## 8. Scheduler 的保持项与可扩展边界

本轮 catalog 不改变 `StageExecutor` 的 owner-pump、ticket generation、slot state、mailbox、multi-class barrier 或 reader 合同。
Q 是 worker 数，D 是 dispatch distance，K 是每 class pool 的精确 slot 数；Flux 两个 K2 retained pool 总共四份 slot backing。

- 已有 `overlap_next_fill_after_claim()` 继续由 adapter 显式声明，不能为了“异步”提前回收仍被 GPU 读取的 slot。
- worker 只 pread/填已分配 backing；不创建 MLX arrays、不执行 kernels、不回调 App。
- I/O job、mailbox 和 in-flight tickets 有界；enqueue 失败按当前异常路径停止，不能临时扩容把 K 约束绕开。
- prefix 和 stage resident 权重是 plan 中显式的额外 backing，不能计入 slot 后又在 memory report 重加一次。
- 跨 pass carry、ordered class barrier 和 retention 都进入 layout identity；preset 修改任何一项就新 revision。
- 不增加每 block catalog lock、device query、JSON serialization、hash 或密集 memory sampling。

future G>1、异构 field conversion 或跨 component prefetch 另开 adapter/reader revision 与验收；本次先完成 G1 子集。
能力“不足”是拒绝未知请求，不是框架不通用；通用性来自相同合同和工具，而非一次开放所有参数组合。

## 9. Catalog 构建、结果与发布策略

### 9.1 文件职责

| 建议文件（新增） | 职责 |
|---|---|
| `native/core/streaming_selector_contracts.hpp` | wire intent，不依赖 MLX |
| `native/runtime/streaming/preset_catalog.hpp/.cpp` | read-only record/index、兼容性和撤回 |
| `native/runtime/streaming/preset_resolver.hpp/.cpp` | exact 筛选/排序、authority 构造 |
| `native/runtime/streaming/resolved_request.hpp` | snapshot 所有权；不加入第二执行器 |
| `resources/streaming/catalog-source.json` | 经审核 build input；不是用户 profile |
| `tools/native/build_streaming_catalog.py` | 严格校验后生成 embedded table；默认仅 proposed |
| `apps/macos/StreamingOptions.swift` | opt-in query/presentation state |

实际 build 接入 `tools/native/build.sh`、`build_app.sh` 和 `Makefile`；CI 验证生成结果与 source digest 一致、索引排序确定。
小 catalog 可用 sorted vector/binary search，不必增加数据库/网络依赖。普通 profile、result JSON 和外部 catalog 不授予 public 权限。

### 9.2 Result 的新增可选字段

在 `RunResult/results.mm` additive 输出；默认请求省略新对象，保持旧字段语义：

```json
{
  "streaming_selection": {
    "selection_source": "public_preset_registry",
    "preset_id": "illustrative.zimage-bf16-k2-p14",
    "preset_revision": 1,
    "catalog_revision": "illustrative-catalog-r1",
    "resolution_digest": "illustrative-not-a-real-digest",
    "target_request_memory_bytes": 12884901888,
    "memory_enforcement": "none",
    "execution_container": "embedded_app"
  },
  "streaming_memory": {
    "scope": "execution_process_tree_v1",
    "estimate_kind": "observed_calibration_not_bound",
    "calibrated_request_bytes": null,
    "observed_request_peak_bytes": null,
    "coverage": "not_sampled",
    "peak_phase": null
  }
}
```

这是结构示意，不是可发布结果：真实 public record 必须有 complete calibration，不能使用上述 null 来通过 resolver。
日常执行未开启 sampler 时 observed 为 null 合理；catalog calibrated 与本次 observed 必须清楚分开。
actual `StreamingRuntimeMetrics` 与 compiled snapshot 一致才成功；出现 adapter 自改 P/K/retention 时返回 internal mismatch。

### 9.3 错误与 gate

文档 23 的错误码之外补充：`artifact_verification_required`、`streaming_resolution_stale`、`streaming_public_adapter_unsupported`、
`engine_busy`、`streaming_quarantined`。默认错误行为不变，新 UI 不用中文错误字符串做逻辑判断。

首版 public gate 来自 embedded catalog 的经过审核记录与 native release policy；不要新增请求字段或通用环境变量来一键绕过。
test-only registry fixture 必须编译隔离，release 不导出新测试 authority setter；现有 candidate 测试入口的隔离状态据实检查，
不能把“private API”误写成当前一定不存在导出符号。App 禁止引用它，并通过 ordinary constructor 做正反测试。

## 10. 默认性能防线与合并约束

1. off 先分支后访问任何 catalog singleton，避免 static initialization 改变默认 engine startup。
2. off 不 tokenize/descriptor/compile twice，不新增 memory sampler thread，不增加 cache clear/unload。
3. 新 report 字段只为 active selector 构造，off 不每 block 判断 public string。
4. 编译后 inspection 测试验证 default 五类 streaming audit 计数仍全零；watchdog/probe 另单独计数。
5. resident↔streaming 转换只在明确 mode boundary；异常路径不能让后续 off 请求残留 authority 或错误 allocator limits。
6. release timing 不编入密集 audit hooks；P0 验证默认冷/热、构造时间和模式切换后恢复性能。
7. 新 symbol/类型尽量放 opt-in translation units；不要把统一 allocator、默认 kernel 优化或全库 request 重构捆绑到 preset PR。
8. API/Swift/parser/metadata可以先合并且 production catalog 为空；发布记录必须最后独立审阅，关闭记录即回滚该 preset。

完整测试数据合同、PR 完成门、基准与发布 checklist 见 [26](26-public-preset-acceptance-and-release.md)。

# 10 · 解码执行、生命周期与产品接入合同

[目录](README.md) · [首发范围](09-release-scope-and-component-contracts.md) · [验收](11-acceptance-profiles-and-feasibility.md)

2026-09-30。以下接口/字段是待实现规范，不是已存在的 CLI 或 ABI。本页冻结 CPU 首版
的同步完成语义，并为 GPU decode/ANE 留独立能力版本；不要求把三种完成模型挤进同一个函数。

## 1. 入口：一个配置来源、一份不可变 plan

拟在 schema2 request 的 `execution.quantized_execution` 添加独立 versioned 对象。
缺省不启用，保留旧 GGUF/resident 行为。以下仅为 execution 片段：

```json
{
  "quantized_execution": {
    "schema_version": 1,
    "enabled": true,
    "mode": "bounded_dequant",
    "source_residency": "packed_resident",
    "decode_backend": "cpu_simd",
    "precision_profile": "z-source-mixed-v1",
    "granularity": "layer",
    "prefetch_layers": 1,
    "persistent_dense_layers": 0,
    "oversized_layer_policy": "reject",
    "ane_compute": "off",
    "allow_requantization": false
  }
}
```

这是请求的 DiT 配置；R2 增加显式 per-component profile 绑定，encoder 不能继承
`z-source-mixed-v1`。外部 schema 版本与 decoder/descriptor/graph revision 分别管理。
`enabled=false` 时只接受 version/enabled，拒绝其他执行参数，避免用户误以为已生效。
未知字段/enum 拒绝，不用拼写近似或环境变量补全数学策略。

### 1.1 与已有配置的关系

| 输入 | 决定 |
| --- | --- |
| `quantized_execution` + manual `streaming` | 允许，但先逐项验证一致；不创建第二个 scheduler |
| `granularity=layer`、前瞻 p | R1：G=1、P=0、K=1+p、D=p、Q=1、retention=request、pass transition=reload |
| 手动布局 K/D/G/P 与上述不一致 | `qe_config_conflict`，不静默改任意一边 |
| 未提供 manual 布局 | adapter 按上述唯一规则生成各 stage 的固定布局；不是按可用 RAM 自动搜索 |
| 旧公开 streaming selector/preset | 不与新手动配置混用；R3 的量化 preset 必须携带完整新 recipe/tuple |
| `memory_constrained` | 唯一 whole-request guard/admission 来源，沿用 limit/buffer/min-free 语义 |
| legacy residency/budget/offload/quantized_cache | R1 拒绝显式混用，不能把旧 working-set hint 当总预算 |
| LoRA、ANE manifest、自动 hybrid 路由 | R1 拒绝；以后按新资格显式开放，不静默忽略 |
| `compile_gpu` | R1 false；新增图的优化阶段显式 true，并通过 profile 和新版能力校验 |
| `allow_approximation` | 不为已选 GGUF 文件再次索取低比特授权；额外 A8、小 tensor 降精度、逆旋转改变数值路径需 true |
| `allow_requantization` | 默认 false；重新 W8 必须它与 allow_approximation 同时为 true，且 recipe 明确 |

manual 路线必须有新 exact adapter qualification 才能进入正常 generate；开发 probe
使用独立实验能力，不伪造 public catalog 资格。旧公开入口目前排除 quantization、ANE、
compiled GPU，应新增版本化分支，不修改旧 v1 的允许集合来绕过审核。

这里的 K=1+p 是新 layer 模式的精确映射，不改写通用 streaming 中 D=0/K>1 的
合法语义。异构 layout-class 的池在 barrier 先释放再建；stage 若不足 K 个 group，
按已有规则拒绝，不能暗中减少 K。很小的固定 stage 可由 descriptor 明确设为 resident。

`max_prefetch_layers` 不作为首版另一控制旋钮：capability 公布最大值；R1 支持 p=0/1，
R3 扩展到 2。`persistent_dense_layers` 首版只允许 0。tile 是 R3 的新 descriptor/
配置 revision：使用 `prefetch_units`，不把“提前两个投影 tile”报告成“提前两层”。
03 的自动 2→1→0 搜索仅属于后续 planner，必须发生在新请求的 resolve/admit 前。

### 1.2 接入顺序

```text
parse/version check → bind component/source leases → directory + adapter validation
→ describe execution units + decode resources → compile immutable layout
→ qualify exact tuple / reserve and admit → stage prepare → execute → drain → receipt
```

plan-only 只描述和检查，不分配模型 tensor、不调用 predict、不下载模型、不编译 ANE。
首次 content verification 是显式准备动作；缺原生 proof 时 report `verification_required`。
plan-only 不能靠临时关闭验证获得可执行资格。

完整 plan digest 包含 source artifact identity、mapping、所有小 tensor dtype、decode
recipe、精度/近似授权、stage/unit 顺序、容量/对齐、source 驻留、图版本、退路资源和 guard。
执行前重新核对 source generation 与 device qualification；同 shape 不足以复用内容。

## 2. 数据接口：把 read、decode 和 publish 分开

02 的 `PackedTensorView` 是逻辑 source descriptor，不应按值复制不可复制的 SourceLease。
具体实现持有 `shared_ptr<const SourceLease>` 与 generation；不序列化 fd/pointer。
建议在 `native/core/gguf_decode.*` 定义纯 CPU kernel 合同，在 typed pager 中组合 I/O。

```text
describe_decode(source, logical_slice, target_dtype, layout, transform)
  -> DecodePlan { read_spans, target_layout, scratch_upper, implementation_revision }

decode_cpu_into(ResidentPackedSpan[], DecodePlan, MutableTarget, ScratchLease, cancel)
  -> DecodeReceipt                     // 同步；函数结束后不再持有/写入这些 spans

submit_decode_gpu(PackedLease[], DecodePlan, TargetLease, ScratchLease, stream_owner)
  -> DecodeSubmission { completion, keepalives }   // 后续能力；不是 CPU 函数的替身
```

`describe_decode` 无 payload 大分配；读取范围和 scratch 上界由 checked geometry 推导。
CPU decoder 不自行 pread、不创建 MLX tensor、不提交设备工作、不调用 allocator 扩容。
pager 按 read plan 将源分块读入固定缓冲，再调用 decoder；cancel 至少在每个有界 chunk
和发布前检查。CPU/GPU 不能偷偷调用另一后端；实际后端写入 receipt。

### 2.1 字段与错误合同

| 对象 | 必须冻结的字段/规则 |
| --- | --- |
| Source | artifact/content identity、tensor ID/type、磁盘/逻辑 shape、block geometry、合法 payload range |
| Slice | row/column 半开区间；gather 为有界 row-index 列表；重复 row 输出语义明确 |
| Target | storage identity、generation、capacity、dtype、shape、byte strides、有效写入 spans、alignment |
| Scratch | 每 worker 最大 packed bytes、FP32 block/row bytes、转换 bytes；slot 外也须 reservation |
| Receipt | task/ticket/revision、源逻辑 bytes、实际提交读取 bytes、written bytes、scratch peak、dtype、status |
| Error | code + tensor/slice/task + expected/actual；保留 primary error，cleanup error 单列 |

R1 仅支持已验证的非负、不重叠 target strides；zero/negative/越界写跨度在运行前拒绝。
decoder 只能写分配给它的 spans，guard bytes/padding 行为由测试验证。源和目标不允许
有读写重叠。两 decoder 写同一 backing 的 disjoint spans 也需 scheduler 显式批准。

GGML block 对齐约束用于**源 tensor 的连续维**；文件 K 不满足该 type 的完整 block
规则时拒绝。合法 tensor 的 slice 可不对齐：读覆盖它的完整 blocks，仅输出请求元素。
M/N kernel tail 与非法磁盘 block tail 是不同问题，不能补零修复损坏文件。

GGUF source→BF16：FP32 oracle 重建，单次目标 RNE。ConvRot inverse recipe 的 FP32
旋转 scratch 必须有界，不先创建整个 FP32 weight。registry 显式区分 raw GGUF、MLX
affine 与 ConvRot；不得继续靠一个 `AffineView` 猜三种格式。

## 3. descriptor 与现有 streaming 的组合

复用 `FieldSpec`、`Materialization`、`SourceRange`、`SourceLease`、ledger 和 slot tracker。
新 typed pager 不放宽 `MlxWeightPager::direct_source` 的 copy-BF16 合同。

| 现有信息 | 新 typed materialization 所需补充 |
| --- | --- |
| `FieldSpec.bytes/alignment/storage_id` | 解码后的 destination 容量；不可填压缩源 bytes |
| `Materialization.reads` | 原 GGUF 文件范围，可多个 block spans；逻辑 I/O 统计独立 |
| `format/storage_mode/conversion` | 例如 target BF16 / MLX shared / `gguf-q8_0-to-bf16-rne-v1` |
| `derived_from/derived_offset` | 仅沿用既有 alias/span 含义，不伪装成通用算子依赖图 |
| adapter revision / workload | 绑定 typed decode plan digest、源 block type、transform 和 target layout |
| required sites / live intervals | 加入 read buffers、worker scratch、decode 输出、编译 scratch 和失败退路 |

第一版在 adapter 的不可变 typed decode plan 保存 recipes，由 descriptor 绑定完整 digest；
canonical encoding 必须覆盖 recipe 内容，不能只存一个可碰撞的人类标签。若需扩展公共
descriptor 字段，则升级相应 codec/ABI 并保留旧版本拒绝不认识字段的规则。

## 4. 所有权、完成时序与 lazy graph

| 资源 | owner / 使用者 | 允许复用或释放的时刻 |
| --- | --- | --- |
| packed-resident source | request/stage source lease；CPU 或 GPU reader | 全部 reader 完成、阶段结束；不得只擦掉字典 |
| packed read buffer | fill task 独占 | CPU decode 返回；GPU decode 则等 decode reader fence |
| CPU scratch | 有界转换 worker 独占 | 同步 decode 返回，失败也先 join |
| dense slot | request owner；GPU compute 读 | tracker 收齐最后 reader fence 后才 Vacant |
| ANE input/weight/output | runtime request lease | prediction 结束且后续 GPU output consumer 完成 |
| conditioning/tapped hidden | 独立输出或显式 lease | 最后下游消费者完成；不依附将被重写的 slot |

CPU 首版：owner 分配 backing/MLX view → fill worker pread/decode → release-publish
completion → owner acquire/验证 generation/bytes → Ready → bind/submit compute。
decoder 失败后目标可含部分数据，但绝不 Ready；队列有固定容量，优先当前 demand。

GPU 后续版保持 Loading，直到 owner 收到 decode-complete 再发布 Ready。这样先复用
现有 CPU-visible Ready 语义，不用“已提交”偷换“写完”；若未来用跨 stream device event
避免 host wait，必须新 revision 定义 dependency-ready，并证明每个消费者都等待它。

MLX view 只指向槽容量，不能使编译图把可变权重捕获为常量。图 key 按 shape/dtype/
layout/math revision，不含 layer ID 或 backing 地址；内容 ticket 必须包含这些运行身份。
重复执行同 shape A/B/A 权重，输出随内容变化，稳定阶段编译次数不得继续增长。

### 4.1 Qwen3 的强制提交边界

新 bounded adapter 使用 `submit_each_layer`，每层必须先把读取当前权重的 lazy 计算
提交并登记最后 reader，再申请复用相应槽。首个正确性版允许每层 `mx::eval`；以后用
经验证的异步完成 token 降低 host wait。不能依赖原有四层 eval interval 来推进双槽 pager。

在新模式中，显式 `TURBOCIDER_QWEN3_DEFER_LAYER_EVAL` 或与此策略冲突的 interval
返回配置冲突；不悄悄忽略用户诊断设置。resident 原路径不改。tap 输出单独物化/持有，
不能为了释放权重把需要返回的 hidden 丢掉。原 `const Weights&` 全模型接口须拆成
逻辑 model metadata + per-layer binding/gather provider，不能后台插改全局 Weights map。

### 4.2 取消和无法 drain

停止 dispatch → cancel/join fillers → 不再发布 partial Ready → 提交/等待已声明 readers
→ owner 消费 completion → revalidate source → release。异常不能跳过其中有活动 reader
的步骤。超时、GPU fence error 或不能证明 prediction 已结束时 quarantine；不接下一请求
覆盖它，也不启动会再占一套预算的“无条件 GPU fallback”。

## 5. 分片执行和失败重算

R3 的 `ExecutionUnit` 使用 `(component, pass, step, layer, projection, tile)` 身份。
layer 模式每层一个 unit；tile 模式以新 descriptor 编译单位顺序和资源 live intervals。
不能将 tile 序号塞进现有 step/layer 字段冒充普通层。01–03 的 layer 前瞻统计只用于 layer 模式。

FFN 的安全 channel-tile 程序固定为：

```text
retain X and create one budgeted FP32 output accumulator
for channels J in deterministic ascending tiles:
    decode Wgate[J,:], Wup[J,:]
    submit gate/up → materialize hidden_J; retire gate/up weight readers
    decode Wdown[:,J] into a reusable slot
    compute partial_J → accumulate once → retire tile readers
add down bias once; cast output once; publish completed FFN
```

target scratch 可复用，但只有各自 fence 后。完整 `[M,H]` accumulator、`[M,|J|]`
gate/up/hidden、保留 X 和后续 attention workspace 都计费。若权重 tile 很小而激活
floor 仍超预算，拒绝；不能暗中 activation spill、缩 M 或少 tokens。

保持旋转域的 ConvRot down 必须按完整 H256 组取 J；gate/up 的 K 部分和不能逐块
施加 SiLU。K tile 归约顺序、FP32 accumulator 与原 GEMM 不同，使用独立 numerical profile。
attention 若逐投影准备，需要保留完整算法所需 Q/K/V，不通过 token-row 分割破坏全局 attention。

ANE/GPU 分支结果采用 transactional publish：保存 FFN 输入和 join 前状态，只在所有
chunks 成功后发布。某 chunk 失败时丢弃未提交 partial，drain 后 GPU 重算整个该 FFN
（首版恢复单元）；禁止把已累加结果再加一次。后续缩小到 chunk 级恢复需新协议。

plan 对 normal 与 recovery 分别求 live-interval 峰值再取 max，不是只预算 normal。
优先 clean drain、释放不再使用的 ANE resources 后复用容量；若 driver 仍 retained 或
无法释放到既定 recovery floor，报告失败，不临时扩预算。所有 fallback 也受 dtype/质量合同约束。

## 6. 图缓存、编译与阶段转换

首版不在请求内自动调参。shape、tile、profile、backend 由不可变 plan 决定；采集数据
供下一请求选路，不把测时噪声变成层间数值变化。新 compiled segment 的首次编译有
单独 bounded scratch/envelope 和 cold 时间；不能把 warmup 放在计费区间外隐藏峰值。

图 cache 容量属于 stage bank，按计划限定；跨层只换参数，不复制一张图一层。
encoder→DiT：先保证 conditioning 独立存活，drain encoder、释放其 weights/pool，
再构建 DiT 资源。是否保留 metadata/shape graph 由 live interval 决定，不默认 retained
session。VAE 同理；p=1 仅限制权重预取，不保证 VAE/encoder 的峰值天然合格。

## 7. 产品工作包 P4 与稳定错误码

P4 明确覆盖以下位置（以实际 schema/桥接所在文件为准，不另建绕过校验的入口）：

- `native/core/contracts.hpp` 及 request/profile parser：typed config、strict enum、冲突测试。
- `native/runtime/streaming/{public_request_validation,public_runtime,resolved_request}.*`：
  新版能力分支、request digest、resolve/admit/execute 一致性。
- model adapter 的 `streaming_source_files`、probe/describe/compile/bind：GGUF 来源闭包与阶段资源。
- `canonical_encoding.*`、`preset_catalog.*` 和发行 catalog：新增 identity，只有真实认证才注册。
- CLI/API/App bridge：先报告 plan/不支持原因，再按具备资格的 tuple 开放选择；不提前展示硬件加速标签。

新增错误类别至少区分：`qe_config_conflict`、`qe_type_unsupported`、
`qe_adapter_mismatch`、`qe_source_changed`、`qe_decode_invalid`、`qe_budget_floor`、
`qe_envelope_unknown`、`qe_capability_unqualified`、`qe_drain_unproven`。
这些名字是拟议稳定码；具体 payload 含 stage/tensor/unit/required/available/reason，
不能只返回一句 OOM。用户不能注入 manifest、unknown=0 或 `qualified=true` 授权自己执行。

P4 的负例必须验证旧请求零行为变化、新配置未资格时 fail-closed、plan 与 generate
使用相同 snapshot，以及每一个上述冲突和错误分支。R1 先交实验 probe 闭环，正式 public
入口在 R3 的 qualification/capacity 证据完成后发布，不能把这一步遗漏在“decoder 完成”之后。

# 15 · Adapter 施工单：从现有源码到真实模型

[目录](README.md) · [模型范围](04-model-adapters.md) · [整合方案](14-layout-first-integration.md) · [统一验收](12-acceptance-playbook.md)

日期：2026-09-16。以下是代码修改建议和验收要求；部分LTX内部adapter已实现并通过tiny实模型parity，范围见 [13](13-implementation-progress.md)。尚未发布新的公共模型路线。
所有新增符号若未明确标“现有”均为建议名。首发 GPU-only，不改变 sampler、precision、block 数或默认 fusion。

## 1. 共用接入包：每个模型交相同六件东西

| 交付物 | 最小内容 | 拒绝条件 |
|---|---|---|
| capabilities | format、shape分支、layout class、K/G/P/D/Q、合法边界 | 未覆盖分支不能用默认值猜测 |
| describe | metadata-only tensor/source/backing/access 描述 | 必须加载权重才能估算则仍plan-only |
| construction view | 对同一resolved plan的模型结构投影 | 不得二次调用budget→prefix |
| adapter ops | allocate/fill/prepare/encode/drain/destroy | 不得再建自己的loader线程 |
| request binding | step输入、组件交接、取消、actual结果 | 不能只跑denoise不覆盖正常请求路径 |
| evidence | 相同数值路径的质量、lifetime、P0/P1 | 合成测试不能代替实模型 |

复用现有 `tc_stream_adapter_v1` 回调，不用改动 public `turbocider` C ABI 来传单个 K/P 参数；请求继续用 JSON。
底层模型 exact create 可新增 versioned API，保留旧 options 内存布局和入口。

### 1.1 每个回调的权限

| 回调 | 线程 | 可做的事情 | 不允许 |
|---|---|---|---|
| allocate_slot | owner、一次setup | 按field计划分配bundle并核对总容量 | 读取完整模型、重选slot数 |
| fill | Q workers之一 | ranged read、确定性转换、写当次spans、查cancel | 新建GPU context、扩slot、修改全局错误缓冲 |
| prefix | owner、每pass | 对resident prefix执行原有计算 | 重复加载prefix、偷偷跳过prefix |
| prepare | owner、Ready之后 | timestep派生表/绑定 | 改写其他InUse槽 |
| encode | owner | 原kernel提交，返回所有queue最后reader | encode返回即假完成、未commit即等reuse |
| drain | owner | 证明GPU及回调不再访问借用对象 | 失败后仍释放/返回session到idle |
| destroy_pool | owner | 释放已实际创建资源，支持partial create | worker/回调仍存活时free |

当前 C bridge 把 fill callback 的1成功转换为内部0成功；其余 C callback 同样1成功。统一封装，不能把返回约定混用。
异步 encode 必须复制 completion sink 的值和 ticket；sink 指针本身是调用栈地址，不能保存该地址给回调。

## 2. LTX：按现有 helper 切开 allocation 和 fill

主要修改范围：`native/models/ltx_runtime/ltx_blocks.c`、`ltx_native.h`、`ltx_gpu.h/.m`、`native/platform/apple/ltx_session.mm`。
先在现有 C 文件内部增加窄 helper，复用 static 数值函数；不要为“模块化”先复制整套 attention/MLP/kernel。

### 2.1 Metadata 与构造计划

已有 `ltx_stream_describe_block()` 位于 `ltx_streaming_layout.*`；复用它，不再实现同义metadata reader。
当前 `ltx_streaming_descriptor.*::StreamingMetadata::describe()` 已完成generic source/field/workload/pass投影；
下一步把该owned snapshot与实际构造/session统一，避免下层独立打开并再解析另一份header。下列仍是完整接入要求：

1. 读取 safetensors header/index 与必要的小型量化 metadata；验证 offset+length、shape乘积、dtype 和 alignment。
2. 按 `load_block_weights()` 的实际访问枚举六组 attention、两组 MLP、norm/gate/scale、parameter tables。
3. 复用 linear metadata resolve 的格式判断；不得调用 `load_linear()` 或 `load_block_weights()` 做 describe。
4. 描述每个 tensor 到 source range 的关系，以及真正创建的 CPU/GPU backing；记录两stage可共享/不可共享的字段。
5. prefix 与slot分别投影同一份 metadata；只有被证明布局兼容的block才放同一个class。

必须有单独的 table 账：当前 `load_table()` 从 F32 临时数组转换为 CPU BF16 `base_values`，再创建 GPU BF16 row buffers。
`scale_shift_table` 等还会被多个表实例重复加载；source相同不等于backing相同。每个实例单独计 CPU base 与 row 容量。
table source bytes、转换 scratch、最终 backing 各自计数，不能用 checkpoint 文件大小代替驻留上界。

先保留 P≥1/G=1；当前 workspace 使用 `weights[0]` 的geometry，P=0需单独拆metadata引用后再开放。
exact路径不能先加载block0再调用 `tc_block_residency_plan_build()`；后者保留给legacy。

### 2.2 Allocation/fill 改造建议

| 现有函数 | 新分支建议 | 保护legacy的方法 |
|---|---|---|
| `load_linear/refill_linear` | metadata构造gpu_linear；fill只写weight/scale/bias现有span | 旧load仍可调用原流程 |
| `load_attention/load_mlp` | 分配projection/norm/gate后生成稳定binding | 不改变run_block参数/算子顺序 |
| `load_table/refill_table` | allocate base/row；`refill_table_into(..., scratch)` 无临时malloc | 新helper供新adapter；旧refill行为先不变 |
| `load_block_weights` | 新 `allocate_block_bundle(view)` 与 `fill_block_bundle(read_plan)` | 不强迫旧loader换实现 |
| `block_resident_bytes` | debug/audit核对实分配与descriptor，不作为前置估算 | 不改旧报告口径；新增source/capacity分列 |

scratch 有两种合法实现：

- 首个 C adapter 可按 K 预分配每槽scratch，fill凭ticket.slot取专属区；简单、无worker ID依赖，但按K计内存。
- 要降到 Q 份scratch，IoExecutor需先支持显式worker context/ID租用，固定按worker预分配；完成独立ABI/并发测试后启用。

当前 `fill` ABI没有worker ID，**不能一边声称仅Q份scratch，一边实际按K分配**；也不能让所有worker共享一个无锁scratch。
初期选择K份就把它写入descriptor/upper；Q份优化更改reader revision并重新验收。table scratch取其实际串行转换最大值，不无理由分配整block大小。

### 2.3 Persistent executor 接入位置

现有内部 `ltx_native_create_streamed_v1()` 已接通C executor；下一步是session/API接线，不重复新增同名入口。
继续保持versioned options，不向未versioned的旧options尾部塞字段。
`ltx_native_denoiser` 的新分支持有 adapter state + `tc_stream_executor*`；legacy字段和 `streaming_slots[3]` 原路径不变。
`run_denoise_schedule()` 的block-stack边界选择旧wrapper或新wrapper，避免每个GEMM判断mode。

新wrapper逻辑示意（不是可编译API）：

```text
create_exact:
    validate construction view against compiled digest
    allocate prefix / step state / scratch
    tc_stream_executor_create_v1(plan, adapter_ops)  // 内部begin，只调用一次

each sampler step:
    bind immutable StepBindings on owner
    run_pass(next_pass, global_step)
    assert all readers retired before destroying StepBindings

after last denoise pass:
    finish -> counters -> destroy handle
    release prefix / scratch / denoiser-only workspace
```

prefix callback只收到pass；其step/workspace从 owner 绑定的 StepBindings 读取，不从 pass 值推断 sigma。
StepBindings包含video/audio workspace、rope、conditioning、sampler局部index和global step映射，生命周期跨完整run_pass。
两stage之间执行upsampler时slot backing保留；中间latent/upsampler/workspace峰值同时计入资源计划。
若shape变化导致weight layout也变，不能继续用原池；必须明确class/domain切换并重新编译支持范围。

### 2.4 真正的最后读者

审核 `run_block()` 到 `ltx_gpu.m` 的所有batch/非batch分支：CPU函数返回不普遍等于GPU完成。
video/audio若使用不同queue，ReaderSet必须各含最后一个真实reader；drain还要涵盖prefix与completion callback。
复用现有算子，不为拿fence禁用batch。必要时新增仅供新分支使用的“提交group并取得completion”窄接口。
同步-only adapter 可帮助排查正确性，但单独标实验，不因它没有race就放行P1。

### 2.5 LTX 专项验收

| ID | 检查 | 通过标准 |
|---|---|---|
| LTX-META | metadata-only vs audit allocation | 无GPU分配；每field容量覆盖实际，复制表不漏计 |
| LTX-FILL | 不同block反复填同槽，short-read/坏range | 无重新malloc/扩buffer；错误不提交compute |
| LTX-STEP | 同一slot跨step派生表变化 | 与legacy相同输入latent一致；无上步残留 |
| LTX-LIFE | 两stage + upsample + VAE + request重复 | 同domain一次pool创建，最后读者后释放；第二请求无旧ticket |
| LTX-FENCE | audio reader延迟、早到completion | 全reader满足才reuse；prefix同样被drain |
| LTX-PERF | K1/2/3支持子集、P/Q完全对齐的旧新 | P0/P1满足12；策略变化另列P4 |

tiny-smoke只打通路径；normal-target通过后才能签该tuple。尚未覆盖的audio/I2V/LoRA分支在preflight拒绝。
当前内部adapter已通过的具体范围见13第7节；完整session逐调用点的修改与故障清单见 [18](18-code-change-matrix.md) 第4–5节。

## 3. H3：复用双槽经验，不继承隐藏调度

截至 2026-09-16，F4 已完成 metadata-only 的第一步：
`h3_streaming_descriptor.hpp/.cpp` 通过 `h3_weight_store` 的只读 header 视图生成统一
descriptor，复用 `h3_stream_uniform_active_mask()`，并以 `StreamingPlanView` 做 K=2/G=1
的严格 projection。该实现只负责 snapshot、shape、range、capacity、active ID 和 pass
identity，不创建 GPU backing，也不进入 public execution。

当前明确边界：只接受原始 BF16 matrix；quantized cache、token reduction、first-block cache、
跨 forward prefetch、跨 block fusion、动态 gate skip 尚未获得 generic executor 表达。PlanView
对这些 shortcut fail closed，防止把 layout-only metadata 误当成数值等价执行模板。对应测试是
`tests/native/h3_streaming_descriptor_test.cpp` / `test_h3_streaming_descriptor.py`，并已进入
`make test-streaming-host` 和 release build。

下一 PR 的最小施工单仍是：把 `first_streamed_block()`、`next_streamed_block()`、实际 slot
generation、GPU 最后 reader fence 和跨 forward carry 编译成显式 candidate pass contract；
先做 fake backend 等价验证，再接真实 H3 session。没有该 contract 之前不得删除 public gate，
也不能以 descriptor 的 source bytes 或 slot capacity 证明 whole-request bounded。

主要位置：`h3_dit.c` 的 `allocate_stream_slot()`、`read_stream_layer()`、`next_streamed_block()`、`stream_ready_slot ^ 1u`，以及 `h3_gpu.m`、`h3.c`、`h3_session.mm`。

1. 保留legacy双槽pager；新exact入口描述常驻prefix、norm/AdaLN、双槽字段和reader格式。
2. adapter沿用allocation/read helper的数据布局，任务派发和线程创建改由框架负责；不能同时跑旧预取线程。
3. 首发冻结K=2/G=1及已验证访问模板；不要把 `active_block_count` 当连续0..N−1而丢掉真实block映射。
4. 将首次prefill、跨forward预读、step gate和final step显式化；当前框架pass末drain并重读，不支持隐式跨pass命中。
5. 如果旧路线跨pass预读对性能重要，先报告P1不等价/策略差异；要保留该优化，需扩展显式模板/生命周期后再测，不能隐藏在adapter里。
6. 跨block fusion读取下一block时必须进external-read闭包；G=1无法覆盖就拒绝该候选，不静默关fusion。

H3专项测试：首块非0、最后streamed块、两pass交接、next-block预读失败、同槽读者延迟、cancel后再请求。
对现有 active-block/approximation 策略固定相同配置再比较质量和性能；框架不能为了省内存额外跳块。
K=1是独立功能，需要串行reuse以及初始/末尾处理；K=3也需新的映射与capacity，不用改数组大小冒充通用支持。

## 4. Flux / Z-Image：先统一组件，再统一块

### 4.1 第一阶段 component-staged

改动范围：`native/models/flux2/flux_{text,transformer,vae}.cpp`、`native/models/z_image/z_image.cpp`、`native/backends/mlx.hpp` 及各自session。
component-staged不是把一个巨大component假装成普通可循环slot；由request orchestrator在组件边界load/execute/materialize/unload。
本阶段block-streamed capability保持false，stage resident lifecycle执行支持也必须显式实现，当前StageExecutor不能直接承担。

交接合同：text输出materialize → 确认不再引用text weights → 释放text/相关graph → transformer；VAE同理。
保留conditioning/latent的真正backing并计账，不能为降低“权重map大小”误删仍被计算图持有的对象。
禁止在默认compiled-step路径插入逐block eval/clear_cache；新分支即使组件可释放，也要测失去cache的真实代价。

### 4.2 第二阶段 block/group

- Flux双流/单流定义不同class和边界；多class executor通过L1后再接，不让每个模型另写barrier循环。
- 新ranged reader必须与 safetensors/GGUF 格式对应；一个shard不是一个执行block，整shard解包要如实计峰值。
- compiled函数显式接收当前slot权重/索引，或者用经过证明的稳定binding；不能捕获上一generation的array。
- cache key至少区分format/layout/backend、shape和binding revision；cache本身的容量/释放单独声明。
- 若backend不能证明异步最后reader，先用明确的group求值完成边界做实验，评估fusion损失；不直接承诺P1通过。

验收额外加入 graph-retention、反复bind同地址不同内容、shape A→B→A、组件释放后峰值、cold compile与warm cache 分列。
无法给出可靠释放证明的backend只能plan-only/experimental，不能借用LTX的资格。

## 5. API、service 与默认路径审计

`native/api/c_api.mm` 当前generate/prepare对新manual统一拒绝。只有真实adapter和资格preflight接好后才替换这道门。
不得简单删除 `streaming_layout_not_certified` 后让旧heuristic路径接收新参数。
`results.mm` 的当前固定plan-only字段之后应来自实际resolve/eligibility；执行结果缺actual时不能填resolved冒充。
service对新路径加入layout identity，旧session key不变；quarantine不重新分配给下一个请求。

每次代码PR对默认路径核查：

- parser缺省语义、profile v1、旧residency和旧memory_budget未改；新mode=false不创建executor。
- 数值kernel调用顺序、batch边界、cache保留策略与基线一致。
- 新descriptor扫描/文件hash/系统probe只发生在显式新路径或诊断工具。
- 构建/host测试通过只是第一步；真实P0包括resident、旧streamed、重复请求和各正式发布模型。

## 6. 实施review清单

提交者必须附：修改符号清单、默认分支diff说明、源/容量/读入字节三份账、owner和borrowed lifetime表、
新增测试ID与命令、未覆盖分支、实际P0/P1状态。测试未跑写NOT RUN，设备不可用写SKIP，不能用“设计上不影响”代替证据。
签核前至少有另一轮独立verifier检查，不由adapter自报“certified”。

## 7. H3 v3 execution bridge 的下一批施工单（2026-09-17）

通用框架已经能表达 H3 K=2/G=1 的跨 pass 首组 carry，但生产 H3 adapter 尚未实现。下一批必须按以下顺序推进，
每一步都保持 legacy `h3_dit` 默认路径原样：

1. **真实 source fill**：从 `h3_dit.c`/weight store 暴露窄接口，按 descriptor materialization 的四个 BF16
   matrix range 填入既有 Metal shared slot；校验 snapshot、short read、取消和 exact byte counter。
2. **slot view/bind**：为一个 block 构造只借用 slot backing 的 qkv/out/fc1/fc2 view；norm、AdaLN、text、
   conditioning 和 workspace 继续由 request owner 持有，不塞进 weight slot。
3. **single-block encode**：抽出不改变数值顺序的 prepare/run helper，先禁用 generic carry，仅证明逐 block
   output 与 legacy 同输入 byte/容差一致。
4. **真实 last-reader**：为 block 的所有 Metal queue 注册 command-buffer completion；跨 block fusion 若读前块
   weight，必须把该 reader 纳入同一 `ReaderSet`，不能用 CPU helper return 代替 fence。
5. **启用 v3 carry**：先验证偶数 suffix，再验证 uniform sparse policy 产生的奇数 suffix rotation；对每 pass
   断言 fill/encode exactly once、step identity 连续、最后 pass 无遗留 carry。
6. **session/candidate owner**：仅接内部 candidate constructor，覆盖 cancel、repeat、shape change、metadata
   mutation、drain failure 和 quarantine retry；public registry 继续为空。
7. **P0/P1**：先证明默认 resident 相对 clean dev 非劣，再做 legacy H3 streamed 与 generic same-layout 对照；
   normal-target 未通过前不开放 production。

建议新增 `H3SlotAdapter` 只实现通用 `ModelSlotAdapter`，不复制 scheduler。模型层负责 materialization 与真实
Metal fence，框架层负责 ticket/window/carry/cleanup。adapter 不得自行维护另一套 slot state 或把 legacy
`stream_ready_slot` 与 generic slot 同时设为 authority。

## 8. Z-Image P1关闭后的 H3 起点（2026-09-17）

Z-Image 已完成 private generic adapter 和冻结 P1，证明同步 MLX backend 可使用通用
`already_complete` reader 快路径。该结论不能直接复制给 H3：H3 必须依据真实 Metal command-buffer 的最后
reader 决定同步/异步 completion；若跨 block fusion 继续读取上一 block 权重，必须保留 callback mailbox。

H3 下一轮先审阅当前工作树已有的 candidate bridge，逐项对照第7节：真实四矩阵 fill、slot view、active
block ordinal、carry、fusion reader、session owner和metrics。任何 helper 已存在但缺真实 fence或失败清理的，
状态仍记部分完成。首个性能 policy 固定 legacy/generic K2/G1、相同 active mask、相同跨 forward carry和
per-request engine；不通过2%门槛时先按 refill/wait/encode telemetry定位，不改P/K掩盖开销。

## 9. H3 Turbo 收口与 Flux 起点（2026-09-17）

第7–8节的 H3 candidate 施工项已经完成到真实四矩阵 fill、slot view、Metal command-buffer reader、
cross-block fusion、cross-pass carry、cancel/drain/quarantine 和 per-request lifecycle。冻结范围只包含
MiniMax H3 Turbo 原始 BF16 `P0/G1/K2/D1/Q1`；不要继续把普通 H3、量化缓存或其他布局追加到该资格。

20-pair H3 release campaign 的 wall/denoise median point estimate 为 `1.01066/1.01185`，P95 为
`0.99511/0.99268`；输出逐对一致，audit steady allocation/thread-create 为0/0。物理 SSD block 方差使
strict bootstrap verifier 保持 `INCONCLUSIVE`，因此不能写成 production P1 PASS。按用户确认的范围与
性能口径，当前冻结 H3 Turbo tuple 已完成工程验收，合理 I/O 波动不再触发重复 TB 级 campaign；bootstrap
结果作为诊断证据保留。public gate、production registry和bounded-memory承诺均不因该结果开放。

H3 施工项至此关闭：只维护 MiniMax H3 Turbo 当前冻结 tuple，不继续接普通 H3、其他 checkpoint、量化缓存
或任意 K/G 布局。除非发现 correctness、生命周期或默认路径性能回归，后续资源转入 Flux 与通用 multi-class
runtime。

Flux 下一步不复制 H3 adapter，而按以下顺序执行：

1. 列出 text、double-stream transformer、single-stream transformer、VAE 的真实对象和 live interval。
2. 对 compiled graph 做 A→B→C slot 内容轮换，证明 graph 使用当前 generation，而不是捕获旧 array。
3. 将 double/single stream 定义成两个显式 layout class；先验证 class barrier，再决定 shared max-capacity
   arena 或 serial pool，不在 adapter 内实现私有 class scheduler。
4. 先做 component-staged 释放证明，再接 block/group fill；默认 compiled 路径不新增 `eval`、cache clear、
   descriptor scan或线程。
5. 冻结一个 legacy/generic 同布局 tuple，按与 LTX/Z-Image/H3 相同的质量、lifecycle、audit 和2%门槛验收。

### 9.1 Flux 9B 当前落地状态

上述1–4项已经以 **Flux.2 Klein 9B BF16 eager GPU** 的受控形式完成：9个fixed tensor常驻，8个dual block和
24个single block进入统一descriptor/compiler；两个layout class采用`retain_all`，setup创建dual/single各
K2 backing，class barrier及下一pass只切换pool authority，不重新分配。通用`MlxWeightPager`在owner线程
创建MLX shared arrays，worker只做多artifact `pread`；每个同步`mx::eval`是最后reader完成点。

冻结工程tuple为`P0/G1/K2/D1/Q2/reload`。两步真实请求完成64 fills，跨pass仍只有2个pool、4个slot bundle；
resident与streaming PNG SHA-256相同，streaming MLX peak降低约6.15 GiB。真实audit为setup worker/pool
`2/2`、steady allocation/thread-create `0/0`，默认resident请求全部streaming audit counter为0。

第5项现已补上可审计的private direct replay：它与generic共用同一P/K/G/D/Q、pager、双pool retention、
reader和同步边界，只绕过通用owner调度循环。正式10-block/20-pair P1中generic/direct wall median、wall
P95和denoise median ratio分别为`0.99971/1.00325/0.99816`，bootstrap上界全部低于门槛；输出、layout、
peak和fill计数一致。campaign policy通过
`expected_implementations`逐请求区分`flux_direct_same_layout_v1`与`generic_stage_executor_v1`，防止两侧
误走同一路径。resident与streaming仍是不同内存策略，二者耗时比不能冒充P1 framework overhead。
Flux 4B compiled graph、LoRA、GPU+ANE、prepare-only、bounded-memory和public route继续fail-closed。

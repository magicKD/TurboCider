# 18 · 布局优先框架：代码施工矩阵与验收交付

[目录](README.md) · [框架合同](17-layout-first-framework.md) · [任务编号](11-work-packages.md) · [性能统计规范](12-acceptance-playbook.md)

日期：2026-09-16。本文件将已有F0–F9落到当前代码，不另建一套里程碑编号。
**所有“建议新增/改造”均为待实施设计**；当前实际完成情况仅见13。本轮未改runtime，未跑模型性能campaign。

最新布局算法/时序算例见[19](19-layout-first-architecture.md)，下一PR的v2输入校验、snapshot所有权与完整session验收见[20](20-layout-first-implementation-and-acceptance.md)。

## 1. 先交付什么：一条真正完整的垂直切片

首发目标：LTX native GPU、确定checkpoint/format/workload、G=1、P≥1、明确K/D/Q，
真实prompt/text conditioning→stage1→upsample→stage2→VAE→输出→安全cleanup。
首条normal-target取现有实际使用workload填入验收卡，不用64×64×9 synthetic-conditioning用例替代。
K2或K3优先取能与legacy明确对齐的候选；tiny K1/2/3已有证据不等于三种都可以发布。

成功标准同时包含：完整质量、requested/resolved/actual一致、失败清理安全、P0/P1通过、production记录覆盖。
bounded另走F5/F6-bounded；它缺失时可以交layout-only，但必须明确没有Y硬上限承诺。

```text
F1真实descriptor/identity ─> F3 session exact接线 ─> F6质量/P0/P1 ─> LTX layout发布
          |                         |
     F9a inspect/verifier       F2异常/owner/清理补强
                                    |
                        F4 H3 -> F8 component -> F8 block

F5全请求资源闭包 -> same-layout guard -> F6完整L2/L3/P2 -> bounded发布
F9a simulate/sweep/preset：辅助选候选；不能跳过以上任何资格门
```

实施顺序表示依赖，不是要求同时大改全部模型。每批保留legacy可用，unsupported候选明确拒绝。

## 2. 当前源码与目标职责的映射

| 路径/符号（相对仓库根） | 现状 | 下一步代码建议 | 不能顺手改的内容 |
|---|---|---|---|
| `native/core/streaming_contracts.hpp`、`streaming/config.*` | 精确参数/presence已存在 | 保持schema；必要新增值类型独立于backend | 默认参数与旧字段语义 |
| `native/runtime/streaming/layout.*` | 权重分组/容量及v2 source/materialization/pass/workload identity | 补完整resource/construction投影、资格与外层接线 | 不探测RAM、不按Y重选P |
| 拟增`streaming/descriptor.*` | 尚无独立完整实现 | checked range/alias/shape/访问闭包校验；metadata-only | 不做model加载或GPU编译 |
| 拟增`streaming/request_context.*`、`registry.*` | 外层编排与layout registry待接 | 持有preflight快照、资格、可选guard、终态 | 不复制StageExecutor pump |
| `native/runtime/session.hpp` | 只有memory hooks，无exact streaming binding/report | 增optional plan/report及默认惰性的metadata/binding钩子 | 不强迫所有旧session实现空调度器 |
| `native/api/c_api.mm` | generate/prepare统一拒绝manual | explicit branch做preflight/资格/bind/终态；共用辅助函数 | 不直接移除gate放行旧heuristic |
| `native/platform/apple/results.mm` | 新配置固定plan-only结果 | 分开serialize intent/resolved/actual/authority | legacy结果不改；未执行actual=null |
| `native/core/stream_slot_c.h`、`streaming/c_bridge.cpp` | 同一C++执行器的v1 C桥 | 必要时新增版本化construction/report view，补ABI反例 | 不静默改变已有v1 struct布局 |
| `streaming/context.*`、`io_executor.*`、`slot_pool.*` | 单活动pool streamed、ordered multi-class barrier可执行 | 先补cancel/fault/owner审计；后做resident/request guard能力 | 不为新模型自建第二套I/O池 |
| `native/models/ltx_runtime/ltx_streaming_layout.*` | 真实metadata/fixed-span fill已存在 | 投影generic descriptor；明确源/内容/容量，复用owned snapshot | 不在fill重复parse/hash |
| `ltx_streaming_slot.*`、`ltx_streaming_adapter.inc` | 内部exact K1..3/G1双stage原型 | 与compiled construction view一致；补source实际计数和故障点 | 不复制run_block数值代码 |
| `native/platform/apple/ltx_session.mm` | 正常路径仍legacy handle | 单独request exact owner、两stage接线、报告和cleanup | 不改旧denoiser cache/key/deleter |
| `native/models/h3_runtime/h3_dit.c`及`h3_session.mm` | 原双槽pager与active-block逻辑 | 新route adapter，显式真实block映射/外部reader闭包 | 不用改数组大小冒充K通用化 |
| `native/models/flux2/*`、`z_image/*`与Apple sessions | 原模型/MLX生命周期 | 先组件交接；再独立reader、class、compiled binding | 默认compiled/eager缓存不重写 |
| `services/turbociderd/service.mm` | 有既有memory隔离逻辑 | 显式route identity与quarantine owner接线 | 默认session key、串行GPU策略不变 |
| `tools/native/build.sh`、`Makefile` | 已编入通用/LTX部分 | 新host utility/test build、独立release timing产物 | 不在执行build中修改脚本 |

表中`streaming/*`均指`native/runtime/streaming/*`。新文件名是建议，不要求机械拆小文件；
保持runtime不依赖Foundation/Metal，模型metadata投影留在model/platform层。

## 3. F1下一PR：真实计划，而非容量演示

### 3.1 修改项

1. source range、destination storage、binding/conversion、workload/pass值类型已有基础实现（13第9节）；后续补construction/resource投影，仍受现有控制平面上限约束。
2. `FieldSpec.storage_id`现已明确为destination身份，source在materialization中；接线保留该语义，旧host fixture迁移有revision标识，不再重复改同名字段含义。
3. 分离`source_read_bytes_per_pass`、`destination_content_bytes_per_pass`、`allocated_capacity_bytes`。
   原`suffix_read_bytes_per_pass`已被`suffix_content_bytes_per_pass`与optional source读量替代；
   padding只进capacity，source未知保持null。最新实现与测试见13第9节，后续报告沿用这一口径。
4. 对齐每个field的capacity，验证tail group有效字段与空缺bias；真实checkpoint160 fields不强行套fixture162。
5. LTX已有C metadata结果投影到generic descriptor；计划中的真实block IDs回投adapter，禁止再自行分组。
6. 将construction view与plan digest绑定；创建时逐field核对实际bytes/format/geometry，不只核对总量相等。
7. descriptor包含实际48块、stage1/upsample/stage2访问模板、输入shape/token/格式/reader revision；
   guard所需unknown资源保持unknown，不因权重计划完整就标whole-request upper完整。

### 3.2 交付与测试

- fake metadata手算golden + 真实checkpoint metadata-only输出；后者不要求GPU。
- 同输入/JSON重排digest稳定；改变Y不变；改变range/shape/pass/reader必须变化。
- 相同source两个真实backing不能去重；真正alias不能重复计；相同总bytes不同field布局必须拒绝或正确适配。
- 截断/替换/修改文件、offset+length溢出、dtype/packing错配、缺field、invalid tail不进入GPU分配。
- 源码审计：compiler不读环境可用RAM，不调用旧`make_block_residency_plan`，不创建worker。

完成门：CMP-01/02/03及metadata专项通过，报告不再用`EXPERIMENTAL-unqualified-checkpoint`冒充可认证identity。
完成本批仍不能generate；production registry继续空。

## 4. F3d下一PR：LTX完整session接线

### 4.1 Preflight和本次handle

`LtxNativeSession`的metadata probe依据现有路径选择规则打开snapshot，但不创建denoiser、Gemma、VAE或清cache。
exact plan绑定后，generate从request context读取layout；不要把`request.residency`伪装成legacy streamed来复用预算推导。

新增独立exact owner成员或本次request-owned state；不要复用现有`unique_ptr`的void deleter。
可以在model方法入口选一次明确handle/执行分支，但避免在每个GEMM中重复检查`streaming.active()`。
新的RAII binding负责unbind；**安全回收是显式状态返回操作，不是无条件析构动作**。

### 4.2 必改调用点

| `ltx_session.mm`现有区域 | exact分支处理 | 必验行为 |
|---|---|---|
| `uses_parent_mlx(request)` | 根据真实text/denoiser/decode backend选择，不能只看streamed布尔 | 不漏MLX依赖；不新增不必要的全局同步 |
| `ScopeExit staged_cleanup` | 覆盖新route异常出口，但unsafe时移交整个owner而非reset | early failure无use-after-free |
| conditioning缺失时释放旧denoiser | preflight/资格通过后才允许清理；text输出独立持有 | cache hit/miss两条都覆盖 |
| `denoiser_key` / `denoiser_cache_hit` | request exact不可命中上次完成的executor | repeated request新generation、无尾块命中 |
| native创建 | 先补v2校验/ownership tests，再调`ltx_native_create_streamed_v2`借用同snapshot和compiled view；v1仅保留原实验回归 | P/K/G/D/Q/field/source与计划一致；失败不释放caller header/fd |
| connector / stage1 / upsample / stage2 | 使用本次exact handle，按pass/step合同调用 | 8+3模式仅对当前已支持tuple；不得丢掉真实conditioning |
| denoiser→VAE handoff | 最后pass finish并安全释放denoiser域；latent仍活 | 输出与引用释放两者都证明 |
| `block_streaming`结果 | exact源于实际counters，不伪造旧budget | 源字节/内容字节/slot分配分开 |
| 尾部`make_block_residency_plan`核对 | exact改为actual对compiled核对，不调用预算启发式 | 同layout换Y不变 |
| `unload()` / destroy | legacy逻辑原样；exact必须owner-safe status drain | failed drain保留handle、engine不可复用 |

### 4.3 完整资源时间表

```text
text load/run   text weights live + raw conditioning
text release   等待输出materialize；保留conditioning独立backing
denoiser create prefix + K slot backing + per-K scratch + helper/connector
connector      raw -> connected context；按最后使用释放raw
stage1         denoiser域 + context + stage1 workspace/latents
upsample       denoiser域保留；old/new latent重叠及upsampler实计
stage2         同一weight域 + stage2 workspace/latents
denoiser drain GPU readers与workers结束；释放denoiser及无后用context
VAE/export     latent、decoder、输出/编码buffer按真实重叠计
terminal       可用输出 + 必要清理完成后才success
```

此表是审计模板，不断言当前Gemma/MLX引用已能完全释放；未知release/envelope逐项标注。
whole-request guard还需F5，不用slot峰值替代表中所有资源。

### 4.4 完成门

真实prompt→成品video全部完成；视频帧数/尺寸/时长和latent质量通过；unsupported audio/I2V/LoRA明确拒绝。
同进程成功→成功、取消→安全结束→新请求、shape A→B→A、stage2失败、VAE/export失败均有结果。
取消不应发布半成品为成功；只清理本请求拥有的临时输出，不碰用户已有文件。
pool/thread创建次数不随11个pass增长；全请求实际参数与compiled一致。此时仍只是实验完整运行。

## 5. F2/F3补强：故障与service安全是发布前工作

### 5.1 必须可注入的故障点

拟在test-only backend/adapter注入：第n个allocation、worker创建、pread短读/EINTR/EOF、fill转换、
prepare、encode/commit、多queue完成、mailbox overflow、owner线程误用、pass边界、terminal cleanup。
生产不能通过普通请求配置触发故障；故障测试不污染release timing构建。

每例断言：primary error稳定、无新dispatch、无partial Ready、无早复写、资源最终安全释放或显式quarantine、
调用者输入/输出提交语义一致。所有异步借用必须到终态后释放。

### 5.2 当前阻塞join的处理次序

先补有限chunk read和协作cancel测试，记录cancel-to-stop-dispatch与cancel-to-safe-terminal；
再做受控非协作fake worker，证明无法drain时保留整个state，不能用“测试进程还能退出”当安全。
若service要求固定恢复deadline，单独设计进程隔离；不要在本批无限扩展线程取消框架。
当前同进程阻塞join问题未解决前，不写hard-deadline SLA。测试自己的超时要由外层受控runner记录TIMEOUT和残留状态。

### 5.3 API和service共同验收

`tc_engine_free`、service session eviction、request异常、cache-clear入口都不能绕过quarantine安全所有权。
新增route/session key只影响新路径；旧默认key保持相同。profile热更新不改正在运行的sealed plan。
准备阶段metadata拒绝不卸载旧模型；unsafe engine后续请求明确拒绝，不自动重新放回worker池。
撤回registry record只阻止未来explicit请求，不强杀已运行pass，不改变legacy fallback策略（新manual始终不自动fallback）。

## 6. F4/F8模型推广的“接入包”

每个模型沿用15的六件共用接入交付物，并提供descriptor fixture、reader/binding实现、reader completion证明、完整workload卡与evidence bundle。
支持面按`model + format + backend + branch + layout + workload`描述，不能只注册一个model名字。

### 6.1 H3

- 先K2/G1，不改旧`stream_ready_slot`/xor逻辑；新route将真实active block序列投影group IDs。
- 从`next_streamed_block`、first-block prefill、step gate抽取访问模板；norm/AdaLN等resident资源单列。
- 跨block fusion/next-block外部reader必须闭包；首发表达不了就unsupported，而非静默关闭fusion。
- legacy跨forward预取若新executor pass drain不等价，记录为P4；先扩显式模板再做合格P1，不能只比较K=2。
- 对首块非0、inactive gap、final pass、下一块read失败、双queue延迟、cancel做专项回归。

### 6.2 Flux与Z-Image

- 第一阶段只统一外层component staged；resident component交接不能伪装成当前StageExecutor支持resident循环。
- 为text输出/latent建立独立backing及materialize边界，清理model对象后审计graph/compiled cache仍持有的权重。
- 不在default内循环插`eval/clear_cache`；new-route新增eval与fusion损失算入完整wall。
- 第二阶段才接format-specific ranged reader、heterogeneous layout class和稳定compiled binding。
- 增加“同地址不同generation内容”、A→B→A shape、cache复用错权重、quant shard解包峰值测试。
- 不能证明引用释放或GPU最后reader时，不授予block-streamed能力，不借用LTX资格。

当前实现状态（2026-09-17）：Z-Image BF16已完成冻结P1；Flux.2 Klein 9B BF16已完成private eager execution，
使用通用`MlxWeightPager`、dual/single两个`retain_all` pool和同步last-reader。Flux默认resident旧/新dylib
warm median未观察到回退，真实Q2 audit稳态分配/建线程为0/0。Flux 4B compiled graph仍保持原路径，未做
地址捕获假设，也未通过插入逐block `eval`来换取表面上的框架统一。Flux正式same-layout P1、P2/P3及public
资格仍待独立证据。

## 7. F5预算组合与机器配置交付

optional guard消费相同plan，不修改布局。required-site closure覆盖17/05的所有非slot资源、控制内存和framework envelope。
source字节、backing容量和physical footprint不是可互相代替的指标；ledger唯一backing计费，logical Vacant不减账。
以B=`floor(Y*(100-X)/100)`做checked admission；系统余量S单独检查。upper=B/B+1边界测试与baseline变化必须覆盖。

配置交付仅使用02的schema和现有examples。建议preset包含完整K/G/P/D/Q及显式retention，
证据/硬件范围作为sidecar，不把`certified`塞进用户可写配置。不要实现未规范的`extends`或`auto_slots`。

| 用户情境 | 推荐入口 | 验证方式 |
|---|---|---|
| 内存充足，重视现有速度 | 不启用新框架 | P0默认resident/旧streamed |
| 希望固定两/三槽便于调优 | 完整manual配置，guard off | 先plan解释资源与支持范围，再有资格执行 |
| 要Y上限与buffer | 相同manual + memory_constrained | 完整closure/资格/admission通过才运行 |
| 不知道如何选参数 | 显式采用有证据preset | 展示硬件/shape/格式/SSD条件及非slot峰值 |

不能发布“16GiB=K1、32GiB=K2”这样的无证据映射。内存容量相同而GPU/SSD/shape不同，最优布局可能不同。
若没有该设备证据，只给experimental候选；失败诊断可以建议新配置，但不替用户执行修改后的布局。

## 8. F9a工具链：最小可用交付顺序

16定义工具行为；这里规定每批验收输出。下列工具均是计划，不是当前可用CLI。

| 交付 | 复用代码 | 必备输出与反例 |
|---|---|---|
| native inspect/plan utility | 同一descriptor/compiler | intent/resolved/unknown/rejection，metadata成本；无GPU也可运行 |
| Python campaign runner | utility + 正式C API测试入口 | 固定workload/配置hash、原始成功/失败/超时样本、环境与build |
| 独立bundle verifier | 独立小算例/不变量，不调用runtime自证 | 缺raw、错identity、P1语义不等价、quality缺失均不得PASS |
| simulator | 编译输出+独立离散事件模型 | 手算oracle、校准/验证分离、未知阶段显式unknown |
| finite sweep / preset exporter | 前述工具和正式资格记录 | 默认simulate-only；独立确认集；不自动启用或签发 |

### 8.1 仿真只回答可回答的问题

先用16的K1/2/3手算例验证安全和事件顺序，再导入真实read/decode/prepare/compute/fence分布。
将带宽竞争、cache命中、pass drain、class切换、prefix首次读、VAE分别建模；未知耗时不能按0处理。
同一trace拟合与验收分离，报告wall误差/候选排名稳定性，而不是仅输出一个speedup。
软件设置Y只测试admission，不等于模拟物理低内存。swap敏感性模型不替代macOS pager实测。

### 8.2 诊断优先级

ready starvation先查read/convert；Ready充足但GPU慢查共享带宽/encode/batch；每pass变慢查重建pool/线程；
默认变慢先查入口、cache策略、额外probe/sync和构建差异。所有优化保持requested/resolved/actual一致。
先修实现成本再扩K/Q；不能用更大prefix掩盖P1回退。

## 9. 本机验收操作清单

### 9.1 本轮可复用的已有命令（不是宣布本轮已运行）

从仓库根执行host快检：

```sh
python3 -B tests/native/test_streaming_layout.py
env TC_STREAMING_SANITIZER=address,undefined python3 -B tests/native/test_streaming_layout.py
env TC_STREAMING_SANITIZER=thread python3 -B tests/native/test_streaming_layout.py
python3 -B tests/native/test_ltx_streaming_layout.py
python3 -B tests/native/test_streaming_contract.py
python3 -B tests/native/test_contract.py
python3 -B tests/repository/test_cpp_boundaries.py
```

API contract依赖与源码匹配的已构建native库；构建及真实checkpoint/Metal命令见13第4/7.5节。
机器无fixture或GPU权限时标SKIP，不能作为PASS。GPU测试在必要权限下显式执行；本轮文档任务不启动GPU/压力。
13的历史PASS只对当时构建与用例有效，任何runtime改动后需重跑对应范围。

### 9.2 下一批新增测试建议

括号中的后缀是既有gate下的子用例，不是另设验收层级；测试文件名为拟议。

| 测试/建议文件 | 覆盖合同 | 断言与通过产物 |
|---|---|---|
| `streaming_descriptor_test.cpp` | CMP-02/03，真实ranges/身份 | 有效fixture一致；TOCTOU/range/alias错配拒绝；unknown不填0 |
| `test_ltx_streaming_session.py` | GPU-01，完整请求 | 真实prompt与normal-target成品/latent，exact实际计数 |
| `streaming_request_context_test.cpp` | CFG-02/FLT-01 | preflight失败无unload；partial construct逆序安全清理 |
| `test_streaming_service_lifecycle.py` | RUN-04/FLT-02 | repeated/cancel/evict/quarantine；unsafe不回idle |
| 扩展`streaming_executor_test.cpp` | RUN-02/03/04 | inline/late/stale回调、不同step、tail、多pass，reader迟到不复写 |
| 扩展`streaming_c_bridge_test.c` | FLT-01/02 | bad size/version/null/wrong owner，C边界错误稳定 |
| default audit build | DEF-01 | new descriptor/pool/worker/probe/clear调用全0；旧路径输出一致 |
| new-route allocator audit | PERF-01 | 稳态framework allocation/thread创建0；模型原有分配单列 |
| `test_streaming_simulator.py` | L0/工具验收 | K1/2/3手算、乱序/延迟、D0、Q限流、pass/class边界 |
| `test_streaming_verify.py` | PERF-01/MEM-02 | 只有aggregate、缺样本、错build、错布局、缺质量不能PASS |

### 9.3 正式性能campaign卡

在GPU运行前封存以下信息；任何空缺决定对应结论不能发布，不事后补猜测值：

```text
comparison_kind / workload_id / normal-target or boundary
before/after binary hash + source manifest（含dirty与untracked）+ compiler/flags
checkpoint content identity + adapter/backend/reader revision
shape/frames/steps/seed/prompt/input branch/precision/numerical flags
P/K/G/D/Q + startup/pass lookahead + retention + all component policies
process_cold/request_warm/retained_model + OS file-cache condition
ABBA blocks + warmup + sample ceiling + exclusions + stopping rule
quality metrics/tolerances + performance statistics/multiplicity policy
GPU/RAM/SSD/OS/power/thermal + pressure scope + stop/cleanup conditions
```

P0使用可信before/after隔离进程；已保存baseline的来源限制见13，不能只拿同一候选build的legacy叫P0。
P1除K相同外还要同Q/D/P、startup、pass、retention和reader语义；不等价就是P4或单独生命周期实验。
LTX旧K1首块同步读与新prefix期间预取不同，已有tiny时序不能证明纯框架收益。

12是阈值与统计方法唯一来源：P0 end-to-end/denoise median比值95%区间上界≤1.02、P95≤1.05；
P1 median≤1.02、P95≤1.05。目标仍是无可检测回退，不把阈值当可随意消耗预算。
按ABBA block重采样；20 matched pairs只是起点，tail至少50次有效请求仍可能不够。
每个条件单独判断PASS/FAIL/INCONCLUSIVE，禁止取跨workload平均掩盖失败，禁止反复取样直到偶然PASS。

完整wall包括metadata/compile、load/text、两stage、upsample、VAE/export和必要cleanup；重叠阶段不能简单求和。
初次编译、冷启动、重复请求分别报告；trace/audit构建单独运行，不把instrumentation差异当调度成本。
质量或安全失败的时间样本保留为失败，不能只统计幸存快样本。

### 9.4 与swap比较

先真实低内存专用机器；大RAM软件Y只能做策略/预算验证。若需人工pressure，另行取得明确授权，
冻结资源上限、时长、磁盘余量、pressure/thermal停止条件；不关swap，不purge用户缓存，不使用无界分配器。
P3匹配质量、checkpoint、shape、外部压力和cache协议；分开logical reads、physical I/O、fault、compression、swap及footprint。
无可靠swap观测时只能称低内存条件对比，不能宣称快于swap。系统counter不能自动归因给本进程。
baseline OOM/timeout报告完成率和删失时间；不把timeout阈值当真实wall算倍速。
明确更慢也是有效结论：保留该条件下更好的legacy路径，不发布无证据preset。

## 10. 可签核的交付门与回滚点

| 门 | 进入条件 | 退出证据 | 未通过时 |
|---|---|---|---|
| F1真实plan | 当前host底座 | metadata/identity/field容量/无GPU校验通过 | 仅plan-only |
| F3完整LTX实验 | F1+安全bridge | 完整请求质量、actual、重复/取消/清理通过 | 不开production |
| F6-layout | normal-target可运行、可信baseline | P0/P1+质量/lifetime证据bundle与review | experimental或INCONCLUSIVE |
| F4/F8推广 | 各adapter完整合同 | 各模型独立质量/性能与能力矩阵 | 仅已过模型保留资格 |
| F5/F6-bounded | required-site/upper完整 | 同布局admission、完整L2/L3、P2 | layout-only，不承诺Y |
| F9 preset | 真实候选与独立确认集 | 工具verifier+资格范围+硬件侧证据 | 仅候选/仿真报告 |

生产启用按tuple新增registry record，不设一次性“所有模型都完成”总开关。撤回record即可停止新路线，旧路径始终可用。
公共parser或shared kernel的回归不是撤record能消除的，必须修复/撤回对应代码变更并重跑P0；不能只关新功能掩盖默认回退。
每个PR同步更新13的“已实现/实测/未覆盖”，附构建hash和artifact；没有分母不写完成百分比。

最终review需能回答：谁决定布局、谁拥有slot、谁证明最后reader、谁释放失败请求、哪个baseline证明没变慢、
哪个完整资源合同支持Y上限。任何一问没有证据，即对应门尚未完成。

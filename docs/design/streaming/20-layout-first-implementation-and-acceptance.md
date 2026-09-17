# 20 · 布局优先实施清单与验收交付

[目录](README.md) · [架构和算法](19-layout-first-architecture.md) · [代码施工矩阵](18-code-change-matrix.md) · [唯一验收规范](12-acceptance-playbook.md)

日期：2026-09-16。本文补充**从当前工作树往前推进**的 PR 内容、依赖、负例和交付证据。
不另设里程碑编号：F0–F9 沿用11，L0–L3 是12中的验收层级，不拿它们重新命名工作包。
当前源码事实与重跑结果归档到13第11–13节。v2、candidate request owner、生命周期 fault/audit 与
multi-class barrier 已实施；public production 资格和 bounded guard 仍待接，不把整个工作树归因于单一批次。

## 1. 当前完成到哪里

| 项目 | 源码核对结果 | 不能据此宣称 |
|---|---|---|
| config/compiler | 有严格参数、canonical、source/materialization/pass/workload | 全请求 upper 已完整 |
| slot/executor/C bridge | 有持久 pool/workers、generation、单活动pool streamed、ordered multi-class barrier、drain | resident/request orchestration 或服务 quarantine 已闭环 |
| LTX metadata | StreamingMetadata 拥有 fd/header，snapshot identity 与 v2 same-source checks 已实现 | content hash/文件不可变已证明 |
| LTX 内部 adapter | v2 K1/K2/K3；candidate connector→两stage→VAE/export、cancel/borrow/quarantine 已验证 | 仍未获得production资格 |
| LTX session | candidate 有独立exact request owner/status destroy；legacy仍保留旧handle/deleter | public exact 或bounded已放行 |
| public API | public generate/prepare仍拒绝active manual；私有candidate authority可执行首批tuple | 配置能解析不等于production执行 |
| 性能 | tiny默认resident 20-pair P0 PASS；post-multiclass 4-pair无明显回退 | normal-target、legacy-streamed、P1/P2/P3已通过 |

当前已有独立 release/audit/test-hook build、host/sanitizer、真实LTX完整candidate生命周期和tiny性能证据；
这些仍不等同于production资格。下一步是normal-target与legacy-streamed P0/P1、whole-request guard、artifact
trust和模型adapter，而不是删除public gate。

## 2. 依赖与交付顺序

```text
F0 route/config 已有底座
  └─ F1 + F3a: 同 snapshot / construction validation
       ├─ F9a: inspect + evidence schema + independent verifier（早做）
       └─ F2 + F3d: request owner / LTX 完整 session / 故障清理
            └─ F6-layout: normal-target quality + P0/P1
                 ├─ 精确 tuple 的 LTX production 记录
                 ├─ F4: H3 K2/G1
                 └─ F8a: Flux/Z-Image component lifecycle

F5: whole-request closure + same-layout guard
  └─ F6-bounded: 完整 L2/L3 + P2 → bounded 记录
F9a P3/P4: 实测决定推荐，不作为安全资格的替代
```

实验 test build 应在 F6 之前就能跑完整请求用于取得证据，但 release gate 继续关闭。
不能把“必须认证后才能测试、必须测试后才能认证”做成循环依赖。

## 3. PR：F1/F3a snapshot、ABI 与 construction 补强（最高优先级）

### 3.1 修改文件和职责

- `native/models/ltx_runtime/ltx_native.h`：明确 v2 struct/version、borrow、failure out、destroy 合同。
- `ltx_blocks.c`：v1/v2 共用 exact options validator，但不改变 v1 ABI 或 legacy options。
- `ltx_safetensors.h/.m`：fd/header 验证、duplicate/map cleanup；保持 caller fd 所有权。
- `ltx_streaming_descriptor.*`：snapshot/change-check 与 immutable artifact 注册的边界。
- `ltx_streaming_adapter.inc`：construction view 对 compiled field/binding 逐项校验。
- tests：新增 v2 malformed/ownership fixture，扩展真实 model harness。

### 3.2 审计接缝与已实施范围

1. `ltx_native_create_streamed_v2` 的入口检查此前没有与 v1 完全对齐；本轮已提取 shared validator：
   两者现在都检查 out、plan struct/version、generation、groups/capacity pointers、K/Q/D/P 等；
   现在先完整校验再做 metadata/GPU构造；测试见`test_ltx_streaming_snapshot.py`。
2. `ltx_st_map_fd`现在duplicate fd、校验device/inode/size/time snapshot并建立整文件 mmap；
   这不同于 metadata projection 的 fd-only 行为。不能写“整个运行没有 mmap”，
   也不能把虚拟 mapping 长度当物理峰值或把 `madvise` 当立即释放证明。
3. borrowed header 是 shallow copy，mapping 则是 native 自己的 duplicate/mmap；
   每条成功/失败/quarantine 路径分别证明 native 只释放自己拥有的资源。
4. 当前 v2 传 header+fd，并未形成 descriptor digest→artifact identity→每个 helper reader 的完整绑定。
   同大小错误fd、stat snapshot失效已有拒绝测试；trusted hash/registry和其他artifact仍待接。
5. `tests/native/ltx_streaming_model_test.cpp` 现在支持`--exact-api 2`并传入`StreamingMetadata`的header/fd；
   v2真实K1/K2/K3与connector smoke已运行，但仍非完整Gemma/VAE/export。

上述已实施部分见13第11节。完整fault/quarantine、field级view、trusted identity与session资格仍不由这些smoke授予。

### 3.3 必增测试（沿用 CMP/FLT/GPU gate）

| 子用例 | 输入/操作 | 期望 |
|---|---|---|
| CMP-02/v2-abi | null out/plan、错误 size/version、null arrays、K0、D≥K、Q>K、invalid P | 稳定错误；无 GPU/权重分配；已提供的 out 正确初始化 |
| CMP-02/v2-source | size 不符、同大小错误 fd/header、range 溢出 | 在构造前拒绝，不能以 total bytes 相等通过 |
| CMP-02/snapshot | replace/truncate/原地改写 fixture | owner 检测失效；并发写入不在支持范围，不能承诺无竞态检测 |
| FLT-01/v2-create | 第 n 次 prefix/slot/helper 分配失败 | 逆序清理；caller fd/header 未释放；unsafe 时保留非空 handle |
| GPU-01/v2 | metadata→generic compiler→v2→两 stage | 同源构造、actual 与 plan 一致、latent parity |
| FLT-02/v2-destroy | wrong owner/cancel/late reader | 不释放借用者；安全 destroy 后 caller 仍可 check_unchanged/read |

fixture 文件修改只针对本测试创建的临时文件，不修改真实 checkpoint。
host 部分与 Metal 部分拆开，设备不可访问时标设备未运行，不以 host PASS 替代。

完成门：同源 construction 与版本化输入验证闭环、测试可复现；仍不授予 production。
请求期间原地修改 artifact 不受支持，资格依赖可信不可变输入假设；不能通过 stat 反例测试就宣布强快照隔离。

## 4. PR：F2/F3d request owner 与公共接线

### 4.1 内部类型（建议，非已存在 API）

```cpp
// 名称示意：实际沿用已有 Result/error 风格，不引入第二套异常体系。
struct StreamingPreflight {
    OwnedMetadataSnapshot snapshot;
    streaming::Descriptor descriptor;
    ImmutableConstructionMetadata construction;
};
struct StreamingRequestContext {
    StreamingPreflight preflight;
    SharedImmutablePlan plan;
    ModelExactOwner model;
    OptionalBudgetBridge budget;
    RequestTerminalState terminal;
};
```

ModelSession 新增 metadata/bind/drain 的窄钩子，默认 unsupported。
legacy 永远不调用这些钩子；不能把“requested exact 但没实现”当 no-op 成功。
优先让 preflight 和 binding request-scoped，避免 session 全局状态泄漏到下一请求。

### 4.2 逐文件任务

| 文件 | 修改建议 | 不变项 |
|---|---|---|
| `native/runtime/session.hpp` | optional preflight/bind/report hook，声明 owner/lifetime | 原 generate 接口/默认实现路径 |
| 拟增 `streaming/request_context.*` | snapshot、immutable plan、drain/quarantine ownership | 不复制 StageExecutor |
| 拟增 `streaming/registry.*` | 精确 tuple 的只读资格；test/release 权限分离 | 不让请求 JSON/profile 自签 |
| `native/api/c_api.mm` | gate 改为 real preflight→资格→bind；统一 generate/prepare 错误处理 | 不直接删除旧 gate，也不改默认 normalization |
| `native/platform/apple/ltx_session.mm` | 独立 exact handle，connector/run/upsample/report 引用本次对象 | 原 denoiser cache/key/void deleter |
| `native/platform/apple/results.mm` | requested/resolved/actual/authority 分开 | legacy 成功 JSON 保持兼容 |
| `services/turbociderd/service.mm` | exact session key、unsafe owner、eviction/engine_free 协议 | 旧 key 与 GPU 单作业策略 |

内部实验权限用专门 test build/harness；release 优化与正式执行代码相同，仅资格来源不同。
不新增用户 unsafe 字段或可任意环境变量绕过资格。带实验权限的产物不得替换默认服务。

### 4.3 LTX 接线核对清单

1. metadata 描述实际 token/shape/format，两 stage/pass 不从 tiny fixture 常量复制。
2. conditioning 缺失或命中旧 cache 时，preflight 失败不清模型；通过后才执行本次 lifecycle。
3. native exact create 使用已验证 v2 view，检查实际字段而非仅总容量。
4. connector、stage1、upsample、stage2 都使用 request exact handle，并明确各 artifact。
5. 最后 pass drain 后释放 denoiser 域；latents、输出仍有独立 owner；再进入 VAE/export。
6. exact 报告取真实 counters，不调用尾部旧 make_block_residency_plan 反推再自证。
7. unsafe cleanup 保留整个 context，包括 sampler 暂存/metadata/callback；不只保留 MTLBuffer。
8. prepare 的未支持模式精确拒绝，不把成功 generate 资格自动授予 prepare。

### 4.4 所有权验收

- 成功→成功、shape A→B→A：无旧 generation 内容误命中，pool/worker 不随 pass 增长。
- 取消→safe terminal→新请求：不发布半成品，不泄漏本次 resources。
- stage2/VAE/export 失败：primary error 不被 cleanup error 覆盖。
- quarantine：不接下一请求，engine_free/session eviction 不绕过安全 owner。
- 非协作 worker：外层测试 runner 限时，记录 TIMEOUT；不在被测进程强杀线程后宣称清理成功。
- metadata/capability/资格拒绝：旧 resident/cache 不变。
- default audit：新 descriptor/pool/workers/probe/cache-clear 调用为0。

完成门：真实 prompt/text→两 stage→VAE→可用输出的实验完整请求；正常 workload 质量/lifetime 通过。
这个门通过仍要跑 F6，不能把实验执行许可写成 production 资格。

## 5. PR：F5 资源闭包与 guard 组合

沿用现有 `memory_manifest/plan/execution/accounting` 等底座，补 adapter bridge；
不要再建一套只计算 slot 的 memory scheduler。布局已选定，guard 不调用旧预算 heuristic。

逐 site 交付：

```text
site_id / destination storage_id / allocator owner / resource domain
creation epoch / first use / last reader / safe release epoch
known upper or unknown(reason) / alias & envelope membership
reserve / allocate / cancel-reserve / pending-release / release events
```

覆盖 text、raw/connected conditioning、prefix、slots、每 K 或每 Q 份 scratch（按真实归属）、
stage activation、upsample old/new latent 重叠、VAE、输出/编码、control、framework/driver envelope。
短暂大额分配也必须纳管，不能只看 100 ms 一次的采样。

验收：

- MEM-01：upper=B 允许、upper=B+1 拒绝；lower baseline 改变 admission 不改变 digest。
- 相同 backing alias 只计一次；独立 derived buffer 各计；reservation→allocation 不双算。
- 已 submit 但未完成的 release 仍计费；Vacant 不减 slot backing。
- MEM-02：required unknown、失效观测、swapout 增长按05 fail-closed；不能 silent success。
- guard off 不创建“无限预算 guard”，但 slot safety 仍完整。
- 首发 request retention 终态不留未声明 backing；未来 session retention 另走 F9b。

完整 L2/L3、P2 才能授予 bounded。P3 是否击败 swap 是独立性能问题，不是 bounded 安全资格的必要条件。
Y 上限是05定义的纳管资源+envelope/观测合同，不能对系统全局 swap 作绝对保证。

## 6. 模型扩展 PR 与支持矩阵

### F4：H3

先审计真实 active-block/first-block/step gate/next_streamed_block，再做 K2/G1；
norm/AdaLN 小权重单列；fusion/外部 next-block reader 进入 last-use 闭包。
如旧跨 forward 预取与新 pass drain 不同，该对照不能叫 P1；先表达等价模板或按 P4 报告。

专项：非0首块、inactive gaps、末步无多余预取、延迟双 queue、read 错误、cancel。
K1/去 xor/更多 K 是 F7 独立任务，不在 K2 发布时暗中放开。

### F8a/F8b：Flux 与 Z-Image

先统一 component scope，验证 materialized text/latent 在卸载后仍独立有效，
检查 compiled graph/weight map/quant shard 的引用与缓存。新 route 的 eval/cache-clear 开销进 wall。
旧 compiled/eager 内循环不改，不把 component unloading 叫 block slot streaming。

block 阶段另交 ranged reader、heterogeneous class、fixed backing binding、content generation 验证；
同地址换内容、不同 shape A→B→A、derived scale/解包峰值均是必测。不能证明 last reader 就留 component-only。

每个模型使用相同 six-part adapter contract（见19），分别提交 exact tuple 资格；
注册到 model 名字级别的无限能力禁止通过 review。

## 7. 工具链与配置交付

沿用16的 T0–T4；工具当前仍为规划，不伪造 CLI 已可运行。

| 先后 | 最小交付 | 独立验收 |
|---|---|---|
| T0 | native inspect/plan，Python 只编排 | 不分配 GPU；真实 source/content/capacity；unknown 可见 |
| T2/T3 基础 | campaign/raw schema + verifier | 缺 build/quality/raw、actual≠resolved、错误 P1 分类都拒绝 PASS |
| T1 | 离散事件 simulator | 19 的 120/80 ms 算例、K1/多 queue、D0、Q限流、pass barrier |
| T2 完整 | release ABBA runner | 保留失败/超时；cache/thermal/计时协议；不隐式施压 |
| T4 | 有界 candidate sweep/preset exporter | 探索/确认集分离，不自动启用，不自签 registry |

schema 分三份，防止混淆：

- request/profile：仅02合法参数，使用 examples；没有 performance policy 或 certified 开关。
- calibration/campaign：工具输入，绑定 workload/build/measurement protocol，不直接当 request。
- evidence/registry sidecar：只读审查产物，性能硬件范围与执行/预算资格分别记录。

preset 至少附适用 checkpoint/format/workload/GPU/SSD/RAM 条件和状态。
unknown 硬件只能显示 experimental 候选，不做 RAM→K 自动映射。修改 K 必须重新校验 D/Q/P/G，
Y 改变不改 layout digest。拒绝时解释峰值来源；VAE 主导不要只建议减少 denoiser slot。

## 8. F6 性能执行卡：怎样证明“不能差”

### 8.1 严格沿用 P0–P4

| ID | A/B | 时间 gate（95%区间上界） | 附加条件 |
|---|---|---|---|
| P0 | 改动前/改动后同 legacy；resident/旧 streamed 分开 | wall/denoise median≤1.02，wall P95≤1.05 | 新 hook/probe/thread/pool/clear=0 |
| P1 | 旧/新 executor 的等价布局与资源策略 | wall/denoise median≤1.03，wall P95≤1.05 | 同质量/格式/kernel，稳态框架 alloc/thread-create=0 |
| P2 | 新同 layout guard off/on | 同执行段 median≤1.05，P95≤1.10 | 总 wall 单列；不能为性能删安全检测 |
| P3 | 匹配条件的 OS-managed/新 streaming | 无预承诺加速百分比 | 需真实 paging 观测才能声称胜过 swap |
| P4 | 两个显式 K/G/P/D/Q/policy 组合 | 同上 | 策略收益不算框架零开销 |

P3/P4 宣称加速要求 speedup 的95%区间下界>1，同时质量/失败率/安全通过。
P1 不只 K 一样，必须核对 P/G/D/Q、startup、跨 pass、retention、text/VAE 策略；
不等价时改实验分类，不能降低 P1 门槛。所有阈值以12为准，后续只在12修订。

### 8.2 执行步骤

1. 冻结 workload 卡：tiny 排错、normal-target、largest-certified、repeated-request 各自用途；
   seed、prompt、shape、steps、format、所有数值开关填实，敏感 prompt 不公开。
2. 封存 before/after commit+dirty patch+untracked source 清单/hash、binary、编译器和依赖。
   没有可信 before build 就不能做 P0 声明；同一新库的 legacy 只适合 P1/P4 等适当实验。
3. quality/lifetime 先过；trace/audit/sanitizer 与 release timing 分开，两侧 instrumentation 一致。
4. matched ABBA blocks；20 matched pairs 是首次分析起点，tail 至少50次有效请求仍可能不足。
   提前约定 sample ceiling、bootstrap/quantile 方法、noise exclusion 和停止规则。
5. 按 ABBA block 重采样，不能把11个 pass 变成11个独立请求样本。上界≤门槛 PASS；
   下界>门槛 FAIL；其他 INCONCLUSIVE。不测到偶然通过才停止。
6. 每个 workload/cache condition 单独签核。process cold 不等于 SSD cold；
   request warm 不等于 retained weights。初次 compile/load 和必要 terminal cleanup 计入对应完整 wall。
7. 保存所有失败/超时记录；baseline timeout 是删失时间，不拿 timeout 阈值计算倍速。
8. 合成 tiny 或仿真只排错/选候选，不作为以上 gate 成绩。

### 8.3 低内存与 swap 专项

优先真实低内存专用机器，P3 两侧匹配外部负载/质量/cache 协议。
人工 pressure 是可选且需另行明确授权的实验，不是必须对用户本机执行的步骤。
限定资源、时长、磁盘余量和 thermal/pressure 停止阈值；不关 swap、不清用户 cache、不无界分配。

分开报告 logical reads、physical I/O、faults、compression、swapin/out、managed peak 和 sampled process peak。
系统计数不自动归因本进程；baseline 未发生可观测 swap 时只能比较当前低内存条件，不能宣传胜过 swap。
软件 Y 只测试 guard；模拟 pager 不能替代实机。无授权/设备/可信观测分别记 NOT RUN/SKIP/INCONCLUSIVE。

## 9. 每个 PR 的完成与发布检查单

```text
work_package: F1/F2/F3d/...（沿用11）
scope: legacy touched? explicit route? bounded?
contract: owner / borrowed resources / ABI / supported & rejected tuples
source changes: 每文件职责，是否改变 kernel/batch/cache
tests: CFG/CMP/RUN/FLT/MEM/GPU/PERF gate 下的具体子用例
evidence: build/source/checkpoint identity、命令、raw artifacts
performance: P0/P1/P2/P3/P4 = PASS/FAIL/INCONCLUSIVE/NOT RUN
release: registry unchanged | reviewed exact record
rollback: new route/record撤回；shared代码回归的修复/撤回方法
```

进入下一阶段必须有明确产物：

| 完成门 | 必备产物 | 缺失时 |
|---|---|---|
| F1/F3a | v2校验/同源/ownership tests + compiled construction | 不接公开执行 |
| F3d 实验完整请求 | normal-target 输出、质量、actual、cancel/repeat/cleanup | 不做 production |
| F6-layout | L0/L1 + L2质量/lifetime + P0/P1 + review | 保留 experimental |
| F5/F6-bounded | 全 site closure、完整L2/L3、P2、可靠观测 | 仅 layout-only |
| F9 preset | 限定硬件/工作负载证据、独立确认集、合法 tuple | 只显示候选 |

撤销 registry 只阻止未来新 route；默认 parser/kernel 若回归必须修复或撤回对应代码，
不能只关新开关。任何“不知道最后谁还在用”“只有总 bytes 没有 field proof”
“没有可信 baseline”“unknown 按0”都应阻断相应门。

## 10. 本机执行范围与可复现命令

当前已运行release/audit/test-hook隔离构建、host/API回归、v2真实Metal smoke、candidate-only完整LTX请求、
生命周期fault矩阵和20-pair tiny default P0；没有normal-target/large、legacy-streamed P0/P1或pressure，且不改
系统设置。最新实际结果见13第13.5节；tiny tuple PASS不替代其余P0–P4。

已有可执行命令（仓库根；native contract 先 build）：

```sh
env MLX_ROOT="$PWD/Python/lib/python3.13/site-packages/mlx" TURBOCIDER_NATIVE_ONLY=1 tools/native/build.sh
python3 -B tests/native/test_streaming_layout.py
python3 -B tests/native/test_ltx_streaming_descriptor.py
python3 -B tests/native/test_ltx_streaming_snapshot.py
python3 -B tests/native/test_streaming_contract.py
python3 -B tests/native/test_contract.py
python3 -B tests/repository/test_cpp_boundaries.py
```

descriptor 脚本不带 --checkpoint 时只测合成48-block fixture，不能写成真实 checkpoint 已重跑。
ASan/UBSan/TSan 与真实 Metal 命令见13；没有本轮执行结果时只保留历史证据，不升级当前资格。

## 11. 当前实现落点与下一施工切片

截至2026-09-16，F3d已经从native harness推进到内部candidate完整请求，但production状态仍为experimental：

| 设计对象 | 当前实现 | 仍需实现 |
|---|---|---|
| Compile | `StreamingMetadata` + generic descriptor/layout + `StreamingPlanView` | trusted artifact/content identity；多class/多stage |
| Bind | plan arrays复制给C executor；header/fd由request owner借用 | service级不可变artifact lease与registry记录 |
| Execute | candidate-only connector→Stage1→upsample→Stage2；G1/P≥1/K1..3 | audio/I2V/LoRA/ANE/近似、H3/Flux/Z适配 |
| Release | status destroy；metadata/first-fill/Stage1/upsample/Stage2/VAE/export故障恢复；session/process quarantine与owner retry已实测 | decoder内部/RGB转换故障；service线程迁移策略 |
| Observe | digest、resolved/actual layout、slot/fill/logical bytes | physical I/O、fault/compression/swap归因、whole-request upper |
| Gate | public constructor仍`streaming_layout_not_certified` | normal-target P0/P1、bounded P2/P3、reviewed registry |

### 11.1 生命周期当前覆盖与剩余项

显式opt-in真实GPU测试已使用同一candidate engine覆盖：metadata、first fill、Stage 1、upsample、Stage 2、
VAE、export failure、success→success、A(64×64)→B(128×64)→A、unsafe destroy session quarantine、engine
teardown process quarantine及owner retry。每次成功均验证actual layout、slot/fill counter和Stage-2 latent hash。
当前剩余：

1. failure：补decoder内部失败、RGB转换失败；media失败不能重新触碰已销毁handle。
2. service：验证请求线程迁移时owner mismatch保持fail-closed，受控worker可在原owner线程重试或隔离退出。

当前`generate()`会在发现quarantine时先调用status destroy重试，成功才继续，失败则保留state并拒绝。
剩余测试通过前，不把K上限扩到3以上，不开放session retention，也不新增用户可绕过资格的unsafe开关。

### 11.2 性能验收分层

- P0默认路径：tiny resident 20-pair已经通过并证明五类新counter为0；仍须对normal-target resident和原legacy
  streamed分别执行同样门禁，不能把tiny tuple扩大解释。
- P1布局框架：legacy与exact必须匹配P/G/K/D/Q、startup、loader并发、conditioning、VAE/export和cache状态；
  `Q=1`结果不能与legacy三loader混为matched P1。
- P2 guard：在同一exact layout上比较guard off/on；只测执行段和完整wall，不以删除安全检查换速度。
- P3 swap：只有在baseline发生可观测paging且两侧外部负载匹配时，才能讨论“比swap快”；本机未授权pressure，当前NOT RUN。

actual weight working set只是denoiser weight backing，不是process峰值。最新tiny中exact约4.27 GB、legacy约8.57 GB，
但两侧process peak都约7.1 GB，因此验收工具必须同时保留model estimate、sampled process footprint和系统级paging证据。

## 12. Campaign runner 与证据闭包（已实现）

为了让上述 P0/P1 规则能够被实际执行，新增了三个工具层文件：

1. `tools/native/run_streaming_campaign.py`：协调器只调度，不加载 native；baseline/candidate 各自拥有独立
   worker process 和 retained engine，按 ABBA/BAAB block 顺序执行，并在每个位置结束后追加并 fsync raw JSONL。
2. `tools/native/verify_streaming_campaign.py`：独立读取 bundle，不依赖 runtime 内部状态；验证 block 结构、pair
   对齐、quality/fault/environment/audit、source provenance、layout identity，并以完整 block 做 bootstrap。
3. `tools/native/capture_streaming_source_identity.py`：记录 commit、source manifest、dirty diff、untracked
   source，避免仅凭 dylib 路径宣称 before/after 可比。

runner 的输出必须至少包含 `campaign-policy.json`、`manifest.json`、`build-identity.json`、`raw-samples.jsonl`、
`warmups.jsonl`、`quality.json`、`faults.json`、`environment.json`、`audit.json` 和 `summary.json`；P1 另需
`semantic-equivalence.json`。warmup 失败、请求超时或 worker abort 不删除位置，而是记录状态并让 verifier
给出 FAIL/INCONCLUSIVE。runner 不启动 pressure、不修改 swap、不清 OS cache。

CPU-only synthetic backend 的独立测试覆盖持久 worker、ABBA/BAAB 顺序、timeout 后完整计划保留、partial audit
不误判 PASS、P0 provenance 缺失和 malformed block。clean source、独立audit和20-pair tiny real GPU已完成；
normal-target/large、legacy-streamed、P1/P2/P3仍缺，因此工具与tiny PASS不等于production资格完成。

## 13. Default audit 与当前性能证据

default audit已经从源码检查推进为可执行证据：独立audit build对framework hook、memory probe、worker、pool、
cache-clear/unload计数；release build完全不导出audit ABI。真实LTX默认resident请求及20-pair audit campaign均
观察到五类计数为0；显式fake streaming executor观察到pool=1、workers=2，防止“计数器未接线”假阳性。

同一冻结tiny tuple的release campaign使用clean `dev@ad343d4` baseline、10个ABBA/BAAB block和20 matched
pairs，wall median/P95及denoise median均通过12定义的P0上限，输出和Stage-2 latent逐pair byte-exact。证据路径
及精确数值见[13第13.5节](13-implementation-progress.md)。

这只关闭DEF-01和tiny default tuple的P0缺口。normal-target/large、原legacy-streamed、同义P1、whole-request
P2、低内存P3以及H3/Flux/Z adapter仍未完成；production registry继续为空。

## 14. 本轮 H3 descriptor 实施记录（2026-09-16）

本轮新增的 H3 descriptor 属于 F4 的 metadata/plan 子门，不改变 F4 的 production 完成定义。
它验证了通用 compiler 可以消费真实 H3 多 shard header，并将非连续 active block ID 按 active
ordinal 投影到 prefix/group/slot；它没有把 inactive block 当作可以任意跳过的 generic executor
工作项，也没有把 norms/AdaLN/text/activation/VAE 计入 streamed slot capacity。

实现文件与验收：

```text
native/models/h3_runtime/h3_streaming_descriptor.hpp
native/models/h3_runtime/h3_streaming_descriptor.cpp
tests/native/h3_streaming_descriptor_test.cpp
tests/native/test_h3_streaming_descriptor.py
```

测试用 sparse file 保留真实 H3 四矩阵 shape/byte range，因此可以验证 25 active blocks、四
shard 和大于 38 GiB 的逻辑文件，而不会读取或分配权重 payload。普通/ASan/UBSan/TSan 与
release native build 均通过。该测试不能替代真实 Metal H3 output parity、跨 forward prefetch
语义或 P0/P1；这些仍是 F4 后续门。

## 15. ABI v3 / carry 实施与验收增量（2026-09-16）

本轮把 H3 plan-only 与 generic executor 之间的一个关键空洞补成可测试协议：pass transition 不再藏在模型
内部。新增 v3 plan 后，layout-first 数据面可以在不改 v1/v2 adapter callback 的前提下表达“下一 pass 首组
提前填充并跨 boundary 保留”。

代码修改矩阵：

| 层 | 文件 | 修改 |
|---|---|---|
| plan | `streaming/layout.hpp/.cpp` | `PassTransition`、digest、K2/G1 compile guard、rotation capacity guard |
| safety | `streaming/slot_pool.*` | Ready ticket 查询、除指定 Ready 外全池 quiescent |
| executor | `streaming/context.*` | carry dispatch/incoming validation、pass rotation、boundary drain、step identity |
| C ABI | `stream_slot_c.h`、`streaming/c_bridge.cpp` | v3 plan/create；v1 callback 复用；async job 自持 blocks |
| H3 | `h3_streaming_descriptor.*` | request generation、v3 group/capacity projection、carry transition |
| tests | executor/C bridge/H3 descriptor | 顺序、奇偶 rotation、exactly-once、cancel/fill failure、ABI rejection |

新增验收要求：

- `reload` 的 3535 layout case、v1/v2 ABI 和默认 resident audit 必须保持不变；
- carry boundary 只能留下一个已验证 Ready ticket，其他 slot 必须 Vacant；
- 下一 `run_pass` 的 pass/step/group/slot 必须与预取 ticket 完全一致；
- K>2、多 pool、多 block group 在当前 revision fail closed；
- fake backend 只完成 F2/F4-plan 子门，不授予真实 GPU、H3 candidate 或 production 资格；
- clean commit 后重新构建 release，以独立 dev baseline 做质量、wall、denoise、audit 和 source identity 对照。

性能设计上，默认路径不创建 v3 plan/executor/worker；新增分支只在显式 H3 plan projection/candidate 路径发生。
carry 稳态不复制 group vectors，不创建线程，不扩 pool；其潜在收益来自把下一 pass 的首个 I/O 与当前 pass 尾部
GPU 工作重叠，而不是减少 source bytes。是否实际提速必须由真实 H3 P1 campaign 证明。

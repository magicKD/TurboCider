# 13 · 实施记录与未完成验收

[目录](README.md) · [实施任务](11-work-packages.md) · [验收合同](12-acceptance-playbook.md)

日期：2026-09-16。设计深化之后继续执行完整实现目标，本次已新增LTX真实模型adapter原型和实机测试，见第7节。完整目标仍是模型接入、正常运行和性能不回退，
不是以 plan-only 或合成测试替代最终交付。下表描述当前工作树，不代表已发布版本。

最新candidate-only完整请求接线、host/真实GPU证据见第12节；第10、11节为此前实现快照。
完整production资格、生命周期故障矩阵和正式性能验收仍未完成。
本次提交前审阅和新增回归见 [21 审阅与交接](21-review-and-handoff.md)；下文历史结果不自动等同于本次重跑结果。

## 1. 已落地的基础代码

| 模块 | 文件 | 当前能力 | 尚未覆盖 |
|---|---|---|---|
| 配置值类型与合并 | `native/core/streaming_contracts.hpp`、`native/runtime/streaming/config.cpp` | presence、stage replacement、manual K/G/P/D/Q、字段来源 | 完整profile硬件组合回归仍需补齐 |
| JSON/API/profile | `streaming_config.mm`、`request.mm`、`profile.mm`、`plan.cpp`、`results.mm` | schema2 execution.streaming、profile v2 reader、旧字段冲突、禁用走旧路径；raw requested与merged intent分开报告 | 模型metadata exact resolve尚未接API |
| 重复键检查 | `native/core/json_keys.*` | 捕获转义后同名JSON字段；用于请求/配置，不额外扫描旧checkpoint metadata | 不替代Foundation语法解析 |
| 纯布局编译 | `native/runtime/streaming/layout.*` | prefix/groups/class pools/modulo mapping、逐field maxima、checked bytes、v2 source/materialization/workload/pass identity、固定策略继承 | 全请求资源live intervals、construction/API投影尚未闭包 |
| C POD合同与adapter bridge | `native/core/stream_slot_c.h`、`native/runtime/streaming/c_bridge.cpp` | 版本化ticket/fence/completion、adapter回调、create/run_pass/finish/cancel/destroy；同一C++调度器；LTX内部入口已接入 | 完整异常/ABI测试矩阵及service quarantine接线 |
| Slot安全 | `native/runtime/streaming/slot_pool.*` | owner检查、generation、Ready/InUse/sealed多queue读者、提前复写拒绝、sticky poison | 与真实model backing/ledger的绑定 |
| 有界I/O | `native/runtime/streaming/io_executor.*` | persistent Q workers、固定job ring、completion mailbox、overflow latch、cooperative cancel+join | 非协作OS I/O的deadline/quarantine管理需补齐 |
| Stage执行器 | `native/runtime/streaming/context.*` | begin/run_pass/finish、单活动pool streamed、同class跨pass保留、ordered multi-class barrier、pass与step分离、lookahead/fence/drain | resident、request orchestration、strict memory schedule、service quarantine接线 |
| LTX metadata/fill/slot | `ltx_streaming_layout.*`、`ltx_streaming_slot.*` | 真实48-block metadata、source/destination/scratch区分、固定shared buffers、chunked pread、无分配fill | session construction与同一snapshot绑定、完整allocator上界/guard |
| LTX generic projection | `ltx_streaming_descriptor.*` | fd-only snapshot、48-block source/derived bindings、11-pass/workload identity；真实checkpoint host验证 | trusted content identity、公开session接线、同fd构造/运行与完整resource plan |
| LTX exact adapter | `ltx_streaming_adapter.inc`、`ltx_blocks.c`、`ltx_streaming_plan.*`、`ltx_session.mm` | versioned create、generic layout→C ABI投影、candidate-only request owner、connector→两stage→VAE/export完整请求、status destroy/quarantine | public production资格、完整故障/重复请求矩阵、normal-target性能和bounded guard |

`layout.*`当前输出只是权重布局，不生成完整memory upper，不做任何production授权。
StageExecutor的backing由adapter拥有，SlotSafetyTracker只维护内容状态，Vacant不会释放backing账。
执行器仍拒绝resident；当前已支持按连续 pool 区间插入 drain barrier、释放旧 pool、建立新 pool 的
ordered multi-class streamed 执行。它不是同时保留多个 class，也不允许跨 class lookahead；编译器支持的
resident、多 stage 或任意 DAG 仍不能误报为执行器已支持。

公共`tc_engine_create_model`的新manual请求仍是plan-only；generate/prepare在进入模型计算/卸载前明确返回
`streaming_layout_not_certified`。内部`tc_engine_create_model_candidate`现在可以执行首批LTX exact tuple，
但不构成production授权。这是开发期资格门，不是最终实现终点，后续须由真实证据和registry preflight取代。
legacy resident/streamed数值kernel与loader继续保留；本次只在外层create/run/cleanup增加新分支，未将旧loader改成通用executor。

## 2. 此前验证记录与本次证据范围

下表除另行说明外保留此前基础实现的验证记录；不自动覆盖之后persistent lifecycle/C bridge变更。
此前文档深化仅取回host结果，见2.2；后续本次实现已重跑build/sanitizer/Metal并新增LTX实模型用例，最新范围见第7节。

| 检查 | 结果 | 能证明什么 |
|---|---|---|
| native-only full build | PASS | 主库/CLI可编译链接 |
| `test_streaming_layout.py` | PASS：3535组有效小布局穷举；14组K/D/Q执行组合 | config/compiler与合成双读者安全、取消/partial create/short fill/drain等 |
| 同测试 `TC_STREAMING_SANITIZER=address,undefined` | PASS | 已运行用例未发现ASan/UBSan错误 |
| 同测试 `TC_STREAMING_SANITIZER=thread` | PASS | 已运行并发用例未发现TSan数据竞争 |
| `test_streaming_contract.py` | 8 tests PASS | 新API参数、精确布局不被预算重选、禁用结果兼容、冲突/类型/重复键 |
| `test_contract.py` | 81 tests，80 PASS+1 fixture SKIP | 原有contract回归；不等于模型性能回归 |
| `test_cpp_boundaries.py` | 2 tests PASS | core/runtime无Foundation/MLX依赖，host policy可编译 |
| `test_memory_execution.py`、`test_memory_plan_compiler.py` | PASS | 既有memory底座回归 |
| `test_streaming_metal.py`（沙箱外） | PASS | 真实Metal K1/K2/K3、3 passes、输出逐元素相等；仅合成权重 |
| `test_ltx_gpu_memory_hooks.py`（沙箱外） | PASS，见下文测试修正 | 既有LTX buffer hooks实机测试 |
| `compare_streaming_legacy_plans.py` | 18组新旧plan/error结果一致 | 隔离进程下6模型×3 legacy配置结果兼容，不证明运行速度 |

沙箱内两个Metal脚本仍可能返回`SKIP: no Metal device is available`；这与沙箱外真实设备可用并不矛盾。
不能把测试运行权限造成的SKIP当作实体机器无GPU，也不能把合成Metal PASS当作LTX/H3质量认证。

### 2.1 既有测试输入修正

LTX hook测试原先`budget=4096`，成功分配1024后，要求1536分配失败；2560没有超4096，因此实机断言失败。
本轮只将该拒绝样例改为申请4096（合计5120），未修改GPU allocator来迎合错误测试。
Python wrapper补充stderr输出，后续真实失败不再被CalledProcessError隐藏。修正后实机测试通过。

### 2.2 Persistent lifecycle / C bridge 最新host结果

本次读取此前仍未收取的`test_streaming_layout.py`执行结果，进程exit0：

```text
PASS streaming layout/config; 3535 exhaustive valid layouts
PASS streaming executor: 14 K/D/Q combinations, two independent readers, faults and cleanup
PASS: C11 persistent slot bridge, separate pass/step, exact pool retention and release
```

C11用例覆盖3槽/4组、两个分别调用的pass（step10/11）、8次fill、池只创建一次并最终释放。
这是同步fake C adapter，不是LTX实模型；跨pass异步fence/取消/异常和更多ABI反例仍需补齐。
上述历史host记录之后，第7节已重跑full build、ASan/UBSan/TSan和真实Metal；C11对象也已补齐同样的sanitizer编译flags。
阻塞join/drain仍不能被解释为已有硬deadline保证。详细待补合同见 [14](14-layout-first-integration.md) 第7节。

## 3. 基线与性能证据边界

已保留实施前既有CLI/动态库副本于本机`build/streaming-baseline-6BnvW2/`；它是本地构建产物，不提交仓库。
初始工作树已脏，不能将整个`git diff`归因于本轮，也不使用reset/checkout覆盖旧工作。

初始identity：

```text
HEAD: 804925cd9ed9248e16cc37a6120ac1045e38d01c
tracked diff SHA256: 703cc8ec61c2c5c495a0a79b7b829d9b1ed0a6608284ad769bc585364825161a
existing CLI SHA256: 0f1c9e025ea25a97461b0f73ef591847d66814c1af5492f04ff47abe7683c969
existing dylib SHA256: fbf4f2f8278d72f2081814860181c51d5c7d774df18d3eaa838cf0a7bd8dd9c6
```

上述tracked diff hash不包含旧untracked源文件，也不能单凭时间戳证明旧二进制与全部初始源码完全一致。
正式P0 campaign仍须完善build/source provenance，保留实际before二进制identity，不伪称已完成全源码封存。
当前未运行真实LTX/H3端到端ABBA，没有P0/P1/P2性能数据，也没有低内存L3/swap对照。
因此目前不能声明“新框架性能没有变差”，只能说默认接口回归和已有hot-path source contract通过。

## 4. 复现命令

```sh
env MLX_ROOT="$PWD/Python/lib/python3.13/site-packages/mlx" TURBOCIDER_NATIVE_ONLY=1 tools/native/build.sh
python3 -B tests/native/test_streaming_layout.py
env TC_STREAMING_SANITIZER=address,undefined python3 -B tests/native/test_streaming_layout.py
env TC_STREAMING_SANITIZER=thread python3 -B tests/native/test_streaming_layout.py
python3 -B tests/native/test_streaming_contract.py
python3 -B tests/native/test_contract.py
python3 -B tests/repository/test_cpp_boundaries.py
python3 -B tests/native/test_memory_execution.py
python3 -B tests/native/test_memory_plan_compiler.py
python3 -B tests/native/test_streaming_metal.py
python3 -B tests/native/test_ltx_gpu_memory_hooks.py
python3 -B tests/native/compare_streaming_legacy_plans.py \
  --baseline build/streaming-baseline-6BnvW2/libturbocider.dylib \
  --candidate build/native/libturbocider.dylib
```

MLX路径/基线目录是本机现有环境；其他环境应显式指定自己的有效路径。Metal测试需要真实设备访问权限，
编译通过或SKIP不能替代实机执行。压力测试未授权为普通测试的隐式步骤，本轮没有启动。

## 5. 下一实施顺序（完整目标仍未完成）

1. 已有真实LTX generic source/materialization/shape/pass投影，见第9节；下一步把snapshot升级为受信artifact身份并绑定session construction，保持同一源与计划。
2. 完成session/API/service exact preflight和实际报告；保留公开API资格门直到证据充足，不能只删除拒绝语句。
3. 对已接入的LTX原型补正常shape、真实text conditioning/输出、allocator/故障/cleanup覆盖；单槽startup语义与旧路线差异要明确。
4. 做真实LTX默认P0和严格同布局P1的ABBA；确认后再决定layout发布资格，当前tiny smoke不算该门通过。
5. 接H3双槽；审计active-block、跨block fusion与跨forward预取；再做Flux/Z-Image component staged。
6. 实现BudgetBridge与完整required-site/upper/schedule闭包，完成L2/L3及P2/P3；不要以合成bounded测试替代实机认证。
7. 补inspect/plan/simulate/sweep/verifier工具，真实evidence驱动registry/preset；全部要求证明后才能标记总目标完成。

目前F0/F1/F2有基础实现，F3已有真实LTX内部垂直切片但未完成公共路由和完整验收；F4–F9仍待推进。没有生产layout资格记录。

## 6. 前次布局优先设计深化记录

新增 [14整合](14-layout-first-integration.md)、[15模型施工](15-adapter-implementation-plan.md)、[16性能工具](16-performance-toolchain-plan.md)：
明确四张计划表、construction view、persistent pool/workers、pass输入借用生命周期、按K/Q分配scratch的接口约束，
以及H3跨pass预读/融合边界、Flux/Z组件先行、仿真与真实P0–P4的分工。
同步修正文档中“bridge尚未实现”“profile只支持v1”等过期表述。未改默认执行路径、未取消API资格门、未进行pressure测试。

本次文档验证：18份当前Markdown、153个本地链接、3段内嵌JSON及6份JSON示例通过；code fence/尾随空白与`git diff --check`通过。
归档原稿SHA-256仍为`826b9ce3089a5878a84533fc1336ab92043d6c12367af33311772a1face3e6cf`，未改历史正文。
这些检查只证明文档结构与示例语法有效，不证明新增设计已实现或性能达标。

## 7. LTX真实adapter实现续接（2026-09-16）

### 7.1 代码交付

- `ltx_streaming_layout.h/.c`：metadata-only枚举28个linear、6个attention和8个table实例；
  验证dtype/shape/offset/ConvRot metadata，保留真实source指针与独立destination身份；table相同source的多个实际副本不去重。
- `ltx_streaming_slot.h/.c`：owner创建shared Metal buffers、CPU表和每槽scratch；fill只使用固定spans，8 MiB chunked pread、协作取消、F32→BF16转换；不在refill重做JSON解析或malloc。
- `ltx_streaming_adapter.inc`：在`ltx_blocks.c`内部复用原kernel；bind字段而不复制数值实现；C bridge持久调度；派生表更新复用slot scratch。
- `ltx_native_create_streamed_v1()`：内部实验入口，接收exact编译结果，检查P≥1/G=1/K1..3/48块/11pass及实际容量；拒绝旧budget、guard hooks、ANE、Sol及未覆盖分支。
- `ltx_native_run()`：按global step接入两stage，upsample期间池保留，最后pass后finish；取消后poison，caller latent不更新。
- `ltx_gpu_streaming_boundary_completed()`：核实当前同步finish helpers已完成且无deferred batch；不新增每组GPU marker或device-wide wait。run_block的audio tasks先join，再报告两queue已完成reader。
- 错误清理：drain无法证明安全时保留sampler/stage资源和executor；status-returning destroy负责后续安全释放；新增owner检查。非协作I/O/join硬deadline、service隔离和完整fault injection仍未完成。

新入口仍不是生产资格，也不保证全请求内存上限。默认API仍拒绝manual generate/prepare；正常session尚未使用此入口。

### 7.2 Metadata实测结构（不是进程内存上限）

本机`ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors`：

| 项目 | 结果 |
|---|---:|
| block数 / 每block字段数 | 48 / 160 |
| GPU backing bytes / slot | 387,981,696 |
| CPU base table bytes / slot | 311,296 |
| scratch bytes / slot（按K份） | 147,456 |
| 全48块一次fill的source read bytes（含重复表） | 18,638,063,616 |
| describe读取的小型quant metadata bytes | 96,768 |

describe还需读取checkpoint header；最后一行不是所有metadata I/O总量。以上不含prefix以外组件、activation、MPSGraph、driver、control及输出，不能据此宣称整请求只用K×slot字节。

### 7.3 实际验证范围

| 检查 | 本次结果 | 限制 |
|---|---|---|
| native-only full build | PASS | 编译/链接，不是性能 |
| 通用compiler/executor/C11 bridge | PASS；3535布局、14组合 | synthetic |
| 通用ASan/UBSan、TSan | PASS；C11对象也已instrument | 只覆盖运行用例，不覆盖整个Metal/model库 |
| `test_ltx_streaming_layout.py` | PASS | 合成checkpoint的geometry/副本/BF16/取消/EOF/越界；真实checkpoint仅metadata |
| 同脚本ASan/UBSan | PASS | 新metadata/reader与C用例，不是全模型sanitizer |
| 同脚本`--metal` | PASS | 合成checkpoint、真实shared buffer，多次fill不换地址 |
| `test_streaming_metal.py` | PASS | K1/K2/K3、三pass真实GPU合成读者 |
| `test_ltx_streaming_model.py --slots 1/2/3` | PASS | 真实48块、两stage+upsample；详见下文 |
| 新API contract | 8 PASS | 公共manual仍plan-only |
| 原有contract | 81 run：80 PASS + 1 Wan fixture SKIP | 非端到端性能 |
| C++ boundaries | 2 PASS | host依赖和policy |
| legacy plan/error对照 | 18项一致 | 隔离进程，新旧saved binary，不证明速度 |

实模型用例统一P=1/G=1/D=K−1/Q=K；64×64×9、11steps、seed42、构造的16-row conditioning，不运行Gemma/text encoder和最终Video VAE/export。
Stage1 video/audio、upsampled video、Stage2 video/audio逐字节相等且finite；每次pool_creates=1、slot_bundles=K、fills=517。
追加suffix中途取消，验证caller latent不变、失败不可重用、destroy/重复destroy；错误旧budget/pass count在执行前拒绝。
最新构建上的K3复跑还断言错误线程调用destroy在触碰GPU前被拒绝，原handle保留，随后owner可正常运行/取消/释放。
这证明真实模型内核接入及tiny路径parity，不覆盖正常prompt成品、所有shape/branch，也不授予layout_validated。

### 7.4 性能观察，不是性能验收

两侧在同一candidate build中依次运行legacy和exact，未做ABBA、冷缓存协议或置信区间。下列仅保留排错用raw wall，不能叫P0/P1通过：

| 测试 | legacy total (s) | exact total (s) |
|---|---:|---:|
| K3首轮，缓存/初始化偏差显著 | 11.6664 | 8.02154 |
| K3第二轮 | 8.16800 | 7.89973 |
| K1 | 17.9865 | 17.0192 |
| K2 | 10.4849 | 10.0563 |
| K3最新owner检查构建复跑 | 8.16327 | 7.89722 |

total包含create、两stage、upsampler和destroy，不含测试前descriptor编译，也不含text/VAE/export。
K1旧路线在prefix之后同步读首槽，新路线可在prefix期间派发：即使K/P相同，也存在startup策略差异，不能把上表算纯框架收益。
并发运行的短synthetic Metal检查、温度、文件缓存均未按正式campaign控制。不得从这里推断低RAM/swap收益或默认无回退。

### 7.5 复现新检查

```sh
python3 -B tests/native/test_ltx_streaming_layout.py
env TC_STREAMING_SANITIZER=address,undefined python3 -B tests/native/test_ltx_streaming_layout.py
python3 -B tests/native/test_ltx_streaming_layout.py --metal
python3 -B tests/native/test_ltx_streaming_layout.py \
  --checkpoint models/LTX-2.5/diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors
python3 -B tests/native/test_ltx_streaming_model.py \
  --checkpoint models/LTX-2.5/diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors \
  --upsampler models/LTX-2.5/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors \
  --video-vae models/LTX-2.5/vae/ltx-2.5-video-vae-conv-bf16.safetensors --slots 3
```

把末尾slots分别改为1/2即可覆盖其他已测布局。真实GPU脚本为显式opt-in，不加入默认host suite；不施加系统pressure。
本节历史测试的descriptor identity为EXPERIMENTAL-unqualified；第9节已替换为真实metadata snapshot投影，仍未保存完整checkpoint digest/正式environment bundle，不能用于签发认证。

最后一次K3/owner/cancel复跑使用的本机候选动态库SHA-256：

```text
143d6ad80828dd55801c7a94fca903f21a6a12f214a9a673f39b13701c8204d2
```

此构建也重新通过81项contract（1项fixture SKIP）和18项legacy plan/error对照。
源码初始已有大量用户改动，不能将完整git diff归因于本次；该binary hash只标识本次候选，不补足正式before/after source provenance。

## 8. 本轮布局优先框架规划（2026-09-16，仅文档）

按本轮最新请求继续细化设计，并未继续修改runtime或放开公共执行入口；已有完整实现目标仍未完成。
新增 [17框架合同](17-layout-first-framework.md) 和 [18施工矩阵](18-code-change-matrix.md)，重点为：

- 控制面/数据面/证据面的单一职责，四种布局与source/backing/binding/generation的分离。
- metadata快照、身份/失效合同、真实source bytes与destination/capacity的区分。
- request coordinator与StageExecutor的边界，公共API资格、实验构建和prepare/generate的不同访问模板。
- LTX独立request exact owner、status-returning销毁、跨pass/upsample与VAE的资源交接、quarantine完整持有。
- multi-slot派发与最后reader合同、K/Q scratch计数、class/pass边界及默认热路径保护。
- 按现有文件/调用点列出的F1/F3d实施顺序、H3/Flux/Z接入要求、工具链及新增验收测试映射。

同步更新README导航、14的下一步状态和15中已过期的拟议LTX符号，避免把已有内部adapter重复规划成尚未开始。
性能阈值继续引用12，未新建更宽松门槛。没有新增真实模型运行、P0/P1配对数据或低内存/swap证据；
第7节tiny smoke仍仅是历史功能/排错证据，不能据此宣称性能不回退或快于swap。

本轮文档检查通过：20份非归档Markdown、170个本地链接、3段内嵌JSON和6份JSON示例；
code fence、尾随空白、末尾换行及`git diff --check -- docs/design`通过。归档原稿SHA-256仍与第6节相同。
上述仅是文档结构/语法检查，不是runtime、模型质量或性能验收。

## 9. 本轮实现增量：真实descriptor与快照投影（2026-09-16）

### 9.1 通用compiler v2

新增 `SourceArtifact`、`SourceRange`、`Materialization`、`PassSpec` 及 `FieldSpec.materialization`：

- destination backing bytes、对齐后的slot capacity、logical content bytes、source range bytes分别保留；
- F32→BF16、row copy等派生字段必须引用同一block中更早的materialized字段，不能同时声明source reads；
- source offset/length、artifact identity、dtype/shape、pass模板、workload字段进入canonical layout digest；
- source读取量未知时返回optional null，不把合成descriptor的destination bytes冒充文件读取量；
- artifact身份区分mutable snapshot与content SHA-256；snapshot不授予内容认证；
- `materializations_complete`只表示descriptor字段闭包程度，不等于whole-request upper或execution qualification。

`StageLayout`现有内容字段更名为 `suffix_content_bytes_per_pass`；新增optional `source_read_bytes_per_pass` 和
`prefix_source_read_bytes`。这是内部C++计划模型的语义改正，旧C ABI仍保持v1布局。

### 9.2 LTX metadata-only projection

新增 `native/models/ltx_runtime/ltx_streaming_descriptor.hpp/.cpp`、
`tests/native/ltx_streaming_descriptor_test.cpp`、`tests/native/test_ltx_streaming_descriptor.py`。

`StreamingMetadata`持有checkpoint header和fd，不创建GPU buffer、不映射整份checkpoint、不读取完整权重；
当前读取48个block的descriptor所需小型metadata。它记录stat/inode/size/time的snapshot identity，并在describe前后
检查fd与路径状态；观察到snapshot身份变化时拒绝继续使用。当前失效反例实测为truncate；原地改写/替换的完整矩阵仍需补齐。
该检查不是内容hash认证，也不保护正在读取的文件免于并发修改，production registry仍需可信artifact identity。

投影输出实际LTX 48 blocks、11 passes（stage1/upsample boundary/stage2）、workload shape/token/text rows、
ConvRot format、F32→BF16表转换、GPU row派生关系和source ranges。测试使用fd-only mapping确认metadata路径没有整文件mmap。

### 9.3 新测试结果

| 检查 | 结果 | 证据边界 |
|---|---|---|
| `test_streaming_layout.py` | PASS：3535布局、descriptor反例、executor/C bridge | host/synthetic |
| `test_ltx_streaming_descriptor.py` | PASS：合成48 block + 本机真实checkpoint；truncate invalidation | metadata-only，不是执行质量 |
| ASan/UBSan | PASS：generic suite及LTX descriptor suite | metadata/compiler/executor/C bridge测试范围，不是全模型instrumentation |
| TSan generic streaming | PASS（本轮） | generic executor/C bridge范围 |
| native-only full build | PASS；candidate dylib SHA-256 `abc1ec18acbf2db86ce3fc14487e7943189ec1b101ed3974bcf565fcad51c2d7` | 编译/链接，不是性能 |
| real LTX K1/K2/K3 exact adapter | PASS；各布局stage1/upsample/stage2 latent byte-exact；cancel/owner cleanup | 64×64×9、fabricated16-row conditioning、无Gemma/VAE/export |
| API/default回归 | streaming 8 PASS；原contract 80 PASS+1 fixture SKIP；C++ boundaries 2 PASS | 不证明性能 |
| isolated legacy plan/error对照 | 18项一致 | saved baseline C API行为对照，不是P0 |
| 原LTX metadata/fill与memory底座 | LTX layout、memory plan compiler、memory execution均PASS | 对应host用例 |

本轮真实descriptor projection的source bytes per pass为 `18,249,770,624`（实际checkpoint），
合成fixture为 `91,349,200`；这些是logical source ranges，不是physical disk traffic或进程峰值。
本机真实GPU运行需沙箱外Metal权限；沙箱内报告 `no Metal device is available`，该wrapper原始退出码为1，
属于设备访问失败；随后沙箱外测试通过。不能把该次沙箱内运行写成PASS或伪造脚本实际未输出的SKIP。

新增host复现命令（其余build/Metal/contract命令沿用第4/7.5节）：

```sh
python3 -B tests/native/test_ltx_streaming_descriptor.py
env TC_STREAMING_SANITIZER=address,undefined python3 -B tests/native/test_ltx_streaming_descriptor.py
python3 -B tests/native/test_ltx_streaming_descriptor.py \
  --checkpoint models/LTX-2.5/diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors
```

最终构建顺序运行的raw smoke total如下，仍不属于ABBA/P0/P1：

| K | legacy total秒 | exact total秒 |
|---|---:|---:|
| 1 | 17.8790 | 17.0334 |
| 2 | 10.4957 | 10.3770 |
| 3 | 8.10822 | 7.91009 |

每次均1套pool、K个slot bundle、517 fills。total不含metadata compile、text、VAE/export；文件缓存、温度、
顺序和CPU竞争未控制，不能从本表得出无回归或加速结论。snapshot投影已替换测试手写descriptor，但未解决下述同fd接线问题。

### 9.4 当时仍未完成的端到端工作（当前状态见第12节）

1. `LtxNativeSession`尚未把公共`generate/prepare`接到exact construction view；API仍保持`streaming_layout_not_certified` gate。
2. descriptor checkpoint identity仍是metadata snapshot；需把可信artifact/hash/registry与session preflight统一，不能仅依赖路径状态。
   第9节测试当时仍用v1独立打开/解析；第11节已完成v2借用header/fd及native专项测试，但尚未接公开session。
   测试前后snapshot检查不等于执行已消费同一fd；完整helper来源与ownership接缝必须在公共路由开放前处理。
3. LTX正常Gemma/text conditioning、真实视频VAE/export、audio/I2V/LoRA分支尚未纳入exact资格。
4. request coordinator、whole-request resource closure、BudgetGuard和quarantine service ownership尚未完成。
5. 尚无可信before/after normal-target ABBA P0/P1、P2 guard、P3低内存/swap campaign；K3 tiny timing只作排错观察。

该阶段提出的session exact接线和tiny完整质量已推进到第12节；生命周期、P0/P1、公共registry和bounded release仍未完成。

## 10. 布局优先设计续补与构建核对（2026-09-16，历史快照）

本节记录文档深化阶段；其后实现及最新证据见第11节，不将两次构建的结果混用。
当时新增19/20并更新README、17/18，仅修改文档，runtime来自既有工作树。

当时发现：v2已存在，但入口校验未与v1对齐、model test尚未调用v2，正常session仍是旧入口。
当时native-only build、3535布局、metadata合成fixture、streaming 8 tests、原contract 80 PASS+1 SKIP、
C++ boundaries 2 tests均通过；未运行真实Metal/性能campaign。

当时candidate dylib SHA-256：

```text
c4242f5bd82f90edd68683efa06ee803c161c7972d2d553d435840f2f7600642
```

本节发现的v2校验和borrowed cleanup问题已推进到第11节的实现，不再作为当前完全未着手的工作。

## 11. LTX v2同源构造与真实GPU验证（2026-09-16）

目标仍是完整模型接入、正常运行和默认性能非劣；本节完成F1/F3a的具体接缝，
没有将整个目标缩减成tiny smoke，也没有取消公开API资格门。

### 11.1 实现变更

- `ltx_streaming_adapter.inc`：v1/v2共用exact options/plan structural validator；
  在metadata/GPU构造前检查ABI、out、generation、K/Q/D/P、group/order/block/capacity、pass count与计数溢出。
  metadata相关容量/几何校验仍在真实describe之后，不把结构合法当成布局已匹配。
- `ltx_safetensors.h/.m`：fd读取的header保存device/inode/size/mtime/ctime snapshot；
  新增`ltx_st_validate_snapshot_fd`，`ltx_st_map_fd`验证同一个未变化的文件后duplicate/mmap。
  这不是content hash或并发写隔离。legacy header路径不增加文件probe。
- 内部`ltx_st_header`结构扩展，所有依赖它的编译单元已重编译；TurboCider公共C ABI及exact options v1/v2结构未改。
- 修复header保留path/metadata字符串分配失败时，已转移的tensor数组被局部cleanup和header cleanup重复释放的问题；
  test-only strdup故障注入覆盖tensor name/path/config/gemma_config。
- `ltx_blocks.c`：v2在构造和stage边界复查snapshot；stage结束source失效时不向caller latents提交成功。
  exact connector复用ctx持有的header/mapping，不再按checkpoint路径重开；legacy connector分支仍保持原方式。
- v2 source/header借用关系未改变：native拥有duplicate mapping/fd，caller拥有header与原fd，安全destroy之前caller不能释放它们。
  增加GPU构造前的`ltx_streaming_validated`取消边界，便于host独立测试完整metadata校验和失败cleanup。
- 新增`test_ltx_streaming_snapshot.py`、`ltx_streaming_snapshot_test.cpp`、`ltx_safetensors_failure_test.m`并接Makefile；
  实模型wrapper新增`--exact-api 1|2`与`--conditioning synthetic|connector`，默认测v2。
  connector用例由fabricated16-row projections经过真实connector得到1024-row context；不是Gemma真实prompt。

### 11.2 本轮实际证据

最终candidate dylib SHA-256：

```text
94dea9b1733853199dd86dbef067784fd28d34c3336c5c8cc992437df6755caf
```

| 检查 | 结果 | 范围 |
|---|---|---|
| native-only build | PASS | 当前源码编译链接 |
| streaming generic suite | PASS；3535布局、14组executor、descriptor/C bridge | host；ASan/UBSan与TSan也重跑通过 |
| LTX metadata/fill suite | PASS | geometry/copy/BF16/cancel/short-read/range反例 |
| LTX descriptor ASan/UBSan | PASS | 合成48-block/11-pass、fd-only、truncate失效；未重跑独立真实checkpoint descriptor命令 |
| snapshot/ABI suite | PASS | v1/v2拒绝、full metadata校验后取消、错误容量拒绝、同大小错误fd、in-place/truncate/replace、重复取消无fd泄漏 |
| parser fault injection ASan/UBSan | PASS | name/path/两种metadata分配失败；legacy与fd两个入口；无double-free/fd泄漏 |
| v2真实Metal K1/K2/K3 | PASS | 64×64×9，两stage/upsample video/audio逐字节一致、pool一次、517 fills、取消保持caller输入、sticky failure、owner destroy |
| v2 K3真实connector | PASS | connector video/audio/mask与后续两stage逐字节一致；原fd/header在destroy后仍可验证 |
| v1 K3真实connector回归 | PASS | 共用validator/connector改造后的旧内部入口回归 |
| shader失败cleanup | PASS（各实模型运行） | metadata/map校验后故障返回空handle，borrowed snapshot仍可用于下一次构造 |
| API contract | streaming 8 PASS；原contract 80 PASS+1 SKIP | Wan base fixture缺失；不是性能证据 |
| C++ boundaries | 2 PASS | host边界回归 |
| isolated legacy plan/error | 18项一致 | saved baseline对照；不是P0 |

**Sanitizer范围**：snapshot脚本的parser故障注入直接编译真实parser并instrument；
v1/v2接口测试链接release dylib，不声称完整native/GPU已经ASan。generic/descriptor另有独立instrumented运行。
Metal测试经授权在沙箱外执行，无人工pressure、swap配置变更或真实模型文件写入。

### 11.3 Raw smoke时间（不能作为P0/P1或加速声明）

单位秒；同一candidate中的legacy与exact顺序运行，只有单次样本，缓存/thermal/CPU竞争未匹配。
部分host编译/测试与smoke交叠；此表仅排错，不与第9节不同build的数值直接计算回归。

| API/条件 | K | legacy total | exact total |
|---|---:|---:|---:|
| v2 synthetic | 1 | 17.8644 | 17.1386 |
| v2 synthetic | 2 | 10.8242 | 10.4008 |
| v2 synthetic | 3 | 8.09717 | 7.87223 |
| v2 connector | 3 | 12.0318 | 9.47254 |
| v1 connector | 3 | 9.87336 | 9.59812 |

total包含本harness的native load/可选connector/两stage/upsample/destroy，
不含metadata compile、Gemma、真实VAE decode、MP4/export；不满足完整请求计时。
v2 connector的legacy首次connector耗时2.96441秒、exact随后0.626304秒，
而v1复测已变成0.690589/0.649681秒，足以说明不能用顺序单次结果宣称框架带来数倍connector加速。

### 11.4 复现命令

```sh
env MLX_ROOT="$PWD/Python/lib/python3.13/site-packages/mlx" TURBOCIDER_NATIVE_ONLY=1 tools/native/build.sh
python3 -B tests/native/test_ltx_streaming_snapshot.py
env TC_STREAMING_SANITIZER=address,undefined python3 -B tests/native/test_ltx_streaming_snapshot.py
env TC_STREAMING_SANITIZER=address,undefined python3 -B tests/native/test_ltx_streaming_descriptor.py
python3 -B tests/native/test_ltx_streaming_layout.py
python3 -B tests/native/test_streaming_layout.py
env TC_STREAMING_SANITIZER=address,undefined python3 -B tests/native/test_streaming_layout.py
env TC_STREAMING_SANITIZER=thread python3 -B tests/native/test_streaming_layout.py
```

实模型（需要Metal设备访问权限；K1/K2把--slots改为1/2，v1回归改--exact-api）：

```sh
python3 -B tests/native/test_ltx_streaming_model.py \
  --checkpoint models/LTX-2.5/diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors \
  --upsampler models/LTX-2.5/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors \
  --video-vae models/LTX-2.5/vae/ltx-2.5-video-vae-conv-bf16.safetensors \
  --slots 3 --exact-api 2 --conditioning connector
```

### 11.5 当时未完成项（已由第12节部分推进）

1. `LtxNativeSession`、API/service尚未绑定exact request owner，generate/prepare仍拒绝active manual。
   legacy void deleter不能承担unsafe owner；必须先接status-returning destroy与整体context保留。
2. text/Gemma、upsampler/VAE等独立artifact的身份/所有权、真实输出、正常shape和重复/失败请求仍需完整验证。
   connector同源已修复，不意味着整个pipeline snapshot闭包完成。
3. source仍是stat snapshot，不是可信content注册；field级construction view、resource plan及资格registry尚待接线。
4. 构造/运行的完整allocation/late-reader/quarantine故障矩阵、service eviction/engine_free安全终态尚未闭环。
5. 可信before/after normal-target ABBA P0/P1、guard P2、低内存P3、策略P4都未完成。
   本轮不能声明“性能保证不回退”“快于swap”或bounded发布。
6. H3/Flux/Z-Image接入和完整预算guard仍属于总目标，不能以本节LTX native smoke替代。

其中request-scoped owner、完整LTX candidate请求和真实prompt到成品输出已在第12节完成；
production gate、故障矩阵、normal-target与bounded验收仍保留。

## 12. dev合并后的LTX完整candidate接线与复核（2026-09-16）

本节是当前状态。`dev@ad343d4`已通过merge commit `566f7a6`进入`feat/stream`；
唯一文本冲突在native contract测试，解决时同时保留memory-constrained与Z-Image streamed合同。
LTX exact的新增执行能力只授予内部candidate constructor，public model constructor继续fail-closed。

### 12.1 代码接线

- `ltx_streaming_plan.hpp/.cpp`新增`StreamingPlanView`：request-scoped拥有metadata/header/fd、generic
  descriptor、immutable layout、稳定capacity/group数组、`tc_stream_stage_plan_v1`与v2 native options。
  native create会复制plan数组，但borrowed header/fd必须活到status-returning destroy成功。
- `LtxNativeSession`新增`LtxExactRequestState`与`LtxExactRequestOwner`。exact handle不进入legacy void deleter；
  success在Stage 2后显式destroy，再进入VAE/export。异常栈展开时先重试安全destroy，仍无法证明安全则把
  handle、metadata、layout整体转移到session quarantine，避免borrowed资源先失效。
- candidate首批tuple限定为C/Metal GPU、dense T2V、无audio/I2V/LoRA/memory guard/Sol/sparse/text pruning、
  `G=1`、`P>=1`、`K=1..3`；unsupported组合在checkpoint/GPU构造前拒绝。
- connector、Stage 1、upsampler、Stage 2全部使用同一个request-owned exact handle；legacy resident、
  component-staged和旧budget streamed仍直接使用原`denoiser_`路径，不构造metadata/layout/executor/worker。
- `tc_engine_create_model_candidate`设置不可由request/profile控制的内部authority；
  `tc_engine_create_model`和public prepare仍返回`streaming_layout_not_certified`。
- actual result现在报告`c_metal_exact_v2`、layout digest、actual counters，并把
  `plan.streaming.resolved_layout/actual_layout`标记为`executed_exact_v2`；这只是candidate执行事实，
  不是production registry资格。

### 12.2 最新源码验证

最终重建native dylib SHA-256：

```text
5d72812ec38af6e6164e7ff318dc362ab7ccbef14091c90ae024bb30107990c2
```

最终构建再次完成真实exact请求，`plan.streaming`同时报告requested、resolved和actual layout，
`resolution_state=executed_exact_v2`、`authority=private_candidate_constructor`；public gate合同仍通过。

| 检查 | 结果 | 备注 |
|---|---|---|
| 最新源码native-only build | PASS | 显式使用本机MLX SDK；全量重编译 |
| `make test-streaming-host` | PASS | 3535 layouts、14 K/D/Q、C bridge、LTX metadata/descriptor/plan view |
| `make test-streaming-contract` | PASS | snapshot、8 API contracts、candidate/public gate |
| native contract | 82项：81 PASS + 1 Wan fixture SKIP | fixture缺失；无LTX失败 |
| memory 13项集合 | PASS；2个Metal hook在沙箱内SKIP | accounting到execution、H3/LTX hooks |
| repository boundaries及后半模型合同 | PASS | layout/independence/C++ boundaries/inventory等 |
| `test_video_timing.py` | 沙箱内writer失败；沙箱外1 PASS | 环境限制，未修改视频代码 |
| exact完整真实请求 | PASS | Gemma cache/connector、两stage、VAE、MP4 export全部完成 |
| latent/video质量 | PASS | Stage-2 BF16 byte-exact；9帧RGB逐帧相同 |

`make test`在沙箱内运行到AVFoundation writer时停止，因此不把一次命令写成全量PASS；
被中断后的memory/model/inventory集合已独立补跑通过。沙箱内Metal不可见，真实LTX请求经授权在实体GPU环境运行。
没有人工memory pressure、swap设置修改或模型文件写入。

### 12.3 默认路径相对dev的短回归证据

同机、真实LTX、64×64×9、11 steps、seed 42、同prompt与conditioning cache；
dev基线dylib SHA-256为`b5334c0b790c7e727aafd5db842ca3f9746399131109386ade8fe403edec3892`。
首次dev resident受冷页缓存影响为41.1871秒，不与后续热样本直接比较。缓存热后的交错复测：

| 默认resident | wall秒 | denoise秒 | current/dev |
|---|---:|---:|---:|
| dev | 9.039951 | 6.915945 | 1.000000 |
| current | 9.069556 | 6.918010 | wall 1.003275；denoise 1.000298 |

当前分支的default request没有descriptor/executor/worker/probe，以上单样本差值为+0.33% wall、+0.03% denoise，
未观察到相对dev的默认路径回退。它只是tiny短回归，不具备20-pair/置信区间/normal-target P0资格。
已有Z-Image七个warm样本对照见21第6.1节，同样未观察到merge回退。

### 12.4 matched legacy/exact短对照

缓存热状态、相同完整请求；legacy由8 GiB budget解析为`P=8/K=3`，实际使用三loader；
exact显式`P=8/G=1/K=3/D=2/Q=3`。第二次交错样本：

| 路线 | wall秒 | denoise秒 | estimated weight working set | fills |
|---|---:|---:|---:|---:|
| legacy streamed | 8.519114 | 7.674939 | 8,570,908,800 bytes | 437 |
| exact v2 | 8.511698 | 7.489339 | 4,271,222,912 bytes | 440 |

exact/legacy ratio为wall 0.99913、denoise 0.97582。Stage-2 BF16两侧SHA-256均为
`7db71bf03027942b53af69ab914714583fe8f8d4c3ed2faf60f4aa4ed43f28fd`；最终视频9帧
correlation/cosine=1、MAE=0、motion relative error=0。layout digest为
`40b13657a746d27a9a44ec11f53485ac4163f23e92408eb7aef2bca1fdfca38a`。

process peak约7.10/7.11 GiB，说明weight working-set估计不是whole-process硬上限；
logical bytes与fill定义也不同，不能只凭本表宣称内存减半或普遍快于swap。该结果属于tiny matched smoke，
不是正式P1，更不是低内存P3。

### 12.5 当前未完成项

1. session级success→success、shape A→B→A、Stage 1/Stage 2取消、VAE边界取消和export失败恢复已覆盖；
   仍需可重复的unsafe destroy/quarantine故障注入、engine teardown和metadata/first-fill/upsample取消。
2. 把mutable-file snapshot升级为可信artifact/content identity，并处理service级隔离；当前stat snapshot
   不能防并发原地改写，process-exit leak只是避免use-after-free的最后防线。
3. 完成normal-target/largest/repeated-request release ABBA P0/P1，补P95与置信区间；当前tiny不签资格。
4. 完成whole-request resource closure、BudgetGuard与可靠swap/pressure观测后才能做P2/P3和bounded承诺。
5. exact report已有actual layout，但production registry仍为空；不得用candidate authority绕过public gate。
6. 按同一adapter contract推进H3 K2/G1，再做Flux/Z-Image component-staged或model-specific block adapter。

### 12.6 同一engine生命周期矩阵（2026-09-16）

新增显式opt-in目标：

```sh
make PYTHON=Python/bin/python3 test-ltx-streaming-lifecycle \
  MODEL="$PWD/models/LTX-2.5" OUTPUT=/private/tmp/turbocider-ltx-lifecycle
```

该目标不会进入默认`make test`，不施加人工内存压力；它使用一个candidate engine完成整个序列，
因此不能用进程退出掩盖request owner泄漏。最终真实GPU结果：

| 顺序/故障 | 结果 |
|---|---|
| Stage 1 streamed block取消→下一请求 | PASS |
| A success→A success | PASS；layout和Stage-2 latent hash相同 |
| A(64×64)→B(128×64)→A | PASS；digest A/B不同，返回A后恢复原digest和latent hash |
| Stage 2 streamed block取消→下一请求 | PASS |
| `video_vae`边界取消→下一请求 | PASS；exact handle已在VAE前安全destroy |
| output父路径为普通文件导致export失败→下一请求 | PASS |
| 每次actual/resolved layout与counter | PASS；P8/G1/K3/D2/Q3、40 groups、11 passes、440 fills |

所有A/recovery成功请求的Stage-2 latent SHA-256均为
`7db71bf03027942b53af69ab914714583fe8f8d4c3ed2faf60f4aa4ed43f28fd`。
A layout digest为`40b13657a746d27a9a44ec11f53485ac4163f23e92408eb7aef2bca1fdfca38a`，
B为`9c1b0b90ab16f89541625a7284489a09b24557f1f88bfb38d8c77505a57120c8`。
热态A/recovery wall约8.30–8.35秒；首个A为10.12秒，不用于正式性能结论。

同时修正session quarantine重试：若异常析构首次无法证明安全，下一次同owner `generate()`先调用
status-returning destroy重试；成功才继续，仍不安全则保留完整state并返回
`streaming_worker_quarantined`。不强制释放borrowed metadata，也不新增用户可控的unsafe开关。
当前尚无确定性session级unsafe-drain注入，因此此分支的release资格仍未完成；native/C bridge层已有
错误线程与drain失败保留handle测试，不能冒充session级覆盖。

## 13. 生命周期闭环与 campaign 工具落地

### 13.1 exact session 生命周期

test-hook build 已补确定性 unsafe destroy 和 first-fill 注入；release build 不定义
`TURBOCIDER_ENABLE_TEST_HOOKS`，也不导出任何 `tc_engine_test_ltx_*` 符号。真实 GPU 生命周期矩阵新增：

- metadata、first fill、Stage 1、upsample、Stage 2、VAE 边界取消后恢复；
- unsafe destroy 首次失败后 session quarantine、下一请求拒绝、owner-thread 后续重试恢复；
- engine teardown 仍不安全时转交 process quarantine，test-only owner retry 后安全 drain；
- process quarantine 使用 request state 内 intrusive link，不在低内存失败路径额外分配 bookkeeping node。

完整 fault artifact：
`/private/tmp/turbocider-ltx-lifecycle-intrusive-quarantine-20260916/lifecycle/lifecycle-summary.json`。
release 生命周期 artifact：
`/private/tmp/turbocider-ltx-lifecycle-release-20260916/lifecycle/lifecycle-summary.json`。
两者的 Stage-2 latent SHA-256 均保持
`7db71bf03027942b53af69ab914714583fe8f8d4c3ed2faf60f4aa4ed43f28fd`。

### 13.2 T2/T3 可执行工具

新增：

- `tools/native/run_streaming_campaign.py`：两个独立持久 worker、ABBA/BAAB、增量 raw JSONL、
  timeout/abort 完整保留、quality/fault/environment/audit/manifest bundle；
- `tools/native/verify_streaming_campaign.py`：manifest/identity/quality/audit 硬门、按 block bootstrap、
  P0/P1 阈值和 PASS/FAIL/INCONCLUSIVE；
- `tools/native/capture_streaming_source_identity.py`：clean export 或 dirty worktree 的可复现源码身份；
- `tests/native/test_streaming_campaign_verifier.py`：持久 worker、block 顺序、失败、timeout、partial audit、
  P0 provenance 反例；
- `tests/native/test_streaming_source_identity.py`：导出树稳定 hash、非法 commit、当前 dirty/untracked 捕获。

这些工具不施加 pressure，不改 swap，不把 synthetic PASS 当真实 GPU 资格。

### 13.3 真实 LTX tiny campaign smoke

沙箱内首次运行在 baseline warmup 明确返回 `Metal GPU unavailable`；runner 修正为仍生成完整
aborted bundle，verifier 给 `INCONCLUSIVE`，不把设备不可用当性能 FAIL 或 PASS。

沙箱外运行 artifact：
`/private/tmp/turbocider-streaming-campaign-smoke-metal-20260916`。配置为真实 LTX
64×64×9、11 steps、两个独立 public resident worker、2 个 ABBA/BAAB block、4 matched pairs。
结果：8/8 measured requests 成功，所有 matched Stage-2 latent byte-exact，无失败/timeout；candidate
每次均报告 `block_streaming.enabled=false`、slot allocations/refills 为0。

| tiny 指标 | dev baseline | current candidate | candidate/dev |
|---|---:|---:|---:|
| request wall median | 42.083178 s | 23.183768 s | 0.55090 |
| request wall P95 | 43.949082 s | 24.122947 s | 0.54888 |
| denoise median | 8.426679 s | 8.692648 s | 1.03156 |
| denoise P95 | 8.724369 s | 9.741839 s | 1.11662 |

wall 被两侧不同的 pre-model/model-load 行为主导，不能拿 0.55 比值宣称框架加速；denoise 的4-pair
小样本则提示 current 约 +3.16% median，95%区间仍宽。verifier 最终为 `INCONCLUSIVE`，原因包括：
样本远少于正式 normal-target 要求、environment/audit 为 partial、dev saved binary 没有完整 source
manifest、candidate 是 dirty build。该信号必须在可信 clean before/after、更多 block 和隔离环境下复测；
在复测前不能声称正式 P0 已通过，也不能用 wall load 差异掩盖 denoise 信号。

随后从 `dev@ad343d4` 的 clean `git archive` 重新构建 baseline，绑定 source manifest
`20958d6f2cfe1ac8b49e67ef4c790650d55714d2b20c97e49f4e2afb28dc2b77`；baseline dylib
SHA-256 为 `6a95b7f6b87961a1b0aafb5d300b80e8499a8572897e7add05136f6c50c0a977`。
同一工具链复测 artifact：
`/private/tmp/turbocider-streaming-campaign-clean-source-metal-20260916`。

| clean-source tiny 指标 | clean dev | current candidate | candidate/dev | block-bootstrap 95% interval |
|---|---:|---:|---:|---:|
| request wall median | 23.344833 s | 23.361947 s | 1.00073 | 0.96806–1.04052 |
| request wall P95 | 23.613862 s | 23.700510 s | 1.00367 | 0.98884–1.01599 |
| denoise median | 8.413172 s | 8.435187 s | 1.00262 | 0.99721–1.00432 |
| denoise P95（诊断项） | 8.462887 s | 8.450525 s | 0.99854 | 0.99472–1.00439 |

这次 clean-source 复测未复现旧 saved dylib 的 +3.16% denoise 信号，支持该信号主要来自不可比
binary/build 状态；同时也证明 source provenance 是 P0 的必要条件。denoise median/P95 诊断区间均在门槛内，
但 wall median 区间上界1.04052仍高于1.02，且只有4 pairs、tiny workload、audit/environment为partial，
因此 verifier 保持 `INCONCLUSIVE`，不能升级为 P0 PASS。

### 13.4 当前下一门

1. 冻结 normal-target resident 与原 legacy streamed 两张 P0 卡，至少20 matched pairs并按预注册 ceiling执行；
2. P1 先证明 legacy/exact 的 P/G/K/D/Q、loader并发、retention、conditioning/VAE/export 同义；
3. 完成 whole-request closure 后再运行 P2；P3 pressure/swap 仍需单独授权；
4. normal-target P0/P1 未通过前 production registry 继续为空，H3/Flux/Z 扩展不越过该门。

### 13.5 独立 audit build 与 20-pair tiny P0

新增 `TURBOCIDER_BUILD_AUDIT_COUNTERS=1` 和可选 `TURBOCIDER_BUILD_OUTPUT_DIR`。release 下
`audit_increment()` 为 inline no-op，dylib 不导出 audit ABI；audit build 才编译原子计数并私有导出：

- `tc_streaming_audit_reset`；
- `tc_streaming_audit_snapshot_json`。

计数覆盖 constrained admission hook、checkpoint capability probe、`IoExecutor` worker 创建、
`StageExecutor` pool 创建以及 constrained route 触发的 cache-clear/unload；普通 engine free 和用户显式
unload 不计入新框架 audit。`run_streaming_audit.py` 可单独生成 verifier 接受的 `audit.json`；campaign worker
若检测到 audit ABI，则在每个请求前 reset、请求后 snapshot。release worker 缺少该 ABI 时仍保持 partial，
不会伪造完整 audit。测试入口为：

```sh
make PYTHON=python3 test-streaming-audit
```

三种隔离构建已验证：release 无 audit/test-hook 符号；audit 只有 audit 符号；test-hook 只有四个
`tc_engine_test_ltx_*` 生命周期符号。真实默认 LTX tiny 请求在 audit build 中成功，五类计数均为0。
显式 fake `StageExecutor` audit 则精确观察到1次pool allocation和2个worker thread，证明计数不是恒零。

重新运行 test-hook 真实 GPU 生命周期矩阵，artifact 为：

```text
/private/tmp/turbocider-ltx-lifecycle-audit-integration-20260916/lifecycle/lifecycle-summary.json
```

metadata/first-fill/Stage1/upsample/Stage2/VAE/export、session/process quarantine和恢复全部通过；所有
成功恢复请求的 Stage-2 latent SHA-256仍为
`7db71bf03027942b53af69ab914714583fe8f8d4c3ed2faf60f4aa4ed43f28fd`。

随后对真实LTX 64×64×9、11 steps、默认resident运行10个ABBA/BAAB block、20 matched pairs。
clean baseline仍为`dev@ad343d4`，release candidate SHA-256为
`c2f9c98a55c4554080a08585a5cc9829aa9f71f5be4696e5e9aa8358038e03d8`。release性能bundle：

```text
/private/tmp/turbocider-streaming-campaign-audit-20260916/bundle-release10
```

独立audit bundle：

```text
/private/tmp/turbocider-streaming-campaign-audit-20260916/bundle-audit10
```

两者均为40/40 measured成功、20/20 pair质量byte-exact、完整环境/source provenance、无fault。release结果：

| tiny default P0指标 | clean dev | release candidate | ratio | block-bootstrap 95% interval |
|---|---:|---:|---:|---:|
| request wall median | 23.933185 s | 23.930811 s | 0.99990 | 0.99241–1.00337 |
| request wall P95 | 24.174613 s | 24.140116 s | 0.99857 | 0.98740–1.01005 |
| denoise median | 8.421231 s | 8.427595 s | 1.00076 | 0.99634–1.00511 |
| denoise P95（诊断） | 8.681186 s | 8.547064 s | 0.98455 | 0.95866–1.02423 |

verifier 对该冻结tiny tuple给出`PASS`，audit hard gate五类计数全部为0。该结论证明本次改造未使这个
默认tiny resident tuple劣化；它仍不是normal-target、largest、legacy-streamed P0，也不授予production
exact streaming或bounded-memory资格。

### 13.6 Ordered multi-class executor 与 v2 C bridge（2026-09-16）

本轮把 compiler 已表达、但旧 executor 拒绝的异构 `layout_class` 接到统一执行框架：

- `StageExecutor` 仍只允许一个活动 pool，但按连续 pool 区间执行；切换前要求当前 pool 的所有 fill、
  GPU reader fence 和 adapter `drain()` 完成，随后才 `destroy_pool()` 并创建下一 pool。
- 同 class 的 pool/backing/worker 在 pass 内按 lookahead/fence 重用；跨 class 不做 lookahead，跨 pass
  回到首个 class 时重新经历有界 barrier。每个 pool generation 的 `pool_creates`、slot bundle、ticket
  和 capacity 独立计数，不能把多 class 伪装成一次建池。
- 原 `tc_stream_executor_create_v1` ABI、回调签名和单 pool 行为保持不变；新增 versioned
  `tc_stream_executor_create_v2`、`tc_stream_pool_plan_v2`、`tc_stream_group_v2` 与
  `tc_stream_adapter_v2`，显式传入有序 pool 列表以及带 pool 的 group。v2 adapter 的 allocate/destroy
  可观察真实 pool id，仍复用同一个 C++ executor，不创建第二套 pager。
- host executor 测试新增两 pool/三 pass 的乱序双 reader、pool 创建失败、非法 class 顺序和非零 pool id；
  v2 C11 测试覆盖 pool-aware allocate/destroy、跨 pass barrier、counter 和 ABI。普通单 class 14 个
  K/D/Q 组合、C bridge fault、ASan/UBSan、TSan 继续通过。

独立 release build：

```text
/private/tmp/turbocider-multiclass-release-20260916/libturbocider.dylib
SHA-256: 65d1da711c278d5568c7b6de158984b0ebbdd0d4f687b02f90527edc3f0679d5
```

release 只导出 `tc_stream_executor_create_v1/v2`，没有 audit 或 lifecycle test-hook 符号。沙箱外提交前
tiny 默认 resident 2-block/4-pair campaign 8/8 成功、质量 byte-exact：candidate/dev wall median
ratio `1.00321`，denoise median ratio `1.00118`；由于只有4 pairs且 release 未启用 audit ABI，verifier
保持 `INCONCLUSIVE`，不能升级为 P0。该结果与此前独立 20-pair release PASS 一致，未发现默认路径的明显
回退；多 class executor 本身未被 default resident 路径实例化。

独立 audit build SHA-256 为
`ba3d0c29685a29c8680a8fc2b8d1a03602174287c1fd5a3647ac5ef85d88f705`；真实默认 resident 请求成功，
`/private/tmp/turbocider-multiclass-audit-run-20260916/audit.json` 中 framework hook、memory probe、worker、
pool allocation、cache-clear/unload 五类计数全部为0。显式 multi-class host test 则观察并断言每个 barrier 的
pool generation 计数，避免默认零计数由 instrumentation 未接线造成。

当前源码另行重建 test-hook dylib 后，真实 LTX lifecycle matrix 全部通过：

```text
/private/tmp/turbocider-multiclass-lifecycle-20260916/lifecycle/lifecycle-summary.json
```

覆盖 metadata/first-fill、Stage 1、upsample、Stage 2、VAE、export、success→success、A→B→A、
取消、unsafe destroy/session quarantine、process quarantine 与恢复；Stage-2 latent SHA-256 仍为
`7db71bf03027942b53af69ab914714583fe8f8d4c3ed2faf60f4aa4ed43f28fd`。该结果证明本次 executor/C ABI
扩展没有破坏现有 LTX 单 class exact candidate lifecycle，但不改变 public gate、normal-target 或 bounded
资格状态。

### 13.7 最终源码重建与合并交付复核（2026-09-16）

在将 pool 激活从按 pool-id 线性查找到按已验证的有序 pool index 直接访问后，重新完成三类构建与验证：

| 构建 | dylib SHA-256 | 预期私有符号 |
|---|---|---|
| release | `e19cbd797eb19f85e1392cf0b91822ff1fadbb41e32ad8b04adf8b575c7225cb` | 无 audit、无 lifecycle hook |
| audit | `0d5d9be922703b5225055fc2ec630d9550f5c9c94859d95cf1fb6db86ac44b7b` | 仅 audit reset/snapshot |
| test-hook | `b925be99d6b6e19bf6a23b71bb9e1b7af4690b8d6f34554579e2a98feeb9defc` | 仅四个 LTX lifecycle hook |

三类构建均成功；release 导出 `tc_stream_executor_create_v1/v2`，没有 audit 或 test-hook 符号；audit 与
test-hook 符号集合互斥。普通 host、ASan/UBSan、TSan 和 Python contract/audit/verifier/source-identity
回归均通过。

最终 audit build 的默认 LTX resident 请求成功，artifact：

```text
/private/tmp/turbocider-audit-default-final-20260916/audit.json
```

`new_framework_hooks`、`new_memory_probes`、`new_worker_threads`、`new_pool_allocations`、
`new_cache_clear_or_unload_calls` 全部为 `0`，证明新 executor/C ABI 不进入默认 resident 热路径。

最终 test-hook 生命周期 artifact：

```text
/private/tmp/turbocider-multiclass-lifecycle-final-20260916/lifecycle/lifecycle-summary.json
```

metadata/first-fill、Stage 1/2、upsample、VAE、export、success→success、A→B→A、取消、session/process
quarantine 与恢复全部通过；Stage-2 latent SHA-256 为
`7db71bf03027942b53af69ab914714583fe8f8d4c3ed2faf60f4aa4ed43f28fd`。

最终 release 对 clean `dev@ad343d4` 的 4-pair ABBA/BAAB smoke artifact：

```text
/private/tmp/turbocider-multiclass-campaign-final-20260916/bundle-post-index
```

8/8 请求成功、quality byte-exact、audit 五类计数全为 0；candidate/dev 比值为 wall median `0.99334`、
wall P95 `0.99232`、denoise median `0.99832`。由于仅 4 matched pairs，verifier 正确保持
`INCONCLUSIVE`，不能替代既有 20-pair P0 PASS；它仅证明最终重建没有出现明显 resident 回退。

## 13.8 H3 metadata-only descriptor 增量（2026-09-16）

本轮在不改变 H3 legacy/resident/SSD streaming 热循环的前提下，完成了第一版 H3
通用 descriptor 与 K=2/G=1 plan 投影：

- 新增 `native/models/h3_runtime/h3_streaming_descriptor.hpp/.cpp`，只打开并解析
  transformer 目录中的 safetensors header，不 mmap、不读取 payload、不创建 Metal buffer、
  不创建 executor/worker。
- descriptor 覆盖多 shard snapshot identity、所有 shard artifact、uniform active-block
  policy 产生的显式 active block ID 序列、四个 BF16 streamed matrix
  (`qkv/out/fc1/fc2`) 的 shape、source range、destination storage identity 和容量。
- `h3_weight_store_header()` 提供只读 header view，descriptor 会检查 shard 排序、重复 tensor、
  缺失 tensor、dtype/shape/byte-range 以及目录增删和 stat 变化；snapshot 失效时在进入 compiler
  前 fail closed。
- `StreamingPlanView` 只接受当前候选边界：单 `denoiser` stage、streamed、G=1、K=2、至少一个
  streamed suffix；token reduction、first-block cache 等尚未显式建模的动态 shortcut 会拒绝，
  不会静默改变访问序列。
- `h3_dit_schedule.h` 公开 H3 shape 常量，legacy `h3_dit.c` 复用同一常量；active mask 公共
  C ABI 增加 `extern "C"` 保护，避免 C++ descriptor 链接到 C 实现时发生符号改名。
- `tools/native/build.sh` 与 `make test-streaming-host` 已接入该 descriptor 测试。

新增测试 `tests/native/test_h3_streaming_descriptor.py` 使用四个 sparse safetensors shard，
覆盖 50-block metadata 的多 shard 投影而不分配实际权重 payload；测试包含 active ordinal
prefix、no-prefix、digest 稳定性、步骤/active-block变化、缺 tensor、错 shape、重复 tensor、
非法 K/G/prefix、动态 shortcut 拒绝及文件截断 TOCTOU。当前结果：普通、ASan/UBSan、TSan
均 PASS；release native build、streaming host/contract、native contract 和 C++ boundary
回归均 PASS（native contract 82 项中 81 PASS、1 个 Wan fixture SKIP）。

这批代码仍然是 metadata/plan-only，不授予 H3 public exact 或 production registry 资格。
H3 现有 `stream_ready_slot ^ 1u`、跨 forward prefetch、跨 block fusion、step gate、norm/AdaLN
和完整 text/latent/VAE 资源仍未映射为 generic executor 的显式 last-reader 合同；下一步必须在
不替换 legacy 热路径的前提下实现 candidate execution bridge 与真实生命周期/性能验收。

当前源码重建的 release dylib SHA-256 为
`c18197a6a25fda3e389b2b6b90e7646d2ea909c29485599785301c53be67b3a8`。
默认 LTX resident audit 的五类新框架计数继续全部为 0；相对 clean `dev@ad343d4` 的
4-pair ABBA/BAAB 回归 smoke 为 8/8 请求成功且 Stage-2 BF16 逐对 byte-exact，candidate/dev
比值为 wall median `0.99506`、wall P95 `1.00518`、denoise median `1.00063`、denoise P95
`1.00289`。这些点估计均在 P0 阈值内，且未观察到 H3 未使用 descriptor 对默认 LTX 热路径的
实质回退；但 wall median 的 block-bootstrap 95% 区间为 `[0.96021, 1.03423]`，因此 verifier
正确保持 `INCONCLUSIVE`。该短 smoke 不能替代既有 20-pair tiny P0 或后续 normal-target P0。

### 13.9 Cross-pass carry、ABI v3 与 dev 再同步（2026-09-17）

远端 `origin/dev`、本地 `dev` 与 fetch 得到的 `FETCH_HEAD` 均为
`ad343d4e5139c9a5f13e29ff9e926e8eb1ae2f39`；该提交已由 merge commit
`566f7a6faf07734255cee229ca31ca3af0bc154c` 合入当前分支。再次执行 `git merge dev` 返回
`Already up to date`；实现提交 `b975b29a8407275532b44078d90bdd4388974c86` 后，当前相对 dev
领先 9、落后 0，没有新增文本冲突或未解决 merge index。

本轮继续实现了此前 H3 legacy 双槽调度缺少的显式 pass transition：

- `PassTransition::{reload, carry_first_group}` 进入 descriptor/layout digest 和 `StageLayout`；
- 新增 `tc_stream_stage_plan_v3` / `tc_stream_executor_create_v3`，v1/v2 ABI 不变；
- executor 可在 pass N 尾部预填 pass N+1 首组，并在 boundary 保留唯一 Ready ticket；
- 奇偶 suffix 均通过 pass-relative slot rotation 保持最后组与下一首组不冲突；
- incoming carry 核验 pass/step/group/slot/generation，要求 scheduler step 连续；
- 当前只认证单 pool、K=2、G=1；未经验证的 K>2、多 pool、多 block carry fail closed；
- H3 `StreamingPlanView` 要求非零 request generation，并投影可直接交给 fake adapter 的 v3 plan。

当前证据链仍是：

```text
H3 safetensors metadata -> generic layout -> ABI v3 plan
-> generic executor -> fake source fill/reader fence
```

它还不是：

```text
real h3_dit/h3_gpu -> block fill -> Metal encode -> command-buffer fence
-> complete denoise request/session lifecycle
```

本轮 focused 回归：3535 个 layout case、14 个 K/D/Q executor 组合、multi-class barrier、v1/v2/v3 C bridge、
H3 sparse descriptor/fake execution、streaming contract、82 项 native contract（81 PASS、1 个既有 Wan fixture
SKIP）、repository boundaries、memory execution/compiler、H3 policy/schedule 全部通过；ASan/UBSan 与 TSan
也通过。

clean commit release：

```text
/private/tmp/turbocider-cross-pass-carry-release-20260916/libturbocider.dylib
SHA-256: 672efa3b41932ee3299eefdaa51d135f9fba2091dc5ada232545734d17f09c56
```

release 导出 v1/v2/v3 executor create，不导出 audit 或 lifecycle test-hook。对应 audit build SHA-256 为
`b5dd56e6bc0a221f8031bdc1dd2100aefc70dc9e177d93f660538817ac3fb05c`；真实默认 LTX resident
请求成功，framework hook、memory probe、worker、pool allocation、cache-clear/unload 五类计数全部为 0。

相对 clean `dev@ad343d4` 的 10-block/20-pair tiny default-resident P0 bundle：

```text
/private/tmp/turbocider-cross-pass-carry-campaign-20260917/bundle-20pairs
```

40/40 measured 请求成功，20/20 pair 的 Stage-2 BF16 SHA-256 逐对一致，fault=0，environment/source/audit
完整，verifier 结果为 `PASS`：

| 指标 | clean dev | `b975b29` candidate | candidate/dev | block-bootstrap 95% interval |
|---|---:|---:|---:|---:|
| wall median | 23.725008 s | 23.916378 s | 1.00807 | 1.00296–1.01152 |
| wall P95 | 24.210594 s | 24.253766 s | 1.00178 | 0.98701–1.04857 |
| denoise median | 8.355715 s | 8.372832 s | 1.00205 | 0.99567–1.00967 |
| denoise P95（诊断） | 8.496699 s | 8.550533 s | 1.00634 | 0.99376–1.05122 |

这证明本轮 carry/ABI v3 增量没有使冻结的 LTX 64×64×9、11-step 默认 resident tiny tuple 越过既有
P0 门槛；它不等于 normal-target、legacy-streamed、真实 H3 carry、bounded-memory 或 swap P3 资格。

## 13.10 LTX P1 语义与生命周期验收工具链（2026-09-17）

本轮先处理 LTX 正式 P1 的可信度，而不是直接把更多模型登记进 production。此前 runner 的两个问题已修正：

- verifier 要求 legacy/exact 的 generic layout digest 相同，但 legacy 没有 generic digest；
- 每个 worker 长期持有 engine，使 legacy denoiser 可以跨请求保留，而 exact owner 每次请求重建，retention
  不同却可能被误报成 framework overhead。

当前实现增加：

1. campaign 顶层 engine_lifecycle，支持 persistent 和 per_request；未显式配置仍保持原 persistent 行为。
2. per_request 保留长期 worker 进程以维持 ABBA 调度，但每个 warmup/measured request 独立 create/free
   engine；request wall 覆盖 engine create、generate 和安全 destroy，native request wall 另行保留。
3. LTX legacy 与 exact 都报告 actual semantic layout：P/G/K/D/Q、group/pass、startup、pass transition、
   retention、reader/weight/kernel revision、conditioning recipe、upsample boundary。
4. LTX 结果新增 request_slot_fills。legacy 为 allocation + refill；exact 直接使用 executor fills，避免把
   exact 首次 fill 再重复加上 slot bundle 数。
5. runner 归一 schema v1/v2 的 workload identity，比较 model/operation/prompt/shape/seed/steps/backend、
   approximation knobs，而不把 output path 或 executor-specific residency 混入。
6. semantic-equivalence evidence 升级为 v2；verifier 逐 pair 比较完整 normalized semantic layout，
   重算 digest，并拒绝缺字段、生命周期不符或仅 generic digest 相同的伪等价；policy 必须冻结
   expected_actual，防止 baseline/candidate 一起偏离 P8/G1/K3/D2/Q3 仍被误判通过。
7. P1 median 门槛统一收紧到 1.02，P95 保持 1.05。

验证结果：

- release native build PASS；
- streaming host PASS：3535 layouts、14 K/D/Q、multi-class、C bridge、LTX/H3 descriptor；
- streaming contract PASS；
- campaign verifier 18 项 PASS，包含 native per-request create/free、schema v1/v2 identity、
  legacy/exact fill 口径、双方相同但偏离 frozen expected、semantic mismatch、
  missing startup 和非法 lifecycle 反例；
- git diff whitespace check PASS。
- 沙箱内 test-streaming-metal 未获得 Metal device：通用 Metal 用例报告 SKIP，LTX 显式 Metal 子测试以
  no Metal device 退出；这不是 GPU PASS，也不是实现失败，真实 GPU 证据仍需沙箱外执行。

真实 Metal one-block P1 smoke policy 已冻结为 per-request、P8/G1/K3/D2/Q3、median 1.02；本次执行所需的
沙箱外 GPU 授权被自动审批链路中断，因此没有生成可签名的 raw bundle，不能声明 P1 PASS/FAIL。下一次取得
Metal 运行权限后，应先完成 one-block 排错，再冻结 2-block smoke，最后执行至少 10-block/20-pair 正式 P1。
production registry 继续为空。

当前工作树还保留一批 H3 真实 K2/G1 candidate adapter 增量；其存在不改变顺序：先完成 LTX P1，再迁移
Z-Image generic adapter，随后对 H3 做正式同义 P1，最后进入 Flux fusion/address-stability 审计。

## 13.11 LTX 冻结同布局 P1 通过与稳态分配修正（2026-09-17）

本轮取得实体 Metal 权限后，按第13.10节冻结协议完成 one-block、2-block 和 10-block/20-pair 三阶段验收。
首次 one-block 在 warmup 前被合同拒绝：policy 中 legacy `memory_budget_bytes=8570908800` 略低于运行时要求的
精确 8 GiB（`8589934592`）。这不是性能或语义失败；正式示例和 smoke policy 已修正为精确 8 GiB，未放宽
P8/G1/K3/D2/Q3 或性能阈值。

one-block 修正后 4/4 measured 请求成功，semantic-equivalence v2 确认双方实际布局、request retention、
conditioning/VAE 边界与每请求 440 fills 完全一致，Stage-2 BF16 pair byte-exact。审阅同时发现原 C bridge
`make_fill_job()` 会在某个 slot 首次 refill 时复制 `group.blocks`，可能在稳态触发 `std::vector` 分配；现改为
借用 `StageExecutor::State::layout` 中生命周期覆盖 worker join 的稳定 block span。

audit ABI 新增 `steady_framework_allocations` 和 `steady_framework_thread_creates`：一次性 executor setup 的
worker/pool 仍由原 counter 统计，setup 完成后的 pool 切换分配单独计入 steady counter。LTX 单 pool audit-only
实模型请求观察到 3 个 setup worker、1 个 setup pool、steady allocation/thread-create 均为0；ordered
multi-class host test 则会对运行期 pool 切换报告非零 steady allocation，防止把当前 multi-pool 实现误签成
零分配热路径。

2-block release smoke 位于：

```text
/private/tmp/turbocider-ltx-p1-two-block-20260917-v1
```

8/8 measured 请求成功、4/4 pair byte-exact，verifier `PASS`；wall median ratio为`0.97735`，其
block-bootstrap 95%上界为`1.00747`。

冻结 10-block/20-pair bundle 位于：

```text
/private/tmp/turbocider-ltx-p1-10block-20260917-v1
```

release dylib SHA-256为
`a5294019b6bf460733e25121425849ee77098630a27fadcc8556837dd46ba8f7`；独立 audit dylib SHA-256为
`f1759d2d20d20e8c6d69406b228d4fba909dd57cafa18d62c0d104f7edecab31`。40/40 measured 请求成功，
20/20 pair Stage-2 BF16逐对一致，fault=0，environment/audit/manifest/semantic证据完整，独立 verifier
复核为`PASS`：

| 指标 | legacy streamed | generic exact | exact/legacy | block-bootstrap 95% interval |
|---|---:|---:|---:|---:|
| wall median | 15.318877 s | 15.136558 s | 0.98810 | 0.97655–0.99570 |
| wall P95 | 15.469798 s | 15.307886 s | 0.98953 | 0.98482–0.99268 |
| denoise median | 7.652459 s | 7.441930 s | 0.97249 | 0.97113–0.97367 |
| denoise P95（诊断） | 7.681396 s | 7.480353 s | 0.97383 | 0.97224–0.97649 |

因此冻结的 LTX 64×64×9、11-step、per-request、P8/G1/K3/D2/Q3 初始 tuple 已通过 P1 ≤2%门槛；
该结论不自动覆盖 normal-target、largest/repeated-request、whole-request P2、低内存/swap P3 或 public
production资格。production registry仍为空。下一实施项是 Z-Image metadata-only descriptor shadow；原
`ZImageWeightStream` 默认/专用路径必须保持不创建 descriptor、executor、worker或额外同步。

### 13.12 Z-Image generic adapter 与冻结 P1 通过（2026-09-17）

本轮完成 private Z-Image generic execution：descriptor/compiler 生成 P14/G1/K2/D0/Q1，统一
`StageExecutor` 管理 ticket、slot、fill、last-reader 和 cleanup；adapter 复用原 safetensors ranged
`pread`、两个 MLX shared slot、fixed/prefix weights、`Weights::bind_arrays`、原 `z_block` 和同步
`mx::eval`。public constructor 仍由 `streaming_layout_not_certified` 拦截，专用
`ZImageWeightStream::begin_pass/acquire/std::async` 路径保留。

为恢复专用 K2/D0 overlap，adapter 使用默认关闭的 `overlap_next_fill_after_claim()`：首个 suffix 仍只
dispatch 一次；当前 group claim 后、同步 encode 前启动下一空闲 slot fill。随后发现同步 `mx::eval` 完成后
再把 reader completion 写入 mailbox 是冗余调度，框架增加 `ReaderSet::already_complete`；executor 仍执行
seal 和 identity 校验，只省去同步 backend 的 post/pop 往返，异步 adapter 行为不变。

性能诊断按原始失败保留：

| bundle/阶段 | wall median ratio | denoise median ratio | 结论 |
|---|---:|---:|---|
| v4 claim-overlap one-block | 1.02124 | 1.00889 | wall略超2%，smoke |
| v5去除重复cache clear one-block | 1.01496 | 1.00201 | 点估计进入门槛 |
| v1 10-block/20-pair | 1.02100 | 1.04039 | 正式拒绝 |
| v2 user-initiated QoS | 1.01926 | 1.02026 | denoise/区间仍未通过 |
| v3同步reader快路径 | **0.99986** | **1.00313** | **PASS** |

最终 release bundle：

```text
/private/tmp/turbocider-z-image-p1-10block-20260917-v3-immediate-reader
```

冻结 policy 为
[`examples/z-image-p1-same-layout-policy.json`](examples/z-image-p1-same-layout-policy.json)，SHA-256
`19c6ced46a0aac73082df74630f4fc173d6542a37c816709f0be2e6a64657949`。release dylib SHA-256 为
`5805045c1904fd94d889911015d4186520c289524f88f6489c94d7416e4a6ba4`；独立 audit dylib 为
`7ef26e5e509120fb32273603be6ef9238d41b063c955a34b915b3bd21be941f9`。

40/40 measured 请求成功，20/20 pair PNG SHA-256 一致，fault=0，semantic/environment/source/audit
完整，独立 verifier 再次返回 `PASS`。wall median/P95 ratio 为 `0.99986/0.99913`，95%上界为
`1.00249/1.00431`；denoise median ratio 为 `1.00313`，95%区间 `1.00099–1.00613`。denoise P95
`1.01266`是诊断项。每请求16 fills，setup worker/pool为1/1，steady framework allocation/thread-create
为0/0。

新增 telemetry 报告 suffix refill total、最慢 refill block/time 和 executor wait；它只用于解释尾延迟，
不影响 verifier 样本选择。该资格只覆盖64×64、1 step、Comfy BF16、per-request、P14/G1/K2/D0/Q1；
normal-target、量化/GGUF/Diffusers、P2/P3和public production仍未完成。下一项转入H3真实K2/G1 adapter。

### 13.13 H3 Turbo generic K2/G1 adapter 与工程验收（2026-09-17）

本轮将 H3 范围明确冻结为 **MiniMax H3 Turbo、原始 BF16、50 个 streamed blocks、4-pass DiT**；
不把普通 H3、量化缓存、动态 block skip、其他 checkpoint 变体或任意 K/G 布局纳入本次资格。
真实 candidate 已复用统一 `StageExecutor`：四个大矩阵 qkv/out/fc1/fc2 ranged fill 到两个既有 Metal
shared slot，norm、AdaLN、conditioning 和 workspace 继续 resident；generic exact 启用后 legacy
`stream_ready_layer/slot` 被失效，slot authority 不重叠。异步 reader completion 来自真实 Metal
command-buffer callback，跨 block fusion 与跨 pass `carry_first_group` 均进入同一 reader/ticket 合同。

功能证据：真实 13-shard 模型的单 block fill 为 `770703360` bytes，source/content 完全一致；取消回归得到
`cancel_queries=2`、`poisoned=true`、`destroyed=true`；legacy/exact 联合 latent FNV-1a 均为
`bcfc8bb6686e0388`。大块读取后的完整 exact 请求完成 4 passes、200 fills、200 submitted groups、一个 pool、
两个 slot bundle，无 poison，输出有限且一致。

首个正式 v2 bundle 保留于：

```text
/private/tmp/turbocider-h3-p1-10block-20260917-v2
```

它 40/40 成功、20/20 artifact 相等，但因 generic cancellable reader 把每个矩阵强制切成 8 MiB `pread`，
wall median/P95 ratio 为 `1.14662/1.14263`，denoise median 为 `1.22061`，正式拒绝。GPU median
`1.5725 s` 与 legacy 基本相同，退化定位为 I/O syscall/read path，而不是 Metal kernel。

修正后 exact reader 与 legacy 一致，每个矩阵执行一次大 `pread`，仍在四个矩阵边界检查取消。冻结
10-block/20-pair release bundle 为：

```text
/private/tmp/turbocider-h3-p1-10block-20260917-v3-large-read
```

policy 为 [`examples/h3-p1-same-layout-policy.json`](examples/h3-p1-same-layout-policy.json)，SHA-256
`923914aaa7807346746468d810d9d10419215dda8ec9c5a784bbcb8d306229cb`。40/40 measured 请求成功，
20/20 pair artifact SHA-256 相同，fault=0，source/protocol/semantic layout 完整：

| 指标 | legacy streamed | generic exact | exact/legacy | block-bootstrap 95% interval |
|---|---:|---:|---:|---:|
| wall median | 22.023271 s | 22.258101 s | 1.01066 | 0.97035–1.03850 |
| wall P95 | 24.718173 s | 24.597181 s | 0.99511 | 0.95166–1.07179 |
| denoise median | 15.284899 s | 15.466089 s | 1.01185 | 0.98438–1.02665 |
| denoise P95（诊断） | 16.225285 s | 16.106575 s | 0.99268 | 0.90709–1.03896 |

点估计全部位于既定 median 2%/P95 5%工程范围内，P95 未观察到退化。10 个 ABBA block 的物理 SSD
refill 方差较大，strict verifier 因置信上界跨线及初次 bundle 未内嵌独立 audit/environment 而保持
`INCONCLUSIVE`，不能改写为统计 PASS。用户已明确将 H3 范围收敛为 H3 Turbo，并接受点估计处于工程范围时
存在合理 I/O 波动，因此本轮状态记为 **冻结 H3 Turbo tuple 的工程验收完成**。bootstrap 区间继续如实保留为
诊断证据，但不再作为本轮继续施工或重复跑盘的阻断项；也不通过增加重复 TB 级 SSD 读取来筛选有利结果。
该决定只关闭 H3 Turbo 当前冻结 tuple 的工程任务，不把 `INCONCLUSIVE` 改写为统计 PASS，也不授予更广发布资格。

独立 audit build SHA-256 为
`0f08b55e57b087ac87cb1ebdf2f4c7ace2f3a8a90125361363ef82e153d8a51a`，真实 exact 请求观察到 setup
worker/pool `1/1`、steady framework allocation/thread-create `0/0`、cache-clear/unload `0`。脱敏的 audit
和环境记录位于 `results/streaming/h3/2026-09-17/`。v2→v3 的 legacy baseline wall median
`22.056429→22.023271 s`、denoise `15.231326→15.284899 s`、GPU `1.572522→1.572361 s`，没有显示
exact-only 大块读取修正影响默认 legacy 路线。

收口回归通过：3535 layouts、14 K/D/Q、multi-class、C bridge v3 carry、LTX/H3/Z-Image descriptor、
三条 public fail-closed gate及21项 campaign verifier测试。H3 public gate和production registry继续关闭；
本节不授予 full video session、normal-target、bounded-memory P2、swap P3或其他 H3 变体资格。除非后续出现
明确 correctness/lifecycle 回归，H3 不再扩展普通版本、量化版本或新布局；下一实施项进入 Flux fusion、
compiled graph address stability 与 multi-class arena 审计。

### 13.14 Flux.2 Klein 9B retained multi-class execution（2026-09-17）

本轮只接 **Flux.2 Klein 9B、Diffusers两分片BF16、eager GPU、无LoRA**。Flux 4B现有compiled graph、
GPU+ANE、prepare-only和其他checkpoint继续fail-closed，避免为了统一接口给默认compiled block loop增加
逐block同步或改变地址生命周期。

实现包括：

- descriptor从真实config/index及两个safetensors header投影9个resident fixed tensor、8×16-field dual block
  和24×4-field single block，source range、shape、dtype、snapshot及resident bytes均进入layout identity；
- `MultiPoolPolicy::retain_all`在setup创建dual/single两类K2 pool，跨class和跨pass只做drain与pool selection；
- 新增通用`MlxWeightPager`：owner创建所有MLX shared array，worker只做`pread`，支持多artifact、resident load、
  pool fill/bind、取消、短读、路径替换和snapshot检查；当前严格限制direct BF16、G1、每field单source range；
- dual和single block复用原eager数学，single首块前执行context/image concatenate；每block `mx::eval`是同步
  last-reader completion，`ReaderSet::already_complete`避免多余mailbox往返；
- public constructor仍返回`streaming_layout_not_certified`，private candidate只允许
  `P0/G1/K2/D0..1/Q1..2/reload`；当前推荐tuple为`P0/G1/K2/D1/Q2/reload`。

新增sparse pager fixture以生产同款256-byte slot alignment覆盖resident load、dual/single retained pool、
fill/bind、预取消、short read、stale path、wrong artifact/range/shape、pool destroy/recreate；沙箱外Metal执行PASS。
同一fixture的ASan/UBSan与TSan构建也在实体Metal上PASS。
完整native build以及`test-streaming-host test-streaming-contract test-streaming-audit`通过：3535 layouts、
14组K/D/Q、retained multi-class、四模型descriptor和四条candidate gate均通过。

两步真实结果：

| 路径 | wall | denoise | MLX peak | fills / logical request bytes |
|---|---:|---:|---:|---:|
| resident | 1.338198 s | 0.859606 s | 18,303,578,036 | 不适用 |
| generic Q2 | 1.951123 s | 1.383473 s | 11,693,804,356 | 64 / 35,605,487,616 |

两条路径PNG SHA-256均为
`5b39235aab9acbbccbce6b1fffdc672cbf5f138c2f935db1fdaf3fb2efa1d108`。Q2 refill worker累计load
`2.395479 s`，executor wait `1.082449 s`；load时间是两个worker时间之和，不可直接与wall相加。peak降低
`6,609,773,680` bytes（约6.15 GiB）。这证明低内存策略可执行并降低MLX allocator峰值；streaming比resident
慢是SSD搬运策略代价，不能写成framework overhead。

audit build真实Q2请求结果：setup worker/pool `2/2`，steady framework allocation/thread-create `0/0`；
请求setup和denoise→VAE交接各有一次weights clear与MLX cache clear，因此
`new_cache_clear_or_unload_calls=4`，block/class/pass稳态内为0。默认resident请求全部streaming audit counter
为0。两步请求证明
dual→single→dual跨pass不会重建pool。
最终release dylib SHA-256为
`d9acb31adaef4d389fa9211ea364078ebcd35119758c7527606f551cb8f4abb9`；对应audit dylib为
`901eb689782a368931bf618efa682249e69ad8c25bf66f71a4f9d0e77524a250`。

默认resident保护采用保存的旧dylib与当前dylib，顺序A1–B1–B2–A2；每个persistent engine运行5次并丢弃
首次加载，得到每variant 8个warm样本。当前/旧版warm median ratio为wall `0.99748`、request wall
`0.99748`、denoise `0.99607`，所有1-step PNG hash均为
`0be91f956cd83457052b70d797bfc4ca81f88e70a77829e94aa1f8d3772393e3`。该小型ABBA工程对照未观察到默认
回退，但只有两个外层block，不能替代预注册20-pair正式P0。
最终audit可观测性修正后的release又执行一组A→B、每variant 5次并丢弃首次加载；wall/denoise warm median
ratio为`0.99665/0.99963`，输出hash仍一致，再次未观察到默认路径回退。

Flux private same-layout direct replay已经完成正式20-pair P1并通过≤2% framework overhead门槛。该结论只
比较相同streaming布局，不能用来声称streaming与resident等速。public registry、whole-request hard cap、
P2、真实低内存/swap P3及Flux 4B仍未完成。

### 13.15 Flux 9B 同布局 direct replay 与 implementation 身份门禁（2026-09-17）

为回答第13.14节尚未回答的纯框架开销问题，private exact candidate新增
`TURBOCIDER_FLUX_DIRECT_STREAMING_BASELINE=1`。该开关仅在已受限的Klein 9B candidate内生效；public和
resident路线不读取它。direct replay复用generic完全相同的descriptor/layout、`MlxWeightPager`、
`SlotSafetyTracker`、两个retained K2 pool、两个持久I/O worker、D1 dispatch window、每block同步
`mx::eval`、reader完成与request retention，只绕过`StageExecutor`通用owner调度循环。

首次两步same-layout smoke结果：

| 实现 | request wall | denoise | MLX peak | implementation |
|---|---:|---:|---:|---|
| direct replay | 1.909523 s | 1.371629 s | 11,693,804,356 | `flux_direct_same_layout_v1` |
| generic executor | 1.896156 s | 1.361718 s | 11,693,804,356 | `generic_stage_executor_v1` |

generic/direct ratio为wall `0.99300`、denoise `0.99277`。两侧layout digest均为
`4d51e19e6560a6c90d5488729ceba58810f46d2390c5c83ecec03b4f68a5ffae`，均为64 fills、2 pool creates、
4 slot bundles、`35,605,487,616` logical request bytes，PNG SHA-256均为
`5b39235aab9acbbccbce6b1fffdc672cbf5f138c2f935db1fdaf3fb2efa1d108`。该单次结果说明路径、质量、内存和
身份接线正确，且未观察到框架回退；它不是统计P1 PASS。

campaign raw sample现新增`streaming_implementation`，可选policy字段`expected_implementations`会由runner
校验schema、由verifier逐成功请求硬校验。新增冻结policy
`examples/flux9-p1-same-layout-policy.json`固定baseline=`flux_direct_same_layout_v1`、
candidate=`generic_stage_executor_v1`，避免环境变量遗漏时两侧误走同一实现仍得到伪P1。

独立audit dylib SHA-256为
`d597f8dba18cfac33129bc55db2231230f62e9292833c0d4a3ebeeef3fb62327`。direct与generic各一次真实请求均观察到
setup worker/pool `2/2`、request-boundary clear/unload `4`、steady framework allocation/thread-create
`0/0`。release dylib SHA-256为
`2d99b9889cdfd9abf85c2a0a98b4addb8c130bc3aa6a82341b196e387f671878`。

先后执行的1-block和2-block smoke均通过；随后使用冻结policy完成正式10-block/20-pair campaign。证据位于
`results/streaming/flux/2026-09-17/p1-same-layout/`：40/40 measured请求成功、20/20 pair PNG byte-exact、
fault=0、semantic/layout/implementation/environment/audit/manifest均完整，独立verifier重跑为`PASS`。

| 指标 | direct baseline | generic candidate | generic/direct | block-bootstrap 95% interval |
|---|---:|---:|---:|---:|
| wall median | 1.946221 s | 1.945657 s | 0.99971 | 0.99620–1.00414 |
| wall P95 | 1.964082 s | 1.970458 s | 1.00325 | 0.97271–1.00623 |
| denoise median | 1.336996 s | 1.334537 s | 0.99816 | 0.99526–1.00262 |
| denoise P95（诊断） | 1.354063 s | 1.353592 s | 0.99965 | 0.98273–1.00629 |

因此Flux.2 Klein 9B BF16 eager GPU的冻结`P0/G1/K2/D1/Q2/reload` tuple正式关闭P1：通用
`StageExecutor`未观察到超过2%的框架开销。该资格不扩展到Flux 4B compiled graph、LoRA、GPU+ANE、
prepare-only、其他shape/checkpoint或production public route。

### 13.16 Public memory-tier 控制面基础（2026-09-17，已提交 `63b73d9`）

在文档23–26的基础上，`feat/stream` 已提交 public selector 的第一批控制面代码，但没有放开任何
模型 public execution：

- `StreamingSelector` schema v2，支持 disabled、`memory_tier`、exact `preset` 和8/10/12/16/20 GiB target；
- request/profile 对 selector 与 manual streaming 进行原子替换，保留 provenance 并拒绝legacy residency/budget/offload冲突；
- plan-only 对 active selector 返回 `streaming_preset_resolution_required`，不把target写成旧memory budget；
- 新增 host-only `preset_catalog.*` skeleton，使用整数 `max(512MiB,ceil(10%×T))` margin，并按
  performance rank、calibrated bytes、logical reads、ID/revision确定排序；
- production catalog 故意为空，revision为`tc-streaming-catalog-empty-v1`；
- 新增 metadata-only `tc_streaming_options_json`，当前只返回tentative五档和明确unavailable；
- Swift新增v2 selector/request/options类型与plan/options/generate overload，但App尚未接入。

本轮实际验证：

```text
tools/native/build.sh          PASS：native及Swift/App全部编译
make test-streaming-contract   PASS：12项request contract；四模型public gate仍fail-closed
make test-streaming-host       PASS：3535 layouts、14 K/D/Q、multi-class/fault/cleanup、
                                preset resolver、LTX/H3/Z/Flux descriptors
```

这些结果只证明控制面基础可编译及host回归，不是新的GPU/内存/性能成绩。production catalog仍为空，
`tc_engine_resolve_streaming_json`、authority、immutable resolved request、`generate_resolved`、App开关/job迁移、
calibration工具和reviewed records均未实现。

代码审阅发现的 unresolved-selector 门禁已经修复并随`63b73d9`提交：`tc_engine_generate` 和 prepare 在全局GPU锁、DeviceLease、
session调用之前拒绝 active selector；普通 public engine 与 private candidate 的 generate/prepare 均已回归
`streaming_preset_resolution_required`。这只消除了静默 resident 回退，engine exact resolver尚未实现，
因此当前状态仍为**不可public执行**。完整施工顺序与验收追踪见
[27](27-public-streaming-delivery-blueprint.md)。

后续可编码类型、engine/API/App接线见[28](28-public-runtime-code-design.md)；逐PR、工具链、四模型候选矩阵、
内存/swap实验和发布门见[29](29-public-implementation-and-acceptance-plan.md)。这两份文档不改变当前“catalog为空、
public执行不可用”的完成边界。

### 13.17 Exact public resolution authority 工作树接线（2026-09-17，尚未提交）

在 13.16 的控制面基础上，当前工作树已继续完成 exact engine resolution 的框架接线，但仍未给任何模型 public 资格：

- 新增 `canonical_encoding.hpp/.cpp`，使用 typed、length-prefixed canonical encoding；
- 扩展 production record 的 source/workload/runtime/device/plan/calibration/performance/release identity；
- 新增 `preset_resolver.hpp/.cpp`、`resolved_request.hpp/.cpp`，实现 deterministic select、exact replay、
  internal-only non-copyable authority 和 immutable resolved execution；
- `ModelSession` 新增 public probe/compile/generate_resolved 默认拒绝 hooks；
- `tc_engine` 保存 model ID、normalized model root、execution container 和 streaming quarantine；
- 新增 C ABI `tc_engine_resolve_streaming_json`；
- active selector generate 在 global GPU lock 前 resolve，取得锁后 revalidate，并且只调用 `generate_resolved`；
- active selector prepare 返回 `streaming_prepare_unsupported`；
- Swift 新增 resolution/selection/identity/error envelope 类型与 `NativeEngine.resolveStreaming`；
- 四模型普通与 candidate gate 均证明空 catalog 时 exact resolve/generate 不进入 ordinary session/GPU execution；
- release dylib 不导出 LTX lifecycle test hooks。

当前 production catalog 仍为 `tc-streaming-catalog-empty-v1`，因此 public selector 仍不可执行。四模型已有 private exact adapter
不会因为新 C ABI 自动获得 public authority。

当前收口审阅发现一个明确待修项：`resolve_public_streaming_locked` 在调用完整 `make_plan` 之前先检查空 catalog，
因此 selector 与 legacy budget、GPU+ANE 或 compiled route 的冲突可能被
`catalog_has_no_public_records` 遮蔽。不能简单把完整 planner 前移，因为这会把模型 recipe/shape validation 和 synthetic fixture
耦合到 catalog availability。下一步应抽取纯 host、无 catalog/GPU/model probe 的共享 public request validator，由 planner 与 C API
共同调用，并冻结“selector/config/route → empty catalog → full model validation → probe/select/authorize”的错误顺序。

本阶段已通过：

~~~text
python3 -B tests/native/test_streaming_preset_resolver.py       PASS
python3 -B tests/native/test_streaming_contract.py              PASS，12项
四模型 candidate public gate                                  PASS
TURBOCIDER_NATIVE_ONLY=1 tools/native/build.sh                  PASS
make test-streaming-contract                                   PASS
tools/native/build_app.sh                                      PASS
git diff --check                                                PASS
~~~

这些结果证明 exact control/runtime 接缝可编译并保持空 catalog fail-closed，不证明 App 已开放、模型 public override 已完成、
任一 8/10/12/16/20 GiB 档位满足完整请求内存，也不增加新的 GPU 性能成绩。

进一步的代码设计、App事务、source execution revalidation、RunResult actual-plan 核对、四模型接入、swap实验和 release checklist
已整理到[32](32-public-streaming-completion-spec.md)。下一阶段应先完成共享 validator、C ABI 所有权/错误合同和 public result，
再做 App 与模型 adapter；production record 必须继续最后加入。

### 13.18 Shared validator 修正与工程规格补充（2026-09-17）

本轮完成了一个编译阻断修正和两份面向实施的设计规格：

- `native/runtime/streaming/public_request_validation.cpp` 不再引用不存在的 `config.hpp`，改为依赖实际存在的
  `core/streaming_contracts.hpp`；validator 仍保持 request-only、无 catalog/GPU/model module/filesystem 副作用的边界。
- 新增[33 Runtime/App 工程实施规格](33-public-runtime-app-engineering-spec.md)：冻结共享 validator、exact resolve
  helper、source lease、actual-plan hard verification、C ABI 所有权、Swift semantic parity、App 高级设置、
  JobStore schema v2、LTX worker-local authority、默认路径零开销审计、逐批 PR 和测试 ID。
- 新增[34 四模型档位校准与发布规格](34-model-tier-calibration-and-release-spec.md)：冻结用户只选择 target、后台选择
  reviewed layout 的产品边界，细化 LTX/H3 Turbo/Z-Image/Flux 9B 候选族、8/10/12/16/20 GiB 完整请求校准、
  resident/streaming/swap 四路实验、simulator/toolchain、evidence、catalog review、撤回和 release gate。
- README 已加入 33/34 文档地图；32 增加到新规格的交叉链接。文档不改变 production catalog 为空和 public 当前不可执行的事实。

本轮重新验证：

~~~text
env TURBOCIDER_NATIVE_ONLY=1 tools/native/build.sh   PASS
make test-streaming-host                            PASS：3535 layouts、14 K/D/Q、multi-class/fault/cleanup、四模型 descriptor/resolver
make test-streaming-contract                         PASS：ABI/plan/selector、12项 Python contract、四模型 public fail-closed gate
tools/native/build_app.sh                            PASS：Swift App 和 integration tests 构建完成
~~~

这些是编译、host/contract 和 App build 证据，不是 public model/target 资格证据。仍未完成：PUB-VAL-001…010 的专门错误顺序断言、
完整 C ABI null/busy/ownership/cancel 测试、source lease execution revalidation、RunResult actual-plan verifier、App transaction、
四模型 public hooks、完整请求档位 calibration、swap P3、ANE public 兼容和 reviewed production records。下一步按 33 的 R0→R6、
再按 34 的单 record 校准/发布顺序推进；在这些完成前不要把任何档位标为 available，也不要声称比系统 swap 更快。

### 13.19 Public actual-plan 回执工作树与实施蓝图 v2（2026-09-17）

`46a3e97` 已提交 exact public resolution、共享 validator、C ABI/Swift resolution、错误优先级测试和 32–34 号规格。
该提交之后，当前工作树继续实现 public result/actual-plan verifier，但尚未形成新 commit：

- `RunResult` 增加 `PublicStreamingSelectionMetrics`；
- `StreamingRuntimeMetrics` 增加 component policy、multi-pool、pool/slot bundle/worker、source lease 和 drain 字段；
- 新增 `native/runtime/streaming/public_result.hpp/.cpp`；
- `tc_engine_generate` 在 `generate_resolved` 返回后、结果序列化前调用 common hard verifier；
- verifier 核对 layout digest、P/G/K/D/Q、group/pass、pass transition、multi-pool、component policy、pool/slot/worker、
  request retention、source lease 和 completed drain；不一致返回 `streaming_actual_plan_mismatch`；
- `results.mm` 开始序列化 public preset/catalog/record/resolution/source/workload/runtime/device/layout/memory scope；
- resolver host test 增加 success、slot mismatch、source lease false、drain false 的覆盖。

本段记录的是当前工作树事实，不等于模型已经能产生这些字段。四个普通模型 session 仍使用基类默认拒绝 public hook，
production catalog 仍为空；因此现阶段只能在 fake resolver/snapshot 测试中验证 common verifier。

actual-plan 增量之后重新执行了完整回归：

```text
env TURBOCIDER_NATIVE_ONLY=1 tools/native/build.sh         PASS
make test-streaming-host                                   PASS
make test-streaming-contract                               PASS
tools/native/build_app.sh                                  PASS
git diff --check                                            PASS
```

contract 构建仍报告已有的 macOS 26.0/26.2 dylib deployment-target warning，但 12 项 Python contract、C ABI/source identity、
四模型 public fail-closed gate 均通过。以上仍只是当前工作树的构建、host/contract 和 fake resolver/result 证据，不是
public model adapter、完整请求 target calibration、swap P3 或 reviewed record 资格证据。

新增[35 Public Streaming 框架实施蓝图 v2](35-public-streaming-implementation-blueprint-v2.md)，进一步明确：

- 从 `c_api.mm` 下沉 pure C++ `PublicStreamingCoordinator` 的建议边界；
- production/test catalog provider 隔离，test injection 不进入 release header；
- source lease 三次校验和结构化 actual receipt 的渐进兼容方案；
- StageExecutor/multi-slot/multi-pool 不变量和 K2 overlap；
- Z-Image → Flux 9B → H3 Turbo → LTX worker 的逐文件 public adapter 施工单；
- App `StreamingChoice`、options stale guard、JobStore v2、resolve→persist→generate 原子事务；
- process-tree sampler、simulator、campaign、independent verifier、catalog builder 的职责；
- resident/streaming/bounded/swap 四路实验、P0–P4 和完整请求内存门；
- PUB-HOST/ABI/RT/APP/MODEL/CAL/REL 测试矩阵、R0–R8 提交边界和 failure injection/revoke/quarantine。

下一步的严格顺序是：先补全并提交 R0 actual result；再实现 coordinator/test catalog/source lease；然后按单模型接
public hook；之后才运行完整 target calibration 和 swap P3；最后加入 reviewed record、开放 App target 并合并最新 dev 回归。

### 13.20 Public coordinator 与 actual-plan 提交后的最新基线（2026-09-17）

本节覆盖 13.19 的“工作树尚未提交”表述。当前分支 feat/stream 已将相关代码提交为：

~~~text
c6cba54 streaming: add public actual-plan result verification
fa1ecd0 streaming: extract public runtime coordinator
~~~

当前已提交的事实：

- PublicStreamingCoordinator 位于 native/runtime/streaming/public_runtime.*，负责纯 C++ preflight、metadata probe、catalog select、snapshot compile、authority 创建和 execution-time revalidation；
- StreamingCatalogProvider 位于 native/runtime/streaming/catalog_provider.*，production provider 读取进程生命周期的 reviewed catalog 快照；
- c_api.mm 只负责 engine lock、GPU lock、C ABI 错误/所有权、Objective-C result dictionary 和 generate 分流；
- RunResult 已包含 PublicStreamingSelectionMetrics；
- common verifier 已核对 layout digest、stage、P/G/K/D/Q、group/pass、pass transition、multi-pool、component policy、pool/slot/worker、request retention、source lease flag 和 drain flag；
- result mismatch 在 JSON 序列化前返回 streaming_actual_plan_mismatch；
- host/contract/App build 和 resolver/fake snapshot/source revalidation 测试已通过。

本次提交仍不代表 public 可用。已知代码缺口：

1. resolve_public_streaming_locked 目前先调用 coordinator.preflight，再调用 resolve_normalized，而后者内部再次 preflight；下一步要冻结一次性 preflight/catalog snapshot；
2. ModelStreamingSnapshot::revalidate_source() 仍是默认 no-op，四个真实模型尚未接入 source lease；
3. actual verifier 当前主要验证汇总 metrics，尚未验证 per-pass/group fill matrix、logical read bytes、reader fence、source generation 和 canonical receipt digest；
4. LTX、H3 Turbo、Z-Image、Flux.2 Klein 9B 均尚未 override probe_public_streaming、compile_public_streaming、generate_resolved；
5. production catalog 仍为空，tc_streaming_options_json 只能返回 tentative/unavailable；
6. App engine-scoped options、StreamingChoice、JobStore v2 和 LTX worker 两阶段握手未实现；
7. 完整请求 process-tree calibration、8/10/12/16/20 GiB record、resident/streaming/bounded/swap 对照和 ANE 独立认证未完成。

本轮新增设计文档：

- [36 Public Adapter Code Implementation](36-public-adapter-code-implementation-spec.md)：逐文件说明 coordinator 收口、ValueModelStreamingProbe/Snapshot、fd-based SourceLease、receipt v2、四模型 public adapter、H3 authority gate 和 LTX worker-local authority；
- [37 Calibration, Performance and Acceptance](37-public-streaming-calibration-performance-acceptance.md)：五档候选探索、完整进程树采样、simulator、resident/streaming/bounded/swap 四路实验、P0–P4、App 事务、evidence/catalog/release。

当前下一步不是重新实现 executor，而是按 36 的 C0→C6 收口真实 public adapter，再按 37 为单个模型/单个 target 生成 reviewed evidence。production catalog 继续保持空，直到 source lease、receipt v2、默认零开销、完整校准和 review 全部通过。

### 13.21 C0 coordinator/catalog snapshot 收口（2026-09-17）

本轮完成并验证 C0 代码阶段：

- `StreamingCatalogProvider` 改为返回 `shared_ptr<const StreamingPresetCatalog>`，production provider 返回进程生命周期的 immutable snapshot；
- 新增不可复制的 `PublicStreamingPreflight`，内部绑定 request digest 和单次 catalog snapshot；
- `c_api.mm` 的 public resolve 顺序固定为 preflight → `make_plan_after_public_streaming_preflight` → `resolve_normalized`；
- planner 新增内部 prevalidated 入口，只跳过已经完成的 public request/selector validation，不跳过完整 model/shape/recipe validation；
- resolve 在 probe 前验证 request digest，防止 preflight 后 request 被替换；
- revalidate 使用新的 provider snapshot，能观察 catalog empty/revision/revoke；
- host fake provider 覆盖 snapshot 稳定性、请求篡改、empty catalog、revalidate stale 和 production snapshot identity。

验证结果：

~~~text
python3 -B tests/native/test_streaming_preset_resolver.py  PASS
env TURBOCIDER_NATIVE_ONLY=1 tools/native/build.sh        PASS
make test-streaming-host                                  PASS
make test-streaming-contract                              PASS
make test-streaming-audit                                 PASS（1项无 audit dylib 环境 skip）
tools/native/build_app.sh                                 PASS
git diff --check                                           PASS
~~~

C0 提交不代表 public 已开放。production catalog 仍为空；四个模型尚未 override public probe/snapshot/generate，
source lease 和 receipt v2 仍待实现，完整 8/10/12/16/20 GiB calibration、swap 对照、App selector 和 ANE
兼容也未完成。下一阶段进入 36 的 C1 source lease/value probe，首个模型仍推荐 Z-Image Turbo。

### 13.22 Framework 代码合同与验收工作簿补充（2026-09-17）

在 `60de338` C0 基线之上，本轮继续审阅了当前 `runtime/streaming`、`StageExecutor`、C ABI v1/v2/v3、
`MlxWeightPager`、四模型 descriptor/reader 和 `ModelSession` public hook 接缝，并新增两份实施文档：

- [38 Framework Code Contracts](38-framework-code-contracts-and-implementation-workbench.md)：将 C1/C2 的
  `SourceLease`、Value Probe/Snapshot、receipt v2、owner pump、K2 carry、multi-pool、C ABI、JobEnvelope、
  四模型逐文件改造、锁/取消/quarantine 和逐阶段停止条件写成代码合同；
- [39 Validation / Benchmark Workbook](39-validation-benchmark-and-release-workbook.md)：将 Host/API/runtime/model/
  memory/performance 六层证据、source/receipt 故障注入、四模型真实验收卡、process-tree target 判定、
  resident/streaming/bounded/swap 四臂、P0–P4、evidence bundle 和 release 签字整理为可执行工作簿。

本轮是设计完善，不是 C1 代码实现。以下事实保持不变：

1. `source_lease.*`、`value_probe.*`、`actual_receipt.*` 尚不存在；
2. 四模型仍未 override `probe_public_streaming`、`compile_public_streaming`、`generate_resolved`；
3. 当前 actual verifier 仍为汇总 v1，没有逐 pass/group/fence/source-generation receipt；
4. production catalog 仍为空，五档均不可 public 执行；
5. 没有新的完整请求 target、swap P3 或 ANE streaming 性能证据。

下一步仍按 36/38 的 C1 执行：先复用现有 pager/descriptor 的 open/fstat/path-replace 逻辑实现通用 fd lease
和 synthetic source mutation tests，再做 receipt v2；不能跳过这两步直接接真实 production record。

### 13.23 C1 工作树审阅与产品化设计深化（2026-09-17）

当前工作树已经完成 C1 common-runtime 实现，但尚未形成阶段提交。本次审阅确认已有：

- `native/runtime/streaming/source_lease.hpp/.cpp`：`OwnedSourceFd`、descriptor capture、`open_and_verify`、request generation、fd duplicate、path/open-fd/post-drain revalidate；
- `native/runtime/streaming/value_probe.cpp` 与 `resolved_request.hpp` value probe/snapshot 接缝；
- `tests/native/streaming_source_lease_test.cpp` 与 Python wrapper，覆盖 canonical ordering、generation、fd duplication、same-size mutation、path replacement、digest mismatch、duplicate logical id 和 snapshot revalidation；
- `SourceLease::capture()` 已实现单一 fd lineage；named path 与 canonical target 双身份校验；empty artifact 拒绝；
- `PublicPresetResolver::select/authorize`、`PublicStreamingCoordinator::revalidate` 和 public result verifier 已要求共享 lease、digest、generation，并在 pre-GPU/post-drain 执行校验；
- 当前单独执行 `python3 -B tests/native/test_streaming_source_lease.py`、resolver、host、contract、audit、native-only build 和 App build 均通过。

该结果仍不代表 public adapter 或 production record 完成。C1 common-runtime 已完成，但进入真实模型前仍必须：

1. 至少一个真实 public adapter 要使用 `SourceLease::capture()`，而不是继续按 path reader；
2. 真实 adapter 的 post-drain receipt 必须在 lease revalidate 成功后再置 `source_lease_verified=true`；
3. receipt v2 仍需补齐 per-pass/group fill、logical bytes、reader fence 和 canonical receipt digest；
4. 四模型 public hooks、完整 request closure、8/10/12/16/20 GiB calibration、App selector 和 production catalog 仍未完成。

本轮新增/扩充的设计文档：

- [40 Public Productization and App Contract](40-public-productization-and-app-contract.md)：冻结 App 只显示 Off/8/10/12/16/20 GiB、options/resolve、JobEnvelope、public/private/legacy 路由和默认路径零工作；
- [41 Scheduler / Multi-slot / Multi-pool](41-scheduler-multi-slot-and-multi-pool-implementation.md)：给出 owner pump、slot state、K/D/Q overlap、carry、serial/retain-all pool、取消和 quarantine 的实现算法；
- [42 Model Adapter Playbooks](42-model-adapter-playbooks.md)：给出 Z-Image、Flux 9B、H3 Turbo、LTX 的 source/component closure、候选族、逐文件接入和错误验收；
- [43 Toolchain / Simulation / Release Gates](43-toolchain-simulation-and-release-gates.md)：定义 inspect→compile→simulate→campaign→verify→builder、process-tree sampler、resident/streaming/bounded/swap 四臂和 P0–P4 门禁；
- [38 Framework Code Contracts](38-framework-code-contracts-and-implementation-workbench.md) 第14节：将当前 C1 工作树缺口设为进入真实 public adapter 前的强制修正。

截至本节，production catalog 仍为空，四模型仍没有真实 public hook，App 仍未开放五档，receipt v2、完整 target calibration、swap P3 和 ANE streaming 认证仍未完成。下一步严格顺序是：C2 receipt v2 → Z-Image → Flux 9B → H3 Turbo → LTX worker → App/工具链 → reviewed record。

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
| Stage执行器 | `native/runtime/streaming/context.*` | begin/run_pass/finish、单class streamed、池/worker跨pass保留、pass与step分离、lookahead/fence/drain | resident/multiclass、request orchestration、strict memory schedule、service quarantine接线 |
| LTX metadata/fill/slot | `ltx_streaming_layout.*`、`ltx_streaming_slot.*` | 真实48-block metadata、source/destination/scratch区分、固定shared buffers、chunked pread、无分配fill | session construction与同一snapshot绑定、完整allocator上界/guard |
| LTX generic projection | `ltx_streaming_descriptor.*` | fd-only snapshot、48-block source/derived bindings、11-pass/workload identity；真实checkpoint host验证 | trusted content identity、公开session接线、同fd构造/运行与完整resource plan |
| LTX exact adapter | `ltx_streaming_adapter.inc`、`ltx_blocks.c`、`ltx_streaming_plan.*`、`ltx_session.mm` | versioned create、generic layout→C ABI投影、candidate-only request owner、connector→两stage→VAE/export完整请求、status destroy/quarantine | public production资格、完整故障/重复请求矩阵、normal-target性能和bounded guard |

`layout.*`当前输出只是权重布局，不生成完整memory upper，不做任何production授权。
StageExecutor的backing由adapter拥有，SlotSafetyTracker只维护内容状态，Vacant不会释放backing账。
执行器目前拒绝resident与multi-class执行，不把编译器支持某种布局误报为执行器已经支持。

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

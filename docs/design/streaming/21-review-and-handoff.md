# 21 · 提交审阅与交接

日期：2026-09-16。本文记录本次代码整理提交的范围、验证结果和明确未完成项。
它是交接清单，不授予任何模型的 production streaming 资格；资格仍以 [12](12-acceptance-playbook.md)
和 [13](13-implementation-progress.md) 的证据为准。

本次是将此前累积的相关 dirty 工作树提交为一个**实验性开发检查点**，不是声称所有文件均在本轮实现。
源码审阅重点是入口资格、默认路径、异步所有权、C ABI 和测试接线；不是对所有模型 kernel 的重新数值认证。

## 1. 本次提交包含

- 通用 `K/G/P/D/Q` layout compiler、source/backing/materialization/workload identity。
- 持久 slot pool、generation、last-reader fence、bounded I/O、completion mailbox 和 owner 检查。
- C bridge 及构造/取消/短读/错误 drain/quarantine 测试；本次额外覆盖标准异常与非标准异常。
- memory manifest、plan、accounting、schedule、execution、watchdog、trace 和 additive C report ABI。
- LTX metadata-only descriptor、fd snapshot、固定 backing/fill、v1/v2 exact native 原型，以及
  generic layout 到 C ABI 的 request-scoped plan view。
- candidate-only LTX exact 完整请求：真实 Gemma/connector、Stage 1、upsampler、Stage 2、VAE 和 MP4 export；
  status-returning destroy、session quarantine 与 actual layout/counter 报告。
- H3/LTX memory hook 接口和 API 层的 fail-closed admission 基础。
- streaming 文档拆分、README 导航、Makefile 专项测试入口，以及归档文档中的机器路径清理。

### 1.1 本次实际修正

- `native/runtime/streaming/c_bridge.cpp`：`create_v1` 的 `catch(...)` 原先没有像
  `catch(std::exception)` 一样转交 quarantine handle。如果 adapter 在部分构造后抛出非标准异常且
  drain 失败，局部 owner 析构会触发 `StageExecutor` 的终止保护。现在返回失败并把 handle 交还 caller，
  仅在后续安全 drain 成功后销毁，不吞掉失败、不强制释放仍可能被引用的对象。
- `tests/native/streaming_c_bridge_failure_test.cpp`：四种标准/非标准异常 × 安全/不安全清理组合；
  验证失败保留 handle、错误线程不能销毁、owner 重试成功和重复 destroy 不 double free。
  由 `test_streaming_layout.py` 统一执行，同样进入 ASan/UBSan 和 TSan。
- `native/core/stream_slot_c.h`：明确 adapter callback 的返回值错误协议和 `destroy_pool` 不抛异常约束；
  本次的防御性异常处理不等于鼓励跨 C ABI 抛异常。
- `Makefile`：抽出 host/contract/Metal 三个专项入口，原 `make test` 的 host/contract 测试集合保留，
  真实模型权重测试依然 opt-in。不新增 runtime 参数，不开启 API 资格门，不改数值 kernel。
- `native/models/ltx_runtime/ltx_streaming_plan.*`：把 immutable generic layout 投影为 native v2 ABI，
  由一个 owner 稳定持有 plan arrays 与 borrowed metadata snapshot，避免 session 手写第二套布局逻辑。
- `native/platform/apple/ltx_session.mm`：内部 candidate constructor 可执行首批 exact tuple；public constructor
  仍拒绝。exact handle 不进入 legacy void deleter，Stage 2 后安全 destroy；不安全时保留完整 request state。
- exact result 写回 resolved/actual layout、candidate authority、digest 和实际 counter，避免“已经执行但 plan
  仍显示 plan-only”的验收歧义。默认 resident/legacy streamed 分支不创建 descriptor、executor 或 worker。

## 2. 已执行验证

| 检查 | 结果 |
|---|---|
| `git diff --check` | PASS |
| native-only `tools/native/build.sh` | PASS |
| `test_streaming_layout.py` | PASS：3535 layout cases、executor/C bridge、构造失败回归 |
| 同脚本 ASan/UBSan | PASS |
| 同脚本 TSan | PASS |
| `test_ltx_streaming_layout.py` | PASS |
| `test_ltx_streaming_descriptor.py` | PASS（合成 48-block metadata；K1/K2/K3 plan view ABI 投影） |
| `test_ltx_streaming_snapshot.py` | PASS |
| `test_streaming_contract.py` | 8 PASS（重建后的 candidate） |
| `test_ltx_candidate_streaming_gate.py` | PASS：public fail-closed，内部 constructor 独占实验 authority |
| `test_contract.py` | 81 PASS + 1 Wan fixture SKIP |
| 仓库 layout/independence/C++ boundaries | PASS |
| `test_memory_execution.py` | PASS |
| `test_memory_plan_compiler.py` | PASS |
| memory/scheduler 13 项独立补跑 | 11 个 host suite PASS；H3/LTX GPU hook 在沙箱内因无 Metal 而 SKIP |
| `compare_streaming_legacy_plans.py` | 当前 merge candidate 与 `dev@ad343d4` 的 18 项一致；只证明 plan/error 兼容，不证明性能。旧的 pre-Z-Image 基线会因 dev 新增 `z-image-turbo/residency=streamed` 而产生预期差异。 |
| 当前文档检查 | 23 个 active Markdown、193 个本地文件链接、3 个内嵌 JSON 与 6 个 JSON 示例有效；归档历史正文不参与格式验收 |
| `make test PYTHON=/usr/bin/python3` | **未完整通过**：沙箱内运行到 `test_video_timing.py` 时 writer 启动失败而停止；后续 memory/model/inventory suite 已逐项补跑通过。部分测试因 numpy/fixture/Metal 不可用 SKIP。 |
| `test_video_timing.py` 沙箱外重跑 | PASS（1 test），支持沙箱限制判断；未修改视频代码。这不自动使整次 `make test` 变成 PASS。 |
| 真实 LTX 完整 candidate 请求 | PASS：64×64×9、11 steps、Gemma cache/connector、两 stage、VAE、MP4 export |
| exact/legacy 质量 | PASS：Stage-2 BF16 byte-exact；9 帧 RGB correlation/cosine=1、MAE=0、motion error=0 |

真实 Metal 和 LTX 实模型证据见 [13 第12节](13-implementation-progress.md)；本次没有把 tiny 结果标记为
正式 P0/P1 性能验收。没有运行系统 pressure 或 swap 破坏性实验。

最终重建 candidate dylib SHA-256 在提交前验证后写入第7节。本机使用显式 `MLX_ROOT` 和
`TURBOCIDER_NATIVE_ONLY=1`；binary hash 只标识本次构建，不补足历史 before 的源码 provenance。

## 3. 当前阻断项

以下问题不阻止保存实验性代码检查点，但**阻止启用 production exact/bounded 路线**。
不能用删除资格门的方式“完成接线”。

1. LTX exact 已接 candidate request owner 和完整请求，但 public manual streaming 资格门仍然保留；
   尚无 production registry 记录，不能将内部 authority 改成用户字段或环境变量。
2. session 级 success→success、shape A→B→A、取消/阶段失败/unsafe destroy/quarantine 矩阵尚未完成；
   normal-target、audio/I2V/LoRA/ANE/近似分支不在首批资格内。
3. whole-request resource closure 未覆盖所有 MLX/MPSGraph/Metal/host/media/driver allocation，
   不能宣称 Y 是 OS/RSS 硬上限或绝对零 swap。
4. executor 现在支持 single-active-pool streamed 和 ordered multi-class barrier；resident、跨 component
   DAG、session retention 和服务级 quarantine/eviction 还未闭环。v2 C ABI 只描述有序 pool 列表，不代表
   任意模型已有真实 adapter 或执行资格。
5. 只有 tiny 短对照，尚无 normal-target release ABBA P0/P1/P2、真实低内存 P3 或策略 P4，
   不能宣称正式性能认证或快于 swap。
6. H3、Flux、Z-Image 尚未获得 exact layout registry 资格。

### 3.1 审阅发现的具体接线风险

| 优先级 | 代码位置 / 现状 | 下一实现必须满足 |
|---|---|---|
| 发布阻断 | `native/api/c_api.mm::finalize_memory_failure` 在 drain 异常后仍尝试 `session.unload()`；`memory_execution` 是调用栈 owner，`tc_engine_free` 直接 delete engine | quarantine 必须保留整个 session/context/metadata/callback owner；无安全 drain 证明时不得 unload/free。目前 production memory registry 为空，不能把已有 bool 标志视为完整隔离机制。 |
| 部分完成，仍发布阻断 | `LtxNativeSession` 已有独立 exact owner、稳定 plan view 和 status destroy；完整 success 请求已通过 | 补 success→success、A→B→A、cancel、Stage 2/VAE/export failure、unsafe destroy/quarantine 和 engine teardown；证据通过前不替换 public gate。 |
| 已在 dev 合并时修正，仍需 P0 | API 层曾对所有请求前置 `make_plan`，正常 session 内还会再规划 | 现在仅显式 `memory_constrained.enabled` 时 API 预规划；默认 resident、dev Z-Image streamed 和既有 streamed 路线不再重复 planning。仍需用 P0 测量，不以源码检查代替时延证据。 |
| 已在 dev 合并时修正，需成功 admission 测试 | API 曾在 `prepare_memory_execution` 调整 runtime denoiser budget 之前复制 request | 现在 admission 后再次从 `request_plan->request` 传播最终预算到 generate/prepare；production registry 仍为空，后续要增加可执行 fixture 验证实际 session 与 report 使用同一预算。 |
| 范围限制 | `retry_drain()` 有阻塞 join；snapshot 是 stat 身份，不是不可变文件隔离 | 不宣称硬超时回收或抗并发改写；必要时单独设计进程隔离/受信不可变 artifact，不扩大现有合同。 |

以上均来自当前源码核对；未执行可触发危险释放的真实 GPU 故障实验，也未在本次整理中仓促重构 request/service 生命周期。
有关施工和退出条件继续使用 [20 第 4–5 节](20-layout-first-implementation-and-acceptance.md)，不另建第二套调度器。

## 4. 下一步顺序

1. 补 session 级生命周期/故障矩阵，并验证 actual report 在 success 与失败终态一致。
2. 完成 normal-target、largest 和 repeated-request 质量/lifetime；随后执行 release ABBA P0/P1。
3. 接入完整 resource ledger 和 guard，执行 P2/P3；结果合格后才登记 LTX 精确 tuple。
4. 把 snapshot 绑定可信 artifact/content identity，明确 service eviction 和最终 quarantine owner。
5. 按 adapter contract 推进 H3 K2/G1，再做 Flux/Z-Image component-staged 或模型专用 block adapter。

任何后续提交若改变 runtime 生命周期或 hot path，都应重新运行本文件第 2 节的最小回归，并在
[13 实施进度](13-implementation-progress.md) 记录 candidate binary identity、测试范围和未通过门禁。

## 5. 复现与提交边界

```sh
# MLX_ROOT 指向本机有效 MLX SDK；先 build，再运行链接当前 dylib 的 contract/snapshot 测试。
env TURBOCIDER_NATIVE_ONLY=1 tools/native/build.sh
make test-streaming-host PYTHON=python3
make test-streaming-contract PYTHON=python3
env TC_STREAMING_SANITIZER=address,undefined python3 -B tests/native/test_streaming_layout.py
env TC_STREAMING_SANITIZER=thread python3 -B tests/native/test_streaming_layout.py
make test PYTHON=python3
```

`make test-streaming-metal PYTHON=python3` 是显式的合成 GPU backing 测试，设备不可访问时可能 SKIP；
不会自动加载真实模型或施加内存压力。实模型 harness 命令见 [13](13-implementation-progress.md)，
[12](12-acceptance-playbook.md) 的性能 campaign 仍需单独运行。

视频时序已在沙箱外单独重跑通过；整仓库测试仍需在合适依赖/权限环境一次跑完，不能将部分 suite
的 PASS 或成功的补跑合并成未经执行的完整 `make test` PASS。

提交只包含源码、测试、构建接线、设计与配置示例；不包含 checkpoint、生成媒体、二进制、原始 benchmark
输出或个人凭证。原始 before binary 仍留本地，其 provenance 限制见 13 第 3 节。提交不等于发布，不 push。

## 6. 合并 dev 的整合记录（2026-09-16）

合并基线为 `dev@ad343d4`。唯一文本冲突位于 `tests/native/test_contract.py`，解决方式是同时保留
memory-constrained fail-closed 用例和 dev 的 Z-Image streamed 合同用例。Z-Image 权重 streaming 的核心
文件与 dev 保持逐字节一致：`z_image.cpp/.hpp`、`weight_stream.hpp`、`z_image_weight_stream.mm` 和
`z_image_module.cpp` 没有为了接入通用框架而改写热路径。

整合时修正了两处 API 接缝：

- `tc_engine_generate/prepare` 只在显式 `memory_constrained.enabled` 时提前 `make_plan`；默认 resident、
  dev Z-Image streamed 和既有 streamed 请求不再在 API 与 session 重复规划。
- constrained admission 调整 denoiser budget 后，再将 `request_plan->request` 传播给实际 session，避免
  执行预算与 terminal report 不一致。

验证结果：

- native-only 和 Swift/App build PASS；82 项 native contract 中 81 PASS、1 个 Wan fixture SKIP；
  streaming compiler/executor、LTX snapshot、memory 11-suite 和 repository boundaries PASS。
- 真实 Metal 合成测试 PASS：Z-Image slot reuse/cancel/bad metadata，以及通用 K=1/2/3 reader fence。
- HubClient localhost fixture、Studio behavior、RunInsights PASS；前两项因 localhost/Trash 权限在沙箱外运行。
- merge candidate 与 dev 的 18 项 legacy plan/error 完全一致。

### 6.1 与 dev 的短性能对照

同机、本地 Comfy BF16 Z-Image、256×256、2 steps、seed 314159；dev dylib SHA-256
`1d32961668237a2941bf43ce89beb458fe95c2a87daadf8dbac0fd671b9098b9`，merge candidate
`f47c0ba1675fc0a92bdf4e54a4563fa681e1c61bebf19f93c5164a976b092807`。每种模式排除首轮后取 7 个
warm 样本：

| 模式 | dev wall median | candidate wall median | candidate/dev | MLX peak |
|---|---:|---:|---:|---:|
| resident | 0.618665 s | 0.618881 s | 1.00035（+0.035%） | 两者均 14,185,782,596 bytes |
| streamed，6 GiB budget | 1.913177 s | 1.890266 s | 0.98802（-1.20%） | 两者均 4,778,391,876 bytes |

resident denoise median 的 candidate/dev 比值为 1.00031；streamed 为 0.99026。全部 resident/streamed、
dev/candidate 输出使用同一 PNG SHA-256：
`a38f7028b441878dc146a316ed1945a104c07b46b36e06292509f97feccd65a9`。

该短 campaign 支持“本次合并未观察到相对 dev 的回退”，但不是 [12](12-acceptance-playbook.md)
定义的 normal-target P0/P1：样本小、没有置信区间、系统状态未隔离，也不用于宣称 streaming 普遍快于
resident。正式发布仍需按 P0–P4 运行。

## 7. LTX exact candidate 最终检查点（2026-09-16）

最终native dylib SHA-256：

```text
5d72812ec38af6e6164e7ff318dc362ab7ccbef14091c90ae024bb30107990c2
```

最终源码重建后，host/contract专项再次PASS；真实GPU完整exact请求再次成功。结果中的
`plan.streaming`不再遗留`plan_only`：

```text
eligibility=experimental_candidate
execution_supported=true
resolution_state=executed_exact_v2
authority=private_candidate_constructor
resolved_layout.digest=40b13657a746d27a9a44ec11f53485ac4163f23e92408eb7aef2bca1fdfca38a
actual_layout.digest=40b13657a746d27a9a44ec11f53485ac4163f23e92408eb7aef2bca1fdfca38a
```

默认resident的热缓存短对照为dev 9.039951秒、candidate 9.069556秒；denoise分别为
6.915945秒和6.918010秒。matched tiny streamed第二次交错样本为legacy 8.519114秒、exact
8.511698秒；denoise分别为7.674939秒和7.489339秒。两路Stage-2 BF16 byte-exact，最终视频逐帧相同。
完整数字、限制和raw artifact范围见[13第12节](13-implementation-progress.md)。

### 7.1 同一engine生命周期补充

新增`test-ltx-streaming-lifecycle`显式目标并在真实GPU运行通过。一个candidate engine连续覆盖：

- Stage 1取消→成功；success→success；64×64→128×64→64×64。
- Stage 2取消→成功；VAE边界取消→成功；export失败→成功。
- 每次成功的actual/resolved layout、P8/G1/K3/D2/Q3 counter和latent hash一致。

所有A/recovery Stage-2 latent SHA-256为
`7db71bf03027942b53af69ab914714583fe8f8d4c3ed2faf60f4aa4ed43f28fd`；
A/B layout digest分别为`40b13657a746d27a9a44ec11f53485ac4163f23e92408eb7aef2bca1fdfca38a`
和`9c1b0b90ab16f89541625a7284489a09b24557f1f88bfb38d8c77505a57120c8`。

session quarantine现在会在下一次同owner `generate()`先重试status destroy：成功才继续，失败则保持完整
owner并返回quarantined错误。确定性unsafe-drain和engine teardown覆盖随后已在本文件第8/9节闭环。

最终默认resident热缓存样本为9.010764秒，denoise 6.950530秒；相对此前dev热样本9.039951秒/
6.915945秒没有观察到wall回退，denoise差约+0.5%，仍属于tiny单样本，不签P0。

该检查点支持保存和继续开发；后续已完成unsafe quarantine矩阵和tiny default P0，但normal-target P0/P1、
whole-request bounded guard、低内存P3和其他模型adapter仍是下一阶段门禁。

## 8. 后续实现：生命周期 fault 与性能 campaign 工具

当前 dirty 工作树又完成两项此前明确未完成的能力，但尚未提交：

1. LTX exact 生命周期 fault 闭环：test-hook build 可注入 first-fill cancel 和多次 unsafe destroy；
   session quarantine、engine teardown process quarantine 及安全 retry 已由真实 GPU 矩阵覆盖。release dylib
   不导出 test-hook 符号，request/schema/public header 没有新增绕过开关。
2. T2/T3 性能工具：两个独立持久 worker、ABBA/BAAB raw campaign、source/build identity、quality/fault/
   environment/audit evidence、按 block bootstrap verifier 和 CPU-only 反例测试。

新增可执行入口：

```sh
make PYTHON=python3 test-streaming-campaign
make PYTHON=python3 test-streaming-source-identity
```

真实 LTX tiny tool smoke 位于：

```text
/private/tmp/turbocider-streaming-campaign-smoke-metal-20260916
```

该 smoke 的4个 matched pair全部成功，Stage-2 latent逐pair byte-exact，candidate 默认 resident
`block_streaming.enabled=false`且slot allocations/refills为0。verifier仍正确返回`INCONCLUSIVE`：
environment/audit/source provenance不完整，且只有tiny 4-pair。diagnostic中 candidate/dev denoise median
约1.03156、P95约1.11662，样本区间很宽；这是需要复测的回归信号，不是可忽略噪声，也不能被wall中
baseline较慢的load阶段掩盖。

交接时应先完成clean dev source export/build和独立default audit，再扩充normal-target P0 block；如果
denoise区间仍越过1.02，先定位stage1/Metal/thermal/编译差异并修复，不以低内存收益抵消默认回归。
production registry仍保持为空，P3 pressure仍未授权、未运行。

补充：随后使用 `dev@ad343d4` clean source export 重新构建 baseline，并由 source identity 工具绑定
完整 source manifest；同一工具链4-pair复测的 candidate/dev 比值为 wall median 1.00073、denoise median
1.00262。旧 saved dylib 的 +3.16% denoise 信号未复现。新 artifact 为
`/private/tmp/turbocider-streaming-campaign-clean-source-metal-20260916`。由于 wall median 的小样本区间
上界仍为1.04052、audit/environment仍为partial，最终结论继续是`INCONCLUSIVE`，不是P0 PASS。

## 9. Audit闭环与20-pair default P0补充

后续已完成独立audit build、私有snapshot/reset ABI、campaign自动采集和release符号隔离。三种构建：

| 变体 | audit符号 | lifecycle test-hook符号 |
|---|---:|---:|
| release | 0 | 0 |
| audit | 2 | 0 |
| test-hook | 0 | 4 |

真实LTX默认resident audit请求成功且五类counter全0；显式executor测试观察到pool/worker非0。新test-hook构建的
完整GPU生命周期矩阵再次通过，artifact：
`/private/tmp/turbocider-ltx-lifecycle-audit-integration-20260916/lifecycle/lifecycle-summary.json`。

20-pair audit bundle：
`/private/tmp/turbocider-streaming-campaign-audit-20260916/bundle-audit10`；20-pair release bundle：
`/private/tmp/turbocider-streaming-campaign-audit-20260916/bundle-release10`。两者均40/40成功、quality完整、
fault=0、environment/source/audit完整。release verifier为`PASS`：wall median ratio 0.99990（95%上界
1.00337）、wall P95 ratio 0.99857（上界1.01005）、denoise median ratio 1.00076（上界1.00511）。

因此当前证据可以签“LTX 64×64×9、11-step、默认resident tiny tuple相对clean dev无回退”，不能扩大为
normal-target、legacy-streamed、P1/P2/P3或production资格。下一优先级变为normal-target resident和既有
streamed P0、同义P1、whole-request closure/P2；pressure/swap仍未授权。

## 10. dev 合并与回归复核（2026-09-16）

`dev@ad343d4e5139c9a5f13e29ff9e926e8eb1ae2f39` 已通过 merge commit
`566f7a6faf07734255cee229ca31ca3af0bc154c` 合入 `feat/stream`；随后保留并继续完成了
LTX candidate lifecycle、fault/quarantine、audit counter 和 campaign 工具改动。当前分支确认满足
`git merge-base --is-ancestor dev HEAD`，没有未解决的 merge index 或冲突标记。

合并后的 streaming 专项回归结果：

- repository、ABI/contract 82 项、host layout/descriptor/executor、LTX snapshot、campaign verifier、
  source identity、audit counter 全部通过；Metal 不可用时按既有规则 SKIP。
- `test_ltx_streaming_snapshot.py`、`test_ltx_streaming_descriptor.py` 和 opt-in
  `test_ltx_streaming_model.py` 的临时 Apple 编译命令统一显式传入 `xcrun --sdk macosx --show-sdk-path`
  得到的 `-isysroot`，避免在不同 Xcode/macOS SDK 默认路径下误报 Foundation header 缺失。
- 合并后的内存、H3、LTX、Z-image、Flux/Wan 相关 Python contract 回归以及 inventory 回归通过；无
  streaming 改动引起的新失败。

完整 `make PYTHON=Python/bin/python3 test` 在当前机器唯一失败的是既有的
`tests/native/test_video_timing.py`：`video_timing_probe` 在 AVFoundation writer 启动阶段返回
`video writer start failed`，尚未进入帧率/时长断言。该测试与本次 merge 的 streaming 文件无 diff，且
失败发生在无 Metal 依赖的媒体 writer 环节；使用同一台机器上 clean `dev@ad343d4` dylib 的独立
probe 也复现同样错误，因此暂记为环境/媒体栈阻塞，不将其误判为 streaming 回退。交付前应在可用
AVFoundation writer 的 clean build 上重跑并保留日志。

性能保护仍以独立 20-pair ABBA/BAAB campaign 为准：release candidate 与 clean dev 的 LTX
64×64×9、11-step 默认 resident tiny tuple 为 40/40 成功、质量 byte-exact，wall median ratio
0.99990（95% bootstrap 上界 1.00337），denoise median ratio 1.00076（上界 1.00511），五类
audit counter 均为零。该证据支持“不影响默认 resident 热路径”，但不扩大为 normal-target、legacy
streamed、低内存 bounded-memory 或其他模型的性能承诺。

## 11. multi-class executor 增量复核（2026-09-16）

本次后续增量把 ordered multi-class barrier 从 compiler-only 推进到 `StageExecutor` 和 versioned C bridge：
v1 ABI 保持兼容，v2 显式描述 pool/group 顺序并把 pool id 传给 adapter 的 allocate/destroy。当前实现仍是
单活动 pool；跨 class 先 drain，再 destroy，再 create，跨 class 不预取，不同时保留异构 pool。

验证证据：

- host 3535 layout cases、14 个 K/D/Q 组合、multi-class 两 pool/三 pass、乱序双 reader、非法顺序、
  pool 创建失败和 C bridge failure 全部 PASS；ASan/UBSan、TSan 全部 PASS。
- release build：`/private/tmp/turbocider-multiclass-release-20260916/libturbocider.dylib`，仅导出
  `tc_stream_executor_create_v1/v2`，无 audit/test-hook 符号。
- audit build：`/private/tmp/turbocider-multiclass-audit-20260916/libturbocider.dylib`；默认 LTX resident
  请求成功，五类新 framework counter 全 0。
- 当前源码 test-hook 真实 LTX lifecycle：
  `/private/tmp/turbocider-multiclass-lifecycle-20260916/lifecycle/lifecycle-summary.json`，所有取消、
  重复请求、A→B→A、阶段失败、unsafe destroy/session/process quarantine 与恢复均通过，latent hash 未变。
- post-change 沙箱外 tiny 默认 resident 4-pair campaign 8/8 成功、质量 byte-exact，wall median 1.00321、
  denoise median 1.00118；样本太小且 release audit 为 partial，verifier 保持 `INCONCLUSIVE`，不作为
  P0 PASS。此前 clean-source 20-pair release PASS 仍是默认路径主要性能证据。

这次增量没有给 H3/Flux/Z-Image 授予 adapter/registry 资格，也没有开启 public exact、bounded memory 或
swap 对比；后续必须继续按本文件第 4 节的 normal-target/P1/P2/P3 顺序推进。

## 12. 最终合并交付核对（2026-09-16）

`dev@ad343d4e5139c9a5f13e29ff9e926e8eb1ae2f39` 已在 merge commit
`566f7a6faf07734255cee229ca31ca3af0bc154c` 合入 `feat/stream`；当前工作树无冲突。随后提交前增量已完成：

- `StageExecutor` 支持 ordered multi-class barrier；跨 class 严格执行 drain → consume → quiescent check →
  destroy → create，且 pool 激活采用已验证 index，避免多 class 扩展产生 O(P²) 查找。
- C ABI v1 保持兼容，v2 显式描述 pool/group 顺序与 pool-aware allocate/destroy；未引入第二套 pager/scheduler。
- release/audit/test-hook 三种最终重建、host/ASan/UBSan/TSan、Python contract/audit/verifier/source identity
  回归全部通过。
- 最终真实 LTX candidate lifecycle 全部通过，最终 audit default resident 五类 counter 全 0。
- 最终 4-pair resident smoke 8/8 成功且质量 byte-exact；wall median 0.99334、wall P95 0.99232、denoise
  median 0.99832。样本量不足以升级为 P0，正式性能结论仍以此前 20-pair tiny P0 PASS（wall median
  0.99990、denoise median 1.00076）为准。

最终 binary/artifact 位置和 hash 详见 `13-implementation-progress.md` 第13.7节。当前仍明确未完成：
normal-target/legacy-streamed P0、same-layout P1、guard/whole-request closure、真实低内存或 swap P3、
public exact production registry，以及 H3/Flux/Z-Image adapter；production registry 必须保持为空。

## 13. H3 descriptor 增量交接（2026-09-16）

提交 `28d351c` 之后，本次整理范围又包含一批 H3 metadata-only descriptor 改动：
`h3_streaming_descriptor.*`、H3 shard header view、shape 常量、build/Makefile 接线和
descriptor 测试。普通、ASan/UBSan、TSan、streaming host/contract、native contract、C++
boundary 与 release build 已通过。

审阅结论：

1. 默认 H3 runtime 的 `configure_active_blocks()` 仍只在 load 阶段调用共享 mask，descriptor
   不会被默认 request 创建；没有新增 worker、pool、probe、cache-clear 或同步点。
2. descriptor 只做真实 shard/header/range/shape 预检，snapshot 变化、重复 tensor、缺字段、
   错 dtype/shape 均 fail closed；sparse fixture 不读取大 payload。
3. `StreamingPlanView` 明确拒绝尚未表达的动态 shortcut 和非 K=2/G=1 布局；不能把该 plan
   误写成 public exact 资格或 whole-request memory upper。
4. 下一步仍是 H3 candidate execution bridge、真实 Metal fence/last-reader、完整资源 ledger、
   session lifecycle 和 normal-target P0/P1。production registry、bounded-memory 和 swap
   对比继续保持未完成。

最终 release SHA-256 为
`c18197a6a25fda3e389b2b6b90e7646d2ea909c29485599785301c53be67b3a8`。与 clean
`dev@ad343d4` 的 4-pair tiny resident ABBA/BAAB smoke 中，8/8 请求成功、Stage-2 BF16
byte-exact、audit 五类计数均为 0；candidate/dev 的 wall median/P95 为 `0.99506`/`1.00518`，
denoise median/P95 为 `1.00063`/`1.00289`。点估计没有显示相对 dev 的性能回退；由于只有
两个 ABBA block，wall median bootstrap 上界为 `1.03423`，总判定仍是 `INCONCLUSIVE`，不得
将它写成 normal-target P0 通过。

## 14. 2026-09-17 续接：dev 已同步，cross-pass carry 检查点

远端同步后 `origin/dev`、本地 `dev` 和 `FETCH_HEAD` 均为 `ad343d4`；该提交已在
`566f7a6` 合入，显式 `git merge dev --no-edit` 返回 `Already up to date`。因此本轮没有新的 merge
冲突；此前唯一的 contract 文本冲突仍按第6节的 additive 方式解决。

本轮增量已完成：

- ABI v3 显式 `reload/carry_first_group`；v1/v2 ABI 与默认行为不变；
- 单 pool K2/G1 跨 pass carry、奇偶 suffix slot rotation 和唯一 Ready boundary；
- carry ticket 的 pass/step/group/slot/generation 核验，错误 step fail closed；
- C adapter 异步 job 自持 blocks，避免临时 group 指针悬空；
- H3 descriptor 投影 v3 plan，并通过 fake fill/encode/reader fence 与 failure matrix。

当前 focused 测试均通过：streaming host/contract、3535 layout case、14 K/D/Q、multi-class、C ABI v3、
H3 descriptor fake execution、82 native contract（1 个既有 fixture SKIP）、repository、memory 和 H3 schedule；
ASan/UBSan 与 TSan 也通过。实现与设计提交为
`b975b29a8407275532b44078d90bdd4388974c86`。

该 clean commit 的 release dylib SHA-256 为
`672efa3b41932ee3299eefdaa51d135f9fba2091dc5ada232545734d17f09c56`；独立 audit build 默认
LTX resident 请求成功，五类新框架计数全部为 0。相对 clean `dev@ad343d4` 的 10-block/20-pair
ABBA/BAAB bundle 位于：

```text
/private/tmp/turbocider-cross-pass-carry-campaign-20260917/bundle-20pairs
```

结果为 40/40 成功、20/20 Stage-2 BF16 pair byte-exact、fault=0、environment/source/audit 完整，verifier
`PASS`。candidate/dev 的 wall median/P95 为 `1.00807`/`1.00178`，denoise median/P95 为
`1.00205`/`1.00634`；wall median 和 denoise median 的 block-bootstrap 95% 上界分别为 `1.01152`
和 `1.00967`，均低于 1.02 门槛。该结论仅覆盖冻结 tiny 默认 resident tuple。

交接时不得把该状态描述为真实 H3 execution：production registry 仍为空，真实 H3 block fill、Metal encode、
last-reader fence、request/session lifecycle、normal-target P0/P1 均未完成。默认 LTX resident/legacy 路径没有
调用 H3 descriptor 或 v3 executor；post-change audit/P0 已验证这一点，但真实 H3 candidate execution 与
normal-target/legacy-streamed/低内存性能仍需后续闭环。

## 15. 2026-09-17 续接：先关闭 LTX P1 证据缺口

本轮新增代码位于 LTX result telemetry、campaign runner/verifier 和 verifier tests。交接后的直接执行顺序：

1. 使用 engine_lifecycle=per_request 运行 LTX legacy streamed 与 generic exact；双方固定
   P8/G1/K3/D2/Q3，worker 进程可复用，但 engine 不跨请求保留。
2. 检查 semantic-equivalence v2：schema v1/v2 workload identity、P/G/K/D/Q、startup reload、
   reader/kernel/format、conditioning、upsample、request retention 和每请求 440 fills 必须完全一致。
3. one-block 成功后运行 2-block smoke；随后冻结 10-block/20-pair，wall/denoise median 上界 1.02、
   wall P95 上界 1.05、Stage-2 latent byte-exact。
4. P1 通过后才开始 Z-Image descriptor shadow 和 generic adapter；原 ZImageWeightStream 保留 fallback，
   同布局 overhead 超过 2% 不切 public 路由。
5. 再整理并验收当前工作树的 H3 real candidate；Flux 在 fusion 与 compiled graph address stability 审计后
   决定 G>1 group streaming，不机械复制 LTX G1。

本轮 host 证据全部通过，但真实 Metal P1 命令的沙箱外授权被自动审批链路中断，没有产生性能 bundle。
这不是性能失败，也不是 P1 通过；下一位执行者必须取得真实 Metal 权限后继续同一冻结协议，不能退回
persistent-engine legacy/exact 对比。

## 16. 2026-09-17 续接：LTX 初始 P1 已关闭，转入 Z-Image shadow

第15节的阻断已经关闭。正式 release campaign 使用同一 dylib 的 public legacy streamed 与 private generic
exact constructor，双方每请求重建 engine，固定 P8/G1/K3/D2/Q3 和 440 fills。10-block/20-pair bundle为：

```text
/private/tmp/turbocider-ltx-p1-10block-20260917-v1
```

40/40 measured 请求成功，20/20 Stage-2 BF16 pair byte-exact，semantic-equivalence、完整 environment、
独立 audit 和 manifest 均通过；wall median/P95 ratio为`0.98810`/`0.98953`，denoise median ratio为
`0.97249`，对应95%上界均低于冻结门槛。steady framework allocation/thread-create为0/0。详细数值和
binary identity见13第13.11节。

实现中同时移除了 C bridge refill 热路径对`group.blocks`的vector复制，并为audit ABI增加setup后稳态计数。
单pool LTX为0；当前multi-pool切换会明确报告非零，不能借LTX资格替多class模型签核。

下一顺序：

1. Z-Image先做只读safetensors metadata descriptor和private shadow projection，不触碰现有默认热路径。
2. 首个目标布局严格匹配现有专用实现：30 blocks、13 tensors/block、K2/G1/Q1/D0、reload、固定权重常驻。
3. sparse fixture覆盖缺tensor、dtype/shape、重叠range、stale snapshot和不支持布局；shadow阶段不读payload、
   不建MLX/Metal buffer或worker。
4. descriptor通过后才用generic adapter包装现有pread/double-buffer/bind/`mx::eval` fence；专用实现保留
   fallback，同布局P1 overhead超过2%不得切public route。
5. Z-Image完成后正式整理H3 real K2/G1 candidate；Flux继续先审计fusion和compiled graph地址稳定性。

LTX当前只取得冻结tiny initial tuple资格，normal-target、largest/repeated-request、P2/P3和public registry
仍未完成，不能把本节扩大为全模型或bounded-memory发布结论。

## 17. 2026-09-17 续接：Z-Image 冻结 P1 已关闭

当前工作树已经把 Z-Image 从 descriptor shadow 推进到 private generic execution，并完成正式
10-block/20-pair P1。最终 bundle：

```text
/private/tmp/turbocider-z-image-p1-10block-20260917-v3-immediate-reader
```

40/40 请求成功、20/20 PNG pair byte-exact，wall median/P95 ratio `0.99986/0.99913`，denoise median
`1.00313`，完整 environment/source/audit/semantic 均通过，steady allocation/thread-create为0/0。
独立 verifier 已重跑并返回 `PASS`。

关键审阅结论：同步 `mx::eval` adapter 必须在返回前证明设备完成，随后可通过
`ReaderSet::already_complete` 由 owner 直接 seal+complete；异步 H3/Metal adapter不得复用该声明，除非真实
command-buffer completion已经发生。Apple generic I/O worker保留user-initiated QoS；更高QoS和显式disk
policy的失败实验未保留。新增refill/wait telemetry为additive字段。

public Z-Image gate和production registry继续关闭。下一位实现者应从当前H3 real candidate审阅开始，不再重复
Z-Image P1；先补真实last-reader/lifecycle，再冻结H3 K2/G1 policy。Flux仍在H3之后进行fusion/address
stability审计。

## 18. 2026-09-17 续接：H3 Turbo 工程验收收口

H3 本轮只覆盖 MiniMax H3 Turbo 原始 BF16。generic candidate 已完成真实 qkv/out/fc1/fc2 ranged fill、
两个 Metal shared slot、真实 command-buffer reader、fusion reader closure、v3 cross-pass carry、取消、
drain 和销毁；legacy slot authority 在 exact 模式下被明确失效。

原 8 MiB 分块 reader 的正式 v2 campaign 为性能 FAIL；改成与 legacy 相同的每矩阵一次大 `pread` 后，
v3 20-pair bundle 40/40 成功且 artifact 全部一致。wall median/P95 ratio 为 `1.01066/0.99511`，denoise
median/P95 为 `1.01185/0.99268`。点估计进入2%/5%工程范围，但10个物理SSD block的bootstrap上界仍跨线，
strict verifier 保持 `INCONCLUSIVE`。这是保留的统计事实；用户确认 H3 仅覆盖 H3 Turbo，并接受该范围内的
合理I/O波动后，当前冻结 tuple 的工程验收已经完成，不再通过追加TB级读取反复抽样。

独立 audit exact 请求为 worker/pool `1/1`、steady allocation/thread-create `0/0`，输出 hash
`bcfc8bb6686e0388`。专项 host/contract/campaign 回归全部通过。H3 public gate和production registry仍为空，
full video、normal-target、P2/P3与其他 H3 变体均未获资格。后续不再规划其他 H3 变体；只有明确的
correctness/lifecycle/default-path回归才重新打开 H3 工作项。

下一位实现者直接进入 Flux：先完成 compiled graph/binding address-stability 探针和 double/single stream
class inventory，再决定 reusable arena；不要重复 H3 campaign，也不要把 H3 的 Metal 异步 reader 合同直接
套到 MLX 同步边界。

## 19. 2026-09-17 续接：Flux 9B private execution 与 audit

Flux范围冻结为Klein 9B BF16 eager GPU。当前工作树已完成真实execution，而非metadata-only：9个fixed tensor
resident，8个dual和24个single block通过统一executor streaming；两个K2 pool采用`retain_all`，Q2 worker
并行`pread`，同步`mx::eval`作为last-reader。Flux 4B compiled graph完全未接入该路径。

本轮新增`MlxWeightPager` sparse Metal fixture并收紧setup source range/compiled aligned span校验。首次真实复测
发现测试误把256-byte slot span等同于source tensor bytes，已经改为分别验证aligned backing与exact read bytes；
修正后pager fixture、完整build、host/contract/audit均通过，真实Flux请求恢复成功。这一失败发生在读取前，未
触碰默认resident路径；pager fixture的ASan/UBSan和TSan也均通过。

两步resident与Q2 exact PNG byte-exact，hash为
`5b39235aab9acbbccbce6b1fffdc672cbf5f138c2f935db1fdaf3fb2efa1d108`。Q2将MLX peak从
`18,303,578,036`降至`11,693,804,356` bytes；wall/denoise分别为`1.951123/1.383473 s`，resident为
`1.338198/0.859606 s`。这是内存策略权衡，不是框架P1结论。

独立audit build的真实Q2结果为worker/pool `2/2`、steady allocation/thread-create `0/0`；请求建立和
denoise→VAE交接合计记录4次weights/cache clear，block/class/pass稳态内为0。默认resident请求所有新框架
counter为0。旧/新dylib的A1–B1–B2–A2 warm smoke中，
当前/旧版wall与denoise median ratio为`0.99748/0.99607`，没有观察到默认回退，但样本规模不足以升级为正式P0。
最终release在audit计数修正后又做一组A→B warm复核，wall/denoise ratio为`0.99665/0.99963`；release/audit
SHA-256分别为`d9acb31adaef4d389fa9211ea364078ebcd35119758c7527606f551cb8f4abb9`和
`901eb689782a368931bf618efa682249e69ad8c25bf66f71a4f9d0e77524a250`。

下一步不是扩H3或把Flux resident当P1 baseline。private same-layout direct replay现已建立，并保持
P0/G1/K2/D1/Q2、两pool retention、pread和每block同步完全一致。正式10-block/20-pair P1已经`PASS`：
wall median/P95和denoise median ratio为`0.99971/1.00325/0.99816`，40/40请求成功、20/20 PNG pair一致、
steady allocation/thread-create为0/0。当前状态为execution/quality/memory/audit/P1完成；下一阶段进入
whole-request closure、budget guard和真实低内存/swap campaign。public gate与production registry继续关闭。

## 20. 2026-09-17 续接：H3范围冻结与Flux同布局基线

H3工作项明确只维护MiniMax H3 Turbo当前原始BF16冻结tuple。合理I/O波动按第18节的工程口径接受，同时保留
strict verifier=`INCONCLUSIVE`的统计事实；不扩普通H3、其他checkpoint、量化缓存或新布局。

Flux新增benchmark-only direct executor和冻结policy
`docs/design/streaming/examples/flux9-p1-same-layout-policy.json`。campaign工具链新增
`streaming_implementation` raw字段与`expected_implementations`硬门，逐请求保证baseline走
`flux_direct_same_layout_v1`、candidate走`generic_stage_executor_v1`。两步smoke的PNG SHA均为
`5b39235aab9acbbccbce6b1fffdc672cbf5f138c2f935db1fdaf3fb2efa1d108`，layout digest、峰值、加载字节和
pool/slot/fill计数一致。audit、1-block、2-block及正式10-block均已完成；正式bundle保存在
`results/streaming/flux/2026-09-17/p1-same-layout/`并由独立verifier重跑为`PASS`。后续不得跳过
implementation身份校验，也不得以resident对照代替新的shape或backend P1。

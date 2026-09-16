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
4. executor 仍是 single-class streamed；resident、多 class、跨 component DAG、session retention
   和服务级 quarantine/eviction 还未闭环。
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
9fe0412da08df4f82b352b6c86c6eee3001a1a4b61133079b76ee9ceec43e138
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

该检查点支持保存和继续开发，但不授予production资格：生命周期故障矩阵、normal-target P0/P1、
whole-request bounded guard、低内存P3和其他模型adapter仍是下一阶段门禁。

# 当前代码状态与下一阶段计划

修订日期：2026-09-19。本文是当前工作树的交接摘要，不替代 [13 实施进度](13-implementation-progress.md)、
[37 校准与验收](37-public-streaming-calibration-performance-acceptance.md) 和 [52 App/档位规格](52-public-streaming-app-config-and-model-tier-spec.md)。

本轮只做代码整理和静态收口，不运行新的真实 Metal、P2/P3、swap 或性能实验。

## 结论

TurboCider 已经具备统一 block/slot/streaming 的控制面、布局编译器、SourceLease、StageExecutor、
actual receipt、App memory-tier selector 和四个模型 adapter 的主要代码骨架。当前代码仍处于
**public-semantics calibration / internal preview**，不能宣称 production public：

```text
production catalog = tc-streaming-catalog-empty-v1
records            = 0
```

因此 App 中的 Off/8/10/12/16/20 GiB 选择器可以显示和查询状态，但没有经 review 的 record 时必须继续
显示 unavailable，并回退到 Off/legacy 默认路径。

## 本阶段整理内容

本阶段提交包含以下已经完成的 runtime 收口：

- LTX public exact 的 multi-stage `RunResult` 在 exec 前由 common verifier 校验；
- verified public envelope 通过继承 fd 传给 Video VAE replacement finalizer；
- finalizer 检查固定 format、schema、stage/boundary/receipt 的容器类型及真正的 JSON boolean
  `actual_plan_verified=true`；完整 receipt 语义由 parent common verifier 校验，child 不重新验证 catalog；
- Video VAE checkpoint 和 envelope 都使用 request-scoped fd authority，path 不重新成为 public authority；
- parent 的 checkpoint/envelope fd 使用现有 `OwnedSourceFd`，覆盖 exec 前 staging、分配和 execve
  异常；本轮不把这项局部异常安全修正扩展为所有取消/错误路径均已通过故障注入；
- campaign probe 支持 `${REQUEST}`，让 disposable process 的请求身份进入采样运行；
- Makefile 和 contract test 覆盖 finalizer envelope；
- 进度文档记录了 P8/K2、P1/K1 的真实证据和准确的 hard-cap 结论。

默认 Off/legacy 路径没有增加 public resolver、SourceLease、StageExecutor 或 sampler 工作。

## 四模型当前状态

| 模型 | 当前代码状态 | 已有证据 | 主要缺口 |
|---|---|---|---|
| Z-Image Turbo | public adapter、K1/K2、public constructor 已接线 | 10 GiB clean public-semantics P2：40/40，byte-exact，peak P95 约 8.97 GB，swap 0 | clean release P0/P1/P3、review、production record、其余档位 |
| Flux.2 Klein 9B | dual/single multi-pool、pager、SourceLease、actual receipt | 同布局 P1：wall median 0.99971、denoise 0.99816；16 GiB memory qualification 曾 PASS | environment evidence 完整化、P3、production review、8/10/12/20 GiB |
| MiniMax H3 Turbo | 50 block、K2/G1、carry、4-pass、C/Metal receipt | 工程 P1 点估计约 +1.07% wall、+1.19% denoise | 当前 checkpoint 不是可信 H3 Turbo merged artifact；缺 merge manifest 和 full public P2/P3 |
| LTX 2.5 | Stage 1/2、SourceLease、fd VAE、exec replacement、verified envelope | public exact MP4 成功；P8/K2 tree peak 约 23.978 GiB，P1/K1 约 18.247 GiB | P0/K1 zero-prefix；20 GiB hard-cap；P2/P3；production record |

## LTX 当前最重要的技术边界

LTX P0 不能只修改 validator 或 JSON。当前 native schedule 仍有显式的 prefix 假设：

- `StreamingPlanView` 要求 `stage.prefix > 0`；
- C adapter 用 `!o->resident_prefix_blocks` 拒绝 zero prefix；
- `ltx_exact_prefix()` 之外，conditioning 和 denoise workspace 仍从 `weights[0]` 读取维度；
- P0 时 `weights[0]` 不再是常驻 block，而是 slot 0 的动态 view；
- ANE attach、first-frame conditioning 和 legacy streamed path 不能因为 exact P0 改动而改变语义。

因此下一阶段应先消除几何信息对常驻权重的依赖。优先考虑读取已经构造的 exact slot view 的稳定几何字段，
或抽出 request-owned `BlockGeometry`；不要求为此重构默认 resident/ANE 路径。conditioning、workspace、rope
只能读取已初始化且生命周期覆盖 stage 的 metadata geometry，不能读取尚未加载的 slot 内容；然后才允许：

```text
resident_prefix_blocks = 0
slot_count              = 1
prefetch_distance       = 0
io_workers              = 1
group_count             = 48
group[i].block          = i
```

P0 实现还要覆盖 Stage 1/Stage 2 的 receipt、source revalidation、取消/quarantine、output parity 和默认
resident 回归。若真实 process-tree peak 仍超过 18 GiB，不能放宽 buffer；应转向 Stage 1/Stage 2 独立
disposable-process pipeline。

具体代码落点与验收：

| 代码位置 | 修改建议 | 必须验证 |
|---|---|---|
| `ltx_streaming_plan.cpp::initialize` | 接受 0≤P<48，保留 G1、slot/group 和 pass 限制 | 48 groups 从 block 0 起，Stage 1/2 offset 为 0/8 |
| `ltx_streaming_adapter.inc::ltx_exact_validate_options` | 允许 zero prefix，同时严格检查 group 覆盖范围 | malformed group、capacity、slot、overflow 仍拒绝 |
| `ltx_blocks.c::apply_scalar_conditioning` | 将维度验证与 resident table 更新数量分离 | P0 时仍 eval conditioning，仅 resident table 更新循环为空 |
| `ltx_blocks.c::ltx_native_run` | rope/workspace 使用 exact 几何来源 | 不访问空 `ctx->weights[0]`；legacy/ANE 保持原行为 |
| `ltx_blocks.c::run_denoise_schedule` | 仅 exact 路径允许 P0；legacy 限制不放宽 | 全部 48 blocks 由 exact executor 执行；T2V 范围不变 |
| snapshot/descriptor/model tests | 增加 P0/K1、P0/K2、split-stage、首次 fill 取消 | ABI/源文件替换/释放/逐阶段 latent parity；功能对照不能误标同布局 P1 |

本轮没有修改上述运行逻辑，也没有创建“已支持 P0”的配置文件。

## App 自动推荐仍需补齐

`StudioState.recommendedStreamingSelection` 已读取物理内存并过滤 unavailable target，
`refreshStreamingOptions` 只在用户尚未手动选择时应用推荐。但是当前规则对 ≥20 GiB 统一推荐上限为
20 GiB，**没有模型/工作负载的 resident-fit 判定**。catalog 为空使它现在回到 Off；catalog 非空之后，
大内存机器也可能被自动切到 streaming。这不满足“内存充足保持默认”的完整需求。

下一阶段需在推荐控制面增加经验证的 resident-fit 信息（按模型、shape、组件、后端和设备），判定顺序为：

1. 用户显式选择优先，不能用推荐覆盖 Off 或指定档位；
2. 常驻路径有足够物理内存余量：推荐 Off；
3. 常驻不足：从匹配且 available 的 records 中推荐已验证的方案；
4. 未知或无已验证档位：保持 Off 并明确提示，不猜测可运行性。

至少增加大内存+available record仍 Off、低内存推荐、显式 Off 保持、catalog 空/撤回、切换模型/shape 后
刷新、乱序查询不覆盖新结果和 App→worker 完整 result 消费测试。目前 CLI exec smoke 不能证明 App UI
端到端已通过，也不能把物理内存档位当成实时可用内存或操作系统强制 hard cap。

## 下一阶段顺序

1. 保存当前代码检查点。本轮只构建、检查边界和文档，不生成新性能结论。恢复实验后先冻结 release commit
   和对照构建；合同一致不等于性能无回退。
2. 修复 App resident-fit 推荐和 worker 结果消费边界；同时实现 LTX 几何信息脱钩及 zero-prefix
   host/ABI/descriptor tests。这两项是有限代码任务，不再增加新的通用框架抽象层。
3. 将 P0/K1 作为内部/test-catalog layout，先验证 metadata、cancel、receipt 和 lifecycle；
   经用户允许恢复实验后再测完整请求。P0 layout 中的 P 是 prefix，不是性能验收阶段 P0。
4. 完成 Z-Image 10 GiB、Flux 16 GiB 的 clean evidence、P3、独立 review 和 production catalog builder 输入。
5. 获取可信 H3 Turbo merged artifact，再做 H3 memory-tier calibration；普通 H3、FastH3、量化 H3 不复用记录。
6. 完成 LTX 20 GiB 后，再向 8/10/12/16 GiB 搜索；每个 `model × workload × device × target` 单独生成 immutable record。
7. GPU-only public record 稳定后，单独认证 ANE/hybrid：Core ML/ANE artifact、partition、peak、quality、P0-P3 和设备档位不得复用 GPU-only record。
8. 最后才把 production catalog 从 empty revision 替换为 reviewed revision，并在 App 中开放对应 available 档位。

## 发布判定

当前不能 public 的原因不是 App 入口缺失，而是证据和 authority 仍未闭合：

- production catalog 为空；
- Z-Image 只有一个接近发布的 10 GiB candidate；
- Flux 16 GiB 仍为 `INCONCLUSIVE`，环境证据不完整；
- H3 缺可信 Turbo artifact；
- LTX 20 GiB 尚超 hard-cap；
- 四模型 public streaming + ANE 均未独立认证；
- 本阶段整理提交是开发检查点，不是 release record；发布仍需独立确认的 clean build/evidence。

只有当 record 通过 source/runtime identity、actual receipt、quality、process-tree headroom、no-swap/P3、
默认路径回归、独立 review 和 catalog builder 后，App 才能把对应 target 标记为 available。

不必等待四模型和所有档位同时通过才首次发布：可以先只开放通过完整验收的 Z-Image 10 GiB，随后
Flux 16 GiB；其他模型/shape/设备/档位保持不可用。ANE 后续独立推进，但仍属于总体需求，不能当作已完成。

## 性能证据应如何解读

- Z-Image clean 10 GiB P2 的 wall median 比值为 1.03234、denoise 为 1.01145：这是固定低内存策略
  的诊断比值，不是同布局框架开销。其既有同布局 P1 wall 比值约 0.99986。
- Flux 9B 的同布局 P1 wall 比值约 0.99971。16 GiB campaign 有 40/40 成功、20 pairs、进程树 peak P95
  12,017,559,951.6 bytes（约 11.19 GiB），但 environment 为 partial，总状态仍是 INCONCLUSIVE，不能只取
  memory qualification PASS 发布；证据来自 `tc-flux9-16g-p2-2a21cf3` 本机 bundle，其内部源码 identity
  实际为 `971bdc8`，不可按目录名推断 commit。
- H3 早期工程 P1 的合理波动不证明当前本机 artifact 已获得 Turbo provenance；正式状态仍保持
  INCONCLUSIVE，不把工程接受改写成统计 PASS。
- LTX 18.247 GiB 是单请求进程树诊断，不是正式 20-pair P2。与 18 GiB 的差距真实存在，单纯观测 swap=0
  也不证明在 20 GiB 物理设备上安全。
- 既有 M5 ANE/hybrid 优化保留。Flux 4B resident 历史耗时下降约 14.6%；Z-Image 特定 512² streaming
  优化下降约 33%，但 1024² hybrid 仍可慢于 GPU。不能推广为四模型通用加速或 public streaming+ANE PASS。
- 本轮没有重测性能。默认热路径无新 scheduler 工作是代码设计事实，“最终 release 全负载不回退”仍待
  对应范围的测量。自然 swap 对照仍缺，不能承诺 streaming 一定比 swap 快。

## 分支与交接

整理起点为 `feat/stream@3e133d3`，本地 `dev@02148b7` 已是当前分支祖先，整理前 `dev...HEAD` 为 0/53。
本轮不 fetch/merge 远端，不宣称已包含远端最新提交；最终合并验收需再次获取并检查当时的 dev。
13 中的实验数字是历史记录；本轮新增的类型校验/fd 异常安全整理没有新增真实推理证据。

本轮实际检查（不是性能实验）：

| 命令 | 结果与范围 |
|---|---|
| `TURBOCIDER_NATIVE_ONLY=1 tools/native/build.sh` | PASS，编译/link release native，不运行模型；未重新构建 Swift App |
| `python3 -B tests/native/test_ltx_finalizer_envelope.py` | PASS，使用本轮重建 binary；合法包前进到缺失 latent，非法类型/缺字段在此之前拒绝；不 decode VAE、不使用 GPU |
| `python3 -B tests/native/test_streaming_campaign_verifier.py` | 33 PASS，synthetic/mock 协议验证；不是模型 campaign |
| `python3 -B tests/native/test_contract.py` | 83 tests，82 PASS，1 缺 Wan fixture 的预期 SKIP；不是完整推理回归 |
| `git diff --check` | PASS |

未在本轮执行：真实模型/Metal 推理、P0/P1/P2/P3 campaign、swap/pressure、ANE benchmark、Swift App
端到端测试。用户恢复实验前保持当前发布 gate，不据此生成新 production record。

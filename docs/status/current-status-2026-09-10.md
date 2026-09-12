# TurboCider 增量状态：H3/LTX 共享驻留策略

更新时间：2026-09-10

H3 与 LTX 的 block working-set 决策已收敛到 `native/runtime/block_residency.c`，并由 C++ `BlockResidencyPlan` 在两个 Session 返回结果前复算校验。H3 保持固定双槽和 budget-driven pinned prefix；LTX 使用自适应 1/2/3 个 refill slot，并在预算足够时自动全驻留。Metal kernel、算子精度和既有 ANE 门禁均未改变。

主机验证（Apple M4 Max，macOS 26.6.2）：

- H3 matched 256×256×22、四次 DiT evaluation、两轮 ABBA、无预算：TurboCider first/retained denoise 中位数为 17.235825/16.833305 s，vpipe 为 20.788215/20.721903 s，TurboCider denoise 约快 1.206×/1.231×。跨 runtime parity 仍失败（relative L2 0.0973546、cosine 0.9952883），因此 vpipe 不能替换现有 runtime。
- LTX 704×448×9、11 steps、video-only、12 GiB denoiser budget：选择 18 pinned、30 streamed、3 refill slots，估算 working set 12,812,451,712 bytes。相对 resident 的 warm request/denoise speedup 为 1.071987×/1.080471×；stage-2 latent 和 9 帧 RGB 均 byte-exact。进程 RSS 仍可能高于 denoiser budget，预算不是硬性 RSS 上限。

代码级策略记录见 [共享驻留策略验证](../design/validation/h3-ltx-shared-residency-2026-09-09.json)，主机复测见 [2026-09-10 H3/LTX 验证](../design/validation/h3-ltx-shared-residency-2026-09-10.json)。LTX 仅 video-only smoke 通过，不开放 I2V、音频或 GPU+ANE；H3 parity、LTX 97 帧全链路矩阵及 LTX ANE 仍保持 fail-closed。

## LoRA 身份与离线缓存

H3/LTX 的进程内权重缓存现在绑定经验证的 LoRA canonical path、字节数、SHA-256、role、strength 精确位模式和算法版本。H3 另外绑定所选 `FL2VA`/`Ref2VA` component 与 provenance manifest SHA-256；LTX 绑定所选 merged checkpoint SHA-256，避免 manifest、adapter 或强度变化后错误复用旧的 prepared DiT。

LTX 离线 merger 已真正把显式 `--strength` 乘到 LoRA delta，并将其写入 resume state、cache identity 和 provenance manifest；有效范围为有限值 `(0, 4]`。runtime-cache 使用不暗示官方 0.8 强度的中性 artifact 文件名，native preflight 已验证该文件名可由 manifest 的 `output.filename` 安全解析，且 cache entry 内的 base/adapter 同目录链接满足现有 provenance 校验。

rank-450 inference-time LTX LoRA 未启用。代表性 Metal 测量仅计算三个 video projection，stage 1/2 已分别约增加 8.1/20.7 ms 每 block，推算 48-block、8+3 schedule 额外约 6 秒，尚未计入其余 adapter projection，因此不满足“新路径不得慢于现有路径”的门槛。H3/LTX 继续只支持 verified `disk_premerge`；ANE/LoRA 仍 fail closed。详细记录见 [LoRA identity/runtime-cache 验证](../design/validation/lora-identity-runtime-cache-2026-09-10.json)。

## LTX ANE 连续窗口验证

LTX ANE profile 现在可以显式限定连续 Transformer block window，并用 stage mask 独立选择 Stage 1、Stage 2 或两者。只有完整 48 blocks、双阶段覆盖时才允许释放全部 GPU MLP；partial window 始终保留未覆盖层的 GPU 权重。解析器只要求被 stage mask 启用的 artifact 目录，禁用阶段不再需要伪造占位路径。所有近似路径仍要求请求显式设置 `allow_approximation=true`。

704×448×97、11 steps、tail-12（blocks 36–47）、Stage-2-only 的修正后复测可以稳定运行，但尚未过门禁：两次 request 为 106.879844/107.777527 s，对照 GPU 为 108.504091/116.468282 s；warm request 约 `1.081×`，未达到 `1.1×`。Stage-2 latent relative L2 为 `0.04099997`、cosine 为 `0.99915925`；RGB mean correlation 为 `0.99180680`、mean cosine 为 `0.99876032`、mean MAE 为 `5.05467/255`，略高于 `≤5/255` 门槛。因此该窗口和完整 ANE 路径都不自动启用，后续仍需交替进程 ABBA、多 prompt/seed 和同时满足质量/速度的 block 选择。

## 代码整理与独立性

离线 vpipe 对比只保留脱敏的验证报告和设计结论；临时 probe 源码、外部 dylib 构建脚本、H3 运行时 dump 环境开关已从工作树移除，TurboCider 的构建与运行不发现或链接 `references/vpipe`、`ltx-mac` 或其他兄弟仓库。GPU coordination lock 和本地 API socket 改用系统临时目录 API，不再硬编码 `/private/tmp` 或开发机路径。

仓库测试新增 shipping-path 独立性门禁，覆盖 `apps/`、`native/`、`services/`、`bindings/` 和 native build/package 脚本，拒绝 `/Users/`、`/home/`、`/private/tmp`、`../references` 与 `references/vpipe`。2026-09-10 已完成 native/Swift 重建；聚焦回归为 `87 passed, 1 skipped, 45 subtests passed`，完整 repository test 集为 `13 passed`，`git diff --check` 无错误。

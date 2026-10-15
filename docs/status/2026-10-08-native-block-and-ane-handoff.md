# 原生 Qwen block 与 private ANE W0 桥：组件进展

日期：2026-10-08，设备 Apple M4 Pro。对应 [独立后端路线 N](../design/qwen-image-gpu-ane-command-graph/10-native-metal-graph-backend.md) 和 [当前接线合同](../design/qwen-image-gpu-ane-command-graph/11-native-block-and-handoff-integration.md)。机器可读数据、40 个源码摘要、日志/二进制 SHA 和清理回执见 [JSON](2026-10-08-native-block-and-ane-handoff.json)。**这是组件正确性/构建验收，没有整图或并行加速结论。**

## 实现范围

新增原生 tiled attention、Qwen norm/modulation/residual、QK norm/RoPE、head layout/copy 和完整 base block。所有 block 运算记录到 TurboCider 自己的 Metal graph，未调用 MLX GEMM/SDPA；prepare 与 GPU workspace 分配在 encode 外。私有 draft 验证成功后才整体追加，晚期非法 FFN 绑定不会留下半个 block。

Linear 补齐 K/N=64、64 对齐列尾和 checkpoint BF16 bias，保留 matmul 输出 cast 后再加 bias。LoRA 尚未接入；本地 Qwen FFN 的 JSON 头确认无 bias，其他来源不能静默省略未支持项。

复核修正两个精度问题：LayerNorm 使用中心方差；Q/K RMSNorm 先归一化→BF16，再乘权重→BF16，然后 FP32 pair RoPE。使用显式 precise tanh，recipe 为 `qwen-ln-centered-fp32-bf16-mod-bf16-rms128-preweight-bf16-pairrope-fp32-v2`。旧 experimental kernel 未在本轮被替换，也不据此授予旧路径相同数值资格。

private ANE 新增 retained surface buffer lease、显式 ready/done 与 provenance、NativeAneHandoff 和 move-only 读取许可。W0 先确认自己提交的 GPU producer 成功且物理完成，再提交 ANE；成功 callback、物理结束和身份匹配后才允许 GPU 读取。新路径没有自动 force-signal 或预排输出 restore；旧 overload 兼容行为独立保留。

## 小规模设备验证

最终源码对应 `outputs/qwen-real-ffn-20261007/native-qwen-block-component.v2.log`。合成数据，无 checkpoint 或 App 加载：

| 检查 | 几何 | 结果 |
|---|---|---|
| Linear | M33，K64/N64，BF16，无 bias | 2112 个输出与 CPU 参考一致 |
| Linear 列尾/bias | M33，K64/N192，BF16 | 6336 个输出与 CPU 参考一致 |
| Linear FP32/bias | M33，K192/N64 | 2112 个输出与 CPU 参考一致 |
| 首步完整 block | M37，H128，F256，prefix5，两个 causal text 段+target | 18 dispatch；非零 query offset、prefix 导出及所测中间值/输出通过 |
| 稳态完整 block | M33，H128，F256，prefix5 | 18 dispatch；K/V concat 及所测中间值/输出通过 |
| 多 head block | M3，H256，2 heads，F256，无 prefix | 14 dispatch；head pack/attention/unpack 通过 |

三个 block 共 9728 个最终输出；所测中间值的观察误差也为 0。readonly 快照、matrix/head padding、canary、晚期绑定和跨步骤 alias 拒绝通过。**这不是真实 H4096 block、所有 prompt、LoRA 或完整模型等价证明。** 前一版合成回放也通过；最终版本在 precise tanh 和完整 readonly/padding 检查补齐后重跑一次。

另执行一次成功的 **256 元素 GPU→private ANE→GPU**：GPU 写 FP16 IOSurface，private 程序执行 add，成功许可后由原生 GPU 消费。输出全部有限、误差为 0；独立事件、请求 provenance、重复许可拒绝通过。导入增加的 charged native allocation 为 **0**，仅测试 FP32 输出新分配一次。

首轮因合法 SharedEvent 的 `device=nil` 被预检误拒绝，GPU producer 已完成但 ANE request 尚未提交。依据 SDK `MTLEvent.h` 修正为 creator-device 必须匹配，reported-device 可为 nil、非 nil 时必须匹配；eventPort/selector 与 W0 身份约束保持。修复后重试成功，错误与最终日志均保留。

小程序 fresh owned source cache 的准备观察值约 **35 ms**；driver/系统编译缓存状态未控制，不能作为模型首次启动或预加载数据。该 FP16 链路不证明 full FFN、W8A8 硬件算术、利用率或 GPU/ANE overlap，没有并行任务或性能对照。

## CPU 合同与构建

Linear 62 项 CPU 检查、attention 59 项、QwenOps 中心方差及 RMS 舍入 fixture、block geometry/phase/精确 workspace、surface/event/provenance/nullable-device admission、sticky ticket 退役均通过。主机 wrapper 严格语法编译通过，设备运行仍要求显式开关。

标准 private-enabled native 库和 CLI 构建退出码 0，runtime identity/catalog 验证通过。库实际包含 `NativeBlock::encode`、`NativeAttention::encode`、`NativeQwenOps::encode_qk_norm_rope`、`NativeAneHandoff::finish` 和 external-event `Program::prepare`。build ID：

```text
tc-runtime-build-v1-d0c288593d9e380080fb68679faf3f9e4961843baa3e7732325d444ba00bb582
```

新 private handoff 源只在 `TURBOCIDER_ENABLE_PRIVATE_ANE=1` 时加入构建；公共分发保持原开关边界。本轮未再编译/启动 Swift App，没有替换 dist App 或启用默认模型路由。构建仍有原先 CommonCrypto/legacy fastMathEnabled 的 SDK deprecation；小桥 builder 对复用旧文件的该警告类别单独关闭，其他警告作为错误。

## 缓存与下一步

没有下载模型、复制 checkpoint、完整生图或新增 E2E 组。清理本轮 227 个 `.o`、静态中间库、module-cache、CPU test 临时目录、旧的小组件二进制目录及两份 private 小程序 owned cache，共约 **19.9 MB 文件**。保留约 **11.9 MB** native 产物/身份清单、约 **0.91 MB** 组件二进制和小型回执。原模型、dist App、旧对照 runtime、capture 和不归属本轮的缓存保留；系统/driver 内部缓存不在清理回执内。

重现入口：

```sh
bash tools/native/build_native_qwen_block_component.sh
build/native-qwen-block-component/native-qwen-block-component --tiny-device

bash tools/native/build_native_qwen_block_component.sh build/native-qwen-block-cpu --cpu-contract
build/native-qwen-block-cpu/native-qwen-block-contract

bash tools/native/build_native_ane_handoff_component.sh
build/native-ane-handoff-component/native-ane-handoff-component \
  --tiny-device <fresh-owned-cache-directory>
```

设备运行要求实际 Metal/private ANE 可用；cache 必须归属该测试。构建使用 process-local `DEVELOPER_DIR=/Library/Developer/CommandLineTools`，没有改全局工具选择。

接下来按 [11](../design/qwen-image-gpu-ane-command-graph/11-native-block-and-handoff-integration.md) 共享 operator/workspace，补齐输入/时间/调制、32 层、prefix 事务、最终 projection/采样与 source/memory/consumer 生命周期，再原生接入 A8/staging/通道 FFN/restore/join 做有限 W1。先验证少量真实 block 输入及完整路径；现有真实 FFN capture 不能冒充完整 block 输入。证明收益后才使用预算内的配对 E2E，base 成立后独立引入 LoRA、编辑、Z-image 和 encoder。

# Native GPU LoRA 组件接入与独立后端设计更新

日期：2026-10-08；`dev-verify`；Apple M4 Pro / 48 GiB。本轮新库 `61fe967c…` 通过完整 native/private-enabled 构建，567 个构建源摘要与当前源码一致。只执行一组生成小张量验证，没有下载或加载真实模型，没有 ANE 调用、图像 E2E 或性能循环。历史 `4fd1e980…` Base 收益、`16fecb41…` 尺度缓存结果不转授本轮源码。

## 改动

- 新增 NativeLoRA，实际 BF16／FP32 A/B、FP32 low／累加、有序 BF16 舍入、bias 最后相加、独立 SiLU×up。不导出 merged checkpoint，不预先扩大整套 factor。
- 原生模型覆盖 8 个全局投影和每层 7 个 attention／FFN 逻辑目标；gate/up 在非线性前修正，down 消费修正后的 hidden。无 adapter 保留原融合路径，不创建 LoRA PSO／scratch；hybrid LoRA 仍拒绝。
- Weights 共用解析器与 held-fd loader；snapshot 保留真实 factor owner、verified SourceLease、顺序、原始 strength 与有效 scale。独立 adapted importer 严格消费 target／range、拒绝不等价 metadata 与未知变体。
- pipeline 强制重载后重新绑定、将 native materialize 移至全部因子绑定之后；NativeSession 接入内容／generation／strength／term／recipe 身份、来源留存、实际 rank 与 FFN scratch，同一内存账本。六步学生与普通 Base schedule 独立校验。
- [独立后端规格](../design/qwen-image-gpu-ane-command-graph/16-splash-native-backend-spec.md)补齐 `ResolvedPlan → BackendGraph → Invocation`、GPU／ANE provider、窗口调度、完整 DiT 事务、资源移交和失败完成域；通用模板及 W2 仍是设计目标。[LoRA 方案](../design/qwen-image-gpu-ane-command-graph/17-native-lora-integration.md)同步实际源码状态。

## 验证证据

| 验证 | 结果与边界 |
|---|---|
| 严格 host／CPU | kernel 合同／source／math 为 106／78／637 项；模型合同 12 接受／32 拒绝；public/private host、metadata、内存预算／owner 回收通过 |
| 请求入口 | 新库 9 项 route 测试通过；入口仍为 BaseOnly 512²／20 步，不开放 Native LoRA CLI／App 请求 |
| CPU loader fixture | 共用解析、fused range、原始 strength、同路径替换、snapshot owner、失败清空通过；MLX 统一分配器需要 Metal 可见，未提交 native GPU／ANE 图 |
| LoRA GPU 算子 | 3 次提交、14 dispatch；BF16 r16/r3 有序负 adapter／fused 范围／bias-last、FP32 r128／in-place／tails、BF16 SiLU×up 共 534 元素均零 BF16 位差 |
| 完整 tiny DiT | H128／F256／2 层、3 文本＋4 目标行、每路 2 步；全部 22 个逻辑投影混合 r16/r128，实际 MLX Transformer＋Euler 对照；三路 RRMS=0、max_abs=0 |
| 重绑与生命周期 | Base→LoRA→Base 恢复逐位一致，LoRA 确实改变输出；adapter／source／padding 不变、3 phase 物理完成且账本归零；native LoRA scratch=10,752 B，两次 Base=0 |

完整 tiny 使用一次确定性输入、同时开启全部 targets；这不是每个目标独立消融，更不是 32 层真实模型的正确性或加速资格。监督耗时仅记录测试预算，不作性能排名。原始记录：[构建](../../outputs/native-lora-gpu-20261008/full-build.log)、[算子](../../outputs/native-lora-gpu-20261008/operator-device.log)、[完整 tiny](../../outputs/native-lora-gpu-20261008/denoiser-device.log)、[监督回执](../../outputs/native-lora-gpu-20261008/generated-group-receipt.json)、[来源／产物绑定](../../outputs/native-lora-gpu-20261008/verification.json)。机器可读汇总见[JSON](2026-10-08-native-lora-gpu-components.json)。

## 清理与下一步

已在进程成功退出、物理完成及零账本确认后清理 243 个本轮自有文件，合计 21,828,889 B（约 20.82 MiB）：对象／静态归档／模块缓存及生成 adapter。新库、CLI、探针、manifest 和日志保留；没有触及模型目录或系统 Metal/driver cache。[清理清单](../../outputs/native-lora-gpu-20261008/owned-cleanup.json)逐文件记录摘要。

请求门继续关闭。下一阶段先处理窄域真实 LoRA 请求合同与一组本地 Viggle r128 六步对照，独立确认准备、去噪、完整请求及图像；之后再扩展 hybrid correction／hidden ABI。W1S 仍有逐层主机准入，W1R／W2 与其事件失败协议、编辑 encoder、Z-image、App／持久会话分别验收。本轮不授任何新的性能收益或产品默认资格。

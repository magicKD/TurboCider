# Core ML / ANE 启动开销与隐藏策略

更新时间：2026-09-08

## 结论

Core ML 的“已编译”“已加载”“已预热”是三个不同状态。`.mlmodelc` 存在不代表 `MLModel` 已创建；`MLModel` 已加载也不代表第一次 prediction 的 runtime setup 已完成。TurboCider 现在把这些阶段分别计时，并通过 `tc_engine_prepare` 允许 App/服务在用户可见生成前异步完成准备。

这能隐藏首请求延迟，但不能凭空减少总计算量。真正有效的优化是跨 block、step 和请求复用同一 shape/provenance 绑定的 session、共享 output backing，并把 preparation 放在非延迟关键路径；单独做一次空 warmup 只能减少随后第一次 prediction 的尾部开销。

## 当前生命周期

```text
source .mlpackage
      │  compile / content-addressed cache
      ▼
compiled .mlmodelc
      │  manifest + checkpoint/LoRA provenance
      ▼
MLModel load + interface setup + caller-owned backing
      │  optional zero-input warmup
      ▼
first runtime prediction
      ▼
subsequent predictions / session reuse
```

`prepare(load-only)` 不执行 denoise、VAE decode 或输出；`prepare(warmup)` 执行无输出完整请求，并可设置 `warmup_iterations=1..8` 触发每个分支的 zero-input prediction。所有状态都绑定模型、shape/bucket、精度、设备、checkpoint 和 LoRA identity。

## 实测证据

验证主机为 Apple M4 Max 64 GB。`mac_transformer` 对已经编译的 `512x1024x4096` FP16 artifact 测得：

| 阶段 | p50 |
|---|---:|
| fresh process wall | 92.378 ms |
| hot heterogeneous MLP | 1.353 ms |
| setup proxy | 91.025 ms |

因此 one-shot 小算子即使获得 1.5–2× hot speedup，也会被进程、manifest、模型对象和 wrapper setup 淹没。private ANE 虽能把 `2048x512` handoff 从 public backing-direct 的 0.424 ms 降到约 0.303 ms，但不能消除 session 生命周期和 sequential Transformer 的 Amdahl 限制。

Z-Image 256²真实拆分如下：

| 路径 | prepare wall | 随后 generate | 说明 |
|---|---:|---:|---|
| cold generate | — | 17.255 s | 首次构造/加载包含在请求内 |
| load-only + generate | 12.792 s | 11.201 s | 约 6.05 s 首请求延迟被移到 prepare |
| zero-input + full warmup + generate | 18.686 s | 10.895 s | 后续比 load-only 再快约 2.7% |

load-only 样本中 manifest/provenance 约 5.25 s、32 个 Core ML model load 合计约 6.88 s、output backing 约 4 ms；有序样本中 model load 也出现过约 0.24 s 的 OS/Core ML cache hit。因此这些数字用于生命周期拆分，不是稳定的首次编译中位数。

## 产品实现

- App 的“加载当前配置”调用 load-only prepare；用户仍可编辑 prompt，不会提前生成图片；
- “预热当前任务”调用完整无输出 warmup，结果不写入历史；
- Core ML source compile 独立由资源服务管理，命中内容寻址 cache 时不重复编译；
- output backing 由调用方持有，runtime 验证对象和 pointer identity，避免默认 CPU bridge；
- 自动模式只使用已测设备/shape/manifest，任何校验、内存或 prediction 失败都回退 GPU；
- 不全量常驻几十个大 program，LTX/H3 使用 bounded/stage-scoped residency；
- 取消在模型阶段边界传播，不把取消异常吞成 GPU fallback。

## 仍可优化的地方

1. 将 manifest hash、tensor index 和 LoRA identity 做安全的增量缓存，但不能跳过内容验证；
2. 对同一 bucket 的多个 block 研究 procedure-bank/多 function 管理，目标是减少 session rotation，而不是把所有 program 全驻留；
3. 在 App 预测用户下一次 shape/prompt 时后台 prepare，但要有内存预算和取消；
4. 继续复用 Metal-owned shared backing，避免 `MLMultiArray` 默认 host copy；
5. 记录 cold、disk-cache-hit、load-only、zero-warmup、first prediction、subsequent prediction 的多轮 ABBA，而不是用单次 wall 推导普遍加速。

详细结构化数据见 [coreml-startup-overhead-2026-09-08.json](validation/coreml-startup-overhead-2026-09-08.json)。

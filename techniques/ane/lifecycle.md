# Core ML 生命周期：准备、缓存、快照与失败回收

<!-- knowledge
id: technique-ane-lifecycle
type: method
stage: pipeline
operator: scheduling
backend: coreml
device: ane, cpu
symptom: compile-bound, stale-cache, lifetime
semantics: contract-only
boundary: weight-runtime, request
api: public
-->

[总目录](../README.md) · [后台调度](../gpu-ane/scheduling.md)

案例扩展：[ANEMLL stateful KV与阶段部署](public-coreml-deployment.md)、[Apple SD资源管理](stable-diffusion.md)、[训练中的动态权重/delta状态](training.md)。状态驻留、缓存命中与本页的artifact/session所有权仍需分别验收。

## 七个计时不能混成“ANE启动”

1. 源 `.mlpackage` 导出 / compile。
2. 内容寻址 compiled artifact 命中与验证。
3. checkpoint / adapter / shape / profile provenance。
4. `MLModel` load。
5. feature interface / caller-owned backing 创建。
6. zero-input warmup / first actual prediction。
7. 后续 prediction 与跨请求复用。

把 load-only 放在用户选模型时可减少点击后延迟，不能说总工作消失；完整 warmup 还运行了计算。fresh process 不等于 OS/driver/compiled cache 都冷；不擅自清系统缓存。

## 缓存 key 与失效

Artifact key 包含 graph ABI、source/recipe、shape/bucket、precision、转换工具/平台依赖；冻结图另绑定 checkpoint、分区和 adapter identity/strength。Session key 还要包含 execution device/profile 与有效 GPU/LoRA 策略。

参数相同的两个文件名不是同一来源；同一路径被替换也不是同一 generation。首次内容验证后可以用持有 FD、stat/generation guard 复用证明，但不能用“已缓存”跳过未验证新内容。

## 从验证到实际使用的间隙

Core ML 可延迟读 compiled 目录。如果逐文件校验后又按用户路径加载，可在两者之间被替换。当前 [ArtifactLease](../../native/backends/ane_artifact_lease.hpp) 和 backend 合同采用校验后私有不可变快照，保留到 worker/model 释放。

这里“私有快照”是进程独占目录，不是 Private ANE API。目录锁、owner/device/inode 与 no-follow/descriptor-relative 清理保护归属；崩溃残留只按明确身份回收，不能扫描临时根目录后猜测哪些可以删。也不承诺 Core ML 的系统缓存完全不落盘。

## 可复用的session/slot设计

将“可共享的不可变资产”“可复用的模型/session”“每请求可变state”和“每代IO slot”分开管理。一个slot至少经历prepared→in-flight→ready→consumer-complete→reusable；异步API返回或生产端完成，不自动跳到reusable。

迁移时记录实际返回backing、逻辑shape/物理stride和generation，并让owner覆盖最后读者。stateful路径还要分清zero reset与checkpoint restore；恢复时同步model外position/mask及optimizer状态。具体可学习[输出backing/代际实验](output-backing-and-stride-ownership.md)、[MLState提交点](stateful-multifunction-and-retry.md)和[Core AI取消/背压](coreai-profiling-reset-and-backpressure.md)。这些是不同边界的证据，不能拼成未经验证的零拷贝并行流水。

## 资源释放的正确顺序

```text
停止新提交 → drain producer/worker/consumer → 释放 graph/model
           → 释放借用的 weights/IO/scratch → 释放 source/snapshot lease
```

可选路线永久失败时，及时退役本实例 graph、worker、slots 和 scratch，避免回退 GPU 后还保留不可用 ANE 内存。取消、外部 callback 错误和永久 backend failure 不一定是同一状态；保留原异常和可复用语义。

任何一个 late chunk 失败，整体 FFN 用原输入完整 GPU 重算；不要保留已成功部分再拼接新路线。若 fallback 本身失败，也不能吞掉异常后返回半成品。

## Session 跨请求复用

[动态shape/hint实测](dynamic-shapes-hints-and-failure-boundaries.md)进一步区分固定三owner、单range owner、per-owner seen shape与persistent compiled-path缓存；enum首次预测SIGTRAP不回填成功时间。相同hint的late plan与instrumented framework interval仍不构成物理ANE、无observer或真实请求性能资格。

保留 graph/backing 可以摊销 load；保留全模型所有层可能压缩 VAE 工作集。依据 phase/预算选择 stage-scoped bank，而不是“能常驻就全常驻”。调度元数据可单独缓存，不能为了 EMA/history 保留数 GB surfaces。

适配器切换、shape/bucket 改变、source generation、精度、memory pressure、取消/失败都要重新判定可复用部分。`base→A→B→base` 是检查错误捕获和 stale scratch 的高价值序列。

## 实测提示与边界

[异构报告](../../docs/design/transformer-heterogeneous-report.md)里一个小 Core ML MLP fresh-process 约 92 ms，hot operator 约 1 ms，说明 one-shot 小任务难摊销 setup；该历史组件数字不代表所有图的固定启动时间。

[backend 维护说明](../../native/backends/README.md)给出真实失败回收/快照边界；[runtime artifact 记录](../../docs/status/runtime-ane-artifact-lease.md)保留旧新库对照。新项目应复制所有权与安全合同，不复制旧 timing 或默认选择。

## 实现参考与实验

可读实现：参考代码；[实验步骤与边界](../examples/README.md)；本项全部 case / controls。

本项控制：convert/compile/load/first/hot；one-shot vs repeated requests。CPU参考/模拟与实际设备入口分开标记；设备环境和更多变体见实验步骤。

可以重开探索的条件：load-only, new cache tier or phase-scoped sessions。历史负结果限定于其来源/shape/边界，不关闭该方向。

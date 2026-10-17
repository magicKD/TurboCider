# DeepCache：U-Net 的深层特征复用

<!-- knowledge
id: technique-algorithms-methods-deepcache
type: method
stage: dit
operator: sampling, attention
backend: mlx, metal
device: gpu
symptom: stale-cache, precision-loss
semantics: approximate
representation: bf16
boundary: request, producer-consumer
api: reference-only
-->

配套：[机制 witness 与证据合同实验](../../examples/README.md)。这是支持性小实验，不是本方法的完整模型复现。

[算法入口](../README.md) · [DiT block cache](block-caches.md) · [模型地图](model-sparsity.md)

来源：作者 `DeepCache/extension/deepcache.py` 与模型实现 [S24](../sources.md#s24)。分类：`approximate-train-free`；不是把 U-Net 输出整步照搬。

## 方法卡：保留浅层更新，复用较贵的深层特征

- **机制**：在多步 U-Net 的完整步保存深层分支特征，命中步只执行必要浅层与恢复路径，减少实际 body 计算。
- **何时考虑**：多步去噪、深层计算占比明显，且任务允许明确评测的近似；不是通用无损缓存。
- **代价/反例**：缓存占用、分支 IO 和状态管理增加成本；极短日程、不同拓扑或条件变化可能不适用。减少计算不保证质量过门槛。
- **实现参考**：[原 helper / U-Net 路径](../case-studies/deepcache-unet-boundaries.md)；[真实 MPS 生成](../case-studies/deepcache-sd15-mps-quality.md)提供速度与预设质量预算拒绝的具体案例，而非推荐阈值。
- **最小验证**：interval=1 对 stock、一次实际跳算、换请求清状态及 disable 恢复；再做完整生成的速度与独立质量评测。

## 分叉位置决定跳过范围

U-Net 有多尺度 down path、mid block 和 up path/skip connections。DeepCache 在完整步缓存深层特征，后续若干步只更新浅层分支，再与缓存深层信息组合。与 DiT 的线性 block 栈不同，它利用的是 U 形多尺度结构和相邻去噪步的高层语义冗余。

通用 helper 将 `cache_branch_id` 分解为 block/layer；`is_skip_step` 根据 interval、down/mid/up 位置决定哪些调用复用 `cached_output`。实现通过 wrap/unwrap 原 forward，disable 时恢复；迁移时需要同样处理异常恢复与同实例多请求，而不把 Python monkey-patch 当成安全的并发隔离。

## 在 Apple 上怎样实现

MLX/PyTorch MPS 可以保持 feature 在 GPU，缓存 down/up 所需 tuple 的完整结构。Core ML/ANE 路线可将 full U-Net 与 fast 分支导出为固定 shape 多图；graph 拆分会增加显式 feature IO、编译资产与峰值内存，所以不能仅从减少卷积层数推导收益。

SD1.5、SDXL 和 Stable Video Diffusion 的具体 block/attention/temporal 结构不同，需要对应 adapter。不能把 deep-cache branch-id 直接当作 FLUX/Qwen 的“第几层”。ControlNet 注入、inpaint mask 和多参考条件变化时，旧 deep feature 是否仍适用要独立验证。

## 适用与失败对照

多步 U-Net 且深层计算有明显占比时值得优先探索；极短日程、浅层主导或缓存挤占权重驻留时可能没有净收益。验证 interval=1、最浅/最深分叉、重试/换 prompt 清状态、skip tuple shape、torch/MLX eval 行为。

检查细节纹理、局部编辑边界、复杂文字；视频还要看时间闪烁。原仓库提供不同模型实现，不证明任意新 pipeline 仅 enable helper 就已正确支持。

论文：[DeepCache: Accelerating Diffusion Models for Free, arXiv:2312.00858](https://arxiv.org/abs/2312.00858)；原始代码与阅读范围见 [S24](../sources.md#s24)。

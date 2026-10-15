# VAE / 媒体阶段：激活峰值、全局统计和采样相位

[阶段地图](stage-map.md) · [本地相位 / Welford 样本](../labs/model_stages.py) · [halo / causal taps](../labs/operators.py)

## 小权重不等于小内存

VAE decode 将低分辨率 latent 扩成巨大 C/T/H/W activation；中间 activation、layout/cast、完整输出和帧队列可能远大于权重。Encode 又有输入预处理、下采样对齐及 latent scaling。先记录整个图的 stage、skip connections、norm 归约域、padding、时间因果性和输出 range/颜色/alpha，不能只统计 conv 权重。

## 保持语义的 tiling 有三个条件

**感受野**：沿普通 conv 链，receptive field 和输入 jump 可递推 `R'=R+(k-1)*d*J; J'=J*s`。输出 tile 必须读包含整个依赖域的 input halo；多层非线性不能通过把 kernel 系数简单相加替代。边界 padding 只在真实全图边界应用，内部 tile 边界不能补 reflect/zero 冒充真实邻居。

**全局统计**：GroupNorm 若跨 spatial/time 归约，独立 tile 均值/方差会改变算子，即使最后 feather 也不还原原语义。可先逐组统计全域 moments，再第二遍 normalize+conv，或设计共享统计的完整子图。全图中有多处 global norm 时，可能需要逐层 pass/scratch，不能声称一次全局统计解决所有层。

**采样相位**：stride/downsample/transpose-conv 的 tile origin 必须映射到全图 output coordinates。Transposed conv 中，一个输入会贡献给多个输出，独立输入 tile 结果直接 concat 会丢 overlap 或错位；output-coordinate gather 或带正确 crop/accumulation 的 overlap-add 才能完整合并。

本地 `transpose_conv1d`以全局输出位置反推合法输入 tap，最后 tile 可为 1 个输出；测试使用独立 scatter oracle，覆盖不同 stride/padding/kernel。`merge_moments`按并行 Welford 公式合并 `(count,mean,M2)`，用 `M2/count`得到 population variance，不平均局部 variance。它们是 1D/标量机制，不假装完整实现任何 VAE。

## 视频时间历史

Causal Conv3d 的 chunk 需要保留上一 chunk 的真实时间 taps 和 phase；首帧 replicate/fold 只在图定义的初始位置成立，后续不能每块重新当首帧。[causal-halo](../labs/operators.py)展示初始折叠和缺 halo 的反例；[media.py](../labs/media.py)演示 bounded frame queue、顺序和 checksum。

流式 encode/decode 要让 queue 有上界、writer 错误反向传播、取消 drain，在保留最后一个 reader 前不复用 frame。Rational FPS、音视频对齐、颜色/alpha、最终文件写完也属于完整请求成功边界。

## ANE / public Core ML 子图怎么选

稳定 shape 的 conv/activation 子图是候选，不是“所有 conv 都必然快”。切图前列出输入输出 layout、norm 统计如何进入、padding、halo、cast、shared backing；测 pack→prediction→restore→后续 consumer 的总边界，包含重叠 halo 的重复算量和多次 prediction 的固定税。仅 compile/load 成功不证明物理 ANE 或性能收益。

FP16/BF16 变化、局部 norm、独立 feather tile、temporal reset 或缩小 decode 都是单独的质量方案。对 seam、慢速运动、长视频首尾、极端像素幅值、alpha/range 比较，不只检查每张 tile finite。

```sh
python3 -B techniques/labs/run.py stage-vae
python3 -B techniques/labs/run.py global-stat-tiles
python3 -B techniques/labs/run.py causal-halo
python3 -B techniques/labs/run.py media-backpressure
```

迁移到真实二维/三维图时重建完整 receptive field、norm groups、skip/upsample phase 和 memory ledger；新的 stride、kernel、tile、temporal chunk、精度、Core ML graph 或媒体格式都要重新验收。更多原理见 [VAE 专章](../media/vae.md)。

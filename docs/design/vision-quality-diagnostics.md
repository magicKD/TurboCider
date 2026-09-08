# Public Vision 感知质量诊断

更新时间：2026-09-08

## 为什么增加这一层

扩散模型在不同后端、精度或求和顺序下可能产生不同纹理。逐位置 RGB correlation/MAE 对这种偏差很敏感，适合定位数值回归，却不能单独代表“主体、布局和整体语义是否仍然相近”。因此 TurboCider 增加了一个可选的 macOS Vision `VNFeaturePrintObservation` 诊断工具：它只使用系统公开 framework，不下载 CLIP、LPIPS 或其他外部模型。

工具构建方式：

```sh
make build-vision-quality
```

输出为被 `.gitignore` 忽略的 `build/native/vision-feature-distance`。它接收成对 PNG，并输出固定 Vision request revision、操作系统版本、每对距离、均值和最大值。距离越小越相似。

视频门禁可通过 `--vision-helper` 对有限数量的 RGB 帧做抽样；使用 `--allow-aligned-rgb-divergence --max-vision-distance <calibrated>` 才会把感知距离用于通过/失败判定。没有经过同一 OS、Vision revision、模型、尺寸和 prompt/seed 校准的阈值时，工具只记录诊断，不改变默认门禁。

## 当前判定规则

默认模式仍是 `aligned_rgb_regression`：

- shape、帧数、帧率和 finite 必须正确；
- 图片要求 correlation ≥ 0.99、cosine ≥ 0.995、MAE ≤ 5/255；
- 视频额外要求最低单帧 correlation ≥ 0.95、运动能量相对误差 ≤ 15%。

可选的 `calibrated_vision_perceptual` 模式只替换对齐 RGB 的最后一层判定，仍保留 shape/fps/finite/运动门禁，并且必须显式给出已校准的最大 Vision 距离。它不是运行时参考推理，也不会自动放宽 `gpu_ane` 的产品资格。

## LTX 现有样本

在 Apple M4 Max 64 GB、LTX 2.5 704×448、97 帧、24 fps 的现有 GPU 与 GPU+ANE 配对中，抽样帧的 Vision feature-print 距离均值为 `0.1862`、最大值为 `0.2079`（request revision 2）；相同解码帧控制值为 `0`，GPU 第 0 帧到第 24 帧的一秒时序控制为 `0.2815`。对应 GPU/GPU+ANE 帧在这个样本中比一秒后的 GPU 帧更接近，说明两条路径仍保留相同的大致主体/构图，但不能仅凭该诊断数字判定可发布：对齐 RGB 门禁仍为 mean correlation `0.8498`、mean MAE `21.59/255`，且尚无多 prompt、多 seed 的 Vision 阈值校准。因此 LTX GPU+ANE 继续保持显式实验候选，`auto` 继续使用 GPU。

完整原始数值见 [vision-feature-print-2026-09-08.json](validation/vision-feature-print-2026-09-08.json)；视频 RGB/运动门禁见 [video-quality-gate-2026-09-08.json](validation/video-quality-gate-2026-09-08.json)。

## 限制

Vision feature print 是操作系统和 request revision 相关的特征，不是模型质量的普适标尺。不同版本、裁剪策略、图片内容和压缩格式会改变距离；它不验证 prompt 遵循、手指/文字细节、音频同步或时序稳定性。正式产品资格仍需 paired tensor/media、性能、取消、内存和 provenance 矩阵共同通过。

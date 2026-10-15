# 辅助图像超分：tiles、alpha与模型预加载

[总目录](../README.md) · [模型配方](../models/recipes.md)

除SeedVR2之外，TurboCider还有Real-ESRGAN x2/x4的Swift/CoreML路径。它能提供一些通用的图像素材处理优化，但不应与完整SeedVR2时序VAE混用资格。

## 可复用技巧

- 固定输入模型用重叠tiles处理矩形图；边缘reflectpadding，重叠区域feather，核对真实scale而非文件名。
- ImageIO先归一orientation；alpha单独按输出几何resize/保留，不能因模型RGB输入而丢透明度。
- 单个resident CoreML model跨jobs复用；key包含resolvedsource身份和device选择。预加载可以包含一次zero predict，把startup前移，不跨App重启保持模型。
- preload与generation共享作业准入，避免预热和当前GPU/ANE任务抢内存。
- 输出staging/decoded尺寸校验/原子发布；失败或取消保留原图，不覆盖已有源文件。

## GPU与ANE优先的真实含义

GPU选择允许CoreML CPU/GPU，ANE优先允许CPU/ANE且排除GPU；不是“所有op已在ANE”。不同模型冷加载与warm图像处理可能排名不同。不能只测preloaded steady prediction就说第一次用户点击更快。

当前公开限制为单张静态图片、最大64Mi输出pixels，拒绝动画/视频；测试包固定输入522的FP16接口只是已测案例，不泛化为所有CoreML图。

## 最小验收

rectangular/edge/smalltile、alpha、rotation、x2/x4真实scale、tile接缝、model/device切换、preload/release、预测进行中取消与后续恢复、publication失败。宽高各放大4倍使pixels变16倍，预算应按实际输出和tile scratch而非输入文件大小。

权威入口：[公开超分合同](../../docs/public/IMAGE_UPSCALING.md)、[历史设备对照](../../docs/status/dev-verify-upscaling-2026-09-28.md)。这些是项目现有实现与经验，不是本轮新跑的性能测试。

独立 [overlap/halo/feather参考](../labs/upscaling.py) 用toy callback比较wholeframe与tiled，并将local-reflect crop作为明确近似对照。实际 [ImageUpscaler.swift](../../apps/macos/ImageUpscaler.swift) 还处理premultipliedRGBA：alpha>0才unpremultiply RGB，alpha单独resize后重组；tile padding/feather与learnedmodel receptivefield需实际quality验证，不能沿用toy的exact结论。

## 实现参考与实验

可读实现：[参考代码](../labs/upscaling.py)；[实验步骤与边界](../labs/media.md)；[本项全部 case / controls](../labs/catalog.json)。

```sh
python3 -B techniques/labs/run.py upscale-tiles
```

本项控制：whole-frame vs overlap/halo/feather callback；local-reflect approximation and alpha preservation。CPU参考/模拟与实际设备入口分开标记；设备环境和更多变体见实验步骤。

可以重开探索的条件：new spatial-only overlap/padding/feather or actual upscaler source。历史负结果限定于其来源/shape/边界，不关闭该方向。

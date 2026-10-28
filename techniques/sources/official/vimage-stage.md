# Accelerate/vImage：像素排列、方向与非原地缓冲合同

<!-- knowledge
id: source-official-vimage-stage
type: source
-->

[总目录](../../README.md) · [媒体交接](../../media/tensor-texture-interop.md) · [本机报告](../../media/case-studies/bgra-to-model-input-cost.md)

读取身份：macOS SDK26.5的`Conversion.h`、`Geometry.h`，2026-10-16。具体固定版本入口与阅读范围在[同一sources登记](../README.md)。这是公开SDK合同与本机API客户端实验，不是开放Accelerate底层实现的源码，也不分发Apple header、framework或网页正文。[Apple Accelerate入口](https://developer.apple.com/documentation/accelerate)供进一步查阅；本地实验依据实际SDK声明，不声称已核对活动网页的最新内容。

## 选读合同

- `vImageConvert_BGRA8888toRGB888`重排/移除通道，和带背景混合的`vImageFlatten_*`不是同一操作。不能把去alpha当成unpremultiply、合成或颜色管理。
- `vImageRotate90_ARGB8888`、`vImageHorizontalReflect_ARGB8888`和`vImageVerticalReflect_ARGB8888`也适用于BGRA/RGBA的四通道排列；geometry操作不能原地完成，输入、输出和中间scratch需要独立存活。
- Rotate90的clockwise常量是3、counter-clockwise是1，不能凭函数名猜数值；尺寸不匹配会裁剪/填背景，奇偶中心条件另有约束。本机只测试完整交换宽高的离散排列，不扩大到任意affine resampling。
- `vImage_Buffer.rowBytes`是实际字节pitch，不是逻辑width×channels。返回kvImageNoError仍需独立完整输出、方向和guard验证，不自动证明更快。

本机编译链接公开Accelerate并实际调用上述API；没有使用私有AMX指令，也没有证明内部SIMD/AMX具体指令、线程数或DRAM流量。SDK header内容hash是本轮读取身份，不为已保存run回填未采集的framework binary/实际加载image attestation。

## 与本地产品源配方的接缝

产品`native/media/rgb_video_reader.mm`请求32BGRA，按实际stride与display transform反查源像素后输出RGB；`video_input.mm`在MLX转FP32/255，`seedvr2/media.cpp`在identity resize后归一化并转模型布局。PE文件decoder直接链接当前`native/media/pe_image.mm`，与独立PNG sample decoder核对全部RGB字节；它不是通用带颜色管理/EXIF的RGBA importer。

这些文件在`dev-tech@4a9a81ac1639ab7b24aaddf6f342380cc675b7a1`有对应内容，独立curator复核既有648项source manifest与五个选定产品文件SHA；当前techniques工作树有未提交改动，不声称整个仓库clean。native产品代码未改，实际客户端没有链接或重建产品runtime库。

完整[阶段实验](../../media/case-studies/bgra-to-model-input-cost.md)既保留vImage相对循环的收益，也保留它输给更强GPU配方和矩形多pass回退的情况。不能由CPU子阶段很快推断完整媒体请求更快。

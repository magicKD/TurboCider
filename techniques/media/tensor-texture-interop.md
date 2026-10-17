# BGRA到GPU模型输入：公开CPU库、融合布局与完整交接成本

<!-- knowledge
id: technique-media-tensor-texture-interop
type: method
stage: io, pipeline
operator: layout
backend: accelerate, mlx, metal
device: cpu, gpu
symptom: wrong-layout, memory-bound, launch-bound
semantics: bitwise, numerically-equivalent
representation: packed, pixelbuffer, fp32
boundary: cpu-gpu, producer-consumer
api: public, reference-only
-->

[总目录](../README.md) · [媒体IO](video-and-audio.md) · [vImage来源](../sources/official/vimage-stage.md) · [全部实测](case-studies/bgra-to-model-input-cost.md)

本页先完成**已准备BGRA字节buffer→归一化GPU tensor**这一段，不宣称已完成CVMetalTextureCache、真实VideoToolbox解码或无copy纹理链路。独立客户端复用实际PE PNG decoder，但视频decode、resize、VAE和最终encode/flush都不在本次分母内。

## 1. 冻结真正下游要的布局

输入为有明确owner、rowBytes、BGRA顺序与宽高的CPU buffer。输出是完整ready、contiguous的FP32 `[1,3,1,H,W]`，数值为`(u8/255)*2-1`。这对齐产品video unit normalization与SeedVR2模型输入配方，再显式包含最终contiguous materialization；不是调用完整产品reader/请求的计时。

```text
prepared padded BGRA
  ├─ source-bound product loop ─ RGB8 import ─ compiled normalization/layout ─┐
  ├─ public vImage geometry/convert ─ RGB8 import ─ same GPU consumer ───────┤
  └─ padded BGRA8 import ─ compiled GPU swizzle/orient/normalize/layout ────┤
                                                        complete FP32 GPU output
```

三个arm都计CPU工作、MLX输入构造、GPU计算和完成同步，不只测vImage函数。`gpu-fused`是编译图配方标签，不是已用trace证明只有一个物理kernel。所有span都是同一process host uptime，不是CPU/GPU硬件独占时间。

## 2. 方向、padding和alpha不是附加选项

方向code的bit4表示交换X/Y，bit1/bit2分别反射输出X/Y；全部8种均保留每个像素。C++按产品display transform反向gather，独立Python按源像素向目标scatter验证，避免两边直接复制同一个索引公式。513×257矩形覆盖宽高交换和odd tail，不只测试正方形。

本次人工准备的row pitch为64-byte向上对齐后再加64字节，**不是观察到的CVPixelBuffer通用对齐规律**。每组原RGB与XOR1版本交替，是freshness控制，不是两个真实视频帧或独立模型输入。padding poison、额外尾guard和完整source hash逐call核对；GPU输出按FP64公式检查全部元素，max abs/relative L2≤1e−6、cosine≥0.999999，并完整恢复RGB8验证字节无误。

BGRA alpha按固定控制序列变化，证明这里只移除alpha而不混背景/除alpha。通用RGBA导入、premultiplied/straight alpha、ICC/EXIF/HDR/YCbCr矩阵与range是另一个合同。旧Qwen capture PNG的alpha253–255被固定opaque-file gate拒绝，未悄悄换成RGB计分；不从该拒绝推断VAE、编码器或权限故障。

## 3. 为什么更快的CPU操作仍可能输

vImage在prepared-buffer路径中通常减轻逐像素循环，却可能增加geometry pass和scratch。当前非原地geometry需要ping-pong两份BGRA缓冲，再convert到紧密RGB；多反射的小矩形案例反而慢于原循环。若下游本来就在GPU，上传较大BGRA后直接完成通道选择、方向、归一化和布局，仍可能比“更快的CPU + RGB import”更快。

[本机30组确认](case-studies/bgra-to-model-input-cost.md)中，真实512×512文件输入的vImage相对循环paired约1.35–1.73×，但融合GPU配方又比vImage完整交接快约1.07–1.24×。这只是该ready-input阶段的warm观察，不能乘上VAE/FFN收益，更不能当模型生成速度提升。

## 4. 真正接回CVPixelBuffer时要补什么

保留actual CMSampleBuffer/CVPixelBuffer、lock与producer/consumer完成边界；GPU未完成前不能unlock/release或让解码器复用源帧。`.alwaysCopiesSampleData=NO`不是无copy证明。rowBytes、plane、pixel format与实际buffer必须核对，不能拿本页CPU-owned数组当真实IOSurface/texture alias。

还要包含decode到BGRA可能已有的转换、真实display transform、resize/color/alpha、PTS/FPS/重排与完整EOF，测试bounded queue、取消/drain和最终encoder flush。当前没有CPU预处理与上一帧GPU工作的流水并行，也没有逐arm峰值/低RAM/长时或完整视频请求资格；CPU分支允许不采用。

## 实现与复核

核心PNG/方向/记录合同、与离线反例自包含；[设备integration](../examples/README.md)显式需要产品source/Apple SDK与installed MLX。PNG reference只支持有界非交错RGB/RGBA8且必须opaque，拒绝压缩尾流、坏CRC、透明输入和未知关键chunk，不冒充所有图片格式decoder。

重开条件：真实CVPixelBuffer/texture ownership、不同尺寸/方向/resize、alpha/color recipe、CPU-only consumer、异步帧流水、OS/SDK/MLX或低内存/长时条件改变。先有完整GPU控制与下游gate，再考虑产品采用。

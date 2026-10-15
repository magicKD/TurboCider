# 媒体 IO：完整帧、backpressure 与端到端时间

[总目录](../README.md) · [VAE](vae.md)

## 为什么是优化的一部分

全clip RGB、FP32 normalization、VAE输出、planar U8、codec queue若并存，可能比权重大。减少DiT时间后，导入、resize、decode、encode和mux的比例也会上升。媒体阶段不能从“完整生成”时间里悄悄删掉。

## Source streaming

一个复用RGB caller buffer → 相同display transform/resize/FP32 normalization → 完成的独立encoder frame owner → 下一帧。不能在GPU尚读时重写reader buffer。

保留原模型时域分组：例如initial-five/subsequent-four、4n+1 padding仅复制最后真实帧；encode后验证EOF，短/多序列、尺寸变化、FPS/PTS或source stamp不匹配要失败，不能静默截断/替换FPS。当前SeedVR2参考：[streaming source记录](../../docs/design/seedvr2-vae-optimization.md)、[RGB reader](../../native/media/rgb_video_reader.mm)。

一个caller buffer不证明codec内部只有一个buffer，也不证明完整请求峰值下降；模型conditioning/DiT和VAEhistory可能仍主导。

## Decode/export streaming

frame sink同步消费或bounded queue/backpressure；精确保留FP32 clamp/round到uint8、frame index/FPS timebase。消费者如果保留每帧tensor，bounded producer就失去意义。

输出写private staging，确认全部geometry/frame/FPS/finish后原子发布。取消/失败只移除owned staging，已存在destination和原输入不修改。backpressure时间属于decode或export哪个span必须写清，不能两个都算再相加。

## GOP与codec边界

GOP参数要明确单位是帧还是秒；源PTS/timebase和codec policy都记录。换H264/HEVC/VideoToolbox、profile、色彩范围或编码器实现后，压缩后pixel差异可能很大，不能直接归因VAE。

优先比较 preencoder RGB / 原始decoded tensor，之后再比最终视频。SeedVR2历史中preencoder RGB只差1，H264后却能出现更大差；见 [virtual causal conv](../../docs/design/seedvr2-vae-virtual-causal-conv.md)。codec差异与模型差异分层归因。

## 音频与视频双流

LTX/H3的音频有自己的latent、VAE/vocoder、sample rate/BWE和mux边界；双queue只有无依赖支路可重叠。cross attention、同步latent与shared scratch必须按真实DAG。把音频关闭获得的时间不能说成同一完整音视频任务提速。

做波形/听感、duration、AV同步、motion/身份与必要的lip-sync检查。MP4可播放、ffprobe字段正确仅证明封装/基本结构，不证明声音语义或口型。

## 长视频与小内存

1000帧reader/writer测试证明codec循环与buffer行为，不证明1000帧模型可跑。真正长视频还需要bounded complete-context conditioning/DiT/history或显式近似temporal window。独立窗口改变上下文，需固定noise、相位、overlap、边界运动和身份，不能改名为exact streaming。

可复用入口：[SeedVR2 pipeline](../../native/models/seedvr2/pipeline.cpp)、[media video](../../native/media/video.mm)、[LTX runtime说明](../../native/models/ltx_runtime/README.md)。

## 实现参考与实验

可读实现：[参考代码](../labs/media.py)；[实验步骤与边界](../labs/media.md)；[本项全部 case / controls](../labs/catalog.json)。

```sh
python3 -B techniques/labs/run.py media-backpressure
```

本项控制：buffered vs bounded full RGBA checksum；rational FPS and nested spans。CPU参考/模拟与实际设备入口分开标记；设备环境和更多变体见实验步骤。

可以重开探索的条件：different codec/source cadence/AV requirements or bounded pipeline。历史负结果限定于其来源/shape/边界，不关闭该方向。

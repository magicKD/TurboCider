# 媒体与VAE实验：halo、global stats、backpressure

[实验室](README.md) · [算子代码](operators.py) · [队列代码](media.py)

## 1. Whole-frame统计与两pass输出

```sh
python3 -B techniques/labs/run.py global-stat-tiles
```

baseline全frame mean/variance；candidate先合并各tile统计再发tile输出；negativecontrol各tile独立norm，能看到结果变化。双pass的reductionorder可能改变低精度末位，生产需Welford/pairwise/FP32witness和可接受误差，不只比double。

独立crop是有效的近似研究路线，而非“不能尝试”：给明确overlap/blend、seam/temporal/质量门，不沿用exact-context标签。

## 2. Causal taps与spatial halo

```sh
python3 -B techniques/labs/run.py causal-halo
```

连续三个temporal taps按`t-2,t-1,t`排列，空间3×3卷积；tile仍从完整上下文取halo，候选与wholeframe一致。缺halo的独立crop作为反例。只有initial重复同帧才可fold三个temporal系数，后续frame不能套此特例。

扩展到真实VAE：stackedConv2D的tap×channel顺序、bias位置、spatialstride、upsample phase、globalnorm/attention、原始FP16系数/FP32累加与historyowner都需补齐。GPU/ANE可按disjointoutputchannels合作，再比较完整pack/predict/assembly与scratch，而不是只比convcore。

## 3. 有限帧队列与准确timebase

```sh
python3 -B techniques/labs/run.py media-backpressure
```

真实Pythonproducer/consumer通过capacity2队列消费25张合成RGBA，所有bytes与bufferedhash一致；caller/queue/consumer的logicalbuffer计数有显式上界，结束归零。保留`24000/1001`的有理timebase，不四舍五入成24。

它不测codec内部RAM、AVFoundation、VAE或千帧模型；计数是payloadowner，不是RSS。实际媒体流程还应测试取消/short/extraframes/EOF/PTS、alpha、原子publication、失败后source/destination不变。

## 4. 探索不止tile大小

建议逐项测：history compactowner、共享norm统计、virtualhalo、norm/SiLU/pack融合、row-file存储、temporal分组、输出sink、DiT/VAEphase释放。局部memory减少后重新找全请求峰值；更大tile/更少calls可能无益，但新硬件/边界可重测。

图像Real-ESRGAN的reflectpadding/feather/alpha与时序VAE的完整globalcontext不同，分别参考 [辅助超分](../media/upscaling.md) 和 [VAE](../media/vae.md)。

```sh
python3 -B techniques/labs/run.py upscale-tiles
```

此spatial-only case执行可读的halo/overlap/weightedfeather callback，并单列local-reflect近似。它的toy model是blur+nearestx2，不是Real-ESRGAN；真实alpha插值、premultipliedRGB和orientation应接实际素材链路。

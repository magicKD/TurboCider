# Encoder 分区、准备资源复用、预取与独立校准

2026-10-08，接续 [20：encoder runtime 优化](20-encoder-runtime-optimization.md)。本轮实现的是现有 MLX/private-ANE 桥接的调度优化，完整测量见 [状态记录](../../status/2026-10-08-qwen-encoder-scheduling.md)。没有完成原生 Metal encoder，也没有把整个 encoder 变成仅由设备事件驱动的执行图。

## 1. 从 Splash 迁移什么

继续固定已核对的 Splash commit `0d5a035ea26d1cc566e3c7ee4e08e5e06fa68cf7`。参考其 [AneFfn.cpp](https://github.com/incoai/splash/blob/0d5a035ea26d1cc566e3c7ee4e08e5e06fa68cf7/runtime/ops/AneFfn.cpp)、[Metal FFN 内核](https://github.com/incoai/splash/blob/0d5a035ea26d1cc566e3c7ee4e08e5e06fa68cf7/runtime/metal/kernels/prefill/ane_ffn.metal) 和 [校准实现](https://github.com/incoai/splash/blob/0d5a035ea26d1cc566e3c7ee4e08e5e06fa68cf7/runtime/ops/AneFfnCalibration.cpp)：先提交当前 GPU complement，再安排下一层 W bank；按真实序列和完整路径判断分区，而不是按单次矩阵乘法决定。

本地 BF16 checkpoint、Qwen3-VL H4096/F12288、GQA/mRoPE/DeepStack 都有自己的语义，不能直接调用上游限制 H 为2560倍数的 FFN API。我们复用已有 Hadamard W8A8、双权重 bank、事件/完成票据和 MPP GPU partial，增加以下四项能力。

## 2. 分区减少调用

[encoder_runtime_policy.hpp](../../../native/models/qwen21/encoder_runtime_policy.hpp) 增加两种有界几何1152/1664及 `balanced` 策略。

```cpp
smallPadded = ceil(rows / 1152.0) * 1152;
largePadded = ceil(rows / 1664.0) * 1664;
useLarge = largeChunks < smallChunks && largePadded <= smallPadded;
```

| 实际行数 | 1152分区 | 1664分区 | balanced |
|---:|---:|---:|---:|
|1071|1次，补81行|1次，补593行|1152|
|2101|2次，补203行|2次，补1227行|1152|
|3131|3次，补325行|2次，补197行|1664|

规则避免以更大padding换取相同调用次数。分区只是资源选择，不改变有效token数或截断prompt。语言FFN独立按行计算，因此padding不进入attention；完整语言attention仍使用原始行数和mask。

实际 `build_runtime_executor` 增加显式fixed bucket参数：先验证完整模型descriptor，再只允许authorized Private、shape-only、partial-channel、512..2048且128对齐的覆盖。Public W8A8模板拒绝该覆盖；不能绕过Public ABI或把每个prompt导出成新程序。旧DiT调用的默认参数不变。

选项 `TURBOCIDER_QWEN21_ENCODER_RUNTIME_BUCKET=descriptor|1152|1664|balanced`，默认仍按descriptor。只有两个研究候选，并不是任意行数缓存。

## 3. Session只缓存一个准备好的分区

[EncoderRuntimeCache](../../../native/models/qwen21/encoder_runtime.hpp) 归Qwen Session所有。显式设置 `TURBOCIDER_QWEN21_ENCODER_RUNTIME_REUSE=1` 后，缓存一个HybridFfn和一个dynamic-weight GPU compiled callable。

缓存key包含descriptor内容摘要、executor配置、GPU recipe、选项、分区、ANE资源预算和校准摘要。切换几何/校准先销毁旧资源。请求借出资源后，必须完成private drain和GPU synchronize才能归还；失败路径不把未完成或退化的资源放回缓存。卸载模型或关闭该入口释放缓存。每个请求重新调用内存准入。

只保留执行程序、绑定、事件、IOSurface、最多双权重bank和小缩放向量，不保留完整BF16 checkpoint、输入activation或旧GPU权重closure。bank可以保留上一次的INT8 codes，但每次新层/新请求仍按真实源重新stage。MLX callable的权重是动态参数，不捕获Session旧Weights。

`reset_phase_metrics`在排空请求边界清空本次模型调用/预取/S1计数；复用请求不再归入constructor/self-test时间。设备内部少量cache统计仍可能是该执行资源的生命周期累计值，不能误当作本次执行量。

验证包括：同一准备缓存更换up权重为0.5倍，重新bind的新Weights输出与新GPU oracle比较；以及两次完整encoder各自创建新SourceLease/Weights，禁止conditioning cache，检查第二次准备复用和结果一致。

## 4. 当前GPU先提交，再预取下一层

[HybridFfn](../../../native/backends/ane_ffn.cpp) 的row/channel路径都把 `maybe_prefetch` 移到当前GPU `async_eval/eval`提交之后。

```text
当前层stage → attention/norm → 唯一packed-input readiness fence
    → launch ANE → submit current GPU complement
    → admit/load next layer sources → stage alternate W bank
    → finish current ANE → FP32 join → residual
下一层：核对immutable allocation/scale/selection → activate matching future bank
```

Encoder的NextWeights provider只提交一层gate/up/down，检查真实shape/dtype/键和额外约288MiB原始源的内存准入。失败或准入不足跳过预取；下层按普通stage执行，不能以预取失败产生成功输出。复用既有消费者/生产者完成栅栏，最多一个future bank。由 `TURBOCIDER_PRIVATE_ANE_PREFETCH=1` 显式启用，默认关闭。

MPP encoder同时去掉额外 `eval(x)`：`run_channels`对packed/padded输入的eval是导入DeviceView前的实际依赖栅栏。owner、offset、物理完成和最终排空保留。legacy对照保留旧显式输入等待。

**仍有限制**：next-source `mx::eval` 是host等待，可能等到当前GPU完成才开始staging；private launch/finish和MLX input pack仍有CPU边界。先提交GPU只避免把未来加载放到当前GPU提交之前，不保证未来stage完全隐藏，也不等于W2。

## 5. Encoder专属SmoothQuant校准

新 [encoder_calibration.hpp](../../../native/models/qwen21/encoder_calibration.hpp) 独立于DiT32层profile，完整采集语言36层真实post-attention-norm输入，视觉塔保持原样。

每层对全部实际token收集 `amax[c]=max(abs(x[...,c]))`，对ANE所选gate/up输出通道收集 `wmax[c]`，得到：

```text
s[c] = clamp(sqrt(max(amax[c],1e-6)) / sqrt(max(wmax[c],1e-6)), 1/16, 16)
x' = x/s;  W' = W*s;  然后执行原来的H128旋转/量化
```

仅改变ANE gate/up输入basis；GPU complement、hidden H512旋转、down与BF16边界不变。weight统计用coalesced原始BF16读取的Metal kernel，每层只输出4096个FP32值，避免生成完整FP32/abs权重副本。

profile只有`s1:float32[36,4096]`，约0.56MiB，无权重/activation。绑定schema、完整GPU trajectory、checkpoint canonical/stat fingerprint、参考图数量/分辨率、ANE channels、H/F/layers、alpha=.5、rotation basis与参考图摘要。加载前后复核小文件SHA256，拒绝不完整/非finite/范围错误/异模型或域不匹配profile。profile明确 `qualified=false`；一次轨迹不代表跨prompt/图像质量合格。

采集入口 `TURBOCIDER_QWEN21_ENCODER_CALIBRATION_OUT=/absolute/new.safetensors` 要求符合encoder研究域的GPU edit、禁用conditioning hit、未设置runtime manifest，并拒绝覆盖。使用入口 `TURBOCIDER_QWEN21_ENCODER_S1_PROFILE=/absolute/profile.safetensors` 要求显式runtime和approximation。不能同时采集和使用。

仅generic HybridFfn的S1 factory扩展为显式32或36层；历史DiT loader/schema仍是32层，encoder profile不能混入DiT。校准路径/内容摘要进入进程recipe和conditioning缓存身份。

## 6. 有界验证和继续优化

先声明FFN RRMS≤5%、完整conditioning RRMS≤8%、完整路径最低省时5%。完整对照使用未经校准采集插桩的GPU基线；采集时间不作为GPU性能基线。只使用已有teapot/fox/puffin和本地Qwen3-VL BF16，无模型下载、完整导出或DiT/VAE生成。

组件覆盖真实首/中/末层0/17/35。1152无S1、1664无S1、1664加S1各warm+AB/BA；数值与完整FFN路径均通过后，才进行完整vision×3+language encoder对照。单样本仍不授予发布/自动路由资格。结果与失败记录见状态文件。

后续设备驱动encoder必须分阶段实现：

1. 为Qwen3-VL实现并核对FP32 RMSNorm→BF16、GQA、interleaved mRoPE、causal mask及DeepStack/residual；现有DiT NativeAttention/NativeQwenOps不能直接代用。
2. 给MLX island增加可封存的produced-buffer/event导入；读者持有producer event和Data owner，以dependency证明readiness，禁止把lazy array直接交给private queue。
3. 将next weight producer放到attention之前可持续工作的队列，减少当前 `mx::eval(next)` host等待，并比较统一内存带宽竞争；无收益则不启用预取。
4. 先完整单层验证异步lane和join，再扩展两层设备票据。current consumer/future producer都完成后才允许bank复用；timeout/cancel需隔离资源并排空，不能提前释放source。
5. 分别验证无S1与encoder S1全轨迹conditioning误差；收集不同prompt、1/2/3图的轻量校准代表性，再考虑更复杂的channel routing、hidden尺度或原生全encoder。

现有桥接改动为这些步骤提供资源、recipe、profile和有限对照基础；完整Metal command graph、设备事件控制的跨层流水及App可发布选项仍需独立验证。

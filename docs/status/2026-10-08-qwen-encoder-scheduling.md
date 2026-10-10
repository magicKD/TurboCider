# Qwen encoder：分区、预取、准备复用和36层校准

2026-10-08，本机M4 Pro48GiB。当前固定三参考图输入出现完整encoder净收益；仍为诊断功能、默认GPU。新的设计及对应代码见[21](../design/qwen-image-gpu-ane-command-graph/21-encoder-scheduling-and-calibration.md)，机器可读数据见[JSON](2026-10-08-qwen-encoder-scheduling.json)。

## 实际完成

- 将row/channel桥接的下一层预取移到当前GPU提交后；encoder提供有内存准入的一层真实BF16权重provider，复用双bank/事件票据。
- 新增1152/1664有界分区与balanced选择；实际3131行变成2个1664分块，完整36层ANE evaluation从108次降为72次。
- Session仅保留一个准备执行资源，排空后归还，卸载/切换配置释放。dynamic GPU图不捕获旧Weights。更换up权重为0.5倍的缓存重绑检查通过，RRMS0.7505%，仅1次本次FFN计数、GPU callable不重建。
- MPP路径用packed-input eval承担唯一导入依赖栅栏，移除之前额外eval(x)。仍存在MLX输入pack、private launch/finish及host join，未完成Splash式完整Metal encoder/W2。
- 新增独立encoder36层S1校准和strict schema，alpha=.5、真实全部token、ANE所选gate/up通道；只保存约0.56MiB通道缩放值，无checkpoint/activation导出。

## 组件筛选

本地Qwen3-VL BF16，全27层视觉×3及36层GPU语言轨迹采集真实FFN输入；测试层0/17/35，rows3131、ANE channels6144。每个组件候选warm+AB+BA。时间包含stage、GPU/ANE、pack和join。实验前声明FFN RRMS≤5%、完整conditioning RRMS≤8%、最低省时5%，没有事后提高门槛。

|分区|量化|六个配对省时范围|平均省时|首／中／末层FFN RRMS|
|---:|---|---:|---:|---|
|1152|Hadamard|22.43–24.60%|23.96%|0.740% / 1.350% / 2.277%|
|1664|Hadamard|27.38–33.34%|29.96%|0.740% / 1.350% / 2.277%|
|1664|encoder S1 + Hadamard|26.83–29.68%|28.61%|0.735% / 1.310% / 2.051%|

S1改善当前三个FFN误差，尤其末层，但平均速度未超过1664无S1。按预声明的完整路径性能规则选1664无S1进入完整对照；S1仅通过组件检查，**尚无完整encoder S1质量/性能资格**。不追加参数/通道扫描。

第一组测试工具在缓存更换权重时，标量乘法将BF16提升为FP32，运行时拒绝该源；保留失败supervisor。修正测试工具为显式BF16后重建probe，复用已经成功采集的36层小profile，再完成同一组有限组件/缓存检查。两个成功逻辑组，另有一次该工具错误导致的失败尝试；没有连续模型搜索。采集插桩时间不充当GPU性能基线。

## 完整encoder对照

同一生产构建 `tc-runtime-build-v1-c4e2cc0d6b5f3598590a199045805cb06d10e6dc169d19c5225a38f54c9ef76b`，单次固定顺序GPU→1152→1664→prefetch/cache二次请求。视觉27层×3、完整语言36层、GQA/mRoPE/DeepStack、pre-final-norm retained conditioning全部执行；没有DiT/VAE/最终编辑图片。输入为已存在teapot/fox/puffin及固定prompt，reference1024；目标App编辑域为512×512，**不能把这些时间说成512 reference的成绩**。

|路径|视觉|语言|准备资源|encoder总时间|相对GPU省时|ANE调用|
|---|---:|---:|---:|---:|---:|---:|
|GPU|4.709s|7.381s|0.000ms|12.373s|—|0|
|1152，无预取|4.743s|6.729s|259.098ms|12.020s|2.85%|108|
|1664，无预取|4.711s|6.341s|307.274ms|11.641s|5.91%|72|
|1664，预取，初次资源准备|4.712s|5.977s|307.457ms|11.277s|8.85%|72|
|1664，预取，复用准备资源|4.722s|6.006s|0.200ms|11.011s|11.01%|72|

1664无预取比本轮1152总时间少3.15%。预取在初次准备样本中比1664无预取总时间少3.13%，语言少5.74%；第二次复用总时间比初次预取少2.36%。这些来自少量固定顺序样本，不能把变化全归因于一个优化，也不是置信区间。

两次预取请求各35次submission/35次hit/0失败，第二次gpu_graph_builds=0、preparation_reused=true；两次都重新创建SourceLease/Weights并运行完整vision/language，不命中conditioning cache。第二次准备约0.20ms，对照初次约307ms；private程序/source/OS/driver缓存已经由组件测试预热，**初次准备不是冷磁盘/首次ANE编译**。

所有hybrid retained conditioning相对GPU的RRMS均为5.75399%，不同分区/预取/准备复用输出字节一致，满足本轮预声明8%encoder筛选。历史5%完整conditioning失败保持原结论，不宣称已经通过5%。这是encoder张量误差，不是像素误差；没有新最终图片质量证据。

组件通过且完整样本达到最低5%收益；仍未授稳定、自动路由或发布资格。此完整请求只有三图1024-reference和一个prompt，1/2图、512-reference及不同文本需后续有限对照。MLX peak约2.16GiB，不能当作IOSurface/driver在内的进程峰值；准备cache约束为1GiB资源预算，并非整App内存容量保证。

## 验证和资源

- 原生CLI/动态库、Swift App和模型管理helper构建成功；575个生产source摘要最终复核一致。
- CPU策略57项、两种bucket×两种channel的MIL生成、4个BF16/FP16动态GPU权重case通过。
- encoder profile复制所有权、错误参考图数量/分辨率/通道域和partial-layer拒绝检查通过，未加载checkpoint tensor。
- S1回归9项、原生contract129项（3项skip）、encoder结果metadata、worker协议/EOF/admission通过。两条旧LoRA源码断言改为检查shared apply_loras_impl/bind_runtime，清理合同保留。沙箱看不到Metal/GPU标识导致的旧失败也保留；开放设备访问后相应检查通过。
- 本轮成功设备组前后swap均4581.69MiB，没有系统thermal/performance warning；不是全程采样，不能排除瞬时压力。
- 清理本轮较慢三图1152重复程序、unused中间probe、owned objects/module cache约214.79MiB，只保留1664有效程序约6MiB、小profile和证据/可运行构建。旧轮用于1/2图的1152缓存及模型/global driver缓存未改动。

## 下一步

优先建立Qwen3-VL专属Native GQA/mRoPE/FP32 norm/DeepStack语义对照，再将输入produced buffer和event封存，移除next-source eval带来的host边界；先证明单层完整路径，再扩展跨两层设备票据。当前约19%的语言收益仍受attention、准备、视觉比例及统一内存带宽限制。S1后续应在完整conditioning上验证误差是否改善，再谈跨prompt校准覆盖；不因组件误差较低默认开启。

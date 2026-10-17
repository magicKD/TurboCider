# ANE性能边界：SRAM、图粒度与共享带宽

<!-- knowledge
id: technique-ane-performance-limits
type: method
stage: ffn
operator: gemm, scheduling
backend: coreml, private-ane
device: ane, gpu
symptom: memory-bound, straggler, launch-bound
semantics: contract-only
representation: fp16
boundary: gpu-ane, producer-consumer
api: public, private
-->

[总目录](../README.md) · [图合同](coreml-graphs.md) · [校准](../gpu-ane/calibration.md)

补充：[硬件/框架能力地图](capability-atlas.md)、[跨项目异构实现](hybrid-runtime-comparison.md)、[负结果和重开条件](negative-results.md)。本轮[合成压缩实验](experiments-20261016.md)也显示小图可能首选CPU、文件压缩率不等于加速比。

后续[GPU-resident完整FFN五臂](resident-ffn-layout-and-gpu-baselines.md)将pack/restore计入并加入原生MPS与MLX对照；large计划首选ANE仍有92输出超原数值门槛，不能只用诊断wall排名。

## 不以TOPS决定迁移

对一个候选算子，先分解：权重读取/转换、输入pack、dispatch/setup、图内计算、片上/片外流量、输出restore、首个consumer等待。ANE的峰值算术、CoreML支持某op、一个micrograph很快，都不足以预测完整FFN或媒体请求。

同一graph可因为rowbucket、channelwidth、K/Ntile、dtype和算子表达而出现不连续的编译/执行行为。不同Apple芯片/OS/工具版本分别验收，不硬编码传闻中的ANE SRAM大小、内部tile数或编译次数限制为产品合同。

## SRAM与tiling应怎样探索

片上容量/带宽是有价值的研究方向，但观测到某shape拐点不自动证明某个SRAM容量。固定dtype/系数/输入，只扫小范围rows/channels/Ktile，同时记录graph结构、compile/load/hot、外部IO和完整consumer。用独立重复与反序排除频率/缓存影响。

优先保留gate/up→activation→down在一个prediction里，让大intermediate不来回跨设备。拆FC2成多个host调用会增加dispatch和partialreduction；如果为适配图限制必须切分，先把这个成本测出来。固定smallrows降低padding税，但过多chunks也会放大每次调用固定开销。

## 量化为什么可能只降低流量

W8/A8可减少surface或图内流量，normalizedcode/scales也可能改变graph布局；然而Q/DQ、rotation、hiddenrestore都要工作。看到INT8features和更快prediction仍不证明物理INT8MAC——必须另有独立执行证据。

把激活由BF16转FP16要处理范围，不只精度。zero/nearzero、outlier、headroomretry与归一化scale都影响图稳定性；不能为了“fit SRAM”把关键状态悄悄降精度。

## 三种典型限制

| 限制 | 观测 | 实验方向 |
| --- | --- | --- |
| 固定调用/加载成本 | 小rows不盈利，cold远大于hot | session/backing复用、最小bucket、rowthreshold |
| ANE straggler | GPU早完、ANE alone也慢 | 减少share/换图边界，别继续加offload |
| UMA / staging竞争 | alone好、concurrent差，prefetchhits无收益 | 测完整traffic、关/延后prefetch、保守share |

若性能退化来自driver/系统热状态，记录只读观察与复测范围；不要通过全局wired/swap/power设置制造不可复现优势。

## 实际参考与适用范围

[mac_transformer异构报告](../../docs/design/transformer-heterogeneous-report.md)覆盖CPU tail、QKV、FC2chunks与publicattention的负结果；[Private prefetch](../../docs/status/private-ane-prefetch-2026-10-03.md)覆盖机制命中却更慢；[VAE feasibility](../../docs/design/seedvr2-vae-ane-feasibility.md)覆盖小tile快、完整边界不快。

外部`gpu_ane/ANE`的SRAM/dispatch探针只作为研究方法参考，不能直接授予TurboCider型号参数或发行兼容性。需要采用某个结论时，固定其source revision并在目标SoC/OS/实际模型重新测量。

## 实现参考与实验

可读实现：[参考代码](../examples/reference/systems.py)；[实验步骤与边界](../examples/README.md)；本项全部 case / controls。

本项控制：fixed overhead vs useful rows；complete independent spans vs exposed wait。CPU参考/模拟与实际设备入口分开标记；设备环境和更多变体见实验步骤。

可以重开探索的条件：graph tiling/SRAM expression/row threshold or hardware changes。历史负结果限定于其来源/shape/边界，不关闭该方向。

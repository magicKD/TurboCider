# Cache-DiT消费者调用链：单次getter、lazy clock与真实gate的小型重放

<!-- knowledge
id: technique-algorithms-case-studies-cache-consumer-callchain
type: case-study
stage: dit
operator: sampling
backend: cpu-reference
device: cpu
symptom: stale-cache, measurement
semantics: approximate, contract-only
boundary: producer-consumer
api: reference-only
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

[算法入口](../README.md) · [前轮状态反例](forecast-state-audit.md) · [FoCa/DMD](../methods/forecast-calibrators.md) · [实验](../../examples/README.md)

2026-10-16第三轮。此前仅验证孤立state类，本轮进一步执行固定版本的forward/manager/context-clock方法，核对哪些前提实际能沿调用链接通。它仍不是完整模型、包初始化或生产请求验证。

## 执行的是真实方法，但替身也写清楚

cache_consumer_probe.py先校验5个完整源文件hash，再提取：`CachedBlocks_Pattern_Base.forward`，manager的`can_cache/similarity/get_Bn_buffer/set_Bn_buffer/apply_cache`及相关getter，context的`mark_step_begin/add_cached_step`，FoCa/DMD校正器。仅去除compiler/装饰器包络，不改被选函数body。

测试替身：context的配置/存储getter与选择、F/M/B计算及参数检查；CFG关闭、无分布式、CPU Torch2.14.1/FP64。F(h)=h+1，middle额外加logical_step，B=identity，故完整组件oracle为2t+1。没有原模型权重、LoRA、scheduler或媒体。

gate不是手工指定的step mask：实际manager用constant front residual=1、阈值0.1、3步warmup、max-consecutive=1，得到full步0/1/2/4/6，hit步3/5/7。完整32行trace保留每次prefix、getter次数、clock、输出与oracle。

## 默认调用路径并没有重复读取Bn

```text
front真实计算
  → mark_step_begin
  → can_cache(prefix=chain_Fn_residual) → 读取Fn比较值
  → hit: add_cached_step → apply_cache
      → get_Bn_buffer一次 → calibrator.approximate
  → full: middle真实计算 → set_Bn_buffer → calibrator.update
  → back / 输出
```

正常两个校正器在每个hit都只有一次Bn getter。因此，上轮“重复读会推进FoCa状态”的反例，**不是证明默认forward自动重复读**。manager也支持Bn前缀的diff路径，但此普通pattern传的是Fn；不能混淆不同consumer。

本轮额外增加诊断臂：仅在step5的cache判断之后手工多读一次Bn，再让正常forward继续。它模拟诊断/日志工具潜在的observer effect，明确不是默认行为。

## 观察结果

| 路径 | step3 output / oracle | step5 output / oracle | step7 output / oracle |
| --- | --- | --- | --- |
| FoCa，正常 | 7 / 7 | 11.5 / 11 | 15.25 / 15 |
| FoCa，step5额外诊断读 | 7 / 7 | 11.75 / 11 | 15.125 / 15 |
| DMD，正常 | 6 / 7 | 11.5 / 11 | 16 / 15 |
| DMD，step5额外诊断读 | 6 / 7 | 11.5 / 11 | 16 / 15 |

full步的组件oracle均匹配；hit步的结果不是因此获得完整生成质量资格。source manager的max-consecutive=1仍允许full anchors之间存在两步间隔，FoCa的full差分不按间隔归一化在此单次getter路径也能产生线性oracle偏差。

DMD校正器虽开启，但这里uniform tail最多3份，始终走Taylor fallback；没有做SVD/eig。不能把开启DMD的日志当作实际DMD拟合成功，也不能把fallback误差归咎于指数模态方法。

## lazy state创建让clock原点更微妙

context先mark，随后第一个full `update(name)`才创建该命名state，trace中calibrator_step=logical_step-1。相对间隔可保持一致，所以这不单独证明off-by-one错误；但DMD中以绝对step条件决定是否产生导数，会使冷启动行为与“先创建state再mark”的孤立探针不同。前轮线性fallback得到4而非3，本轮首次hit的偏差方向不同，必须保留两种调用前提。

FoCa的预测会推进rolling pair，额外诊断读改变本步及后步；DMD的读取没有同样的rolling更新，因此这组诊断臂未改变其结果。不要把getter在某一实现中纯读的经验推广到所有calibrator。

## 可迁移的工程合同

区分logical model evaluation、CFG branch、context clock、named-state clock和full-anchor间隔。统计/日志应读取已提交的decision/prediction memo，不重新调用有副作用的getter。为每种路径记录fit/fallback/full/replay实际次数，失败不提交半份history。

本轮不修改上游，也没有证明实际Qwen/LTX模型会产生这些输出偏差；只证明给定微型F/M/B和真实gate/方法body可以到达相应状态。后续模型trace应继续核对clock创建、step重启、callback重复读取、CFG与实际snapshot owner。

来源：manager见[S37](../sources.md#s37)，pattern/context见[S06](../sources.md#s06)，预测器见[S33](../sources.md#s33)。默认离线测试只核对整理trace/来源，不自动执行上游方法或GPU。

# 预测缓存状态审计：12个固定源码CPU观察与4个合同反例

<!-- knowledge
id: technique-algorithms-case-studies-forecast-state-audit
type: case-study
stage: dit
operator: sampling
backend: cpu-reference, mlx, metal
device: cpu, gpu
symptom: stale-cache, measurement
semantics: contract-only
representation: bf16, fp32
boundary: producer-consumer, request
api: reference-only
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

[算法入口](../README.md) · [FoCa/DMD](../methods/forecast-calibrators.md) · [Spectrum](../methods/spectrum-forecasting.md) · [配套实验](../../examples/README.md)

2026-10-16第二轮。目标是检查**实际开源实现的时间/历史合同**，不是用toy质量排名论文。本轮没有加载权重、生成图片/视频、提交GPU计算、修改上游或发起外部PR。

## 怎样保证测的是原实现

source_state_probe.py 先校验固定commit和完整文件SHA-256，再从AST只提取白名单状态类/依赖函数；不导入上游package initializer/CLI。实际执行Cache-DiT的FoCaState/DMDState和Spectrum的forecaster声明，张量及线性代数显式CPU，Torch2.14.1。

普通`.venv`缺少torch，首次启动失败且没有生成成功报告；随后使用本库已有、已装torch的环境，未安装新依赖。复现只需任意明确具备torch的CPU Python，默认unittest不会运行上述source execution。

整理数据保存12条完整观察、每个source的commit/hash/声明名单与执行范围。脚本return0表示观察完成，**不表示12个oracle都通过**；`oracle_met`及`observed_oracle_misses`保留失败，不能被总状态覆盖。

## 实际观察

| 场景 | observed / oracle | 判读 |
| --- | --- | --- |
| FoCa：full在0/1，预测2，F(t)=t | 2 / 2 | 相邻anchors的线性控制通过 |
| FoCa：full在0/2，预测3，F(t)=t | 4 / 3 | full差分没有除以两步间隔 |
| FoCa：每步一次读取，预测3 | 3 / 3 | 单次读取控制通过 |
| FoCa：step2多读一次，随后预测3 | 2 / 3 | getter推进rolling状态；重复读取影响未来 |
| DMD：full0/1/2后，预测3，F(t)=t | 4 / 3 | 此时走短历史Taylor fallback，不是DMD eig拟合 |
| DMD：full0/1/2/4的等间隔tail | 长度2、spacing2 | 正确拒绝把混合间隔当成一个uniform窗 |
| DMD：full0..3，F(t)=0.9^t，预测4 | 0.656099996 / 0.6561 | 小型CPU指数路径正常，绝对误差约4.18e-9 |
| Spectrum：仅1份观测，ready | true / true | 源码没有按已填history长度等待，不是充分观测保证 |
| Spectrum：t75的τ | 2 / 2 | 当前代码固定时间域0..50 |
| Spectrum：T4(2) | 97 / 97 | 区间外基增长；不是实测生成误差 |
| Spectrum：first update后原地改调用者tensor | 7 / 1 | 首份stored snapshot不是独立副本 |

表中DMD tail长度与spacing在JSON中是两条观察，所以共12条。4个oracle miss都由解析函数或原tensor身份独立界定，不用近似输出与自己比较。

## 源码原因与适用范围

FoCa保存了`prev_full_step`，但所读`approximate`的`deriv_full=F_full-F_full_prev`未使用它；当full anchors不相邻，单位逻辑step的斜率会放大。`approximate`同时推进F_k/F_km1，因此一次step取多次同stream预测需要幂等缓存或严格single-read合同。没有检查完整Cache-DiT所有consumer是否实际触发这些用法，不能宣布所有模型都已受影响。

DMD `_update_taylor`用旧`dY_prev`算差分，之后才交换`dY_prev=dY_current`；在这个最小轨迹上读到了比上一full更旧的anchor。该问题只证明所读版本的fallback线性合同不满足，不否定DMD指数基原理，也不否定其足量uniform-history路径。

Spectrum第一次`h.reshape(...).to(same_device)`可以保留输入存储，未clone；后续cat分配不修复首次写入后的alias窗口。是否在真实pipeline中被覆写取决于buffer复用，不能把反例直接当成已发生的生成回归。

本轮没有给第三方打补丁。若移植，应先冻结坐标/每step调用次数/状态提交点/owner，再决定规范化、幂等memo、fallback修正和owned snapshot；任何修改都另建A/B，不重写旧证据为已修复。

## 复现

第一条是显式source-code execution，有完整digest拒绝；不是对任意用户脚本的沙箱。不同源码digest会拒绝，需重新审读后才能更新白名单。第二条是原创标准库数学实验，不依赖torch/兄弟仓库。

重开条件：上游commit、state update/read协议、logical step与sigma映射、fallback/owner变化，或获得真实pipeline调用trace。GPU性能与完整媒体质量尚无本轮新结论。

第三轮已补[消费者接缝重放](cache-consumer-callchain.md)：真实forward/manager/gate/context-clock、微型F/M/B与明确替身。默认每hit仅一次Bn读取，不能把此处重复读取反例误报成默认consumer行为；lazy创建改变冷启动clock，DMD实际仍走fallback。原12条观察与4个miss保持不变。

## 第二轮交付复核

并行新增文档导致的中途全库链接/metadata失败保留在本地旧报告中；最终重跑已消除，不将旧失败改写为从未发生。本轮没有修改产品默认、上游预测实现或提交外部issue/PR；真实模型消费者的forecast调用前提与新稀疏Metal速度/媒体质量仍待继续研究。

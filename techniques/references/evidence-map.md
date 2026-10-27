# 来源与案例阅读地图

<!-- knowledge
id: technique-references-evidence-map
type: method
-->

[总目录](../README.md) · [固定版本来源](../sources/README.md) · [设计样本](../designs/README.md) · [历史经验](history-lessons.md)

本目录是知识库，不分发实验数据、验证回执或机器清单。这里说明怎样理解不同材料，不将阅读记录当成当前运行资格。

## 区分五种认识

1. **源码事实**：固定 revision 的实现、guard、数据流与 API 合同。
2. **作者报告**：作者在特定硬件和任务下的结果，不等于本机复现。
3. **历史观察**：案例中记录的局部数值、速度或失败；原始实验包不随库提供。
4. **独立推导**：代数、流量或成本模型，有明确前提，不是实测。
5. **待验证假设**：值得下一次尝试的解释，不能改写成已证明原因。

比较范围不同时并列说明，不拼成排名，不把组件倍率乘成整模型倍率。

## 按学习目标定位

| 目标 | 入口 |
| --- | --- |
| GPU 数据流、选核和低比特消费 | [Metal 方法](../metal/README.md) · [Metal 来源](../metal/sources.md) |
| ANE 图、生命周期和训练接缝 | [ANE 方法](../ane/README.md) · [ANE 来源](../ane/sources.md) |
| 跨设备切分与消费者 ABI | [协同方法](../gpu_ane/README.md) · [协同来源](../gpu_ane/sources.md) |
| 缓存、稀疏、预测与质量取舍 | [算法地图](../algorithms/README.md) · [算法来源](../algorithms/sources.md) |
| 把机制迁移到新模型 | [阶段地图](../models/stage-map.md) · [实现样本](../examples/README.md) |

产品当前行为应回到实际产品代码和配置确认，不能从旧笔记推导已部署。需要新实验证据时，在独立任务和本地工作目录中取得，而不是把全套输出重新塞回知识库。

# Qwen Image 2.1 Base 的 QKV 实验

这里保存 2026-10-18 至 10-19 的 512² Base QKV 实验。报告分别记录单投影、
完整 block 和 40 步图像请求的输入、执行路线与测量范围。

| 记录 | 测量范围与限制 |
| --- | --- |
| [固定 Private W8 组件](qwen-base512-private-qkv-component-2026-10-18.md) | 256/512 行固定图，保留真实组件结果和当时未接整请求的边界 |
| [单组权重前看](qwen-base512-qkv-lookahead-2026-10-19.md) | 固定图复用、生成和编辑的匹配完整请求，以及输出一致性的范围 |
| [权重提交前沿](qwen-base512-qkv-frontier-2026-10-19.md) | 等待从 QKV 转到 FFN 后，完整消费者未获得稳定收益 |
| [W8 producer 合并](qwen-base512-qkv-w-stage-batch-2026-10-19.md) | 逐组提交和 metadata 开销，保留完整投影的负结果 |
| [Private FP16 调度](qwen-base512-private-fp16-qkv-2026-10-19.md) | 输入准备和真实投影筛选，最快受测 FP16 路线仍慢于当时的 W8 路线 |

[缓存归档](../qwen21-cache/README.md)记录首步 QKV 的 GPU 选择如何影响
text-prefix 命中的质量。[Runtime ANE 归档](../runtime-ane/README.md)保存其他
几何、精度和模型范围的对照。当前可用路线以[用户指南](../../public/README.md)为准。

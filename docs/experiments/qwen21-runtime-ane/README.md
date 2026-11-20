# Qwen Image 2.1 Base 的固定 Runtime ANE 实验

这些记录覆盖 2026-10-18 至 10-19 的 512²、40 步 Base 实验。各篇记录
对应构建、设备、工作负载与失败边界。组件和完整请求按原记录的范围比较。

| 记录 | 测量范围与限制 |
| --- | --- |
| [固定 Runtime ANE 与 vpipe](qwen-base512-runtime-ane-profile-2026-10-18.md) | 原 BF16 请求、分阶段 profiling 和同机对照，保留小样本及竞争负载限制 |
| [固定 QKV＋FFN 联合运行](qwen-base512-joint-qkv-ffn-2026-10-18.md) | 共享固定图、完整生成和编辑，保留观察窗口与资格限制 |
| [联合分工、固定 K 与质量](qwen-base512-joint-share-quality-2026-10-18.md) | 份额和 K 分片筛选及五组质量对照，保留参数试验无新增收益的结果 |
| [固定 Program 与失败分支](qwen-base512-fixed-program-2026-10-18.md) | 热重建、失败处理、固定行数和图生命周期 |
| [多行数 W8 完整 FFN](qwen-base512-w8-function-family-real-2026-10-19.md) | 原权重和冻结输入的完整 FFN 筛选，拼接长行样本按原范围记录 |
| [Function family 的 40 步接入](qwen-base512-function-session-2026-10-19.md) | 显式生成/编辑接入、行数选择和完整请求边界 |

相关记录见 [QKV](../qwen21-qkv/README.md)、[FFN](../qwen21-ffn/README.md) 和
[缓存](../qwen21-cache/README.md) 专题。跨模型及其他工作负载见
[Runtime ANE 归档](../runtime-ane/README.md)。

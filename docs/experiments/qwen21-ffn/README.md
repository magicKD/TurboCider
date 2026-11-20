# Qwen Image 2.1 Base 的 FFN 实验

这里保存 2026-10-18 的 512² Base FFN 对照。各篇完整保留设备、BF16 来源、
请求条件、固定图约束和对应的性能或画质证据。

| 记录 | 测量范围与限制 |
| --- | --- |
| [固定投影粒度](qwen-base512-ffn-projection-2026-10-18.md) | 大 K 与整 K 的完整 block、40 步文生图和图生图对照，保留无收益结果 |
| [gate/up 与 down 分别隔离](qwen-base512-ffn-mixed-projection-2026-10-18.md) | 两因素完整 block 消融，两种混合策略均未取得新增收益 |
| [固定图内的激活表达](qwen-base512-ffn-activation-2026-10-18.md) | SiLU 失败归因、组件与完整请求、反向和质量对照 |
| [encoder 恢复和 GPU 分工](qwen-base512-ffn-handoff-encoder-2026-10-18.md) | 区分交接等待与实际运行，保留 gate/up 合并无净收益的结果 |
| [F32 down 与 FP16 gate/up](qwen-base512-ffn-down-tiles-2026-10-18.md) | 12 种 tile 与精度消融，完整 block 未获得可靠净收益 |

相关对照见 [Base QKV 实验](../qwen21-qkv/README.md)和
[缓存实验](../qwen21-cache/README.md)。复验前先核对原报告的版本、输入和图几何，
按[实验归档约定](../README.md)记录新的设备和测量范围。

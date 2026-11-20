# Qwen Image 2.1 Base 的 FFN 实验

这里保存 2026-10-18 至 10-19 的 512² Base FFN 对照。各篇完整保留设备、BF16 来源、
请求条件、固定图约束和对应的性能或画质证据。

## 固定投影与激活

| 记录 | 测量范围与限制 |
| --- | --- |
| [固定投影粒度](qwen-base512-ffn-projection-2026-10-18.md) | 大 K 与整 K 的完整 block、40 步文生图和图生图对照，保留无收益结果 |
| [gate/up 与 down 分别隔离](qwen-base512-ffn-mixed-projection-2026-10-18.md) | 两因素完整 block 消融，两种混合策略均未取得新增收益 |
| [固定图内的激活表达](qwen-base512-ffn-activation-2026-10-18.md) | SiLU 失败归因、组件与完整请求、反向和质量对照 |
| [encoder 恢复和 GPU 分工](qwen-base512-ffn-handoff-encoder-2026-10-18.md) | 区分交接等待与实际运行，保留 gate/up 合并无净收益的结果 |
| [F32 down 与 FP16 gate/up](qwen-base512-ffn-down-tiles-2026-10-18.md) | 12 种 tile 与精度消融，完整 block 未获得可靠净收益 |

## GPU 分工与交接

| 记录 | 测量范围与限制 |
| --- | --- |
| [互补激活与输出交接](qwen-base512-complement-share-handoff-2026-10-18.md) | 分工、预测与等待归因，保留 5120 通道较慢的实测 |
| [显式动态权重的共享编译](qwen-base512-compiled-complement-2026-10-19.md) | 完整混合 FFN 的组件筛选，输出一致并保留小样本时间信号 |
| [共享编译的 40 步验收](qwen-base512-compiled-complement-session-2026-10-19.md) | 生成与单参考编辑正反序，组件信号未形成稳健整图收益 |
| [融合 gate/up 与 SwiGLU](qwen-base512-dual-gup-block-2026-10-19.md) | 合成 oracle、真实 BF16 权重完整 block 与来源重放 |
| [固定图下融合 GPU 分工](qwen-base512-gpu-dual-runtime-2026-10-19.md) | 40 步匹配请求与 RGBA 逐字节对照，保留未达默认采用门槛的结果 |

## 旋转、归一化和混合精度

| 记录 | 测量范围与限制 |
| --- | --- |
| [H512 旋转因式分解](qwen-base512-h512-factorization-2026-10-19.md) | 固定 FP16 算子的延迟和逐点误差门，保留八个未采用的候选 |
| [归一化倒数与相邻通道分工](qwen-base512-hidden-quant-neighbor-share-2026-10-19.md) | 倒数表达和 5632/6656 通道筛选均无稳定收益 |
| [FP16 hidden 与 BF16 down 权重](qwen-base512-mixed-hidden-down-2026-10-19.md) | 保留原权重布局的 down 投影重放，没有稳定整块收益 |

相关对照见 [Base QKV 实验](../qwen21-qkv/README.md)和
[缓存实验](../qwen21-cache/README.md)。复验前先核对原报告的版本、输入和图几何，
按[实验归档约定](../README.md)记录新的设备和测量范围。

# Qwen 三参考图：runtime LoRA 的长序列瓶颈

接续 [1024² base 对照](runtime-ane-qwen-1024.md)。本轮补齐统一测试工具的
真实三图编辑证据，而不是把文生图或缩小参考后的另一条路线冒充编辑加速。

## 匹配的 GPU/runtime ABBA

M4 Max 64 GB，Qwen-Image-2.1 BF16 base、Viggle v0.2.1 r256 原始 adapter，
`inference_time`、6 步、seed29、512² 输出、三张参考都按同一 ref512 策略处理，
resident，未开启 GPU FP16-LoRA 或其他额外实验。LoRA 没有写进 checkpoint、
Core ML artifact 或 runtime base slots；runtime 使用 v2 c288/t1024 图。

原生库 SHA：`bdc78921f6b07c170faae8d66fd3f57ff137fa94a87cfe3c2613bf0ea0c3936c`。
GPU → runtime → runtime → GPU，每 trial 独立 resident batch，一冷两热。
完整提示词、按顺序的参考 SHA256、请求、JSONL、PNG 和 preflight 在：
`outputs/runtime-ane/qwen-edit3-ref512-runtime-v2-auto-abba/`。

| 路线 | 全部热请求（秒） | 热中位 |
| --- | --- | ---: |
| GPU | 12.954017 / 12.940136 / 12.902591 / 12.928569 | 12.934353 s |
| runtime c288/auto | 14.352766 / 14.054341 / 14.079771 / 13.788860 | 14.067056 s |

GPU/runtime 为 **0.919×**，runtime 慢 **8.8%**。两路实际 `reference_tokens`
均为 3072，操作均为 `image.edit`；summary 完成、无错误回退，LoRA 正常绑定。
这里没有冻结 `lora_fused` 对照，不与历史 component-staged 的编辑时间混比。

runtime 两个 trial 最后一次热请求分别只有 10/5 个 hybrid block，其余
182/187 个 block 已走 GPU。预测没有报错，但不是“持续全程 ANE 加速”。
它仍有 gate/up、hidden/down 交换、block 同步和 adapter 身份校验成本。

已查看 GPU 与 runtime 最后 PNG：米色茶壶在左、蓝色茶壶在右、橙色龙贴纸
居中，暖光木桌和主体细节接近，没有明显新增崩坏。不过提示词要求的蓝壶
透明质感在两路都没有很好保留；这是本样本的编辑质量限制，不能把“与 GPU
接近”写成“全部指令完美完成”。一组样本也不代表广泛参考身份质量。

## 为什么加长 sequence 仍不够

同一 build 的单独诊断在
`outputs/runtime-ane/qwen-edit3-ref512-runtime-v2-profile/`，一冷六热、开启
profile。诊断时间不能与未开启 profile 的 ABBA 混成正式速度结论。

实际 FFN 有两种形状：

- 首步 prefill：4226 rows（包括图像/参考及文本部分），每张图每层仅一次。
- 后续 decode：1024 rows，参考条件缓存后不再每步重复整条长序列。

调度按 layer/rows 隔离。最初两次 hybrid 之后还要取两次 GPU probe；
所以 prefill 的调参横跨多个**请求**，而不是一张六步图就一定收敛。
诊断第 5 次请求才开始实际使用 3–4 chunks，而最初 ABBA 每个 session
仅三次请求。不能因为早期没有 multi-chunk 就认为实现不支持循环。

诊断累计 profile 样本的完整 block 中位（不同层和 warmup 混合，仅诊断）：

| rows / 路径 | 样本数 | block 中位 |
| --- | ---: | ---: |
| 1024 / GPU probe | 96 | 45.009 ms |
| 1024 / 1 chunk | 74 | 52.485 ms |
| 4226 / GPU probe | 64 | 172.279 ms |
| 4226 / 1 chunk | 64 | 182.618 ms |
| 4226 / 4 chunks | 62 | 168.454 ms |

prefill 多 chunk 的名义收益约 2.2%，不是整请求 2.2%；六步里只有一个
prefill，而短 decode 反而更慢，故不少层关闭。第 5–7 次诊断请求仍约
14.13–14.22 s，没有显示跨过 GPU 的趋势；这不构成无 profile 的收敛验收。

因此不能仅优化 ANE matmul 或强制更多 chunks：需要同时处理串行修正、
小 shape 的边界成本，以及不削弱身份校验前提下的短请求固定开销。
既有 adapter SHA 热中位约 0.56 s 是独立历史测量，不能据此从本组结果
直接减掉一个常量当作“优化后成绩”。

## 本轮新增可观测性

在 `HybridFfn` 增加累计计数和逐块 profile 字段：

- `lora_gate_up_seconds_session_total` / `lora_gate_up_seconds`：GPU tail 的
  gate/up 修正、contiguous/eval 和输入准备，ANE launch 之前的串行窗口。
- `post_join_seconds_session_total` / `post_join_seconds`：join 后的输出
  恢复、可选 down-LoRA、拼接及既有 output-ownership eval。失败时可能包含
  GPU tail 重算，必须结合 failure counters；不是纯 down-LoRA kernel 时间。

没有新增 eval、线程或依赖链，不改变 SiLU/LoRA 顺序、数值计算、分区策略
或默认路由。验证工具接受旧报告，对新字段检查有限、非负、累计不回退，
且两段合计不超过 hybrid FFN 总窗口。微图测试覆盖无 adapter 时 gate/up
计时增量为零、有 adapter 时非零以及整体时间包含关系。

下一步应先量出这些窗口，再验证针对性优化；不能用单次时差推断物理
ANE/GPU overlap，更不能用缺少串行项的理论吞吐保证整请求加速。

## 新字段实测：串行项不能忽略

原生库 `58d6f005489ec8dc07811e170485288304b29296c58beed8b4a7d2512c09e562`
构建完成后，等待全部构建/测试退出，再运行同一 workload 一冷六热 profile：
`outputs/runtime-ane/qwen-edit3-ref512-runtime-v2-spans-profile/`。
summary 完整、实际参考 tokens 仍为 3072，无错误回退，最后 PNG 已查看，
主体/构图和前述 GPU 图接近，同样没有恢复好蓝壶透明材质。

以下为本轮所有相应 profile 样本的中位，不是设备 kernel 时间：

| rows / chunks | 样本数 | gate/up 串行 | post-join 串行 | FFN 窗口 | 串行占比 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1024 / 1 | 97 | 2.905 ms | 2.490 ms | 31.019 ms | 18.8% |
| 4226 / 1 | 64 | 4.552 ms | 6.736 ms | 110.555 ms | 10.4% |
| 4226 / 4 | 76 | 10.567 ms | 7.601 ms | 98.831 ms | 19.0% |

占比先逐样本计算 `(gate_up + post_join) / ffn` 再取中位，不能用表中三个
独立中位数相除复算。另有两条 3-chunk 样本，不作为稳定形状结论。
4-chunk 并行分支的 GPU/ANE host spans 中位约 79.591/71.439 ms；两者
看似平衡，完整 FFN 却还要付修正/恢复费用。post-join 包含内存处理和拼接，
不能把 7.601 ms 全称为 down-LoRA 算力损失。

本轮六次热请求为 14.140624 / 13.741185 / 13.786060 / 13.641288 /
13.688574 / 13.834177 s。没有改调度或算法；相较旧 profile 的变化伴随
不同的 auto 决策，例如末请求为 18 hybrid / 174 GPU，因此**不能称新增
计时让模型更快**。当前还需要无 profile、同配置的针对性优化对照。

下一步优先针对 gate/up 到 ANE 的依赖与 post-join 数据/低秩计算；调度
成本模型若纳入串行项，也要区别固定成本和随 rows 变化的成本，不能简单
把一个 chunk 的全部固定开销按 chunk 数线性放大。完整 block on/off
仍作为收益门禁，不强制开启已经确认更慢的 decode。

新构建通过 `make test-acceleration-contract`（4 host、27 报告/预检、11 CLI）、
`make test-runtime-ane`（4 host + 8 图/集成）、`make test-qwen21` 和完整
`make test`。全量测试的既有 GPU opt-in、缺夹具/专用构建 skip 不算已覆盖。
Core ML SDK 的临时目录清理警告仍在，但微图测试成功退出。

## 最快 base 路线回归

同一 `58d6f00…` 下另跑 Z BF16 512²/8 步狐狸样本，一冷两热、无 profile，
顺序 frozen → runtime → GPU，原始结果：
`outputs/runtime-ane/z-base-serial-spans-regression/`。

| 路线 | 热请求 | 中位 | GPU/该路线 |
| --- | --- | ---: | ---: |
| frozen | 5.351513 / 5.349151 s | 5.350332 s | 1.310× |
| runtime c288/auto | 6.679258 / 6.594776 s | 6.637017 s | 1.056× |
| GPU | 7.009646 / 7.005525 s | 7.007586 s | 1.000× |

GPU 与 frozen 的最后 PNG SHA 分别和 `bdc7892…` 回归一致，最快 frozen
性能保持接近；runtime 比上一组 6.591926 s 名义慢约 0.7%，样本少且波动
较大，不能称本轮带来速度提升，也不能将全部差额归因为两个新计时。
runtime 图已查看，狐狸主体/构图正常；本轮累计 685 hybrid、558 untimed
hybrid、无错误回退，无 adapter 的 gate/up 计时为 0，post-join 累计
0.152516 s。各 trial 前后系统 swapusage 为 958.38 MiB，swapins/out
计数未变；这只是前后快照，不是峰值归因或设备驻留认证。

249 个既有暂存移除及对应本地文件保持不动，没有提交 commit 或删除产物。
本轮确立了真实编辑的负结果和串行成本证据，**尚未完成**一图/两图、更多
训练 LoRA/提示词、物理 overlap/驻留等验收，不将观测改进当成加速完成。

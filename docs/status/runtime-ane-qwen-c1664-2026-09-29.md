# Runtime ANE：Qwen c1664 的组件收益未传到整请求

2026-09-29，接续[1024² c1792 调优](runtime-ane-qwen-chunks.md)与
[同库三路内存对照](runtime-ane-memory-admission-2026-09-29.md)。本轮只新导出
Qwen graph-v2 c1664/K1024/N512，和现有 c1792 同接口图交错比较。
不修改产品原生库、默认选路、权重、LoRA 或冻结图；**保留 c1792**。

## 单层：c1664 稍快

M4 Max 64 GB，真实 Qwen BF16 block0 FFN 权重，但 4096-row activation 是
探针 seed41 的**合成**输入，不是生成中捕获的激活。两个无模型权重常量的
graph-v2 都传零 LoRA 修正；H4096/F12288、exp SiLU、K1024/N512。
各一个 ANE chunk，c1664/c1792 分别留下 2432/2304 GPU rows，因此同时
改变分区与图形状。按 c1792→c1664→c1664→c1792 运行四个独立进程；
每次两次 warmup 后交错测十次 GPU/并行 FFN，各配置 20 个计时样本。

| Chunk | GPU FFN | Core ML prediction | 并行 FFN | 并行含 staging | GPU head 后 join |
| --- | ---: | ---: | ---: | ---: | ---: |
| c1792 | 82.807 ms | 46.560 ms | 51.512 ms | 54.018 ms | 3.361 ms |
| c1664 | 82.799 ms | 44.800 ms | 50.033 ms | 52.560 ms | <0.001 ms |

c1664 并行窗口快约 2.9%，含 staging 快约 2.7%；probe 没有 attention，
这里的 staging 全暴露，不把差额直接套到整模型。四 trial 均成功、每样本
恰好一次 prediction、无 overflow retry；relative L2 为 c1792 0.002160、
c1664 0.002084，不同 ANE rows 的误差不能直接视为同一数值样本。
独立 probe SHA256：`c992325af819b4a52467fffd50ee47804facbe5336371114aea904a72f05b7a8`。

原始逐样本、身份与 complete 汇总在被忽略的
`outputs/runtime-ane/qwen-c1664-c1792-n512-component-v2-2026-09-29.jsonl`，
驱动为同名前缀去掉 `-v2` 的 `.py`；新图在
`outputs/runtime-ane/qwen-image-c1664-k1024-n512-v2-exp-2026-09-29/`。
沙箱内首次编译因 Core ML 系统临时目录权限失败，获权限后成功；首次组件
预检因 `ps` 沙箱权限失败，只留下 identity 的 incomplete JSONL，已保留、
未当作完成的 trial。

## 同库 1024² base：没有整请求收益

原生库 SHA256 在全程保持
`f3f6b6a05ce6eecb476f314f1fa1fccbdb1d2ef24fc09fdc5b72049b0390ca22`。
Qwen-Image-2.1 BF16 base、狐狸雪景/seed42、1024²、40 步、无 LoRA，
全部显式 runtime、auto chunks、GPU Q/K norm-RoPE 开启、profile 关闭。
按 c1792→c1664→c1664→c1792，各独立 resident trial 一冷两热，
共 12 次请求；计时为包含 VAE/PNG 的 `request_wall`，不含加载/冷请求。
两图 manifest/runner/库哈希绑定在总 summary；不是跨构建比较。

| Trial | Chunk | 冷请求 | 热 1 | 热 2 |
| --- | ---: | ---: | ---: | ---: |
| 0 | 1792 | 150.297 s | 147.887 s | 148.830 s |
| 1 | 1664 | 160.469 s | 148.396 s | 149.300 s |
| 2 | 1664 | 152.359 s | 148.339 s | 149.211 s |
| 3 | 1792 | 151.982 s | 147.680 s | 148.677 s |

四热请求池化中位 c1792 **148.282 s**、c1664 **148.804 s**；c1664
名义慢约 0.35%，没有整请求性能证据支持替换 c1792。样本仍少，不能把
0.35% 解释为精确的硬件劣势或通用结论。两配置同样每 trial 累计
3,648 次真实 runtime prediction、3,648 hybrid blocks、192 正常完整
GPU probe，均无失败、错误 fallback 或 overflow retry；不是路由标记下的
全 GPU 成绩。12 个请求、12 张 PNG、四份子 summary 与总 summary 均 complete。

四条 100 ms 进程树采样流独立重新运行 verifier，合计 18,045 样本，
最大间隙 111.641 ms；全部无新增系统 swap-in/out。首个 c1792 窗口有
33.063 MB 系统 compression，其余三个窗口为零；不能将系统计数归因于
Core ML/ANE，也不能推断 driver/wired 或低内存机型的资格。
进程树峰值 footprint 为 31.480–32.123 GB（十进制），覆盖加载、冷/热
和退出，不含所有外部服务或驱动内部计量。

实际打开反向两路末次热 PNG：狐狸、姿态、松枝、雪地和色调接近。
相同 chunk 的正反向末次 PNG 各自字节一致，不同 chunk 的 PNG 哈希不同；
肉眼检查不要求逐像素一致。单个 prompt/seed 不构成
更多提示词、编辑或 LoRA 的质量资格。

本地原始请求、stdout、PNG、采样流与 complete 报告：
`outputs/runtime-ane/qwen-base-1024-c1664-abba-2026-09-29/`；一次性驱动：
`outputs/runtime-ane/qwen-c1664-model-abba-2026-09-29.py`。
冻结图/GPU 未参加本轮 chunk ABBA，不把其他 campaign 的分母拼接进表。
组件负结果与整模型负结果均保留；产品 c1792 图、512²最快冻结图及默认
选择不变。物理 ANE 驻留、Q4_1、低内存/driver、编辑和广泛视觉仍未验收。
本轮未改产品代码、未重建原生库或重跑完整 `make test`。文档更新后
`make test-acceleration-contract` 沙箱外退出 0：5 host、10 placement、
42 screen、6 memory screen、17 switch、12 repository 检查；
`git diff --check` 也通过。

# 本机组件对照：编译融合、量化权重与 fused SDPA

[GPU 导航](README.md) · [全部样本/环境/哈希](experiment-evidence-20261016.json) · [可运行探针](../labs/metal/engine_probe.py)

日期：2026-10-16。M4 Max，40 GPU cores，64 GiB；macOS 26.6.2，MLX wheel 0.32.0。两次独立进程/seed（17、29），每次 14 个比较、每比较 8 个 ABBA block，共 **896 个正式计时输出**。这是合成 resident 组件实验，**不是引擎排行榜、真实模型 E2E、GPU-only 或 M5/NAX 测试**。

## 合同、原始证据与失败记录

每臂第一次调用单列（不称全进程冷启动），随后各 3 次预热；每 block 顺序 A/B/B/A，每次调用构造计算并等待 `mx.eval`，CPU 拷贝与验证放在计时外。每个正式输出都全量检查 finite、shape 与 CPU FP64 参考。比值是每 block 的 A 均值 / B 均值再取中位，不是表中两列独立中位数的商；全部 8 个比值和原始样本均保留。

CPU 参考由 NumPy `einsum(optimize=False)` 做 FP64 直接求和，不经 BLAS matmul。affine Q4/Q8 由独立 CPU uint32 unpack+group scale/bias 解释，非同一个 GPU decoder 自比；softmax 在 CPU FP64 做全 Q/K/V。三个 family 的门槛预设为 relative L2：norm-chain ≤1e-5、affine ≤0.02、SDPA ≤0.005；本轮 checks 最大值约 9.483e-4。门槛是组件数值预算，不是生成质量标准。

第一轮 BLAS 参考发出 divide/overflow/invalid 警告，虽 finite gate 通过仍排除；另两份 warning-free 诊断因随后修改日期 metadata 不进正式表。三份前序 raw JSON 的哈希、版本与排除理由都在 capsule，不把它们悄悄当作不存在。正式两次使用 `-W error`，无警告通过，probe SHA256 为 `3a36b57e3bb0bb40944671c61217c16718b3037984bc1e47a7f33cf3e74ad7f2`。raw 输出在 ignored 的 `outputs/metal-survey-20261016/`，可分发 capsule 不包含机器路径、序列号或输入权重。

## 全部正式结果

单位 ms；下面时间为 seed17，最后两列分别为 seed17/29 的 paired 倍率。倍率 >1 表示 candidate 更快，不表示统计显著或可直接采用。

| 比较 | A median ms | B median ms | paired seed17 | paired seed29 |
| --- | ---: | ---: | ---: | ---: |
| compile-rms-chain-M1 | 0.1776 | 0.1526 | 1.161× | 1.158× |
| compile-rms-chain-M128 | 0.2100 | 0.1683 | 1.277× | 1.240× |
| compile-rms-chain-M1024 | 0.3279 | 0.2476 | 1.317× | 1.349× |
| affine4-packed-vs-resident-M1 | 0.1477 | 0.1515 | 0.996× | 0.944× |
| affine4-packed-vs-resident-M16 | 0.1681 | 0.1814 | 0.934× | 0.954× |
| affine4-packed-vs-resident-M256 | 0.2579 | 0.2598 | 1.010× | 1.037× |
| affine4-packed-vs-resident-M1024 | 0.4864 | 0.5136 | 0.950× | 0.945× |
| affine8-packed-vs-resident-M1 | 0.1186 | 0.1216 | 0.972× | 1.002× |
| affine8-packed-vs-resident-M16 | 0.1556 | 0.1635 | 0.952× | 0.962× |
| affine8-packed-vs-resident-M256 | 0.2578 | 0.2573 | 1.011× | 0.997× |
| affine8-packed-vs-resident-M1024 | 0.4976 | 0.5284 | 0.947× | 0.946× |
| sdpa-Q1-K4096-D64 | 0.2533 | 0.1901 | 1.341× | 1.388× |
| sdpa-Q512-K512-D64 | 0.3193 | 0.2295 | 1.417× | 1.382× |
| sdpa-Q192-K257-D80 | 0.2392 | 0.1987 | 1.195× | 1.190× |

每行 baseline/candidate、shape、dtype、第一调用、误差和样本在 capsule 中完整列出，不只保留最快格子。

## 三组结果各自能说明什么

**编译融合**：A 是 eager/lazy FP32 RMS 归一化→scale→tanh→residual 表达，B 是同一函数 `mx.compile`。本组约 1.16–1.35×，包含 host 构图/提交的减少；没有 capture/counter，不能断言全部来自某个 shader 融合或 GPU 指令。M1 小负载尤其受固定提交与 host 开销影响。这不是完整 FFN，更不是完整 denoiser 的同幅度收益。

**量化 packed vs resident decode**：A 为 MLX affine group64 `quantized_matmul`，B 在计时前将**同一份编码权重**展开为 FP16 后 matmul。固定 N=K=1024，M=1/16/256/1024；大多数格子 B 持平或慢一些，M256 小差异/配对与独立中位数方向不同也全部保留，不宣布赢家。权重较小、可驻留 cache，不能外推到大模型 streaming 的带宽瓶颈。

Q4 的 codes+scale/bias 为 0.5625 MiB，Q8 为 1.0625 MiB；resident decode 另加 2 MiB。它没有比较原始未量化模型质量，decode setup 不计入 B，两个 arm 共享准备好的分配。逻辑 bytes 不是隔离的进程/GPU 峰值；新增 cache 与 source/LoRA invalidation 仍需单独资格。

**SDPA**：A 为 materialized QK→FP32 softmax→FP16 P→PV，B 为 `mx.fast.scaled_dot_product_attention`；maskless、GQA1、FP16，测试 decode-like Q1/K4096 与两组 prefill（含 D80/K257 tail）。约 1.19–1.42×，但实际 dispatch 没有 capture，不凭函数名证明每行用了哪种 kernel。全部参考为 full-output CPU FP64，数值通过仍不覆盖 causal、empty-row、GQA、量化 KV、训练或真实视频。

## 与真实消费者的关系

[真实 Qwen affine FFN](../review/gpu-affine-20261016.md)和[真实 Qwen attention](../review/qwen-attention-20261016.md)是更强但不同边界的证据；它们不被此处的小合成结果替代。已有 Qwen 数据显示 resident decode 的小收益不一定超过原始 BF16 图，attention 的看似合理重排也未必改善完整消费者。当前不修改任何生产默认。

本实验未锁频、未测功耗/DRAM/register/NAX counters、未隔离每臂峰值、未做跨框架 PyTorch/MNN/Candle 对照。ABBA 能缓和漂移但不能消除系统背景/CPU 检查间隔的影响；8 个 block 也不是全面统计证明。不提供外推的 tok/s、TFLOPS 或“最优引擎”。

## 复现

在已安装 NumPy/MLX 的 macOS 环境，从仓库根目录运行；脚本不安装依赖、不下载模型，拒绝已有输出：

```sh
.venv/bin/python -W error -B techniques/labs/metal/engine_probe.py \
  --seed 17 --blocks 8 --output outputs/metal-survey-replay/seed17.json
.venv/bin/python -W error -B techniques/labs/metal/engine_probe.py \
  --seed 29 --blocks 8 --output outputs/metal-survey-replay/seed29.json
python3 -B -m unittest discover -s techniques/tests -p test_metal_survey.py -v
```

重放时记录自己的 core binary hash、OS/device、probe hash 和全部结果；不要期待跨环境逐 ms 重现。单元测试验证 ABBA 口径、NaN/缺失拒绝、错误 nibble/group 负控与 capsule 完整性；无 NumPy 的标准库环境会显式 skip 两个可选 CPU packing 测试，不谎称执行过设备测试。

## 可执行学习控制（不替代上游验证）

本页实现参考：[labs/metal/engine_probe.py](../labs/metal/engine_probe.py)；case：`mlx:engine-mechanisms`。CPU/旧 Metal 控制只验证对应数学与合同；`mlx:engine-mechanisms` 对应[显式组件 CLI](experiments-20261016.md)，不是自动启动 GPU 的普通单元测试，也不证明全部上游引擎或 PR 已运行。设备、source、shape、精度或消费者改变后重新验证。

# Qwen 三权重打包 QKV：runtime-weight 组件筛选

Q 单投影有长序列正信号后，本组使用同一 BF16 checkpoint 的 block-0
`attn.to_q/k/v.weight` 三个真实 `[4096,4096]` 矩阵，按 Q/K/V 顺序
**在计时前**打包成 `[12288,4096]`。图不含 checkpoint 常量；runtime
MatMul 输入是打包权重。合成 4096-row BF16 激活，ANE 1536 rows、GPU
2560 rows；K1024/N512。两种 GPU 对照均用相同的打包权重与输入，
输出为完整三投影矩阵，等 GPU head 与 Core ML tail 拼接就绪才停止计时。

每种 GPU 对照两独立进程、各排除两个 warmup 后交替测十对，即池化20对
的中位数；输入、图、checkpoint 和 probe 的 SHA 与逐样本、预检进程、
系统内存前后快照保存在被忽略的 JSONL。单位毫秒：

| GPU-only 对照 | GPU-only | 并行投影 | 含 staging | GPU / 含 staging |
| --- | ---: | ---: | ---: | ---: |
| 一次打包矩阵 GEMM | 27.409 | 18.422 | 19.361 | 1.416× |
| 三次 Q/K/V 独立 GEMM + concat | 27.931 | 18.691 | 19.639 | 1.422× |

两组都正常退出、40 对样本、无 overflow retry，relative L2 均为
0.000985；两组 `summary.status=complete`。后一种 GPU 对照从同一打包
张量的三个切片分别执行 GEMM，比单个打包 GEMM 更接近当前普通产品路径，
但仍不是完整产品 block。曾在两个打包 GEMM 试次之间发现其他推理忙碌，
预检等待至空闲后继续；没有逐时段 GPU 独占证明。

证据及入口：

```text
outputs/runtime-ane/qwen-qkv-packed-c1536-k1024-n512/
outputs/runtime-ane/qwen-qkv-packed-screen.py
outputs/runtime-ane/qwen-qkv-packed-screen.jsonl
outputs/runtime-ane/qwen-qkv-separate-gpu-screen.py
outputs/runtime-ane/qwen-qkv-separate-gpu-screen.jsonl
```

两轮 probe 的 SHA256 分别为 `632ef34447a5587424f4aa3bf47703e37cccfe40ad2b7a36d9e0e5368fcf4a34`
与 `3548d7a32ae5f8cd65e134a8cc815f8b089f62a8352803499765321908d8b438`；
最终 `make test-runtime-ane` 的 5 项 host、12 项 Core ML/MLX 集成回归通过。
当前产品库 `28ff6aaf…` 未重建、默认路线未修改。

**这不等于已实现产品 QKV tier**：打包/分配在计时前发生，若为32层
保留 BF16 `[12288,4096]` 副本，额外权重理论值约 3 GiB；还需
实际内存准入、转换时机和生命周期设计。Q/K norm、RoPE、attention、
FFN 的 ANE 串行竞争及完整请求耗时均未测，`cpuAndNeuralEngine` policy
不证明物理 ANE 驻留。下一门槛是避免永久双份权重或计入真实打包成本，
在 Qwen block 边界测完整 QKV+norm/RoPE 与 GPU-only，再做内存/请求级
对照；若完整窗口没有正收益，应保留现有产品路径。

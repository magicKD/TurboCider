# M5 GPU / ANE 优化记录

2026-09-16；M5 Pro、16 核 GPU、24 GiB、macOS 26.4.1、MLX 0.32.0。

## 优化点

- **只读 GPU 后缀**：Z-Image streaming 中，ANE 计算 MLP 前 4096 通道，GPU 只加载后 6144 通道。w1/w3 跳过前缀，w2 在会话首次准备时整理为临时后缀文件，减少重复读取并保留更多层。
- **GPU 前后段编译**：分别编译 ANE 交接前的注意力/输入准备、交接后的合并/归一化/残差，减少调度开销；中间继续并行执行 GPU 后缀与 Core ML 前缀。
- **Core ML 输出拷贝**：同步上游的变长输出安全处理后，将输出指针和步长移到循环外读取，连续行批量拷贝，避免逐元素访问框架属性。
- **客户端自动选行分区**：按当前图片尺寸和实际文本长度，在相同通道划分中选择最小可用行桶，自动发现、按需编译及登记。修复从 1024 切回 512 后仍沿用大分区的问题。
- **小内存管理与 M5 策略**：约束 Z-Image 闲置缓存，换提示词前释放图像工作集；新增实测通过的 M5 Pro 24 GiB FLUX 自动混合案例。

## 保留的实验结果

以下均为完整生成含 PNG 导出的热态中位数；各行是独立实验，不能跨行拼接比较。
Z-Image 为 Comfy BF16、无 LoRA，streaming 的 GPU 预算均为 10 GiB。
性能数值对应各 JSON 中记录的历史构建；合并上游后的额外验证单独记录，不混入原采样。

| 实验 | 条件 / 每路线热态样本数 | 优化前 → 后 | 结果与证据 |
| --- | --- | ---: | --- |
| FLUX 自动混合 | 512²、4 步、resident / 18 | GPU 1.784 → 混合 1.524 s | 耗时 −14.6%；[数据](validation/flux2-m5pro-2026-09-16.json) |
| Z-Image resident 调研 | 512²、9 步 / 12 | GPU 6.948 → 混合 7.576 s | 混合慢 9.0%；[数据](validation/z-image-m5pro-2026-09-16.json) |
| 后缀优化前的 streaming | 512²、9 步 / 8 | GPU 10.953 → 混合 11.936 s | 未见收益；a6144 未过质量门槛；[数据](validation/z-image-m5-streaming-2026-09-16.json) |
| GPU 后缀加载 | 512²、9 步 / 8 | 原混合 10.773 → 7.217 s | 耗时 −33.0%，应用权重读取量 55.36 → 26.48 GB；同轮 GPU 10.747 s；[数据](validation/z-image-m5-gpu-suffix-2026-09-16.json) |
| GPU 前后段编译 | 1024²、8 步 / 4 | 原混合 34.666 → 33.476 s | 耗时 −3.4%；同轮 GPU 28.895 s，混合仍较慢；[数据](validation/z-image-m5-hybrid-fusion-2026-09-16.json) |
| 512 融合复核 | 512²、8 步、1120 行 / 4 | 原混合 7.338 → 7.119 s | 耗时 −3.0%；同轮 GPU 10.578 s；[数据](validation/z-image-m5-512-partition-routing-2026-09-16.json) |
| 512 分区选择修复 | 同上、新融合路径 / 4 | 误用 4128 行 12.387 → 1120 行 7.119 s | 耗时 −42.5%；两个分区实验分别预热；[数据](validation/z-image-m5-512-partition-routing-2026-09-16.json) |

[1024 分段计时](validation/z-image-m5-1024-profile-2026-09-16.json) 表明 MLP 并行有收益，但整图仍受交接前后处理和内存压力影响。后续编译优化将主层前段 60.478 → 55.028 ms、后段 3.266 → 2.090 ms，并行窗口基本不变。512 误用大分区时 Core ML 调用由 8.147 增至 30.001 ms，是明显慢速的主要原因。

## 验证与边界

- FLUX 24 对、Z-Image resident 18 对图片通过原有近似质量门槛；后缀优化 10 对、融合验证 26 次生成及 512 复核 40 次生成中，同一混合路线的优化前后 PNG 逐字节一致。这不表示 INT8 混合与 BF16 GPU 输出相同。
- 原生后缀读取、缓冲区复用、取消/重试、路由切换测试通过；Swift 回归及真实分词器的 512 → 1024 → 512 分区选择通过。`make test` 的 VDN 精度测试在修改前库也失败，未放宽阈值。
- PR 同步 `dev@ad343d4` 后，原生/Swift 构建、78 项契约测试（3 项跳过）及客户端回归通过；额外 6 次 512 生成验证新旧图和输出拷贝修复前后 PNG 一致。记录附在 512 验证 JSON 的 `pr_integration_validation` 中。
- Z-Image M5 的 `auto` 仍选 GPU；显式 ANE streaming 支持 Comfy BF16、无 LoRA。FLUX 自动案例仅限 M5 Pro 24 GiB、512²、4 步、resident、无 LoRA/输入图、1025–1088 tokens、a6144/b1088，且需允许近似及有效本机分区。
- 首张包含模型加载、编译及后缀整理；后缀临时文件约 1.41 GiB，两次准备为 0.385 / 0.236 s，随会话释放。10 GiB 是 GPU 采样规划预算，不含 Core ML，也不是整进程内存上限。
- 未清空系统缓存，存在桌面负载及换页波动；打点是含同步的主机耗时，不能当作纯内核时间。1024 测试的 VAE 后 MLX 峰值约 17.35 GiB。结论不推广至其他 M5 配置，也不承诺零 swap。

## 复现与产物

使用 `tools/native/benchmark_z_image_streaming.py` 比较驻留方式，使用下面的驱动复测融合或分段计时；路径替换为本机配置，每次使用新输出目录：

```sh
.venv/bin/python tools/native/profile_z_image.py --library build/native/libturbocider.dylib --model "$TC_MODEL" --manifest "$TC_MANIFEST" --prompt-file "$TC_PROMPT_FILE" --size 512 --steps 8 --budget-gib 10 --experiment fusion --output outputs/m5-recheck
```

`--experiment stages` 复测分段计时；`TURBOCIDER_Z_HYBRID_EAGER_SEGMENTS=1` 切回原混合图。打点默认关闭。FLUX 准备步骤见[指南](../public/FLUX_PREPARATION.md)。

7 份验证 JSON 保留全部采样、版本/哈希、失败记录和限制，路径已归一化。本机 `outputs/*m5*20260916/` 保留原始计时、日志及图片；重复图片共用存储。未采用/已替代的分区权重和临时构建已清理，清单及退役分区元数据位于 `outputs/m5-experiment-cleanup-20260916/`；客户端登记的三个分区保留。

# Comfy Qwen FP4 mixed 与 INT8 GPU/Core ML 内存实验

日期：2026-09-10。机器：M4 Max Mac Studio，64 GiB；MLX 0.32.2。
本次为隔离实验，没有修改 App 的模型加载器或重新打包 App。

后续已完成 12 对最终图片，以及 VAE 变慢的阶段/缓存对照，见
[最终出图与加速复查](QWEN_FP4_IMAGE_QUALITY_2026-09-10.md)。后续同口径
限制空闲缓存的测试恢复约 1.20×；下文未限制缓存的旧读数保留为历史证据。

## 下载与直接运行

从 ModelScope `Comfy-Org/z_image_turbo` 下载
`split_files/text_encoders/qwen_3_4b_fp4_mixed.safetensors`。
文件 3,479,416,193 bytes（3.48 GB），SHA-256 与仓库公布值一致：

`7ca32dcf07dfe7692945d80fff86e3a74cb83c6206b9b223ac6836b939bb85d6`

真实 `comfy_quant` 元数据包含 189 个 `nvfp4` 投影、58 个
`float8_e4m3fn` 投影，其他权重包括 BF16 embedding/norm/部分投影。
MLX `load` 可以读取文件，但将 FP8 数据及 NVFP4 打包数据都表示为 uint8。
文件可读不代表模型可以直接执行。

未修改的 `build/native/qwen3-quant-probe` 实际失败：

```text
NVFP4 weights must be packed before projection: model.layers.0.self_attn.q_proj
```

## 不重新量化的 MLX 适配

隔离脚本 `tools/validation/qwen_mixed_probe.py` 做了以下处理：

1. NVFP4：高 nibble 在前的 U8 改为低 nibble 在前的 U32；cuBLAS 分块
   E4M3 scales 改为 MLX row-major group-16；保留原始全局 scale 乘数。
2. FP8 的最终路径：原始 E4M3 字节打包成 U32，添加值为 127 的 E8M0
   group-32 scales（代表 1），保留外部 Comfy tensor scale。这样可调用
   MLX `quantized_matmul(mode="mxfp8")`，没有重新量化 FP8 数值。
3. BF16 权重不做压缩转换。

这是原生 MLX 低位**权重**矩阵乘，不是直接把原 checkpoint 塞入 App，
也不是 NVIDIA W4A4。沿用现有 Qwen 的 FP32 residual/投影输入契约，
不声称这轮文本编码采用 W4A16。没有把整个模型展开为 BF16。

### 正确性检查

- 同一脚本跑 BF16，在 128-token 确定性 IDs 上与此前 C++ probe 的
  conditioning 逐元素一致（max abs / relative L2 = 0）。
- 三个真实 NVFP4 投影的独立 NumPy FP32 解码，与适配后 MLX 解码
  逐元素一致；实际 packed matmul 相对解码矩阵参考的 relative L2
  为 0.98e-6 至 1.24e-6。
- 一个真实 FP8 投影独立 E4M3 解码，与 `from_fp8`、MXFP8 重打包后
  的解码都逐元素一致；实际矩阵乘 relative L2 为 3.94e-7。
- 最终 mixed conditioning 有限；对 BF16 的 cosine 为 0.999276（32
  tokens）/ 0.997992（128 tokens），128-token relative L2 为 0.063882。
  这是编码数值偏差，不是最终图像画质评分；本次没有用 mixed 文本生成图片。

## 同口径文本测试

相同 Python/MLX conditioning 实现；32 或 128 个确定性 token IDs，取第
35 block 后、最终 RMSNorm 前输出；无 padding。1 次预热、3 次暖测量；
每次明确求值、清除空闲缓存并重置 MLX 峰值。纯编码时间排除加载、分词、
导出和 prompt cache，包含 Python 调度；各配置单独进程串行执行。

| 配置 | 权重驻留 GiB | 128-token MLX 峰值 GiB | 暖编码中位数 ms |
|---|---:|---:|---:|
| BF16 | 7.492 | 7.914 | 156.88 |
| 本地 affine Q8 g32 | 4.532 | 4.597 | 99.46 |
| 本地 affine Q4 g32 | 2.840 | 2.905 | 97.69 |
| Comfy FP4 mixed，FP8 按投影解码 | 3.240 | 4.225 | 158.05 |
| Comfy FP4 mixed，FP8 预展开为 BF16 | 4.258 | 4.679 | 116.18 |
| **Comfy FP4 mixed，FP8 无损打包 MXFP8** | **3.272** | **3.595** | **99.65** |

最终低位路径在 32 tokens 下为 38.86 ms，峰值 3.570 GiB。
128-token 峰值比 BF16 少约 54.6%，比 affine Q8 少约 21.8%，占本机
64 GiB 的约 5.62%；这里只是文本阶段 MLX 占用，不是完整 App/整图内存。
它比 affine Q4 多占约 0.69 GiB。本测试中速度与 affine Q8 接近，不能
凭亚毫秒差异排名；数值保真不如 Q8、优于 Q4（限这组确定性 IDs）。

最终路径两次独立进程的 load+materialize 为 0.087 / 0.529 s，布局适配
为 0.065 / 0.107 s。未清空 OS 文件缓存，不能当作稳定的 SSD 冷加载结果。
FP8 展开路径到低位路径的优化显著影响速度与内存，说明“FP4 mixed”文件名
本身不足以预测运行性能。

## INT8 图像模型加 GPU/Core ML（ANE 配置）

### 分区大小及是否省掉 GPU 权重

使用已有、与 INT8 ConvRot checkpoint SHA 匹配的
`models/coreml/z_image_convrot_native_a4096_s8_o32_b4128_all/` 产物。

- 32 个 block，hidden=3840，FFN intermediate=10240。
- Core ML 分担 `[0,4096)`：FFN 通道的 40%；GPU 执行其余 6144 通道，
  attention 等仍在 GPU。这不是整个图像模型 40% 都交给 ANE。
- 三个 FFN 主矩阵的分区参数量为
  `32 × 3 × 3840 × 4096 = 1,509,949,440`，约为整 DiT 参数量的 24.5%。
  单按 INT8 主矩阵估算为 1.406 GiB，尚未含 ConvRot 旋转矩阵与 scales。
- 实际每 block 的 `weight.bin` 为 49,258,048 bytes（46.98 MiB），
  32 个共 1,576,257,536 bytes（1.468 GiB）。全部编译产物共
  1,576,502,220 bytes（1.468 GiB）。磁盘上的 source/compiled-cache
  各有一套，不能相加当作运行内存。
- Core ML 产物为 per-channel INT8 压缩权重，FP16 计算契约。导出时
  从 ConvRot 权重构造分区再压缩，并非直接共享 GPU 的 packed Q8 buffer。
- 当前 GPU 仍保留**完整** INT8 权重；`project_range` 只选择计算范围，
  不会删除 ANE 对应的那段底层权重。32 个 Core ML 分区另外加载。
- Core ML 共享输出 backing 为 `4128 × 3840 × 2`，约 30.23 MiB；
  这是输出缓冲大小，不是全部 Core ML 工作内存。

相关代码：`native/models/z_image/z_image.cpp` 的 `z_hybrid_gpu_suffix`、
`native/backends/mlx.cpp` 的 `project_range`、`native/backends/coreml.mm`
及 `tools/coreml/export_z_image.py`。

### 本次实际运行

1024×1024、9 steps、seed 42、同一个狐狸提示词，INT8 DiT + BF16 Qwen，
无 LoRA，固定 4128-row Core ML 分区。每配置独立进程，1 冷 + 2 暖请求，
暖请求命中文本缓存；先 GPU 后 GPU/Core ML，未控制其他系统负载或清空文件缓存。

| 执行配置 | 暖出图中位数 s | MLX 历史峰值 GiB | 自进程 physical footprint 峰值 GiB |
|---|---:|---:|---:|
| INT8 GPU | 40.887 | 18.163 | 37.527 |
| INT8 GPU/Core ML | 39.791 | 18.192 | 39.160 |

GPU 暖样本为 40.994 / 40.780 s；GPU/Core ML 为 41.283 / 38.300 s。
差异不足以证明稳定加速，不能把之前其他日期约 32.6 s 的结果混入本轮排名。
两张首次输出图像 pixel cosine 为 0.999957；没有作广泛画质验证。

内存读数必须区分：

- MLX 只多了约 30 MiB，看不到全部 Core ML 分配。
- 自进程 physical footprint 多了约 **1.633 GiB**，本轮占 64 GiB 的
  比例约为 58.64% → 61.19%。使用 Mach `TASK_VM_INFO` 的 footprint
  及 ledger peak，25 ms 采样；包括进程计账的设备分配和未归还缓存。
- Physical footprint、RSS、MLX 活跃/峰值不是互斥的内存份额，**不能相加**。
  高水位也不能当作最低物理 RAM 要求。
- 这里只测当前进程，不包含所有其他 Core ML 服务进程，不能称为整机
  总占用，更不能将差值精确归因为 ANE 独占内存。
- Core ML 设置为 `cpuAndNeuralEngine`，公开接口允许 CPU 执行；
  `observed_ane_residency` 仍为 `unknown`，未直接测量 ANE SRAM 或证明
  每个算子均在 ANE 执行。1.468 GiB 是产物大小，不是 ANE SRAM 占用。

因此当前 GPU/ANE 路线是计算分担方案，不是显存/统一内存卸载方案。
更低内存需另行设计删除 GPU 重复分区权重、处理 GPU fallback 与 LoRA
重载的策略，不能只打开 ANE 开关就假定内存减少。本次没有实现此改动。

## 复现与产物

原始结果：`outputs/qwen-mixed-20260910/`，各目录 `report.json`；
`layout-verification.json` 保存独立解码校验；两个图像目录还有
`footprint.json` 和生成的 PNG。新增权重保留在原 ModelScope 组件目录，
没有删除原权重，没有额外保存整个转换后的 mixed checkpoint。

```sh
Python/bin/python3 tools/validation/qwen_mixed_probe.py \
  --weights models/Comfy-Org-z_image_turbo/split_files/text_encoders/qwen_3_4b_fp4_mixed.safetensors \
  --output outputs/qwen-mixed-rerun --tokens 128 --fp8-mxfp8

xcrun clang -dynamiclib -O2 tools/validation/task_footprint.c \
  -o build/native/task-footprint.dylib
```

`benchmark_with_footprint.py` 接受与 `tools/native/benchmark_native.py`
相同的参数，输出目录必须是新的。所有 GPU 测试需允许 Metal 访问。
已通过 Python 语法检查、C helper 编译、上述真实模型编码/解码校验与
两组整图测试。因为未修改生产路径，没有把本次实验说成 App 已支持 mixed。

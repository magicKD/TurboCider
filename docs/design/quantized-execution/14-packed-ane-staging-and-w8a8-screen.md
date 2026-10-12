# 14 · GGUF/ConvRot 直填 ANE 槽与 W8A8 筛选（2026-10-01）

[目录](README.md) · [验收合同](11-acceptance-profiles-and-feasibility.md)

本阶段交付 **A0 的底层 staging API、ConvRot CPU SIMD 转换和 C3 数学 oracle**，
并执行 A2 的公开 Core ML 单投影筛选。不是完整 R4/R5 交付：新 source view 尚未接入
模型级 typed pager/HybridFfn；没有公开量化 ANE 请求资格、硬件 INT8 证明或全模型加速声明。
既有发行库/CLI/App、默认选路、ConvRot/量化执行的产品门禁不变。

## 1. 直接 staging 与数学身份

`native/backends/ane_runtime.hpp` 新增 `GgufView`、`ConvrotView`，扩展内部 `WeightView`。
`RuntimeGraph::stage_weights` 仍由持久 worker 执行、`wait_stage` 收齐转换，全部成功才
发布 staged；失败不允许用上一层的内容 predict。caller 必须持有不可变输入到完成。
这是内部 C++ 接口变化，使用者须与库一起重建，不能拿新头文件链接旧 runtime 二进制。

`ane_runtime_packed.hpp` 的两种转换都直接写已分配的 FP16 surface 行：

- GGUF：已注册 F32/F16/BF16、Q4_0/1、Q5_0/1、Q8_0、Q4_K/Q5_K/Q6_K；原 blocks
  →既有 checked decoder；headroom 在 FP32 重建之后、唯一 FP16 舍入之前施加。
- ConvRot：原 signed I8、原 FP32 per-row scales；按完整 H256 group 做四级整数 Comfy H4
  蝶形，再 `/16 → F32 source scale → headroom → RNE FP16`，不重新量化 codes。
  ARM NEON 同时优化蝶形与最终转换；scalar 是同数学、同二进制的控制。
- source/metadata 有独立 row pitch；校验短 buffer、几何、dtype、checked extent、地址溢出。
  target 必须具有足够的对齐 uint16 容量，拒绝与 source/scales alias；非法数值/FP16
  溢出失败，不 clamp。成功转换不申请 heap，也不生成整张 FP32/dense 中间权重。

ConvRot 的 recipe 为 `convrot-integer-h256-source-f32-scale-headroom-f16-rne-v1`。
整数变换后乘 scale 在实数域等价于逆旋转，但 FP32 舍入顺序不同于先构造 D 再 `D H^T`，
更不等于旧 BF16-scale packed profile；必须另做 N1/媒体资格，不能把这个名字用于原图默认。
LoRA corrections、bias、输入旋转和模型级 source lease 的接入仍需独立 consumer 合同。

## 2. 有界转换并行

新 packed 两种 source 使用 16-row group、最多 8 个活跃转换 callback；原 dense/affine
路径保持既有调度。callback 通过原子索引取下一 group，坏行使其他 callback 停止取新任务，
仍等待已启动任务退出后才返回。dispatch owner 是独立 std::thread，不在转换 pool 内等待自己。

`plan_packed_conversion` 可纯 host 测试：小矩阵串行，零 hardware count 按 1 worker，
大矩阵最多 8。GGUF headroom 分支的大数组 scratch 每 worker ≤2048 bytes；ConvRot
≤1024 bytes；admission estimate 显式增加 8×2048 bytes。此处仅是 kernel 数据 scratch，
**不含 worker stack/GCD 控制、Core ML/driver envelope**，不能据此认证 whole-request hard cap。
没有新增第二套 FFN slots 或无限预取队列。

## 3. 真实 ConvRot CPU 转换性能

M4 Max、macOS 26.6.2。Z layer0 原始 ConvRot w1/w2，source scales 保持 FP32。
每 arm 3 warmups、24 个交错样本；串行转换，包含 inverse H/scale/headroom/FP16 写入，
排除 I/O、分配、Core ML predict、GPU/ANE fork/join。两张整 target 的 scalar/SIMD
bit-exact；每张固定三 row 另与独立 dense Kronecker INT64 oracle 对照。

| 原 tensor `[N,K]` | scalar median | SIMD median | scalar/SIMD |
| --- | ---: | ---: | ---: |
| w1 `[10240,3840]` | 62.580 ms | 12.949 ms | 4.833× |
| w2 `[3840,10240]` | 63.042 ms | 13.014 ms | 4.844× |

这只是 **CPU 转换算子** 的提升，不是 Z-Image 或 ANE 出图倍率。两个 FP16 target 是实验
对照资源，生产 staging 只写既有 surface；回执的 input/target capacity 也不是进程峰值。
没有全 checkpoint content binding，只有真实投影 codes/scales SHA，不能资格化模型。

便携回执：[convrot-ane-staging-20261001.json](validation/convrot-ane-staging-20261001.json)。
保留全部原始样本、源码/二进制 hash、源和目标 hash。复现：

```sh
.venv/bin/python tools/native/benchmark_convrot_staging.py \
  --checkpoint models/Comfy-Org-z_image_turbo/split_files/diffusion_models/z_image_turbo_int8_convrot.safetensors \
  --iterations 24 --output outputs/new-convrot-staging.json
```

## 4. W8A8 数学 oracle 和动态权重筛选

`convrot_w8a8_math.py` 独立 NumPy oracle 覆盖 Comfy H256、per-row maxabs/127 A8、零 row、
ties-to-even、原 weight -128、INT64 dot 与 INT32 上界、GPU FP32 epilogue 的乘法顺序、
per-group partial-dot 各自恢复再累加，以及 normalized FP16 出口/ULP。大矩阵使用
有整数精确范围证明的 FP64 BLAS，并抽样复核 INT64，不链接被测 Core ML 算子。
这不是 M5 primitive 或 GPU A8 kernel，也尚未完成真实 FFN/latent 的 N2/holdout。

`export_runtime_w8a8_probe.py` 导出 FP16、frozen QDQ 和 runtime QDQ 三个独立图；normalized
FP16 I/O、QDQ 常量 scale=1/128，不把 interface 称为压缩 INT8 slots。图内 tile 有尾处理和
4096 MatMul 节点上限；dynamic W 是输入，不捕获 checkpoint。导出/编译/文件身份单列。

`run_runtime_w8a8_proof.py` 使用一张已加载图/arm，至少十轮 A/B/A、5 warmups、30 样本。
A/B 的非均匀 scales 和 bias 在外部 FP32 恢复，分别记录 SHA 与恢复时间；不是图内动态
scale lowering 的证明。实际编译 artifact 不变；host export/load 各 1 次，driver 内部
runtime compile count 仍 unknown。提前准备的 H/A8/normalized arrays 不在 predict 计时内。
各 arm 顺序运行，只有 arm 内换权交错，**不是正式跨 arm 配对 performance campaign**。

最终实测使用真实完整 gate codes/scales，M=256/K=3840/N=10240；激活是 synthetic Gaussian
经 Comfy H256/A8，不是模型捕获的 FFN 输入。coremltools=8.3.0、CPU_AND_NE：

| arm | predict median | normalized 最大 ULP | A/B/A 与 graph 文件 |
| --- | ---: | ---: | --- |
| runtime FP16 控制 | 55.622 ms | 1 | 通过 |
| frozen QDQ 控制 | 7.118 ms | 1 | 固定 W、重复通过 |
| runtime QDQ 候选 | 54.911 ms | 1 | 通过 |

compute plan 的 MatMul preferred device 为 NeuralEngine，但这只是 placement 意图。
运行期 placement、硬件 arithmetic 仍 **unknown**；三个 arm 的 feasibility 均为
`inconclusive_arithmetic_not_observed`。动态 QDQ 未显示值得集成的优势；不能用冻结控制
的速度证明动态 INT8、更不能把它直接推广为 ConvRot 全 FFN 加速。保留 runtime FP16 主线。

另复核 K1024 分片、N512 的 FP16 partial-sum 候选：FP16/runtime QDQ 最大 ULP=12032，
frozen=9600，全部失败；最终工具返回 1，门槛仍为 ≤2 ULP。这里的 FP16 分片归约不满足
一次 normalized 出口的整数 oracle，不能用 A8 模型容差掩盖它，也不改变既有 float tile profile。

便携回执包括完整 raw samples：

- [完整 gate 筛选](validation/runtime-w8a8-full-gate-20261001.json)
- [K-tile 负结果](validation/runtime-w8a8-k-tile-negative-20261001.json)

```sh
.venv/bin/python tools/validation/run_runtime_w8a8_proof.py \
  --rows 256 --hidden 3840 --width 10240 --iterations 30 \
  --convrot-checkpoint models/Comfy-Org-z_image_turbo/split_files/diffusion_models/z_image_turbo_int8_convrot.safetensors \
  --output outputs/new-runtime-qdq-screen
# 负例：width 改512，再加 --tile-k 1024 --tile-n 512，使用新output；预期exit1。
```

screen exit 0 只表示三个 arm 的筛选数据/数值/换权检查可用，不表示 INT8 或产品资格；
1 是观测数值/reuse 失败，2 是 export/执行/必需证据不足。测试拒绝缺 arm、重复 arm、
短样本、无变化 W、stale graph 和假 NE-intent→硬件证明。首次只读 slice reader 也测试
重复 header key、错误 shape/dtype/offset、短文件和 nonfinite scale，不冒充完整 SourceLease。

## 5. 已执行的工程与真实模型回归

- 98 项选定 CPU/图结构/既有 Z 回归全部通过：旧 GGUF gate 4、ANE host 7、W8A8 math 7、
  exporter 3、screen/reader 5，以及 10 个既有 Z 模块 72。
- 固定 GGML oracle/真实 Z Q8/Q4 decoder 24 项通过，含同基类复跑；不是 24 个新格式。
- 新 Core ML direct-slot CPU_ONLY/CPU_AND_NE 两项通过：11 种 type、独立 dense control、
  ConvRot 10 轮 A/B/A、坏 scale→staged=false→launch拒绝→fresh fill恢复，同 graph/surfaces。
- 既有 runtime ANE 图/真实 Core ML/MLX 12 项通过，使用新实验 dylib重编 model-facing
  test；覆盖 LoRA、async output lifetime、失败 GPU 重算、取消、artifact lease/tamper。
- 独立 packed host 程序 ASan/UBSan、strict warnings、成功路径禁止 operator new 通过。
  包括全部 H256 basis、independent INT64 matrix oracle、strides、alias/未对齐 target、
  溢出及 headroom-before-rounding。没有把 sandbox 初始化失败或未启用 hardware 的 skip 计为通过。
- 最新隔离目录完整 native lib 构建通过；未替换 `build/native` 发行库。Python compile、
  bash syntax、`git diff --check` 通过。
- 用最终 dylib 做真实 Q8、512²、seed42、4 steps、`z-mlx-compat-affine-v1` 单/双槽完整请求：
  两次 status0、各120 fills；initial/所有 step/final latent、decoded 和 PNG exact。
  它是 GPU-only 回归，不是新增 GGUF+ANE 模型验收。cold/prompt-cache 不同，墙钟不比较加速。

[GPU 回归摘要](validation/gguf-ane-staging-regression-20261001.json) 保存加载库身份、source
hash、slot/managed统计、所有 tensor/PNG身份和 raw-report SHA。whole-request certified=false。

```sh
TURBOCIDER_BUILD_OUTPUT_DIR=build/quantized-execution TURBOCIDER_BUILD_LIB_ONLY=1 \
  TURBOCIDER_BUILD_EXPERIMENTAL_PROBES=1 TURBOCIDER_BUILD_PYTHON=.venv/bin/python \
  bash tools/native/build.sh
TC_GGML_ORACLE_ROOT=../references/llama.cpp TURBOCIDER_TEST_GGUF_MODELS=1 \
  .venv/bin/python -m unittest discover -s tests/native -p test_gguf_execution.py -v
TURBOCIDER_TEST_RUNTIME_PACKED=1 \
  .venv/bin/python -m unittest discover -s tests/native -p test_ane_runtime_packed.py -v
TURBOCIDER_TEST_RUNTIME_ANE=1 TURBOCIDER_NATIVE_LIBRARY_DIR="$PWD/build/quantized-execution" \
  .venv/bin/python -m unittest discover -s tests/native -p test_ane_runtime.py -v
.venv/bin/python tools/native/run_z_image_gguf_quantized.py \
  --library build/quantized-execution/libturbocider.dylib --model models/z-image-runtime-gguf-q8 \
  --output outputs/new-q8-staging-regression --prefetch 0 1 \
  --precision z-mlx-compat-affine-v1 --steps 4 --size 512 --seed 42 --dump
```

## 6. 仍需推进

A0 的模型 binding/lease/recipe/admission、ConvRot 全 FFN N1/媒体、真正 A8 FFN/holdout、
static S1/S2 artifact 与 graph-bank预算、M5 primitive/实机、动态 ANE arithmetic/placement
都未完成。不扩大 production catalog，也不因为单个 CPU 算子快了就开启自动 ANE。
R0/R1 的完整 campaign、R2 encoder/K真实组件、R3 p=2/source-streamed/tiles/whole-envelope
继续按 09–11 推进；本阶段不缩小原目标。磁盘余量约14 GiB，未删除缓存、模型或历史证据。

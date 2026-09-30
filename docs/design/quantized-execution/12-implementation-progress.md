# 12 · 实施进度：GGUF CPU 解码基础（2026-09-30）

[目录](README.md) · [范围](09-release-scope-and-component-contracts.md) · [验收政策](11-acceptance-profiles-and-feasibility.md)

状态：P1a/P1b 的第一阶段源码与 CPU 验收已交付；**R0 的完整 fixture binding 和 R1 的
有界整图执行尚未完成**。不以这些单元/投影测试替代 48 张图片、全请求内存或 W8A8/ANE 验收。

## 1. 已实现的模块

| 模块 | 实际能力 | 仍不包含 |
| --- | --- | --- |
| `native/core/gguf_directory.hpp` | held fd、v2/v3/little-endian、checked geometry、block 对齐、payload 范围/重叠、重复 key、metadata/数组/目录容量、split 集合一致性 | 全量 tokenizer/config 语义、内容 SHA 校验、任意未知 GGML type |
| `native/core/gguf.hpp` | 旧 MLX gate 使用完整安全目录检查；仍只放行 F32/F16/BF16/Q4_0/Q4_1/Q8_0 | MLX loader 的 path reopen 尚未改成 SourceLease；不开放 K 生成 |
| `native/core/gguf_decode.*` | F32/F16/BF16、Q4_0/1、Q5_0/1、Q8_0、Q4_K/Q5_K/Q6_K → 预分配 F32/FP16/BF16 | IQ4/Q8_1/Q8_K、旋转变换、I/O/pager、GPU decode |
| 同模块 slice/gather | 任意合法 row/column slice、跨 block 切片、positive strides、embedding 重复/乱序 row gather、overlap/容量拒绝 | 无界 row cache 或自动全词表展开 |
| 同模块 Q8 SIMD | Apple ARM NEON：源 FP16 scale → FP32 multiply → BF16 RNE，一次直填，无中间 dense FP32 矩阵 | INT8 GEMM / A8，不能称 W8A8 |

解码成功路径不申请 heap、不保留输入/输出 span，固定 FP32 block scratch 为 1024 bytes。
失败使用固定消息 `DecodeError`，没有 owned string；C++ exception runtime 的控制内存仍
属于后续整体 envelope，不能仅据该 API 声称 whole-process hard cap。非有限源、目标
FP16/BF16 溢出、非法几何/stride、source/target/gather-index overlap 显式失败。

`DecodeOptions.use_simd=false` 提供相同 API/布局/数值的 scalar 控制；非 ARM、切片
边块、非连续或未对齐 target 安全走 scalar。receipt 报 processed bytes、written bytes、
scratch 与真正执行的 SIMD block 数，不把 I/O bytes 混入 decoder 计数。

Q4/Q5/K 的解码表达式与格式参考按固定 GGML 快照适配，MIT 声明见
[第三方说明](../../../THIRD_PARTY_NOTICES/ggml-quantization.md)。没有复制 Unsloth Studio
代码，没有生产依赖 `../references` 或外部推理引擎。

## 2. 实际 CPU 验证

```sh
TC_GGML_ORACLE_ROOT=../references/llama.cpp \
TURBOCIDER_TEST_GGUF_MODELS=1 \
.venv/bin/python -m unittest discover -s tests/native -p test_gguf_execution.py -v

.venv/bin/python -m unittest discover -s tests/native -p test_native_gguf.py -v
```

第一条实际运行 22 项通过，其中同一基类也在选定 oracle 环境下复跑；不是 22 种
独立硬件资格。主要证据：

- 固定 `64e9bceb2c3a856efed96feda784a50947049feb`，oracle checkout 未修改；测试提取
  原始 scalar 函数和原始头文件单独编译，不链接被测 decoder；关闭 FMA contraction。
- 8 个量化 type × 32 组 × 12 blocks，共 3072 个随机 block 的 FP32 输出 bit-exact。
- 所有 63,488 个有限 FP16 bit patterns 的转换/往返，以及 ties/subnormal/overflow。
- 所有已注册 type 的目标 dtype、非对齐 column slice、strides、gather 与 guard bytes。
- split 缺片/重复/架构不符、payload 重叠/短缺、block 几何和 metadata 容量负例。
- 本地 Z Q8/Q4 各抽取 w1、w2、最后一层 QKV，每张前三 row；FP32 与固定 GGML oracle
  exact，BF16 与 oracle 的一次 RNE exact。不是完整模型数值/图片质量验证。

旧格式边界 4 项通过；仅将旧 fixture 修成实际对齐、非重叠且 payload 足量的合法文件，
没有把不支持格式的测试改成可执行。

独立 native 安全程序实际以 `-fsanitize=address,undefined`、`-Wall -Wextra -Werror`
构建执行，通过成功路径禁止 `operator new`、source/target alias、gather index alias、
奇数 row pitch/未对齐 target、overflow、pre-cancel 检查。程序入口：
`tests/native/gguf_decode_safety_test.cpp`，编译链接 `native/core/gguf_decode.cpp`。
这不证明异步 pager 的所有 cancel/fence 状态；pager 尚未接入。

既有 10 个 Z unittest 模块共 72 项通过；`mlx.cpp` 在 managed headers 下 C++20
syntax-only 通过；构建脚本 bash syntax、Python compile 和 `git diff --check` 通过。
没有重建/替换完整产品 dylib；下一阶段产品构建会纳入已加到 build.sh 的 decoder 源文件。

## 3. 真实 Q8 完整投影的解码性能

M4 Max（另由 `sysctl` 核对）、macOS 26.6.2，CPU-only。最终版本回执：
[gguf-cpu-decode-20260930-v2.json](validation/gguf-cpu-decode-20260930-v2.json)。

```sh
.venv/bin/python tools/native/benchmark_gguf_decode.py \
  --checkpoint models/z-image-runtime-gguf-q8/z_image_turbo-Q8_0.gguf \
  --iterations 24 \
  --output outputs/quantized-execution-20260930/new-gguf-cpu-decode.json
```

输出路径必须尚不存在；部分 sandbox 需要授权读取 sysctl 设备身份。工具默认每次仅
保留一个投影的 packed 数据及两个相同 BF16 target，buffer cap=256 MiB；不加载全模型。
每 arm 3 warmups、24 交错样本，包含同步 decode 函数调用，排除读取/分配/哈希。

| 实际 tensor | shape `[N,K]` | scalar median | SIMD median | scalar/SIMD | 整张目标 |
| --- | --- | ---: | ---: | ---: | --- |
| layer 0 FFN w1 | `[10240,3840]` | 69.528 ms | 5.813 ms | 11.961× | bit-exact |
| layer 0 FFN w2 | `[3840,10240]` | 68.687 ms | 6.314 ms | 10.879× | bit-exact |

单个 target=75 MiB；packed=41,779,200 bytes；对照实验的两个 target + source buffer
为 199,065,600 bytes，另报 1024-byte scratch。不把这当作进程/OS page-cache 峰值。
回执含原始样本、实际 SIMD 覆盖、tensor payload SHA、输出 SHA 和源码/二进制身份；
没有全 checkpoint SHA，因此不是 bound campaign profile。

保留的 [第一版回执](validation/gguf-cpu-decode-20260930.json) 使用较慢的 scalar 错误
检查实现，约 164 ms vs 5.4 ms。最终版同时优化了 scalar 的固定错误/检查路径，必须
使用上表更强 baseline，不能拿第一版约 30× 宣传最终速度。两版不是同一源码的两个
独立 campaign。初始 probe 编译/JSON 输出错误已修正；对应失败未产生可用回执。

这里证明的是 **CPU 权重解码更快**。仍需测 dense GEMM、GPU bandwidth 竞争、预取
供给、fence、完整 FFN/step/request；不能说 Z-Image 出图快了 11×，更不能据此宣称
ANE W8A8 已实现。暂无正式全请求性能推广资格。

## 4. 下一阶段及不能标记完成的项

1. 将 Directory + SourceLease + decode plan 接到 typed pager 与唯一 ledger/slot tracker；
   实现 p=0/1，而不是全模型 dense load；修复 legacy loader 的路径重开接缝。
2. 接入 Z per-block source binding、保留 mixed 小字段精度，跑同 profile O1 exact 和
   source-quant/N1、真实图片；完整 source/binary/input/profile binding 仍需完成。
3. 逐步扩大 K 文件整图与 Qwen3 embedding/per-layer adapter；目前只有 CPU 格式解码
   支持，不能把它当作 encoder/GPU 支持交付。
4. ConvRot 有界转换/旋转质量、GPU/ANE 并行和 W8A8 继续按 C*/A*/G* 工作包。
   本阶段没有修改其默认路径或删除门禁。

磁盘检查余量约 15 GiB，这一阶段不需要模型下载/缓存清理；没有删除文件。完整目标
仍处于 active，不因为 CPU 子阶段通过而把 R0–R5 标记完成。

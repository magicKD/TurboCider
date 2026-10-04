# 16 · Q4/Q5/K CPU SIMD 解码与真实 encoder 加速（2026-10-01）

[目录](README.md) · [Qwen3 接入](15-qwen3-gguf-conditioning.md) · [验收合同](11-acceptance-profiles-and-feasibility.md)

本阶段修复15实测的 K-format scalar decoder吞吐瓶颈，交付ARM NEON直填以及同源
scalar/SIMD控制。没有更换量化文件、放宽dtype/质量政策或增加全模型dense缓存；15的
原BF16媒体差异仍未资格化，完整目标保持active。

## 1. 内核与数学不变量

`gguf_decode.cpp` 的SIMD从Q8扩展至 **Q4_0/1、Q5_0/1、Q4_K/Q5_K/Q6_K**，所有
已注册quantized types均支持F32/F16/BF16 contiguous完整block目标：

- nibble与Q5高bit、K scales/minimum、Q6 high-bit planes与signed subscale按固定GGML
  geometry展开。仍保留源FP16 master scales，FP32重建之后仅一次目标RNE。
- FP32乘法/加减顺序与固定scalar oracle一致，不使用FMA替代、fast-math或重新W8量化。
  Q6先计算master×signed-subscale，再乘code；group尺度不移到整个dot后。
- BF16用整数bit RNE；FP16明确检查65520边界，溢出/nonfinite失败，不clamp。
  finite FP16 scales与受限integer codes/subscales的重建范围可装FP32/BF16。
- 不申请heap/整row或矩阵scratch；既有decoder固定1024-byte scratch上界不增。
  完整block/contiguous/目标element对齐才走SIMD；部分slice、scatter或unaligned目标
  保留scalar。源unaligned load合法，任何路径都遵守既有checked target/alias合同。
- 非ARM编译隔离保留scalar；本阶段无非ARM实机成绩，不把build guard当硬件验收。

`GgufWeightPager`增加immutable DecodeOptions，fill/gather统一传递；Qwen3 plan workload/
cache identity与实际receipt绑定 `cpu_simd` 或 `cpu_scalar`。数学profile与槽预算相同，
不会用上一个模式的prompt-cache成果伪造另一个模式的计时。

实验模型控制 `TURBOCIDER_QWEN3_GGUF_SCALAR_DECODE=0/1`，0为既有默认SIMD选择，
1为同源控制；严格拒绝其他值。standalone probe可在末尾传`simd|scalar`。它们只是
CPU decoder选择，不是硬件W8A8、ANE或产品qualification入口。

## 2. 实际正确性与安全验证

- 固定GGML checkout/关闭FMA oracle、随机所有type与真实Z样本24项通过，FP32 bit-exact
  并核对target RNE；包括同基类复跑，不能解释为24个不同type。
- 扩展独立ASan/UBSan/no-operator-new程序：8个quantized types×3目标dtype，signed-zero/
  最小FP16 subnormal/正负finite最大scale、FP16 overflow接受/拒绝一致、guard bytes、
  odd row pitch/unaligned target、duplicate gather、nonfinite scale均通过。
  ARM完整block receipt确认每type实际进入SIMD；unaligned fallback不冒充SIMD覆盖。
- 31个选定旧GGUF/ANE host/ConvRot math/QDQ/screen/Qwen sweep测试、72个既有Z测试通过。
  真实CoreML raw GGUF/ConvRot direct-slot两项、Metal pager/gather/lease释放一项和
  prefill policy四项通过。
- 真实Qwen3 Q8/mixed-K短21/长273 rows、p0/p1共8个component与5负例全部通过；全部
  conditioning与15的pre-SIMD输出SHA一致。既有源量化差异没有因内核优化而改变。
- 完整隔离实验lib、probe、Python/bash syntax及diff检查通过，发行库未替换。

## 3. CPU完整投影（同源同预算）

M4 Max/macOS26.6.2；Qwen3-4B Q4_K_M固定source，layer0两张完整投影。BF16输出，
每arm3 warmups、24交错样本，包含decode函数，排除parse/read/allocation/GEMM。
此处按实际Qwen3 `[N,K]`计费，每张目标49,807,360 bytes，不沿用Z的矩阵尺寸。

| tensor / type | `[N,K]` | scalar median | SIMD median | scalar/SIMD |
| --- | --- | ---: | ---: | ---: |
| ffn_gate / Q4_K | `[9728,2560]` | 49.872 ms | 4.707 ms | 10.596× |
| ffn_down / Q6_K | `[2560,9728]` | 51.223 ms | 5.566 ms | 9.203× |

整target bit-exact，receipt含payload/target/source/binary hash、所有samples与实际SIMD
block计数。[独占回执](validation/gguf-k-simd-20261001.json)。最初/并发其他测试的screen
仍保留在outputs，不删除慢样本；此表使用其他模型测试结束后的新独占campaign，不能混池。
这些是CPU权重解码倍率，不是图片生成倍率。

```sh
.venv/bin/python tools/native/benchmark_gguf_decode.py \
  --checkpoint models/Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M.gguf \
  --tensors blk.0.ffn_gate.weight blk.0.ffn_down.weight --iterations 24 \
  --output outputs/new-k-decode.json
```

## 4. 完整 Qwen3 encoder matched screen

同一个mixed-K文件、21 tokens、p1/two slots、8GiB managed ceiling、同dtype/math。
同进程Native verified-source proof warm后，每次仍重新构造source、读compressed weights/
填slots/运行35层并drain；elapsed含这些工作，不只计算GEMM。
两个模式分别3 warmups、8交错measured samples，每一次都对control逐元素exact。

| mode | encoder wall median | managed peak / slots / fills |
| --- | ---: | --- |
| cpu_scalar | 7.903681 s | 2,831,541,248 bytes / 2 / 35 |
| cpu_simd | 1.393214 s | 同上 |

筛选倍率 **5.673×**；存储不扩大，source/tokens/所有模式实际计数一致。
[完整raw回执](validation/qwen3-k-simd-encoder-20261001.json)。单会话、小样本、单short
cell不是11的正式双会话/24-request/cold/bootstrap/p90推广campaign；不注册accelerated。

```sh
.venv/bin/python tools/native/benchmark_qwen3_decoder.py \
  --checkpoint models/Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M.gguf \
  --config models/Tongyi-MAI-Z-Image-Turbo/text_encoder/config.json \
  --tokenizer models/Tongyi-MAI-Z-Image-Turbo/tokenizer/tokenizer.json \
  --prefetch 1 --iterations 8 --output outputs/new-encoder-decode-screen
```

## 5. 完整 Z 真实输入回归

固定mixed-K encoder、Q8 DiT/compat-affine p1、portrait/512²/seed42/4steps，独立进程
单次diagnostic（dump与20ms内存observer开启，**不是正式计时**）：

| mode | request wall | encoder decode active |
| --- | ---: | ---: |
| scalar | 21.6466 s | 7.123704 s |
| SIMD | 15.1988 s | 0.738150 s |

conditioning、initial/所有step/final latent、decoded、PNG完全相同，encoder各35 fills，
DiT各120 fills；同一个PNG SHA。看到了decoder瓶颈改善传到整请求，但不据这两次有
observer、cold状态不同的样本宣称正式约1.42×推广资格。whole-request envelope仍unknown，
质量与15的source-mixed结果相同，不继承原BF16画质。

[全请求与component-after回执](validation/k-simd-z-regression-20261001.json)，本地
`outputs/quantized-execution-k-simd/`保存图片、全部张量、原始samples/失败和receipt。
复现沿用15的模型命令，只改变`TURBOCIDER_QWEN3_GGUF_SCALAR_DECODE=1/0`和新output。

## 6. 仍需推进

CPU SIMD不等于所有GGUF/encoder/模型资格：Q5/K的真实模型扩展、source-quant/O1与媒体
门、正式per-component schema/whole-envelope、短prompt packed kernels和更紧预算仍需继续。
ConvRot bounded模型/ANE并行、M5/static/runtime W8A8、p2/packed-streamed/tiles与IQ不因
这次提速被标为完成。默认生产路由、manifest/catalog资格不变，没有清理模型或历史记录。

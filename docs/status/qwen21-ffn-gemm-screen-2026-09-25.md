# Qwen Image 2.1：FFN GPU GEMM 形状匹配筛选

BF16 Qwen21／M4 Max：`tools/native/qwen21_gemm_probe.cpp` 为独立 GPU
单算子筛选，不加载 checkpoint，也不代表完整生图加速。随机但确定性
BF16 输入，模拟 512² decode 的 1024 行和 1024² 的 4096 行；
gate/up 权重矩阵 `[24576,4096]`，down 为 `[4096,12288]`。
MLX `matmul` 与 MPP Metal TensorOps 的调用按交替顺序成对运行，
丢弃前三轮预热；1024 行每臂 5 个样本，4096 行每臂 3 个样本。
输出为毫秒中位数，包含单次 API 调度和 GPU 完成等待，不包含
权重加载、量化或完整 block。重复运行和不同机器仍需单独验证。

| 行数／算子 | 匹配 MLX | 最快 MPP tile | MPP 耗时 | 数值 |
| --- | ---: | --- | ---: | --- |
| 1024 gate/up | 14.072 ms | 32×128 | 13.718 ms | max abs 0 |
| 1024 down | 7.401 ms | 32×128 | 7.164 ms | max abs 0 |
| 4096 gate/up | 54.820 ms | 32×128 | 54.481 ms | max abs 0 |
| 4096 down | 28.253 ms | 32×128 | 27.679 ms | max abs 0 |

成对 gate/up 投影并在同一个 Metal dispatch 内完成 SwiGLU 的候选
（packed 双矩阵＋双 accumulator）在 1024 行最快 13.930 ms，对照
14.005 ms；4096 行最快 55.208 ms，对照 55.804 ms。
相对 MLX 的随机输入输出 rL2 约 `3.88e-5`，max abs 8；
由于激活函数计算次序不同，该候选不能称为逐像素精确，且微基准
收益不足 1.1×。没有将这些 tile 或激活融合提升为默认模型路径。

GPU W8A16：先离线把随机 BF16 权重转成 FP16／affine W8（预处理不计时），
再分别测 g32/g64/g128 的 `quantized_matmul`。1024 行 gate/up
为 `17.30–17.51 ms` 对约 `13.88–14.05 ms` BF16；down 为
`9.09–9.24 ms` 对约 `7.35–7.42 ms`。4096 行 gate/up 为
`68.84–69.82 ms` 对约 `55.07–55.18 ms`，down 为
`35.72–35.88 ms` 对约 `28.20–28.49 ms`；所有受测形状均更慢。
W8A16 对随机 BF16 输出的 rL2 在 `0.00508–0.00618` 范围。
这支持继续将生产候选的 GPU FFN 后缀保留 BF16，不能据此推断
所有真实权重分布、设备或其他 GPU 精度的性能/画质。

补充检查了**把相同 BF16 数值转成 FP16 表示**的 MLX 计算（1024 行，
每臂 7 次交替样本，前三轮预热；含 API 调度与同步，不含离线转换）：

| 算子 | BF16 中位 | FP16 中位 | 对 BF16 rL2 |
| --- | ---: | ---: | ---: |
| gate/up | 14.023 ms | 14.003 ms | 0.00169 |
| down | 7.346 ms | 7.392 ms | 0.00168 |
| gate/up＋SwiGLU | 14.099 ms | 14.207 ms | 0.00291 |

FP16 没有足够的单算子收益覆盖真实模型的 dtype 边界、画质验证和
可能的中间值重排，故没有改生产 GPU 路线。这里是随机 BF16 输入，
不是图像质量结果；筛查开关为 `qwen21-gemm-probe gate_up 1024 7 --fp16`
（`down` 和 `gate_swiglu` 同理）。

同一 1024 行随机 BF16 形状又筛查了 FP16 activation＋affine W4 weight
（每臂 5 次交替样本，group 32/64/128；不计离线量化）：gate/up
`17.20–17.53 ms` 对 BF16 `13.95–14.08 ms`，down
`8.98–9.24 ms` 对 BF16 `7.33–7.35 ms`；单算子 rL2
`0.081–0.100`。W4 在这些 GPU 大矩阵乘上同样更慢且随机输入误差
更高，没有进入生图路径。可用 `qwen21-gemm-probe gate_up 1024 5 --w4`
及 `down` 复现；此测试不能代表真实模型的图像质量或其它硬件。

复现（同机、独占 GPU）：

```sh
bash tools/native/build_qwen21_gemm_probe.sh
build/native/qwen21-gemm-probe gate_up 1024 5
build/native/qwen21-gemm-probe down 1024 5
build/native/qwen21-gemm-probe gate_swiglu 1024 5
build/native/qwen21-gemm-probe gate_up 4096 3
build/native/qwen21-gemm-probe down 4096 3
```

即使 MPP 在一部分 GEMM 上略快，单算子小幅收益也不能替代
512² 文生图／编辑的配对 Session、质量和 1–3 参考图验证。当前尚无
足够证据把它接入生产 DiT；纯 GPU 1.1–1.2× 目标未达到。

## 三完整参考图首步真实行数补测（2026-09-26）

三参考 512² 首步包含 12,288 reference + 130 text + 1,024 target，
即 13,442 行。筛查器现接受该行数；仍只用确定性随机 BF16 输入，
非 checkpoint 权重或模型级验证。每对仅两次交替样本，缺 ABBA／
温度控制，不适合凭小差值推广。形状匹配 GPU 单算子中位：

| 算子，13,442 行 | MLX BF16 | MPP 最佳（48×128） | GPU W8A16 最快 |
| --- | ---: | ---: | ---: |
| gate/up | ~180.5 ms | ~179.2 ms（逐位相同） | ~225.2 ms（g128） |
| down | ~92.7 ms | ~90.2 ms（逐位相同） | ~116.7 ms（g128） |
| gate/up＋SwiGLU | ~182.7 ms | ~179.9 ms（rL2 约 3.84e-5） | 未测 |

补测不支持在首步盲目替换为 W8A16 GPU，也没有足够的 MPP
收益支付真实模型集成/布局与质量验证成本；保留既有最快默认 GPU。
这项 GEMM screen 和完整首步 18.04 秒测量边界不同，不能把表内
毫秒直接乘 32 当作整次提速。

## 缩到 512px 参考图的 1–3 图首步行数

probe 现在额外接受 2091/3157/4226 行，分别对应 1/2/3 张
512px 参考图的真实首步（包括 text + output）。确定性随机 BF16
张量、交错顺序每臂三次计时，不能等同于模型级加速：三种行数的
gate/up MLX 约 28.4/43.1/57.4 ms、最快 MPP 约
28.2/42.2/56.6 ms，逐位相同且单算子仅有小幅收益。
4226 行 down 为约 29.5 vs 28.8 ms；gate/up+SwiGLU
为约 58.2 vs 57.3 ms（舍入有差异）。该行数的 W8A16
gate/up 约 71.1–72.5 ms、down 约 36.9–37.2 ms，
均慢于 BF16。没有替换当前最快的 GPU FFN 路径。

同一 3 图 512px 五步 GPU 的同步 block-0 诊断测得首步
约 4.72–4.75 秒、后续每步约 1.20–1.22 秒；首步 block-0
的 FFN gate/up、down、QKV、attention、Q/K norm+RoPE
分别约 63.6、31.3、29.8、14.7、13.0 ms。诊断禁用了
block-0 编译并强制同步，**不可乘 32 外推**整张图的占比；
诊断 PNG 与旧纯 GPU 基线逐字节相同。

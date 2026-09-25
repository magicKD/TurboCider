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

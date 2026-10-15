# 量化实验：把每一个比例因子写出来

[实验室](README.md) · [CPU实现](quantization.py) · [公共图实现](ane/bench.py)

## 1. Packed Q4/Q8与physical pitch

`pack_codes/unpack_codes`使用little-endian U32、最低位在前的unsigned affine codes；`affine_slice`保留完整矩阵行宽和group metadata，再取逻辑列范围。它刻意不是GGUF K-quant解析器。

```sh
python3 -B techniques/labs/run.py quant-formats
python3 -B techniques/labs/run.py precision-boundaries
```

对照3×24矩阵的中间8列：正确路径和独立索引witness完全一致；错误地以局部8列作为pitch时，所有读取仍可能在界内但矩阵已错。迁移时替换`encoding/group/scales/offsets`，先覆盖非零range、rowoffset、tail与原metadata dtype，再换成Metal decoder。

`precision-boundaries`展示BF16/FP16指数范围差异，以及`half(half(scale*code)+offset)`和`half(scale*code+offset)`可不同。它是明确两次舍入的反例，不声称所有编译器都会作同样FMA lowering。

## 2. Normalized W8A8 FFN

对一行实数`z`，安全scale（不含device额外headroom）定义为：

```text
s_norm = round_fp16(max(abs(z))/127 * 128)
q = clamp(RNE(z / s_norm * 128), -127, 127)
z_hat = q/128 * s_norm
```

零行使用非零安全scale；极小scale保护、-128源编码、非有限和FP16 overflow要分别处理。这里对新量化使用[-127,127]，**已有signed ConvRot的-128不应被擅自截掉**。

输入X与gate/up权重各自旋转/量化，恢复gate/up后计算完整SiLU×up；hidden再按down输入basis旋转并A8量化。最终down整数dot的独立witness：

```text
Y[m,n] = integer_dot(q_hidden[m,:], q_down[n,:])
         * s_hidden_norm[m] * s_down_norm[n] / 128²
```

CPU实现同时做DQ matmul与整数dot恢复，两条应在double误差内一致；原FFN→W8A8的误差另报，不把“量化实现自洽”当成模型质量通过。

```sh
python3 -B techniques/labs/run.py w8a8-reference
python3 -B techniques/labs/run.py w8a8-reference --options '{"rows":7,"hidden":32,"width":64}'
```

## 3. Rotation的位置与sign recipe

`R=D H/√B`时，线性边界有`(XR)(WR)ᵀ=XWᵀ`；逆变换是先H再D，不能假设同一个forward等于inverse。SiLU前后不能随意把rotation移来移去。

本实验用小power-of-two blocks学习同一代数；生产H128/H512、Comfy H256的符号/排列、normalization、source rounding按 [真实配方](../../native/backends/ane_w8a8_math.hpp)核对。Splash另用其固定PRNG与graph常量；相同seed数字不证明sign序列相同，应逐项对比。

## 4. SmoothQuant与diffusion calibration

`X' = X/S, W'=WS`保持线性实数等价；`S_j=max|X_j|^α/max|W_j|^(1-α)`只是一个候选scale构造，极值floor/clamp需要明确。

```sh
python3 -B techniques/labs/run.py smoothquant
```

代码分开calibration/holdout行，扫α并分别报exact transform与holdout Q8误差。迁移到DiT时按层/step/reference/LoRA采activation：早/中/晚step、敏感projection保留单独profile，和 [Sana diffusion PTQ参考](../references/kernelwiki-sana.md)所讨论的粒度搜索区分CUDA具体格式。

## 5. 设备实现与探索轴

本地[完整W8A8 CPU实现](quantization.py)包含rotation、normalized scales、hidden Q/DQ与down恢复；[ANE CLI](ane.md)的native样本隔离单INT8 matmul共享feature ABI，不把它当完整FFN。公共INT8特征要求由CoreML9/macOS26/SDK26明确检查。原完整产品FFN emitter与native fixtures仍可通过[可选接入](../integrations/README.md)运行，其他合法W8A8表达可独立实验。

建议探索：per-token/per-group A8、W4A8与LUT、S1平滑、grouped Hadamard、K-segmentation、敏感层GPU、headroom的动态scalar、GPU FP32 restore。每次固定来源与当前最强GPU分母；新recipe分别做结构、数值和完整请求测试，不把某个失败group或现有guard理解为永久不可行。

完整runtime代码参考 [Public W8 graph](../../tools/coreml/export_runtime_w8a8_ffn.py)、[shared GPU stager](../../native/backends/ane_gpu.hpp)、[Private executor](../../native/backends/private/ane_w8_executor.mm)。Private是独立授权research环境，本文不自动开启。

## 6. W4A8 / INT8 LUT的完整FFN参考

```sh
python3 -B techniques/labs/run.py w4a8-reference
```

原toy weights→per-row signedQ4/F16scale→U4packed codes→共享16项signed-I8 LUT；输入和correctedhidden各做per-tokenA8，整数gate/up/down点积分别恢复scale，并与LUT-DQ matmul独立对照。source→candidate误差仍单列。

W4描述storage，INT8 LUT/展开bank描述计算表达，二者不自动等于原生INT4MAC。若runtime先展开INT8，bankbytes会比packed source大；若图内lookup则需验证palette/constexpr支持和完整吞吐。这个uniformsymmetric LUT不是任意GGUF/MLXaffine，也不是NVFP4；扩展新格式先写code/zero-point/group/LUT/scale合同。

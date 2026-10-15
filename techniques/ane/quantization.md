# 量化、SmoothQuant、Hadamard 与 ConvRot

[总目录](../README.md) · [存储格式](../memory/storage.md)

## 先拆开五个概念

| 名称 | 实际含义 | 不能由此推出 |
| --- | --- | --- |
| Q4/Q8 packed storage | checkpoint / GPU codes 压缩 | ANE 正在做 INT4/INT8 算术 |
| W8 + FP16 IO/compute | 常量权重量化、浮点激活 | W8A8 |
| runtime FP16 from GGUF | 将 packed weights stage 为浮点 | 保留压缩的 ANE 计算 |
| W8A8 graph | code/scales 和激活量化都在合同中 | 物理原生 INT8 MAC 已观察 |
| FP32 GPU restore/join | epilogue 在 GPU 用 FP32 | ANE 图使用 FP32 accumulator |

精度报告还需写 source dtype、scale/bias dtype、rotation basis、rounding、accumulator、output cast；只有“INT8”远远不够。

## 两种等价变换与其误差来源

使用行向量、`Y = X Wᵀ`：

- 正交旋转 `R`：`(X R)(W R)ᵀ = XWᵀ`。精确实数等价；旋转后的量化、有限精度和 reduction order 仍产生误差。
- 平滑 scale `S`：`(X S⁻¹)(W S)ᵀ = XWᵀ`（对角 S）。用校准降低 activation outlier，必须把同一 S 施于对应 physical source 范围。

权重和激活的 basis、sign seed、归一化、group、转置方向必须完全匹配。SwiGLU gate/up 的输入旋转与 down 的 hidden 旋转是两个不同维度，不能一张 H 表随意复用。

## 本仓库的两个不同旋转配方

[ane_w8a8_math.hpp](../../native/backends/ane_w8a8_math.hpp)明确区分 Sylvester `D·H/√B`（H128/H512）与 Comfy H256 的 H4 Kronecker 顺序。都是 Hadamard 家族不等于符号/排列相同。

Comfy direct 路线使用原 signed codes 和 row scales（包括 -128），不再次 W8 量化；legacy affine packed `q+128`、重复 scale/offset 的表示又是另一种合同。不要用普通 affine/GGUF decoder 误读 ConvRot。

RNE ties、-128/127 范围、零行安全 scale、subnormal、BF16/FP16 activation rounding、headroom normalization、hidden restore 都应有独立 CPU witness。recipe 版本进入 graph/cache/receipt。

## 校准数据该从哪里来

真实模型、真实 prompt/参考图、不同 step 和层深度的 activation；独立 holdout 做最终质量。不要只用随机正态决定所有层 S1，也不要拿 calibration 数据同时当唯一验收集。S1/SmoothQuant metadata 必须绑定 source generation/shape/recipe，不能让 cache 继续命中旧 scales。

[Splash FFN 历史](../references/related-projects.md)提供 FFN-only SmoothQuant 的参考，但 LLM retrieval/输出相同不证明扩散 latent trajectory 相同。

## 误差诊断顺序

先比 conditioning / initial latent，再做：原系数 decode → gate/up projection → corrected hidden → down partial → join → block residual → 每步 velocity/latent → decoded media。为 input/hidden grouping、FP16 reduction、FP32 join 和敏感层保留独立对照。

[ConvRot group scope](../../docs/status/convrot-group-scope-2026-10-05.md)中，input-only/hidden-only 都实际跑通，却仍没通过既定 final latent gate；不能凭组件 pass 或更好 cosine 升默认。相似最终图也不抹掉已失败的数值 gate。

## 采用策略

按 operation/phase 考虑 GPU、Public frozen/runtime、Private research，敏感层可明确 GPU。不要自动减少 step、跳激活或漏 adapter。遇到不能表达 / 非有限 / 未资格 shape，保留完整 GPU fallback 和真实 label。

Public W8 的共享 GPU staging 与完整 FFN 结构见 [实现记录](../../docs/status/public-w8-shared-ffn-2026-10-07.md)；它证明特定实现存在，不证明所有模型尺寸、LoRA、硬件或 ≥1.2× 目标均已验收。

## 实现参考与实验

可读实现：[参考代码](../labs/quantization.py)；[实验步骤与边界](../labs/ane.md)；[本项全部 case / controls](../labs/catalog.json)。

```sh
python3 -B techniques/labs/run.py w8a8-reference
```

本项控制：integer dot vs DQ and source error；holdout alpha sweep and source pitch。CPU参考/模拟与实际设备入口分开标记；设备环境和更多变体见实验步骤。

可以重开探索的条件：W4A8, grouping, rotations, S1 or sensitive-layer policies。历史负结果限定于其来源/shape/边界，不关闭该方向。

# 08 · 本轮实际交付与验证账本

本篇历史记录已归档到[量化执行实验目录](../../experiments/quantized-execution/08-experiments-and-delivery.md)。
实验顺序及证据边界见[专题索引](../../experiments/quantized-execution/README.md)，
设计与验收合同见[设计目录](README.md)。
以下保留原章节标题，供已有章节链接继续使用。

## 1. 实际源码改动

<<<<<<< Updated upstream
[阅读归档章节](../../experiments/quantized-execution/08-experiments-and-delivery.md#1-实际源码改动)。
=======
- 新增 `native/models/z_image/ffn.hpp`：复用现有 `Weights::project_many`，使
  ConvRot gate/up显式共享同一次输入旋转；down仍在SwiGLU后独立旋转。
- `native/models/z_image/z_image.cpp` 的 `z_ffn` 调用此helper。
  ConvRot INT8 默认开启；`TURBOCIDER_Z_CONVROT_SHARED_GATE_UP=0` 回退旧路线。
  不宣称已有streaming/混合channel-slice路线全部覆盖。
- 新增 `tools/native/convrot_ffn_probe.cpp` 和构建脚本：使用生产inline helper与
  已有C++ Weights primitives，非Python重写的替代算法。
- 新增 `tests/native/test_convrot_ffn.py`：一个源码开关契约测试与显式GPU数值回归。
- 新增 `tools/native/benchmark_convrot_ffn.py`：有benchmark锁、warmup、交错顺序、
  新路径不覆盖、源码/二进制hash和原始样本。合成数据，不加载真实模型。

没有新增K/IQ生产decoder、通用GGUF pager、M5 INT8 shader或runtime ANE W8A8；
这些工作的具体实现/验收步骤在02–07。没有放宽unsupported格式、ANE或内存门禁。
>>>>>>> Stashed changes

## 2. GPU 数值回归

[阅读归档章节](../../experiments/quantized-execution/08-experiments-and-delivery.md#2-gpu-数值回归)。

## 3. 组件性能测量

[阅读归档章节](../../experiments/quantized-execution/08-experiments-and-delivery.md#3-组件性能测量)。

## 4. 复现与后续升级条件

[阅读归档章节](../../experiments/quantized-execution/08-experiments-and-delivery.md#4-复现与后续升级条件)。

## 5. 尚未完成 / 不得声称完成

[阅读归档章节](../../experiments/quantized-execution/08-experiments-and-delivery.md#5-尚未完成--不得声称完成)。

## 6. 其他实际检查

[阅读归档章节](../../experiments/quantized-execution/08-experiments-and-delivery.md#6-其他实际检查)。

## 7. 文档审阅补充（2026-09-30，文档范围）

[阅读归档章节](../../experiments/quantized-execution/08-experiments-and-delivery.md#7-文档审阅补充2026-09-30文档范围)。

## 8. 后续实际代码实施

[阅读归档章节](../../experiments/quantized-execution/08-experiments-and-delivery.md#8-后续实际代码实施)。

## 9. 后续 GGUF GPU 实验纵切

[阅读归档章节](../../experiments/quantized-execution/08-experiments-and-delivery.md#9-后续-gguf-gpu-实验纵切)。

## 10. GGUF/ConvRot ANE staging 与 W8A8 筛选（2026-10-01）

[阅读归档章节](../../experiments/quantized-execution/08-experiments-and-delivery.md#10-ggufconvrot-ane-staging-与-w8a8-筛选2026-10-01)。

## 11. Qwen3 GGUF conditioning 实验（2026-10-01）

[阅读归档章节](../../experiments/quantized-execution/08-experiments-and-delivery.md#11-qwen3-gguf-conditioning-实验2026-10-01)。

## 12. Q4/Q5/K CPU SIMD 加速（2026-10-01）

[阅读归档章节](../../experiments/quantized-execution/08-experiments-and-delivery.md#12-q4q5k-cpu-simd-加速2026-10-01)。

## 13. packed-streamed 与三槽（2026-10-01）

[阅读归档章节](../../experiments/quantized-execution/08-experiments-and-delivery.md#13-packed-streamed-与三槽2026-10-01)。

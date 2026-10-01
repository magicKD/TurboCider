# 08 · 本轮实际交付与验证账本

[目录](README.md) · 2026-09-30

第 1–6 节保留首次设计/组件优化任务的历史记录；随后文档审阅补充见第 7 节。
本次文档修改没有重新运行下述 GPU 测试，不能把历史回执当作新增能力的验证。

## 1. 实际源码改动

- 新增 `native/models/z_image/ffn.hpp`：复用现有 `Weights::project_many`，使
  ConvRot gate/up显式共享同一次输入旋转；down仍在SwiGLU后独立旋转。
- `native/models/z_image/z_image.cpp` 的 `z_ffn` 调用此helper。
  仅 `TURBOCIDER_Z_CONVROT_SHARED_GATE_UP=1` 开启；未设置、`0`或其他值保持旧路线。
  默认不变，不宣称已有streaming/混合channel-slice路线全部覆盖。
- 新增 `tools/native/convrot_ffn_probe.cpp` 和构建脚本：使用生产inline helper与
  已有C++ Weights primitives，非Python重写的替代算法。
- 新增 `tests/native/test_convrot_ffn.py`：一个源码开关契约测试与显式GPU数值回归。
- 新增 `tools/native/benchmark_convrot_ffn.py`：有benchmark锁、warmup、交错顺序、
  新路径不覆盖、源码/二进制hash和原始样本。合成数据，不加载真实模型。

没有新增K/IQ生产decoder、通用GGUF pager、M5 INT8 shader或runtime ANE W8A8；
这些工作的具体实现/验收步骤在02–07。没有放宽unsupported格式、ANE或内存门禁。

## 2. GPU 数值回归

实际执行：

```sh
TURBOCIDER_TEST_GPU=1 .venv/bin/python -m unittest discover \
  -s tests/native -p test_convrot_ffn.py -v
```

96组native组合：dense-H与已有Metal butterfly各自的shared/unshared对照；
raw ConvRot、packed g32/g64/g128；BF16/FP16/FP32 activation；1/33rows及
二维/三维输入；bias、mixed dense-up回退。每组要求shape/dtype不变、有限值、
输出逐元素exact。**96组全部通过**。这不是dense-H与butterfly彼此bit-exact的断言。

未覆盖：真实checkpoint数值/画质、runtime LoRA adapter切换、更多stride、大范围
完整请求、低内存峰值。`project_many`原有LoRA回退保留，但本轮未新建其真实LoRA fixture。

初次沙箱内MLX初始化失败，直接probe只返回`vector`；获准访问Metal后两项unittest
均通过。此环境失败不记产品数值失败，也不把未启用GPU时的skip算通过。

## 3. 组件性能测量

执行入口：

```sh
.venv/bin/python tools/native/benchmark_convrot_ffn.py \
  --rows 1056 4128 --iterations 24 \
  --output outputs/quantized-execution-20260930/convrot-shared-ffn.json
```

固定H=3840/F=10240，BF16 activation、packed ConvRot g32/BF16 scales、原dense-H；
只比较shared gate/up。3 warmups/arm、24交错样本/arm，包含graph构造与eval，排除
权重生成/packing；另做该shape最终输出exact检查。

M4 Max、macOS 26.6.2、MLX 0.32.0，实际结果：

| rows | 原FFN median | 共享H median | 比值 control/shared | 输出 |
| --- | ---: | ---: | ---: | --- |
| 1056 | 20.3992 ms | 20.2628 ms | 1.00673× | exact |
| 4128 | 78.6236 ms | 78.0619 ms | 1.00720× | exact |

改善仅约0.7%，低于07提出的新路线推广门。保留为研究开关和可复现实验，**不升级
默认**。本轮没有重复独立进程campaign、功耗/竞争负载trace或完整内存采样；这组
紧凑交错样本不支持更广泛的统计/产品结论。

便携原始回执：[convrot-shared-ffn-20260930.json](validation/convrot-shared-ffn-20260930.json)，
含全部样本、源hash、库hash、probe hash和时间。本地原始输出保留在上述`outputs/`。
backend库SHA256为
`201a4926417ca3448d77638171f2d5ba5e84fa1bd2ea141aa9feb964ac524958`；
probe SHA256为
`e354b6517731ca941b8ce215fa2ece119e386071cdf60e84074a25dee27a869f`。
旧backend库不包含新的产品环境开关wrapper；新inline helper编入本probe，
调用该库中未修改的Weights primitives，因此这是helper/组件验证而非产品整图验证。

测量边界：没有attention、encoder、VAE、真实prompt/seed、ANE或M5；不表示完整
Z-Image提速或受限内存认证。不将上一轮Hadamard/Q8单算子结果与本轮相加。

## 4. 复现与后续升级条件

probe构建需要已有 `build/native/libturbocider.dylib` 和managed MLX依赖；inline
helper在probe内编译，backend primitives来自回执指定库。若要在产品生成使用
新环境开关，必须重新构建native库；本轮不会默默替换CLI/App发行包。

完整真实模型ABBA、same-route图片比对、slot内存和高分辨率覆盖通过之前，不将
该开关默认开启、不登记新性能资格。即使本轮单层收益显著，也不外推全请求倍率。

## 5. 尚未完成 / 不得声称完成

- Q4–Q8全类型native GGUF与encoder组合。
- 0/1/2层前瞻decoder pager、分片fallback和整体内存认证。
- ConvRot A8校准、static原生旋转W8A8。
- 动态runtime INT8图的编译、placement、arithmetic及实际速度证明。
- M5 shader/运行资格；本轮无该硬件。
- 全模型性能、跨prompt画质与低容量真实机器验证。

本设计的作用是把这些任务变成有界、可复核的工程步骤，而不是用一个INT8标签
合并所有问题。参考仓库只读，模型/图产物/已有性能基线未被修改。

## 6. 其他实际检查

- `z_image.cpp` 使用managed MLX headers的C++20 `-fsyntax-only`通过；未重建/替换
  完整产品dylib，也未执行新的完整模型生成。
- 新增Python工具/test的`py_compile`、构建脚本`bash -n`通过。
- `test_native_gguf.py` 的4项CPU格式边界测试通过，证明本轮没有放宽旧格式门禁。
- 10个现有Z-Image unittest模块共72项通过：benchmark、audit、row split、sharded
  checkpoint、SmoothQuant、W8 suffix、Core ML W8A8图、layers、routing和SQ search。
- 本目录本地链接及JSON样本数量检查通过；`git diff --check`通过。

72项回归的可复现入口：

```sh
PYTHONPATH=tests/native .venv/bin/python -m unittest \
  test_z_image_benchmark test_z_image_benchmark_audit \
  test_z_image_row_split_probe test_z_image_sharded_checkpoint \
  test_z_image_smoothquant test_z_image_w8_suffix_benchmark \
  test_z_image_w8a8_coreml test_z_image_w8a8_layers \
  test_z_image_w8a8_routing test_z_image_w8a8_sq_search -v
```

最初尝试宽泛`discover -p 'test_z_image*.py'`得到6个入口错误：该目录也有需要
checkpoint/manifest参数的CLI脚本，会在import时执行argparse，不可当unittest
模块导入。已按上面的明确模块列表重跑；没有修改这些既有脚本，未提供fixture的
整请求CLI验收仍未运行，不能把初次失败或其skip重标为全套通过。

## 7. 文档审阅补充（2026-09-30，文档范围）

- 新增 09：R0–R5、首个 Z/Qwen3 adapter 的名字/shape/tap、源 mixed 与优化 dtype profile。
- 新增 10：版本化配置、旧 streaming 接入、read/decode/publish、reader 所有权、Qwen3
  lazy eval 接缝、tile 和 transactional FFN recovery、P4 产品入口工作包。
- 新增 11：固定输入、D0/D1/N1/N2、媒体审核、内存/性能门、未来 runner/verifier 合同，
  以及 M5/ANE placement/arithmetic 证据与有限可行性实验。
- 新增未绑定验收 JSON 模板；null hashes/设备/预算意味着不能执行正式认证，不是缺数据的 PASS。
- 同步 01–07 与目录；没有实现 runner、扩充 decoder、修改生产路由或登记新 capability。

本次只读检查了本地 Q8/Q4 的 header/tensor directory，具体 bytes、type histogram
与差异见 09；没有读取模型 payload 来做数值验证，没有全文件 hash，因此仅为候选
fixture 审计。没有模型生成、性能重测、ANE/M5 执行或完整内存认证。

新增政策中的 48 配对图片、24 warm 样本/arm 和误差阈值都是后续 campaign 要求，
不是本轮执行次数/实测值。源码已有未提交改动保持原样；本次交付仅设计文档与模板。

实际文档检查：12 份 Markdown 的 62 个本地链接、围栏配对/行尾空白、2 个 JSON
代码块及 validation 下 2 个 JSON 文件解析通过；48 配对输入、每性能 cell 每 arm
24 个 warm 样本、4 个性能 cells、单/双槽映射的模板一致性断言通过。
模板仍为 unbound/not_run，`git diff --check` 通过。这些是文档/结构检查，不是产品验收。
另以 09 的 required-key/shape 表核对两个本地 GGUF 目录，各 453 个 tensor 全部匹配，
仅应用声明的 pad-token reshape；type histogram 一致。未解码 payload，不代表 tensor 数值通过。

## 8. 后续实际代码实施

安全目录与预分配 CPU decoder 的第一阶段实现、固定 GGML oracle、真实 Z Q8/Q4
切片、sanitizer 与最终 SIMD 投影回执见 [12](12-implementation-progress.md)。本节
是首次进入代码实施的增量，不重写前面历史边界；R0/R1 完整资格仍未完成，W8A8/ANE
新增路线仍未实现。

## 9. 后续 GGUF GPU 实验纵切

SourceLease/managed ledger/slots已接到Z真实Q8/Q4生成。实际通过与失败、原MLX
BF16→FP16浮点加载差异、明确compat profiles、取消重试及限制见
[13](13-gguf-bounded-runtime-progress.md)。没有完整R1发布资格，没有新增production
catalog、W8A8或ANE资格；发行库/CLI/App未替换。

## 10. GGUF/ConvRot ANE staging 与 W8A8 筛选（2026-10-01）

实际 API、bounded conversion workers、CPU SIMD整投影回执、公开CoreML同图换权、
完整gate QDQ筛选与K-tile负例、既有ANE/真实Z回归见
[14](14-packed-ane-staging-and-w8a8-screen.md)。只交付其明确列出的组件/研究增量，
没有硬件INT8或新增GGUF+ANE整图资格，没有替换发行库或放宽生产门禁。

## 11. Qwen3 GGUF conditioning 实验（2026-10-01）

模型级SourceLease/config/tokenizer绑定、embedding gather、per-layer固定槽、Q8/Q4_K_M
组件和真实Z组合已实现/执行，详见[15](15-qwen3-gguf-conditioning.md)。单/双槽exact不替代
原BF16差异或媒体门；原BF16完整请求参考出现N1失败，原样保留，默认和生产资格不变。

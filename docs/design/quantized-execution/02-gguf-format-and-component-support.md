# 02 · GGUF Q4–Q8 与 encoder→DiT 支持

[目录](README.md) · [内存](03-bounded-dequant-and-memory.md)

## 1. 分层边界

```text
GGUF metadata / tensor directory / split set
             ↓ checked descriptors + source leases
format-specific decoder registry
             ↓ bounded target views (not a whole-model float dictionary)
component adapter: names, axes, embedding, norms, RoPE, hidden states
             ↓
existing GPU / static Core ML / runtime Core ML execution
```

格式层不运行 tokenizer、attention 或采样器；模型层不重复实现每种量化 decoder。
不得让 `load_gguf_file()` 先物化全模型浮点再声称下游只占两个槽。

## 2. 首发格式矩阵

此处“首发”是 Q4–Q8 产品目标，不要求第一个可运行 PR 全部实现。
[09 的 R1](09-release-scope-and-component-contracts.md) 先完成 Q8_0+浮点的端到端纵切，
R2 再扩充其他格式；实际模型映射和精度以 09 为规范。

下表 block bytes 是 GGML 常见磁盘定义，包含量化 metadata、不含 GGUF 文件对齐；
实现仍须从固定版本 type registry 验证，不能只用 `bits/8` 估内存。

| Tensor type | 元素/block | bytes/block | 首发策略 |
| --- | ---: | ---: | --- |
| F32 / F16 / BF16 | 1 | 4 / 2 / 2 | 必需；直接复制或有界 dtype 转换 |
| Q4_0 / Q4_1 | 32 | 18 / 20 | 原生 packed 可保留；新增目标直填 decoder |
| Q5_0 / Q5_1 | 32 | 22 / 24 | 新 decoder；合并高 bit 与低 nibble，正确 offset |
| Q8_0 | 32 | 34 | 必需；保留 signed codes 和源 FP16 scale |
| Q4_K / Q5_K / Q6_K | 256 | 144 / 176 / 210 | 必需；准确展开嵌套 scales/min/high bits |
| IQ4_NL / IQ4_XS | 32 / 256 | 18 / 136 | 第二批；查表语义，不冒充 affine Q4 |
| Q8_1 / Q8_K | 以 registry 为准 | 以 registry 为准 | 若文件实际含有则单独 decoder/fixture；不因名字含 Q8 自动接受 |
| Q2/Q3/IQ1–3 或新扩展 type | — | — | 本轮产品目标外，先明确拒绝；混合文件包含时不能部分成功 |

`Q4_K_M`、`Q4_K_S`、`Q5_K_M`、`Q5_K_S` 是文件级配方，可能混有 Q6_K、
Q8_0 和浮点小 tensor。首发覆盖“Q4–Q8”意味着这些文件中**每个实际 type**
均有 decoder，不是支持文件后缀就完成。Unsloth 的动态配方同理。

原始参考：`../references/llama.cpp/ggml/src/ggml-common.h` 中 block 结构及
`ggml-quants.c` 中 `dequantize_row_*`。本地参考提交为
`64e9bceb2c3a856efed96feda784a50947049feb`。落地需固定依赖 commit/校验和，不能在
生产构建时读取开发机 `../references`；可窄化 vendoring 或版本固定的解码库，
不必引入其整个推理引擎。本轮未复制第三方实现。

## 3. 拟议类型与接口

```cpp
// Design sketch, not an already exported API.
struct PackedTensorView {
    std::shared_ptr<const SourceLease> source;
    TensorId id;
    GgmlType type;
    Shape logical_shape;
    ByteRange payload;
    QuantGeometry blocks;
    AxisMapping axes;
    TransformRecipe transform; // none / convrot_h256 / derotate_to_dense
};
struct DecodeTarget {
    MutableByteSpan backing;
    DType dtype;               // BF16 / FP16 / explicitly planned FP32 fields
    Shape shape;
    Strides strides;
    LogicalSlice slice;
};
DecodeReceipt decode_into(const PackedTensorView&, DecodeTarget,
                          BoundedScratch&, CancellationToken);
```

真实 C++ 结构应复用现有 source/shape/stride 类型，以上只说明合同：

具体 read/decode 分离、同步 CPU 返回、GPU completion 和 descriptor 绑定见
[10 第 2–4 节](10-execution-and-product-integration.md)。源浮点小 tensor 可按 profile
保留 FP32；不因此允许量化矩阵默认展开整模型 FP32。

- 不返回自由增长的全模型 `vector<float>`；调用方提前分配目标和 scratch。
- 允许 CPU decoder 内用一个 block/小段 FP32 scratch，再一次舍入写 BF16/FP16。
- K slice 未对齐量化 block 时，读完整所需 blocks 后仅写请求元素；报真实读取字节。
- FP16 overflow、非有限 scale、错误 shape/stride 均显式失败，不能 clamp 权重蒙混过关。
- scale、zero-point、布局和 Hadamard recipe 属于数学身份；源 FP32 scale 不能先
  降为 BF16，再声称保留了源 checkpoint 精度。
- target 可为 MLX 共享 backing 或 ANE FP16 slot；GPU/ANE 格式不同的副本分别计费。
- embedding 支持按 token ID 收集 row 的解码，不展开整个词表。

按 tensor/layout/目标 dtype 注册能力；未知输入拒绝，不能调用 generic affine
解码“试试看”。原有 `AffineView` 只表达 MLX affine，不扩充字段后继续伪装 GGUF。

## 4. 文件、分片与可信读取

在大分配前完成：

1. magic/version、little-endian 字段、metadata 类型及嵌套数组长度上界。
2. rank/维度乘法、type/block 大小、alignment/offset/end 的 checked arithmetic。
3. payload 在文件范围内；禁止重复逻辑 tensor、未解释重叠、非法零维/超大维。
4. split 集合编号/总数/架构一致；缺片在执行前失败，tensor 不重复、不漏。
5. 固定目录扫描和 metadata 内存上限，防止仅解析文件也耗尽内存。
6. SourceLease 绑定打开的 fd 和身份，后台 reader 不按路径重新打开被替换文件。
7. parser 能描述某 type 不代表 executor 支持；生成 capability report，整请求
   所有组件完整覆盖后才允许 generate。

禁止针对格式不支持静默下载 dense 原模型或切到另一个 backend。它可能突破用户
内存/网络/隐私边界。需要替代文件时明确返回缺失项。

## 5. Encoder 不能只换 Linear

### 5.1 第一适配目标：Z / 共享 Qwen3 conditioning

接入 `native/components/text/qwen3.*`，增加独立的 GGUF source adapter，保持：

- 原 tokenizer、特殊 token、prompt template 和最大有效长度。
- Q/K/V、FFN 命名映射及 Q/K layout/旋转约定。
- causal mask、padding、GQA、RoPE theta/缩放、norm epsilon。
- 输出 hidden layer 索引、是否 final norm，以及多个层拼接的顺序。
- 无需语言模型 logits 时，不展开/计算 lm_head；若与 embedding tied，共享 source
  identity，但不能误将 packed backing 当作浮点 embedding。

GGUF LLM 可以完成文本生成，不代表其最后一层 hidden 就等于图像模型的 conditioning。
以同 prompt/token IDs、逐层 hidden 和最终 embedding 为验收对象。

### 5.2 分阶段组件矩阵

| 组件 | 接入目标 | 必须单独验证 |
| --- | --- | --- |
| Z DiT | 第一纵切，Q8→Q4/K | 两个 refiners/main blocks、mixed dtype、调制层 |
| Qwen3 encoder | 第一 encoder 纵切 | embedding gather、hidden taps、tokenizer/mask |
| Flux2 Qwen3 conditioning | 复用 source/decoder | 自己的层选择与拼接，不复用 Z 配置 |
| Qwen21 VL encoder | 后续 | mRoPE、视觉投影、deepstack、多图位置与 image tokens |
| UMT5/T5 | 后续 | relative position bias、非 causal mask、embedding/tied weights |
| LTX Gemma | 后续 | 专属 encoder、projection/connector、输出层选择 |
| CLIP/其他 vision tower | 后续 | pooling、patch/position embeddings、卷积和 layer norm |
| VAE | 默认保留现有精度 | 卷积质量/tiling 另立任务，不因“全链路”强制量化 |

不能承诺所有 Unsloth GGUF 自动可用；模型架构、配置和 tensor 几何仍必须在本项目
支持范围内。多组件请求允许 encoder GGUF + DiT ConvRot 或反过来，但每个组合独立资格。

## 6. Encoder 的低内存策略

encoder 通常只 prefill 一次，且 M 是实际 prompt tokens，不能套用 DiT 的大 M 阈值。

- 短 prompt：优先 packed GEMV/GEMM 或按投影解码；解码成本可能比计算更大。
- 长 prompt：有限层展开 + dense，按实际 M 筛选，mask/padding 不改变语义。
- 输出 conditioning 后，等待所有 reader 完成，再释放 encoder 槽/图/权重。
- 缓存只保留有身份的 conditioning；key 包含 encoder checkpoint、quant recipe、
  tokenizer、prompt/token IDs、hidden-layer recipe、LoRA、精度及 vision 输入身份。
- 不为了跨组件预取而同时驻留完整 encoder 和 DiT；阶段重叠必须被预算允许。
- 若支持自回归生成，KV cache 另算；本任务的 conditioning prefill 不伪造 decode
  性能，权重量化也不意味着 KV/activations 量化。

## 7. 精度、校准和模型适配退出条件

按次反量化只改变执行表示，不增加新的低比特权重量化；仍有目标 dtype 舍入。
Golden 分两级：GGML FP32 decoder oracle；本项目指定 BF16/FP16 舍入后的执行 oracle。
两者都记录，不要求低比特模型恢复原 BF16 模型。

首发 GGUF 路线不需要重新训练或 imatrix；imatrix 是制作某些 checkpoint 的输入，
不是解码已完整存储的 IQ 权重所必需的运行时模型数据。额外 A8 才需新的校准/质量评估。
一个 decoder 通过合成 fixture，不等于对应 encoder、DiT 和媒体质量通过。

首发 Z/Qwen3 的 required-key、shape、hidden tap 和 tokenizer 冲突规则见
[09](09-release-scope-and-component-contracts.md)；逐 type 解码、模型数值及媒体门分别
执行 [11](11-acceptance-profiles-and-feasibility.md)，不由同一“出图成功”布尔值代替。

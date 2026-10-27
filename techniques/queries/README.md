# 知识索引：直接来自 Markdown

[知识库](../README.md) · [使用与维护](../guides/knowledge-base.md) · [检索工具](../scripts/README.md)

从页面标题、链接和短元数据生成，不使用 JSON catalog，不登记实验或验收状态。完整结果通过查询按需展开。

## 内容类型

| 类型 | 页数 | 查询 |
| --- | --- | --- |
| `method` | 151 | `--page-type method` |
| `implementation` | 10 | `--page-type implementation` |
| `case-study` | 132 | `--page-type case-study` |
| `guide` | 28 | `--page-type guide` |
| `navigation` | 24 | `--page-type navigation` |
| `source` | 23 | `--page-type source` |

## stage

| 标签 | 页数 | 查询 |
| --- | --- | --- |
| `decode` | 61 | `--stage decode` |
| `dit` | 86 | `--stage dit` |
| `encoder` | 17 | `--stage encoder` |
| `ffn` | 71 | `--stage ffn` |
| `io` | 11 | `--stage io` |
| `pipeline` | 72 | `--stage pipeline` |
| `prefill` | 62 | `--stage prefill` |
| `vae` | 19 | `--stage vae` |

## operator

| 标签 | 页数 | 查询 |
| --- | --- | --- |
| `attention` | 88 | `--operator attention` |
| `codec` | 5 | `--operator codec` |
| `conv` | 25 | `--operator conv` |
| `gemm` | 85 | `--operator gemm` |
| `gemv` | 20 | `--operator gemv` |
| `layout` | 104 | `--operator layout` |
| `norm` | 27 | `--operator norm` |
| `quantization` | 44 | `--operator quantization` |
| `rope` | 15 | `--operator rope` |
| `sampling` | 53 | `--operator sampling` |
| `scheduling` | 113 | `--operator scheduling` |
| `swiglu` | 30 | `--operator swiglu` |

## backend

| 标签 | 页数 | 查询 |
| --- | --- | --- |
| `accelerate` | 1 | `--backend accelerate` |
| `coreai` | 2 | `--backend coreai` |
| `coreml` | 74 | `--backend coreml` |
| `cpu-reference` | 169 | `--backend cpu-reference` |
| `metal` | 155 | `--backend metal` |
| `mlx` | 100 | `--backend mlx` |
| `mpsgraph` | 5 | `--backend mpsgraph` |
| `private-ane` | 43 | `--backend private-ane` |
| `videotoolbox` | 1 | `--backend videotoolbox` |

## device

| 标签 | 页数 | 查询 |
| --- | --- | --- |
| `ane` | 102 | `--device ane` |
| `cpu` | 191 | `--device cpu` |
| `gpu` | 192 | `--device gpu` |
| `media-engine` | 5 | `--device media-engine` |

## symptom

| 标签 | 页数 | 查询 |
| --- | --- | --- |
| `compile-bound` | 34 | `--symptom compile-bound` |
| `graph-interruption` | 9 | `--symptom graph-interruption` |
| `launch-bound` | 75 | `--symptom launch-bound` |
| `lifetime` | 68 | `--symptom lifetime` |
| `measurement` | 136 | `--symptom measurement` |
| `memory-bound` | 56 | `--symptom memory-bound` |
| `precision-loss` | 133 | `--symptom precision-loss` |
| `stale-cache` | 65 | `--symptom stale-cache` |
| `storage-pressure` | 6 | `--symptom storage-pressure` |
| `straggler` | 22 | `--symptom straggler` |
| `wrong-layout` | 71 | `--symptom wrong-layout` |

## semantics

| 标签 | 页数 | 查询 |
| --- | --- | --- |
| `approximate` | 100 | `--semantics approximate` |
| `bitwise` | 13 | `--semantics bitwise` |
| `contract-only` | 170 | `--semantics contract-only` |
| `numerically-equivalent` | 71 | `--semantics numerically-equivalent` |
| `requires-training` | 6 | `--semantics requires-training` |

## representation

| 标签 | 页数 | 查询 |
| --- | --- | --- |
| `bf16` | 70 | `--representation bf16` |
| `fp16` | 93 | `--representation fp16` |
| `fp32` | 108 | `--representation fp32` |
| `gguf` | 10 | `--representation gguf` |
| `iosurface` | 17 | `--representation iosurface` |
| `mlx-affine` | 5 | `--representation mlx-affine` |
| `packed` | 27 | `--representation packed` |
| `pixelbuffer` | 8 | `--representation pixelbuffer` |
| `q4` | 21 | `--representation q4` |
| `q8` | 28 | `--representation q8` |
| `w8a8` | 7 | `--representation w8a8` |

## boundary

| 标签 | 页数 | 查询 |
| --- | --- | --- |
| `cpu-gpu` | 38 | `--boundary cpu-gpu` |
| `gpu-ane` | 35 | `--boundary gpu-ane` |
| `producer-consumer` | 210 | `--boundary producer-consumer` |
| `request` | 104 | `--boundary request` |
| `tensor-pixelbuffer` | 3 | `--boundary tensor-pixelbuffer` |
| `weight-runtime` | 41 | `--boundary weight-runtime` |

## api

| 标签 | 页数 | 查询 |
| --- | --- | --- |
| `private` | 43 | `--api private` |
| `public` | 154 | `--api public` |
| `reference-only` | 203 | `--api reference-only` |

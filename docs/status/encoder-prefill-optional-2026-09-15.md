# Encoder prefill 与 GPU/ANE 可选路径状态（2026-09-15）

## 结论

TurboCider 的默认 execution 仍是 `gpu`。GPU/ANE 没有成为隐式默认：

- denoiser/DiT hybrid 使用 `ane_manifest`，必须由 `gpu_ane` 或经过资格匹配的
  `auto` 路由选择；请求仍需 `allow_approximation=true`；
- encoder-only hybrid 使用独立的 `encoder_ane_manifest`，不会复用或覆盖
  denoiser manifest，并且必须显式设置 `allow_approximation=true`；
- 没有 encoder manifest 时，FLUX、Z-Image、H3 和 LTX encoder 都走精确 GPU
  路径；
- 模型目录接口现在分别报告 `gpu_ane_policy` 和
  `encoder_gpu_ane_policy`。前者为 `optional_manifest_gated` 或
  `unsupported`，后者为 `optional_explicit_manifest` 或 `unsupported`；
- `supports_gpu_ane` 只表示 denoiser/DiT 能力，新增的
  `supports_encoder_gpu_ane` 单独表示 encoder 能力，避免把两类 artifact
  混为一谈。

所有 encoder ANE 路径仍按实验性近似路径报告为
`gpu_ane_experimental`。显式请求在 capability、近似许可、manifest ABI 或
provenance 不满足时 fail closed。LTX Gemma 的逐层 ANE 执行另外保留精确 GPU
MLP fallback；它不会因为某层 Core ML 失败而输出缺失的 partial result。

## 代码整理

本轮只收敛边界，不改 encoder 数学实现：

- 请求解析不再维护一份易失真的模型 ID 白名单，而是读取各模型的
  `supports_encoder_gpu_ane` capability；
- encoder telemetry 按模型区分 Qwen3、Qwen3-VL 和 Gemma4。LTX 现在正确报告
  `metal_mps(+coreml)`、`gemma4_encoder_*` 与
  `gemma4_encoder_mlp_int8_per_channel`，不再错误标成 Qwen3/MLX；
- 新增的 LTX isolated MLP probes 和 H3 encoder benchmark/hybrid probes 不再
  强制进入默认 native build。需要这些实验工具时显式设置：

  ```sh
  TURBOCIDER_BUILD_EXPERIMENTAL_PROBES=1 \
  MLX_ROOT=/path/to/mlx \
    bash tools/native/build.sh
  ```

  默认值为 `0`。该开关只影响四个 benchmark/probe 可执行文件，不移除运行库
  中 manifest-gated 的 encoder hybrid 支持，也不改变默认 GPU executor。

## 当前覆盖

| Consumer | Encoder | GPU 工作 | 可选 ANE 工作 | 当前决定 |
| --- | --- | --- | --- | --- |
| FLUX Klein 4B/9B | Qwen3 | attention、MLP suffix、consumer taps | 连续 MLP prefix | 已通过若干 full-encoder buckets；仍需显式 encoder manifest |
| Z-Image / GGUF | Qwen3 | attention、MLP suffix、tap 34 | 连续 MLP prefix | 已通过若干 full-encoder buckets；仍需显式 encoder manifest |
| MiniMax H3 MLX / VSA / VDN | Qwen3-VL | MRoPE、attention、剩余 MLP/layers | exact-256 的 50-layer MLP prefix bank | 历史 run 通过，但本轮同口径复跑未复现；保持 explicit optional，其他长度回 GPU |
| LTX 2.5 | Gemma4 ConvRot | fused gated MLP、GPU taps、未覆盖 layers | 当前 8-layer MLP prefix bank | 质量通过但未达到 1.1x；不晋升 |

FLUX/Z-Image 与 H3/LTX 使用不同 encoder ABI。LTX hidden=3840、
intermediate=15360 的 Gemma4 ConvRot 路径不能复用 Qwen3 artifact 或 telemetry。

## 2026-09-15 实测状态

这些数字是 Apple M4 Max 上的 resident encoder-prefill 数据，不是完整图片或
视频生成的端到端加速结论。

- FLUX Klein：64–2048 tokens 中保留 256/512/1024/2048 buckets；已记录的
  retained warm speedup 最高约 1.256x。
- Z-Image：64/128/256/512/1024/2048 buckets 均通过当轮 full-encoder
  quality、stability 与 runtime gates，warm speedup 约 1.315–1.445x。
- H3 Qwen3-VL：exact-256 50-layer 路径曾在当日较早的 qualification 中通过。
  hot-cache matched run 为
  4.57608 s 对 5.11944 s（1.119x）；更长 settling run 的 median 为
  4.94330 s 对 10.37870 s。64/128 tokens 以
  `below_min_profitable_rows` 回 GPU，512 以
  `manifest_capacity_exceeded` 回 GPU。Core ML load 约 29.606 s，因此不能把
  resident prefill 结果表述成首请求或端到端收益。
- LTX Gemma exact-128 settled run：resident fused GPU+taps 为 0.508186 s，
  persistent hybrid 为 0.499790 s，仅 1.0168x。8 个 ANE layers 全部成功、
  0 fallback、caller-owned backing 与完整 video/audio/mask quality 均通过，
  但仍低于 1.1x 门槛，所以当前 topology 保持 optional，不继续自动推广。

LTX settled 和 H3 settling 的原始 report 保留为本机临时证据，不提交进仓库，
也不作为可移植 artifact。本文只记录可复核的 workload、聚合指标和采用决定。

## 请求示例

只启用 encoder ANE、保持 denoiser 在 GPU：

```json
{
  "model": "flux2-klein-4b",
  "execution": "gpu",
  "allow_approximation": true,
  "encoder_ane_manifest": "/path/to/qwen3-encoder-manifest.json"
}
```

同时启用 denoiser 和 encoder hybrid 时必须提供两个独立 manifest：

```json
{
  "model": "flux2-klein-4b",
  "execution": "gpu_ane",
  "allow_approximation": true,
  "ane_manifest": "/path/to/dit-manifest.json",
  "encoder_ane_manifest": "/path/to/qwen3-encoder-manifest.json"
}
```

LTX 的 `encoder_ane_manifest` 指向 `block-XX/manifest.json` bank 根目录或对应
入口；它和 LTX denoiser 的 `ane_manifest` 不是同一个 artifact。

## 验证边界

代码整理后的最低验证集：

```sh
Python/bin/python -m pytest -q \
  tests/native/test_benchmark_ltx_gemma_encoder.py \
  tests/native/test_ltx_gemma_prefill_policy.py \
  tests/native/test_qwen3_prefill_policy.py \
  tests/native/test_h3_mlx_cache.py \
  tests/native/test_h3_mlx_source_contract.py

TURBOCIDER_NATIVE_ONLY=1 \
MLX_ROOT="$PWD/Python/lib/python3.13/site-packages/mlx" \
  bash tools/native/build.sh
```

需要重新编译并运行 isolated experimental probes 时，在 build 命令上增加
`TURBOCIDER_BUILD_EXPERIMENTAL_PROBES=1`。完整硬件 sweep 应复用已有 artifact，
避免在磁盘空间紧张时重新导出大型 Core ML bank。

本轮整理后的实际结果：

- `git diff --check` 与 `bash -n tools/native/build.sh` 通过；
- 默认 `TURBOCIDER_NATIVE_ONLY=1` native build 通过；
- `TURBOCIDER_BUILD_EXPERIMENTAL_PROBES=1` 的 opt-in native build 通过，四个
  experimental probes 均成功生成；
- encoder-prefill 定向回归：115 passed、1 skipped、71 subtests passed；
- 完整 `tests/native`：282 passed、7 skipped、141 subtests passed。Unix socket
  与 AVFoundation 用例需要沙箱外系统能力，最终统一结果已在允许这些系统
  能力的同一主机上复跑通过。
- H3 exact-256 真实 Metal/Core ML 复跑保持相同输出质量与执行完整性，但没有
  稳定复现早先 speedup。缩短版 2-sample smoke 为 9.2750 s hybrid 对
  10.1993 s exact GPU（1.0996x），只比 1.1 门槛低约 0.03%；同既有口径的
  7-run/discard-3 复跑为 10.2683 s hybrid 对 10.2295 s exact GPU（0.9962x）
  和 10.2715 s fused-SDPA GPU（1.0003x）。400 次 ANE 调用全部成功，runtime
  failure 和 output copy 都为 0，relative L2 为 0.001280、cosine 为
  0.99999921。默认 GPU 与早先 10.2153 s reference 只差约 0.14%，没有明显
  主路径回归；未复现的是可选 hybrid 的缓存/系统状态相关收益。因此最新结论
  是继续保持 explicit optional，不把 H3 exact-256 自动晋升。

整理后的 H3 smoke 和 qualified 原始报告同样只保留为本机临时证据，不提交进
仓库。上面的样本数、计时和质量聚合值构成本文的可移植记录。

## 尚未完成

- LTX 8-layer ANE bank 不满足 1.1x full-encoder 门槛；除非新 topology 先证明
  可能获得完整 encoder 收益，否则不扩展 artifact。
- H3 exact-256 的收益需要在可控缓存/负载条件下重新证明；其他 sequence
  buckets 也尚未 qualification。
- FLUX/Z-Image 的通过结果仍是 encoder-prefill 资格，不代表完整 generation
  的同等 speedup，也没有让 encoder ANE 成为默认。
- 不存在 same-checkpoint、same-token、same-layer、same-output 的 patched oMLX
  对照；Qwen3.5 language-model prefill 数字不能作为 LTX/H3/FLUX/Z-Image 的
  直接基线。

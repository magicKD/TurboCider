# TurboCider Native

C++/Objective-C++ 推理库、SwiftUI App、C/Swift SDK、CLI 与 Unix socket 服务。FLUX.2 Klein 4B/9B 支持文生图、图生图、1–8 张参考图编辑；MiniMax H3 有 manifest-gated native executor；FastMetal 1.3B QAD 有显式配置的持久 MLX/TAEHV worker；LTX 2.5 的 video-only 文生视频已开放 native executor，I2V 与音频仍按能力门禁。除 FastMetal 的显式 Python worker 外，FLUX/H3/LTX 原生推理不依赖 Python 模型运行时，也不会自动下载模型。

## 构建与发行目录

```sh
make setup
make build
make package
```

需要兼容的 MLX C++ 0.32.x、完整 Xcode 和 Apple Silicon arm64。通过 `DEVELOPER_DIR`/`SDKROOT` 选已有编译器；`tools/native/dependencies.sh` 优先使用项目托管依赖，也接受显式 `MLX_ROOT`。构建脚本会读取所绑定 `libmlx.dylib` 的 deployment target，同时用于 C/C++/Objective-C++、Swift 和发行包最低系统版本，可用 `TURBOCIDER_DEPLOYMENT_TARGET` 显式覆盖。构建不下载依赖。输出 `dist/TurboCider.app` 和 `dist/cli/`，包含 MLX dylib/metallib、H3/LTX shader 和 LTX clean-exec helper；本机 ad-hoc 签名不等于 Developer ID 公证。模型保持在用户选择的原目录。

## 请求与 CLI

```sh
build/native/turbocider doctor
build/native/turbocider models
build/native/turbocider plan request.json
build/native/turbocider generate /path/to/FLUX.2-klein-4B request.json
build/native/turbocider batch /path/to/FLUX.2-klein-4B first.json second.json
```

LoRA preparation uses the existing audited `h3.c` merge implementations rather
than duplicating safetensors/ConvRot arithmetic inside the native inference
library. The resulting checkpoint and provenance manifest are then consumed by
the native H3/LTX sessions:

```sh
build/native/turbocider prepare-lora h3 \
  /path/to/FL2VA/transformer /path/to/h3-lora.safetensors \
  /path/to/merged-transformer --device mps
build/native/turbocider prepare-lora ltx \
  /path/to/ltx-2.5-22b-dev-transformer-comfy-int8-convrot.safetensors \
  /path/to/ltx-refiner-lora.safetensors \
  /path/to/ltx-2.5-22b-dev-refiner-lora-0.8-comfy-int8-convrot.safetensors \
  --device mps
```

Use `--check-only` first for a shape/identity-only pass. Preparation may use
Python and MPS; native generation does not. H3 and LTX inference remain
fail-closed until the generated manifest, base, adapter, and merged output
hashes all match the request.

推荐 schema 2（schema 1 仍兼容）：

```json
{
  "schema_version": 2,
  "model": "flux2-klein-4b",
  "operation": "image.edit",
  "inputs": [
    {"kind": "text", "role": "prompt", "text": "把这只狐狸放在雪地里，保持毛色与外观"},
    {"kind": "image", "role": "reference", "path": "/absolute/reference.png"}
  ],
  "outputs": [{"kind": "image", "path": "/absolute/result.png", "width": 512, "height": 512}],
  "sampling": {"steps": 4, "seed": 42},
  "execution": {"policy": "gpu", "residency": "resident"},
  "parameters": {"dynamic_text": true}
}
```

文生图使用 `image.generate`，仅保留 prompt 输入。图生图使用 `image.transform`，一张图片角色 `init_image`，可附 `strength: 0.5`。strength 表示保留原图程度；正值从 `max(1, floor(steps*strength))` 的采样阶段开始，1 不执行 DiT，0 执行全部步骤。编辑参考图的 strength 不用于加噪控制；顺序影响 reference 位置编码。

stdout 输出最终结果 JSON，stderr 输出事件 JSON。Ctrl-C 在安全边界取消，返回码 2；导出开始前可取消，文件原子提交后返回成功。使用唯一输出路径以保留历史。`batch` 复用同一会话，要求模型一致。动态文本最大 512 tokens；尺寸 64–2048 且 16 倍数，实际受内存预算限制；种子 0–2147483647，步数 1–50。

## FastMetal 1.3B QAD

FastMetal 是固定形状的 3-step、16 fps、`4n+1` 帧视频路径。它要求一个显式 profile（`TURBOCIDER_FASTMETAL_CONFIG`，或模型目录下的 `turbocider-fastmetal.json`），profile 绑定 Python、FastVideo engine、上游 entrypoint、TurboCider worker，以及可选的 Core ML ANE bridge；示例见 [`profiles/fastmetal.example.json`](../profiles/fastmetal.example.json)。没有 profile 或依赖不完整时，Session 会 fail closed，不会退回伪造媒体。

GPU+ANE 只接受固定的 30-block manifest：`rows=32760`、`hidden=1536`、`intermediate=8960`、ANE/GPU split=`4096/4864`，并校验 checkpoint 与 packed `mlx_dit.json` 身份。worker 在 Session 内持久复用 DiT、TAEHV decoder 与 prompt cache；ANE bindings 在每次 denoise 前按需激活、在 decode 前释放，避免 Core ML/compiled shards 占用统一内存拖慢 TAEHV。取消会终止并重建 worker，视频通过临时文件原子提交。结果中会记录 `compile_enabled`、`denoise_step_s`、decoder 分段耗时、worker RSS 和 ANE 调用指标，便于区分模型计算与框架开销。

FastMetal 接受至多一个 `transformer` LoRA，但不会把任意 adapter 动态注入 MLX。请求中的 LoRA 必须有同名 sidecar（`ADAPTER.safetensors.manifest.json`），或同目录的 `fastmetal-lora.manifest.json`；manifest schema 为 `turbocider-fastmetal-premerged-lora-v1`，绑定 `FastVideo/FastMetal-1.3B-QAD` revision `2dac0154b217adabf8895d6cde7d6d93e68b7bec`、固定 base `mlx_dit.safetensors/json`、LoRA 文件/role/strength、merged checkpoint 两个文件、完整 mapping、固定 shape 与 MLX INT8 config。所有 artifact 都按大小和 SHA-256 校验，worker 通过独立的 `--mlx-checkpoint` 加载 merged DiT，model root 仍只服务 tokenizer、text encoder 和 VAE。可先用只读 ABI `tc_fastmetal_lora_preflight_json` 检查安装；生成结果会报告 checkpoint/LoRA SHA、strength 和 `lora_fusion=premerged_manifest_verified`。当前 LoRA 只开放 GPU；`gpu_ane` 会 fail closed，直到安装与 merged checkpoint SHA 对应的完整 ANE artifacts。

FastMetal base checkpoint 的真实 GPU/GPU+ANE latent parity 已与对应 direct baseline 逐元素一致。当前同热状态成对复测中，GPU worker 为 76.90 秒、direct 为 78.12 秒；GPU+ANE 两个 direct 样本中位数约 74.78 秒，三个优化后 TurboCider 完整 callback 样本中位数同为约 74.78 秒，暖请求为 74.29 秒，满足 5% 回归门槛。历史 72.93/69.97 秒绝对记录在当前热状态未复现，因此仍需多机器、冷暖交错矩阵，不能宣称普遍更快。FastMetal LoRA 尚无真实 merged checkpoint，因此只有 resolver/worker contract 验证，不能声称 LoRA 质量、latent parity 或性能已验证。LTX video-only 已有公共 executor；I2V 数值 parity、音频 Session parity、LoRA merged checkpoint 和完整 GPU+ANE 性能矩阵仍未完成。

## LTX 2.5（公共 video-only executor）

LTX 2.5 的 `video.generate`、`audio=false` 计划返回 `executable=true`，并默认选择 `component_staged`。单次 CLI 和 service worker 都直接复用 ltx-mac 的生命周期：Transformer 完成后 `exec` 到已有 C++/MLX Video VAE finalizer，使 decoder 不继承 denoiser 的 Metal/MPSGraph allocator 状态。CLI 默认把 conditioning cache 放在用户 Caches 目录；service 放在任务状态目录。两者均按 checkpoint/Gemma/tokenizer/prompt identity 绑定。`video.image` 与 `audio=true` 计划仍返回 `executable=false`，因为 I2V 数值 parity、音频资产 provenance 和完整音频 Session parity 尚未完成。

M4 Max 64 GB、704×448、97 帧、24 fps、原始 8+3 schedule 的公共 CLI 实测：首次动态 Gemma 为 102.83 秒；相同 prompt 再次执行命中 connected-conditioning cache 为 81.37 秒。cache-hit 中 checkpoint/model 建立约 22.75 秒，扣除后从已加载模型到最终 MP4 的链路约 58.61 秒；ltx-mac 的 matched decoded-pixel 参考为 59.483 秒，两者处于同一性能水平。干净 exec finalizer 的 Video VAE 为 2.84–2.97 秒；旧 CLI 子进程方案为 16.35 秒。

模型目录可通过 C ABI 的只读 `tc_ltx_audio_preflight_json(model_path, ...)` 检查音频资产。它只接受固定候选文件名，并要求旁置 `turbocider-ltx-audio-assets-v1` manifest，校验 Lightricks/LTX-2.5 revision、artifact 大小/SHA-256、Audio VAE/base vocoder/BWE/mel-STFT 组件、16/48 kHz 和双声道声明。返回 `assets_verified` 不等于音频 executor ready；即便 manifest 完整，音频 operation 仍保持关闭，直到端到端 Session parity 和所有 artifact provenance 门禁完成。当前本机 `models/LTX-2.5/vae/ltx-2.5-audio-vae-bf16.safetensors` 是共享目录的符号链接，能看到完整 safetensors 组件，但尚无 TurboCider provenance manifest，因此 preflight 状态为 `unverified`。

Audio VAE 的 native runtime 已加入 `native/models/ltx_runtime/ltx_mlx_audio_vae.cpp`，实现 latent→mel 的 MLX/Metal 权重加载、因果 Conv2d、PixelNorm、残差块和时间/频率上采样；工具为 `build/native/ltx-audio-vae-decode CHECKPOINT INPUT_BF16 BATCH TOKENS OUTPUT_BF16`。它需要可见 Metal device，输出 `[B,2,4*tokens-3,64]` BF16 mel。已在可见 Metal 环境对 `[1,101,128]` 真实 latent 做逐阶段 Python oracle 对比：12 个阶段全部 `max_abs=0`、逐元素 100% 一致。

16 kHz base vocoder 与 48 kHz BWE 已作为独立 runtime 加入，分别使用 667 与 560 个 resident FP32 tensors；BWE 工具为 `build/native/ltx-bwe-decode CHECKPOINT WAVE16_F32 BATCH SAMPLES OUTPUT48_F32`。真实 Metal 对比中 STFT/mel/skip 逐元素一致，生成器 stage 最大误差约 `1.21e-5`，residual/最终 48 kHz waveform 最大误差约 `5.6e-8`。`native/media/audio.mm` 负责有限值/clip 检查、原子 WAV 写出、视频时长同步和 AVFoundation AAC mux；独立媒体测试验证 97 帧@24fps 的 H.264/AAC 轨道时长均为 `4.041667` 秒。

## 设备配置与编译缓存

SDK 缺省为 GPU；App 的 `auto` 在本机、权重、桶、MLP 分区和近似许可匹配时选择自有 GPU/ANE 分区，否则回退 GPU。当前自动 GPU+ANE 仅对通过完整端到端门槛的 exact device profile 开放：M4 Max 64 GB 的 FLUX 4B a6144 与 Z-Image base a4096，M4 Pro 48 GB 的 FLUX 4B full-MLP profile。请求 `execution.profile` 可指向本地 JSON。`profiles/apple-m4-pro-48gb.example.json` 使用完整 ANE MLP，`profiles/apple-m4-max-64gb.example.json` 使用 6144-channel FLUX 前缀和 4096-channel Z-Image 前缀并由 GPU 并行补算后缀；两者默认关闭，复制后填写 artifact 路径并显式启用。配置覆盖请求的 policy/residency，计划含配置内容 hash。可控制 allocator cache、预算和 Core ML warmup 次数。带独立 LoRA 时基础 ANE artifact 不再匹配，App 会安全使用 GPU；只有 provenance 完整的 LoRA-bound manifest 才能显式使用混合路径。

混合 `gpu_ane` 必须 `allow_approximation=true`，使用本地 schema 2 `ane_manifest`。当前支持20个 single block MLP、K=N=3072、单固定桶。量化 MLP 改变算法精度，结果明确标注；公开 `cpuAndNeuralEngine` 不保证子图全部实际驻留 ANE。旧 artifact provenance 只有源路径/大小，故仍为实验。超过 bucket 明确失败，不裁剪输入、不静默改 GPU。

```sh
build/native/turbocider compile-coreml /path/to/block.mlpackage /path/to/cache
```

按源文件 SHA、OS、GPU/架构身份缓存编译结果，带互斥和原子提交。仅编译已有 Core ML 模型，不自动转换/切分任意网络。驻留模式速度优先；`component_staged` 释放阶段权重，FLUX 不支持 block streamed offload。

## 常驻后台服务

```sh
build/native/turbocider serve /tmp/turbocider.sock /absolute/job-state
build/native/turbocider rpc /tmp/turbocider.sock rpc.json
```

RPC 为每连接一个换行终止 JSON，响应 `{ "ok": true, "result": ... }` 或错误。客户端可直接用 Unix socket。支持：

| action | 其他字段 |
|---|---|
| submit | `model_path` 与完整 `request` 对象，返回 job id |
| status / cancel | `id` |
| jobs | 可选 `offset`、`limit`（默认20，最大100） |
| plan | `request` |
| models / doctor | 无 |

单 worker、最多32排队、持久化状态、取消、分页、模型/conditioning 复用。重启将中断任务标记 interrupted，不自动重做。socket 限当前用户；崩溃后可回收同用户 stale socket。App 当前使用嵌入模式，共享队列需要通过服务 RPC。跨进程 GPU 锁使独立嵌入会话冲突返回 busy，不会隐式等待。

## SDK 与 App

`bindings/c/include/turbocider/turbocider.h` 提供 ABI 1。`tc_engine_create_model` 选择注册模块，`tc_engine_generate` 同步执行；返回字符串必须 `tc_string_free`。回调同步发生在生成线程，不能阻塞或重入生成；Engine 只能在所有调用完成后释放。`cancel` 可跨线程调用。

`bindings/swift/TurboCiderNative.swift` 提供 async generate、cancel、plan、system/models，主线程不执行推理。取消使用 `engine.cancel()`；Swift Task.cancel 尚未自动映射。App 的 JobStore 共用此接口，持久化历史并恢复 interrupted 状态。App 支持素材选择、图像模式、参数/配置、生成/取消、预览及历史参数复用。

## 开发验收

```sh
python3 tests/native/test_contract.py
build/native/turbocider self-test
build/native/turbocider-lifecycle-test /path/to/model /tmp/new-lifecycle-directory
python3 tests/native/test_service.py --model /path/to/model --output /tmp/new-service-directory
```

H3 的性能回归使用当前 `h3.c` 与 TurboCider native Session 做同模型、同请求、
fresh-process 的顺序 AB/BA 对照。精确模式要求输出 MP4 字节一致，并默认拒绝超过 5% 的
TurboCider 中位数回归：

```sh
Python/bin/python tools/native/benchmark_h3.py \
  --h3-bin ../h3.c/h3 \
  --turbocider-bin build/native/turbocider \
  --model ../h3.c/models/MiniMax-H3-LightX2V-Turbo \
  --output-dir outputs/benchmarks/h3-comparison
```

如需检查请求级 H3 LoRA provenance，再传 `--lora PATH --lora-strength 0.0625`。
direct 与 TurboCider 都读取已经预合并的同一 Transformer；TurboCider 额外验证请求中的
adapter 大小、SHA-256 和 strength。近似 kernel 测试必须同时显式传
`--allow-approximation --allow-output-difference`，不能被记录为 exact parity。

真实推理需要 Metal 权限。生命周期目录应为空。oracle 工具需要已有 mflux/MLX/transformers 的开发 Python 环境并强制离线：

```sh
/path/to/dev-python tools/native/flux_reference.py --model MODEL --request REQUEST --output REFERENCE
/path/to/dev-python tools/native/compare_tensors.py REFERENCE CANDIDATE --require-exact --report parity.json
/path/to/dev-python tools/native/benchmark_comparison.py --model MODEL --engine ORIGINAL_ENGINE --mflux MFLUX --manifest MANIFEST --bridge BRIDGE --output BENCHMARK
```

原生请求 `dump_tensors` 指定候选张量目录。性能测试不使用 dump；比较器任何缺失、shape/finite/数值差异均返回失败。最新证据与限制见 [重构状态](design/rewrite-implementation-status.md)、[性能对比](design/flux-performance-comparison.md)、[视频模型验收](design/video-model-acceptance.md)。

## Studio App

图像与视频创作、图片插入/粘贴、有序参考、随机种子、独立 LoRA 文件、模型选择及模型 Load/Unload 的已实现范围与测试入口见 [Studio 实现记录](design/app-studio-implementation.md)。模型中心根据 native descriptor 切换 FLUX 4B/9B、H3、LTX 和 FastMetal 的操作与默认帧参数；视频输出使用 MP4 预览，模型自身的 profile/provenance/audio 门禁仍由 native runtime 最终校验。

模型页现支持当前配置加载、无输出完整预热、卸载、GPU/ANE 选择，以及已有 Core ML 分区的预编译/缓存清理。接口与复现见 [模型准备与性能](design/model-preparation-and-performance.md)。

Core ML 模型/磁盘管理：`turbocider coreml request.json` 支持 `inventory`、`export`、`compile`、`delete_artifacts`、`clear_compiled`、`clear_runtime`。清理默认只预览；配置、请求范例及 C/Swift API 见 [资源管理说明](design/coreml-artifacts-and-storage.md)。

项目独立性、托管依赖与隔离验收见 [独立部署说明](design/standalone-project.md)。原始模型可以位于任意用户指定目录；自动加速产物不再从相邻 workspace 发现。

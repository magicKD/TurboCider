# TurboCider 当前目标状态

更新时间：2026-09-08

验证机器：Apple M4 Max，64 GB unified memory，MLX 0.32.2

## 总结

当前 `dev` 已经把 Z-Image GGUF、LLaDA-Image-Turbo 原生 C++/MLX、Z-Image ConvRot packed Q8 GPU 和原生 ConvRot Core ML/ANE 候选接入 TurboCider。LLaDA 正式 descriptor 只公开自包含的原生文生图；尚需外部 Python/reference source 的单参考图编辑已从发行能力移除，仅保留显式开发诊断入口，不再扫描兄弟仓库或随正式 package 分发。LLaDA 1024² 原生 GPU+ANE 已完成真实 resident E2E，达到相对匹配 GPU 的 1.203×，因此可以作为显式近似候选；仍不自动覆盖 exact GPU 路径，直到多机器/多 seed 矩阵完成。Z-Image BF16 基础模型仍是当前最快的稳定路径；ConvRot 原生 GPU+ANE 相对同一 ConvRot GPU 达到 1.249×，但仍比已有 BF16-derived ANE 路线慢 8.65%，因此保持显式候选。

完整目标尚未达成。公共 `lora_strategy` 三模式契约现已贯穿 schema 1/2、descriptor、plan/result、Swift 和 Studio App；量化 Z-Image 保持 packed base 的原生 inference-time 低秩分支已经可以真实生成，并通过 strength=0 精确退化检查，但 256² 对内存融合的最终 latent correlation 仍只有 0.986464，因此继续保持显式候选。Q4_K_M 官方独立 LoRA 在补齐 `--cfg-scale 1.0` 并统一双方 100 ms polling 后，严格 ABBA×2 的 TurboCider/direct 比值为 0.999417、4/4 输出逐像素一致，已通过该 workload 的“不慢”门禁。LLaDA GPU+ANE 已在当前单机 1024²验证中达到 1.203×；主要缺口变为它的多机器/多 seed 门禁、packed LoRA 的质量门禁、FLUX/Z-Image LoRA 异常和取消矩阵，以及更多机器和尺寸的重复性能验证。

## 本轮架构与发行边界收口

- 已将 `gpu_ane/mac_transformer@9f322e1` 的 tensor/sequence/head/CPU/GPU/ANE、public/private ANE 和 1/multi-block 结论整理为 [Transformer 异构并行技术报告](../design/transformer-heterogeneous-report.md)。报告明确区分 hot operator、fresh process 和模型 E2E，不把 1.5–2.6× MLP micro 写成 H3/LTX 点击生成加速。
- 已补充 [Core ML / ANE 启动报告](../design/coreml-ane-startup.md)：编译、load、interface/backing、zero-input warmup、first/subsequent prediction 分开计时；`prepare(load-only)` 能前移首请求开销，但不能减少 prepare+generate 总工作。
- 已补充 [vpipe/H3/LTX 量化与 streaming 对照](../design/quantized-streaming-vpipe-comparison.md)：Z-Image GGUF streaming 已能降低约 36.1% physical footprint；H3 已有 BF16 双 slot 后台 `pread`，LTX 仍缺 per-block streaming 和动态 pinned prefix。
- private ANE 已严格移入实验边界：正式 `native/` 删除调用 `_ANEInMemoryModel*` 的 bridge/MLP/linear 实现，产品构建链接 `h3_ane_disabled.c`；真实研究实现只留在 `experimental/video/h3/vendor`。完整 build 后 `libturbocider.dylib` 中没有 `_ANEInMemoryModel`、`_ANERequest`、`_ANEIOSurfaceObject` 或 AppleNeuralEngine 未解析符号。
- `tools/native/build.sh`、`tests/repository/test_layout.py` 已加入静态 fail-closed 回归；public H3 Core ML、其他 native 模型和 Swift App 在移除 private bridge 后重新构建通过。

## 目标逐项状态

| 目标 | 当前实现 | 证据 | 未完成 |
|---|---|---|---|
| Z-Image GGUF 多量化 | mixed K-quant 使用常驻 stable-diffusion.cpp Metal；Q8_0/Q4_0/Q4_1 使用 native MLX affine quantized matmul；F16/BF16/F32 可走 native floating 路径 | Q3_K_S、Q4_K_M、Q8_0 已真实生成；Q4_K_M base 1024² TurboCider/direct 为 179.6199/179.7119 s；Q4_K_M LoRA 256² 为 13.1163/13.1240 s，均逐像素一致且严格不慢 | 更多量化的 native ANE、LoRA 1024²/多 seed 矩阵 |
| GGUF GPU+ANE | Q8_0 使用 checkpoint-bound a4096 FFN 分区 | Q8 GPU 45.9634 s，GPU+ANE 37.5848 s，1.223× | 自动设备策略、多尺寸、Q4_0/Q4_1 和 floating GGUF 矩阵 |
| GGUF LoRA | mixed K-quant 使用 sd.cpp 请求期独立文件；native-compatible GGUF 支持内存 delta，并可在 native GPU 开关启用时使用 packed-base inference-time A/B；LoRA-bound Core ML manifest | Q4_K_M request-time LoRA 严格 ABBA×2 ratio 0.999417、4/4 pixel exact；native 路线官方 LoRA 命中 238 projection；ConvRot packed A/B 把 active MLX 降低 46.7% | Q8 LoRA GPU+ANE 65.02 s 慢于 GPU 55.76 s；packed A/B 仍未通过严格 parity，不能自动启用；Q4 LoRA 1024²待补 |
| LLaDA 文生图 | 原生 C++/MLX tokenizer、QueryFormer、MoE 文本编码器、DiT、4-step stochastic scheduler、Flux2 VAE、PNG；GPU block compile；大图 staged text | 1024² native GPU warm `17.971 s`；官方 PyTorch/MPS `19.601 s`；TurboCider 快约 `8.3%` | 多机器/多 seed 质量矩阵；LoRA；编辑原生化 |
| LLaDA 图像编辑 | 正式 descriptor 不再公开；旧兼容 worker 仅能通过显式开发环境变量调用，不进入 package | 历史诊断为 TurboCider 52.64 s load / 28.39 s first、PNG correlation 0.94047 | 需要原生 reference-image conditioning/DiT/VAE 全链路后才能重新开放 |
| LLaDA GPU+ANE | 2 个 noise refiner + 30 个主 block 的 FFN `[0,4096)` 使用 checkpoint-bound Core ML，GPU 计算 `[4096,10240)`；原生 MLX/Core ML 进程内 fork/join；支持 256²和 1024² text-to-image | 1024² native GPU warm `17.971 s`、GPU+ANE warm `14.934 s`，speedup `1.203×`；PNG correlation `0.995194`、cosine `0.998969`、MAE `2.649/255`；Core ML output copy `0` | ANE residency 仍只能报告 `unknown`（public API）；多机器/多 seed 及自动策略门禁尚未完成；LoRA 未实现 |
| LoRA 三模式 | `disk_premerge`、`in_memory_merge`、`inference_time` 已进入 schema 1/2、descriptor、plan/result、Swift 和 Studio；GGUF 策略会实际选择 sd.cpp 或 native MLX；native packed projection 可追加 FP32 累积的低秩 A/B | 自动解析、显式不支持策略和无 adapter 的显式策略均 fail closed；strength=0 对 packed base PNG 逐像素一致；保留旧 `lora_fusion` 兼容字段 | packed A/B 的 1024²、多 seed 和 ComfyUI/Unsloth parity；完整多 adapter、取消和异常 E2E 矩阵 |
| ConvRot 原生 ANE | exporter 直接读取 I8/F32 scale/U8 metadata，在 Core ML 图内执行 grouped H256；GPU 使用 packed affine Q8 | 1024² ConvRot GPU 40.7288 s，GPU+ANE 32.5979 s，1.249× | 仍慢于 BF16-derived ANE 的 30.0014 s；LoRA-bound ConvRot ANE 未过完整门禁 |

## Z-Image ConvRot GPU

Comfy ConvRot 的 signed INT8 权重被映射成 MLX affine Q8：权重存为 `q + 128`，每 32 个值重复 tensor-wise row scale，并使用 `-128 * scale` bias。推理期执行 grouped H256 后直接调用 `quantized_matmul`，不再为每个 projection 物化完整 FP32 权重。

1024×1024、9 steps、resident：

| 路径 | Warm E2E | Active MLX | 判断 |
|---|---:|---:|---|
| 旧 ConvRot dense dequantization | 43.4134 s | 6.53 GB | 对照 |
| packed Q8、BF16 scales | 40.7288 s | 6.53 GB | 1.066×；当前 ConvRot GPU 默认实现 |
| packed Q8、FP32 scales | 45.92 s | 约 6.53 GB | 更慢，仅保留诊断开关 |
| BF16 compiled GPU | 36.3685 s | 约 12.69 GB | 更快但内存更高 |

packed Q8 对旧 ConvRot PNG correlation 为 0.999596，cosine 为 0.999949。

## Z-Image ConvRot GPU+ANE

原生模式不在导出阶段永久 derotate 权重，而是在每个 Core ML FFN 图内对输入和 `w2` 输入执行 grouped H256。ANE 负责 intermediate `[0,4096)`，GPU 负责 `[4096,10240)`；attention 先在 GPU 完成，然后才在同一个 FFN 输入处分叉。

1024×1024、9 steps、resident：

| 路径 | Cold | Warm 中位数 | 相对 ConvRot GPU |
|---|---:|---:|---:|
| ConvRot packed Q8 GPU | 42.0264 s | 40.7288 s | 1.000× |
| ConvRot native a4096 GPU+ANE | 36.4961 s | 32.5979 s | 1.249× |

GPU+ANE 每请求 288 次 Core ML 调用，output copy 为 0，active MLX 约 6.56 GB。相对匹配 GPU 的 PNG correlation 为 0.999659、cosine 为 0.999957。它通过了相对自身 GPU 的 1.2×目标，但比现有 BF16-derived a4096 的 30.0014 s 慢 8.65%，所以不替换当前全局最优默认。

## LoRA 状态和可靠性

FLUX 与 safetensors Z-Image 接受最多八个独立 LoRA 文件，并根据 path、bytes、SHA-256、role 和 strength 识别身份。基础 GPU 路径在权重加载时把 delta 融合到内存，不写 merged checkpoint。ANE 不能复用基础 artifact：必须使用绑定相同 LoRA identity 的独立 Core ML artifact，否则 manifest 校验拒绝执行。

Z-Image 官方 BF16 LoRA 已完成 1024² ComfyUI parity，238 个 projection 全部应用；latent cosine 0.998929，PNG correlation 0.999156。ConvRot + 官方 LoRA 已完成 256²真实运行，warm 2.6348 s，对 BF16+LoRA PNG correlation 0.993584、cosine 0.999315。

ConvRot/GGUF 的内存融合会把 LoRA 命中的量化 projection 解量化为 dense BF16。官方 LoRA 命中 238 个 projection 后，ConvRot active MLX 从约 6.53 GB 增到 12.52 GB。新的 `inference_time` 分支保留 packed base projection，并在输出处追加 FP32 累积的 `x @ A.T @ B.T * scale`；同一 256²请求 active MLX 为 6.67 GB，降低约 46.7%。其 warm wall 为 2.9408 s，而内存融合为 2.6349 s，慢约 11.6%；PNG correlation/cosine 为 0.978807/0.997737，最终 latent correlation/cosine 为 0.986464/0.986503。strength=0 与 packed base PNG 逐像素一致。该结果证明独立文件、低内存和非零效果，但尚不足以通过默认质量门禁。

LLaDA 原生 executor 仍拒绝 LoRA；Core ML bridge 的 provenance 只绑定基础 Transformer checkpoint。旧兼容 worker 也拒绝 LoRA，仅作为显式开发诊断入口保留。统一契约会在 descriptor 中把 LLaDA 的策略列表保持为空，不能把 LLaDA 的 hybrid artifact 当作支持 LoRA 的证据。

2026-09-08 的代码审计还发现，共享 `Weights::apply_loras` 原先边验证边替换权重；后续 adapter 失败或取消时可能保留部分 delta。当前分支已改成失败时清空整个 component，下一次请求必须从不可变 base checkpoint 重载，避免同 identity 重试运行部分融合权重。该方案不保留第二份 checkpoint-sized rollback copy。

## 当前推荐配置

- Z-Image BF16 base、Apple M4 Max 64 GB、1024²/9 steps：`auto` + checkpoint-bound a4096 manifest，warm 约 30.0014 s。
- Z-Image ConvRot base：需要低 active MLX 内存时可显式使用 native a4096 GPU+ANE，warm 约 32.5979 s。
- Z-Image 带 LoRA：质量优先使用默认 `in_memory_merge` GPU；内存受限时可显式尝试 packed `inference_time`，但当前有约 11.6% warm wall 代价且 parity 未过默认门禁；只有匹配 LoRA identity 的 artifact 才显式使用 GPU+ANE。
- Z-Image mixed K-quant GGUF：继续使用常驻 sd.cpp Metal；Q8_0 可显式选择 native MLX GPU/GPU+ANE。
- FLUX 4B base、M4 Max 64 GB、512²/4 steps：`auto` + a6144，warm 1.6279 s，相对 GPU 1.392×。
- LLaDA 文生图：默认使用原生 C++/MLX GPU；需要显式设置 `gpu_ane`、匹配 manifest 和 `allow_approximation=true` 才会启用当前单机达到 1.203× 的近似候选。`image.edit` 不属于当前正式自包含能力。

## 验证与未完成项

最新 managed Python 定向测试为 80 passed、47 subtests passed；`make test`、native CLI/library、Swift App、integration test binary 重建和 App behavior test 通过（当前无 pasteboard service，因此剪贴板检查按设计跳过）。原始生成和性能报告位于被忽略的 `outputs/`，可提交摘要位于 `docs/design/validation/`。

本轮 private-ANE 隔离后的增量验证还包括：完整 native/Swift build 成功；`models` 正常列出八个模块；`doctor` 正常返回；`otool -L` 只显示 MLX/JACCL、系统库和公开 Apple frameworks，没有 AppleNeuralEngine private framework。结构化实验索引见 `docs/design/validation/transformer-heterogeneous-2026-09-08.json`，发行隔离证据见 `docs/design/validation/private-ane-shipping-isolation-2026-09-08.json`。

Core ML prepare/lifecycle 已补齐 load-only 与阶段遥测。一次真实 M4 Max 256² Z-Image GPU+ANE 对照中，load-only prepare wall 为 `12.77 s`，其中 manifest/provenance `5.25 s`、32 个 `.mlmodelc` model load 合计 `6.88 s`、共享 output backing `4 ms`；随后 generate wall 为 `11.20 s`。配置 `warmup_iterations=1` 后，零输入 warmup 约 `1.10 s`，后续真实请求约 `10.90 s`；该结果受 OS/Core ML cache 状态影响，只作为启动开销拆分证据，不是多轮性能门禁。当前没有把 Core ML `prepare` wall 误写成编译时间，真实 `.mlmodelc` compile 仍由独立 cache 操作计时。

LLaDA 原生 C++/MLX session 也已通过公共 `tc_engine_prepare` 支持真正的 load-only 过程；兼容 Python worker 不伪装成可预热的 native session。M4 Max 256² GPU 实测 prepare wall `10.26 s`，随后命中 conditioning cache 的 generate wall `2.97 s`，准备阶段未生成输出文件。对应代码在 `native/platform/apple/llada_session.mm`，完整原始报告为 `outputs/llada-preparation-load-2026-09-08/report.json`（本地忽略）。

跨后端不再以逐 bit/逐像素一致作为唯一正确性定义。新增 `tools/native/quality_gate.py`，默认离线 RGB 门禁为 correlation ≥ 0.99、cosine ≥ 0.995、MAE ≤ 5/255；LLaDA benchmark 会始终记录门禁结果，并可用 `--require-quality` 让未通过结果返回失败。阈值属于可记录、可覆盖的 workload 策略，不会把单一阈值冒充为所有图像/视频模型的普适标准。FLUX load-only prepare 同时修复为返回 resolved `ExecutionPlan`，准备遥测现在与随后实际执行路由一致。

视频路径现已增加流式解码的 `tools/native/video_quality_gate.py`，不把完整 H3/LTX RGB 序列同时载入内存。现有 matched GPU 产物中，TurboCider 对 H3 direct 和 LTX direct 的 MP4 都逐文件一致；但 LTX GPU 对 GPU+ANE 的 704×448、97 帧输出只有 mean correlation `0.8498`、minimum correlation `0.7751`、mean MAE `21.59/255`，虽然 frame-to-frame motion energy 最大相对误差为 `8.04%`，仍明确未过门禁。因此 LTX `auto` 保持 GPU，不得把约 1.20× 的局部/历史加速写成已验收的正式 E2E hybrid。结构化证据见 `docs/design/validation/video-quality-gate-2026-09-08.json`。

重新生成的 `dist/cli` 已复制到源码目录之外，在 `env -i` 仅保留系统 PATH 的环境中成功执行 `models` 和 `doctor`；`libturbocider.dylib` 的非系统依赖仅为 `@rpath/libmlx.dylib` 与 `@rpath/libjaccl.dylib`，发行包 rpath 为 `@loader_path`。App 深度签名校验通过。摘要见 `docs/design/validation/packaged-portable-2026-09-08.json`。

下一阶段按优先级：

1. 将 LLaDA 1024² hybrid 的单机结果扩展到多机器/多 seed，并更新自动策略门禁；当前路径真实 resident warm 为 `1.203×`，但仍保留显式 `gpu_ane`，不能把单机单 seed 结果当成普遍保证。
2. 把已经实现的 packed `inference_time` LoRA branch 推进到默认质量门禁：完成 1024²、多 seed、ComfyUI/Unsloth 和 in-memory parity，并继续优化当前约 11.6% 的 warm wall 代价；mixed K-quant GGUF 继续由 sd.cpp 执行，native-compatible GGUF 可在 native GPU 开关下使用低秩分支。
3. 补齐统一策略下的 provenance、cache invalidation、多 adapter、失败和取消 E2E 矩阵。
4. 优化 ConvRot Core ML 原始 I8/scale 表达和两个 grouped H256 边界，把 32.60 s 降到 30.00 s 以下。
5. 补齐 FLUX 4B/9B、Z-Image BF16/ConvRot/GGUF 的 LoRA 异常、取消、多 adapter、多尺寸和多机器 E2E 门禁。
6. 为 LLaDA 增加独立 LoRA 文件识别、内存融合/inference-time 分支和 provenance-bound artifact 后再评测。

详细数据见 [ConvRot 原生 ANE 验证](../design/validation/z-image-convrot-native-ane-2026-09-08.json)、[GGUF 状态](z-image-gguf-2026-09-07.md)和 [GPU/ANE 并行方案](../design/parallel-acceleration.md)。

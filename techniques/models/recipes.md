# 模型配方：从相似拓扑迁移，而非照搬开关

<!-- knowledge
id: technique-models-recipes
type: guide
stage: encoder, dit, ffn, vae, pipeline
operator: attention, swiglu, conv
backend: metal, mlx, coreml, private-ane
device: gpu, ane
symptom: precision-loss, memory-bound, wrong-layout
semantics: numerically-equivalent, approximate
representation: bf16, fp16, q4, gguf
boundary: request, gpu-ane
api: public, private
-->

先按[模型阶段地图](stage-map.md)和[本地设计](../designs/README.md)理解算法与可运行样本；本篇的具体模型/产品源码路径是额外接入线索，不是离线学习前置条件。

[总目录](../README.md) · [新模型纵切](../guides/new-model.md)

以下是优化入口与历史经验，不是当前所有operation/机器的支持表。实际可用性看当前module、`plan`、manifest和actual result；历史日期/精度/冷暖不能混用。

## FLUX.2 Klein 4B / 9B

起点：[flux2 transformer](../../native/models/flux2/flux_transformer.cpp)。优先compiled single block、Q/K RoPE、attention heuristic、conditioning复用；parallel attention/MLP拓扑适合FFN channel split。ANE返回hidden partial，GPU attention和suffix保留同一source。

4B历史a6144不是所有设备最优；M4 Pro/M5 Pro与Max分别验收。当前Q4来源若走dense-prefix图，要建立来源核验的派生view并与它自己的GPU控制比较，不能直接把packed U32交dense GEMM。

9B既有packed/nativeGPU、预解码/compiled与hybrid都有后续专项，早期“9B无ANE”不能当永久结论。分别读 [FLUX9 hybrid](../../docs/benchmark/FLUX9_HYBRID.md)、[predecode](../../docs/benchmark/FLUX9_PREDECODE.md)、[4B derived状态](../../docs/benchmark/FLUX4_CURRENT_STATUS.md)。预展开可能加速热计算但增加RAM/磁盘和冷成本；不叫低内存。

## Z-Image Turbo / ConvRot / GGUF

起点：[z_image](../../native/models/z_image/z_image.cpp)、[GPU kernels](../../native/models/z_image/metal_kernels.hpp)。attention→FFN串联，合法并行为FFN内部成对channel split。强GPU基线含norm/RoPE/SDPA/compiled与调优SwiGLU；当前默认步数与历史9步实验不同，匹配时显式写steps。

BF16、ConvRot signed Q8、普通GGUF必须与自身source/precision控制比较。ConvRot原H256和signed codes不可套Sylvester recipe；GGUF raw block和MLX affine不同。runtime-weight可省跨层图，但staging/chunk/restore税明显。

M5 Pro24GiB只读GPU suffix/最小bucket与生命周期优化见 [适配记录](../../docs/design/m5-ane-adaptation.md)。caption长度会改变rows，512长caption不一定能用短bucket。历史W prefetch、过大tile、QKV split都有负结果，先读 [排障](../guides/troubleshooting.md)。

## Qwen Image 2.1：生成、编辑与encoder分开

起点：[transformer](../../native/models/qwen21/transformer.cpp)、[encoder prefill](../../native/models/qwen21/encoder_prefill.hpp)、[condition cache](../../native/models/qwen21/conditioning_cache.hpp)。

普通40步base、六步Viggle student、普通LoRA、多参考编辑、prompt enhancement是不同workload。先正确tokenizer/vision resize/rotary/scheduler；跨框架noise/conditioning共享用于定位。

编辑先区分encoder、首步full-prefill和后续prefix-KV-hit。GPU encoder/GQA/vision padding、shared LoRA ranks与jointA/B可帮助整图；首步通道分担后五步GPU是特定候选，不是把整步交ANE。CPU-load检查失败和share排名反转仍保留资格限制。

M5 Pro24GiB的 [layer streaming / compact suffix cache](../../docs/status/qwen21-m5pro-24gb-streaming-ane.md)有明确硬件/operation限制和临时MLX wired-limit恢复合同。它不是系统全局wired设置建议；held-out图仍有明显细节变化，不是普遍无损ANE。

DBCache residual reuse是独立近似，默认off。最新合并对照与cache取舍见 [10月15日记录](../../docs/status/2026-10-15-main-into-dev.md)。

## MiniMax H3：C/Metal、FastH3 INT6、VDN不要混排

起点：[H3 C runtime](../../native/models/h3_runtime/h3_dit.c)、[H3 MLX](../../native/models/h3_mlx/dit.cpp)、[VDN](../../native/models/h3_mlx/vdn.cpp)。

传统H3串联attention→MLP，局部ANE MLP/QKV/row split收益被50层生命周期和first-consumer deadline限制；genericMPSGraph上CPU tail有效不代表专用runtime也能赶上deadline。

FastH3 INT6/g64、dense/VSA checkpoint与VDN另有来源/schedule。参考 [FastVideo parity合同](../../docs/design/h3-mlx-int6-fastvideo-parity-plan.md)：MRoPE精确表达、top-k的double稀疏率、原始token tags、同noise两种音视频head都重要。VSA有reference gather实现不代表Metal SIMD sparse已交付，也不代表比dense更快。

大H3先typed SSD slots、pinned subset和conditioning/DiT阶段释放；完整video/audio VAE和mux纳入时间。TAEH3等轻量decoder若更改输出质量必须独立profile，不能当fullVAE同语义提速。

## LTX 2.5：双流与两阶段的生命周期

起点：[ltx runtime](../../native/models/ltx_runtime/README.md)、[ltx_shaders](../../native/models/ltx_runtime/ltx_shaders.metal)。先Fast A/V双GPUqueue与audio command batching，再阶段1/2生命周期、最后一步逐block释放、cleanexec VAE finalizer、720p tiledVAE。只spawn child而父进程保留巨大资源，不一定恢复cleanexec性能。

GPU/ANE FFN、stage1 textKV、stage2 sparse/contextpruning都分别近似/资格。不要用某一fast hotpath 1.2×保证完整视频。参考 [ltx-mac对照](../../docs/design/ltx25-ltx-mac-acceleration-parity-plan.md) 与更新的 [Stage2 sparse负结果](../../docs/design/ltx-stage2-sparse-results.md)；后者的实测否决不能被早期推荐设计覆盖。

## SeedVR2：超分 / restoration而非文生模型

起点：[pipeline](../../native/models/seedvr2/pipeline.cpp)、[VAE](../../native/models/seedvr2/vae.cpp)、[hybrid](../../native/models/seedvr2/hybrid.cpp)。保留original-FP16/FP32-state稳定配方，先history/inputowner释放与norm融合，再source/output streaming、globalstats row-file schedules与单个CoreML卷积边界。DiT FFN hybrid和实验VAE conv不是同一manifest/发布范围。

当前serial verified block weights是独立compile-isolated原型，不是App memory policy；VAE降低active后whole-request峰值可能仍在DiT。121帧模型限制不由1000帧codec测试解除。详细方法见 [VAE](../media/vae.md)。

[Comfy对照](../../docs/benchmark/SEEDVR2_COMFYUI.md)的速度来自GPU路线，不能称ANE收益；actual Comfy dtype和MPS兼容补丁也必须记录。

## Wan / FastMetal QAD 与 LLaDA

Wan看 [native module](../../native/models/wan_module.cpp) 与 [hybrid](../../native/models/wan/hybrid.cpp)：离线转换产物、固定QAD/少步schedule、adapter provenance与manifest先完成，再判断FFN分工；早期Python worker的历史速度不能冒充当前nativeexecutor。premerged LoRA资格不允许任意runtime adapter替换。

LLaDA看 [transformer](../../native/models/llada/llada_transformer.cpp)：原生GPU、noise refiner与主layers的FFN可参考Z通道分工，context/sigvq分支保持自己的数学。历史nativehybrid 1.203×是具体单机候选，不等于所有reference/edit/adapter或默认资格。

## 辅助超分：Real-ESRGAN

Swift/CoreML加载、overlap tiles/reflectpadding/feather、alpha与orientation、preload与原子发布是另一套轻量链路；ANE优先与生成模型ANE toggle相互独立。参考 [超分手册](../media/upscaling.md)，不可把x2/x4不同输出分辨率横排成等质量速度比。

## 实现参考与实验

可读实现：参考代码；[实验步骤与边界](../examples/README.md)；本项全部 case / controls。

本项控制：actual source dtype/table vs names；model capabilities/topology vs copied profile。CPU参考/模拟与实际设备入口分开标记；设备环境和更多变体见实验步骤。

可以重开探索的条件：new architecture, source, shape, adapter or hardware。历史负结果限定于其来源/shape/边界，不关闭该方向。

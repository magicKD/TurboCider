# 算法调研来源：固定版本、阅读范围与证据等级

[算法入口](README.md) · [机器可读清单](sources.json) · [评测合同](evaluation.md) · [配套实验](../labs/algorithm-research.md)

核查日期：2026-10-16。来源以作者开源实现、论文、作者技术笔记和 Apple/框架官方资料为主。S 表示指定源码接缝/章节已读，D 表示只核对文档；不是完整仓库审计。下载候选文件与实际阅读分开，下面只列实际采用的阅读范围。

`sources.json` 保存 commit、文件 SHA-256 与长度，不镜像第三方正文。摘要为原创概括，原代码和模型继续遵循各自许可。活动项目页只供导航，固定文件链接用于追溯。没有把网页标题中的 lossless 改写为数值无损，也不把作者测量冒充本轮本机复现。

本轮网页搜索工具未返回可用结果；GitHub API 返回403。已转用获准的 Git 只读源码下载和 raw 文件端点，覆盖下列公开项目；因此不宣称穷尽全网或核实远端所有最新分支。远端快照与本地既有 checkout 分别固定，不能混称同一更新时点。

<a id="s01"></a>
## S01 — Sana sol-engine

仓库：[NVlabs/Sana](https://github.com/NVlabs/Sana)；commit：`d0c0a4685ab5dc2336d18b7213d85f13def92418`。

阅读：S；控制器、effect组合、token与SAP策略、Metal summary/routing/softmax的关键片段和算法文档；不是所有模型adapter审计。

运行边界：仅阅读；未安装/运行上游模型或基准。作者性能数字仅作来源报告。

- [techniques/compose.py](https://github.com/NVlabs/Sana/blob/d0c0a4685ab5dc2336d18b7213d85f13def92418/techniques/compose.py)
- [techniques/methods/teacache.py](https://github.com/NVlabs/Sana/blob/d0c0a4685ab5dc2336d18b7213d85f13def92418/techniques/methods/teacache.py)
- [techniques/methods/step_cache.py](https://github.com/NVlabs/Sana/blob/d0c0a4685ab5dc2336d18b7213d85f13def92418/techniques/methods/step_cache.py)
- [techniques/methods/payload_cache.py](https://github.com/NVlabs/Sana/blob/d0c0a4685ab5dc2336d18b7213d85f13def92418/techniques/methods/payload_cache.py)
- [techniques/methods/token_prune.py](https://github.com/NVlabs/Sana/blob/d0c0a4685ab5dc2336d18b7213d85f13def92418/techniques/methods/token_prune.py)
- [techniques/sparse_attention_policies.py](https://github.com/NVlabs/Sana/blob/d0c0a4685ab5dc2336d18b7213d85f13def92418/techniques/sparse_attention_policies.py)
- [techniques/sparse_backends/sol_attn/mps/preprocess.py](https://github.com/NVlabs/Sana/blob/d0c0a4685ab5dc2336d18b7213d85f13def92418/techniques/sparse_backends/sol_attn/mps/preprocess.py)
- [techniques/sparse_backends/sol_attn/mps/metal.py](https://github.com/NVlabs/Sana/blob/d0c0a4685ab5dc2336d18b7213d85f13def92418/techniques/sparse_backends/sol_attn/mps/metal.py)
- [site_docs/techniques/cache/easycache.md](https://github.com/NVlabs/Sana/blob/d0c0a4685ab5dc2336d18b7213d85f13def92418/site_docs/techniques/cache/easycache.md)
- [site_docs/techniques/cache/teacache.md](https://github.com/NVlabs/Sana/blob/d0c0a4685ab5dc2336d18b7213d85f13def92418/site_docs/techniques/cache/teacache.md)
- [site_docs/techniques/cache/cache_dit.md](https://github.com/NVlabs/Sana/blob/d0c0a4685ab5dc2336d18b7213d85f13def92418/site_docs/techniques/cache/cache_dit.md)
- [site_docs/techniques/cache/taylorseer.md](https://github.com/NVlabs/Sana/blob/d0c0a4685ab5dc2336d18b7213d85f13def92418/site_docs/techniques/cache/taylorseer.md)
- [site_docs/techniques/sparse/sol_attn.md](https://github.com/NVlabs/Sana/blob/d0c0a4685ab5dc2336d18b7213d85f13def92418/site_docs/techniques/sparse/sol_attn.md)
- [site_docs/techniques/sparse/pisa.md](https://github.com/NVlabs/Sana/blob/d0c0a4685ab5dc2336d18b7213d85f13def92418/site_docs/techniques/sparse/pisa.md)

<a id="s02"></a>
## S02 — TeaCache

仓库：[ali-vilab/TeaCache](https://github.com/ali-vilab/TeaCache)；commit：`7c10efc4702c6b619f47805f7abe4a7a08085aa0`。

阅读：S；FLUX调制信号、累计多项式、首末full、residual主路径。

运行边界：仅阅读；未安装/运行上游模型或基准。作者性能数字仅作来源报告。

- [README.md](https://github.com/ali-vilab/TeaCache/blob/7c10efc4702c6b619f47805f7abe4a7a08085aa0/README.md)
- [TeaCache4FLUX/teacache_flux.py](https://github.com/ali-vilab/TeaCache/blob/7c10efc4702c6b619f47805f7abe4a7a08085aa0/TeaCache4FLUX/teacache_flux.py)

<a id="s03"></a>
## S03 — mlx-teacache 与作者研究笔记

仓库：[IonDen/mlx-teacache](https://github.com/IonDen/mlx-teacache)；commit：`b56f2882ca237b3b971c976f78804154cbeb1014`。

阅读：S；gate完整安全控制与FLUX prelude/body/tail接缝；README的基准更正/支持边界；Qwen混合精度作者笔记前半及配方；另读短日程负结果全文。

运行边界：仅阅读；未安装/运行上游模型或基准。作者性能数字仅作来源报告。

- [README.md](https://github.com/IonDen/mlx-teacache/blob/b56f2882ca237b3b971c976f78804154cbeb1014/README.md)
- [src/mlx_teacache/_kernel/gate.py](https://github.com/IonDen/mlx-teacache/blob/b56f2882ca237b3b971c976f78804154cbeb1014/src/mlx_teacache/_kernel/gate.py)
- [src/mlx_teacache/variants/flux1_dev/integration.py](https://github.com/IonDen/mlx-teacache/blob/b56f2882ca237b3b971c976f78804154cbeb1014/src/mlx_teacache/variants/flux1_dev/integration.py)
- [docs/papers/qwen-image-mixed-precision-on-a-32-gb-mac.md](https://github.com/IonDen/mlx-teacache/blob/b56f2882ca237b3b971c976f78804154cbeb1014/docs/papers/qwen-image-mixed-precision-on-a-32-gb-mac.md)
- 补充已读原文：[固定文件/作者笔记](https://raw.githubusercontent.com/IonDen/mlx-teacache/b56f2882ca237b3b971c976f78804154cbeb1014/docs/papers/why-teacache-does-not-engage-on-short-distilled-schedules.md)（不在上述文件哈希范围内）

<a id="s04"></a>
## S04 — EasyCache

仓库：[H-EmbodVis/EasyCache](https://github.com/H-EmbodVis/EasyCache)；commit：`bfc63232cfae8c77e2dccf9d72b8f2f56bd995b8`。

阅读：S；README、Wan CFG调用对计数、控制与cache_even/cache_odd主体片段。

运行边界：仅阅读；未安装/运行上游模型或基准。作者性能数字仅作来源报告。

- [README.md](https://github.com/H-EmbodVis/EasyCache/blob/bfc63232cfae8c77e2dccf9d72b8f2f56bd995b8/README.md)
- [EasyCache4Wan2.1/easycache_generate.py](https://github.com/H-EmbodVis/EasyCache/blob/bfc63232cfae8c77e2dccf9d72b8f2f56bd995b8/EasyCache4Wan2.1/easycache_generate.py)

<a id="s05"></a>
## S05 — ParaAttention FirstBlockCache

仓库：[chengzeyi/ParaAttention](https://github.com/chengzeyi/ParaAttention)；commit：`4de137c5b96416489f06e43e19f2c14a772e28fd`。

阅读：S；CacheContext、相对L1、CachedTransformerBlocks、两流residual与剩余block边界。

运行边界：仅阅读；未安装/运行上游模型或基准。作者性能数字仅作来源报告。

- [README.md](https://github.com/chengzeyi/ParaAttention/blob/4de137c5b96416489f06e43e19f2c14a772e28fd/README.md)
- [src/para_attn/first_block_cache/utils.py](https://github.com/chengzeyi/ParaAttention/blob/4de137c5b96416489f06e43e19f2c14a772e28fd/src/para_attn/first_block_cache/utils.py)

<a id="s06"></a>
## S06 — Cache-DiT

仓库：[vipshop/cache-dit](https://github.com/vipshop/cache-dit)；commit：`089df6b6895a3bc15ebcd3f7546d90ec6eb592e8`。

阅读：S；框架文档、F/M/B forward关键片段、context字段/计数、TaylorSeer导数/anchor；不审计所有adapter或分布式路径。

运行边界：仅阅读；未安装/运行上游模型或基准。作者性能数字仅作来源报告。

- [README.md](https://github.com/vipshop/cache-dit/blob/089df6b6895a3bc15ebcd3f7546d90ec6eb592e8/README.md)
- [src/cache_dit/caching/cache_blocks/pattern_base.py](https://github.com/vipshop/cache-dit/blob/089df6b6895a3bc15ebcd3f7546d90ec6eb592e8/src/cache_dit/caching/cache_blocks/pattern_base.py)
- [src/cache_dit/caching/cache_contexts/cache_context.py](https://github.com/vipshop/cache-dit/blob/089df6b6895a3bc15ebcd3f7546d90ec6eb592e8/src/cache_dit/caching/cache_contexts/cache_context.py)
- [src/cache_dit/caching/cache_contexts/calibrators/taylorseer.py](https://github.com/vipshop/cache-dit/blob/089df6b6895a3bc15ebcd3f7546d90ec6eb592e8/src/cache_dit/caching/cache_contexts/calibrators/taylorseer.py)

<a id="s07"></a>
## S07 — TaylorSeer

仓库：[Shenyi-Z/TaylorSeer](https://github.com/Shenyi-Z/TaylorSeer)；commit：`704ee98c74f7f04da443daa3c0aa2cc7803d86e3`。

阅读：S；作者公式、step坐标、差分与预测函数；不运行其CUDA pipeline。

运行边界：仅阅读；未安装/运行上游模型或基准。作者性能数字仅作来源报告。

- [README.md](https://github.com/Shenyi-Z/TaylorSeer/blob/704ee98c74f7f04da443daa3c0aa2cc7803d86e3/README.md)
- [TaylorSeers-Diffusers/taylorseer_flux/taylorseer_utils/__init__.py](https://github.com/Shenyi-Z/TaylorSeer/blob/704ee98c74f7f04da443daa3c0aa2cc7803d86e3/TaylorSeers-Diffusers/taylorseer_flux/taylorseer_utils/__init__.py)

<a id="s08"></a>
## S08 — stable-diffusion.cpp 缓存移植

仓库：[leejet/stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)；commit：`2f886889e6e8b78738d6b87f7191f6018557c551`。

阅读：S；cache modes、EasyCache host控制与条件残差、Taylor状态/配置片段。主仓库ggml/frontend子模块有既有修改；这些选中文件按HEAD取证，不依赖子模块改动。

运行边界：仅阅读；未安装/运行上游模型或基准。作者性能数字仅作来源报告。

- [docs/caching.md](https://github.com/leejet/stable-diffusion.cpp/blob/2f886889e6e8b78738d6b87f7191f6018557c551/docs/caching.md)
- [src/runtime/easycache.hpp](https://github.com/leejet/stable-diffusion.cpp/blob/2f886889e6e8b78738d6b87f7191f6018557c551/src/runtime/easycache.hpp)
- [src/runtime/cache_dit.hpp](https://github.com/leejet/stable-diffusion.cpp/blob/2f886889e6e8b78738d6b87f7191f6018557c551/src/runtime/cache_dit.hpp)

<a id="s09"></a>
## S09 — VideoSys PAB

仓库：[NUS-HPC-AI-Lab/VideoSys](https://github.com/NUS-HPC-AI-Lab/VideoSys)；commit：`4cce477878b0683b5c05cd001ce45ac0e2fc3034`。

阅读：S；三种attention周期、timestep窗口、MLP输出池与清理。

运行边界：仅阅读；未安装/运行上游模型或基准。作者性能数字仅作来源报告。

- [videosys/core/pab/pab_mgr.py](https://github.com/NUS-HPC-AI-Lab/VideoSys/blob/4cce477878b0683b5c05cd001ce45ac0e2fc3034/videosys/core/pab/pab_mgr.py)

<a id="s10"></a>
## S10 — FasterCache

仓库：[Vchitect/FasterCache](https://github.com/Vchitect/FasterCache)；commit：`02d05ef7edb48bdec01d2b4df6edfe7f472a41b9`。

阅读：S；CogVideoX counter、attention双历史外推、FFT低高频差与CFG恢复的相关片段。

运行边界：仅阅读；未安装/运行上游模型或基准。作者性能数字仅作来源报告。

- [README.md](https://github.com/Vchitect/FasterCache/blob/02d05ef7edb48bdec01d2b4df6edfe7f472a41b9/README.md)
- [scripts/cogvideox/fastercache_sample_cogvideox.py](https://github.com/Vchitect/FasterCache/blob/02d05ef7edb48bdec01d2b4df6edfe7f472a41b9/scripts/cogvideox/fastercache_sample_cogvideox.py)

<a id="s11"></a>
## S11 — ComfyUI-SolAttn-MPS

仓库：[yshenaw/ComfyUI-SolAttn-MPS](https://github.com/yshenaw/ComfyUI-SolAttn-MPS)；commit：`45071126b0c1ee30b0e6b7103fa9d70924828ba5`。

阅读：D；Apple环境、版本/header fallback、kernel机制与作者component/E2E表；Metal算法代码另外由S01核对，本项不宣称独立逐行审计。

运行边界：仅阅读；未安装/运行上游模型或基准。作者性能数字仅作来源报告。

- [README.md](https://github.com/yshenaw/ComfyUI-SolAttn-MPS/blob/45071126b0c1ee30b0e6b7103fa9d70924828ba5/README.md)

<a id="s12"></a>
## S12 — PISA

仓库：[xie-lab-ml/piecewise-sparse-attention](https://github.com/xie-lab-ml/piecewise-sparse-attention)；commit：`655648df86cafd75042dc32d6b6be78c4ea0eca8`。

阅读：S；fused summary、centroid top-k、exact/zero-order/first-order三段forward和wrapper；0th独立kernel未逐行审计。

运行边界：仅阅读；未安装/运行上游模型或基准。作者性能数字仅作来源报告。

- [README.md](https://github.com/xie-lab-ml/piecewise-sparse-attention/blob/655648df86cafd75042dc32d6b6be78c4ea0eca8/README.md)
- [piecewise_attn/kernels/piecewise_sparse_attn_hyd.py](https://github.com/xie-lab-ml/piecewise-sparse-attention/blob/655648df86cafd75042dc32d6b6be78c4ea0eca8/piecewise_attn/kernels/piecewise_sparse_attn_hyd.py)

<a id="s13"></a>
## S13 — Sparse VideoGen / SVG2 / SVG-EAR

仓库：[svg-project/Sparse-VideoGen](https://github.com/svg-project/Sparse-VideoGen)；commit：`f89aedaf169ac2ae5b186bda674e53c3dc08c476`。

阅读：S；Wan attention online sample误差/分支、dynamic-map/kmeans接缝与重排；EAR只读作者README机制，没有完整kernel审计。

运行边界：仅阅读；未安装/运行上游模型或基准。作者性能数字仅作来源报告。

- [README.md](https://github.com/svg-project/Sparse-VideoGen/blob/f89aedaf169ac2ae5b186bda674e53c3dc08c476/README.md)
- [svg/models/wan/attention.py](https://github.com/svg-project/Sparse-VideoGen/blob/f89aedaf169ac2ae5b186bda674e53c3dc08c476/svg/models/wan/attention.py)
- [svg/kmeans_utils.py](https://github.com/svg-project/Sparse-VideoGen/blob/f89aedaf169ac2ae5b186bda674e53c3dc08c476/svg/kmeans_utils.py)
- [svg/kernels/triton/permute.py](https://github.com/svg-project/Sparse-VideoGen/blob/f89aedaf169ac2ae5b186bda674e53c3dc08c476/svg/kernels/triton/permute.py)

<a id="s14"></a>
## S14 — SpargeAttention

仓库：[thu-ml/SpargeAttn](https://github.com/thu-ml/SpargeAttn)；commit：`ae5b629ebb41e41f86b3ea2ab5a3283f13ac151a`。

阅读：S；recommended topk API、map/quant调用链、架构与tile分派、V低精度分支。

运行边界：仅阅读；未安装/运行上游模型或基准。作者性能数字仅作来源报告。

- [README.md](https://github.com/thu-ml/SpargeAttn/blob/ae5b629ebb41e41f86b3ea2ab5a3283f13ac151a/README.md)
- [spas_sage_attn/core.py](https://github.com/thu-ml/SpargeAttn/blob/ae5b629ebb41e41f86b3ea2ab5a3283f13ac151a/spas_sage_attn/core.py)

<a id="s14a"></a>
## S14a — SageAttention

仓库：[thu-ml/SageAttention](https://github.com/thu-ml/SageAttention)；commit：`d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5`。

阅读：S；per_block_int8 / per_warp_int8、K smoothing接口、scale/log2及layout。

运行边界：仅阅读；未安装/运行上游模型或基准。作者性能数字仅作来源报告。

- [sageattention/quant.py](https://github.com/thu-ml/SageAttention/blob/d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5/sageattention/quant.py)

<a id="s15"></a>
## S15 — FastVideo STA/VSA 与 MLX

仓库：[hao-ai-lab/FastVideo](https://github.com/hao-ai-lab/FastVideo)；commit：`a943220c115228ade5d57b3bab9a6a87fd600a10`。

阅读：S；STA归档状态、VSA checkpoint/gate/Metal质量限制、MLX窗口切片+sink的完整核心。

运行边界：仅阅读；未安装/运行上游模型或基准。作者性能数字仅作来源报告。

- [docs/attention/sta/index.md](https://github.com/hao-ai-lab/FastVideo/blob/a943220c115228ade5d57b3bab9a6a87fd600a10/docs/attention/sta/index.md)
- [docs/attention/vsa/index.md](https://github.com/hao-ai-lab/FastVideo/blob/a943220c115228ade5d57b3bab9a6a87fd600a10/docs/attention/vsa/index.md)
- [fastvideo/mlx_runtime/windowed_attention.py](https://github.com/hao-ai-lab/FastVideo/blob/a943220c115228ade5d57b3bab9a6a87fd600a10/fastvideo/mlx_runtime/windowed_attention.py)

<a id="s16"></a>
## S16 — ToMeSD

仓库：[dbolya/tomesd](https://github.com/dbolya/tomesd)；commit：`27a14a372beecaa85c101fb588631643f379b8ce`。

阅读：S；随机2D二分匹配、MPS gather workaround、scatter mean、unmerge和attn/cross/MLP默认范围。

运行边界：仅阅读；未安装/运行上游模型或基准。作者性能数字仅作来源报告。

- [README.md](https://github.com/dbolya/tomesd/blob/27a14a372beecaa85c101fb588631643f379b8ce/README.md)
- [tomesd/merge.py](https://github.com/dbolya/tomesd/blob/27a14a372beecaa85c101fb588631643f379b8ce/tomesd/merge.py)
- [tomesd/patch.py](https://github.com/dbolya/tomesd/blob/27a14a372beecaa85c101fb588631643f379b8ce/tomesd/patch.py)

<a id="s17"></a>
## S17 — Nunchaku / SVDQuant

仓库：[nunchaku-ai/nunchaku](https://github.com/nunchaku-ai/nunchaku)；commit：`302e0e97024ebd68688fe890e5df83731edf7b54`。

阅读：S；方法说明、SVDQW4A4Linear quantize/forward_quant、低秩投影融合接口与packing配置。

运行边界：仅阅读；未安装/运行上游模型或基准。作者性能数字仅作来源报告。

- [README.md](https://github.com/nunchaku-ai/nunchaku/blob/302e0e97024ebd68688fe890e5df83731edf7b54/README.md)
- [nunchaku/models/linear.py](https://github.com/nunchaku-ai/nunchaku/blob/302e0e97024ebd68688fe890e5df83731edf7b54/nunchaku/models/linear.py)

<a id="s18"></a>
## S18 — SmoothQuant

仓库：[mit-han-lab/smoothquant](https://github.com/mit-han-lab/smoothquant)；commit：`c61476d728e42ae0d8a35e7e78494edcac3237b5`。

阅读：S；scale公式、LN/RMSNorm与多linear共同补偿；LLM实现边界。

运行边界：仅阅读；未安装/运行上游模型或基准。作者性能数字仅作来源报告。

- [smoothquant/smooth.py](https://github.com/mit-han-lab/smoothquant/blob/c61476d728e42ae0d8a35e7e78494edcac3237b5/smoothquant/smooth.py)

<a id="s19"></a>
## S19 — MLX quantized Metal

仓库：[ml-explore/mlx](https://github.com/ml-explore/mlx)；commit：`36969c094354086e15d2a327a9c50feb20b9cf11`。

阅读：S；Metal量化dispatch/tile/alignment片段；另读固定commit的Python quantize/QuantizedLinear/QQLinear原始文件，不在此文件hash清单。

运行边界：仅阅读；未安装/运行上游模型或基准。作者性能数字仅作来源报告。

- [mlx/backend/metal/quantized.cpp](https://github.com/ml-explore/mlx/blob/36969c094354086e15d2a327a9c50feb20b9cf11/mlx/backend/metal/quantized.cpp)
- 补充已读原文：[固定文件/作者笔记](https://raw.githubusercontent.com/ml-explore/mlx/36969c094354086e15d2a327a9c50feb20b9cf11/python/mlx/nn/layers/quantized.py)（不在上述文件哈希范围内）

<a id="s20"></a>
## S20 — mflux

仓库：[filipstrand/mflux](https://github.com/filipstrand/mflux)；commit：`8c00dab2505a96019df9d30bc9c223bf20d733c4`。

阅读：S；README第三方MLX cache入口与当前Qwen量化predicate。

运行边界：仅阅读；未安装/运行上游模型或基准。作者性能数字仅作来源报告。

- [README.md](https://github.com/filipstrand/mflux/blob/8c00dab2505a96019df9d30bc9c223bf20d733c4/README.md)
- [src/mflux/models/qwen/weights/qwen_weight_definition.py](https://github.com/filipstrand/mflux/blob/8c00dab2505a96019df9d30bc9c223bf20d733c4/src/mflux/models/qwen/weights/qwen_weight_definition.py)

<a id="s21"></a>
## S21 — Latent Consistency Models

仓库：[luosiallen/latent-consistency-model](https://github.com/luosiallen/latent-consistency-model)；commit：`a9ad79587cc8bd1e404ccd1a3056a3da969b2f62`。

阅读：D；蒸馏/LCM-LoRA/scheduler说明与MPS使用入口，不运行训练或推理。

运行边界：仅阅读；未安装/运行上游模型或基准。作者性能数字仅作来源报告。

- [README.md](https://github.com/luosiallen/latent-consistency-model/blob/a9ad79587cc8bd1e404ccd1a3056a3da969b2f62/README.md)

<a id="s22"></a>
## S22 — LTX-2-MLX

仓库：[dgrauet/ltx-2-mlx](https://github.com/dgrauet/ltx-2-mlx)；commit：`91e6f6c9bd621ff2ae31adfee643e113d67d6ae8`。

阅读：S；两阶段角色、TeaCache校准说明/构建、2.5拒绝guard及sampler signal/cached residual片段。

运行边界：仅阅读；未安装/运行上游模型或基准。作者性能数字仅作来源报告。

- [README.md](https://github.com/dgrauet/ltx-2-mlx/blob/91e6f6c9bd621ff2ae31adfee643e113d67d6ae8/README.md)
- [packages/ltx-pipelines-mlx/src/ltx_pipelines_mlx/ti2vid_two_stages.py](https://github.com/dgrauet/ltx-2-mlx/blob/91e6f6c9bd621ff2ae31adfee643e113d67d6ae8/packages/ltx-pipelines-mlx/src/ltx_pipelines_mlx/ti2vid_two_stages.py)
- [packages/ltx-pipelines-mlx/src/ltx_pipelines_mlx/_base.py](https://github.com/dgrauet/ltx-2-mlx/blob/91e6f6c9bd621ff2ae31adfee643e113d67d6ae8/packages/ltx-pipelines-mlx/src/ltx_pipelines_mlx/_base.py)
- [packages/ltx-pipelines-mlx/src/ltx_pipelines_mlx/utils/samplers.py](https://github.com/dgrauet/ltx-2-mlx/blob/91e6f6c9bd621ff2ae31adfee643e113d67d6ae8/packages/ltx-pipelines-mlx/src/ltx_pipelines_mlx/utils/samplers.py)

<a id="s23"></a>
## S23 — TAESD

仓库：[madebyollin/taesd](https://github.com/madebyollin/taesd)；commit：`027d3ef404a04a141e60d5ec0fee149c2c8bdd74`。

阅读：S；模型家族/latent合同/质量限制；Conv/upsample、F16/F32与带全局norm的变体。

运行边界：仅阅读；未安装/运行上游模型或基准。作者性能数字仅作来源报告。

- [README.md](https://github.com/madebyollin/taesd/blob/027d3ef404a04a141e60d5ec0fee149c2c8bdd74/README.md)
- [taesd.py](https://github.com/madebyollin/taesd/blob/027d3ef404a04a141e60d5ec0fee149c2c8bdd74/taesd.py)

<a id="s24"></a>
## S24 — DeepCache

仓库：[horseee/DeepCache](https://github.com/horseee/DeepCache)；commit：`1691a3ab22308fe06685e4ff151a9ab175be2dc4`。

阅读：S；U-Net缓存机制说明与helper位置/周期/wrap/unwrap/reset控制。

运行边界：仅阅读；未安装/运行上游模型或基准。作者性能数字仅作来源报告。

- [README.md](https://github.com/horseee/DeepCache/blob/1691a3ab22308fe06685e4ff151a9ab175be2dc4/README.md)
- [DeepCache/extension/deepcache.py](https://github.com/horseee/DeepCache/blob/1691a3ab22308fe06685e4ff151a9ab175be2dc4/DeepCache/extension/deepcache.py)

<a id="s25"></a>
## S25 — Diffusers caching / scheduler

仓库：[huggingface/diffusers](https://github.com/huggingface/diffusers)；commit：`d42ebf9c6e0b46103b74283f5324357a67acaa04`。

阅读：S；cache机制与MPS文档入口；Sea indicator/metadata/gate和Mag config/ratio/状态控制片段；不是所有hooks或全部Sea adapter审计。

运行边界：仅阅读；未安装/运行上游模型或基准。作者性能数字仅作来源报告。

- [docs/source/en/optimization/cache.md](https://github.com/huggingface/diffusers/blob/d42ebf9c6e0b46103b74283f5324357a67acaa04/docs/source/en/optimization/cache.md)
- [docs/source/en/api/schedulers/multistep_dpm_solver.md](https://github.com/huggingface/diffusers/blob/d42ebf9c6e0b46103b74283f5324357a67acaa04/docs/source/en/api/schedulers/multistep_dpm_solver.md)
- [src/diffusers/hooks/sea_cache.py](https://github.com/huggingface/diffusers/blob/d42ebf9c6e0b46103b74283f5324357a67acaa04/src/diffusers/hooks/sea_cache.py)
- [src/diffusers/hooks/mag_cache.py](https://github.com/huggingface/diffusers/blob/d42ebf9c6e0b46103b74283f5324357a67acaa04/src/diffusers/hooks/mag_cache.py)

<a id="s26"></a>
## S26 — Radial Attention

仓库：[mit-han-lab/radial-attention](https://github.com/mit-han-lab/radial-attention)；commit：`72788d4f0a6d202f1ec5f1c98a6e4c8b2e34fdbc`。

阅读：S；距离band/密度规律、LoRA延长与原长度区分、mask/CSR转换、时空窗口与sink代码片段。

运行边界：仅阅读；未安装/运行上游模型或基准。作者性能数字仅作来源报告。

- [README.md](https://github.com/mit-han-lab/radial-attention/blob/72788d4f0a6d202f1ec5f1c98a6e4c8b2e34fdbc/README.md)
- [radial_attn/attn_mask.py](https://github.com/mit-han-lab/radial-attention/blob/72788d4f0a6d202f1ec5f1c98a6e4c8b2e34fdbc/radial_attn/attn_mask.py)

<a id="s27"></a>
## S27 — Apple Core ML Stable Diffusion

仓库：[apple/ml-stable-diffusion](https://github.com/apple/ml-stable-diffusion)；commit：`ea2805dc1945be20561c77e5f6d1d9a5a637cda2`。

阅读：D；官方weight compression/MBP/activation sensitivity说明与后端条件；不重跑转换脚本。

运行边界：仅阅读；未安装/运行上游模型或基准。作者性能数字仅作来源报告。

- [README.md](https://github.com/apple/ml-stable-diffusion/blob/ea2805dc1945be20561c77e5f6d1d9a5a637cda2/README.md)

<a id="s28"></a>
## S28 — mlx-taef

仓库：[IonDen/mlx-taef](https://github.com/IonDen/mlx-taef)；commit：`cd4340e4190b7964e816a3dcb67fba5bcd27ca2d`。

阅读：D；纯MLX tiny decoder、latent变体/预览callback、normalization及旧基准修正说明；不审计所有模型源码。

运行边界：仅阅读；未安装/运行上游模型或基准。作者性能数字仅作来源报告。

- [README.md](https://github.com/IonDen/mlx-taef/blob/cd4340e4190b7964e816a3dcb67fba5bcd27ca2d/README.md)

<a id="s29"></a>
## S29 — TurboCider 既有 Apple 正反实验

仓库：TurboCider（本地历史，可选产品溯源）；commit：`614fdf2bc5abf05b050688f8b40dcc65580e1af4`。

阅读：D；历史LTX sparse质量/阶段负结果与Qwen DBCache完整请求报告；本轮只读文档，没有重跑原始媒体基准。

运行边界：既有本地报告，非本轮重跑；仓库公开URL未验证，本项以本地相对路径为权威。

- [docs/design/ltx-stage2-sparse-results.md](../../docs/design/ltx-stage2-sparse-results.md)
- [docs/status/2026-09-30-qwen21-dit-cache.md](../../docs/status/2026-09-30-qwen21-dit-cache.md)

## 重放来源取证

[source-plan.json](source-plan.json)是候选采集范围，不是已审计/支持列表；[review-plan.json](review-plan.json)是本页采用的具体阅读文件。

```sh
python3 techniques/algorithms/source_probe.py --help
python3 techniques/algorithms/curate_sources.py --check techniques/algorithms/sources.json
```

源码下载需显式 `--download`，输出放到新的本地忽略目录；不安装依赖、不运行项目脚本、不更新既有 HEAD。partial clone 的缺失 blob 仅在显式下载模式获取。来源收集没有 GPU 工作。

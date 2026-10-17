# 算法调研来源：固定版本、阅读范围与证据等级

<!-- knowledge
id: technique-algorithms-sources
type: source
operator: scheduling
backend: cpu-reference
device: cpu
symptom: measurement
semantics: contract-only
boundary: request, producer-consumer
api: reference-only
-->

[算法入口](README.md) · [方法地图](methods/README.md) · [案例](case-studies/README.md)

这里保留原始阅读笔记中的固定 revision、关键文件、论文链接与机制解读，不保存来源 JSON、文件 hash 或逐轮日志。S 表示所列源码接缝已读，D 表示文档阅读；不等于审计完整仓库、复现作者结果或确认最新版本。案例数字是历史摘要，不附验证环境。

<a id="s01"></a>
## S01 — Sana sol-engine

仓库：[NVlabs/Sana](https://github.com/NVlabs/Sana)；commit：`d0c0a4685ab5dc2336d18b7213d85f13def92418`。

阅读：S；控制器、effect组合、token与SAP策略、Metal summary/routing/softmax的关键片段和算法文档；第四轮深读online_mask_search_reuse的前置score、drift缺省、borrowed mask与indices；不是所有模型adapter审计。

历史观察范围：第三轮执行固定Metal前向/summary（只修独立untimed debug库）；第四轮CPU白名单AST执行六种route-policy接缝，计数wrapper不改函数body；不是Sana模型pipeline。

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

历史观察范围：仅阅读；未安装/运行上游模型或基准。作者性能数字仅作来源报告。

- [README.md](https://github.com/ali-vilab/TeaCache/blob/7c10efc4702c6b619f47805f7abe4a7a08085aa0/README.md)
- [TeaCache4FLUX/teacache_flux.py](https://github.com/ali-vilab/TeaCache/blob/7c10efc4702c6b619f47805f7abe4a7a08085aa0/TeaCache4FLUX/teacache_flux.py)

<a id="s03"></a>
## S03 — mlx-teacache 与作者研究笔记

仓库：[IonDen/mlx-teacache](https://github.com/IonDen/mlx-teacache)；commit：`b56f2882ca237b3b971c976f78804154cbeb1014`。

阅读：S；gate完整安全控制与FLUX prelude/body/tail接缝；README的基准更正/支持边界；Qwen混合精度作者笔记前半及配方；另读短日程负结果全文。

历史观察范围：仅阅读；未安装/运行上游模型或基准。作者性能数字仅作来源报告。

- [README.md](https://github.com/IonDen/mlx-teacache/blob/b56f2882ca237b3b971c976f78804154cbeb1014/README.md)
- [src/mlx_teacache/_kernel/gate.py](https://github.com/IonDen/mlx-teacache/blob/b56f2882ca237b3b971c976f78804154cbeb1014/src/mlx_teacache/_kernel/gate.py)
- [src/mlx_teacache/variants/flux1_dev/integration.py](https://github.com/IonDen/mlx-teacache/blob/b56f2882ca237b3b971c976f78804154cbeb1014/src/mlx_teacache/variants/flux1_dev/integration.py)
- [docs/papers/qwen-image-mixed-precision-on-a-32-gb-mac.md](https://github.com/IonDen/mlx-teacache/blob/b56f2882ca237b3b971c976f78804154cbeb1014/docs/papers/qwen-image-mixed-precision-on-a-32-gb-mac.md)
- 补充已读原文：[补充资料](https://raw.githubusercontent.com/IonDen/mlx-teacache/b56f2882ca237b3b971c976f78804154cbeb1014/docs/papers/why-teacache-does-not-engage-on-short-distilled-schedules.md)（不在上述文件哈希范围内）

<a id="s04"></a>
## S04 — EasyCache

仓库：[H-EmbodVis/EasyCache](https://github.com/H-EmbodVis/EasyCache)；commit：`bfc63232cfae8c77e2dccf9d72b8f2f56bd995b8`。

阅读：S；README、Wan CFG调用对计数、控制与cache_even/cache_odd主体片段。

历史观察范围：仅阅读；未安装/运行上游模型或基准。作者性能数字仅作来源报告。

- [README.md](https://github.com/H-EmbodVis/EasyCache/blob/bfc63232cfae8c77e2dccf9d72b8f2f56bd995b8/README.md)
- [EasyCache4Wan2.1/easycache_generate.py](https://github.com/H-EmbodVis/EasyCache/blob/bfc63232cfae8c77e2dccf9d72b8f2f56bd995b8/EasyCache4Wan2.1/easycache_generate.py)

<a id="s05"></a>
## S05 — ParaAttention FirstBlockCache

仓库：[chengzeyi/ParaAttention](https://github.com/chengzeyi/ParaAttention)；commit：`4de137c5b96416489f06e43e19f2c14a772e28fd`。

阅读：S；CacheContext、相对L1、CachedTransformerBlocks、两流residual与剩余block边界。

历史观察范围：仅阅读；未安装/运行上游模型或基准。作者性能数字仅作来源报告。

- [README.md](https://github.com/chengzeyi/ParaAttention/blob/4de137c5b96416489f06e43e19f2c14a772e28fd/README.md)
- [src/para_attn/first_block_cache/utils.py](https://github.com/chengzeyi/ParaAttention/blob/4de137c5b96416489f06e43e19f2c14a772e28fd/src/para_attn/first_block_cache/utils.py)

<a id="s06"></a>
## S06 — Cache-DiT

仓库：[vipshop/cache-dit](https://github.com/vipshop/cache-dit)；commit：`089df6b6895a3bc15ebcd3f7546d90ec6eb592e8`。

阅读：S；框架文档、F/M/B forward关键片段、context字段/计数、TaylorSeer导数/anchor；不审计所有adapter或分布式路径。

历史观察范围：第一轮只读；第三轮提取forward/context clock等指定方法，用CPU微型F/M/B替身检查consumer接缝；不导入完整包或模型。

- [README.md](https://github.com/vipshop/cache-dit/blob/089df6b6895a3bc15ebcd3f7546d90ec6eb592e8/README.md)
- [src/cache_dit/caching/cache_blocks/pattern_base.py](https://github.com/vipshop/cache-dit/blob/089df6b6895a3bc15ebcd3f7546d90ec6eb592e8/src/cache_dit/caching/cache_blocks/pattern_base.py)
- [src/cache_dit/caching/cache_contexts/cache_context.py](https://github.com/vipshop/cache-dit/blob/089df6b6895a3bc15ebcd3f7546d90ec6eb592e8/src/cache_dit/caching/cache_contexts/cache_context.py)
- [src/cache_dit/caching/cache_contexts/calibrators/taylorseer.py](https://github.com/vipshop/cache-dit/blob/089df6b6895a3bc15ebcd3f7546d90ec6eb592e8/src/cache_dit/caching/cache_contexts/calibrators/taylorseer.py)

<a id="s07"></a>
## S07 — TaylorSeer

仓库：[Shenyi-Z/TaylorSeer](https://github.com/Shenyi-Z/TaylorSeer)；commit：`704ee98c74f7f04da443daa3c0aa2cc7803d86e3`。

阅读：S；作者公式、step坐标、差分与预测函数；不运行其CUDA pipeline。

历史观察范围：仅阅读；未安装/运行上游模型或基准。作者性能数字仅作来源报告。

- [README.md](https://github.com/Shenyi-Z/TaylorSeer/blob/704ee98c74f7f04da443daa3c0aa2cc7803d86e3/README.md)
- [TaylorSeers-Diffusers/taylorseer_flux/taylorseer_utils/__init__.py](https://github.com/Shenyi-Z/TaylorSeer/blob/704ee98c74f7f04da443daa3c0aa2cc7803d86e3/TaylorSeers-Diffusers/taylorseer_flux/taylorseer_utils/__init__.py)

<a id="s08"></a>
## S08 — stable-diffusion.cpp 缓存移植

仓库：[leejet/stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)；commit：`2f886889e6e8b78738d6b87f7191f6018557c551`。

阅读：S；cache modes、EasyCache host控制与条件残差、Taylor状态/配置片段。主仓库ggml/frontend子模块有既有修改；这些选中文件按HEAD取证，不依赖子模块改动。

历史观察范围：仅阅读；未安装/运行上游模型或基准。作者性能数字仅作来源报告。

- [docs/caching.md](https://github.com/leejet/stable-diffusion.cpp/blob/2f886889e6e8b78738d6b87f7191f6018557c551/docs/caching.md)
- [src/runtime/easycache.hpp](https://github.com/leejet/stable-diffusion.cpp/blob/2f886889e6e8b78738d6b87f7191f6018557c551/src/runtime/easycache.hpp)
- [src/runtime/cache_dit.hpp](https://github.com/leejet/stable-diffusion.cpp/blob/2f886889e6e8b78738d6b87f7191f6018557c551/src/runtime/cache_dit.hpp)

<a id="s09"></a>
## S09 — VideoSys PAB

仓库：[NUS-HPC-AI-Lab/VideoSys](https://github.com/NUS-HPC-AI-Lab/VideoSys)；commit：`4cce477878b0683b5c05cd001ce45ac0e2fc3034`。

阅读：S；三种attention周期、timestep窗口、MLP输出池与清理。

历史观察范围：仅阅读；未安装/运行上游模型或基准。作者性能数字仅作来源报告。

- [videosys/core/pab/pab_mgr.py](https://github.com/NUS-HPC-AI-Lab/VideoSys/blob/4cce477878b0683b5c05cd001ce45ac0e2fc3034/videosys/core/pab/pab_mgr.py)

<a id="s10"></a>
## S10 — FasterCache

仓库：[Vchitect/FasterCache](https://github.com/Vchitect/FasterCache)；commit：`02d05ef7edb48bdec01d2b4df6edfe7f472a41b9`。

阅读：S；CogVideoX counter、attention双历史外推、FFT低高频差与CFG恢复的相关片段。

历史观察范围：仅阅读；未安装/运行上游模型或基准。作者性能数字仅作来源报告。

- [README.md](https://github.com/Vchitect/FasterCache/blob/02d05ef7edb48bdec01d2b4df6edfe7f472a41b9/README.md)
- [scripts/cogvideox/fastercache_sample_cogvideox.py](https://github.com/Vchitect/FasterCache/blob/02d05ef7edb48bdec01d2b4df6edfe7f472a41b9/scripts/cogvideox/fastercache_sample_cogvideox.py)

<a id="s11"></a>
## S11 — ComfyUI-SolAttn-MPS

仓库：[yshenaw/ComfyUI-SolAttn-MPS](https://github.com/yshenaw/ComfyUI-SolAttn-MPS)；commit：`45071126b0c1ee30b0e6b7103fa9d70924828ba5`。

阅读：D；Apple环境、版本/header fallback、kernel机制与作者component/E2E表；Metal算法代码另外由S01核对，本项不宣称独立逐行审计。

历史观察范围：仅阅读；未安装/运行上游模型或基准。作者性能数字仅作来源报告。

- [README.md](https://github.com/yshenaw/ComfyUI-SolAttn-MPS/blob/45071126b0c1ee30b0e6b7103fa9d70924828ba5/README.md)

<a id="s12"></a>
## S12 — PISA

仓库：[xie-lab-ml/piecewise-sparse-attention](https://github.com/xie-lab-ml/piecewise-sparse-attention)；commit：`655648df86cafd75042dc32d6b6be78c4ea0eca8`。

阅读：S；fused summary、centroid top-k、exact/zero-order/first-order三段forward和wrapper；0th独立kernel未逐行审计。

历史观察范围：仅阅读；未安装/运行上游模型或基准。作者性能数字仅作来源报告。

- [README.md](https://github.com/xie-lab-ml/piecewise-sparse-attention/blob/655648df86cafd75042dc32d6b6be78c4ea0eca8/README.md)
- [piecewise_attn/kernels/piecewise_sparse_attn_hyd.py](https://github.com/xie-lab-ml/piecewise-sparse-attention/blob/655648df86cafd75042dc32d6b6be78c4ea0eca8/piecewise_attn/kernels/piecewise_sparse_attn_hyd.py)

<a id="s13"></a>
## S13 — Sparse VideoGen / SVG2 / SVG-EAR

仓库：[svg-project/Sparse-VideoGen](https://github.com/svg-project/Sparse-VideoGen)；commit：`f89aedaf169ac2ae5b186bda674e53c3dc08c476`。

阅读：S；Wan attention online sample误差/分支、dynamic-map/kmeans接缝与重排；EAR只读作者README机制，没有完整kernel审计。

历史观察范围：仅阅读；未安装/运行上游模型或基准。作者性能数字仅作来源报告。

- [README.md](https://github.com/svg-project/Sparse-VideoGen/blob/f89aedaf169ac2ae5b186bda674e53c3dc08c476/README.md)
- [svg/models/wan/attention.py](https://github.com/svg-project/Sparse-VideoGen/blob/f89aedaf169ac2ae5b186bda674e53c3dc08c476/svg/models/wan/attention.py)
- [svg/kmeans_utils.py](https://github.com/svg-project/Sparse-VideoGen/blob/f89aedaf169ac2ae5b186bda674e53c3dc08c476/svg/kmeans_utils.py)
- [svg/kernels/triton/permute.py](https://github.com/svg-project/Sparse-VideoGen/blob/f89aedaf169ac2ae5b186bda674e53c3dc08c476/svg/kernels/triton/permute.py)

<a id="s14"></a>
## S14 — SpargeAttention

仓库：[thu-ml/SpargeAttn](https://github.com/thu-ml/SpargeAttn)；commit：`ae5b629ebb41e41f86b3ea2ab5a3283f13ac151a`。

阅读：S；recommended/legacy topk入口、utils的pool/sim/CDF/LUT/量化、独立Triton quant、C++绑定与sm80 causal flag/模板接缝；不审计所有架构kernel。

历史观察范围：第五轮仅提取两个Python入口和CDF控制函数的原AST，以CPU Torch与显式CUDA/selector/quant/kernel桩核对8+5个接缝；未加载CUDA扩展或执行其attention数学。

- [README.md](https://github.com/thu-ml/SpargeAttn/blob/ae5b629ebb41e41f86b3ea2ab5a3283f13ac151a/README.md)
- [spas_sage_attn/core.py](https://github.com/thu-ml/SpargeAttn/blob/ae5b629ebb41e41f86b3ea2ab5a3283f13ac151a/spas_sage_attn/core.py)
- [spas_sage_attn/utils.py](https://github.com/thu-ml/SpargeAttn/blob/ae5b629ebb41e41f86b3ea2ab5a3283f13ac151a/spas_sage_attn/utils.py)
- [spas_sage_attn/quant_per_block.py](https://github.com/thu-ml/SpargeAttn/blob/ae5b629ebb41e41f86b3ea2ab5a3283f13ac151a/spas_sage_attn/quant_per_block.py)
- [csrc/qattn/attn_cuda.h](https://github.com/thu-ml/SpargeAttn/blob/ae5b629ebb41e41f86b3ea2ab5a3283f13ac151a/csrc/qattn/attn_cuda.h)
- [csrc/qattn/pybind.cpp](https://github.com/thu-ml/SpargeAttn/blob/ae5b629ebb41e41f86b3ea2ab5a3283f13ac151a/csrc/qattn/pybind.cpp)
- [csrc/qattn/qk_int_sv_f16_cuda_sm80.cu](https://github.com/thu-ml/SpargeAttn/blob/ae5b629ebb41e41f86b3ea2ab5a3283f13ac151a/csrc/qattn/qk_int_sv_f16_cuda_sm80.cu)
- [csrc/qattn/qk_int_sv_f16_cuda_sm80.cuh](https://github.com/thu-ml/SpargeAttn/blob/ae5b629ebb41e41f86b3ea2ab5a3283f13ac151a/csrc/qattn/qk_int_sv_f16_cuda_sm80.cuh)

<a id="s14a"></a>
## S14a — SageAttention

仓库：[thu-ml/SageAttention](https://github.com/thu-ml/SageAttention)；commit：`d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5`。

阅读：S；per_block/per_warp接口、K/V smoothing、scale/log2/layout；Triton add-half/trunc与CUDA amax floor/RNE helper及入口dispatch选段；不审计全部attention内核。

历史观察范围：上游代码仅阅读，未执行CUDA/Triton。第五轮另做原创NumPy量化/稀疏消融，不是Sage/Sparge后端逐位复现或性能测量。

- [sageattention/quant.py](https://github.com/thu-ml/SageAttention/blob/d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5/sageattention/quant.py)
- [sageattention/triton/quant_per_block.py](https://github.com/thu-ml/SageAttention/blob/d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5/sageattention/triton/quant_per_block.py)
- [csrc/fused/fused.cu](https://github.com/thu-ml/SageAttention/blob/d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5/csrc/fused/fused.cu)
- [csrc/numeric_conversion.cuh](https://github.com/thu-ml/SageAttention/blob/d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5/csrc/numeric_conversion.cuh)

<a id="s15"></a>
## S15 — FastVideo STA/VSA 与 MLX

仓库：[hao-ai-lab/FastVideo](https://github.com/hao-ai-lab/FastVideo)；commit：`a943220c115228ade5d57b3bab9a6a87fd600a10`。

阅读：S；STA归档状态、VSA checkpoint/gate/Metal质量限制、MLX窗口切片+sink的完整核心。

历史观察范围：仅阅读；未安装/运行上游模型或基准。作者性能数字仅作来源报告。

- [docs/attention/sta/index.md](https://github.com/hao-ai-lab/FastVideo/blob/a943220c115228ade5d57b3bab9a6a87fd600a10/docs/attention/sta/index.md)
- [docs/attention/vsa/index.md](https://github.com/hao-ai-lab/FastVideo/blob/a943220c115228ade5d57b3bab9a6a87fd600a10/docs/attention/vsa/index.md)
- [fastvideo/mlx_runtime/windowed_attention.py](https://github.com/hao-ai-lab/FastVideo/blob/a943220c115228ade5d57b3bab9a6a87fd600a10/fastvideo/mlx_runtime/windowed_attention.py)

<a id="s16"></a>
## S16 — ToMeSD

仓库：[dbolya/tomesd](https://github.com/dbolya/tomesd)；commit：`27a14a372beecaa85c101fb588631643f379b8ce`。

阅读：S；随机2D二分匹配、MPS gather workaround、scatter mean、unmerge和attn/cross/MLP默认范围。

历史观察范围：仅阅读；未安装/运行上游模型或基准。作者性能数字仅作来源报告。

- [README.md](https://github.com/dbolya/tomesd/blob/27a14a372beecaa85c101fb588631643f379b8ce/README.md)
- [tomesd/merge.py](https://github.com/dbolya/tomesd/blob/27a14a372beecaa85c101fb588631643f379b8ce/tomesd/merge.py)
- [tomesd/patch.py](https://github.com/dbolya/tomesd/blob/27a14a372beecaa85c101fb588631643f379b8ce/tomesd/patch.py)

<a id="s17"></a>
## S17 — Nunchaku / SVDQuant

仓库：[nunchaku-ai/nunchaku](https://github.com/nunchaku-ai/nunchaku)；commit：`302e0e97024ebd68688fe890e5df83731edf7b54`。

阅读：S；方法说明、linear/quantize/GEMM wrapper的输入/输出/scale布局、M padding与FP32低秩hidden；CUDA LoRA up/down helper及tile epilogue选段；另读作者blog的方法/融合与offload口径，不审计完整模型。

历史观察范围：上游仅阅读，未执行Nunchaku或CUDA；第六轮另做原创MLX packed affine W4A16+分离FP16低秩的合成GPU组件，不是W4A4/NVFP4、完整FFN或模型复现。

- [README.md](https://github.com/nunchaku-ai/nunchaku/blob/302e0e97024ebd68688fe890e5df83731edf7b54/README.md)
- [nunchaku/models/linear.py](https://github.com/nunchaku-ai/nunchaku/blob/302e0e97024ebd68688fe890e5df83731edf7b54/nunchaku/models/linear.py)
- [nunchaku/ops/quantize.py](https://github.com/nunchaku-ai/nunchaku/blob/302e0e97024ebd68688fe890e5df83731edf7b54/nunchaku/ops/quantize.py)
- [nunchaku/ops/gemm.py](https://github.com/nunchaku-ai/nunchaku/blob/302e0e97024ebd68688fe890e5df83731edf7b54/nunchaku/ops/gemm.py)
- [src/kernels/zgemm/lora.cuh](https://github.com/nunchaku-ai/nunchaku/blob/302e0e97024ebd68688fe890e5df83731edf7b54/src/kernels/zgemm/lora.cuh)
- 补充已读原文：[补充资料](https://hanlab.mit.edu/blog/svdquant)（不在上述文件哈希范围内）

<a id="s18"></a>
## S18 — SmoothQuant

仓库：[mit-han-lab/smoothquant](https://github.com/mit-han-lab/smoothquant)；commit：`c61476d728e42ae0d8a35e7e78494edcac3237b5`。

阅读：S；scale公式、LN/RMSNorm与多linear共同补偿；LLM实现边界。

历史观察范围：仅阅读；未安装/运行上游模型或基准。作者性能数字仅作来源报告。

- [smoothquant/smooth.py](https://github.com/mit-han-lab/smoothquant/blob/c61476d728e42ae0d8a35e7e78494edcac3237b5/smoothquant/smooth.py)

<a id="s19"></a>
## S19 — MLX quantized Metal

仓库：[ml-explore/mlx](https://github.com/ml-explore/mlx)；commit：`36969c094354086e15d2a327a9c50feb20b9cf11`。

阅读：S；Metal量化dispatch/tile/alignment片段；另读固定commit的Python quantize/QuantizedLinear/QQLinear原始文件，不在此文件hash清单。

历史观察范围：仅阅读；未安装/运行上游模型或基准。作者性能数字仅作来源报告。

- [mlx/backend/metal/quantized.cpp](https://github.com/ml-explore/mlx/blob/36969c094354086e15d2a327a9c50feb20b9cf11/mlx/backend/metal/quantized.cpp)
- 补充已读原文：[补充资料](https://raw.githubusercontent.com/ml-explore/mlx/36969c094354086e15d2a327a9c50feb20b9cf11/python/mlx/nn/layers/quantized.py)（不在上述文件哈希范围内）

<a id="s20"></a>
## S20 — mflux

仓库：[filipstrand/mflux](https://github.com/filipstrand/mflux)；commit：`8c00dab2505a96019df9d30bc9c223bf20d733c4`。

阅读：S；README第三方MLX cache入口与当前Qwen量化predicate。

历史观察范围：仅阅读；未安装/运行上游模型或基准。作者性能数字仅作来源报告。

- [README.md](https://github.com/filipstrand/mflux/blob/8c00dab2505a96019df9d30bc9c223bf20d733c4/README.md)
- [src/mflux/models/qwen/weights/qwen_weight_definition.py](https://github.com/filipstrand/mflux/blob/8c00dab2505a96019df9d30bc9c223bf20d733c4/src/mflux/models/qwen/weights/qwen_weight_definition.py)

<a id="s21"></a>
## S21 — Latent Consistency Models

仓库：[luosiallen/latent-consistency-model](https://github.com/luosiallen/latent-consistency-model)；commit：`a9ad79587cc8bd1e404ccd1a3056a3da969b2f62`。

阅读：D；蒸馏/LCM-LoRA/scheduler说明与MPS使用入口，不运行训练或推理。

历史观察范围：仅阅读；未安装/运行上游模型或基准。作者性能数字仅作来源报告。

- [README.md](https://github.com/luosiallen/latent-consistency-model/blob/a9ad79587cc8bd1e404ccd1a3056a3da969b2f62/README.md)

<a id="s22"></a>
## S22 — LTX-2-MLX

仓库：[dgrauet/ltx-2-mlx](https://github.com/dgrauet/ltx-2-mlx)；commit：`91e6f6c9bd621ff2ae31adfee643e113d67d6ae8`。

阅读：S；两阶段角色、TeaCache校准说明/构建、2.5拒绝guard及sampler signal/cached residual片段。

历史观察范围：仅阅读；未安装/运行上游模型或基准。作者性能数字仅作来源报告。

- [README.md](https://github.com/dgrauet/ltx-2-mlx/blob/91e6f6c9bd621ff2ae31adfee643e113d67d6ae8/README.md)
- [packages/ltx-pipelines-mlx/src/ltx_pipelines_mlx/ti2vid_two_stages.py](https://github.com/dgrauet/ltx-2-mlx/blob/91e6f6c9bd621ff2ae31adfee643e113d67d6ae8/packages/ltx-pipelines-mlx/src/ltx_pipelines_mlx/ti2vid_two_stages.py)
- [packages/ltx-pipelines-mlx/src/ltx_pipelines_mlx/_base.py](https://github.com/dgrauet/ltx-2-mlx/blob/91e6f6c9bd621ff2ae31adfee643e113d67d6ae8/packages/ltx-pipelines-mlx/src/ltx_pipelines_mlx/_base.py)
- [packages/ltx-pipelines-mlx/src/ltx_pipelines_mlx/utils/samplers.py](https://github.com/dgrauet/ltx-2-mlx/blob/91e6f6c9bd621ff2ae31adfee643e113d67d6ae8/packages/ltx-pipelines-mlx/src/ltx_pipelines_mlx/utils/samplers.py)

<a id="s23"></a>
## S23 — TAESD

仓库：[madebyollin/taesd](https://github.com/madebyollin/taesd)；commit：`027d3ef404a04a141e60d5ec0fee149c2c8bdd74`。

阅读：S；模型家族/latent合同/质量限制；Conv/upsample、F16/F32与带全局norm的变体。

历史观察范围：仅阅读；未安装/运行上游模型或基准。作者性能数字仅作来源报告。

- [README.md](https://github.com/madebyollin/taesd/blob/027d3ef404a04a141e60d5ec0fee149c2c8bdd74/README.md)
- [taesd.py](https://github.com/madebyollin/taesd/blob/027d3ef404a04a141e60d5ec0fee149c2c8bdd74/taesd.py)

<a id="s24"></a>
## S24 — DeepCache

仓库：[horseee/DeepCache](https://github.com/horseee/DeepCache)；commit：`1691a3ab22308fe06685e4ff151a9ab175be2dc4`。

阅读：S；U-Net缓存机制说明与helper位置/周期/wrap/unwrap/reset控制。

历史观察范围：仅阅读；未安装/运行上游模型或基准。作者性能数字仅作来源报告。

- [README.md](https://github.com/horseee/DeepCache/blob/1691a3ab22308fe06685e4ff151a9ab175be2dc4/README.md)
- [DeepCache/extension/deepcache.py](https://github.com/horseee/DeepCache/blob/1691a3ab22308fe06685e4ff151a9ab175be2dc4/DeepCache/extension/deepcache.py)

<a id="s25"></a>
## S25 — Diffusers caching / scheduler

仓库：[huggingface/diffusers](https://github.com/huggingface/diffusers)；commit：`d42ebf9c6e0b46103b74283f5324357a67acaa04`。

阅读：S；cache机制与MPS文档入口；Sea indicator/metadata/gate和Mag config/ratio/状态控制片段；不是所有hooks或全部Sea adapter审计。

历史观察范围：仅阅读；未安装/运行上游模型或基准。作者性能数字仅作来源报告。

- [docs/source/en/optimization/cache.md](https://github.com/huggingface/diffusers/blob/d42ebf9c6e0b46103b74283f5324357a67acaa04/docs/source/en/optimization/cache.md)
- [docs/source/en/api/schedulers/multistep_dpm_solver.md](https://github.com/huggingface/diffusers/blob/d42ebf9c6e0b46103b74283f5324357a67acaa04/docs/source/en/api/schedulers/multistep_dpm_solver.md)
- [src/diffusers/hooks/sea_cache.py](https://github.com/huggingface/diffusers/blob/d42ebf9c6e0b46103b74283f5324357a67acaa04/src/diffusers/hooks/sea_cache.py)
- [src/diffusers/hooks/mag_cache.py](https://github.com/huggingface/diffusers/blob/d42ebf9c6e0b46103b74283f5324357a67acaa04/src/diffusers/hooks/mag_cache.py)

<a id="s26"></a>
## S26 — Radial Attention

仓库：[mit-han-lab/radial-attention](https://github.com/mit-han-lab/radial-attention)；commit：`72788d4f0a6d202f1ec5f1c98a6e4c8b2e34fdbc`。

阅读：S；距离band/密度规律、LoRA延长与原长度区分、mask/CSR转换、时空窗口与sink代码片段。

历史观察范围：仅阅读；未安装/运行上游模型或基准。作者性能数字仅作来源报告。

- [README.md](https://github.com/mit-han-lab/radial-attention/blob/72788d4f0a6d202f1ec5f1c98a6e4c8b2e34fdbc/README.md)
- [radial_attn/attn_mask.py](https://github.com/mit-han-lab/radial-attention/blob/72788d4f0a6d202f1ec5f1c98a6e4c8b2e34fdbc/radial_attn/attn_mask.py)

<a id="s27"></a>
## S27 — Apple Core ML Stable Diffusion

仓库：[apple/ml-stable-diffusion](https://github.com/apple/ml-stable-diffusion)；commit：`ea2805dc1945be20561c77e5f6d1d9a5a637cda2`。

阅读：D；官方weight compression/MBP/activation sensitivity说明与后端条件；不重跑转换脚本。

历史观察范围：仅阅读；未安装/运行上游模型或基准。作者性能数字仅作来源报告。

- [README.md](https://github.com/apple/ml-stable-diffusion/blob/ea2805dc1945be20561c77e5f6d1d9a5a637cda2/README.md)

<a id="s28"></a>
## S28 — mlx-taef

仓库：[IonDen/mlx-taef](https://github.com/IonDen/mlx-taef)；commit：`cd4340e4190b7964e816a3dcb67fba5bcd27ca2d`。

阅读：D；纯MLX tiny decoder、latent变体/预览callback、normalization及旧基准修正说明；不审计所有模型源码。

历史观察范围：仅阅读；未安装/运行上游模型或基准。作者性能数字仅作来源报告。

- [README.md](https://github.com/IonDen/mlx-taef/blob/cd4340e4190b7964e816a3dcb67fba5bcd27ca2d/README.md)

<a id="s29"></a>
## S29 — TurboCider 既有 Apple 正反实验

仓库：TurboCider（本地历史，可选产品溯源）；commit：`614fdf2bc5abf05b050688f8b40dcc65580e1af4`。

阅读：D；历史LTX sparse质量/阶段负结果与Qwen DBCache完整请求报告；本轮只读文档，没有重跑原始媒体基准。

历史观察范围：既有本地报告，非本轮重跑；仓库公开URL未验证，本项以本地相对路径为权威。

- [docs/design/ltx-stage2-sparse-results.md](../../docs/design/ltx-stage2-sparse-results.md)
- [docs/status/2026-09-30-qwen21-dit-cache.md](../../docs/status/2026-09-30-qwen21-dit-cache.md)

<a id="s30"></a>
## S30 — ToCa token-wise feature caching

仓库：[Shenyi-Z/ToCa](https://github.com/Shenyi-Z/ToCa)；commit：`e84096ffd85af4540a6a1f64e3334e428e1b7377`。

阅读：S；DiT full/ToCa forward、gate前分支缓存、MLP fresh gather/scatter、attention-map成本、age/空间bonus和ratio schedule；未审计全部模型版本。

历史观察范围：仅阅读固定源码，未运行上游package、模型、基准或Swift测试。

- [README.md](https://github.com/Shenyi-Z/ToCa/blob/e84096ffd85af4540a6a1f64e3334e428e1b7377/README.md)
- [DiT-ToCa/models.py](https://github.com/Shenyi-Z/ToCa/blob/e84096ffd85af4540a6a1f64e3334e428e1b7377/DiT-ToCa/models.py)
- [DiT-ToCa/cache_functions/attention.py](https://github.com/Shenyi-Z/ToCa/blob/e84096ffd85af4540a6a1f64e3334e428e1b7377/DiT-ToCa/cache_functions/attention.py)
- [DiT-ToCa/cache_functions/scores.py](https://github.com/Shenyi-Z/ToCa/blob/e84096ffd85af4540a6a1f64e3334e428e1b7377/DiT-ToCa/cache_functions/scores.py)
- [DiT-ToCa/cache_functions/score_evaluate.py](https://github.com/Shenyi-Z/ToCa/blob/e84096ffd85af4540a6a1f64e3334e428e1b7377/DiT-ToCa/cache_functions/score_evaluate.py)
- [DiT-ToCa/cache_functions/cache_cutfresh.py](https://github.com/Shenyi-Z/ToCa/blob/e84096ffd85af4540a6a1f64e3334e428e1b7377/DiT-ToCa/cache_functions/cache_cutfresh.py)
- [DiT-ToCa/cache_functions/fresh_ratio_scheduler.py](https://github.com/Shenyi-Z/ToCa/blob/e84096ffd85af4540a6a1f64e3334e428e1b7377/DiT-ToCa/cache_functions/fresh_ratio_scheduler.py)
- [DiT-ToCa/cache_functions/update_cache.py](https://github.com/Shenyi-Z/ToCa/blob/e84096ffd85af4540a6a1f64e3334e428e1b7377/DiT-ToCa/cache_functions/update_cache.py)

<a id="s31"></a>
## S31 — DuCa dual feature caching

仓库：[Shenyi-Z/DuCa](https://github.com/Shenyi-Z/DuCa)；commit：`0a7151b7bfea3822295e8d29f26cd2e10756670c`。

阅读：S；DiT实际执行类的full/ToCa/aggressive/fallback与调度；区分README FlashAttention声明、attention-map配置和保留实验分支。

历史观察范围：仅阅读固定源码，未运行上游package、模型、基准或Swift测试。

- [README.md](https://github.com/Shenyi-Z/DuCa/blob/0a7151b7bfea3822295e8d29f26cd2e10756670c/README.md)
- [DuCa-DiT/models.py](https://github.com/Shenyi-Z/DuCa/blob/0a7151b7bfea3822295e8d29f26cd2e10756670c/DuCa-DiT/models.py)
- [DuCa-DiT/cache_functions/attention.py](https://github.com/Shenyi-Z/DuCa/blob/0a7151b7bfea3822295e8d29f26cd2e10756670c/DuCa-DiT/cache_functions/attention.py)
- [DuCa-DiT/cache_functions/cal_type.py](https://github.com/Shenyi-Z/DuCa/blob/0a7151b7bfea3822295e8d29f26cd2e10756670c/DuCa-DiT/cache_functions/cal_type.py)
- [DuCa-DiT/cache_functions/cache_init.py](https://github.com/Shenyi-Z/DuCa/blob/0a7151b7bfea3822295e8d29f26cd2e10756670c/DuCa-DiT/cache_functions/cache_init.py)
- [DuCa-DiT/cache_functions/score_evaluate.py](https://github.com/Shenyi-Z/DuCa/blob/0a7151b7bfea3822295e8d29f26cd2e10756670c/DuCa-DiT/cache_functions/score_evaluate.py)

<a id="s32"></a>
## S32 — Spectrum spectral forecasting

仓库：[hanjq17/Spectrum](https://github.com/hanjq17/Spectrum)；commit：`4e41f91c93ff66b045332530dd7bcb4c60eafbab`。

阅读：S；Chebyshev ridge/Cholesky、fixed clock、ready/history/alias、blend、主体feature更新和FLUX refresh/tail接缝。

历史观察范围：对basis_utils白名单AST声明做CPU状态观察；不导入完整项目、不生成媒体，具体12-case范围见source-state-round2-20261016.json。

- [README.md](https://github.com/hanjq17/Spectrum/blob/4e41f91c93ff66b045332530dd7bcb4c60eafbab/README.md)
- [src/utils/basis_utils.py](https://github.com/hanjq17/Spectrum/blob/4e41f91c93ff66b045332530dd7bcb4c60eafbab/src/utils/basis_utils.py)
- [src/utils/__init__.py](https://github.com/hanjq17/Spectrum/blob/4e41f91c93ff66b045332530dd7bcb4c60eafbab/src/utils/__init__.py)
- [src/pipelines/flux_forward.py](https://github.com/hanjq17/Spectrum/blob/4e41f91c93ff66b045332530dd7bcb4c60eafbab/src/pipelines/flux_forward.py)
- [configs/algo/spectrum.yaml](https://github.com/hanjq17/Spectrum/blob/4e41f91c93ff66b045332530dd7bcb4c60eafbab/configs/algo/spectrum.yaml)
- [configs/base.yaml](https://github.com/hanjq17/Spectrum/blob/4e41f91c93ff66b045332530dd7bcb4c60eafbab/configs/base.yaml)

<a id="s33"></a>
## S33 — Cache-DiT FoCa / Dynamic Mode Decomposition

仓库：[vipshop/cache-dit](https://github.com/vipshop/cache-dit)；commit：`089df6b6895a3bc15ebcd3f7546d90ec6eb592e8`。

阅读：S；完整状态类、rolling/full anchors、时间差分/重复读取、DMD uniform suffix、FP64/complex128拟合、fallback与finite-cast。

历史观察范围：固定digest的CPU state控制及第三轮manager/gate/forward接缝重放；保留所有oracle偏差与诊断读取的区别，未跑完整模型。

- [src/cache_dit/caching/cache_contexts/calibrators/foca.py](https://github.com/vipshop/cache-dit/blob/089df6b6895a3bc15ebcd3f7546d90ec6eb592e8/src/cache_dit/caching/cache_contexts/calibrators/foca.py)
- [src/cache_dit/caching/cache_contexts/calibrators/dmd.py](https://github.com/vipshop/cache-dit/blob/089df6b6895a3bc15ebcd3f7546d90ec6eb592e8/src/cache_dit/caching/cache_contexts/calibrators/dmd.py)

<a id="s34"></a>
## S34 — DiTFastAttn redundancy composition

仓库：[thu-nics/DiTFastAttn](https://github.com/thu-nics/DiTFastAttn)；commit：`91bfbaee962d839c433e68c6fab1aeaa78c30c28`。

阅读：S；WA-RS full-window差、output/CFG共享、校准hook防state污染、step/layer计划、breakpoint与外部metric依赖。

历史观察范围：仅阅读固定源码，未运行上游package、模型、基准或Swift测试。

- [README.md](https://github.com/thu-nics/DiTFastAttn/blob/91bfbaee962d839c433e68c6fab1aeaa78c30c28/README.md)
- [dit_fast_attention.py](https://github.com/thu-nics/DiTFastAttn/blob/91bfbaee962d839c433e68c6fab1aeaa78c30c28/dit_fast_attention.py)
- [modules/fast_attn_processor.py](https://github.com/thu-nics/DiTFastAttn/blob/91bfbaee962d839c433e68c6fab1aeaa78c30c28/modules/fast_attn_processor.py)

<a id="s35"></a>
## S35 — VMoBA block routing

仓库：[KlingAIResearch/VMoBA](https://github.com/KlingAIResearch/VMoBA)；commit：`48aaccd4f14c5adb7db961058bfbb2113e392003`。

阅读：S；noncausal chunks、valid tail、query/head/global选择器、minmax阈值、own/remote分离、varlen/LSE合并与empty-remote dense fallback；不审计backward。

历史观察范围：仅阅读固定源码，未运行上游package、模型、基准或Swift测试。

- [README.md](https://github.com/KlingAIResearch/VMoBA/blob/48aaccd4f14c5adb7db961058bfbb2113e392003/README.md)
- [src/vmoba.py](https://github.com/KlingAIResearch/VMoBA/blob/48aaccd4f14c5adb7db961058bfbb2113e392003/src/vmoba.py)

<a id="s36"></a>
## S36 — Draw Things Apple-native caching graphs

仓库：[drawthingsai/draw-things-community](https://github.com/drawthingsai/draw-things-community)；commit：`8d9c62ad38539ef8bb123f8b8e78cab76caafd0c`。

阅读：S；TeaCache控制器完整核心、H3首块/全局层号/tail/replay-head片段、参数共享、marker、多模态max、raw同步与相关测试定义；未执行Swift测试。

历史观察范围：仅阅读固定源码，未运行上游package、模型、基准或Swift测试。

- [Libraries/SwiftDiffusion/Sources/TeaCache/TeaCache.swift](https://github.com/drawthingsai/draw-things-community/blob/8d9c62ad38539ef8bb123f8b8e78cab76caafd0c/Libraries/SwiftDiffusion/Sources/TeaCache/TeaCache.swift)
- [Libraries/SwiftDiffusion/Sources/Models/MiniMaxH3.swift](https://github.com/drawthingsai/draw-things-community/blob/8d9c62ad38539ef8bb123f8b8e78cab76caafd0c/Libraries/SwiftDiffusion/Sources/Models/MiniMaxH3.swift)
- [Libraries/SwiftDiffusion/Tests/TeaCacheTests.swift](https://github.com/drawthingsai/draw-things-community/blob/8d9c62ad38539ef8bb123f8b8e78cab76caafd0c/Libraries/SwiftDiffusion/Tests/TeaCacheTests.swift)

<a id="s37"></a>
## S37 — Cache-DiT consumer buffer dispatch

仓库：[vipshop/cache-dit](https://github.com/vipshop/cache-dit)；commit：`089df6b6895a3bc15ebcd3f7546d90ec6eb592e8`。

阅读：S；can_cache/similarity、Fn/Bn prefix、buffer/calibrator dispatch、apply_cache与相关getter；结合S06 pattern/context执行指定方法；不审计全部pipeline/CFG/refresh。

历史观察范围：白名单AST方法在Torch CPU执行；context存储/选择与F/M/B为显式微型替身，使用原gate得到full/hit表，另注入一次诊断读取；无GPU/模型。

- [src/cache_dit/caching/cache_contexts/cache_manager.py](https://github.com/vipshop/cache-dit/blob/089df6b6895a3bc15ebcd3f7546d90ec6eb592e8/src/cache_dit/caching/cache_contexts/cache_manager.py)

<a id="s38"></a>
## S38 — DiCache online probe and trajectory alignment

仓库：[Bujiazi/DiCache](https://github.com/Bujiazi/DiCache)；commit：`fdbe20b669c9174bbed5ec994de073fd881c8010`。

阅读：S；README、三个dicache_forward的probe/gate/resume/history/gamma/reset与启动配置；FLUX ControlNet恢复索引、零分母，Wan奇偶CFG与Hunyuan边界；未审计全部基础模型模块。

历史观察范围：仅提取FLUX原forward AST，用明确CPU四个加法block及identity prelude/tail重放5个接缝；34行保留NaN与ControlNet恢复反例，不执行顶层CUDA/下载/monkeypatch；其余两个adapter只读。

- [readme.md](https://github.com/Bujiazi/DiCache/blob/fdbe20b669c9174bbed5ec994de073fd881c8010/readme.md)
- [FLUX/run_flux_dicache.py](https://github.com/Bujiazi/DiCache/blob/fdbe20b669c9174bbed5ec994de073fd881c8010/FLUX/run_flux_dicache.py)
- [HunyuanVideo/run_hunyuanvideo_dicache.py](https://github.com/Bujiazi/DiCache/blob/fdbe20b669c9174bbed5ec994de073fd881c8010/HunyuanVideo/run_hunyuanvideo_dicache.py)
- [WAN2.1/run_wan_dicache.py](https://github.com/Bujiazi/DiCache/blob/fdbe20b669c9174bbed5ec994de073fd881c8010/WAN2.1/run_wan_dicache.py)

<a id="s39"></a>
## S39 — MLX v0.32.3 affine decode and packed QMM

仓库：[ml-explore/mlx](https://github.com/ml-explore/mlx)；commit：`64ea011cb65f14d9ce2737e60db9a4ae91ed7441`。

阅读：S；v0.32.3 tag固定；API affine_dequantize/dequantize的scales dtype与末端astype、Metal dispatch/type选择、affine解码与QMV/QMM/split-K/NAX守卫选段；未逐行审计全部kernel。

历史观察范围：使用已有MLX0.32.3 wheel执行实际GPU affine quantize/QMM和显式FP32-promoted解码控制，记录wheel/native/stub SHA；tag与binary不是reproducible-build证明，不从source静态分支宣称实测ISA或kernel计数。

- [mlx/ops.cpp](https://github.com/ml-explore/mlx/blob/64ea011cb65f14d9ce2737e60db9a4ae91ed7441/mlx/ops.cpp)
- [mlx/backend/metal/quantized.cpp](https://github.com/ml-explore/mlx/blob/64ea011cb65f14d9ce2737e60db9a4ae91ed7441/mlx/backend/metal/quantized.cpp)
- [mlx/backend/metal/kernels/quantized.h](https://github.com/ml-explore/mlx/blob/64ea011cb65f14d9ce2737e60db9a4ae91ed7441/mlx/backend/metal/kernels/quantized.h)
- [mlx/backend/metal/kernels/quantized.metal](https://github.com/ml-explore/mlx/blob/64ea011cb65f14d9ce2737e60db9a4ae91ed7441/mlx/backend/metal/kernels/quantized.metal)

<a id="s40"></a>
## S40 — SLA public main and SageSLA forward

仓库：[thu-ml/SLA](https://github.com/thu-ml/SLA)；commit：`7db4039111a9002f900be537c2b2061d26d73744`。

阅读：S；完整Python core/route helper、稀疏kernel forward与Autograd接口选段、SageSLA构造/forward/README；核对全局线性分支、零初始化、topk与density、epsilon、可训练参数和SLA2 citation，不视为SLA2代码等价证明。

历史观察范围：仅core/get_block_map白名单AST及Sage构造在Torch CPU执行；内部dtype显式FP32、CUDA autocast no-op，pool/attention/反向为CPU参考替身；两seed结构/梯度接缝，不执行Triton、CUDA、QAT或模型训练。

- [README.md](https://github.com/thu-ml/SLA/blob/7db4039111a9002f900be537c2b2061d26d73744/README.md)
- [sparse_linear_attention/core.py](https://github.com/thu-ml/SLA/blob/7db4039111a9002f900be537c2b2061d26d73744/sparse_linear_attention/core.py)
- [sparse_linear_attention/utils.py](https://github.com/thu-ml/SLA/blob/7db4039111a9002f900be537c2b2061d26d73744/sparse_linear_attention/utils.py)
- [sparse_linear_attention/kernel.py](https://github.com/thu-ml/SLA/blob/7db4039111a9002f900be537c2b2061d26d73744/sparse_linear_attention/kernel.py)
- [SageSLA/README.md](https://github.com/thu-ml/SLA/blob/7db4039111a9002f900be537c2b2061d26d73744/SageSLA/README.md)
- [SageSLA/core.py](https://github.com/thu-ml/SLA/blob/7db4039111a9002f900be537c2b2061d26d73744/SageSLA/core.py)
- [SageSLA/utils.py](https://github.com/thu-ml/SLA/blob/7db4039111a9002f900be537c2b2061d26d73744/SageSLA/utils.py)
- 补充已读原文：[补充资料](https://arxiv.org/abs/2602.12675v1)（不在上述文件哈希范围内）

<a id="s41"></a>
## S41 — SLA-1 complement-linear branch

仓库：[thu-ml/SLA](https://github.com/thu-ml/SLA)；commit：`325cc7519f79883778ce0ee64a95533ce5eab35d`。

阅读：S；Python类/route完整、native summary/preprocess/fused forward与Autograd接口选段；重点全局减selected得到补集、full选取的Inf denominator和缺少epsilon；不审计全部GPU backward。

历史观察范围：与main相同FP32 CPU参考包络，原class/helper body不改；严格同shape state load后比较语义、scalar见证和单步projection梯度，不授予原CUDA/BF16/训练pipeline资格。

- [README.md](https://github.com/thu-ml/SLA/blob/325cc7519f79883778ce0ee64a95533ce5eab35d/README.md)
- [sparse_linear_attention/core.py](https://github.com/thu-ml/SLA/blob/325cc7519f79883778ce0ee64a95533ce5eab35d/sparse_linear_attention/core.py)
- [sparse_linear_attention/utils.py](https://github.com/thu-ml/SLA/blob/325cc7519f79883778ce0ee64a95533ce5eab35d/sparse_linear_attention/utils.py)
- [sparse_linear_attention/kernel.py](https://github.com/thu-ml/SLA/blob/325cc7519f79883778ce0ee64a95533ce5eab35d/sparse_linear_attention/kernel.py)

<a id="s42"></a>
## S42 — MLX TeaCache gate, Qwen CFG and generation transactions

仓库：[IonDen/mlx-teacache](https://github.com/IonDen/mlx-teacache)；commit：`b56f2882ca237b3b971c976f78804154cbeb1014`。

阅读：S；完整gate/cache/stats/window/lifecycle/pairer、Qwen integration与branch-pairing测试；重点FP16先减后cast、forced/finite优先级、正负独立residual、active img2img窗口、staging/commit/cleanup及callback身份。

历史观察范围：实际MLX0.32.0 CPU执行原gate/cache/stats/window/lifecycle/pairer及白名单Qwen helper/class；14个gate与两seed/30个generation包络，两affine block/固定signal为明确模型替身。没有mflux/package initializer、GPU、真实Qwen模型或媒体资格；违反配对/shape/条件不变前提的诊断单列，不声称正常pipeline有同类缺陷。

- [LICENSE](https://github.com/IonDen/mlx-teacache/blob/b56f2882ca237b3b971c976f78804154cbeb1014/LICENSE)
- [src/mlx_teacache/errors.py](https://github.com/IonDen/mlx-teacache/blob/b56f2882ca237b3b971c976f78804154cbeb1014/src/mlx_teacache/errors.py)
- [src/mlx_teacache/_kernel/cache.py](https://github.com/IonDen/mlx-teacache/blob/b56f2882ca237b3b971c976f78804154cbeb1014/src/mlx_teacache/_kernel/cache.py)
- [src/mlx_teacache/_kernel/gate.py](https://github.com/IonDen/mlx-teacache/blob/b56f2882ca237b3b971c976f78804154cbeb1014/src/mlx_teacache/_kernel/gate.py)
- [src/mlx_teacache/_kernel/stats.py](https://github.com/IonDen/mlx-teacache/blob/b56f2882ca237b3b971c976f78804154cbeb1014/src/mlx_teacache/_kernel/stats.py)
- [src/mlx_teacache/_kernel/window.py](https://github.com/IonDen/mlx-teacache/blob/b56f2882ca237b3b971c976f78804154cbeb1014/src/mlx_teacache/_kernel/window.py)
- [src/mlx_teacache/integrations/mflux/lifecycle.py](https://github.com/IonDen/mlx-teacache/blob/b56f2882ca237b3b971c976f78804154cbeb1014/src/mlx_teacache/integrations/mflux/lifecycle.py)
- [src/mlx_teacache/variants/qwen_image/pairing.py](https://github.com/IonDen/mlx-teacache/blob/b56f2882ca237b3b971c976f78804154cbeb1014/src/mlx_teacache/variants/qwen_image/pairing.py)
- [src/mlx_teacache/variants/qwen_image/integration.py](https://github.com/IonDen/mlx-teacache/blob/b56f2882ca237b3b971c976f78804154cbeb1014/src/mlx_teacache/variants/qwen_image/integration.py)
- [tests/test_qwen_branch_pairing.py](https://github.com/IonDen/mlx-teacache/blob/b56f2882ca237b3b971c976f78804154cbeb1014/tests/test_qwen_branch_pairing.py)

<a id="s43"></a>
## S43 — DeepCompressor low-rank calibration and SVDQuant conversion semantics

仓库：[nunchux-ai/deepcompressor](https://github.com/nunchux-ai/deepcompressor)；commit：`69f3473f5e1c1504bae35cc50c7858ef900a9b17`。

阅读：S；LowRankBranch双精度SVD/因子布局、QuantLowRankCalibrator的w-qw迭代与best/early-stop、one-shot compensate旁路、共享A/分B、smooth/eval/hook调用链及固定INT4/default/fast配置；README方法/支持边界选段，不审计完整toolbox或转码器。

历史观察范围：仅固定Git源码阅读，未安装或执行DeepCompressor/CUDA/model converter。新增本库真实Qwen实验使用原创range80随机分解和diagonal-RMS变体、实际MLX CPU affine W4/W8，不能标为作者100轮输出目标/平滑网格/W4A4实现复现。

- [README.md](https://github.com/nunchux-ai/deepcompressor/blob/69f3473f5e1c1504bae35cc50c7858ef900a9b17/README.md)
- [LICENSE](https://github.com/nunchux-ai/deepcompressor/blob/69f3473f5e1c1504bae35cc50c7858ef900a9b17/LICENSE)
- [deepcompressor/nn/patch/lowrank.py](https://github.com/nunchux-ai/deepcompressor/blob/69f3473f5e1c1504bae35cc50c7858ef900a9b17/deepcompressor/nn/patch/lowrank.py)
- [deepcompressor/calib/lowrank.py](https://github.com/nunchux-ai/deepcompressor/blob/69f3473f5e1c1504bae35cc50c7858ef900a9b17/deepcompressor/calib/lowrank.py)
- [deepcompressor/calib/config/lowrank.py](https://github.com/nunchux-ai/deepcompressor/blob/69f3473f5e1c1504bae35cc50c7858ef900a9b17/deepcompressor/calib/config/lowrank.py)
- [deepcompressor/quantizer/config/lowrank.py](https://github.com/nunchux-ai/deepcompressor/blob/69f3473f5e1c1504bae35cc50c7858ef900a9b17/deepcompressor/quantizer/config/lowrank.py)
- [deepcompressor/quantizer/processor.py](https://github.com/nunchux-ai/deepcompressor/blob/69f3473f5e1c1504bae35cc50c7858ef900a9b17/deepcompressor/quantizer/processor.py)
- [deepcompressor/app/diffusion/quant/weight.py](https://github.com/nunchux-ai/deepcompressor/blob/69f3473f5e1c1504bae35cc50c7858ef900a9b17/deepcompressor/app/diffusion/quant/weight.py)
- [deepcompressor/app/diffusion/quant/smooth.py](https://github.com/nunchux-ai/deepcompressor/blob/69f3473f5e1c1504bae35cc50c7858ef900a9b17/deepcompressor/app/diffusion/quant/smooth.py)
- [examples/diffusion/configs/svdquant/__default__.yaml](https://github.com/nunchux-ai/deepcompressor/blob/69f3473f5e1c1504bae35cc50c7858ef900a9b17/examples/diffusion/configs/svdquant/__default__.yaml)
- [examples/diffusion/configs/svdquant/int4.yaml](https://github.com/nunchux-ai/deepcompressor/blob/69f3473f5e1c1504bae35cc50c7858ef900a9b17/examples/diffusion/configs/svdquant/int4.yaml)
- [examples/diffusion/configs/svdquant/fast.yaml](https://github.com/nunchux-ai/deepcompressor/blob/69f3473f5e1c1504bae35cc50c7858ef900a9b17/examples/diffusion/configs/svdquant/fast.yaml)

<a id="s44"></a>
## S44 — MLX v0.32.0 affine dtype and GPU full-FFN QMM dispatch

仓库：[ml-explore/mlx](https://github.com/ml-explore/mlx)；commit：`7a1d4f5c12ac82f4b4d0a6e71538d89ca0605247`。

阅读：S；v0.32.0 tag固定，affine_dequantize的scale dtype与末端astype、QuantizedMatmul的contiguous准备/qmv阈值、qmm_splitk几何/中间dtype、qmm与NAX守卫选段；不审计全部shader或编译优化。

历史观察范围：使用已安装MLX0.32.0公共GPU API执行实际BF16 packed QMM/独立低秩与完整SwiGLU组件，不从该tag宣称binary可复现构建、实测shader/ISA或物理计时。源码split-K预测和真实component调用分开，未安装/重编译此checkout。

- [mlx/ops.cpp](https://github.com/ml-explore/mlx/blob/7a1d4f5c12ac82f4b4d0a6e71538d89ca0605247/mlx/ops.cpp)
- [mlx/backend/metal/quantized.cpp](https://github.com/ml-explore/mlx/blob/7a1d4f5c12ac82f4b4d0a6e71538d89ca0605247/mlx/backend/metal/quantized.cpp)

<a id="s45"></a>
## S45 — TurboCider native Qwen BF16 request-local DBCache consumer

仓库：TurboCider（本地历史，可选产品溯源）；commit：`451161f99cc6530e414e793955d447ac4d73b456`。

阅读：S；固定Git对象与当前worktree逐文件一致；F8/W8预设/显式Off、相对L1与末步/连续守卫、front/middle anchor更新、leaf与异常清理、Euler consumer/step profile/result receipt、layer-streaming准入和C ABI字符串/engine ownership选段；不是全产品审计。

历史观察范围：现有source-matched648输入的原生库，纯GPU/BF16/component_staged/20步，两prompt各九次完整PNG-export生成；实际step跳过计数与源码loop/eval边界互证，不是物理shader计数、连续GPU独占或媒体/产品资格。未重建/修改产品；header/stat身份不宣称完整model payload SHA。

- [native/models/qwen21/diagnostic_options.hpp](../../native/models/qwen21/diagnostic_options.hpp)
- [native/models/qwen21/transformer.hpp](../../native/models/qwen21/transformer.hpp)
- [native/models/qwen21/transformer.cpp](../../native/models/qwen21/transformer.cpp)
- [native/models/qwen21/pipeline.cpp](../../native/models/qwen21/pipeline.cpp)
- [native/models/qwen21/memory_policy.hpp](../../native/models/qwen21/memory_policy.hpp)
- [native/models/qwen21_module.cpp](../../native/models/qwen21_module.cpp)
- [native/runtime/plan.cpp](../../native/runtime/plan.cpp)
- [native/platform/apple/results.mm](../../native/platform/apple/results.mm)
- [native/api/c_api.mm](../../native/api/c_api.mm)
- [bindings/c/include/turbocider/turbocider.h](../../bindings/c/include/turbocider/turbocider.h)

<a id="s46"></a>
## S46 — FasterCache CogVideoX actual CFG order, spatial FFT and attention-refresh coupling

仓库：[Vchitect/FasterCache](https://github.com/Vchitect/FasterCache)；commit：`02d05ef7edb48bdec01d2b4df6edfe7f472a41b9`。

阅读：S；CogVideoX脚本FFT/DiT/layer完整cache接缝、sampling_main reset；DynamicCFG prepare/merge及VideoDDIM denoise/训练timestep坐标；Denoiser/Wrapper保持batch次序、原DiffusionTransformer入口与固定50-step config选段。不审计所有模型/并行adapter。

历史观察范围：实际Torch2.14.0 CPU执行固定AST函数/方法，微型affine attention/MLP、zero time embedding、identity denoiser scaling与两模式reshape adapter为明确替身；两seed四臂400行、counter-reset20行、换条件1行、六个refusal/partial-state诊断及两FFT轴控制。不是SAT/einops或完整CogVideoX/sampler/视频质量、Apple FFT/GPU性能复现。

- [scripts/cogvideox/fastercache_sample_cogvideox.py](https://github.com/Vchitect/FasterCache/blob/02d05ef7edb48bdec01d2b4df6edfe7f472a41b9/scripts/cogvideox/fastercache_sample_cogvideox.py)
- [configs/cogvideox/fastercache_sample.yaml](https://github.com/Vchitect/FasterCache/blob/02d05ef7edb48bdec01d2b4df6edfe7f472a41b9/configs/cogvideox/fastercache_sample.yaml)
- [fastercache/models/cogvideox/diffusion_video.py](https://github.com/Vchitect/FasterCache/blob/02d05ef7edb48bdec01d2b4df6edfe7f472a41b9/fastercache/models/cogvideox/diffusion_video.py)
- [fastercache/models/cogvideox/dit_video_concat.py](https://github.com/Vchitect/FasterCache/blob/02d05ef7edb48bdec01d2b4df6edfe7f472a41b9/fastercache/models/cogvideox/dit_video_concat.py)
- [fastercache/models/cogvideox/sgm/modules/diffusionmodules/guiders.py](https://github.com/Vchitect/FasterCache/blob/02d05ef7edb48bdec01d2b4df6edfe7f472a41b9/fastercache/models/cogvideox/sgm/modules/diffusionmodules/guiders.py)
- [fastercache/models/cogvideox/sgm/modules/diffusionmodules/sampling_utils.py](https://github.com/Vchitect/FasterCache/blob/02d05ef7edb48bdec01d2b4df6edfe7f472a41b9/fastercache/models/cogvideox/sgm/modules/diffusionmodules/sampling_utils.py)
- [fastercache/models/cogvideox/sgm/modules/diffusionmodules/sampling.py](https://github.com/Vchitect/FasterCache/blob/02d05ef7edb48bdec01d2b4df6edfe7f472a41b9/fastercache/models/cogvideox/sgm/modules/diffusionmodules/sampling.py)
- [fastercache/models/cogvideox/sgm/modules/diffusionmodules/denoiser.py](https://github.com/Vchitect/FasterCache/blob/02d05ef7edb48bdec01d2b4df6edfe7f472a41b9/fastercache/models/cogvideox/sgm/modules/diffusionmodules/denoiser.py)
- [fastercache/models/cogvideox/sgm/modules/diffusionmodules/wrappers.py](https://github.com/Vchitect/FasterCache/blob/02d05ef7edb48bdec01d2b4df6edfe7f472a41b9/fastercache/models/cogvideox/sgm/modules/diffusionmodules/wrappers.py)
- [fastercache/models/cogvideox/sgm/modules/diffusionmodules/discretizer.py](https://github.com/Vchitect/FasterCache/blob/02d05ef7edb48bdec01d2b4df6edfe7f472a41b9/fastercache/models/cogvideox/sgm/modules/diffusionmodules/discretizer.py)

<a id="s47"></a>
## S47 — MLX v0.32.0 spatial FFT, shifts and real-cache ownership

仓库：[ml-explore/mlx](https://github.com/ml-explore/mlx)；commit：`7a1d4f5c12ac82f4b4d0a6e71538d89ca0605247`。

阅读：S；FFT前端complex64/float32与norm/axis、Python FFT2默认axes、fftshift/ifftshift全轴roll；Metal supported radices/Stockham与nd_fft_op/FFT::eval_gpu选段；Real primitive、unary output allocation及同itemsize donation守卫。非完整shader/编译器或物理dispatch审计。

历史观察范围：已安装MLX0.32.0公共GPU API执行四个benign ready-tensor50-call cache周期cohort、六compute/七timing臂×14formal trials和独立offset精度诊断；保留两trial pilot但排除formal。原创FasterCache数学移植/空间residual变体，full NumPy FP64与小shape原Torch CPU参考；不是本tag重编译、loaded-image attestation、真实模型/视频、物理shader计数或每方法peak。

- [mlx/fft.cpp](https://github.com/ml-explore/mlx/blob/7a1d4f5c12ac82f4b4d0a6e71538d89ca0605247/mlx/fft.cpp)
- [mlx/backend/metal/fft.cpp](https://github.com/ml-explore/mlx/blob/7a1d4f5c12ac82f4b4d0a6e71538d89ca0605247/mlx/backend/metal/fft.cpp)
- [python/src/fft.cpp](https://github.com/ml-explore/mlx/blob/7a1d4f5c12ac82f4b4d0a6e71538d89ca0605247/python/src/fft.cpp)
- [mlx/ops.cpp](https://github.com/ml-explore/mlx/blob/7a1d4f5c12ac82f4b4d0a6e71538d89ca0605247/mlx/ops.cpp)
- [mlx/backend/metal/unary.cpp](https://github.com/ml-explore/mlx/blob/7a1d4f5c12ac82f4b4d0a6e71538d89ca0605247/mlx/backend/metal/unary.cpp)
- [mlx/backend/common/unary.h](https://github.com/ml-explore/mlx/blob/7a1d4f5c12ac82f4b4d0a6e71538d89ca0605247/mlx/backend/common/unary.h)
- [mlx/backend/common/utils.h](https://github.com/ml-explore/mlx/blob/7a1d4f5c12ac82f4b4d0a6e71538d89ca0605247/mlx/backend/common/utils.h)
- 补充已读原文：[补充资料](https://ml-explore.github.io/mlx/build/html/python/fft.html)（不在上述文件哈希范围内）
- 补充已读原文：[补充资料](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.real.html)（不在上述文件哈希范围内）

<a id="s48"></a>
## S48 — ToMeSD actual CPU matching, generator/CFG and patch consumer

仓库：[dbolya/tomesd](https://github.com/dbolya/tomesd)；commit：`27a14a372beecaa85c101fb588631643f379b8ce`。

阅读：S；原merge/split/unmerge完整接缝、partial edge/r clamp、无epsilon cosine、MPS rank-gather；compute_merge奇数batch随机守卫、CPU generator fallback、pre-attention metric跨stage复用及Diffusers wrapper/default MLP-off；README支持/方法/配方选段与MIT license。

历史观察范围：Torch2.14.0 CPU受限执行原函数和明确toy stages，无上游初始化/生产模型patch；七toy路由、五wrapper控制、两geometry拒绝、两CPU gather helper及MPS metadata-only generator；12个既有真实Qwen BF16 FFN输入的36个源CPU地图/均值，不是MPS/full ToMeSD媒体或GPU性能复现。

- [README.md](https://github.com/dbolya/tomesd/blob/27a14a372beecaa85c101fb588631643f379b8ce/README.md)
- [LICENSE](https://github.com/dbolya/tomesd/blob/27a14a372beecaa85c101fb588631643f379b8ce/LICENSE)
- [tomesd/merge.py](https://github.com/dbolya/tomesd/blob/27a14a372beecaa85c101fb588631643f379b8ce/tomesd/merge.py)
- [tomesd/patch.py](https://github.com/dbolya/tomesd/blob/27a14a372beecaa85c101fb588631643f379b8ce/tomesd/patch.py)
- [tomesd/utils.py](https://github.com/dbolya/tomesd/blob/27a14a372beecaa85c101fb588631643f379b8ce/tomesd/utils.py)

<a id="s49"></a>
## S49 — MLX0.32 array.at duplicate-index add and Metal scatter reduction

仓库：[ml-explore/mlx](https://github.com/ml-explore/mlx)；commit：`7a1d4f5c12ac82f4b4d0a6e71538d89ca0605247`。

阅读：S；ArrayAt::add→mlx_add_item→scatter_add/Scatter::Sum前端、shape/index检查、Metal Scatter::eval_gpu的copy/update/reduction kernel选择与scatter_impl atomic_update选段；不是全indexing/atomic/ISA/编译器审计。

历史观察范围：实际MLX0.32.0 GPU公共API编译FP32归约、BF16 centroid/FFN与restore，固定CPU partition、两层/两prompt/三step的12点、八计算/九timing臂×18trial；全输入mean/shape/routing控制及仅两row全宽FP64 FFN参考。原创FFN-local高风险迁移，不是tag重编译、shader观测或原ToMeSD attention/default/full Qwen图像资格。

- [python/src/array.cpp](https://github.com/ml-explore/mlx/blob/7a1d4f5c12ac82f4b4d0a6e71538d89ca0605247/python/src/array.cpp)
- [python/src/indexing.cpp](https://github.com/ml-explore/mlx/blob/7a1d4f5c12ac82f4b4d0a6e71538d89ca0605247/python/src/indexing.cpp)
- [mlx/ops.cpp](https://github.com/ml-explore/mlx/blob/7a1d4f5c12ac82f4b4d0a6e71538d89ca0605247/mlx/ops.cpp)
- [mlx/backend/metal/indexing.cpp](https://github.com/ml-explore/mlx/blob/7a1d4f5c12ac82f4b4d0a6e71538d89ca0605247/mlx/backend/metal/indexing.cpp)
- [mlx/backend/metal/kernels/indexing/scatter.h](https://github.com/ml-explore/mlx/blob/7a1d4f5c12ac82f4b4d0a6e71538d89ca0605247/mlx/backend/metal/kernels/indexing/scatter.h)
- 补充已读原文：[补充资料](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.at.html)（不在上述文件哈希范围内）

<a id="s50"></a>
## S50 — MagCache author FLUX/Qwen/Wan calibration and residual consumers

仓库：[Zehong-Ma/MagCache](https://github.com/Zehong-Ma/MagCache)；commit：`df81cb181776c2c61477c08e1d21f87fda1cd938`。

阅读：S；根README方法/新模型接入/许可、三adapter README用法与作者A800测量选段；FLUX/Qwen/Wan标准forward/calibration、ratio插值/prefix、parity分支、预算/阈值/映射11守卫、生命周期与类状态；不包括Wan VACE/其它模型完整调用链或原论文逐式审计。

历史观察范围：Torch2.14 CPU选定原函数，fresh tiny affine/identity/zero-time替身；实际Qwen initializer仅改自有toy class，Wan显式no-op AMP与in-memory JSON sink；校准→曲线→推理、旋转向量反例、阈值/二次请求/失败/cold-hit与同类实例stress。不是原模型/CFG runtime顺序、CUDA混合精度、性能或媒体复现。

- [README.md](https://github.com/Zehong-Ma/MagCache/blob/df81cb181776c2c61477c08e1d21f87fda1cd938/README.md)
- [LICENSE](https://github.com/Zehong-Ma/MagCache/blob/df81cb181776c2c61477c08e1d21f87fda1cd938/LICENSE)
- [MagCache4FLUX/README.md](https://github.com/Zehong-Ma/MagCache/blob/df81cb181776c2c61477c08e1d21f87fda1cd938/MagCache4FLUX/README.md)
- [MagCache4FLUX/magcache_flux.py](https://github.com/Zehong-Ma/MagCache/blob/df81cb181776c2c61477c08e1d21f87fda1cd938/MagCache4FLUX/magcache_flux.py)
- [MagCache4QwenImage/README.md](https://github.com/Zehong-Ma/MagCache/blob/df81cb181776c2c61477c08e1d21f87fda1cd938/MagCache4QwenImage/README.md)
- [MagCache4QwenImage/magcache_generate.py](https://github.com/Zehong-Ma/MagCache/blob/df81cb181776c2c61477c08e1d21f87fda1cd938/MagCache4QwenImage/magcache_generate.py)
- [MagCache4Wan2.1/README.md](https://github.com/Zehong-Ma/MagCache/blob/df81cb181776c2c61477c08e1d21f87fda1cd938/MagCache4Wan2.1/README.md)
- [MagCache4Wan2.1/magcache_generate.py](https://github.com/Zehong-Ma/MagCache/blob/df81cb181776c2c61477c08e1d21f87fda1cd938/MagCache4Wan2.1/magcache_generate.py)

<a id="s51"></a>
## S51 — Diffusers MagCache actual HookRegistry and pipeline lifecycle

仓库：[huggingface/diffusers](https://github.com/huggingface/diffusers)；commit：`d42ebf9c6e0b46103b74283f5324357a67acaa04`。

阅读：S；MagCache config/head/tail/apply完整接缝；真实HookRegistry嵌套new_forward、state/context/reset与identifier tuple；CacheMixin启停/上下文/重置；Qwen cond/uncond consumer和成功结束cleanup；PipelineUtils.maybe_free_model_hooks重置优先于offload早退。其余pipeline/框架文件仅定位，不作全文件审计。

历史观察范围：实际registry+MagCache+CacheMixin选定声明CPU执行，移除相对import并注入依赖及明确toy metadata registry；单/双/三block、named/shared contexts、clock/threshold、shape/FP16减法/失败诊断和原cleanup方法。3969组原NumPy/Torch interpolation对照；不执行production架构registry或完整Qwen pipeline，不授予模型/Apple GPU/媒体资格。

- [src/diffusers/hooks/mag_cache.py](https://github.com/huggingface/diffusers/blob/d42ebf9c6e0b46103b74283f5324357a67acaa04/src/diffusers/hooks/mag_cache.py)
- [src/diffusers/hooks/hooks.py](https://github.com/huggingface/diffusers/blob/d42ebf9c6e0b46103b74283f5324357a67acaa04/src/diffusers/hooks/hooks.py)
- [src/diffusers/hooks/_common.py](https://github.com/huggingface/diffusers/blob/d42ebf9c6e0b46103b74283f5324357a67acaa04/src/diffusers/hooks/_common.py)
- [src/diffusers/models/cache_utils.py](https://github.com/huggingface/diffusers/blob/d42ebf9c6e0b46103b74283f5324357a67acaa04/src/diffusers/models/cache_utils.py)
- [src/diffusers/pipelines/qwenimage/pipeline_qwenimage.py](https://github.com/huggingface/diffusers/blob/d42ebf9c6e0b46103b74283f5324357a67acaa04/src/diffusers/pipelines/qwenimage/pipeline_qwenimage.py)
- [src/diffusers/pipelines/pipeline_utils.py](https://github.com/huggingface/diffusers/blob/d42ebf9c6e0b46103b74283f5324357a67acaa04/src/diffusers/pipelines/pipeline_utils.py)

<a id="s52"></a>
## S52 — AdaCache original adaptive-period/MoReg STDiT consumer and RFlow CFG sampler

仓库：[AdaCache-DiT/AdaCache](https://github.com/AdaCache-DiT/AdaCache)；commit：`2c6d3ca0320afd1f3ac41ac52673c79c8f6e00a4`。

阅读：S；README方法/运行与三配方、脚本CLI覆盖、推理model复用/循环与sampler调用；STDiT3Block的metric/MoReg/forward/reset及root全局rate发布；RFlow sample的cond-first packed CFG/Euler、timestep_transform、t2i_modulate与scheduler初始化/noise选段。非全部OpenSora/FlashAttention/训练/论文逐式审计。

历史观察范围：Torch2.14 CPU受限选取原sampler/root/block/helper，原构造器之外均明确tiny affine/identity/zero-position、Torch reshape facade和direct-checkpoint依赖替代；14对blocks/8个100-step合成请求贯通全局layer13、CFG与Euler；19原metric控制、current-mask/gate hit与注入prefix失败。完整FP32 scalar重放，不是实际STDiT权重/模型/视频、SP/CUDA/MPS/Apple性能复现。

- [README.md](https://github.com/AdaCache-DiT/AdaCache/blob/2c6d3ca0320afd1f3ac41ac52673c79c8f6e00a4/README.md)
- [LICENSE](https://github.com/AdaCache-DiT/AdaCache/blob/2c6d3ca0320afd1f3ac41ac52673c79c8f6e00a4/LICENSE)
- [configs/sample.py](https://github.com/AdaCache-DiT/AdaCache/blob/2c6d3ca0320afd1f3ac41ac52673c79c8f6e00a4/configs/sample.py)
- [configs/sample_adacache.py](https://github.com/AdaCache-DiT/AdaCache/blob/2c6d3ca0320afd1f3ac41ac52673c79c8f6e00a4/configs/sample_adacache.py)
- [configs/sample_adacache_moreg.py](https://github.com/AdaCache-DiT/AdaCache/blob/2c6d3ca0320afd1f3ac41ac52673c79c8f6e00a4/configs/sample_adacache_moreg.py)
- [run_sample_video.sh](https://github.com/AdaCache-DiT/AdaCache/blob/2c6d3ca0320afd1f3ac41ac52673c79c8f6e00a4/run_sample_video.sh)
- [inference.py](https://github.com/AdaCache-DiT/AdaCache/blob/2c6d3ca0320afd1f3ac41ac52673c79c8f6e00a4/inference.py)
- [opensora_base/opensora/models/stdit/stdit3.py](https://github.com/AdaCache-DiT/AdaCache/blob/2c6d3ca0320afd1f3ac41ac52673c79c8f6e00a4/opensora_base/opensora/models/stdit/stdit3.py)
- [opensora_base/opensora/schedulers/rf/__init__.py](https://github.com/AdaCache-DiT/AdaCache/blob/2c6d3ca0320afd1f3ac41ac52673c79c8f6e00a4/opensora_base/opensora/schedulers/rf/__init__.py)
- [opensora_base/opensora/models/layers/blocks.py](https://github.com/AdaCache-DiT/AdaCache/blob/2c6d3ca0320afd1f3ac41ac52673c79c8f6e00a4/opensora_base/opensora/models/layers/blocks.py)
- [opensora_base/opensora/schedulers/rf/rectified_flow.py](https://github.com/AdaCache-DiT/AdaCache/blob/2c6d3ca0320afd1f3ac41ac52673c79c8f6e00a4/opensora_base/opensora/schedulers/rf/rectified_flow.py)

<a id="s53"></a>
## S53 — Draw Things original Swift TeaCache and H3/legacy consumer wrapper

仓库：[drawthingsai/draw-things-community](https://github.com/drawthingsai/draw-things-community)；commit：`8d9c62ad38539ef8bb123f8b8e78cab76caafd0c`。

阅读：S；完整TeaCache配置/gate/compile/cache/replay；UNetProtocol的ModelBuilderOrModel及H3 first-block→audio/video signal→tail/head与marker消费者、Wan/FLUX调用选段；测试的anchor/async/head输入/参数共享选段；固定Package依赖、AGENTS全文及GPL3许可头。不作完整UNet/应用/所有测试或许可法律分析。

- [LICENSE.md](https://github.com/drawthingsai/draw-things-community/blob/8d9c62ad38539ef8bb123f8b8e78cab76caafd0c/LICENSE.md)
- [Package.swift](https://github.com/drawthingsai/draw-things-community/blob/8d9c62ad38539ef8bb123f8b8e78cab76caafd0c/Package.swift)
- [AGENTS.md](https://github.com/drawthingsai/draw-things-community/blob/8d9c62ad38539ef8bb123f8b8e78cab76caafd0c/AGENTS.md)
- [Libraries/SwiftDiffusion/Sources/TeaCache/TeaCache.swift](https://github.com/drawthingsai/draw-things-community/blob/8d9c62ad38539ef8bb123f8b8e78cab76caafd0c/Libraries/SwiftDiffusion/Sources/TeaCache/TeaCache.swift)
- [Libraries/SwiftDiffusion/Sources/Models/UNetProtocol.swift](https://github.com/drawthingsai/draw-things-community/blob/8d9c62ad38539ef8bb123f8b8e78cab76caafd0c/Libraries/SwiftDiffusion/Sources/Models/UNetProtocol.swift)
- [Libraries/SwiftDiffusion/Tests/TeaCacheTests.swift](https://github.com/drawthingsai/draw-things-community/blob/8d9c62ad38539ef8bb123f8b8e78cab76caafd0c/Libraries/SwiftDiffusion/Tests/TeaCacheTests.swift)

<a id="s54"></a>
## S54 — s4nnc pinned native tensor readback, Model and named parameter sharing

仓库：[liuliu/s4nnc](https://github.com/liuliu/s4nnc)；commit：`f64e4a6ed2d656b5d169d39fc1316c8df793d09f`。

阅读：S；BSD3与Package NNC依赖/源列表；DynamicGraph init/rawValue/joined、Tensor shape与toCPU/toGPU、Model Parameters/Input、参数copy/share和Dense/ModelIO运算符接缝。其余大文件未全式审计；不逐个审计所有编译单元/传递包。

历史观察范围：实际固定NNC公共API CPU与MPS GPU执行FP32 abs/mean/subtract、model/head/Dense和share，不用接口桩；Float16 cache泛型与FP32 tensor边界分开。SwiftPM本地同ref路径override有identity warning，原Package及跟踪工作树未改；不是完整依赖闭包内容证明、训练或物理共享峰值证据。

- [LICENSE](https://github.com/liuliu/s4nnc/blob/f64e4a6ed2d656b5d169d39fc1316c8df793d09f/LICENSE)
- [Package.swift](https://github.com/liuliu/s4nnc/blob/f64e4a6ed2d656b5d169d39fc1316c8df793d09f/Package.swift)
- [nnc/DynamicGraph.swift](https://github.com/liuliu/s4nnc/blob/f64e4a6ed2d656b5d169d39fc1316c8df793d09f/nnc/DynamicGraph.swift)
- [nnc/Tensor.swift](https://github.com/liuliu/s4nnc/blob/f64e4a6ed2d656b5d169d39fc1316c8df793d09f/nnc/Tensor.swift)
- [nnc/Model.swift](https://github.com/liuliu/s4nnc/blob/f64e4a6ed2d656b5d169d39fc1316c8df793d09f/nnc/Model.swift)
- [nnc/ModelIOAddons.swift](https://github.com/liuliu/s4nnc/blob/f64e4a6ed2d656b5d169d39fc1316c8df793d09f/nnc/ModelIOAddons.swift)
- [nnc/ModelAddons.swift](https://github.com/liuliu/s4nnc/blob/f64e4a6ed2d656b5d169d39fc1316c8df793d09f/nnc/ModelAddons.swift)
- [nnc/Operators.swift](https://github.com/liuliu/s4nnc/blob/f64e4a6ed2d656b5d169d39fc1316c8df793d09f/nnc/Operators.swift)

<a id="s55"></a>
## S55 — ccv pinned SwiftPM model constructor and CPU/MPS reduction/transfer

仓库：[liuliu/ccv](https://github.com/liuliu/ccv)；commit：`66dc16a3a3b688adfa6dc1d19eeeb3071beeb701`。

阅读：S；代码/文档/第三方许可区别、SwiftPM目标/框架与HAVE_MPS/Accelerate flags；Model新图topological-sort/input-bitmask断言；CPU FP32 mean loop与MPSGraph mean前向、GPU→CPU blit/fast-fence分支选段。不作全core/utility/MFA shader审计或具体硬件ISA归因。

历史观察范围：实际SwiftPM release构建的ccv由NNC调用CPU及GPU API，原构造断言保留、客户端分离Input后完成；三样本Device0准入、显式stream行为与全部合成head值独立检查，无权重、速度、连续独占、物理dispatch/peak或全模型资格。

- [COPYING](https://github.com/liuliu/ccv/blob/66dc16a3a3b688adfa6dc1d19eeeb3071beeb701/COPYING)
- [Package.swift](https://github.com/liuliu/ccv/blob/66dc16a3a3b688adfa6dc1d19eeeb3071beeb701/Package.swift)
- [lib/nnc/ccv_cnnp_model_core.c](https://github.com/liuliu/ccv/blob/66dc16a3a3b688adfa6dc1d19eeeb3071beeb701/lib/nnc/ccv_cnnp_model_core.c)
- [lib/nnc/cmd/reduce/ccv_nnc_reduce_mean_cpu_ref.c](https://github.com/liuliu/ccv/blob/66dc16a3a3b688adfa6dc1d19eeeb3071beeb701/lib/nnc/cmd/reduce/ccv_nnc_reduce_mean_cpu_ref.c)
- [lib/nnc/cmd/reduce/mps/ccv_nnc_reduce_mean_mps.m](https://github.com/liuliu/ccv/blob/66dc16a3a3b688adfa6dc1d19eeeb3071beeb701/lib/nnc/cmd/reduce/mps/ccv_nnc_reduce_mean_mps.m)
- [lib/nnc/cmd/util/mps/ccv_nnc_util_mps.m](https://github.com/liuliu/ccv/blob/66dc16a3a3b688adfa6dc1d19eeeb3071beeb701/lib/nnc/cmd/util/mps/ccv_nnc_util_mps.m)

<a id="s56"></a>
## S56 — s4nnc real gate reductions, fixed-arity scalar packing and shape representation

仓库：[liuliu/s4nnc](https://github.com/liuliu/s4nnc)；commit：`f64e4a6ed2d656b5d169d39fc1316c8df793d09f`。

阅读：S；Identity、ReduceMean、Concat及Functional.concat新建对象与固定Model根；Functional.abs/reduced/full的命令接缝、TensorShape完整12槽相等与数组投影、withUnsafeBytes、raw/直接variable构造与withNoGrad/withStream选段。不审计全文件或全部编译闭包。

历史观察范围：固定原TeaCache(S53)和真实NNC/MPS API的ready-FP32 gate窗口，四计算臂及host floor、三shape×1/3query、20平衡trial；原source/批量读回/缓存分母分离，完整signed bank SHA与标量统计参考；CPU、排除pilot、变更anchor/代际与原生shape合同诊断分开。不代表完整模型/媒体/原cache replay或物理GPU计数。

- [nnc/ModelAddons.swift](https://github.com/liuliu/s4nnc/blob/f64e4a6ed2d656b5d169d39fc1316c8df793d09f/nnc/ModelAddons.swift)
- [nnc/FunctionalAddons.swift](https://github.com/liuliu/s4nnc/blob/f64e4a6ed2d656b5d169d39fc1316c8df793d09f/nnc/FunctionalAddons.swift)
- [nnc/Tensor.swift](https://github.com/liuliu/s4nnc/blob/f64e4a6ed2d656b5d169d39fc1316c8df793d09f/nnc/Tensor.swift)
- [nnc/DynamicGraph.swift](https://github.com/liuliu/s4nnc/blob/f64e4a6ed2d656b5d169d39fc1316c8df793d09f/nnc/DynamicGraph.swift)

<a id="s57"></a>
## S57 — ccv concat alias-format-transform, model absorb and CPU image-bridge metadata

仓库：[liuliu/ccv](https://github.com/liuliu/ccv)；commit：`66dc16a3a3b688adfa6dc1d19eeeb3071beeb701`。

阅读：S；concat构造/alias输出与FORMAT_TRANSFORM graph node、model_absorb input_size断言及compile重建；dynamic_graph_evaluate完整params memcmp触发compile/absorb、reduce tensor_auto复制输入参数后仅改reduce轴；ccv_nnc_tensor_new的CPU NHWC三维dense-matrix桥接把row step写入dim[4]、终止0之后的metadata选段。不将concat count当shader数，也未instrument正式timing的重编译次数；非全文件/allocator/ISA审计。

历史观察范围：真实固定ccv依赖执行CPU/MPS统计与固定2/4-input packing Model；保留bare variadic Concat重用abort、不同构造器可见shape相同而原生shape不等的guard拒绝，以及同构造/不可变bank的正式timings。原库未改、断言未关闭，不授予所有版本bug、物理backing/峰值/连续独占或完整H3推理速度资格。

- [lib/nnc/ccv_cnnp_model_addons.c](https://github.com/liuliu/ccv/blob/66dc16a3a3b688adfa6dc1d19eeeb3071beeb701/lib/nnc/ccv_cnnp_model_addons.c)
- [lib/nnc/ccv_cnnp_model.c](https://github.com/liuliu/ccv/blob/66dc16a3a3b688adfa6dc1d19eeeb3071beeb701/lib/nnc/ccv_cnnp_model.c)
- [lib/nnc/ccv_nnc_tensor.c](https://github.com/liuliu/ccv/blob/66dc16a3a3b688adfa6dc1d19eeeb3071beeb701/lib/nnc/ccv_nnc_tensor.c)
- [lib/nnc/ccv_nnc_dynamic_graph_evaluate.c](https://github.com/liuliu/ccv/blob/66dc16a3a3b688adfa6dc1d19eeeb3071beeb701/lib/nnc/ccv_nnc_dynamic_graph_evaluate.c)
- [lib/nnc/cmd/reduce/ccv_nnc_reduce.c](https://github.com/liuliu/ccv/blob/66dc16a3a3b688adfa6dc1d19eeeb3071beeb701/lib/nnc/cmd/reduce/ccv_nnc_reduce.c)

<a id="s58"></a>
## S58 — ccv read-only model-input signature boundary and compile/absorb predicate

仓库：[liuliu/ccv](https://github.com/liuliu/ccv)；commit：`66dc16a3a3b688adfa6dc1d19eeeb3071beeb701`。

阅读：S；私有model struct的graph/inputs/input_size布局；16个int参数字段与12维ABI；公开variable/symbol params声明及只读getter；dynamic_graph_evaluate的cold/full-params memcmp分支和compile中copy/absorb选段。仅这些接缝，不是完整内部ABI、编译闭包或所有后端审计。

历史观察范围：原创只读C适配器链接固定原ccv，真实CPU/GPU packing前后各读完整参数字段；共享角色、分离角色、four-stats及跨几何正控各有独立chronological replay。未改上游、未装compiler hook、不导出地址/opaque padding；这是分支谓词及保存签名转移，不是直接compile函数次数、timing、稳定私有ABI、模型或媒体资格。

- [lib/nnc/_ccv_cnnp_model.h](https://github.com/liuliu/ccv/blob/66dc16a3a3b688adfa6dc1d19eeeb3071beeb701/lib/nnc/_ccv_cnnp_model.h)
- [lib/nnc/ccv_nnc_tfb.h](https://github.com/liuliu/ccv/blob/66dc16a3a3b688adfa6dc1d19eeeb3071beeb701/lib/nnc/ccv_nnc_tfb.h)
- [lib/nnc/ccv_nnc.h](https://github.com/liuliu/ccv/blob/66dc16a3a3b688adfa6dc1d19eeeb3071beeb701/lib/nnc/ccv_nnc.h)
- [lib/nnc/ccv_nnc_dynamic_graph.c](https://github.com/liuliu/ccv/blob/66dc16a3a3b688adfa6dc1d19eeeb3071beeb701/lib/nnc/ccv_nnc_dynamic_graph.c)
- [lib/nnc/ccv_nnc_symbolic_graph.c](https://github.com/liuliu/ccv/blob/66dc16a3a3b688adfa6dc1d19eeeb3071beeb701/lib/nnc/ccv_nnc_symbolic_graph.c)
- [lib/nnc/ccv_nnc_dynamic_graph_evaluate.c](https://github.com/liuliu/ccv/blob/66dc16a3a3b688adfa6dc1d19eeeb3071beeb701/lib/nnc/ccv_nnc_dynamic_graph_evaluate.c)
- [lib/nnc/ccv_cnnp_model.c](https://github.com/liuliu/ccv/blob/66dc16a3a3b688adfa6dc1d19eeeb3071beeb701/lib/nnc/ccv_cnnp_model.c)

<a id="s59"></a>
## S59 — s4nnc tensor reflection and real model-evaluate dispatch

仓库：[liuliu/s4nnc](https://github.com/liuliu/s4nnc)；commit：`f64e4a6ed2d656b5d169d39fc1316c8df793d09f`。

阅读：S；AnyTensor.underlying/_AnyTensor._tensor及公开graph/shape getter、Model.cModel与所有权选段、Functional Model.callAsFunction到DynamicGraph.Tensor.evaluate的真实ccv_nnc_dynamic_graph_evaluate调用。非全文件/反射API稳定性或完整训练审计。

历史观察范围：真实固定Swift/NNC CPU和MPS执行norm/difference及2/4-input packing，私有tensor句柄只在客户端反射后传给只读适配器，不序列化地址。每设备201次packing、510个完整标量input描述和独立数值参考；GPU初始及三组setup均三样本Device≤5%，不证明连续独占、性能或模型/媒体质量。

- [nnc/DynamicGraph.swift](https://github.com/liuliu/s4nnc/blob/f64e4a6ed2d656b5d169d39fc1316c8df793d09f/nnc/DynamicGraph.swift)
- [nnc/Model.swift](https://github.com/liuliu/s4nnc/blob/f64e4a6ed2d656b5d169d39fc1316c8df793d09f/nnc/Model.swift)
- [nnc/Functional.swift](https://github.com/liuliu/s4nnc/blob/f64e4a6ed2d656b5d169d39fc1316c8df793d09f/nnc/Functional.swift)

<a id="s60"></a>
## S60 — HiDiffusion RAU-Net schedules,MSW-MSA windows and patch lifecycle

仓库：[megvii-research/HiDiffusion](https://github.com/megvii-research/HiDiffusion)；commit：`e9ec39a58805c3c44c650e2e745c0aa600824f0e`。

阅读：S；Apache2许可头、README支持/配方/示例及setup/依赖声明；原key/ratio/aggressive常量、transformer window/roll/odd bicubic、cross-down/up pool/restore、conv stride/dilation与legacy LoRA返回、hook/apply/remove完整选段；utils类名MRO与RNG fallback。模块key文本仅核路径样式/首尾/行数及严格set识别消费者，非逐行架构审计；大段ControlNet pipeline/完整UNet仅查看接口及resize/upsampler接缝，不审计或执行完整body。补充选读论文2311.17528v2的3.2/G.4/G.5/H/I及作者项目页；网页正文未纳入文件hash，也非全篇审计。

历史观察范围：固定AST原方法在Torch CPU执行7个window对照、2个无效metadata/mask诊断、178个conv调用、74对down/up微型chain、legacy/异常状态及6个patch观察；真实CPU均匀SDPA但无原learned QKV/norm/resnet，明确affine/identity替身与独立全数组参考。小tensor和root-size路由metadata分开、scaffold并非Diffusers真实pipeline；未改上游，不授予GPU/MPS性能、完整模型/ControlNet/训练/媒体、所有版本或可直接集成资格。

- [LICENSE](https://github.com/megvii-research/HiDiffusion/blob/e9ec39a58805c3c44c650e2e745c0aa600824f0e/LICENSE)
- [README.md](https://github.com/megvii-research/HiDiffusion/blob/e9ec39a58805c3c44c650e2e745c0aa600824f0e/README.md)
- [hidiffusion/__init__.py](https://github.com/megvii-research/HiDiffusion/blob/e9ec39a58805c3c44c650e2e745c0aa600824f0e/hidiffusion/__init__.py)
- [hidiffusion/hidiffusion.py](https://github.com/megvii-research/HiDiffusion/blob/e9ec39a58805c3c44c650e2e745c0aa600824f0e/hidiffusion/hidiffusion.py)
- [hidiffusion/sd_module_key/sd15_module_key.txt](https://github.com/megvii-research/HiDiffusion/blob/e9ec39a58805c3c44c650e2e745c0aa600824f0e/hidiffusion/sd_module_key/sd15_module_key.txt)
- [hidiffusion/sd_module_key/sdxl_module_key.txt](https://github.com/megvii-research/HiDiffusion/blob/e9ec39a58805c3c44c650e2e745c0aa600824f0e/hidiffusion/sd_module_key/sdxl_module_key.txt)
- [hidiffusion/utils.py](https://github.com/megvii-research/HiDiffusion/blob/e9ec39a58805c3c44c650e2e745c0aa600824f0e/hidiffusion/utils.py)
- [requirements.txt](https://github.com/megvii-research/HiDiffusion/blob/e9ec39a58805c3c44c650e2e745c0aa600824f0e/requirements.txt)
- [setup.py](https://github.com/megvii-research/HiDiffusion/blob/e9ec39a58805c3c44c650e2e745c0aa600824f0e/setup.py)
- 补充已读原文：[补充资料](https://arxiv.org/html/2311.17528v2)（不在上述文件哈希范围内）
- 补充已读原文：[补充资料](https://hidiffusion.github.io/)（不在上述文件哈希范围内）

<a id="s61"></a>
## S61 — Diffusers0.39.0 complete Tiny U-Net consumer,real projections and DDIM feedback

仓库：[huggingface/diffusers](https://github.com/huggingface/diffusers)；commit：`a3608b512ed7248499a44c61d954965ed9bdae4d`。

阅读：S；AGENTS/CLAUDE指向的.ai规则及Apache2许可头；__init__版本/lazy导出、ModelMixin构造、Config/Pipeline注册与name_or_path；U-Net构造参数/下采样skip stack/显式upsample_size消费者、CrossAttnDown/Up的resnet/attention循环；BasicTransformerBlock兼容norm flags、原AttnProcessor2_0的QKV/SDPA/输出投影、Resnet2D与Transformer2D flatten/restore、down/up插值接缝；DDIM32-step固定beta初始化、10-step leading时表与eta0 epsilon step选段。13个运行源码完整字节与release匹配不等于全文件语义/完整依赖或native二进制审计。

历史观察范围：完整未改HiDiffusion包(S60)导入，真实Diffusers0.39.0 Tiny四/三stage U-Net随机初始化，无identity/affine替身；四shape/topology×五臂20轨迹的19条完成，1条组合odd网格RuntimeError保留，191预测/CFG-DDIM更新独立条件数值重放和完整回调shape程序重放。20次unpatch同输入原U-Net读回、9次中断重用/重apply/fresh控制分开。官方route名仅选择adapter，不是预训练SD/SDXL、完整pipeline/ControlNet/媒体、独立U-Net math、GPU或性能资格；原库/包未改。

- [.ai/AGENTS.md](https://github.com/huggingface/diffusers/blob/a3608b512ed7248499a44c61d954965ed9bdae4d/.ai/AGENTS.md)
- [AGENTS.md](https://github.com/huggingface/diffusers/blob/a3608b512ed7248499a44c61d954965ed9bdae4d/AGENTS.md)
- [CLAUDE.md](https://github.com/huggingface/diffusers/blob/a3608b512ed7248499a44c61d954965ed9bdae4d/CLAUDE.md)
- [LICENSE](https://github.com/huggingface/diffusers/blob/a3608b512ed7248499a44c61d954965ed9bdae4d/LICENSE)
- [src/diffusers/__init__.py](https://github.com/huggingface/diffusers/blob/a3608b512ed7248499a44c61d954965ed9bdae4d/src/diffusers/__init__.py)
- [src/diffusers/configuration_utils.py](https://github.com/huggingface/diffusers/blob/a3608b512ed7248499a44c61d954965ed9bdae4d/src/diffusers/configuration_utils.py)
- [src/diffusers/models/attention.py](https://github.com/huggingface/diffusers/blob/a3608b512ed7248499a44c61d954965ed9bdae4d/src/diffusers/models/attention.py)
- [src/diffusers/models/attention_processor.py](https://github.com/huggingface/diffusers/blob/a3608b512ed7248499a44c61d954965ed9bdae4d/src/diffusers/models/attention_processor.py)
- [src/diffusers/models/downsampling.py](https://github.com/huggingface/diffusers/blob/a3608b512ed7248499a44c61d954965ed9bdae4d/src/diffusers/models/downsampling.py)
- [src/diffusers/models/modeling_utils.py](https://github.com/huggingface/diffusers/blob/a3608b512ed7248499a44c61d954965ed9bdae4d/src/diffusers/models/modeling_utils.py)
- [src/diffusers/models/resnet.py](https://github.com/huggingface/diffusers/blob/a3608b512ed7248499a44c61d954965ed9bdae4d/src/diffusers/models/resnet.py)
- [src/diffusers/models/transformers/transformer_2d.py](https://github.com/huggingface/diffusers/blob/a3608b512ed7248499a44c61d954965ed9bdae4d/src/diffusers/models/transformers/transformer_2d.py)
- [src/diffusers/models/unets/unet_2d_blocks.py](https://github.com/huggingface/diffusers/blob/a3608b512ed7248499a44c61d954965ed9bdae4d/src/diffusers/models/unets/unet_2d_blocks.py)
- [src/diffusers/models/unets/unet_2d_condition.py](https://github.com/huggingface/diffusers/blob/a3608b512ed7248499a44c61d954965ed9bdae4d/src/diffusers/models/unets/unet_2d_condition.py)
- [src/diffusers/models/upsampling.py](https://github.com/huggingface/diffusers/blob/a3608b512ed7248499a44c61d954965ed9bdae4d/src/diffusers/models/upsampling.py)
- [src/diffusers/pipelines/pipeline_utils.py](https://github.com/huggingface/diffusers/blob/a3608b512ed7248499a44c61d954965ed9bdae4d/src/diffusers/pipelines/pipeline_utils.py)
- [src/diffusers/schedulers/scheduling_ddim.py](https://github.com/huggingface/diffusers/blob/a3608b512ed7248499a44c61d954965ed9bdae4d/src/diffusers/schedulers/scheduling_ddim.py)

<a id="s62"></a>
## S62 — T-GATE original SD/SDXL/SVD phase policies and attention processor

仓库：[HaozheLiu-ST/T-GATE](https://github.com/HaozheLiu-ST/T-GATE)；commit：`ab2a88fa50319cbb0bfcdc2772405573b266fa13`。

阅读：S；MIT全文、README简介/版本/要求/结果表及SD配置；tgate_utils完整register/scheduler/processor、SD原latent pipeline函数、SDXL后段negative条件/pooled/time IDs与MPS dtype guard选段、SVD条件准备及后段选择。__init__及PixArt模块仅import接口，DeepCache两loader的函数内可选import/enable边界；未完整审计PixArt/DeepCache/SVD forward或论文。补充arXiv2404.02747v3摘要与版本身份，非全篇/作者媒体验收复现。

历史观察范围：完整原包导入，真实Diffusers0.39.0随机Tiny U-Net+原StableDiffusionPipeline/SD.tgate的预计算embedding/latent-only16请求：15条10步完成，no-CFG边界3步后拒绝。加入CFG-only null/官方positive cutoff和cross-only强分母消融；全投影调用/cache manifest及两个选定完整cache接缝、条件CFG/DDIM逐元素重放；qk_norm无cache差异和4D hit拒绝分开。未运行text/VAE、预训练模型、PixArt/DeepCache/SVD模型或GPU/媒体/性能；原库未改。

- [LICENSE](https://github.com/HaozheLiu-ST/T-GATE/blob/ab2a88fa50319cbb0bfcdc2772405573b266fa13/LICENSE)
- [README.md](https://github.com/HaozheLiu-ST/T-GATE/blob/ab2a88fa50319cbb0bfcdc2772405573b266fa13/README.md)
- [tgate/PixArt_Alpha.py](https://github.com/HaozheLiu-ST/T-GATE/blob/ab2a88fa50319cbb0bfcdc2772405573b266fa13/tgate/PixArt_Alpha.py)
- [tgate/PixArt_Sigma.py](https://github.com/HaozheLiu-ST/T-GATE/blob/ab2a88fa50319cbb0bfcdc2772405573b266fa13/tgate/PixArt_Sigma.py)
- [tgate/SD.py](https://github.com/HaozheLiu-ST/T-GATE/blob/ab2a88fa50319cbb0bfcdc2772405573b266fa13/tgate/SD.py)
- [tgate/SDXL.py](https://github.com/HaozheLiu-ST/T-GATE/blob/ab2a88fa50319cbb0bfcdc2772405573b266fa13/tgate/SDXL.py)
- [tgate/SDXL_DeepCache.py](https://github.com/HaozheLiu-ST/T-GATE/blob/ab2a88fa50319cbb0bfcdc2772405573b266fa13/tgate/SDXL_DeepCache.py)
- [tgate/SD_DeepCache.py](https://github.com/HaozheLiu-ST/T-GATE/blob/ab2a88fa50319cbb0bfcdc2772405573b266fa13/tgate/SD_DeepCache.py)
- [tgate/SVD.py](https://github.com/HaozheLiu-ST/T-GATE/blob/ab2a88fa50319cbb0bfcdc2772405573b266fa13/tgate/SVD.py)
- [tgate/__init__.py](https://github.com/HaozheLiu-ST/T-GATE/blob/ab2a88fa50319cbb0bfcdc2772405573b266fa13/tgate/__init__.py)
- [tgate/tgate_utils.py](https://github.com/HaozheLiu-ST/T-GATE/blob/ab2a88fa50319cbb0bfcdc2772405573b266fa13/tgate/tgate_utils.py)
- 补充已读原文：[补充资料](https://arxiv.org/abs/2404.02747v3)（不在上述文件哈希范围内）

<a id="s63"></a>
## S63 — ComfyUI_TGate sigma proxy,semantic labels and cache frontier

仓库：[JettHu/ComfyUI_TGate](https://github.com/JettHu/ComfyUI_TGate)；commit：`cc1437bf44f609f6a7251b65132119c31c7f3ffd`。

阅读：S；GPL3许可头、README节点/历史MPS指针/旧后端数字及__init__；TGate.py完整forward-cache/clear、sigma proxy、custom CFG、deprecated pre-CFG及apply节点注册。已读取作者2024-04-25所链MPS comment2077823182，环境M1 Pro/32GB/Sonoma14.4.1(23E224)/Torch2.2.2；仅历史异常/后端假设报告，不据其推断当前MPS可用性；非所有Comfy版本/内核审计。

历史观察范围：原TGate.py模块加载，直接执行原TGateProxy和TGateSamplerCFG配明确CPU affine apply_model，并调用S65原cfg_function；16组正反/单路batch与sigma上下/相等/零点独立scalar重放。仅所列helpers，不运行make_tgate_forward真实attention、节点图/ModelPatcher、Comfy应用、GPU/cpu-cache传输或媒体性能；界限/位置反例不升级为所有图像错误。

- [LICENSE](https://github.com/JettHu/ComfyUI_TGate/blob/cc1437bf44f609f6a7251b65132119c31c7f3ffd/LICENSE)
- [README.md](https://github.com/JettHu/ComfyUI_TGate/blob/cc1437bf44f609f6a7251b65132119c31c7f3ffd/README.md)
- [TGate.py](https://github.com/JettHu/ComfyUI_TGate/blob/cc1437bf44f609f6a7251b65132119c31c7f3ffd/TGate.py)
- [__init__.py](https://github.com/JettHu/ComfyUI_TGate/blob/cc1437bf44f609f6a7251b65132119c31c7f3ffd/__init__.py)
- 补充已读原文：[补充资料](https://github.com/JettHu/ComfyUI_TGate/issues/4#issuecomment-2077823182)（不在上述文件哈希范围内）

<a id="s64"></a>
## S64 — Diffusers0.39.0 latent SD pipeline and true SVD embedding branch identity

仓库：[huggingface/diffusers](https://github.com/huggingface/diffusers)；commit：`a3608b512ed7248499a44c61d954965ed9bdae4d`。

阅读：S；SD pipeline构造的steps_offset修正、预计算prompt编码/dtype、CFG batch与prepare_latents、callback/latent postprocess；PipelineCallback/SDCFGCutoffCallback的guidance与conditional切换接口、VaeImageProcessor latent直接返回；SVD _encode_image/_encode_vae_image的False返回conditional、True拼零分支选段。不完整审计其它图像处理/视频pipeline或新UI。

历史观察范围：所选4个已安装文件与固定tag匹配；原SD latent-only请求执行source/RNG/shape/条件反馈，VAE/CLIP为支持的None路径并提供完整embedding/latents，不是替代encoder网络。SVD helper语义仅静态阅读，不运行视频模型；与S61的13文件合计17匹配，不授予全包/compiled native闭包、MPS或媒体资格。

- [src/diffusers/callbacks.py](https://github.com/huggingface/diffusers/blob/a3608b512ed7248499a44c61d954965ed9bdae4d/src/diffusers/callbacks.py)
- [src/diffusers/image_processor.py](https://github.com/huggingface/diffusers/blob/a3608b512ed7248499a44c61d954965ed9bdae4d/src/diffusers/image_processor.py)
- [src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py](https://github.com/huggingface/diffusers/blob/a3608b512ed7248499a44c61d954965ed9bdae4d/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py)
- [src/diffusers/pipelines/stable_video_diffusion/pipeline_stable_video_diffusion.py](https://github.com/huggingface/diffusers/blob/a3608b512ed7248499a44c61d954965ed9bdae4d/src/diffusers/pipelines/stable_video_diffusion/pipeline_stable_video_diffusion.py)

<a id="s65"></a>
## S65 — ComfyUI pinned positional conditioning batches and CFG callback consumer

仓库：[Comfy-Org/ComfyUI](https://github.com/Comfy-Org/ComfyUI)；commit：`b0b743566f65daafc423b4fea8a2fbda94b3384a`。

阅读：S；AGENTS约束与GPL3许可头；samplers._calc_cond_batch按条件索引建队列、to_batch_temp.reverse与按位置cat/scatter，以及sampling_function/cfg_function的cond=0/uncond=1、cfg1优化和custom CFG返回坐标选段。不是完整Comfy调度/插件兼容审计。

历史观察范围：仅抽取原cfg_function声明，用真实Torch CPU tensor和S63原custom CFG执行16组affine callback控制；未导入Comfy core初始化或执行完整_calc_cond_batch/实际模型。反序batch合法性来自所读构造/回填接缝，非全部sampler/节点/app/GPU/媒体复现。

- [AGENTS.md](https://github.com/Comfy-Org/ComfyUI/blob/b0b743566f65daafc423b4fea8a2fbda94b3384a/AGENTS.md)
- [LICENSE](https://github.com/Comfy-Org/ComfyUI/blob/b0b743566f65daafc423b4fea8a2fbda94b3384a/LICENSE)
- [comfy/samplers.py](https://github.com/Comfy-Org/ComfyUI/blob/b0b743566f65daafc423b4fea8a2fbda94b3384a/comfy/samplers.py)

<a id="s66"></a>
## S66 — PyTorch2.14.0 pinned MPS synchronize,fallback and SDPA eligibility

仓库：[pytorch/pytorch](https://github.com/pytorch/pytorch)；commit：`08187d9e0fba026dc8217405802ab5381dc88d90`。

阅读：S；BSD类许可及归属头；torch.mps公开synchronize/device API选段、MPSStream endKernelCoalescing/COMMIT_AND_WAIT/全Torch stream同步；MPSFallback全局环境控制及embedding_renorm_显式CPU例外；Attention.mm的head-dim/short-Q/vector/prefill/MPP dtype/mask/stride选定条件。不审计全allocator/shader/框架/native二进制，未instrument实际kernel或ISA。官方2.14 synchronize/环境文档另为D级导航，不在Git文件hash内。

历史观察范围：运行库声明Torch2.14.0/git08187d9且安装MPS Python wrapper与固定blob相等，区别旁路2.14.1；真实MPS resident随机U-Net四cohort五臂40平衡trial/200正式完整latent窗口、20cold40warm排除，260GPU输出与20同配方CPU reference逐元素固定门槛核查。准入、完整journal、参数/input bytes与host post-run supplement分开。环境fallback0不消除所有显式CPU注册或CPU metadata；不授予所有op纯GPU、物理branch/峰值/ISA、预训练媒体或默认资格。

- [LICENSE](https://github.com/pytorch/pytorch/blob/08187d9e0fba026dc8217405802ab5381dc88d90/LICENSE)
- [aten/src/ATen/mps/MPSFallback.mm](https://github.com/pytorch/pytorch/blob/08187d9e0fba026dc8217405802ab5381dc88d90/aten/src/ATen/mps/MPSFallback.mm)
- [aten/src/ATen/mps/MPSStream.mm](https://github.com/pytorch/pytorch/blob/08187d9e0fba026dc8217405802ab5381dc88d90/aten/src/ATen/mps/MPSStream.mm)
- [aten/src/ATen/native/mps/operations/Attention.mm](https://github.com/pytorch/pytorch/blob/08187d9e0fba026dc8217405802ab5381dc88d90/aten/src/ATen/native/mps/operations/Attention.mm)
- [torch/mps/__init__.py](https://github.com/pytorch/pytorch/blob/08187d9e0fba026dc8217405802ab5381dc88d90/torch/mps/__init__.py)
- 补充已读原文：[补充资料](https://docs.pytorch.org/docs/2.14/generated/torch.mps.synchronize.html)（不在上述文件哈希范围内）
- 补充已读原文：[补充资料](https://docs.pytorch.org/docs/2.14/mps_environment_variables.html)（不在上述文件哈希范围内）

<a id="s67"></a>
## S67 — VDE original FLUX/Qwen velocity decomposition and fixed-snapshot availability

仓库：[Tan-Junwen/VDE](https://github.com/Tan-Junwen/VDE)；commit：`3d08e45de4378ba0e4848c0dfa6c7216b4311f2c`。

阅读：S；Apache2许可及README/模型README；两份Python完整读取：全tensor norm/dot、epsilon分解、两锚点真实t外推、fixed interval与末步强制、局部history、CLI类patch/device_map、Qwen分路估计后per-token CFG rescale和MPS dtype补偿。固定递归tree仅两份Python，Wan/Z README待发布、Comfy目录无节点实现；不把README支持勾选/外部倍率当可执行发布或Apple结论。另选读正确Tan等arXiv2605.23381v1方法/算法并核对CVPR目录身份，不与其它题名/作者或arXiv身份混用，非全文逐式审计；CVF单篇页面本轮抓取失败，目录元数据不能代替其全文。

历史观察范围：两份完整未改模块导入；真实随机Diffusers FLUX latent-only CPU九条10步请求完成，69次完整网络返回/21估计步，完整Euler反馈与所有估计逐元素独立重放；interval1等原pipeline、正常warm重用等fresh，batch单独/合并反例。原Qwen0.39调用在txt_seq_lens接口拒绝，未静默适配、未计算Qwen网络/CFG。direct原helper另测epsilon/零/tiny、有限FP16 norm溢出、空/单/重复/非均匀锚点；无预训练/文本/VAE/媒体、GPU、计时或独立transformer数学资格。

- [LICENSE](https://github.com/Tan-Junwen/VDE/blob/3d08e45de4378ba0e4848c0dfa6c7216b4311f2c/LICENSE)
- [README.md](https://github.com/Tan-Junwen/VDE/blob/3d08e45de4378ba0e4848c0dfa6c7216b4311f2c/README.md)
- [VDE4FLUX/README.md](https://github.com/Tan-Junwen/VDE/blob/3d08e45de4378ba0e4848c0dfa6c7216b4311f2c/VDE4FLUX/README.md)
- [VDE4FLUX/inference_flux1.py](https://github.com/Tan-Junwen/VDE/blob/3d08e45de4378ba0e4848c0dfa6c7216b4311f2c/VDE4FLUX/inference_flux1.py)
- [VDE4Qwen-Image/README.md](https://github.com/Tan-Junwen/VDE/blob/3d08e45de4378ba0e4848c0dfa6c7216b4311f2c/VDE4Qwen-Image/README.md)
- [VDE4Qwen-Image/inference_qwenimage.py](https://github.com/Tan-Junwen/VDE/blob/3d08e45de4378ba0e4848c0dfa6c7216b4311f2c/VDE4Qwen-Image/inference_qwenimage.py)
- [VDE4Wan2.1/README.md](https://github.com/Tan-Junwen/VDE/blob/3d08e45de4378ba0e4848c0dfa6c7216b4311f2c/VDE4Wan2.1/README.md)
- [VDE4Z-image/README.md](https://github.com/Tan-Junwen/VDE/blob/3d08e45de4378ba0e4848c0dfa6c7216b4311f2c/VDE4Z-image/README.md)
- 补充已读原文：[补充资料](https://arxiv.org/html/2605.23381v1)（不在上述文件哈希范围内）
- 补充已读原文：[补充资料](https://openaccess.thecvf.com/CVPR2026?day=2026-06-07)（不在上述文件哈希范围内）

<a id="s68"></a>
## S68 — Diffusers0.39 FLUX/Qwen consumer and FlowMatch Euler contract

仓库：[huggingface/diffusers](https://github.com/huggingface/diffusers)；commit：`a3608b512ed7248499a44c61d954965ed9bdae4d`。

阅读：S；各文件Apache2归属头；FLUX/Qwen constructor/forward接口、prompt预计算embedding路径、packed latent/shape、stock loop与latent输出；CacheMixin.cache_context无finally的上下文生命周期；FlowMatch配置、动态shift/set_timesteps/index与确定性Euler step选段。原Qwen代码txt_seq_lens与0.39实际forward签名不匹配。未审计完整依赖/所有attention算式/compiled binary或其它版本兼容性；AGENTS沿用S61已读约束，本轮复读。

历史观察范围：六个已安装Python文件字节等固定tag；真实小型随机FLUX双流/单流transformer与原FluxPipeline、原FlowMatch scheduler CPU执行，无替身。Qwen实际类构造和原调用pre-hook到达，但绑定参数即拒绝，未执行Qwen forward计算；无成功适配/完整视频/媒体/GPU或全包native构建身份资格。

- [src/diffusers/models/cache_utils.py](https://github.com/huggingface/diffusers/blob/a3608b512ed7248499a44c61d954965ed9bdae4d/src/diffusers/models/cache_utils.py)
- [src/diffusers/models/transformers/transformer_flux.py](https://github.com/huggingface/diffusers/blob/a3608b512ed7248499a44c61d954965ed9bdae4d/src/diffusers/models/transformers/transformer_flux.py)
- [src/diffusers/models/transformers/transformer_qwenimage.py](https://github.com/huggingface/diffusers/blob/a3608b512ed7248499a44c61d954965ed9bdae4d/src/diffusers/models/transformers/transformer_qwenimage.py)
- [src/diffusers/pipelines/flux/pipeline_flux.py](https://github.com/huggingface/diffusers/blob/a3608b512ed7248499a44c61d954965ed9bdae4d/src/diffusers/pipelines/flux/pipeline_flux.py)
- [src/diffusers/pipelines/qwenimage/pipeline_qwenimage.py](https://github.com/huggingface/diffusers/blob/a3608b512ed7248499a44c61d954965ed9bdae4d/src/diffusers/pipelines/qwenimage/pipeline_qwenimage.py)
- [src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py](https://github.com/huggingface/diffusers/blob/a3608b512ed7248499a44c61d954965ed9bdae4d/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py)

<a id="s69"></a>
## S69 — PyTorch2.14 MPS scalar blocking-copy,norm eligibility and default-device observer re-entry

仓库：[pytorch/pytorch](https://github.com/pytorch/pytorch)；commit：`08187d9e0fba026dc8217405802ab5381dc88d90`。

阅读：S；BSD类及Apple归属头；Scalar.mm完整_local_scalar_dense_mps、Copy.mm的MPS→CPU/同dtype blit→copy_and_sync(false)、MPSStream copy_and_sync的COMMIT_AND_WAIT及公开synchronize；ReduceOps.mm完整norm_kernel_mps的scalar/inner条件及sum dispatcher选段，不审计shader/全文件其它reduction。_device.py完整DeviceContext mode堆栈及__torch_function__重派发，不能把Python hook重入当多次物理copy。沿S67/S68未改VDE/Diffusers阅读，本条不覆盖所有PyTorch版本/native binary/ISA或逐op实际分支。

历史观察范围：固定Torch2.14.0声明08187d9，MPS Python wrapper与DeviceContext各精确匹配安装字节；六个Diffusers选定文件沿S68匹配。实际MPS共同50-step去噪循环四cohort/40平衡trial200正式窗，60cold/warm排除，20diagnostic+8原pipeline alignment另计，全部288GPU输出同配方CPU门槛通过。首次完整run的去guard变体多算一次weight，整体排除并保留原client/receipt；修正为共享单weight后按原计划全量重跑，不混用计时。原VDE helper与去guard变体端点byte等值；观察器原VDE每请求68entry并非68读回，独立CPU34逻辑请求在默认设备模式观察68层级entry，unset/reset各34，显式restore。CPU+MPS原/stock pipeline各八次端点对齐共同loop，不把公共pipeline额外边界放入性能窗口；无真实权重/文本/VAE/媒体、profile/物理分支/continuous exclusivity或默认资格。

- [LICENSE](https://github.com/pytorch/pytorch/blob/08187d9e0fba026dc8217405802ab5381dc88d90/LICENSE)
- [aten/src/ATen/mps/MPSStream.mm](https://github.com/pytorch/pytorch/blob/08187d9e0fba026dc8217405802ab5381dc88d90/aten/src/ATen/mps/MPSStream.mm)
- [aten/src/ATen/native/mps/operations/Copy.mm](https://github.com/pytorch/pytorch/blob/08187d9e0fba026dc8217405802ab5381dc88d90/aten/src/ATen/native/mps/operations/Copy.mm)
- [aten/src/ATen/native/mps/operations/ReduceOps.mm](https://github.com/pytorch/pytorch/blob/08187d9e0fba026dc8217405802ab5381dc88d90/aten/src/ATen/native/mps/operations/ReduceOps.mm)
- [aten/src/ATen/native/mps/operations/Scalar.mm](https://github.com/pytorch/pytorch/blob/08187d9e0fba026dc8217405802ab5381dc88d90/aten/src/ATen/native/mps/operations/Scalar.mm)
- [torch/mps/__init__.py](https://github.com/pytorch/pytorch/blob/08187d9e0fba026dc8217405802ab5381dc88d90/torch/mps/__init__.py)
- [torch/utils/_device.py](https://github.com/pytorch/pytorch/blob/08187d9e0fba026dc8217405802ab5381dc88d90/torch/utils/_device.py)
- 补充已读原文：[补充资料](https://docs.pytorch.org/docs/2.14/generated/torch.mps.synchronize.html)（不在上述文件哈希范围内）
- 补充已读原文：[补充资料](https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.item.html)（不在上述文件哈希范围内）

<a id="s70"></a>
## S70 — DeepCache original U-Net extension,branch boundaries and lifecycle

仓库：[horseee/DeepCache](https://github.com/horseee/DeepCache)；commit：`1691a3ab22308fe06685e4ff151a9ab175be2dc4`。

阅读：S；Apache2/XinyinMa许可、两README、两个init及extension helper完整；stable_diffusion.py完整的CUDA/0.24定制实验入口但不执行。helper divmod3编码、down>=/up>非对称边界、父block先挡住子模块、raw tuple/Tensor缓存、timestep.item+list.index首值匹配、enable/disable/reset与normal重复请求。另选读arXiv2312.00858v2第3节/模型评估语境，非全文逐式审计或论文CUDA倍率复现。沿S24扩展，不回写其历史只读范围。

历史观察范围：完整原DeepCache包导入/未改helper，真实Diffusers0.39随机207404参数U-Net/latent pipeline13条10步完成、2条0步拒绝；130网络返回/99840预测scalar，49920条件CFG-DDIM反馈scalar和36530个leaf决策独立重放，25376 Conv2d/Linear body调用。正常warm重复和disable恢复等值；直接kwargs/double-enable/reset-enabled拒绝，重复时间list fixture4个完整网络返回，非PNDM solver。另以同一原helper及明确受限ordinal-root变体做真实镜像SD1.5 MPS完整生成，数据范围/预算/排除由独立image capsule限定，非所有SDXL/SVD/Img2Img/视频、物理kernel或媒体/默认资格。

- [DeepCache/README.md](https://github.com/horseee/DeepCache/blob/1691a3ab22308fe06685e4ff151a9ab175be2dc4/DeepCache/README.md)
- [DeepCache/__init__.py](https://github.com/horseee/DeepCache/blob/1691a3ab22308fe06685e4ff151a9ab175be2dc4/DeepCache/__init__.py)
- [DeepCache/extension/__init__.py](https://github.com/horseee/DeepCache/blob/1691a3ab22308fe06685e4ff151a9ab175be2dc4/DeepCache/extension/__init__.py)
- [DeepCache/extension/deepcache.py](https://github.com/horseee/DeepCache/blob/1691a3ab22308fe06685e4ff151a9ab175be2dc4/DeepCache/extension/deepcache.py)
- [LICENSE](https://github.com/horseee/DeepCache/blob/1691a3ab22308fe06685e4ff151a9ab175be2dc4/LICENSE)
- [README.md](https://github.com/horseee/DeepCache/blob/1691a3ab22308fe06685e4ff151a9ab175be2dc4/README.md)
- [stable_diffusion.py](https://github.com/horseee/DeepCache/blob/1691a3ab22308fe06685e4ff151a9ab175be2dc4/stable_diffusion.py)
- 补充已读原文：[补充资料](https://arxiv.org/html/2312.00858v2)（不在上述文件哈希范围内）
- 补充已读原文：[补充资料](https://huggingface.co/docs/diffusers/main/en/optimization/deepcache)（不在上述文件哈希范围内）

<a id="s71"></a>
## S71 — MLX DeepCache public port:whole-block boundary,CFG order and static interface limits

仓库：[Barry0121/mlx_deepcache](https://github.com/Barry0121/mlx_deepcache)；commit：`6b3564b95b41050a543597d477efad0da020e948`。

阅读：S；两README及__init__/unet/sampler/config/txt2image完整，model_io头/模型表/UNet与CLIP权重映射选段。whole-down/up stage cache位置、NHWC、positive-first CFG与Euler、CLI mx.eval、cache_interval membership及I2I positional routing；非均匀Python负底数分数幂与type=bool CLI边界只做静态/原创stdlib域示例。固定递归tree未找到LICENSE，仅若干Apple copyright头，不把所称上游来源自动当作所有新增改动的复用授权。

历史观察范围：仅阅读公开源码，不复制其函数body到Wiki、不执行该port/其发布基准或加载其模型；许可与当前MLX API兼容性待确认。作者README链接它不等于本轮M1/MLX速度、bitwise等价、SDXL Turbo/I2I可用性或媒体资格。与Torch branch2/MLX layer0的拓扑对应限四stage/2down3up结构，不是跨layout/权重/采样器的数值或二进制cache兼容证明。

- [README.md](https://github.com/Barry0121/mlx_deepcache/blob/6b3564b95b41050a543597d477efad0da020e948/README.md)
- [mlx_deepcache/README.md](https://github.com/Barry0121/mlx_deepcache/blob/6b3564b95b41050a543597d477efad0da020e948/mlx_deepcache/README.md)
- [mlx_deepcache/stable_diffusion/__init__.py](https://github.com/Barry0121/mlx_deepcache/blob/6b3564b95b41050a543597d477efad0da020e948/mlx_deepcache/stable_diffusion/__init__.py)
- [mlx_deepcache/stable_diffusion/config.py](https://github.com/Barry0121/mlx_deepcache/blob/6b3564b95b41050a543597d477efad0da020e948/mlx_deepcache/stable_diffusion/config.py)
- [mlx_deepcache/stable_diffusion/model_io.py](https://github.com/Barry0121/mlx_deepcache/blob/6b3564b95b41050a543597d477efad0da020e948/mlx_deepcache/stable_diffusion/model_io.py)
- [mlx_deepcache/stable_diffusion/sampler.py](https://github.com/Barry0121/mlx_deepcache/blob/6b3564b95b41050a543597d477efad0da020e948/mlx_deepcache/stable_diffusion/sampler.py)
- [mlx_deepcache/stable_diffusion/unet.py](https://github.com/Barry0121/mlx_deepcache/blob/6b3564b95b41050a543597d477efad0da020e948/mlx_deepcache/stable_diffusion/unet.py)
- [mlx_deepcache/txt2image.py](https://github.com/Barry0121/mlx_deepcache/blob/6b3564b95b41050a543597d477efad0da020e948/mlx_deepcache/txt2image.py)

<a id="s72"></a>
## S72 — Diffusers0.39 learned SD pipeline,VAE decoder and DDIM/PNDM clock

仓库：[huggingface/diffusers](https://github.com/huggingface/diffusers)；commit：`a3608b512ed7248499a44c61d954965ed9bdae4d`。

阅读：S；各文件Apache2头；沿S61/S64的SD construct/encode_prompt、CFG/prepare_latents、完整原loop/公共callback、VAE decode到postprocess/PIL路径；新增AutoencoderKL constructor/_decode/decode和Decoder constructor/forward正常非tile/非slice路径选段。DDIM set_timesteps/step及PNDM set_timesteps的skip_prk重复值选段，不将value-first index叫调用ordinal。非VAE encoder/全部tiling/PNDM solver逐式审计；AGENTS沿既有完整阅读，未改上游。

历史观察范围：六个新增选定文件及沿S61/S64合计20个安装Diffusers Python文件等固定tag；真实本地SD1.5镜像fp16 safetensors转FP32的原公共pipeline，含实际CLIP/50DDIM步/VAE/PIL/PNG，非预计算embedding/随机模型替身。模型/decoder/NN数值不由源码身份独立证明；common软件census/finite检查与helper setup/cleanup计入窗口，加载/准入/初始latent准备在外。PNDM仅静态阅读，MLX port不执行，无全wheel/native闭包、纯GPU或媒体/默认资格。

- [src/diffusers/image_processor.py](https://github.com/huggingface/diffusers/blob/a3608b512ed7248499a44c61d954965ed9bdae4d/src/diffusers/image_processor.py)
- [src/diffusers/models/autoencoders/autoencoder_kl.py](https://github.com/huggingface/diffusers/blob/a3608b512ed7248499a44c61d954965ed9bdae4d/src/diffusers/models/autoencoders/autoencoder_kl.py)
- [src/diffusers/models/autoencoders/vae.py](https://github.com/huggingface/diffusers/blob/a3608b512ed7248499a44c61d954965ed9bdae4d/src/diffusers/models/autoencoders/vae.py)
- [src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py](https://github.com/huggingface/diffusers/blob/a3608b512ed7248499a44c61d954965ed9bdae4d/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py)
- [src/diffusers/schedulers/scheduling_ddim.py](https://github.com/huggingface/diffusers/blob/a3608b512ed7248499a44c61d954965ed9bdae4d/src/diffusers/schedulers/scheduling_ddim.py)
- [src/diffusers/schedulers/scheduling_pndm.py](https://github.com/huggingface/diffusers/blob/a3608b512ed7248499a44c61d954965ed9bdae4d/src/diffusers/schedulers/scheduling_pndm.py)

<a id="s73"></a>
## S73 — Transformers4.57.6 CLIP text/config/tokenizer pinned learned-conditioning consumer

仓库：[huggingface/transformers](https://github.com/huggingface/transformers)；commit：`753d61104116eefc8ffc977327b441ee0c8d599f`。

阅读：S；完整AGENTS约束、Apache2归属/许可；CLIP配置接口/默认字段、TextEmbeddings/TextTransformer/TextModel与causal mask/last-hidden/legacy EOS pool选段；slow tokenizer constructor/vocab-merges/特殊token与byte BPE tokenize选段。不是Vision/完整CLIP语义或全tokenizer库审计。v4.57.6 annotated tag解析到该commit，所选实际安装的3个CLIP Python文件逐字节匹配。

历史观察范围：原CLIPTextModel/CLIPTokenizer从固定本地safetensors/config/vocab/merges完整加载并在MPS生成中实际执行正/负两次text forward；加载info无missing/mismatch/error，legacy非persistent position buffer与旧文件77项完全相等，所有浮点计算FP32。不是预计算embedding替身、CLIP视觉/感知评分、原始全FP32checkpoint、完整Transformers/native闭包或模型/媒体/生产资格。

- [AGENTS.md](https://github.com/huggingface/transformers/blob/753d61104116eefc8ffc977327b441ee0c8d599f/AGENTS.md)
- [LICENSE](https://github.com/huggingface/transformers/blob/753d61104116eefc8ffc977327b441ee0c8d599f/LICENSE)
- [src/transformers/models/clip/configuration_clip.py](https://github.com/huggingface/transformers/blob/753d61104116eefc8ffc977327b441ee0c8d599f/src/transformers/models/clip/configuration_clip.py)
- [src/transformers/models/clip/modeling_clip.py](https://github.com/huggingface/transformers/blob/753d61104116eefc8ffc977327b441ee0c8d599f/src/transformers/models/clip/modeling_clip.py)
- [src/transformers/models/clip/tokenization_clip.py](https://github.com/huggingface/transformers/blob/753d61104116eefc8ffc977327b441ee0c8d599f/src/transformers/models/clip/tokenization_clip.py)

## 重放来源取证

及是候选采集范围，不是已审计/支持列表；是本页采用的具体阅读文件及固定commit。

源码下载需显式 `--download`，输出放到新的本地忽略目录；不安装依赖、不运行项目脚本、不更新既有 HEAD。partial clone 的缺失 blob 仅在显式下载模式获取。来源收集没有 GPU 工作。

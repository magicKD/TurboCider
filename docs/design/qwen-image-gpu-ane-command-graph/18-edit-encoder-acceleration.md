# 18 · Qwen Image 2.1 Edit：1–3 张参考图 encoder 加速

更新：2026-10-08。实现分为现有 MLX encoder 的 GPU 融合路径与未来 Native Metal Graph encoder。前者本轮接入，后者没有实现。本轮使用本地模型，限定生成数据、真实 1–3 图 conditioning 和一组完整请求配对；不下载、转换或复制 checkpoint，不导出 Core ML 模型。

后续ANE runtime桥与最新Splash参考见[19](19-edit-encoder-ane-runtime.md)：单层有收益，但完整三图encoder变慢且未通过5%conditioning筛选，默认仍GPU。本文件的测量/“未接ANE”描述属于此前GPU融合轮次，不转授后续构建。

## 1. 与 Splash 的关系

不是完全相同的 encoder，但有直接可复用的结构和执行方法。固定 [Splash #260 的 Vision.hpp](https://github.com/incoai/splash/blob/80827e1c3593f7d10c266394e4d4381321b73ad0/runtime/ops/Vision.hpp) 指定 Qwen3.5 视觉塔：27 层、1152 hidden、16 heads、head dimension 72、patch16、merge2、position grid48。`VisionLayout` 的 intermediate 为 4304，使用 4352 的 kernel padding；它不能代替本地 checkpoint 的张量形状。[Vision.cpp](https://github.com/incoai/splash/blob/80827e1c3593f7d10c266394e4d4381321b73ad0/runtime/ops/Vision.cpp) 的 Q/K scratch 宽度为 80，V/attention 保留 72，query/key tiles 为 64/128；[vision.metal](https://github.com/incoai/splash/blob/80827e1c3593f7d10c266394e4d4381321b73ad0/runtime/metal/kernels/shared/vision.metal) 将 softmax running state 保留在寄存器，避免整个 NxN attention matrix。

本地 Qwen3-VL 8B 的视觉几何高度接近，区别仍需独立验证：参考图处理、视觉 GELU／归一化／projection 舍入、三份 DeepStack merger（视觉 8/16/24 层输出进入语言 0/1/2 层）、language 输出宽度和 pre-final-norm hidden 接口。语言部分是一次完整多模态 causal prefill，36 层、32 Q heads、8 KV heads、D128，mRoPE 三轴 interleave；不需要 Splash 的逐 token decode、语言 logits、paged decode KV 或采样器。参考图 VAE 是另一个 convolutional encoder，不能拿 LLM FFN 或视觉塔 attention 加速方案替代。

因此优先迁移“保持数学的融合注意力和布局”，再考虑 arena／PSO／command graph；直接调用现有 DiT native attention／FFN 会违反 geometry 与 FP32/BF16 边界。

## 2. 本轮实现

入口：[encoder_attention.hpp](../../../native/models/qwen21/encoder_attention.hpp)、[text_encoder.cpp](../../../native/models/qwen21/text_encoder.cpp)、[vision.cpp](../../../native/models/qwen21/vision.cpp)、[pipeline.cpp](../../../native/models/qwen21/pipeline.cpp)。

### 2.1 语言 prefill

- 序列至少 128 行时直接传入原始 8 组 K/V，使用 MLX 原生 GQA；去掉每层 K/V 4 倍 repeat。小行数沿用已有 common Qwen3 的 expanded-KV 策略。
- 无 padding 时使用 SDPA 的 `causal`，不创建 count×count FP32 additive mask。有 padding 时保留原 causal + `key >= valid` additive mask；padding 不参与 attention。
- 保留 Q/K norm FP32 accumulation、mRoPE、attention FP32 Q/K/V、输出 BF16 边界、残差／SiLU FFN／DeepStack 顺序、层级 eval 和 source release callback。
- 不截断 prompt 或参考图 token，不降低图像处理尺寸。

### 2.2 视觉 attention

本地 pinned MLX 0.32.0 的 [Metal SDPA](https://github.com/ml-explore/mlx/blob/v0.32.0/mlx/backend/metal/scaled_dot_product_attention.cpp) full fused 支持 D64/80/128，D72 会 fallback。对实际 Qwen21 的 FP32 Q/K/V，在 feature axis 补 8 个零，保持 scale **1/sqrt(72)**，调用 D80 fused SDPA，再切回 72、transpose/merge heads。MLX 接口要求 Q/K/V 宽度一致，所以本轮 V 也补到 80；这与 Splash 的 V72 原生 kernel 有执行差异，但零填充及切回保留数学。

视觉 BF16 fixture 的初始测试未通过 0.5% RRMS 筛选：旧 fallback 存在 BF16 中间舍入，不能认为数学等价即逐边界等价。因此最终实现只在 Q/K/V **全部 FP32**、D72、rows>8 时使用 padding fusion；其余 dtype／geometry 走原 `attend`。PE Qwen3.5 eager attention 完全保留，不能借本轮资格启用新路径。

### 2.3 接线与 cache

优化接入现有 Qwen21 Session 的视觉／语言 encoder，默认启用。诊断环境变量 `TURBOCIDER_QWEN21_ENCODER_REFERENCE=1` 保留旧路径，用于同构建的独立进程配对；只允许 0/1，首次读取后固定，不是 request/App 用户选项。这样一个进程不会因切换 recipe 误复用 conditioning。

已有 exact edit conditioning 与最多 3 张图的视觉／VAE 条件缓存继续使用；同图改变 prompt 可复用视觉／VAE，图像 hash 或 source generation 变化失效。缓存功能是既有实现，不列为本轮新收益。配对必须启动新进程，避免 conditioning cache 命中跳过 encoder。

当前 `encoder_ane_manifest` 对 Qwen21 的拒绝保留：本轮实际执行 GPU，既没有新增 ANE program，也没有 private API／W8A8 placement 证据。App 复用同一原生 pipeline，本轮 Swift App 编译通过；已安装 App／UI 尚未验收。

## 3. 验证方法和数值边界

[原生 probe](../../../tools/native/qwen21_edit_encoder_probe.mm) 与[构建脚本](../../../tools/native/build_qwen21_edit_encoder_probe.sh)链接显式指定的新库，核对 compiled runtime identity。CPU 模式验证计划／非法 counts，不运行设备或加载模型。

小型设备组：两层完整语言 blocks × 三种序列 × padded/unpadded，含 mRoPE／DeepStack／raw 或 final norm，RRMS≤2%；视觉 FP32/BF16 × 三种 rows，RRMS≤0.5%。BF16 应保留旧路径。shape/dtype／finite 独立检查，误差上限不能覆盖错误 mask、溢出、source drift 或未完成执行。

真实组件组：本地 teapot／fox／puffin，默认 1024 reference processor，每张图完整 27 层视觉塔旧/新各一次，顺序 AB/BA/AB，merged 与三个 DeepStack 各 RRMS≤1%。继而对 1／2／3 张图组合完整 36 层语言 conditioning 旧/新各一次，顺序 AB/BA/AB，最终 retained conditioning 使用用户允许的5%筛选，独立保留初始2%严格线的失败状态。组件显式 cache cap256 MiB，预物化视觉后释放，再物化语言 source；测量不含 load／resize／assembly，首个真实 shape compile 包含在 arm 时间内。图数间没有频繁循环或 warmup sweeps。

仅组件通过后，运行一组 512²、三参考图、Viggle r128 strength1 六步、seed42、reference_size512 的完整 MLX 请求配对，旧/新各一次。512 reference 是请求中既有显式近似，不能把它用于声称默认 1024 完整编辑质量已验收。关闭近似 DiT cache，其他 request/source/adapter 相同，比较 request、text/vision、VAE 和 denoise，检查生成图。一次配对不构成稳定性能或全机型资格。

所有有效结果需要实际 `mx::eval + mx::synchronize`、held-fd stat generation 复核与 reference hash 一致。监督器保存 source/build/binary 身份、time bound、failure 和系统观察；失败／超时不能靠进程退出解释为物理完成。真实模型不作全量复制或频繁全模型 hash。

真实1/2/3图计算省时18.70%/20.54%/21.77%，conditioning RRMS2.0023%/2.1506%/2.4564%；初始2%严格筛选未通过，不能改写成严格等价。完整512 reference三图请求总耗时32.058→31.669s，但text/vision phase为3.871→3.919s，未证明encoder净收益。默认1024完整编辑尚未测试，有限配对不获稳定性能资格。

本轮实际结果与 owned 清理记录见[状态](../../status/2026-10-08-qwen-edit-encoder.md)。

## 4. 后续 Native／GPU+ANE 边界

下一阶段不是简单把 `encoder_ane_manifest` 放开，而是独立建立 encoder op inventory 和预算：

1. 首先覆盖 FP32 visual norm/projection/GELU、D72 padding 与 mRoPE／DeepStack；同 input/source 的完整 conditioning 数学成立后再切 backend。
2. Native language attention 要支持 Q32/KV8 的 GQA 与 causal/key-valid masks；当前 DiT D128 相同不代表 K/V layout 相同。保留 VL norm FP32 semantics。
3. 采用 Splash arena 的 maximum patches、复用 PSO／scratch，以及最大图尺寸预准入。1–3 图顺序使用同一个视觉 workspace；输出 merger/DeepStack owner 保留到语言消费结束。
4. encoder 是一次 prefill：36 层 FFN 首次 weight staging／ANE program prepare 不一定能摊薄。真实 prefill 行数随文本和 reference processor 变化；先用实际输入测总 path（stage + sync + join），再选固定／可变 bucket 与 GPU/ANE 通道划分。
5. ANE 量化 recipe、校准与 content key 独立于 DiT；不能把 DiT W8A8／LoRA 或 base 性能证明转授 encoder。动态 1–3 图可先由纯 GPU fused prefill 执行，避免为短生命期 workload 生成大量低命中 program/cache。
6. VAE 先测其在整个请求的比例；本轮没有优化它，不将 visual transformer 的收益记成 VAE 提速。

只有完整请求中首次／重复编辑的净收益成立，再增加自动路由与 App 专门设置。不同 backend 的角色、来源、缓存、内存、资格仍由共同 model/runtime 层管理。

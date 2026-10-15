# 单图编辑：完整 Native Metal Qwen3-VL encoder

2026-10-08，接续 [21：调度与校准](21-encoder-scheduling-and-calibration.md)。现在已实现并执行完整的视觉27层和语言36层 Native Metal 模型计算，完成一次单图编辑整体验证。该路径独立于21的MLX/private-ANE FFN桥；本轮实际仅运行GPU。数据、构建和限制见[状态记录](../../status/2026-10-08-qwen-native-encoder.md)。

## 1. 从 Splash 迁移的执行组织

固定参考 Splash `0d5a035ea26d1cc566e3c7ee4e08e5e06fa68cf7` 的 [AneFfn.cpp](https://github.com/incoai/splash/blob/0d5a035ea26d1cc566e3c7ee4e08e5e06fa68cf7/runtime/ops/AneFfn.cpp)：当前计算先提交，下一窗口权重准备放到提交之后；图、缓冲区、PSO及完成票据由应用拥有。结合vpipe/MLX Steel的head-major attention，复用TurboCider已有Backend、NativeLinear、NativeFfn和MLX Data-owner桥。本轮不链接Splash/vpipe runtime。

这一步补齐完整GPU模型图，先证明单图净收益。没有直接复制上游LLM encoder的模型数学：Qwen3-VL具有视觉位置插值、D72、GQA、interleaved mRoPE和DeepStack。本轮没有把原来的private-ANE桥直接嵌入新图，没有声称达到设备事件驱动的W2，也没有迁移DiT/VAE。

## 2. 实际代码和边界

入口为 [NativeEncoder](../../../native/models/qwen21/native_encoder.hpp) 和 [实现](../../../native/models/qwen21/native_encoder.cpp)。两条执行API分别为：

```cpp
VisionFeatures vision(Weights&, const Tensor& patches, int grid_h, int grid_w,
    const Event&, std::atomic<bool>&, VisionConfig);
Tensor language(Weights&, const Tensor& embeddings, const Tensor& positions,
    int valid_tokens, const std::vector<Tensor>& deepstack,
    const Event&, std::atomic<bool>&, TextConfig);
```

API会释放已经物理执行完成的层权重。复用执行器时必须重新创建/加载Weights，不能重复使用已消费的映射。PSO按几何缓存在NativeEncoder内，不捕获旧权重或activation。Session卸载会销毁执行器；每次新执行清空阶段统计。

MLX仅负责原checkpoint的lazy reader、图像预处理、tokenizer/embedding lookup及prompt assembly、元数据reshape和输出retain等边界操作。视觉和语言模型塔的norm、projection、RoPE、attention、非线性、残差、merger全部由Native Metal编码执行。没有用MLX算一部分层再把结果标成完整Native encoder。

完整图进入 [Qwen Session](../../../native/models/qwen21/pipeline.cpp)，继续复用held source lease、视觉/conditioning cache、参考VAE及既有MLX DiT。环境recipe在进程内冻结，拒绝在已有缓存的进程中切换后端或视觉精度。

## 3. 视觉塔完整图

```text
FP32 patches → flatten-conv-equivalent GEMM + bias
  → FP32 bilinear position interpolation
  → 27 × {LayerNorm → QKV → 2D RoPE / pack D72→80
          → FP32 SDPA(scale=1/sqrt72) → unpack/drop zero tail
          → projection → residual → LayerNorm
          → FC1 → approximate GELU → FC2 → residual}
  → DeepStack mergers at vision layers 8/16/24
  → final merger
```

视觉保留FP32模型边界：位置插值系数是FP32，不能因checkpoint是BF16就把全部视觉state改为BF16。二维RoPE的频率、2×2 patch顺序和四点位置插值都与本地VisionEncoder相同。

DeepStack merger先2×2 shuffle再归一化，final merger先归一化再shuffle；各自经过两次projection与GELU。输出包括merged和三个DeepStack，不遗漏任何输出。默认视觉GEMM将原始BF16权重临时转换为FP32，仅保留当前command graph的转换平面，不复制完整checkpoint或导出模型。

可选BF16 GEMM将FP32 projection输入临时转换为BF16，仍保留FP32 accumulator、bias、norm、attention、RoPE、GELU、残差及输出。它改变乘法输入精度，需要显式近似许可；本轮只通过条件张量筛选，没有整图/发布资格，Native默认视觉仍为FP32。

## 4. 语言塔完整图

```text
BF16 embedding + int32 positions[3,M]
  → 36 × {FP32 RMSNorm reduction/affine → BF16
          → Q/K/V projection
          → Q/K head RMSNorm → BF16 interleaved mRoPE
          → causal GQA SDPA → output projection → BF16 residual
          → FP32 RMSNorm → BF16
          → fused gate/up/SiLU/hidden → down → BF16 residual
          → early-layer DeepStack addition}
  → official edit pre-final-RMSNorm hidden → retain conditioning
```

生产几何为H4096、F12288、Q32/KV8、D128、36层。保留原始8组KV，原生shader按`query_head/4`选择KV，不展开到32组。原生 [NativeAttention](../../../native/backends/metal_graph/native_attention.mm) 新增显式KV头数及FP32/D80特化，默认BF16/D128/equal-head的DiT调用保持兼容。80字节参数ABI不变，原reserved0承载显式GQA分组。

注意不同模型的精度边界：语言RMSNorm在FP32乘scale后才转BF16；Q/K norm与RoPE的中间BF16舍入保留；残差先舍入，再加入DeepStack并再舍入。NativeFfn保留BF16 gate/up、Sigmoid/SiLU、hidden以及down输出边界。DeepStack进入语言层0/1/2，不是语言层8/16/24。生成数据另验证可选final RMSNorm，编辑生产路径不添加该norm。

输入行数是实际M，范围1..4096，无固定prompt桶或静默截断。当前只接受无padding且valid=M的完整语言序列；未知mask/geometry明确拒绝。视觉范围≤4096 patches、偶数网格及merge2。完整单图实测是1066行，与prompt及参考图大小绑定。

## 5. 提交、准备和物理完成

初始化视觉patch graph单独提交，随后视觉/语言按两层窗口提交。完整27+36层共33次提交、1053次dispatch。

```cpp
auto ticket = backend.submit(std::move(current_graph));
// 当前Native queue已经开始工作；再让lazy reader准备未来最多两层。
if (system_available_bytes > 5_GiB) ready(next_two_layer_sources);
auto completion = ticket.wait();
require(completion.ok && completion.physically_complete);
erase(consumed_layer_sources);
```

记录30个future weight window。`ready`仍是host `mx::eval`，读取/CPU准备与当前Native queue有重叠机会，不能从计数推断完全隐藏I/O或已实现W2。可用内存不足则跳过预取，下一窗口正常同步读取。当前只允许非constrained-memory研究域，尚未纳入MemoryExecutionContext的native workspace账本，因此不能宣称24GiB/任意内存档位可用。

所有graph保留BufferView、MLX Data owner、PSO和参数。未来来源准备异常必须先wait当前票据，再抛异常；取消只在有物理完成证据的边界释放资源。Backend超时/失败会变为unhealthy，下次Session执行拒绝复用并要求重启。不能以逻辑失败或event已signal代替物理完成。

源权重上界为当前两层加未来两层；语言约4×368MiB原始权重，临时activation另计。视觉FP32 workspace约377MiB，转换平面与merger临时buffer由各graph持有至完成。此处为几何分析，不是全进程容量认证或通用内存准入策略。

## 6. 实验入口与结果披露

保持默认MLX。CLI单图512² GPU编辑显式设置：

```sh
TURBOCIDER_QWEN21_ENCODER_BACKEND=metal \
TURBOCIDER_QWEN21_NATIVE_VISION_PRECISION=fp32 \
build/qwen21-native-encoder-r3-20261008/turbocider generate LOCAL_MODEL REQUEST.json
```

Request需Qwen-Image-2.1、image.edit、恰好1张图、output512²、reference512或1024、GPU、allow_approximation=true、MLX DiT、hybrid_mlp_mode=auto、不启用prompt enhancement、不提供ANE manifest、不绑定constrained-memory context。拒绝与旧MLX encoder runtime/S1/calibration混用。这里没有禁用DiT LoRA，六步整图验证使用本地Viggle；encoder自身的runtime LoRA仍明确拒绝。

结果报告`native_metal_graph_qwen3_vl_experimental`、实际层数/提交数/dispatch、host source-ready/wait与准备耗时、缓存命中和physically_complete。conditioning cache命中报告0层且conditioning_cache_hit=true，不能冒充重新执行完整encoder。`experimental=true`、`release_qualified=false`保持明确，DiT backend仍为MLX。

## 7. 已证明与后续边界

完整组件、真实前两层、27+36层和一组512²单图六步编辑全部执行通过。本轮完整encoder独立对照省时24.93%（FP32视觉）／26.77%（BF16视觉GEMM）；实际Session首次encoder省时7.49%，请求省时1.59%。这两个口径分别保留：不能用独立encoder的26.77%宣称App整图快26.77%。

本轮没有长时间重复跑图、下载模型、导出Core ML或W8模型、清理系统cache。PSO/driver cache在各独立进程之间没有重置；只有有限样本，不授自动路由或发布资格。真实图片与conditioning的误差和完整证据在状态文件。

原生GPU塔完成后，进一步的异构路线应接Native FFN的packed-input、GPU complement、ANE physical completion和FP32 join，逐层验证语言专属量化后再扩大设备窗口。不能把21的三图FFN收益转授单图Native路径，也不能在当前33个host窗口上预先宣称设备驱动的跨层并行完成。首次PSO预热、source-reader调度、完整MemoryExecutionContext以及无缓存跨prompt稳定测量仍是独立工作；在已有单图收益基础上推进这些工作。

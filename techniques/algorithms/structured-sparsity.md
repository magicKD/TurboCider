# 结构化稀疏：窗口、STA、VSA 与径向注意力

配套：[机制 witness 与证据合同实验](../labs/algorithm-research.md)。这是支持性小实验，不是本方法的完整模型复现。

[算法入口](README.md) · [模型 sparsity](model-sparsity.md) · [Sol-Attn](sol-attention.md)

来源：FastVideo 文档/MLX 实现 [S15](sources.md#s15)、径向 attention [S26](sources.md#s26)。分类随方法而变：窗口/STA 的推理近似可不训练；带学习压缩 gate 的 VSA 需要匹配训练权重。

## 为什么 tile 布局是算法的一部分

视频序列来自 `(time,height,width)` 展平。空间邻近与时间邻近在物理内存上并不等价。STA（Sliding Tile Attention）把窗口与 kernel tile 对齐，让选中的 block 尽量是完整、连续的计算块，减少边界碎片和被 mask 掉但仍执行的矩阵工作。通常要按 layer/head 搜索窗口，并对敏感阶段 dense fallback。

一维 `abs(i-j)<r` 不是三维时空邻域：换一个 flatten 顺序便改变语义。若先做 permutation，Q/K/V、位置与输出逆变换必须协调；已施加 RoPE 的向量可以一同重排，但不能给重排后的 slot 重新分配位置。

径向注意力利用时空距离与 attention 能量衰减设计不同距离的预算；它与固定窗口“半径以外全删”不同。阅读与迁移时记录模型/帧布局、距离定义、是否保留 anchor、窗口增长规律及层/步例外，不把论文渐近复杂度等同于当前尺寸速度。

## VSA：不要忘记 gate 权重

FastVideo 的 VSA 用低分辨率/压缩表示选择 top-k blocks；与 train-free 均值路由不同，某些模型依赖训练过的压缩 gate。其 Apple H3 路径说明区分 dense-only checkpoint（去掉 gates）与 VSA-capable checkpoint（保留 gates 并写 metadata）。仅重命名/补一个随机 gate 不是有效移植。

本次固定版本的文档还指出 MLX `auto` 是 chunked gather+SDPA reference；`simd` Metal tile64/dim128 是显式候选，有 fallback，480p 四步生成仍有视频不匹配的限制。存在 Metal kernel 不等于其输出已获模型级资格。[S15](sources.md#s15)

FastVideo 的 STA 完整 inference 接入已归档到 `sta_do_not_delete`，main 保留 kernel；项目倾向 VSA 是其维护决策，不能推广为“VSA 在所有 Mac 和模型都更好”。

## MLX 最小实现应该长什么样

对每个 query chunk，构造它真正需要的局部 K/V 切片与独立 exact sink；合并重叠区域，避免 sink 被计两次，再运行局部 mask 的 fused SDPA。这样可以少读取无关 keys。把同一个全长 N×N boolean mask 传给 full SDPA，语义相同但不保证后端跳过计算。

FastVideo `windowed_attention.py` 正是这类 scaling reference；它不等于高效 STA/VSA 论文 kernel。每个 chunk 的 host loop、拼接、mask、launch 都计入耗时；不能将只测 SDPA 的数字当作完整分支速度。

## Apple 实验优先级

先 full SDPA → all-exact chunk 控制 → 全长 local-mask → 实际 sliced local+sink；同一个局部语义应数值匹配。再比较真实 3D tiles、动态 top-k 和 learned gate。高稀疏率却没有速度可能来自 launch/拼接/tile 低效，不能不断加大窗口裁剪掩盖实现成本。

非因果 DiT 不可误用 LLM causal mask；text-query 的全局交互、I2V 条件帧、音频、尾块及空行要专门控制。纯局部图对远距离主体互动、镜头运动和身份保持风险很高。

论文：[STA, arXiv:2502.04507](https://arxiv.org/abs/2502.04507)、[VSA, arXiv:2505.13389](https://arxiv.org/abs/2505.13389)；径向作者代码见 [S26](sources.md#s26)。

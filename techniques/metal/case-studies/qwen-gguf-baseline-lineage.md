# GGUF量化比较先证明基线配对：原始BF16、官方FP16与Q8不是同一个标签

<!-- knowledge
id: technique-metal-case-studies-qwen-gguf-baseline-lineage
type: case-study
stage: prefill, decode, ffn
operator: quantization, gemm, norm, layout
backend: metal, cpu-reference
device: cpu, gpu
symptom: measurement, precision-loss, wrong-layout
semantics: approximate, contract-only
representation: q8, fp16, bf16, fp32, gguf
boundary: weight-runtime, producer-consumer, request
api: public, reference-only
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

> 固定版本实现案例，供理解机制与复核边界；学习先读[方法导航](../README.md)，数据保留规则见[证据说明](../../MAINTENANCE.md)。

[GPU导航](../README.md) · [held-out评分](qwen-heldout-distribution-scoring.md) · [初始来源](../sources.md#qwen-quality-sources-20261017) · [验证（过程原件已移出）](../../MAINTENANCE.md)

2026-10-17，第十九轮后验控制。初始方案以官方BF16原权重和官方Q8 GGUF作完整数学对照，却发现**仅2/170个Q8矩阵可直接从BF16按参考codec逐字节重建**，未量化FP32 tensor只有72/121相同。不能把总差异称为“纯8-bit量化误差”或归给Metal。

看到差异后，额外下载**同一GGUF仓库/同revision的官方FP16文件**，原8篇/64target不变。明确标注post-hoc，不伪装成原预置方案。

## 1. 用真实字节证明配对

原BF16为`Qwen/Qwen2.5-0.5B-Instruct@7ae557604adf67be50417f59c2c2f167def9a775`，988097824 bytes，SHA `fdf756fa...`。GGUF同源仓库`Qwen/Qwen2.5-0.5B-Instruct-GGUF@9217f5db79a29953eb74d5343926648285ec7e67`：Q8为675710816 bytes、SHA `ca59ca7f...`；FP16为1266425696 bytes、SHA `8e0ae260...`。完整SHA/URL/逐tensor身份在capsules，简称不作为校验。

后验客户端读取全部FP16 GGUF参数，保留decoded FP32 SHA，按[固定原始Q8_0参考编码器](https://github.com/ggml-org/llama.cpp/blob/64e9bceb2c3a856efed96feda784a50947049feb/ggml/src/ggml-quants.c)逐32-lane块复算：`d=max(abs(x))/127`，保存FP16 scale，但int8计算用未压缩FP32 d的倒数和`roundf`（half-away-from-zero）。保存后scale不能冒充原d，Python bankers-round也不能默认相同。

| 比较对象 | 全张量/位流结果 | 允许结论 |
| --- | --- | --- |
| 原BF16→发布Q8 | 2/170矩阵编码相同；72/121 FP32 tensor相同 | 不是未变换BF16的一步Q8对照 |
| 同仓库FP16→发布Q8 | 170/170矩阵所有half scale/int8 lanes逐字节相同；121/121 FP32 tensor相同 | 在相同短prefix数学图下可隔离这对artifact的codec差异 |
| 原BF16↔同仓库FP16 | 291个映射角色仅72个完全相同 | 表示/参数变换或checkpoint来源差异仍需解释 |

原BF16只有290个独特tensor，tied embedding映射GGUF embedding/head两个角色，不多算独立训练参数。极小BF16值转FP16可下溢/舍入，但layer norm和projection差异并非都能用普通转换解释；文件名“fp16”不是无其他变换的保证。

## 2. 第一次拒绝揭示context metadata不同

最初要求所有模型/tokenizer metadata一致，实际拒绝：FP16 `qwen2.context_length=8192`，Q8为32768。失败receipt保留，没有改artifact或隐藏差异。

进一步检查其余Qwen数学参数与`tokenizer.ggml.*`完全相同，291个名字/形状一致。当前oracle在32–39长度fresh-prefix方程中**不读取context_length作数学缩放**；因此只对当前短prefix图建立配对，记录context差异，不称全部metadata相同。8192/32768部署、RoPE外推、KV分配和原生FP16路径等价仍未证明；旧数学门槛/预置位置没有改变。

## 3. 三段差异分开，KL不能相加

相同64个target的完整数学结果，KL方向均为该行“参考→候选”：

| 参考→候选 | mean ΔNLL nats | mean forward KL | 样本PPL比 | top1相同 |
| --- | --- | --- | --- | --- |
| 原BF16权重→官方FP16 GGUF权重 | +0.0243314 | 0.0166853 | 1.024630 | 60/64 |
| 官方FP16 GGUF→Q8 GGUF | +0.0034047 | 0.0006937 | 1.003410 | 62/64 |
| 原BF16权重→Q8 GGUF | +0.0277361 | 0.0185370 | 1.028124 | 58/64 |
| Q8数学→Metal，FA off/fresh | −0.0000334 | 0.0000033 | 0.999967 | 64/64 |

NLL是同observed token的标量，差值可逐位置相加，验证误差<1e-12；**KL不满足此可加性**。不能用`0.018537−0.0000033`算“纯量化KL”，也不能把不同方向KL直接相加。

主capsule的初始字段名`quantization_vs_BF16`保留原实验记录；应按本页解读为该BF16与发布Q8的总artifact对照，而非已隔离的纯codec损失。真正通过字节配对的FP16→Q8控制在后验capsule中另列，未回写初始证据。

样本PPL约6.229695（BF16）、6.383131（官方FP16）、6.404901（Q8）；实际引擎四路径约6.403628–6.405006。没有原生BF16/FP16引擎运行，只有同一完整方程加载不同权重。512个实际Q8输出仍全未过原tensor预算，分布指标不升级其资格。

前两段8文章条件bootstrap ΔNLL区间约[−0.02456,+0.07368]和[−0.00130,+0.00808]，均跨0。64个特定短target不足以证明总体损失，也不足以证明总体无损；未证明显著差异不等于已证明等价。没有事后通过阈值或速度许可。

## 4. 权重不同不自动说明训练checkpoint不同

[固定SmoothQuant源码](https://github.com/mit-han-lab/smoothquant/blob/c61476d728e42ae0d8a35e7e78494edcac3237b5/smoothquant/smooth.py)提供**同义参数重标定**例子：RMSNorm gamma除逐channel scale，下游各linear列乘同scale，理想数学consumer可不变；只改一边则错。权重分布/量化误差可以变化，运行结构未必改变。

本轮仅阅读选定folding源码，未执行SmoothQuant、未下载activation scales，**没有证据证明Qwen用了它/AWQ或特定校准集**。这是解释“tensor相等、函数相等、来源历史”区别的类比，不为观察结果补造历史。发布artifact的实际变换与来源因果仍待证据。

## 5. 实现与复查

CPU合同/tests分开检查平移、参数folding、KL方向/不可加性、BF16格式、完整配对和context差异：

主方案、128份初始CPU参考、512份实际输出未重写；追加64份FP16 CPU参考及256份canonical实际输出对它的指标，不增加held-out样本量。模型只在ignored outputs，参考仓库只读，无上游提交/生产默认修改。

复查条件：artifact/revision/SHA、codec scale/rounding、mapping/tied角色、graph/context/RoPE、参考数学精度、target、样本规则及统计预算。长文本/更多模型语言、真实任务、native BF16/FP16和来源历史未完成，不将此小样本泛化成“Q8普遍只有0.34%质量损失”。

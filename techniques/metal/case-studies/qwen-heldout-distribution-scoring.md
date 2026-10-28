# 完整Qwen的held-out评分：logits预算拒绝、NLL与分布偏移是不同证据

<!-- knowledge
id: technique-metal-case-studies-qwen-heldout-distribution-scoring
type: case-study
stage: prefill, decode, pipeline
operator: sampling, attention, gemm, norm
backend: metal, cpu-reference
device: cpu, gpu
symptom: measurement, precision-loss
semantics: approximate, contract-only
representation: q8, gguf, bf16, fp32
boundary: weight-runtime, producer-consumer, request
api: public, reference-only
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

> 固定版本实现案例，供理解机制与复核边界；学习先读[方法导航](../README.md)，数据保留规则见[证据说明](../../MAINTENANCE.md)。

[GPU导航](../README.md) · [来源](../sources.md#qwen-quality-sources-20261017) · [参数与量化配对](qwen-gguf-baseline-lineage.md) · [验证（过程原件已移出）](../../MAINTENANCE.md)

2026-10-17，第十九轮。此前完整Qwen严格logits预算大量拒绝，但生成同词。本轮换用**预先固定的WikiText-2 test文本和teacher-forced真实后续token**，独立计算全模型参考，不改容差或用生成文本恢复性能资格。

4个原始llama.cpp进程/512份完整151936维输出仍全部拒绝旧数学预算；但相对**同一decoded GGUF数学参考**，64个不同评分位置的平均forward-KL仅约2.3–3.3×10⁻⁶，top1均一致。这说明严格tensor拒绝不能直接翻译成“语言质量明显坏了”；**小KL也不授权原tensor通过、通用质量无损或性能排名**。

## 1. 先冻结输入和统计单位

driver先下载核验资产，独占创建`plan.json`，随后才运行smoke和正式GPU。固定[WikiText数据卡](https://huggingface.co/datasets/Salesforce/wikitext/blob/b08601e04326c79dfdd32d625aee71d232d685c3/README.md)及[llama.cpp下载脚本](https://github.com/ggml-org/llama.cpp/blob/64e9bceb2c3a856efed96feda784a50947049feb/scripts/get-wikitext-2.sh)所用`ggml-org/ci` archive的commit/zip SHA；不执行会删文件的原脚本。

确定性规则：test前8篇符合条件的顶层文章，各选第一条至少40个原tokenizer IDs的正文行。前32 token作上下文，预测位置32–39的8个真实token；不加BOS/EOS/chat template，不拼接文章，不用实际argmax代替observed target。完整标题、行号、原文SHA和40个ID均保留。文章为Robert Boulter、Du Fu、Kiss You、Ise-class battleship、Dick Rifenburg、1933 Treasure Coast hurricane、Second Battle of Naktong Bulge、Hed PE。

这是未用于此前本地两prompt或阈值调参的**本地研究held-out**，不证明模型训练未见过它们。每篇开头、短context、英语Wikipedia及instruction模型会产生选择偏差。不是完整WikiText PPL、随机代表性样本、中文/代码/聊天benchmark或任务准确率。

## 2. 两种真实路径消费相同前缀

stock `llama.cpp@64e9bce`，Flash off/on × replicas17/29，共4进程，每进程128份输出，每篇执行：

- fresh-full-prefix：每个评分位置清状态，重算32–39长度的完整prefix。
- cached-one-token：32-token prefill后逐个输入**真实文本token**，不是模型选择的token。

两replica反转文章/路径顺序。512个调用仍只有**8篇文章、64个不同target**；重复进程不增加统计样本。256个Flash/route/article/position分组在两replica间payload逐位相同。

复用上一轮sealed原生客户端，不改旧源码、binary、stock库或public ABI。实际加载库路径/逐库SHA匹配stock，没有observer符号；last-only、F32 KV、请求context128/实际256保持不变。1个4输出smoke不计入正式512。普通tests不启动引擎。

## 3. 两套完整数学参考，原门槛仍不变

继承SHA固定的独立24层oracle，每个位置重新计算整个prefix和151936个head值：FP64 reductions、显式FP32 stage边界，不用llama.cpp CPU量化forward或仅top1。

第一套独立解码Q8 GGUF，全部291个参数SHA匹配前轮见证；第二套读取官方BF16 safetensors位模式，直接扩展为精确FP32权重：290个源tensor全部映射，tied embedding/head映射291个角色。无Torch/Transformers forward或BF16硬件算术，因此叫**BF16权重数学参考**，不是原生BF16引擎或性能。

初始64×2=128份完整CPU向量保存，后验同仓库FP16另有64份，见[配对页](qwen-gguf-baseline-lineage.md)。每份GPU输出完整SHA/有限值/实际argmax核验，并对两套参考逐份全词表检查。旧门槛仍是逐元素`.005+.005*abs(ref)`及relative-L2≤.005，512/512拒绝。新指标仅补充诊断，所有`qualified_model_performance_ms`保持null，无速度排名。

## 4. 分布指标与逐元素指标回答不同问题

用完整支持集的稳定FP64 log-softmax：先减最大值，再log-sum-exp；不只留top-k、不丢词表rows、不改支持集。

- NLL：observed target的`−log p(target)`，不是argmax命中率。
- paired ΔNLL：同target的candidate NLL减reference NLL，正值表示其概率下降。
- forward KL：`Σ p_reference (log p_reference−log p_actual)`，方向不能换。
- TV：概率差绝对值和的一半。
- 样本PPL：`exp(mean NLL)`，不是逐token perplexity平均。

全部logits同加常数可让逐元素预算失败而softmax不变；相同argmax也可让某observed token NLL大幅恶化。CPU正负控证明这些区别，**不声称本批实际误差全是常数偏置**。

对读[原始perplexity实现](https://github.com/ggml-org/llama.cpp/blob/64e9bceb2c3a856efed96feda784a50947049feb/tools/perplexity/perplexity.cpp)：其base log-prob缓存压缩成uint16，选定KL路径只累加base log-prob>−16的词。本轮由完整FP32 logits作FP64归一化，不沿用这两种近似，没有运行其CLI，更不称默认benchmark复现。8个真实完整词表case再用标准库scalar scorer交叉检查向量化指标，误差<1e-11。

## 5. 实际结果与有限样本不确定性

每行仅用canonical replica17的64个target；replica29只作重复控制。参考为decoded-GGUF数学分布，ΔNLL单位nats。

| Flash /路径 | mean ΔNLL | mean forward KL | mean TV | top1相同 |
| --- | --- | --- | --- | --- |
| off /fresh | −0.00003340 | 3.305×10⁻⁶ | 0.0007285 | 64/64 |
| off /cached | −0.00019882 | 2.320×10⁻⁶ | 0.0005957 | 64/64 |
| on /fresh | +0.00001641 | 2.923×10⁻⁶ | 0.0006666 | 64/64 |
| on /cached | +0.00000250 | 2.962×10⁻⁶ | 0.0006616 | 64/64 |

负ΔNLL只表示这批真实token更受偏好，不叫“引擎比参考更正确”。对原BF16权重数学参考，top1只有58/64；不能把六个差异全归因Metal/Flash，权重基线本身已不同。

预置seed17、2000次**文章块bootstrap**，每篇8个相关位置一起重采样，不把64 token当64独立文档。四种runtime ΔNLL条件区间均跨0，整体约−0.00098到+0.00080。它们只描述选定8篇文章，不证明总体等价或无损；没有设置事后新质量通过阈值。

## 6. 复现、来源边界与复查

CPU tests检查平移、同top1恶化、KL方向/不可加性、PPL、block重采样、BF16格式范围、保存的全部prefix/target/shape/门槛与来源哈希，不需模型/GPU。大参考和设备实验需显式supervisor、新ignored目录及help所列参数，GPU串行。

新BF16与后验FP16模型合计约2.3GB，仅留ignored outputs；WikiText原文同样不进入Wiki。遵守原Wikipedia贡献者/WikiText CC-BY-SA3.0/GFDL及模型Apache2许可，不重新授权。复查条件：模型/词表/模板、target对齐、context/mask/position、dtype/FA/输出需求、统计单位或任何质量阈值变化。更多语言/长context/真实任务及原生BF16/FP16消费者仍缺，本轮不关闭研究目标。

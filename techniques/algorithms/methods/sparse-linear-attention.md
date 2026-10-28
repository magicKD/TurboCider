# SLA / SageSLA：补集与全局线性修正不是同一个checkpoint语义

<!-- knowledge
id: technique-algorithms-methods-sparse-linear-attention
type: method
stage: dit
operator: attention, gemm, layout
backend: cpu-reference, mlx, metal
device: cpu, gpu
symptom: precision-loss, wrong-layout, measurement
semantics: requires-training, approximate, contract-only
representation: fp32, bf16
boundary: weight-runtime, producer-consumer
api: reference-only
-->

[算法入口](../README.md) · [训练型稀疏前沿](spargeattention2.md) · [Sol补偿](sol-attention.md) · [VMoBA合并](vmoba.md) · [实验](../../examples/README.md)

2026-10-16第七轮。实际读取`thu-ml/SLA`两个固定版本：[main/SageSLA S40](../sources.md#s40)与[SLA-1 S41](../sources.md#s41)。main README引用SLA2，但本轮所查类/helper没有验证到论文所述learnable routing/QAT路径；**citation不是实现等价证明**。没有加载模型checkpoint、训练完整模型或运行CUDA/Triton。

## 先分清哪些东西在学习

两份公开类的新增参数都是`proj_l.weight/bias`，维度D×D和D，沿最后一维作用于各head；构造器没有head数参数，因此不是每head单独一套P。softmax/ELU+1/ReLU feature maps在这些类里是固定函数，不含可学习的Q/K compression参数。路由是mean-pool Q/K后的离散top-k，不是已核验的SLA2 learned router。

训练整个Transformer仍可更新外部QKV/FFN；不能把“本模块新增参数少”说成“整个微调只训练P”。也不能因公开推理类可调用就省略配套微调与checkpoint验证。

## 两个实际分支的计算图

令φ是通道维的feature map，`S_all=Σφ(k_j) v_jᵀ`、`z_all=Σφ(k_j)`。稀疏支路o_s是仅在选中KV blocks上重新归一化的softmax attention。

| 版本 | 线性支路 | 组合 | 本轮核查范围 |
| --- | --- | --- | --- |
| SLA-1 `325cc7…` | 全局S/z减去选中block摘要，只对**未选补集**归一化 | o_s + Linear_P(o_l) | native preprocess/fused forward与接口选段；Python body CPU包络 |
| main `7db403…` | `φ(q)S_all/(φ(q)z_all+1e−5)`，覆盖**全部KV** | o_s + Linear_P(o_l) | 完整Python类/helper；原稀疏kernel只读，CPU参考后端 |
| SageSLA同一main | 量化稀疏forward＋main式全局线性项 | 同上 | 构造/调用链源码；未执行量化forward或其反向 |

SLA-1为每个K block建立S/z，再减selected摘要；main只算全局KφᵀV与Kφ总和，省去了该补集workspace，但也改变了函数。main不是在同一函数上单纯“把kernel换快”。

即使SLA-1的连接集合不重叠，两支路也分别归一化，且feature kernel不同，之后由P学习修正。它不是full softmax的精确分子/分母合并，也不是[VMoBA的同核LSE合并](vmoba.md)或[Sol当前summary补偿](sol-attention.md)。不能把它们都称为remainder便互换归一化。

## 相同state_dict可成功加载，输出仍不同

sla_source_probe.py核对11份完整源文件hash，提取原core class/get_block_map；真正的pool、attention和backward换成明确CPU参考，内部dtype改为FP32、CUDA autocast为no-op。**不验证原BF16、Triton kernel、量化或原custom backward。**

使用两seed的合成QKV（B1/H1/L256/D64），两类严格加载相同P/b state_dict，keys/shapes/值相同。P=.25I等非零值是人为构造，不是论文训练得到的checkpoint。观察如下：

| 条件 | main | SLA-1 | 学习点 |
| --- | --- | --- | --- |
| P=0、b=0，keep1/4 blocks | 两者同一稀疏输出，但不等dense | 同左 | zero初始化保留的是稀疏支路，不是原dense |
| P=0、b=0，全部选取 | 等dense | 等dense | 可用的full控制必须同时关闭修正 |
| P=.25I，keep1/4 blocks | 与SLA-1输出不同 | 与main不同 | strict state load成功不能证明语义兼容 |
| P=.25I，全部选取 | 仍加全局线性项，不等dense | 补集置零，等dense | topk=1单独不是通用all-exact控制 |
| P=0、b=.03125，全部选取 | 加bias，不等dense | 加bias，不等dense | 关闭矩阵权重还不等于关闭affine修正 |
| ReLU令φ(Q/K)全0，partial keep | epsilon使线性项为0，finite | 补集0/0后为NaN | 零P乘NaN仍为NaN；默认softmax外的边界要审计 |
| K只有1块、topk=.2 | floor为0，参考稀疏支路NaN | 同左 | 不能用topk=0当纯线性模式 |

非零partial P下，两seed跨分支max abs约0.0119；全部选取的main与dense也有非零差。这些是构造输入上的接缝反例，不代表实际训练模型已发生同样生成回归。

SLA-1 kernel对full selection显式令线性denominator为Inf，使补集输出0；partial时没有main的epsilon。CPU引用的是该公式，但不模拟原summary dtype、masked-load、MMA或累加顺序，因此NaN与finite观察不得冒充CUDA实机复现。

建议checkpoint元数据绑定：source revision、`linear_scope=global/complement`、feature map、epsilon、P布局/共享方式、bias、block几何、route语义和quant backend。square D×D权重即使转置错了也可能通过shape检查，端口必须保留Torch Linear的out×in含义。

## top-k与返回的“sparsity”是什么

所读helper取`min(K_blocks, int(topk_ratio*K_blocks))`，没有保底1；2seed×2分支×7配置的28个完整route观察见。负ratio可到达Torch RuntimeError，>1被clamp到全部，小ratio可为0。

`return_sparsity=True`返回的是`real_topk/K_blocks`，实际为**保留block比例**，不是1减该值。L256、topk=.25时返回.25而非.75；L65有64/1两种尾长，选1/2 blocks也不等于选1/2 tokens，更不代表完整支路FLOPs。

L65观察使用valid-only mean pooling的CPU替身；它验证Python路由与有效token计数，不验证原Triton tail masked-load、pooling舍入或稀疏kernel的尾块行为。

源码kernel以q的L同时索引Q/K/V；公开接口描述equal-length非因果self attention，不是任意Q≠KV、GQA、causal或packed mask API。MLX/MPS adapter应显式拒绝未验证语义，不能借CPU参考支持更宽shape便宣称原kernel支持。

## 零初始化与梯度要看具体路径

另有4个CPU案例：在原类＋参考后端中，以dense参考为teacher，仅对4160个P/b参数做一步SGD。固定fixture损失均下降；没有训练完整模型、发布checkpoint或做性能/质量验收。

P=0时，P/b梯度非零，而φQ/φK这条线性路径的梯度为0；Q/K/V仍通过稀疏参考支路得到非零梯度。不能将其误读成整个网络无法训练。若移植时新增learnable feature maps又冻结P，应审计这种路径初始化，而非仅检查`loss.backward()`未报错。

CPU参考的autograd不验证原Triton custom backward；SageSLA被作者描述为量化forward，本轮也未验证其QAT/反向能力。不能把一个快速quantized forward直接当作可微训练后端。

## 文档/接口也要核对

两个core docstring列出`hedgehog`，实际构造均NotImplementedError；SageSLA README给的BLKQ/BLKK构造关键字，固定class签名不接受，CPU直接TypeError。这些均限定本次commit，不把问题永久贴到项目名上。

完整数据含28个forward、14个同state比较、4个梯度案例、3个构造拒绝及scalar math.fsum见证。标准库整理器检查源/输入身份、mask/LUT/尾长、SSE与非finite保留；默认tests不执行上游或设备代码。

## Apple迁移顺序与成本

先选择确切分支数学，再在FP32参考中验证checkpoint及full控制；linear φ/S/z/P可由普通GEMM/reduction表达，不绑定CUDA。真正省算的稀疏支路仍需Metal tile/gather或已验证后端；全长mask＋dense SDPA仅是数值参考，不证明跳过QK/PV。

main的linear cost约O(LD²)，还要P projection；SLA-1另存per-block S/z和query补集workspace。稀疏density很低时，summary/selection/线性路径、layout与launch都可能成为主成本，不能直接移植作者CUDA倍率或由复杂度断言胜过强MLX SDPA。

训练端需要匹配前向的gradient/custom VJP或有明确语义的QAT/STE；推理端需要对应训练参数。保留真实text/reference/audio spans，校准完整velocity/latent与媒体，不能以本轮tiny projection SGD替代这一过程。

本轮无GPU工作，也没有原SLA/SLA2完整模型速度或画质资格。重开条件包括分支/权重、feature核/合并/epsilon、dtype/backend、mask与训练recipe变化。

## 本轮离线回归记录

2026-10-16完成以下检查；这是当次工作树快照，不将测试数量作为研究覆盖或模型资格证明：

- `python3 -B -m unittest discover -s techniques/tests -p test_algorithm_round7.py -v`：22项通过。包括完整重新整理、raw/frontier不同字节序列的hash、opt-in/新输出保护、source拒绝，以及route/NaN/SSE/scalar/state/gradient/论文资格字段的篡改负控。
- `python3 -B -m unittest discover -s techniques/tests -p 'test_algorithm*.py'`：104项通过。
- `python3 -B -m unittest discover -s techniques/tests`：779项，777通过、2跳过；首轮仅生成索引未同步而失败，索引同步后重跑通过，不删除失败测试或放宽断言。
- `python3 -B techniques/scripts/check_standalone.py --isolate`：仅复制`techniques`，779项同结果、全部67个CPU cases及wiki/catalog/knowledge检查通过；不包含产品、兄弟源码、模型、上游checkout或optional框架环境。
- 来源元数据42条/139个文件hash、生成索引一致性及`git diff --check`通过。CPU原始报告经CLI重新整理，与公开JSON逐字节一致。

默认tests不执行原项目、Torch/MLX/Triton或设备，也不读取activation capsule。原始源码执行仅属于另行显式选择的CPU probe；本轮没有新增GPU计时。

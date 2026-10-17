# ToMeSD consumer 与真实 Qwen FFN：token 合并的成本、恢复和长尾

<!-- knowledge
id: technique-algorithms-case-studies-token-merge-consumer-and-real-ffn
type: case-study
stage: dit, ffn
operator: gemm, layout, sampling
backend: mlx, metal, cpu-reference
device: gpu, cpu
symptom: precision-loss, wrong-layout, measurement, launch-bound
semantics: approximate, contract-only
representation: bf16, fp32
boundary: weight-runtime, producer-consumer, request
api: public, reference-only
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

[算法入口](../README.md) · [token reduction机制](../methods/token-reduction.md) · [ToCa/DuCa](../methods/token-feature-caching.md) · [真实BF16 FFN基线](real-ffn-gpu-study.md) · [实验](../../examples/README.md)

2026-10-17，第十四轮。先执行固定ToMeSD原CPU接缝，再以原BF16 Qwen-Image-2.1 image FFN权重与真实输入测试原创**FFN-local centroid迁移**。M4 Max/64GiB、macOS26.6.2、MLX0.32.0；两层0/15、两prompt/seed42/43、原20步trajectory中的step1/10/19，共12点。

核心观察：真正缩短FFN行数可以取得组件净收益，包含在线匹配的50%移除约1.81–1.86×、75%约3.14–3.31×；但早层早步和中层输出误差显著，个别row relative-L2超过100%。**这不是原ToMeSD默认attention路径、完整block/生成轨迹或媒体质量资格**，不改产品默认。数据分为原CPU接缝/地图与实际GPU组件。

## 原源码真正启用了什么

[S48](../sources.md#s48)固定`dbolya/tomesd@27a14a372beecaa85c101fb588631643f379b8ce`，读取README、MIT license、merge/patch/utils。实际Torch2.14.0 CPU执行原函数，不导入上游初始化或Diffusers，不patch生产模型。

原`compute_merge`的顺序为：

```text
block入口hidden_states
  → 按原image size/max_downsample判定几何与r
  → 随机/固定2D划分 + normalized cosine匹配，生成一对merge/unmerge闭包
  → norm1 → 可选merge → self-attention → 可选unmerge → 加原residual
  → norm2 → 可选merge → cross-attention → 可选unmerge → 加residual
  → norm3/调制 → 可选merge → FFN → 可选unmerge → 加residual
```

**map在block入口只计算一次，三个启用的stage共享同一闭包**，不是到FFN时重新对其normalized输入选图。本轮local迁移恰恰改了这一边界：在已有FFN input上选图，因此不能写成原patch的模型复现。

真实原wrapper在明确toy stages上记录：

| 配方 | self-attn输入N | cross-attn输入N | MLP输入N |
| --- | ---: | ---: | ---: |
| 默认，N=16/r=8 | 8 | 16 | 16 |
| MLP-only | 16 | 16 | 8 |
| 三stage全启 | 8 | 8 | 8 |
| max_downsample旁路 | 16 | 16 | 16 |

原默认`merge_attn=True / merge_crossattn=False / merge_mlp=False`；作者对MLP明确不推荐。所以上面GPU实验是**主动标记为高风险的非默认迁移**，不能借作者Stable Diffusion/4090/FID数据给它授予质量或M4速度资格。原patch针对BasicTransformerBlock体系，不是现成可patch本库native Qwen/MLX的入口。

raw wrapper日志有两条route-call：一条属于原forward，另一条来自harness在forward后独立调用compute_merge检查闭包身份；不是原forward跑了两次selector，不据日志长度翻倍计成本。

## 随机网格、CFG与MPS不能混为一个保证

每个完整sx×sy区域选一个destination，其余source；不整除的底/右边缘仍进入source。source按normalized metric找最相似destination，再按source最大cosine降序选r个；`r`最多为source数。没有逐次重算centroid的聚类迭代，也不是仅在局部窗口内选最近destination：网格控制候选池，匹配仍可跨图像远处。

代码对**奇数batch强制关闭随机destination**，偶数才按use_rand启用；它没有显式CFG branch身份验证，偶数本身也不证明是正负分支。batch共享同一src/dst划分，但score/argmax/rank是逐batch的，**不保证两路得到相同map**。原CPU双分支小例中确实看到同partition、不同assignment。

MPS相关实际读取/观察：

- gather workaround是给last-dim=1的tensor加一个维度、调整负axis再调用gather，不是把数据自动搬回CPU。本轮仅在真实Torch CPU验证其rank变换等价，没有运行MPS tensor。
- utils对非CPU/CUDA的generator请求可回退CPU；传fallback时返回原对象，状态继续推进。本轮`torch.device("mps")`只作为metadata传入此helper，不是MPS后端执行。
- zero-r直接返回identity函数，原toy输入/输出为同一对象；partial3×5、stride2的例子成功覆盖15 slots，r99被clamp到13。
- metric normalization没有epsilon。全零metric的cosine为NaN，后续mean/unmerge仍可能返回finite零payload；finite输出不能证明route有效。本轮真实12点metric非零，GPU端另检查该前提。
- 故意让token数/原image size不一致分别观察到RuntimeError和ZeroDivisionError；这些破坏Stable Diffusion几何前提，不当作正常demo失败。

## 恢复的是slot，不是独立信息

原`scatter_reduce(mean)`包含destination自己：每个cluster的原dest和选中sources各贡献一次。unmerged sources先排在compressed序列前面，dest代表排在后面；unmerge把同一个代表输出复制到其各source slot，恢复N。

例如标量[1,3]的cosine为1，合并得2，unmerge得[2,2]而非[1,3]；若FFN为平方，合并后复制得到[4,4]，原为[1,9]。加原residual也不会恢复这两个独立FFN响应。这说明normalized cosine阈值/合并比例本身不提供非线性输出误差上界，norm差、Jacobian/曲率和后续depth都需要额外预算。

本次MLX port用FP32 `zeros.at[group].add(value)`及group counts实现mean，再转BF16；所有原tokens都映射一次，所以包含dest self。重复destination必须做归约，不能用普通indexed assignment/`x[idx]+=updates`代替。后者对重复索引的语义不同。

[S49](../sources.md#s49)固定MLX0.32.0五个文件，贯通ArrayAt.add→mlx_add_item→Scatter::Sum→Metal scatter的atomic_update。它是选定source机制，不是实际shader/ISA取证。FP32加法顺序的可变性仍需考虑；本轮96个repeat数值输出、map与merged BF16全部逐位相同，不把这个cohort观察推广成通用原子归约确定性承诺。

## 实际Qwen输入与GPU对照边界

沿[第九轮](real-qwen-lowrank.md)原始captures，使用层0/15、茶壶seed42/自行车seed43、0起始sampler step1/10/19的`[1,1024,4096]` BF16 image-FFN inputs，target网格32×32。它们来自cache-off/full原trajectory，近似结果没有反馈进下一层/step。没有prefix text、reference、audio token，不测试prefill/edit/video。

每层原始GU[24576,4096]、SiLU(gate)×up、Down[4096,12288]，两矩阵共288MiB BF16驻留；不降bit、不加低秩，无bias/LoRA。norm/modulation、tanh gate与原residual addition在本边界之外，不能把FFN relative-L2直接当整个block、latent或画质损失百分比。

移除r=256/512/768，分别保留768/512/256行。原CPU router在FP32展开的真实BF16 input上执行，采用奇数batch对应的固定no-rand partition。GPU online以同一partition重新做FP32 norm、768×256 cosine、argmax/rank；partition独立于feature，预备在计时外。**完整原compute_merge的几何/RNG初始化不在online计时内**，这里测的是cached fixed-partition设计，不宣称原patch全部host开销。

八计算臂为eager/compiled dense，以及三种r各自的：

- `prepared-r…`：原CPU预先选图，计时内做GPU group mean→BF16→FFN→full-N restore；是可复用/离线图的上界臂，不当作online净收益。
- `online-r…`：计时内还包含当前input的norm/cosine/rank与map构造；算完仍输出1024×4096。

计时函数返回完整输出及可审计map/merged BF16，eval这些依赖后synchronize，包含建图/compiled调用、mean/scatter、两线性/SwiGLU和恢复；不是仅短序列GEMM、物理GPU timestamp或最大融合实现。input/weight加载、CPU路由/FP64控制、编译/数值warmup在计时外，全部参数显式传入，无训练/产品改动。

每point18轮、九个计时臂（含ready输入host控制）轮转/反序，各位置两次；合计1944段formal interval，其中1728计算臂、216 host控制。一次layer0/step1两轮pilot的18段不混入。没有把组件速度乘到cache或旧量化报告上，也没有cold/长期热稳态/每方法peak结果。

## 包含选择与恢复的组件速度

每层六点各自配对中位比的范围，不是置信区间；所有point完整值见GPU JSON。分母为预固定原BF16 compiled dense，12点的中位耗时约21.18–21.58ms，不逐点改选较慢eager。

| 移除比例 / 剩余N | prepared相对dense | online相对dense | online耗时范围ms |
| --- | --- | --- | --- |
| 25% / 768 | 约1.28–1.29× | 约1.23–1.26× | 16.84–17.55 |
| 50% / 512 | 约1.89–1.94× | 约1.81–1.86× | 11.42–11.82 |
| 75% / 256 | 约3.42–3.55× | 约3.14–3.31× | 6.44–7.07 |

online比prepared多付当前feature匹配，少token数也不是恰好4×。本轮真的减少FFN M维算量，和[第十轮resident packed/低秩仍执行全行](real-ffn-gpu-study.md)的成本机制不同；不能据不同误差/边界/日期的数据宣布token merge普遍优于量化。

## 原dense误差：step/layer差异与token长尾

prepared/online在本cohort36对恢复输出逐位一致；下表列online global relative-L2对原同点GPU dense的范围（六点/层），不设质量通过线：

| 移除比例 | layer0 global | layer15 global | layer0最大row | layer15最大row |
| --- | --- | --- | --- | --- |
| 25% | 1.40–21.73% | 12.08–19.58% | 91.83% | 74.23% |
| 50% | 2.40–32.44% | 18.79–29.42% | 174.55% | 107.82% |
| 75% | 4.28–43.93% | 27.61–38.24% | 283.51% | 178.26% |

step差异不可藏在范围内：50%移除时，layer0早步两prompt约32.1/32.4%，step10约10.1/9.8%，step19约2.40/2.63%；中层step19仍约18.8/20.5%。这不是已校准的晚步安全策略，更不是可将2.4%解释成2.4%画质损失；未运行近似轨迹、媒体或新holdout校准。

即使layer0 step19整体较小，50%移除的最大row仍约7.84/9.10%，75%约18.02/12.49%。保留全部1024行SSE/energy、p95/max，不用median=0或全局范数遮住少数token。relative row error也会受较小reference能量影响，应与原SSE/energy一起读，不当作像素质量评分。

### Shape控制没有制造另一个原因

对实际返回的merged BF16 input，另做`dense(unmerge(merged_input))`，保持1024行；与`unmerge(dense(merged_input))`比较，区分input投影与M变化的数值效应。72个控制全部输出hash一致，shape-path SSE=0；本cohort未观察到该来源的误差。不能为解释失真虚构split-K/fusion原因，也不将它推广为所有shape/版本逐位相同；第十轮33-row尾形状确有不同观察。

另用独立FP64完整GU/SiLU/Down计算原token0、1023对应的两行（各4096输出通道，K/F/N不裁剪）。实际最大global约0.003919、最大sample-row约0.004016，低于冻结同公式3% implementation gate；不是全部1024行FP64，也不是对原dense误差≤3%的质量线。FP32 mean单独对FP64 group mean，计时图实际返回的BF16 centroid再核对1%舍入门槛；单独mean32审计不是对compiled图每个中间bit的取证。

### Raw slot编号不等于聚类身份

GPU与源CPU有18个online图的unmerged slot顺序不同，最大31个raw map entries不同；按原token代表ID canonical化后，全部36图聚类一致。所有prepared/online恢复输出也一致。若只比较compressed slot编号会误判路由变了；若只看canonical group又不能忽略下游layout/order合同。两种信息都保存，不能任意替换。

## 可靠性与学习边界

两个formal GPU child、一个pilot、一个原CPU child均正常退出。每GPU setup、point数值阶段和formal trial前重新检查3个driver样本Device≤5%，formal最后样本到开始≤2秒。正式保留218个先前busy窗口，pilot另2；很多在自身warmup/上一trial后出现，不标成外部任务次数。附脱敏pid/pcpu/comm，无argv，不杀其它进程；无连续per-PID独占或热/频率控制。

四个原BF16 weight spans及12个input、partition/prepared maps在前后hash相同，checkpoint检查dev/inode/size/mtime generation；不是完整大checkpoint payload SHA。所有输出数值控制在formal前完成，并做一次repeat；不声称1944个timed outputs都独立回读检查。未测每方法物理内存，减少token不改变288MiB权重payload，scatter/restore仍有full-N成本。

CPU标准库整理器重算tiny mean/sum/unmerge及source map/rank/CFG/default合同，不读取真实capsule或重算其完整cosine；GPU整理器重算全row SSE/分位数、map/count/canonical身份、projection/shape误差三角界、gate及全部paired intervals，不执行checkpoint FFN/NumPy/GPU。默认tests无Torch/MLX/上游、权重或真实activation读取。实际source与设备probe均须显式`--run`。

接入前应先固定真正的metric boundary、token/branch/geometry身份和真实质量预算，保护文字/reference/audio及RoPE；晚步/浅层候选需另做完整feedback trajectory和独立媒体验证。可以探索norm/variance/距离约束或更保守局部候选，但这些都是新控制器，不是本次结果已证明安全的默认。原ToMeSD的attention路径与MLP-only迁移继续分开，不按组件赢点自动启用产品。

## 交付复核：2026-10-17

两份公开JSON经独立CLI重新整理逐字节一致，S48/S49十个Git对象重新收集与ledger一致；累计来源50条/200个文件hash条目。版本登记绑定真实GPU与CPU证据、各probe和父数据；匹配历史环境仍不授予媒体/默认ToMeSD/完整Qwen资格。没有删除坏row或改质量门槛，也没有修改上游/产品默认。

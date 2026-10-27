# Runtime ANE：LoRA与Z-Image/LTX接入分析

2026-10-19，M4 Max/64GiB。用户新增范围现在包括Z-Image Turbo、LTX2.5；
此前“暂不关注Z”的限制已由新请求替代。总体目标ACTIVE，App/default未改。
本页区分当前代码已实现、数学可行与尚待整请求测量，不将它们混为速度资格。

## LoRA可以一起划分，但不能漏掉非线性和完整down修正

采用W[out,in]、A[rank,in]、B[out,rank]：

```text
linear(X) = X Wᵀ + strength × (X Aᵀ) Bᵀ
FFN = H_G Wd_Gᵀ + H_A Wd_Aᵀ
down-LoRA = strength × (H_G Ad_Gᵀ + H_A Ad_Aᵀ) Bdᵀ
```

G/A为GPU/ANE的intermediate-channel分区。每份H必须先在gate/up的base结果
上加对应LoRA，再经过SiLU×up。down的两个rank partial先相加，只执行一次
完整B投影；不能漏ANE侧LoRA，不能把down-LoRA重复加两次，也不能把LoRA
移过SiLU。Q/K/V的A若不同，不能因为rank相同就共享一次AX。

当前Qwen/Z的`HybridFfn::Adapter`路径已经让base由GPU/ANE分担，GPU生成
gate/up修正，ANE在SiLU之前消费它们，并将corrected hidden交给完整down-LoRA。
这并不等于低秩A/B matmul已经都在ANE。Qwen的具体回调在
`native/models/qwen21/runtime_ffn_graphs.hpp`与`pipeline.cpp`；generic接缝在
`native/backends/ane_ffn.hpp/.cpp`。已有share/split-rank选项也不自动证明ANE
执行了低秩运算，应查真正program inputs/算子和actual calls。

可探索三种不同方案：

| 方案 | 灵活性/准备 | 需要实测的成本 |
| --- | --- | --- |
| 现有base-ANE＋GPU低秩修正 | 动态adapter、完整数学，无merge | GPU低秩、两份修正就绪、hidden恢复/完整down |
| 一次生成有效W，再作为runtime权重输入 | 换LoRA/strength需重算BA；无需把W烘焙进ANE程序 | 准备时间、峰值、重新量化与merge舍入；不能只报热图 |
| 将A/B作为固定ANE ABI的动态输入 | 按rank bucket补零可换adapter，不必按adapter编译 | 小GEMM效率、因子staging、FP16范围、同步与down完整归约 |

第二种的当前Qwen/Z runtime入口明确只准入`inference_time`，并不能仅换一个
request字段就当作已支持。第三种已有隔离FP16 capability emitter
`tools/native/qwen21_runtime_factor_mil.hpp`（rank32/128、ANE宽度≤2048）：base
与gate/up的A/B均为动态输入，但down-LoRA仍GPU。它不是W8A8整图，也不是
“全部LoRA已ANE”的完整图片性能证据。`native_ane_lora_contract.hpp`另有
rank128/down-A partial合同，亦不可据emitter存在便宣称已接入生产Session。

若尝试ANE down-A partial，仅返回小rank矩阵可少恢复一份完整hidden；若再把
down-B也塞进一个ANE图，图会依赖GPU-head rank，可能在launch之前先等GPU
head完成，反而失去base并行窗口。需要用完整消费者比较，不能按FLOPs推倍率。

本地训练Qwen适配器目前为Viggle v0.2.1 r128/r256。它们是蒸馏学生LoRA，
应在同一adapter/strength/六步/任务下做GPU与runtime对照，不能用base40作
分母。r128原文件SHA256为
`bafb91d0047df3f9b8a5a850b0c967f051164314d8aad778dfa34d9c24ec345b`。
尚不能代表任意风格LoRA；用户给定其它适配器时仍需检查目标投影、rank、
alpha、scale和encoder terms。旧Public/FP16负结果在各自原报告中保留，不用
历史更慢GPU分母替代当前direct-QKV/typed-SwiGLU强控制。

### 本轮已完成的r128完整请求

同V4 native、原BF16 base＋原r128 strength1、六步，seed73→74→73，各独立
进程三请求，无减步/temporal reuse。GPU使用actual original-buffer direct-QKV
及typed-LoRA/SwiGLU强路径；编辑为相同ref512，GPU真实命中跨请求prefix。

| 场景/路径 | 热中位数 |
| --- | ---: |
| 生成，强GPU | 7.2076s |
| 生成，Private R1024/F_ANE6144，独立ranks | 约7.759s |
| 生成，同6144，共享input-rank＋split-down | 7.8269s |
| 生成，Private R1024/F_ANE4096，独立ranks | 7.1391s |
| 单ref512编辑，强GPU/prefix hit | 7.3293s |
| 同编辑，Private F_ANE4096，无runtime-prefix | 8.6529s |

6144生成约慢7.6%；4096仅约快0.95%的微弱局部信号，不能称稳健加速。编辑
约慢18.1%，当前不能默认选ANE。所有runtime窗口每请求192个成功混合FFN
block、全部227项训练LoRA实际绑定、fallback/retry/rebuild0；共享试验每请求
实际192个rank producer sets、384个rank arrays，两侧消费者均完成，split-down
也覆盖全部block，但净时间未改善。原文件679604800 bytes由native注册/验证，
parent也核完整pin；不是用合成adapter或只有flag存在代替真实LoRA计算。

这些仍是GPU算低秩修正的既有runtime路径。下一项不应继续盲扫分区，而是用
真实输入与全部source检查ANE down-A rank partial能否节省hidden恢复成本；
先过完整组件/消费者，再决定是否修改Session。生成/编辑的完整PNG及source/
memory原件位于`outputs/qwen-expanded-lora-*-v9-20261019`，独立审计另列。

GGUF/ConvRot＋LoRA、普通风格LoRA与Turbo8＋LoRA没有由本轮自动获得资格。
尤其ConvRot现有source adapter仍拒绝runtime LoRA，不能混用其H256与普通
BF16路线的Sylvester基或绕开guard。

### 为什么LoRA混合更慢：融合边界、依赖和参考缓存要分开

不能概括为“所有fused都没了”。强GPU控制使用original-buffer direct-QKV及
paired LoRA-B/typed-SwiGLU专用融合；当前hybrid complement不经过这两条完整
GPU专用路径，但仍有compiled/fused运算。ANE每个chunk仍执行完整FFN，
不是每个低秩matmul都单独调用ANE。额外关键依赖是：

```text
GPU normalized X → gate/up LoRA修正就绪 → ANE(base+修正→SiLU×up→down)
                                                        ↓ corrected hidden
GPU complement ───────────────────────→ 完整down-LoRA → partial join
```

gate/up修正不能放到SiLU之后；完整down-LoRA必须包含ANE侧hidden。当前为此
需要回收hidden。主机async-wait包含真正计算、GPU排队和交接，不是纯“同步
开销”，也不能只凭包络断言物理ANE/GPU重叠程度。最后热请求的非互斥包络：

| r128路径 | pre-FFN | LoRA-ready（含attention） | FFN | async-wait | 暴露W-stage等待 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 生成C6144 | 2.896s | 2.860s | 4.281s | 3.951s | 约61µs |
| 生成C4096 | 2.777s | 2.736s | 3.773s | 3.352s | 约63µs |

不能把这些列相加。未看到W staging显著阻塞的证据，因此继续只改权重缓存
不是最有根据的下一步。编辑还有独立的额外成本：强GPU实际命中参考prefix，
当前LoRA runtime没有对应命中；每请求ANE预测从生成224增加到编辑256。
首prefill从1048行/64次/约1.345s变为2083行/96次/约2.526s。
余下decode为1024行/160次，约5.21–5.50s。先恢复合法prefix复用，再评估
ANE down-A rank partial减少hidden回传，才比笼统“加强融合”更可检验。

新V10b诊断仅在自己的进程将`TURBOCIDER_SOURCE_REGISTRATION_CACHE`设为空，
首次实际读全679604800 bytes、cachehit0/boundtrue，热请求读0 bytes、hit1。
未删除共享注册缓存，也未放宽严格source-proof。首个因注册cache命中而被拒的
诊断窗口保留，不计入合格窗口。V10b合并独立审计覆盖30窗/80 PNG/861实际步。

普通adapter的旧runtime实验六步域保持不变。V5增加默认OFF的
`TURBOCIDER_QWEN21_RUNTIME_BASE_LORA=1`，为resident512原BF16、单普通
transformer LoRA、20–40步base Euler提供独立研究准入；不允许Viggle/Turbo/
packed source/encoder ANE/PE/cache/内存配方混用。它不是任意LoRA性能资格。
硬件验证使用明确标记的合成非训练rank2 gate/up/down fixture，仅能证明准入、
非零绑定与完整采样功能；不得当作普通风格LoRA质量或真实adapter速度结论。

V5现已实际完成该合成fixture：GPU和runtime各完整20步、三项投影绑定；runtime
640个健康混合FFN block、一个实际ANE model、无回退/overflow/rebuild。
这是一次首请求功能smoke，没有热倍率资格。新增模式关闭时，原Turbo8与真实
Viggle r128六步的完整首张PNG SHA分别仍为
`119eee99f90a4e41c0212e6d93fc2297624ae33acd96184fc036724e03002643`、
`17b90aa88ddbed433511e54ebcc23e3762a2023bf3b51d0e97f726e749c7324d`，
与对应V4控制逐字节相同。这不是跨任意prompt的回归或性能资格。

另一个可检验方向是按adapter/strength生成有效权重`W+sBA`，再旋转量化写入
既有动态W槽。数学上可以同时覆盖gate/up/down的两侧分区，避免逐block的修正
依赖与hidden回收；换LoRA改变数据而不改变图。但不能省略首次merge/quant成本，
也不能将量化后的有效W当作原在线LoRA逐位等价。若缓存32层C6144的三投影，
光W8 codes便约2.25GiB，另有scale/metadata；这不是ANE model文件变大，而是
runtime数据驻留的代价。只缓存小bank则需要重复准备，必须实测摊销和带宽。
现有入口仍只准入在线LoRA；此有效权重缓存方案没有接入或测得收益。

## Z-Image与LTX不是复制一个开关

Z的`native/models/z_image/z_image.cpp`已经接入generic runtime FFN，H3840、
F10240，实际含2个noise-refiner和30个主block；支持在线LoRA修正。下一步可以
用本机原模型重做GPU/Private固定图与LoRA对照，保留全部8步及实际32个block，
不能沿用Qwen的source、F12288、quality或速度。现有入口仅文生图，不宣称编辑。

### 本轮Z-Turbo：功能已测，性能门槛未过，不提交

V5原Comfy BF16 checkpoint、完整8步、seed73→74→73。C5120固定R1024与
R1152均完成全部32层/步、fallback/overflow/rebuild0。R1024每请求496次预测
（image-only noise refiners与joint main blocks行数不同），R1152为256次；
不要把两层noise refiner漏出计数，或误以为所有32层具有相同行数。

同一R1152 owner完成512²→768×512→1024²→512²，每请求预测256/512/1024/
256；独立selector显示一个实际model、hot compile/load0，首请求另有3次能力
调用，回512同seed完整RGBA exact。所有token与8步保留，没有热换descriptor。
这证明一个固定FFN图可以处理这些长度，不证明每个size已采用最佳bucket。
例如Z的主block每步还处理caption，不能照抄Qwen只对target行的最优R1536。

但连续外部ComfyUI Python负载或load采样间隙使这些性能窗口全部未通过gate。
原raw wall保留：GPU512热约6.99–7.02s，ANE R1024约9.52–9.61s、R1152约
6.61–6.63s；变尺寸timeline约7.52/13.19/39.37/8.41s。它们不是干净匹配对照，
**不生成或宣布GPU加速比**，不据此更改默认/执行条件commit，也不自动推广到
长caption、Z-Image base/edit或任意参考数。共享进程未被停止，失败记录保留。

后处理独立审计`outputs/runtime-image-independent-audit-v2-20261019.json`覆盖
7个功能窗口、18张完整PNG、168实际步，包含前述合成LoRA但不含最初因错误
progress假设拒绝的pilot。已核对原模型/组件完整SHA、663输入V5 exact archive、
请求/进度、原始memory hash-chain、实际selectors及同seed恢复。显式
`--allow-contaminated-functionality`只恢复功能/图片审计资格，不恢复计时资格。
R1152对GPU两个seed RGB SSIM为0.98979/0.96682；样本目视检查仅覆盖seed73，
可见总体形状/构图一致而局部像素有变化，不能据此承诺所有prompt质量可接受。

### LTX：下一阶段，尚未接入同款runtime FFN

LTX本地2.5配置为48层、video hidden4096/audio hidden2048，现有C/Metal FFN
使用fc1→BF16 tanh-GELU→fc2，不是Qwen/Z的SwiGLU。当前
`native/models/ltx_runtime/ltx_ane_mlp.m`按artifact加载每层MLModel，
`ltx_blocks.c`为每层/两stage持有ANE MLP；IO复用不等于一个跨层runtime模型。
runtime迁移需独立的GELU两权重图、Video/Audio几何、bias、ConvRot H256、
scale、残差和跨stage工作集合同。Audio FFN含bias，不能沿用无bias的Video图。
本地原始LTX checkpoint/encoder/VAE已在，不需要先重复下载整套模型。

当前尚无本轮Z/LTX runtime整请求新倍率。不能将旧冻结ANE成绩当新runtime
成绩，或将短视频功能smoke替代完整11步/8+3采样、VAE/音频/mux的测量。

## Draw Things / vpipe：可借鉴的是具体边界

已只读检查本地`references/ccv@66dc16a3a3b688adfa6dc1d19eeeb3071beeb701`的
`lib/nnc/mfa/ccv_nnc_mfa_ane_rowwise_coreml.mm`。其生成器接受动态INT8的W/X，
使用固定DQ scale、可拆K的matmul，按(M,N,K)缓存program，按几何/dtype缓存
IOSurface-backed scratch，并检查Core Video可能增加的pitch。W并未作为每层
模型常量；改变源权重仍须正确更新生产者及缩放，surface命中不等于内容命中。

与本地`vpipe@18ddb3b5bd9c74ab7f2b4408ee78efc6cecd48e7`的固定chunk/row-split
相比，TurboCider当前Qwen主要是intermediate-channel split。前者避免尾部padding
但短序列可能没有合法ANE chunk；后者可分担短序列，却有两个partial与join，
不平衡/补齐可能抵消收益。现有Qwen已有GPU/IOSurface交接和双权重bank，不能
把“增加共享内存”当作尚未实现的神奇优化。

这些是固定源码机制对照，不是本机重跑Draw Things/vpipe的速度排名，也不
称上述commit为最新。源代码的int8张量/DQ图或CoreML compute-unit设置，均
不单独证明物理INT8 MAC、零拷贝、无驱动内部compile或物理GPU/ANE重叠。

### Splash：同model多固定procedure，比仅缓存多个owner更贴合目标

只读核对官方`incoai/splash@be3b083c5f413673d44e44881042e875e674e333`
（commit时间2026-10-14），不是凭旧记忆推断。`runtime/ops/AneFfn.cpp`的
`functionRows`/`function`/`program`把多个固定rows的`ffn<rows>`放在同一个MIL
program；构造时创建一个`ane::Program`，按函数名称解析真实procedure。
每个procedure有双bank binding，输入/partial复用最大工作区；`encode`选择
第一个足够大的行数。`runtime/ane/Program.mm`读取model attributes的procedure
ID与input/output symbol映射，request传实际procedure，而不是假设所有函数
都使用0或相同symbol index。GPU/ANE shared-event handoff仍需要正确的生命周期。

可借鉴的实现顺序：固定rows集合（例如512/1024/1536/2048）、最大IOSurface
带真实pitch合同、按名字解析procedure与symbols、预建双bank binding、只选择
procedure和tail padding；换adapter/层只更新动态W，不改图。热窗仍要独立检查
model身份、compile/load、失败退路和端到端收益。生产FFN仍使用单procedure。
后续新增显式研究构造器，已经真实验证同model双procedure、共享最大pitch和
32→64→32 A/B/A，详见[基础能力报告](runtime-ane-multi-procedure-2026-10-19.md)。
后续V7已增加W8 SwiGLU后端显式入口，并完成H128/F512的完整合成消费者与
Exp/Complement独立复算；尚未测真实Qwen完整FFN或接入生成Session，没有
新的图片速度或多procedure收益声明。

相似处在于FFN对token逐行独立，可以把不同size/参考数转化为不同row数与
chunk/tail；不是把diffusion整个attention当作LLM prefill。Qwen的静态参考
prefix可复用依赖其特定mask；目标noisy tokens、Z的统一caption/image attention
不能任意套用跨步KV缓存。图像比例还影响位置编码/attention/VAE，单看总token
相同不足以宣布全模型无重编译。硬件代际与LLM吞吐数字也不转移到本机M4 Max。

## 验证顺序与可接受误差

先完成Qwen各size固定图的stock/reverse/selector/PNG审计，再做同LoRA强GPU
与Private runtime对照；只有具体边界有净收益，才接入更多native因子计算。
随后给Z同样的完整请求工程，再实现LTX独立runtime GELU原型。

用户允许一定误差，因此不把逐位相同当W8采用前提；但SSIM也不等于语义质量。
要检查文字、身份/形状、材质、构图、编辑指令及透明边缘，保留可查看的GPU/
ANE样本。比如当前ConvRot seed73壶嘴有可见变化，不能代用户断言可接受。
每个候选均记录source/LoRA身份、sampler实际步骤、首次准备、热请求、实际
compile/load/calls、fallback与内存，不能为“更快”省LoRA、token或步骤。

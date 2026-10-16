# 真实Qwen共享FFN：完整五臂消费者与实际join绑定

[总目录](../README.md) · [完整FFN边界](../gpu-ane/private-ffn-boundary.md) · [backing机制](../gpu-ane/shared-surface.md) · [全部样本/非执行记录](qwen-shared-ffn-evidence-20261016.json)

2026-10-16，M4 Max/64GiB、macOS26.6.2/build25G83、SDK26.5，已安装MLX0.32.0/实际architecture applegpu_g16s。C++20独立客户端不链接产品库；沿用原BF16 checkpoint、forced-GPU capture与原BF16 FFN replica reference，layer0/base/no LoRA、M1024/H4096/F12288，冻结private share2048。原模型/参考checkout/App和产品默认均未改。

## 1. 从真正二维MIL端口接入，不拼接两个toy

先对已有且来源已核对的MIL/blob在新owned目录做[严格文件clone](../sources/official/clonefile.md)，实际compile/load/unload，**不执行FFN**。读到的input/output都是FP16 **`[1,1,1024,4096]`**，row8192、plane/batch8388608 bytes，而非前面小图的`[1,C,1,S]`。这限定于该二维matmul MIL/本机版本，不是所有ANE图的映射定律。

后续四个新进程真正运行完整消费者：

- 完整GPU：原始BF16 input、full gate/up、SiLU×up、down、完整BF16 output，compiled MLX；所有112次GPU输出对保存BF16 replica全元素误差0。它是强公式replica，不声称测到未改Transformer大kernel内部。
- copy-serial / copy-hybrid：同一`_ANEClient`、相同图/常量和GPU pack；额外两次CPU feature copy，分别把GPU staging FP16输入送入模型、把FP16模型输出送到独立Metal buffer，合计16MiB/call。
- shared-serial / shared-hybrid：GPU直接BF16→FP16 pack到实际input IOSurface；private同步evaluate；实际output surface严格导入MLX view作为**compiled join运行时参数**，完整GPU remainder和FP32恢复/相加后cast BF16。

五臂同一group/point，十行循环+反向顺序平衡位置。边界都是**ready原始BF16 GPU input→完整BF16 GPU output ready**；GPU pack、CPU copy/lock、private调用、suffix、恢复/join与drain计入。prepare、常量compile/load、poison与完整CPU验证在外，下一arm前排空验证转换。不比较单个execute数字，不把serial当GPU分母。

这个copy控制不是旧e5rt/NumPy bridge：private入口和输出恢复方式不同，旧桥接有host FP32 partial；本轮相同client控制传FP16，再GPU cast。不能把两批倍率直接归因为少了一项copy。

## 2. 全部560次与两版协议

四个成功运行分别65、165、165、165次，共560次完整调用：每run每臂first1/warmup2，measured10或30组。前两个真实point循环为`1→19→10`与`19→10`；后两轮`19→10`和`10→19`。这是相同轨迹下新进程/反序重复，不是未见prompt/seed/层的hold-out确认。

全部first/warm/measured通过固定relative L2≤0.03、cosine≥0.999：完整输出对原BF16 reference、完整private partial对BF16 prefix reference，以及独立full/prefix FP64抽样。全输出最大L2约0.002667，full FP64 sample约0.006079，完整prefix约0.004585，prefix FP64 sample约0.011051。FP64每个实际row保留完整K/F，但只选4row×4outputcols，不是全输出FP64或画质资格。

最初两轮没有额外exact join witness，历史不回填。后两轮对264个完整candidate output，独立CPU按**实际GPU tail＋实际private partial**做FP32加法和显式BF16 RNE，全4194304个元素逐位比较：全部零错误、完整hash相等。不是只看3%预算能否遮住相邻timestep的旧partial。

另64项负控把上一实际point的真实shared-hybrid输出对下一point的实际join期望比较，每项4189812/4194304个元素拒绝；只是CPU检查旧真实结果，没有冒充新GPU dispatch或计性能。**join位级一致不等于模型无损**：两个分支本身仍有BF16/FP16混合精度误差。

每个candidate前后实际request inputArray/outputArray返回surface ID与目标一致，Metal contents与实际base一致，MLX restore view与正确buffer一致；只发布bool/hash，无raw ID/地址。四run均完成所有owned GPU/MLX/private调用、两份restore view deleter恰好释放一次/份、unload后释放owned mappings。没有实际取消/失败注入或物理ANE counter资格。

## 3. Warm完整边界结果，保留方向不同的观察

| run / 协议 | GPU ms | copy-hybrid ms | shared-hybrid ms | GPU/shared paired中位 | copy/shared paired中位 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 三point/10组，原协议 | 21.610 | 19.090 | 19.345 | 1.11718× | 0.99038× |
| 两point/30组，新进程重复 | 21.601 | 18.995 | 18.810 | 1.14900× | 1.01154× |
| load入口/30组，exact join | 21.608 | 19.055 | 18.858 | 1.14575× | 1.01142× |
| 反向point/30组，exact join | 21.632 | 19.086 | 18.853 | 1.14715× | 1.01097× |

两个exact-join run的GPU/shared探索性paired bootstrap区间分别[1.14396,1.14766]、[1.14452,1.14958]；copy/shared约[1.00753,1.01429]、[1.00780,1.01327]。这是同机有序重复的探索区间，不授予长期独立样本或正式显著性。

长确认shared约1.146–1.149× GPU，相当于本**warm FFN**边界约12.7%–13.0%延迟下降，不是整次生成下降15%。同client copy→shared只有约1.1%局部观察，首轮三point方向相反；不能将全部GPU收益归因于零外部copy，或从此自动晋升通用share/profile。serial两臂约26.5–27.0ms，都慢于完整GPU；serial→hybrid约1.4×另列，不借作GPU提速。

实际private与GPU suffix的同process host窗口相交；这是host依赖证据，不是物理放置/并发。`external_feature_copy_count=0`只属于shared计时边界；初始化/poison/验证仍访问CPU内存，runtime内部复制未知。逻辑16MiB copy量不是DRAM流量counter。

## 4. 存储admission、复用与setup不消失

新增clone/compile流程要求256MiB free；一次后续尝试在约254MiB被拒绝，未启动native/device，记录保留，不能叫硬件失败。另设**不同的只读prepared/load工作流**：不新建常量clone、不调用compile selector、不fallback编译，128MiB门槛仅覆盖该流程；原256MiB门槛没有降低。

后两轮复用上一已验收owned MIL/blob/manifest，内容前后hash相同，原始source spans重新核对。compile selector未调用，compile duration保持null，不伪造0。两次load却分别约**650.99ms、32.11ms**；前两轮显式compile约510.83/534.42ms、load20.51/21.35ms。系统cache未清，不能从selector名字推出内部无编译或固定cold load成本，也不能说load-only优化已赚回setup。

GPU逻辑权重共存576MiB、private逻辑权重48MiB；未测逐arm独立系统峰值、低内存或长时热状态。没有计完整wrapper/CPU FP64 witness/GPU reference preparation为请求wall，也没有按32层/LoRA做常量cache失效实验；cold request可能否决warm组件收益。

## 5. 取证、复现与采用范围

[可选集成](../integrations/README.md)保存native/client/prepare；核心[独立合同与RNE](../labs/real_shared_ffn.py)、[data verifier](../scripts/curate_real_shared_ffn.py)、[反例测试](../tests/test_real_shared_ffn.py)只需标准库。curator独立复核原BF16 checkpoint/capture、fixture及NPY行pitch上的native prefix、MIL/blob、全部per-run source/binary、383 headers/三library与owned退出receipt，重算完整顺序/gates/clock依赖/ratio。

curator不重跑设备、private输出或FP64数学；local oracle vectors仅绑定manifest/source hash，不发布数值。exact join的独立CPU计算实际在native采集时执行，verifier只复核其受源码约束的计数/hash，不是硬件attestation。早期inspect与两轮无exact witness的源码保持原身份；当前CLI只额外拒绝“prepared replay却未显式选benchmark”，不为旧run换源码标签。

本批采OS build/SDK/MLX assets与源码编译参数，没有另在每份wrapper中采NumPy/clang版本或dyld actual-loaded-image attestation，不回填旧记录。原始模型/激活/F64向量/机器路径保留ignored目录，公开只含hash、数值误差、范围与相对host区间。没有自动commit/push。

保留为**实际shared layer0完整warm FFN候选**。仍未完成：完整block/request和图像质量、更多层/独立prompt/seed/LoRA、冷启动/常量bank/load复用、逐arm峰值/低内存/长时、取消/失败回收与物理placement/overlap。也不覆盖以前Public CoreML/affine/attention的负结果；见[完整工作台](research-campaign-20261015.md)。

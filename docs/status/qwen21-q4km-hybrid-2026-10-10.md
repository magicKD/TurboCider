# Qwen Q4_K_M GPU/ANE：40步正反序约17%收益，目标仍未完成

2026-10-10，M4 Max64GB/macOS26.6.2。接续
[混合GGUF首版](qwen21-q4km-support-2026-10-10.md)。两份原Q4_K_M文件和
原BF16 VAE保持不变。本轮实现实际GPU/Private ANE FFN通道分工、显式
GPU down-kernel候选与Q6_K导入SIMD，未将完整目标缩成组件通过。

## 通道并行接入与完整GPU恢复

全部32层FFN保留原immutable affine codes/scales/biases。fused gate/up
按输出rows切两份时，三个平面一起切，明确bits4/8、group32和FP16
metadata。`stage_weights`消费真实packed source，不将UINT32解释成dense。
FFN的norm/attention仍按依赖在GPU执行，不是整步全部交ANE。

GPU范围和完整fallback均用原packed QMM，不能退回旧dense matmul而漏掉
coefficients。borrowed-source scope在正常/异常路径drain。只允许明确
approximate GPU或固定正Private通道的base生成，原源/finite/shape/ledger/
memory guards保留。当前FP16 mixed-K路线拒绝仅BF16-qualified的F32-join，
GGUF encoder/LoRA/edit hybrid和Public完整性能覆盖仍待推进。

## 实际40步正反序

512²、40steps、seed29、三个同源fox fresh提示词，只改natural/soft/warm
winter light；每臂独立process、一cold两warm，conditioning全miss。
encoder为相同admitted retained原GGUF source，warm实际reuse=true。
固定异步、c1056、GPU IOSurface I/O、W8A8 Hadamard、cache/prefetch/
lookahead关闭；后续39步确实参与分工，不使用时间复用。

| window / route | warm request s | warm encoder s | warm DiT s |
| --- | ---: | ---: | ---: |
| forward / complete GPU | 44.550319 | .231413 | 43.730452 |
| forward / ANE4096 + GPU8192 | 38.608917 | .233109 | 37.776076 |
| forward / ANE5120 + GPU7168 | 37.009189 | .231974 | 36.176666 |
| reverse / ANE5120 + GPU7168 | 36.968787 | .227882 | 36.148263 |
| reverse / complete GPU | 44.576176 | .233911 | 43.725750 |

5120整请求对各自GPU名义耗时少16.93%/17.07%，约1.20–1.21×；4096首轮
少13.34%，但只有首轮，不说有同任务反序资格。reverse使用加入default-off
kernel开关的新build，但开关0，与forward库不互称同一binary；两个window
各自用自己的GPU分母。推荐继续显式5120候选，尚非全局自动calibration。

每hybrid request实际1280次Private predictions/成功channel blocks，phase
分别32/1248，三请求累计1280/2560/3840。failure/fallback/retry0、headroom1，
actual async blocks匹配，FP16 join和实际GPU IO明确验证。encoder未转ANE。
图像查看forward soft-light整图GPU/5120：狐狸姿态/脸、毛色、雪林构图
非常接近，仅细微毛纹/背景差别；仅这些scene/seed，不是全面质量验收。

forward各臂tree peak约16.18–16.51GB，swap-in/out0。reverse5120 peak
16,515,850,472bytes，system swap-in2,097,152bytes、swap-out0；GPU peak
16,087,474,312bytes，swap0。不将系统counter归因单个kernel/模型，不称
整机RAM上限或GPU/ANE物理overlap。没有连续quiet-load正式资格。

## GPU kernel：真实shared-word decode/MPP down，额外收益仍小

新默认关闭 `TURBOCIDER_QWEN21_GGUF_SHARED_DOWN=1` 限制approximate
resident512 mixed-K。在原packed FFN down把每个word typed解码一次到
有界TG tile，四SIMD groups共用MPP matmul；不建立全矩阵dense W、不
改gate/up、attention或其他模型。partial range使用原physical pitch/
offset，不用切片endpoint冒充完整K。cold bank赋值会替换Weights，故
recipe snapshot在实际加载后安装；切换drain/清prefix graph，防旧trace。

原layer0真实Q4 down、合成FP16 X、3warmup/9cyclic hot，M1024/1048，
full K12288和partial K8192的actual evaluated projection spans：

| M / K | 原QMM ms | shared MPP ms |
| --- | ---: | ---: |
| 1024 /12288 | 7.716459 | 7.659334 |
| 1024 /8192 | 5.582959 | 5.240333 |
| 1048 /12288 | 7.954416 | 8.226958 |
| 1048 /8192 | 5.734334 | 5.615166 |

四个当前synthetic cases输出对QMM relL2=0，但部分shape不盈利；不是
GPU timestamps、所有shape或完整模型保证。576个实际Metal原typed
Q4/Q8、FP16/BF16、offset/pitch/尾部/strided/narrow/F32 cases及108个
malformed contracts通过，不能把研究数值测试当整请求收益。

同新build的后续40步screen，kernel开：GPU44.515447s、5120混合36.617037s，
DiT43.689806/35.776378s；与前一同build反序kernel-off共用route各自比较，
名义混合额外少.95%，GPU仅.14%。这不是四臂同一window ABBA或正式
稳定倍率，保持default-off。所有完整样本/PNG保留，不拼各臂最佳phase。

## Encoder冷加载与Q6 SIMD

Q6_K affine import新增ARM NEON六位解码、signed subscales、group-max、
固定ties-to-even和clamp；仅128bytes stack scratch，finite typed metadata/
cancel/alias/precision guards不变。CPU108场景的scalar/SIMD fields逐字节
一致，旧exhaustive affine回归通过，native旧packed-bank source/lifetime/
cancel/floor回归也通过；mixed bank backend标签更新为实际ARM NEON。

新build4步cold screen：GPU text4.227717s、5120 text4.184059s，前一scalar
5120 cold5.141165s。同source/逻辑recipe，但不是同window causal A/B；
不能全归因SIMD。warm约.214/.224s，完整请求5.225053/4.464201s。
首次提前启动尚未链接的CLI在source/output前失败，failed runner保留；
等同一build handle terminal后另开v2完成，没有重启live build。

冷encoder总阶段仍包括首次native full-SHA/source导入，不能拿.22s compute
差额与对方whole-condition比较后宣告冷端到端目标完成。仍需要减少
重复source读取、对齐load/compute与更多场景及权重生命周期。

## Unsloth同40步与完整缺口

只读原Unsloth args builder，固定其u1d02858包、相同GGUF/BF16 VAE、
512²/40steps/seed29/cfg1、max FA/direct-conv，默认Apple clip-on-cpu。
真实model child完成，condition3.37s、sampling109.95s、generation131.68s，
tree peak10,587,955,264bytes、memory complete/system swap0。Native DiT
约36s有明确方向信号，但它的source加载发生在denoise计时之前，对方
sampling可能包含首次lazy参数加载；不能直接推广成跨所有冷/热状态的
3×保证。Native warm encoder约.23s与其cold3.37s也不是匹配warm对照。

**未完成项保持原目标：** 同预算BF16真实streaming实现/内存与整请求
比较；encoder冷端到端超过最强matched Unsloth；更多scene/seed、quiet
顺序和完整阶段/质量审计。不会用较小尺寸/更少步骤代替streaming，不
把仅功能接入或当前热态优势当完整目标完成。

工具 `qwen21_gguf_hybrid_screen.py` 验证实际typed counters、phase、source/
encoder retention、成功prediction/GPU IO，拒绝fallback/重试/伪造channel；
支持同一参数的order和显式shared-kernel开关，qualification始终false。
本轮33个native requests与11份memory evidence、全部原source/PNG hashes
独立核对通过，包含4步smoke/NEON与三个40步windows。

[机器记录](../design/validation/qwen21-q4km-hybrid-20261010.json)绑定完整
raw路径、库/源、样本、内存、kernel及未完成要求。模型/adapter/ref与参考
工程未改写，所有logs/PNG/build artifacts不入Git，完整目标保持active。

最后确认owned模型/测试/编译全部terminal，清理本轮三个isolated build的
654个可重建`.o`、51,786,048 logical bytes及三个空module-cache；library/
CLI/probe和全部证据保留。最新NEON库503个source inputs匹配当前tree。
4项CPU/host（含108 typed K scalar/SIMD cases、旧exhaustive affine）通过，
最终host counter负例2项通过；1项实shared-kernel unit（576/108cases）和
1项旧packed-bank实Metal回归通过，无skip。未做全部产品矩阵验收。

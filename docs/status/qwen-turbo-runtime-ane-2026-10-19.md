# Qwen 2.1 Turbo、encoder、GGUF与变长 runtime ANE

2026-10-19，Apple M4 Max/64GiB、macOS26.6.2。此页为持续探索记录，整体目标
ACTIVE。本页只记录Qwen Image 2.1生成/编辑；后续追加的Z/LTX另见
[扩展研究](runtime-ane-lora-and-model-expansion-2026-10-19.md)，不混用模型结果。
所有速度是本机有限窗口，不代表独占、thermal或语义质量资格。App/default不变。

## 已完成的Turbo支持

从ModelScope `Comfy-Org/Qwen-Image-2.1`固定revision
`8c798285d642b81a1af415b70f2e444e809820d7`下载原文件：

- `qwen_image_2.1_turbo_bf16.safetensors`，14230280616 bytes，SHA256
  `8b74237a7f505773e2c66726fa696444b4c704585c93a347cff9044937f7914a`。
- 原完整文件发布前验证size/全SHA，无第二份Hub cache，原base权重未覆盖。
- 安装`models/Comfy-Org-Qwen-Image-2.1-Turbo`，使用原Turbo文件名、只读共享的
  encoder/VAE/processor链接。没有把Turbo改名成base；base根目录仍优先选原base。
- Native/CLI使用`model: qwen-image-2.1`、`model_variant: qwen-image-2.1-turbo`、
  `steps: 8`。显式variant与实际checkpoint不符会拒绝，不静默换权重/采样器。

采样采用作者`Qwen/Qwen-Image-2.1-Turbo@fc13ba1986fcb905854dc8de2f52eee395f49b5b`
的model_index/scheduler元数据：CFG1、shift1、dynamic shifting关闭、terminal
stretch关闭，最终sigma为
`[1,.978453,.95418,.926626,.89508,.845148,.704534,.414568,0]`。
这是固定8步，不是base Euler少取8步，也不是Viggle六步LoRA。512²、768×512、
1024²节点与实际GPU执行的测试通过。暂不将Turbo与base/student LoRA、DiT cache、
PE、streaming/constrained路线混用；编辑初始范围为1–2参考图。

## 完整8步、同模型stock对照

每臂fresh process，同library/权重/任务/seed73→74→73，一首请求、两次热请求。
无减步/LoRA/temporal reuse。单位秒，热中位数：

| 工作负载 | GPU | runtime ANE FFN | 结论 |
| --- | ---: | ---: | --- |
| Turbo512生成 | 9.1142 | 7.3081 | 约少19.8%，约1.247× |
| Turbo512单参考编辑，未启用runtime prefix | 10.4825 | 13.3000 | ANE反而慢，不能宣称编辑加速 |
| 同编辑，启用runtime prefix | 10.4825 | 8.2752 | 约少21.1%，约1.267× |

生成的GPU/ANE各3张完整PNG、24步；最初编辑对照也完整执行。GPU已有跨请求
参考前缀复用，而旧Turbo runtime不在base40的prefix准入域，热请求仍重复
长prefill。这是实际找到的瓶颈，不是把所有额外等待归给ANE硬件。

新增明确限定的Turbo8 runtime-prefix准入，保持原fixed1024/C6144、F32 join、
sigmoid-complement、健康/失败/drain规则。独立Turbo recipe标签隔离K/V；不扩展
原Private QKV/base40前看域。miss后保留2165833728 bytes已完成K/V；命中后实际
FFN预测416→256/请求，仍完成全部8个sampler步骤。开启/关闭3组完整RGBA全部
逐位一致；反向关闭热13.2582s，保留信号。首次on16.8366s单列，不称OS冷缓存。

prefix独立timeline完成3请求，一个稳定ANE model，hot compile/load0；其时间
约8.34s仅诊断，不混入stock分母。没有把base40→Turbo8的少步收益算成ANE倍率。

W8路线相对GPU本身仍近似：两个生成seed的RGB SSIM约0.9564–0.9936，最初编辑
约0.9961–0.9978，alpha最大差分别5/1。前缀优化的增量RGBA exact不等于W8对
GPU无损，不代表其它prompt/文字/身份/材质质量。当前不默认采用。

## 同一个固定ANE图服务不同输出尺寸

同一engine、同一FFN1024/C6144图：512²→768×512→1024²→512²；全部8步、相同
seed73，GPU与ANE各四张。以下是timeline诊断wall，尚非各尺寸stock重复资格：

| 尺寸 | GPU | ANE | 观察 |
| --- | ---: | ---: | --- |
| 512²首次 | 10.2015 | 9.5499 | 首请求单列 |
| 768×512 | 13.7870 | 13.9808 | 没有获益 |
| 1024² | 39.3168 | 37.5205 | 本窗口约少4.6% |
| 回512² | 9.0410 | 7.2206 | 原输出恢复 |

一个实际ANE model，所有热请求无compile/load；每请求workload预测增量为
288/512/1056/288，首请求另有3次能力验证。实际完整FFN block仍各256个。
前缀/目标行数增加只改变固定chunk调用次数，不为每个layer/sequence重建模型。
1024²进程树峰值约29.57GB（十进制），swap-in/out0。

768目标1536行，1024-row图需两块、其中padding512；GPU分支仍处理真实1536
行，负载不平衡可能抵消offload。新增固定512-row描述符供后续独立进程筛选，
不把“更小模型”预设为更快，更不在热路径按尺寸换图。此处保留原时间线结果；
尺寸特定stock重复的新结果见下节。更多参考图不因文生图变尺寸资格而自动通过。

### 后续：按尺寸选择固定图，完成stock重复

同一V4库、原Turbo BF16、8步、seed73→74→73，每臂独立初始化，GPU和ANE
均包含完整encoder/denoise/VAE/export，不使用LoRA或少步分母。

| 尺寸/重复 | GPU热中位数 | ANE热中位数 | 行数/ANE宽度 |
| --- | ---: | ---: | --- |
| 768×512首次 | 13.77905s | 11.15577s | R1536/C6144 |
| 同尺寸独立重复，尾部GPU控制 | 13.79563s | 11.03260s | R1536/C6144 |
| 1024²首次 | 39.30021s | 35.76655s | R2048/C6144 |
| 同尺寸反向重复 | 39.30791s | 35.79471s | R2048/C6144 |

768约1.235–1.250×；1024²约1.098×。R512/R1024在768只有约0–2%信号，
1024²的R1024约37.4532s，均不如表中候选。每请求实际256个混合block，
R1536/768预测288次，R2048/1024²预测544次，首请求另有3次能力验证。
两组独立timeline各一个实际ANE model，hot compile/load0，actual bucket与
descriptor一致，没有截断token。这里的“按尺寸”是在owner初始化前选择固定图；
尚未实现预热多size池或同model多procedure，不宣称任意热切换均无新load。

相对GPU的两个seed RGB SSIM：768为0.947363/0.984775，1024²为
0.993844/0.995868；alpha最大差分别6/4、3/2。仅两个seed不代表语义质量资格。
最终独立审计`outputs/qwen-expanded-size-lora-audit-v10b-20261019/audit.json`
覆盖30成功窗口、24配对、80 PNG、861实际步，绑定每窗真实工具版本、native
源码归档、完整原模型/adapter SHA、原始memory hash-chain、进度与selectors。
单步功能smoke不参与完整采样速度对照。

## Encoder：实现可用，单参考没有净收益

每次换neutral/warm/cool指令，conditioning全部miss；不能用cache hit冒充新的
encoder执行。仅语言36层FFN offload，视觉27层、attention和其余运算仍GPU。
两组相同source策略，均完成实际8步编辑。热text_encode中位数：

| source策略 | GPU encoder | ANE1024 | ANE512 | ANE1152 |
| --- | ---: | ---: | ---: | ---: |
| 原17.53GB source保留 | 1.4840 | 1.9180 | 1.9902 | 未测 |
| request-local、编码后释放 | 1.5257 | 2.2299 | 未测 | 1.8154 |

1024/512/1152分别72/108/36次实际预测/encoder。标准单参考约1070语言行，
固定1152避免两块1024或三块512的padding；它改善候选约18.6%，但仍比GPU慢
约19%，完整请求约17.1087 vs16.7705s，不能将候选内部改进称作GPU净加速。

保留source窗口进程树峰值约38.8GB，并记录约15GB系统压缩/13GB解压活动。
反向GPU对照发生source逐次重载（loads1→2、reused=false），严格collector在
第二张后拒绝该驻留对照；失败原件保留，不计成三请求成功窗口。没有强制放宽
内存准入或停止ComfyUI。request-local两组峰值约31GB，压缩0、swap0，仍未盈利。

1152候选最后一次request的host包络约FFN0.56s、pre-FFN0.92s、stage work0.35s、
exposed stage wait0.000016s。stage/等待/上游GPU工作可能重叠，不能求和当作
硬件算子时间或唯一归因；补充staging/边界/竞争可能抵消FFN局部节省。目前
此单参考场景保留GPU encoder，长序列/多参考与其它backend仍需另测。

## GGUF：当前完整请求有收益

原pinned Unsloth Q4_K_M DiT＋Qwen3-VL encoder、原BF16 VAE，512²/base40、
三fresh prompts，实际保留压缩encoder、native fast fused-prefill默认保持，
不是选一个较慢CPU encoder作分母。混合K的Q6→affine Q8等来源语义保留，
GPU/ANE均使用原导入三平面，不展开完整BF16 checkpoint。

| 路线 | 首请求 | 热中位数 | 热denoise约 |
| --- | ---: | ---: | ---: |
| GGUF GPU | 50.1712 | 44.4674 | 43.7873 |
| Private FFN C5120 | 40.3546 | 36.9587 | 36.2695 |

同格式/同40步约少16.9%、约1.203×，不是Turbo或少步倍率。FP16 typed IO、
F32 channel join关闭是此GGUF域的独立数值配方，不借用BF16/F32-join资格。
每请求1312实际FFN预测，fallback/retry0；独立3请求timeline一个稳定model，
hot compile/load0，stock窗口无swap/压缩。当前只支持GGUF生成，encoder hybrid/
editing仍有独立guard，不能宣称所有GGUF格式/操作均已加速或质量合格。

## ConvRot：正确接线后的完整请求结果

已从同ModelScope仓库下载原base `qwen_image_2.1_int8_convrot.safetensors`，
7256783064 bytes，SHA256
`cb74113cb03faecd79611b01fd7fd642f0aa60d6f0b95086abee214d75eaa57d`。
安装独立`models/Comfy-Org-Qwen-Image-2.1-ConvRot`，不覆盖原base。

实际文件的FFN为原signed I8、每输出行F32 scale、Comfy H256基；它不是普通
BF16或Sylvester-H128/H512输入。新接线保留原codes（q+128）、F32 scale/bias
的group128布局，使用显式ComfyH256Inverse source views；Private选择独立
`convrot_w8a8`、fixed-program和F32 join。H256-aware staging/F16 carriers本身
仍近似，不因源metadata为F32便称ANE所有算术保留F32或物理INT8 MAC。

这里的ConvRot确实是Hadamard rotation→A8→ANE W8A8：输入X由GPU stager做
Comfy H256再量化；ANE gate/up计算后，SiLU×up的hidden在固定图内做H256再A8，
然后down投影。checkpoint的W已经在对应H256基下保存为I8，直接使用原codes，
不重复旋转/重新W8量化，也不先展开整份BF16 W。每处rotation的基/符号/归一化
必须对齐，不能混用普通Sylvester配方。原GPU对照也使用该checkpoint的H256语义。

原GPU F32 partial helper只接受activation/metadata同dtype；不能为了接线就
把F32 row scale静默改为BF16。新分支保留F32 coefficient并使用已有F32 Q8
operator，在普通BF16输出cast之前取F32 partial。真实新库的tail/非零range/
H256-source-view测试通过。新I8 dot→F32 row-scale后乘GPU原型虽过数值筛选，
但最快4.51ms仍慢于strong ready-F32 4.20ms，未采用、没有将其当整图收益。

V3完整GPU成功；ANE早期两次失败原件保留：complement激活不在Comfy准入域，
修正为Exp后又暴露真实metadata布局错误，首张40步实际全GPU回退。二者不算
ANE加速。实际private-stage合成回归复现了同一geometry错误。

V4修正MLX惰性split求值后才显露的广播stride：先求值，再检查/有界物化F32
scale/bias，不复制codes/完整dense模型。仅保留当前/未来两层共12份metadata
plane的LRU，保留原producer generation和稳定copy身份，使实际预取能够命中。
同一准备函数用于ordinary stage和prefetch；新分配重新做内存准入，失败/释放时
清空缓存。真实Private ANE staging/launch/future-bank hit测试通过；单步原权重
smoke完成32个混合block、64次预测、fallback/retry/rebuild0，不用于性能/质量。

V4同库完整40步、seed73→74→73，各三请求的stock配对已完成：

| 原ConvRot base40、512² | 首请求 | 热中位数 |
| --- | ---: | ---: |
| GPU | 55.7042s | 53.9182s |
| H256→W8A8 ANE C6144 | 41.7662s | 39.6599s |

本窗口约少26.44%、1.3595×；不是与BF16/base不同checkpoint的比较，也没有
减少40步。每请求1280个混合FFN block、1312次实际预测，fallback/retry/rebuild0。
独立三请求timeline显示1个实际ANE model，首次compile/load各1，后续均0；每热
请求1312次实际evaluate，首请求另有3次能力调用。初始compile约1.07s单列，
不称冷启动零成本或不可见driver内部绝不编译。完整RGBA/来源审计另列下节。

四窗10张完整PNG、361实际步的独立审计通过：
`outputs/qwen-expanded-convrot-audit-v4b-20261019/audit.json`。其中单步smoke只算
功能，不混入40步速度。全部原模型/encoder/VAE全SHA、663输入exact native
归档、原始请求/PNG/进度、内存hash-chain与实际selectors核对完成。

相对同checkpoint GPU，两个seed RGB SSIM约0.95535/0.99458，alpha最大差1；
seed73的壶嘴等局部几何可见变化，不是无损。stock/timeline三组完整RGBA逐位
相同；同seed回归也exact。近似本身不因重复性/observer一致性便获得语义质量
资格。当前不默认采用，后续比较group256 A8、BF16边界等独立固定配方的成本/
误差；不得在热owner中更换配方后声称同一图无重编译。

GPU/ANE stock峰值footprint约15.30/15.52GB（十进制），swap-in/out0；GPU
系统压缩/解压约0.38/0.93MB，ANE约0/0.066MB。ANE权重槽165707776 bytes，
graph估计396361984 bytes；它们不等于包含checkpoint/encoder/VAE的整进程
内存。metadata缓存最多12份plane（本形状18MiB），不驻留32层的ANE权重模型。

实际stage/预取命中/源metadata换代拒绝、新库lifecycle及采样/策略等20项回归
通过；另7项CPU审计测试通过，含跨stock/timeline只能用`--quality-pair`，
禁止混用它们计算速度。首个审计误用普通matched pair对比stock/timeline被
严格拒绝，失败日志保留；新增明确quality-only入口后重新审计，没有放宽
source/实际工作要求。测试初期BF16 scalar读取dtype和未启用prefetch的fixture
问题也保留日志，只修fixture，不借失败窗口宣称native性能。

此前未采用的factorized GPU-dot原型亦完成独立四窗source/完整输出/F64见证
审计：`outputs/qwen-convrot-factorized-independent-audit-v2-20261019/audit.json`。
该审计对完整F64 oracle加显式finite gate；Accelerate matmul仍报告浮点状态
warnings，但实际所有oracle/输出/误差均有限且过阈值。此项仍只是ready-dot
负结果，不将它当rotation/FFN/ANE或图片加速。

当前V4库`build/qwen-expanded-convrot-v4-20261019/libturbocider.dylib`，SHA256
`b1e9c733e5eb0b1cf65acddcd64d8b93e2b2728ac22807258cd75d73856d7cff`，runtime ID
`tc-runtime-build-v1-89c10cda391aee1bdcfc4722fb8db16481b31490656a33e2a3b6f1154a7bd587`。
source归档`outputs/qwen-expanded-convrot-native-source-v4-20261019.tar.gz`，SHA256
`5a6b390654a8664255b3970b85181eb85b7a37dedaa6a15c1f4ad1124de3ffbd`。

禁止借用原dense-BF16 QKV、prefix-cache、LoRA或GPU-only单算子资格。

后续独立固定量化配方没有取代row方案：input/hidden均group256的热请求
68.4711s，慢于row39.6599s和同源GPU53.9182s；两个seed SSIM约
0.96320/0.99371，也没有一致质量优势。BF16 value-boundary候选在cold setup
约3分钟仍无图，终止的仅是本次拥有的inference PID及其sample工具，不触碰
共享ANE服务/ComfyUI。没有stack文件，不能声称拿到堆栈；失败不当作0ms或
成功图片。原件及取消journal保留，最终30窗审计不计入这些失败。

## 工程、来源与审计

- `tools/models/download_qwen21_modelscope.py`：逐个pinned资产、断点续传、全SHA、
  无第二Hub副本、原文件名variant根目录。
- `native/models/qwen21/checkpoint_recipe.hpp`/`scheduler.hpp`：真实source/配方
  对齐与作者固定8步，默认base优先级不变。
- `tools/profiling/qwen_base/expanded_collect.py`：完整progress/PNG、fresh/hit、
  同engine变长、source/实际驻留、真正Private selector及process-tree采集。
- `expanded_audit.py`：独立原请求/完整RGBA、来源全SHA、exact native archive、
  memory原始hash-chain与实际预测/能力调用核对，不把host spans相加。
- 首批总审计`outputs/qwen-expanded-full-audit-v1-20261019/audit.json`：20成功窗、
  62张完整PNG、784实际步、11组matched比较；失败GPU reverse的2张/16步不
  借作成功三请求或性能资格。后续ConvRot等新窗另作增量审计。
- V1/V2分别662 inputs，V3/V4为663；各匹配库/完整source archive与历史工具V1–V7
  保留，C++ helper不得跨ABI/内容身份混用。BSD tar的AppleDouble附加项造成
  初审拒绝后，用`native_source_archive.py`从原归档逐文件核SHA重封exact manifest；
  原归档/失败日志保留，没有放宽source gate。

按用户新授权清理7个Oct11–13旧mflux/Flux派生view/ANE export，先lsof检查并
归档386份小型配置/记录，实际free增加29889044480 bytes（约27.8GiB）。原始
模型、当前Qwen证据/图片、references、ComfyUI不动。删除的大型派生物不能从
小归档恢复，须从保留原模型重新生成。清理journal为
`outputs/qwen-expanded-retired-exports-v1-20261019.json`。

全部性能窗串行；自有build/test/CPU数组审计在性能窗外。持续竞争CPU负载
使`qualification_passed=false`保持，未授独占、opaque-driver永久不编译、
硬件INT8或全部场景语义资格。后续继续做各尺寸stock重复/分工、长encoder、
ConvRot完整请求和更丰富图像质量对照，不因单项正结果将总体目标标complete。

# 本轮目标逐项核验

更新2026-10-12。按原目标核对下载/格式、GPU/ANE实现、Unsloth encoder/DiT
对照、BF16 resident/相近内存streaming，以及本地base模型benchmark。
“本轮交付”不等于所有设备/尺寸/操作或生产/Public/hard-RAM发布资格。
历史incomplete、质量fail、busy diagnostic和下载receipt均不改写成通过。

下表保留原阶段审计，不代表新增扩大目标已经完成。最新扩大catalog
已补有效Qwen五路线40步、Z七路线8步、FLUX stock三精度完整对照，
再加Qwen cache0、FLUX compiled、LTX两边界、Z十四参数候选和9B完整
hybrid、两种Q4/g32匹配及提前解码、Qwen1024/4096MiB缓存、Z当前
mflux0.22 BF16/FP16对照、Qwen0.22完整40步四路线，共52个dated campaigns；最新证据见[Qwen](QWEN_BF16_FRAMEWORKS.md)、
[Z](Z_IMAGE_FRAMEWORKS.md)、[FLUX](FLUX_PRECISIONS.md)。更新上游框架版本、
更多native分区/尺寸、FLUX匹配量化/其它9B混合、LTX进一步加载/驻留与成功外部
对照仍缺证据，扩大目标仍active/partial/goal_complete=false。

DiT-only Q4是已完成匹配，不是全组件量化的替代完成：native7.987s比
自身BF16 7.922s略慢、mfluxQ4 8.031s也比自身BF16 7.380s慢，但
native peak18.16GB更低；[真实模型与payload/内存/误差审计](FLUX9_MATCHED_Q4.md)。
随后encoder+DiT eligible linears Q4/g32也完整匹配，native8.005s、
peak11.99GB，未省时；[36层完整覆盖与兼容性边界](FLUX9_FULL_Q4.md)。
其它precision/布局/更多GPU-ANE分区仍未因此完成。

source-Q4 DiT预解码BF16+compiled五路线7.211s对自己packed8.001s省
9.88%，peak24.4GB；153次初始decode、后续96compiled events、40PNG/
10memory与额外舍入均核验。[完整候选](FLUX9_PREDECODE.md)仍为显式
近似高内存配置，不替代低RAM/严格BF16/全设备发布资格。

| 原要求 | 已核验的当前/保留权威证据 | 结论与限制 |
| --- | --- | --- |
| 原unsloth Q4_K_M DiT与Q4_K_M encoder下载，VAE保持BF16 | 固定revision/LFS SHA下载receipt；实际各4,199,565,024 /5,027,785,568bytes；后续native imports/原BF16 VAE SHA绑定 | 已完成；原模型/参考仓库未改写，磁盘足够，未删除用户模型 |
| TurboCider GGUF模型执行 | 原两文件实际512²/40步/base/encoder GPU + packed DiT +原VAE完整PNG；source/type/required-key与typed affine planes；真实默认/关闭/恢复session | 已完成实测原组合；量化须allow_approximation，编辑/LoRA/其它尺寸不自动启用512 fast默认 |
| GPU encoder/kernel加速，参考Splash设计 | typed RMS/NeoX QK norm-RoPE/GQA、QKV18/QK18/gate-up36实receipt；64 GPU cases、原源完整tensor/PNG回归；bounded2-buffer pread/8decoder实测试 | 已实现并测量；原GPU direct-U4/dense候选更慢，负结果保留，不冒充加速 |
| GPU/ANE并行FFN | Private Hadamard W8A8 ANE5120/GPU7168、fixed async/IOSurface；两phase真实32+1248=1280calls/request，failure/fallback/retry0 | 已实现并完整测量，不是首步全部交ANE；host调度/driver路径不是hardware occupancy/native INT8 MAC trace |
| encoder与DiT比原Unsloth快 | 相同原模型/三个fresh prompts；每路前置相同实际三次BF16-stream40步；只读原flags、实际MTL0 conditioner；cold/warm stage与complete generation | 指定512²/40步任务已证明领先；不声称所有cache/shape/prompt或同seed=同noise |
| 相近内存BF16 streaming与resident对比 | 原四路40步；stream prefix28/two slots/13GiB managed；tree physical peak stream/mixed ratio0.9587、在±10%观测范围；全部raw/memory完整 | GGUF GPU比stream省2.395%，mixed省19.005%；resident GPU更快但peak更大；不把预算或observed-fit当硬RAM |
| SHA只初次校验，后续inference复用 | native owner-private authenticated registration；新进程encoder/DiT/processor SHA bytes0；24跨进程/失效负例；Z兼容loader原文件保护 | 已完成Qwen默认注册与真实复用；文件generation/记录变化仍重新验证；Z guard缺陷已修，不忽略ctime |
| docs/benchmark充实简洁、base-only，GPU/最好实际混合 | 主总览、Z五格式16配置完整正反序、32memory/128PNG、当前FLUX9B同源stock/mflux ABBA、当前LTX10全尺寸音视频/5完整树/97帧比较、18个dated samples catalog | 已整理；只选同campaign自己的最快已测GPU/混合，不拼最小样本/不同库；N/A清楚，LoRA/VDN overlay不入榜 |
| 常见框架如Unsloth、mlx-serve | 原Unsloth GPU conditioner/DiT成功计时；只读mflux0.19.2原FLUX9B BF16成功计时；mlx-serve26.10.1原FLUX键/layout与LTX stock memory preflight实际terminal拒绝 | 适用成功对照已完成；原组合不可运行的框架是有原因N/A，不用失败启动时间/不同quant/conversion凑分母 |

## 关键原始记录与实现

- `models/unsloth-qwen21-q4-k-m/download-receipt.json`是历史下载验证，
  其中runtime/performance=false不是后续qualified source；本轮不改该receipt。
- [Qwen同前置负载原GPU参考](../design/validation/qwen21-conditioned-reference-20261010.json)：
  GPU/mixed warm encoder0.091071/0.090424s，原0.13s；DiT43.712706/
  36.168671s，原107.30s；complete44.427732/36.870464s，原125.365s。
- [当前四路同内存机器记录](../design/validation/qwen21-bounded-import-read-ahead-20261010.json)
  的`matched_memory`、[早期同内存分析](../status/qwen21-matched-gguf-bf16-2026-10-10.md)、
  [有界读取/解码](../status/qwen21-bounded-import-read-ahead-2026-10-10.md)、
  [真实无flags默认](../design/validation/qwen21-fast-defaults-20261011.json)。
- [Z当前全格式](Z_IMAGE_FORMATS.md)、[FLUX原BF16 stock参考](../status/flux9-bf16-framework-2026-10-11.md)、
  [当前LTX全链路](LTX_VIDEO.md)、[完整逐请求catalog](results-20261011.json)。
- 当前代码：`native/models/qwen21/gguf_weights.hpp`、`encoder_prefill.hpp`、
  `pipeline.cpp`、`native/runtime/streaming/source_registration_store.hpp`与
  `source_lease.cpp`、`gguf_packed_bank.cpp`/bounded worker；source保护为
  `native/backends/gguf_read_snapshot.hpp`，不修改权重或算术。

原阶段发布回归：`make test-base-benchmark`的30项全部通过；source注册24项、
Qwen options1项、GGUF snapshot1项和SourceLease原生fixture通过。样本
中位数、campaign SHA、所有benchmark Markdown本地链接、portable路径、
97帧媒体范围与完整进程树均复核。未运行全量`make test`，未新增模型推理。

扩大阶段新回归：44个campaign/有效Qwen与Z/FLUX真实compiled+参考精度/cache0/
artifact配置边界已进入定向测试；当前make test-base-benchmark的112项
通过（包含6项FLUX源generation/SHA复用），新实验另保留raw。未运行
全量make test，不将扩大参数、9B混合或外部LTX空缺当成通过。
LTX full-config包含E5警告粘连JSON/未知文本拒绝/actual plan配置三项
新增回归；v1 collector失败保留，新v2 32MP4/20memory完整，所有8个
候选的97帧RGB/motion完成，batch/estimate分开，不推广旧小收益。

10-12新增9B完整混合使catalog为45组。8进程/32PNG/8memory、96实际
calls/request、原index/两shards和全部compiled partitions身份核验；
fresh7.841/compiled7.844s差0.031%，参考7.355s仍胜，repeat收益另列。
[新完整证据](FLUX9_HYBRID.md)不改变正式/自动/其它形状资格。
本次设置TURBOCIDER_FLUX9_GATE_LIBRARY指向实验库运行
make test-base-benchmark，120项全部通过，包括真实API的环境/尺寸/
步数/dynamic-text拒绝；未设置该变量时这两项显式skip，不伪称通过。
仍未运行全量make test；所有测试在性能实验终态后运行，没有并行干扰。

随后DiT-only Q4/g32的完整same-payload native/reference四路线完成，
新增source extent/dtype/group预算、完整saved参数与真实packed/dense
分支拒绝回归。设置同一FLUX9_GATE_LIBRARY运行定向目标，128项全部
通过；未运行全量make test，量化weights/媒体/raw不进Git。

encoder+DiT Q4/g32扩展后定向目标133项通过，包含saved36层推导、
生成RoPE常量排除/其它unknown拒绝、902参数/252encoder linears，
以及派生mlx-serve拒绝不作原格式或latency分母的回归。所有测试在
模型实验terminal后运行；仍未运行全量make test。

提前解码扩展后使用新experimental library运行定向目标137项通过；
另显式GPU opt-in的微型Weights fixture验证Q4/Q8两case、历史FP16
不变、新BF16准确与复制源不变。同真实session的packed/predecoded/
packed/predecoded四完整请求与pixels恢复通过；未作额外计时分母。
所有tests/probes在性能实验terminal后运行，未运行全量make test。

Qwen1024/4096MiB两组完成后catalog为50组，各自24PNG/6memory，
actual cache limits、完整三路线回文、四fresh与cold/repeat独立验收。
新增错误cache实际值/不完整memory/日期/控制顺序拒绝，以及保存reference
自己cohort的native分母回归；首次测试发现cache0配置抄录的尾数不一致，
按原验收值45.96821891650325修正，不放宽比较门槛。设置上述新实验库
运行make test-base-benchmark，142项全部通过，未运行全量make test。
仍不将更多shape/current-upstream/成功外部LTX缺口当成完成。

Z当前mflux0.22.0独立声明依赖环境/源码snapshot、完整四路线8process/
32PNG/8memory已验收，使catalog为51组。nativeGPU7.354/frozen5.959/
referenceBF16 7.976s，mixed优势1.339×；referencepeak22.51GB比native
约32GB低。partialFP16 8.355s反而慢4.746%，306 FP16/215 BF16参数、
实际每图metadata、同engine8组误差与16对正反序RGBA均核验。
新增label-only compute/偷偷改BF16 control、实际cache/日期/无完整memory
拒绝回归，设置实验库运行make test-base-benchmark，148项全部通过。
所有tests/audits在模型实验terminal后运行，仍未运行全量make test。
Qwen/FLUX当前版本、更广native配置与成功外部LTX仍未完成，goal保持active。
随后Qwen0.22两路线短预检完成，4步单cold、288F16/9BF16 DiT参数和
两memory/两PNG核验，只保存precheck、不进入51组完整catalog。补充
预检不得当完整排名的回归后，make test-base-benchmark为149项全过；
仍未运行全量make test，40步新版本对照和其它缺口未完成。

随后Qwen0.22完整40步四路线8process/32PNG/8memory已验收，catalog为
52组：GPU43.309/frozen31.810/referenceBF16 41.952/partialFP16 42.000s，
新版参考胜pureGPU约3.2%，mixed仍1.319×；FP16仅0.114%差异没有
稳定收益，同engine warm-light SSIM0.997113/MAE0.563，288F16/9BF16
参数、actual1GiB cache、所有required weights与1248frozen calls均核验。
新增当前reference胜出/错误cache拒绝和保存自己cohort controls回归。
FLUX新版本七路线工具新增BF16 compiled/hybrid/Q4/predecode及参考
BF16/partialFP16/matchedQ4完整14进程协议、source/precision/calls/PNG/
memory验收与12项CPU协议测试；此时只是工具，不冒充真实新模型结果。
使用实验库运行make test-base-benchmark，163项全部通过；所有tests/
audits在Qwen实验terminal后执行，仍未运行全量make test。
FLUX新版真实运行、更多native shape/precision与成功外部LTX仍pending。

## 不是缺少结果的隐含通过

当前MiniMax原video VAE、FastH3 INT6 DiT不在预期可执行资产树，只能列
N/A与明确历史数据；不能拿VDN/蒸馏附件/其它checkpoint替代。FLUX9B
精确512/4实验hybrid已有完整测量，fresh无稳定收益，正式/自动/其它
尺寸仍未qualified；NVFP4 GPU-only，Core ML成功不是physical trace。
LTX INT8 nominal不到1%且完整媒体漂移大、FP16更慢，当前推荐GPU。
Q4/Q8 Z的低peak比BF16优势大于其混合时延收益，不泛称量化必然更快。

报告只保证本机指定任务的真实保留证据，不保证未来版本、任意尺寸/
seed、产品发布质量、LoRA/编辑/所有硬内存设备。原阶段下载/实现/
性能对照/文档有上述证据；随后用户明确新增的四家族精度/参数/最佳
GPU/ANE及成功框架扩大目标是当前必须继续交付的工作，不因原阶段
完成而缩小或标完成。原失败、参考胜出和未运行扩展继续保留。

# Qwen512：direct QKV 与不可变编辑前缀的组合

2026-10-18，Asia/Singapore，M4 Max /64GiB。仅 Qwen Image 2.1 的512²文生图和
单ref512图生图；本轮新增速度能力针对重复编辑，文生图只作恢复回归。
不涉及 Z Image，不替换 App、不改默认值。持续优化目标保持 active。

## 本轮结果

同提示词/参考图的热编辑，包含全部采样、原VAE及PNG导出：

下表“无前缀”指关闭跨请求snapshot复用，仍保留原有请求内K/V与完整文本/参考
tokens；没有删除前缀内容或缩小attention范围。

| 采样档 | 原始，无前缀 s | direct，无前缀 s | 原始，前缀hit s | direct，前缀hit s |
| --- | ---: | ---: | ---: | ---: |
| 六步 | 8.8560224 | 8.7018571 | 7.6588858 | 7.4628850 |
| 五步 | 7.6793879 | 7.5742873 | 6.5191677 | 6.3404893 |

五步组合相对**同轮五步无前缀原始路径**减少17.43496%；相对已经开启前缀的强
基线减少2.74082%。六步分别减少15.73096%和2.55913%。没有相乘历史倍率，
没有把五步与六步当作等价NFE/画质，也没有把重复编辑时间说成新提示词/新参考
或冷启动时间。

60次质量矩阵中，24对同采样档、同前缀状态的原始/direct QKV完整RGBA全部逐位
一致。另12对前缀hit与完整prefill也全部RGBA exact；这个批次的SSIM为1、alpha
差0，不推广为任意prompt、参考图、身份保持或透明素材资格。

八个计时窗口中七个检测到外部推理CPU负载；六步组合窗口未检出，但CPU静默
不能证明GPU独占。每臂仅一个进程窗口、五个热请求，无置信区间，全部保留
`qualification_passed=false`。这些数字是有边界的本机观察，不是统计性能认证。

## 省掉什么，保留什么

接续[五步×direct四组验证](qwen512-five-step-qkv-composition-2026-10-18.md)，
原有corrected-QKV不能与snapshot同时开启。本轮新增独立默认关闭的
`TURBOCIDER_QWEN21_CORRECTED_QKV_PREFIX=1`，只允许既有有界direct请求中的
单ref512编辑，同时显式开启`PREFIX_SNAPSHOT=1`。packed bank仍拒绝此组合。

相同原始BF16组件、发布固定Viggle v0.2.1 r128、strength1/inference-time、joint
BF16 A/B与F32 ranks、原encoder/VAE，实际五/六步作者节点不变。无ANE、时间
复用、权重merge、额外packed bank、encoder/VAE精度实验或新模型下载。

snapshot miss仍由普通GPU完成完整prefill；导出完成且独立紧凑的静态K/V。
hit在第一步就只计算target，direct融合可用于该步及后续所有decode。不是复用
旧target noise、旧目标输出或FFN residual，也不省略sampler更新。

准入保持原resident512、单adapter pin、GPU/MLX、显式近似许可和其他pilot冲突。
组合flag单独开启、缺少direct/main/snapshot/joint、生成、多参考、ref1024、dump
或非法literal均拒绝。Session仍核对原始BF16来源、原encoder和adapter内容pin。
研究runner逐请求设置全部开关，返回旧路径不会靠残留环境状态碰巧工作。

Transformer的import/export各新增显式`allow_direct_qkv`，默认false。只允许
original-buffer direct路径；packed、DB与direct同时豁免、live callback、trace/
profiling/其他kernel和temporal cache等原有所有权guard不放宽。拷贝仍要求
evaluated leaf、独立storage、紧凑row/data_size及完整geometry/条件身份核对。

Session key额外绑定direct storage和独立composition recipe；原有sampler recipe/
步数、adapter、原始来源、sigma、geometry和不可变条件身份继续参与失效。
切换原始/direct即使prefix producer算术相同也重新填充，不借旧policy的hit。

## 两类命中与实际覆盖

conditioning cache与DiT K/V snapshot不是同一个cache。源码中text-only和edit
conditioning分别有一个entry；文生图请求不必淘汰原有edit conditioning，却会
清除本组合的K/V snapshot。因此切回编辑可以同时是conditioning hit、snapshot
miss。不能用一个布尔值替代两种lifetime。

首步是否prefill取决于实际snapshot命中，逻辑融合覆盖为：

| 档位 | miss：首步普通prefill | hit：全部target decode |
| --- | ---: | ---: |
| 六步 | 160 block | 192 block |
| 五步 | 128 block | 160 block |

完成请求按实际状态核对全部32层；`completed_sampler_steps`独立记录实际5/6，
不再用`decode_blocks/32+1`推算。新增`completed_prefill_steps`、prefix
enabled/hit/bytes回执，plan新增
`qwen21_direct_qkv_immutable_prefix_composition_experimental`。这些是逻辑执行
覆盖，不是硬件kernel数、GPU利用率或GPU/ANE重叠证据。

## 数值夹具：保留反例，不把小shape冒充生产geometry

早期合成夹具使用H256、M128，逐位门槛失败。先修正诊断host读取：转换F32后
仍可能是非紧凑view，须contiguous后按逻辑顺序读完整数组。修正后真实差异在
base GEMM：Q/K/V分别4/2/1个BF16元素；其后B、norm、RoPE也受影响。

同输入反事实只把MPP base替换为原生base，四个阶段全部恢复exact。这定位了
该小GEMM bucket的累积/舍入差异，不是prefix旧状态泄漏，也不是全shape等价
证明。`test_small_gemm_counterexample_is_retained_not_given_equivalence`保留
这个负例及native-base隔离，未换seed挑选通过样本或放宽数值门槛。

正例按真实Qwen投影geometry：H4096、32 heads×128、M1024、rank128；仍为两层
合成权重、小context/channel/FFN的原生Transformer夹具，不冒充完整checkpoint。
源Transformer销毁后，新原始/direct接收者共享同一已完成前缀；两组噪声、每组
三步的完整输出及更新后状态逐位一致，hit首步计入全部decode。正例不安装block
observer，保留正常lazy/compiled路径；预算、隐式import/export和callback拒绝通过。

没有用小模型正例替代完整图片验证，也没有采集完整checkpoint的cached整图
final-latent dumps：dump请求在组合准入中拒绝。这里的整图结论是RGBA；原生
夹具覆盖cached tensor，二者不能混写为整图latent全称证明。

## 完整图片、恢复与失败原件

先执行27次恢复序列，图片均生成，但脚本错误预期切回编辑时conditioning miss，
在第25项回执验收失败。原件`outputs/qwen512-qkv-prefix-restoration-20261018/`
保持failed/inconclusive；未回填runtime identity或修改其summary使之通过。

核对`conditioning_cache.hpp`和实际回执后，只修正冻结序列的独立conditioning
预期，增加对应host负控制；不更改模型cache实现、不放松snapshot失效或数值门槛。
新目录完整重跑27次通过：六→五→六、原始/direct、前缀opt-out、prompt改变、
文生图往返；14对重复状态RGBA exact，换seed重新计算，生成原始/direct也exact。

质量窗口60次：六/五步×改色/玻璃材质/添加柠檬，每组原始73、direct73、原始74、
direct74，再各自prefix prime73/hit73/hit74。24对kernel比较、12对prefix与prefill
比较全部RGBA exact；prime图也复现相同算术的完整prefill控制。无挑选seed或
自动语义评分，未增加身份、人脸、遮罩、多参考或任意prompt的批准。

八个独立固定arm计时窗口各六次完整请求。六步按原始→direct→prefix→combined，
五步反序；首请求含加载/JIT/prime单列，后五次conditioning-hot取中位数，seed73/74
交替。prefix两臂五个计分项均为真实hit；首请求不进分母，恢复/质量时间不作测速。
没有并行自己的构建、测试、模型或CPU图片审计。

八个首请求依序为12.5331、11.2676、11.5060、11.3151、10.1653、10.3175、10.1956、
10.0826秒；系统cache/JIT与顺序不同，不推导冷启动倍率。

## 来源、独立复核与内存

`tools/validation/qwen_qkv_prefix_composition_audit.py`复核complete，结果为
`outputs/qwen512-qkv-prefix-composition-audit-20261018.json`。核对650项native/build
来源及完整归档，重读完整RGBA/实际请求、冻结序列、coverage/命中、时间分母、
active ownership与memory digest。51个无snapshot控制还与前一封存v5矩阵的相同
实际请求/PNG逐位一致，不把旧时间借作新分母。

进入complete审计的135次完整请求：恢复27、质量60、测速48，actual denoise steps
合计752；其中direct78次、completed direct decode block合计12640。另有首次failed
窗口27次已完成图片请求，分开保留，不作合格数据。因此本轮实际执行162次完整
请求；上述次数不是独立prompt数或硬件kernel数。

```text
build/qwen512-qkv-prefix-v6-20261018/libturbocider.dylib
SHA256 48fc47b2586c2682675ee702179b5756a4cffeef9f46cbad250a9d7b82cab936
runtime tc-runtime-build-v1-7a9f94bb76407117f9546b01af0382338eb99250efa3770ad6502d847e229e74

outputs/qwen512-qkv-prefix-native-source-20261018.tar.gz
SHA256 bd4f321b4e4dfa9d64726623412f9af5e2b16a898dfb18dd9a557d90b4b18a07
```

归档恢复到新目录，不用以后修改的native tree冒充该库来源。固定adapter仍为
Viggle v0.2.1 r128原文件，SHA256
`bafb91d0047df3f9b8a5a850b0c967f051164314d8aad778dfa34d9c24ec345b`。

在六个质量task/tier组内，原始/direct在相同prefix状态的active bytes相同，
没有新增持久base权重bank。改色例compact K/V为555,220,992 bytes；snapshot-on
的实际active增量555,745,280 bytes，含其他被保留的owner，不混称恰好只有K/V。
其余prompt按实际token数记录；一份有上限的bank，不累积六/五或两policy副本。

十个complete窗口的memory采样均complete，system swap-in/out均0。恢复/质量
process-tree峰值physical footprint为30,829,270,952/30,806,644,816 bytes；系统
compression约0.79/1.27GB，不归因到单进程。八个计时窗口峰值约18.78–19.76GB、
compression均0。不把active bank增量当作相同幅度的OS内存或每进程物理归因。
首个failed窗口system swap-in为64KiB、swap-out0，亦单列保留，不能算进上述全0。

## 回归与继续方向

本库C API139项：138通过、1个既有Wan fixture缺失跳过；真实GPU scheduler、普通
Transformer lifecycle、两个QKV prefix原生测试通过。新runner host3、既有sampler/
QKV host及options继续回归；知识库检查仅证明文档/检索，不构成模型资格。

原件：`outputs/qwen512-qkv-prefix-restoration-v2-20261018/`、`quality-20261018/`
及`six-{original,direct,prefix,combined}-20261018/`、对应四个`five-*`目录（均带
`qwen512-qkv-prefix-`前缀）。小shape早期失败、native-base反事实与最终原生测试日志
也保留在outputs，不改写失败历史。

复测一个固定配置的独立窗口（output必须为新目录）：

```sh
.venv/bin/python -B tools/validation/qwen_qkv_prefix_composition_screen.py \
  --library build/qwen512-qkv-prefix-v6-20261018/libturbocider.dylib \
  --output outputs/new-qwen-qkv-prefix-timing \
  --mode timing --steps 5 --arm combined
```

该runner逐请求设置研究开关，不影响已安装App；`--mode restoration`和`--mode quality`
分别重跑完整恢复/质量窗口，不能替代无干扰重复测速。

所有权重、历史库/图片、用户改动保留，没有模型下载、用户缓存清理或停止用户
ComfyUI。结束可用磁盘约11GiB，owned构建、设备、测试和审计任务均已退出。
后续继续主要DiT成本、其他GPU/ANE分工与重复反序整请求窗口；这条路径
仍显式实验性，不自动扩大到其他模型、参考数/尺寸或默认部署。

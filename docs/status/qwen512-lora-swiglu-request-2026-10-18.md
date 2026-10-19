# Qwen512：typed LoRA-B/SwiGLU 的完整请求验证

2026-10-18，Asia/Singapore，M4 Max /64GiB。仅 Qwen Image 2.1 的512²文生图和
单ref512图生图；不涉及Z Image，不替换已安装App、不改默认值。持续优化目标
保持active，本页只交付本轮默认关闭的GPU研究入口和整请求证据。

## 结果与资格

在已有direct-QKV强基线上，同档完整热请求用时减少约1.15%–1.83%。五步两类
主要路径另做独立反序进程复测，方向仍一致：文生图约6.16–6.17秒，重复编辑
的前缀命中约6.24秒。没有把历史block差值乘层数/步数或相乘旧倍率。

338次完整请求复核complete：原始输入/final latent对照通过、七任务/两seed的
92对FFN开关RGBA全部逐位一致，全部48对prefix与完整prefill比较也RGBA exact。
336次标准参考请求复现历史同配置PNG，另两次为不同参考图恢复，不冒称历史
矩阵已覆盖这张新参考。

所有窗口仍检出外部推理CPU负载，`qualification_passed=false`。CPU负载不是
GPU利用率或每PID归因；每配置首轮只有一对独立进程，五步主要路径再各一对，
没有置信区间、GPU独占或冷启动保证。数值结果不推广到任意prompt、dtype、
硬件、多参考、身份保持或生产默认资格。

## 从局部候选到有界完整路径

[上一轮block实验](qwen512-lora-swiglu-block-2026-10-18.md)修正typed sigmoid边界，
但加强逐次字节检查后局部wall明显漂移，不能由其宣布整图提速。本轮新增默认
关闭的`TURBOCIDER_QWEN21_LORA_SWIGLU=1`，用完整请求继续验证，而不是把局部
信号当作采用结论或修饰旧失败。

原生helper位于`native/models/qwen21/metal/lora_swiglu.hpp`，与封存研究helper的
表达式、shader和边界相同，只改变命名空间、入口名与说明。固定BM16/BN128，
未按当前结果事后挑选tile。数据流保持：

```text
原始BF16 X @ Wgate_upᵀ → 原base[gate|up]
原在线独立A_gate/A_up → F32 ranks → 原B输入BF16 narrowing
两路B/scale/base修正 → BF16 gate/up → typed sigmoid/SiLU/multiply
完整BF16 hidden → 原down投影及全部down-LoRA → 原residual
```

不融合或更换base GEMM、不合并权重、不共享不同A、不裁token/层/step、不复用
旧target/FFN输出。两路B和激活合为单consumer，只发布完整hidden。Transformer
构造时解析一次各层term的原range/index/up/scale，up只共享原owner，不复制bank；
这项host设置与每次在线rank计算都仍计入完整请求成本。

只在decode生效。snapshot miss时首步完整prefill仍普通GPU；hit时首步也是target
decode，因此融合覆盖也包含它。完整layer权重/因子须为原dense BF16几何
`[24576,4096]`、两条独立r128 gate/up `[12288,128]`及原range，无bias、quantized/
ConvRot替换或额外adapter。

## 准入、cache与回执

- 默认关闭；仅显式近似许可、resident512、普通GPU/MLX、发布固定r128
  strength1/inference-time，六步或显式作者五步。原BF16组件和原encoder/VAE。
- 生成无参考；编辑单ref512。r256、merge、GGUF、ANE/回调、streaming/staged、
  memory override、PE、temporal reuse、其他encoder/VAE/kernel pilot及packed
  QKV组合仍拒绝。非法literal拒绝，不更改旧六步observer/pin predicates。
- 普通QKV或original-buffer direct-QKV均可；编辑的显式immutable prefix也覆盖。
  direct＋prefix仍须原独立`CORRECTED_QKV_PREFIX`opt-in。dump只用于snapshot-off
  原始数组对照，不静默把dump请求算成hit。
- prefix runtime key包含新recipe和tile。变FFN/QKV/采样器/条件必须miss，seed
  改变不应失效静态K/V。import/export另加默认false的显式allow_lora_swiglu；
  DB组合、live callback、非leaf/非紧凑/别名或错误条件等旧guard不放宽。

新增`qwen_lora_swiglu`回执，recipe为`native-base-paired-b-swiglu-b16-v1`，包含
实际`completed_sampler_steps`、`completed_prefill_steps`、`completed_decode_blocks`、
prefix enabled/hit、native base、online independent F32 ranks、complete down-LoRA、
extra_weight_bytes0。plan标记`qwen21_typed_lora_swiglu_decode_experimental`。

成功请求核对全部32层，计数为`32*(actual_steps-actual_prefill_steps)`：六步miss/
hit为160/192，五步为128/160。它们是逻辑完整消费者覆盖，不是物理kernel次数。
采样更新仍为实际5/6，不从decode计数加1推算。

## 原始数组、恢复和质量

raw独立16次完整请求：生成/编辑×六/五步×普通/direct QKV×FFN off/on，seed73、
snapshot-off。八对完整noise/text/reference/final-BF16-latent及RGBA逐位一致，
不把dump/readback时间用于性能分母。

恢复窗口42次，包含六→五→六、FFN/QKV/prefix切换、seed73→74→73、prompt变化、
参考内容改变以及生成/编辑往返。20对重复状态RGBA exact，七对FFN开关控制也
exact；换seed重新计算。conditioning与K/V分别预测，不要求操作切换时两者同时
miss。备用参考为既有v6改色矩阵`trial/8.png`，与primary图像内容hash不同，未修改
任何参考原件。两次备用参考请求只验证对应恢复，不授予新的广泛编辑质量。

质量窗口184次：照片、英文、中文、空间布局、改色、玻璃材质、添加柠檬；七任务
×两seed×两采样档，普通/direct×FFN四组完整对照；编辑另为各配置执行prefix
prime73/hit73/hit74。92对同QKV/prefix/采样档的FFN控制全部RGBA exact；prime
复现对应full prefill，48对prefix hit比较SSIM1/alpha0且完整RGBA相同。

这不是OCR/语义评分，不把原模型已有的指令或参考保真问题说成已解决。完整
checkpoint cached final latents未dump：cached结论为RGBA，原生夹具另覆盖实际
H4096/F12288/r128的tensor与完整down-LoRA。两种证据不混成所有整图latent证明。

## 无dump完整请求中位数

固定direct-QKV，各配置独立进程六次完整请求，首请求含load/JIT/prime单列，其后
五次conditioning-hot取中位数，seed73/74交替并验证恢复。prefix两臂计分均为真hit。
计入实际采样、原VAE、PNG及request内准备，不把脚本写request JSON、后置CPU
图片审计或单独engine constructor当作GPU工作。首次项不代表严格系统冷启动。

| 任务/档位 | snapshot | FFN off s | FFN on s | 同轮减少 |
| --- | --- | ---: | ---: | ---: |
| 文生图六步 | 无跨请求复用 | 7.3402274 | 7.2523579 | 1.19710% |
| 文生图五步 | 无跨请求复用 | 6.2531705 | 6.1570132 | 1.53774% |
| 编辑六步 | off | 8.7191103 | 8.6031658 | 1.32977% |
| 编辑六步 | hit | 7.4725295 | 7.3466549 | 1.68450% |
| 编辑五步 | off | 7.5844661 | 7.4974367 | 1.14747% |
| 编辑五步 | hit | 6.3538564 | 6.2373085 | 1.83428% |

“off”仅关闭跨请求snapshot，仍保留完整文本/参考和请求内K/V；未减少attention
范围或参考编码尺寸。六/五档不混分母，更不乘历史NFE、prefix或QKV百分比。

在复测前固定反序计划：五步生成首轮on→off、复测off→on；五步prefix编辑首轮
off→on、复测on→off。没有因其中一轮较好而选最佳窗口：

| 五步复测 | FFN off s | FFN on s | 同轮减少 |
| --- | ---: | ---: | ---: |
| 文生图 | 6.2470658 | 6.1707095 | 1.22227% |
| 编辑，实际prefix hit | 6.3433373 | 6.2448148 | 1.55317% |

12个首轮＋4个复测窗口均完整退出，所有原始五项时间及首次项保留。未并行自己
的模型、构建、测试或CPU数组审计。额外图片比较在整个模型窗口结束后，不在
各block间插入逐次GPU→CPU hash；原请求必需的finite检查和导出仍保留。不能由
两种协议差异唯一解释上一轮block漂移，亦不由重复两对授予统计稳定保证。

## 独立CPU复核与来源

`tools/validation/qwen_lora_swiglu_request_audit.py`复核complete：651项native/build
输入与链接manifest、当前tree及完整source archive一致；重算原始数组、恢复、
质量、全部PNG、各自时间分母和memory digest。全部338请求/1860个实际采样
更新中，FFN融合179次、完整逻辑decode block合计27680，不是独立prompt或
硬件kernel数量。

336次标准参考请求与当前质量矩阵对应请求及PNG一致，同时194次snapshot-off
复现封存v5控制，142次snapshot路径复现封存v6控制。全部16次raw请求的原始
输入和final latent也与对应历史snapshot-off控制逐位一致。另两次备用参考恢复
单列，不假称历史矩阵覆盖。旧窗口的时间没有进入新性能分母。

```text
build/qwen512-lora-swiglu-v7-20261018/libturbocider.dylib
SHA256 683612e77dffb4bd5585c6886d89084d1dde820bf6c6bbdabe0d7e1fb65ec37b
runtime tc-runtime-build-v1-a5e4e0679534b4e07bf7eb31eb096e5a63ad81fda6a0e51719439eecdf79f6f2

outputs/qwen512-lora-swiglu-request-native-source-20261018.tar.gz
SHA256 5905ec90ac23353300bf034b6283f19d21f87f9979726f735acebe03de1b122c
```

完整审计：`outputs/qwen512-lora-swiglu-request-audit-20261018.json`。以后source变化
时按归档恢复到新目录，不用新tree冒充此库来源。原固定adapter文件/679604800
bytes未改，SHA256仍为
`bafb91d0047df3f9b8a5a850b0c967f051164314d8aad778dfa34d9c24ec345b`。

## 资源、回归与复现

92对匹配条件/采样器/prefix状态的MLX active bytes差全部0；代码只共享原up因子，
没有额外持久weight bank。不是承诺所有activation、MLX或driver staging零复制，
也不是同幅度OS物理内存下降。snapshot自身的原K/V成本仍存在。

19个完整窗口memory采样均complete，system swap-in/out0。raw/恢复/质量的进程树
峰值physical footprint分别31,718,808,352 /32,075,734,056 /31,593,896,688 bytes；
系统compression约3.27/5.54/3.40GB，不能归因某一进程。16个计时峰值约18.76–
19.72GB；前四窗口compression为.0007–.320GB，之后为0。不是每臂DRAM/driver
归因、能耗或GPU独占证据；全部load checks失败，未停止用户ComfyUI。

构建通过；C API140：139通过、1个既有Wan fixture缺失跳过；真实prefix GPU4
（含完整F12288/down-LoRA、普通/direct QKV组合）、普通生命周期1、调度器1通过。
options host1、promoted-source相同表达式1、request host5、request audit host2，
既有prefix/composition/QKV host3/4/8通过。文档检查不授予模型数值或性能资格。

原件统一前缀`outputs/qwen512-lora-swiglu-request-`，包含raw、restoration、quality、
gen/edit的six/five、prefix/noprefix、off/on及四个repeat目录。每窗保留source、
request、loaded runtime、metrics、PNG、load与memory；raw另保留原始tensor。

一个固定配置复测示例（output须为新目录）：

```sh
.venv/bin/python -B tools/validation/qwen_lora_swiglu_request_screen.py \
  --library build/qwen512-lora-swiglu-v7-20261018/libturbocider.dylib \
  --output outputs/new-qwen-swiglu-timing \
  --mode timing --case edit --steps 5 --ffn on --prefix
```

runner逐请求设置所有研究开关，不影响App；五步仍为作者支持的独立采样/画质
档位，不能把本轮等价kernel收益说成五/六档也等价。未下载、merge或清理用户
文件，所有历史库/图片、权重和用户改动保留，结束可用磁盘约10GiB。

后续继续重复干净整请求窗口、主要DiT成本与合适的GPU/ANE分工；本轮不是
全局最快或完整优化目标已完成的证明。

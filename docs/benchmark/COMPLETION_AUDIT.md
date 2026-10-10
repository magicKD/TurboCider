# 本轮目标逐项核验

2026-10-11。按原目标核对下载/格式、GPU/ANE实现、Unsloth encoder/DiT
对照、BF16 resident/相近内存streaming，以及本地base模型benchmark。
“本轮交付”不等于所有设备/尺寸/操作或生产/Public/hard-RAM发布资格。
历史incomplete、质量fail、busy diagnostic和下载receipt均不改写成通过。

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

最终发布回归：`make test-base-benchmark`的30项全部通过；source注册24项、
Qwen options1项、GGUF snapshot1项和SourceLease原生fixture通过。样本
中位数、campaign SHA、所有benchmark Markdown本地链接、portable路径、
97帧媒体范围与完整进程树均复核。未运行全量`make test`，未新增模型推理。

## 不是缺少结果的隐含通过

当前MiniMax原video VAE、FastH3 INT6 DiT不在预期可执行资产树，只能列
N/A与明确历史数据；不能拿VDN/蒸馏附件/其它checkpoint替代。FLUX9B
DiT hybrid未验证、NVFP4 GPU-only；Core ML配置/成功不是physical trace。
LTX INT8 nominal不到1%且完整媒体漂移大、FP16更慢，当前推荐GPU。
Q4/Q8 Z的低peak比BF16优势大于其混合时延收益，不泛称量化必然更快。

报告只保证本机指定任务的真实保留证据，不保证未来版本、任意尺寸/
seed、产品发布质量、LoRA/编辑/所有硬内存设备。用户要求的本轮下载/
实现/性能对照/文档已逐项有证据；这些更大产品/研究扩展没有被偷换成本轮
必需交付，也没有因此声称已经通过。原失败和未运行扩展继续保留。

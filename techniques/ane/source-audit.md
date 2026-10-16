# ANE 调研来源审计：固定版本、阅读范围与未验证项

[ANE 目录](README.md) · [机器可读文件身份](research-manifest.json) · [实验证据](experiments-20261016.json) · [最终验证记录](research-validation-20261016.json)

日期：2026-10-16。已阅读路线图的 ANE/异构/profiling 主线，并检查现有 `ane/`、相关来源卡和实验。此页回答“实际看了什么”，不是全仓库代码审计、全网穷尽列表或所有项目性能复现。

## 证据标签

官方资料 D、选定源码 S、作者报告 U、本库实测 L 是独立维度。获取仓库/tree/README 不自动标 S；源码存在不自动标 L。JSON 的37个文件包含**选定行范围或显式关键词检索**，不是37个文件全文审核；未读的下载文件不纳入 inspected 清单。

本轮新增固定18个仓库/资料入口的37个文件身份，另复用既有 canonical 来源账本。`research-manifest.json` 保存完整 commit、文件SHA-256、行范围、检索符号、URL和未运行声明；下载源码仅在 ignored output。没有把活动分支链接当不可变版本。

## 新读取的实现与资料

| 来源与固定ref前缀 | 实际重点 | 深度/执行 |
| --- | --- | --- |
| [maderix/ANE](https://github.com/maderix/ANE/tree/d91c9845c0784dec7753048954fc6d0e8411fe29) `d91c9845` | README；dynamic MIL/IO/训练循环；INT8 evaluate与Q/DQ检索 | S选段，不运行原训练/benchmark |
| [ANEgpt](https://github.com/vipuldivyanshu92/ANEgpt/tree/0f60ed113306221944e9091b1dc1e160c8698e7d) `0f60ed11` | README训练分工、dynamic weights和CPU offload表 | D/U；获取的其他代码未借此标已读 |
| [Orion](https://github.com/mechramc/Orion/tree/4dcc1c2a626c9ee1f35576bd0b290d489db17d08) `4dcc1c2a` | reload/patch_weights；delta测试；LoRA frontend | S选段，未执行；不接受全部README硬件解释 |
| [ANE-LM](https://github.com/johnmai-dev/ANE-LM/tree/04b1af12aa30b3f280f511d286398469764b4326) `04b1af12` | Conv/fused-FFN生成器、compile/cache、matvec IO | S选段，未跑整模型 |
| [Field Guide](https://github.com/skyfallsin/apple-neural-engine-field-guide/tree/83e73eb94051dc67431829d0f3d3eb67db0b8e9b) `83e73eb9` | README、tests导读、results.tsv | D/U，未执行对应C++探针或ane.cpp |
| [ANEMLL](https://github.com/Anemll/anemll/tree/f4ad26d061dc2e426faa19f81c367343bdeb9f0d) `f4ad26d0` | Conv MLP、KV buffer、StateType、LUT config | S选段，未下载/转换LLM |
| [anemll-bench](https://github.com/Anemll/anemll-bench/tree/865e78bf8199188be612809356dc90f78b925957) `865e78bf` | bytes/time、warm/timed失败处理、型号映射helper | S选段，未执行/上传报告 |
| [Apple Core ML SD](https://github.com/apple-aiml-research/ml-stable-diffusion/tree/ea2805dc1945be20561c77e5f6d1d9a5a637cda2) `ea2805dc` | attention.py、压缩导出、Swift生命周期 | S选段/attention完整范围，未重跑完整SD |
| [Core AI Models](https://github.com/apple/coreai-models/tree/90cc644367f27338c1ed512b9c4b68eb513d4317) `90cc6443` | README组件/OS27要求 | D，不安装上游skills、不运行 |
| [whisper.cpp](https://github.com/ggml-org/whisper.cpp/tree/0c7be696371069e0baa2515ed7aa08d5a5473aa6) `0c7be696` | Core ML encoder bridge | S该短文件，其他模型调用链未全审 |
| [asitop](https://github.com/tlkh/asitop/tree/74ebe2cbc23d5b1eec874aebb1b9bacfe0e670cd) `74ebe2cb` | utils中的sudo sampler/临时文件清理 | S选段，不运行/提权 |
| [旧mactop](https://github.com/context-labs/mactop/tree/eaa2fab31481d10bc7544466434eb74a2e2318c3) → [新mactop](https://github.com/metaspartan/mactop/tree/8dbfaaed7426cff3bcfbcb7951a2e15dfb6d1667) | README迁移、采样/估算/控制功能边界 | D；两入口分列身份，不当两份独立硬件证据 |
| [WhisperKit](https://github.com/argmaxinc/WhisperKit/tree/9f99b5cb6a907730543823ad271b82532782a899) | README定位与关键词 | D，未审encoder/decoder源码 |
| [hollance/neural-engine](https://github.com/hollance/neural-engine/tree/beef0000ab38df4807acef6dc331ddc97525b897) | README问题地图/经验免责声明 | D，旧限制不升级为当前规格 |
| [iPhone ANE训练](https://github.com/slavko-at-klincov-it/ANE-Training-iPhone/tree/2564a23d5e8f0f37d55b72c0367e9b626db57846) | README、训练分工及MLCompute陈述 | D/U；未审核/部署App，未复现速度和后台行为 |
| [ANEForge](https://github.com/sbryngelson/ANEForge/tree/896c4943c5ebf811c0e815f8d1381fd1e157fa7d) `896c4943` | 本轮补读autograd/VJP/Trainer/optimizer/resident | S选段；旧推理实测另列，训练未复跑 |
| [ANE Transformers](https://github.com/apple-aiml-research/ml-ane-transformers/tree/590eb35913622cdef217ebc5329047aab389ac98) `590eb359` | FFN、LayerNorm、attention检索与既有来源 | S选段；不继承历史模型倍率 |

本轮 `MrZaKaRiA/ane` 与 `maderix/ANE` 选到同一个 `d91c9845`；GitHub页面确认前者是后者的fork，而非URL重定向。因此登记为同snapshot的fork关系，不算第19个独立研究。`context-labs/mactop` 则确实是迁移提示页，应沿新地址学习。

## 复用现有深读与本机证据

没有重新克隆/重跑这些完整项目；来源链保持原身份：

- [coremltools compute plan](../sources/repos/coremltools-compute-plan.md)：固定 `a207b1a3`；实际实验仍用已安装8.3.0，不混称整个包同版本。
- [vpipe runtime tiling](../sources/repos/vpipe-runtime-tiling.md)：`c2989a76`；[ccv rowwise](../sources/repos/ccv-coreml-rowwise.md)：`66dc16a3`；[Splash历史FFN](../references/splash-w8a8-walkthrough.md)：`80827e1c`；含原始文件SHA、许可和未运行声明。
- [anemll-profile安全阅读](../sources/repos/anemll-profile-safety.md)：`bf852810`；[PMU来源](../sources/repos/ane-pmu-provenance.md)：`20080a94`；[capacity](../sources/repos/ane-capacity-pmu.md)：固定ref见原卡。
- [PMU81次完整输出](../review/ane-pmu-20261016.md)、[ANEForge13组](../review/aneforge-20261016.md)、[真实Qwen五臂](../review/qwen-shared-ffn-20261016.md)是此前已整理的本机证据，不算成本轮新运行。
- Draw Things、当前Splash入口、Perfetto是框架/上层学习导航；没有虚构本轮已读整个应用调用链或取得ANE设备trace。

## 官方资料、论文/博客与精确版本

| 一手来源 | 本报告采用的内容 |
| --- | --- |
| [ANE Transformers研究](https://machinelearning.apple.com/research/neural-engine-transformers) | 布局/算子表达的设计原因；不迁移历史倍数 |
| [Core ML量化性能](https://apple.github.io/coremltools/docs-guides/source/opt-quantization-perf.html) | M4/A17 Pro整数路径与weight-only区别 |
| [LUT概述](https://apple.github.io/coremltools/docs-guides/source/opt-palettization-overview.html)、[LUT性能](https://apple.github.io/coremltools/docs-guides/source/opt-palettization-perf.html)、[组合压缩](https://apple.github.io/coremltools/docs-guides/source/opt-joint-compression.html) | 表示/部署版本、按需解压、grouped LUT和bit开销 |
| [WWDC24压缩](https://developer.apple.com/videos/play/wwdc2024/10159/)、[性能分析](https://developer.apple.com/videos/play/wwdc2024/10161/) | 新表示、计划/成本与工具；不等同逐op实测 |
| [MLCompute.ane文档](https://developer.apple.com/documentation/mlcompute/mlcdevicetype/ane)与SDK26.5 `MLCTypes.h` | `.ane`仅inference；JSON登记SDK文件SHA与82–110行 |
| [M4 Pro/Max规格](https://www.apple.com/newsroom/2024/10/apple-introduces-m4-pro-and-m4-max/) | 16-core ANE，纠正按GPU命名猜核数 |
| [Core ML SD研究](https://machinelearning.apple.com/research/stable-diffusion-coreml-apple-silicon)、[官方混合量化配方](https://github.com/apple-aiml-research/ml-stable-diffusion/blob/ea2805dc1945be20561c77e5f6d1d9a5a637cda2/README.md#compression-lower-than-6-bits) | 阶段/内存/压缩的原始解释 |
| maderix [Part1](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine)、[Part2](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine-615)、[Part3](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine-c8b) | 作者的逆向、benchmark和训练叙述；旧INT8推断需结合后续代码/官方资料纠正 |

网页按2026-10-16获取/核对，可变页面不伪造Git commit。网页无法覆盖的实现细节回到固定代码；博客的硬件解释是作者推断，不是Apple公开规格。收费/登录或工具未返回的正文不称全文已读；主要以可取得的一手正文、源码和完整本地实验交叉核对。

## 获取与许可边界

[小型抓取器](../labs/ane/research_sources.py)仅下载指定路径，记录commit/SHA，默认所有文件为 acquired-not-read；人工选定范围才进入本JSON。GitHub API对WhisperKit/旧mactop两次403原样保留，随后用 `git ls-remote` 固定HEAD再取raw README，未因此克隆大仓库。一次扩展抓取因README所列的 `gpu_ane_share.m` 在所选tree找不到而中断；该尝试不算完成阅读，后续改为显式存在文件并保存receipt。

manifest的 `license_api_hint` 只是GitHub根许可提示，不替代逐文件授权。本轮不再分发上游源代码、模型或binary；文档是原创综合，代码是原创API客户端/数学/验证器。既有引文资产继续按 [NOTICE](../NOTICE.md)与原来源许可；无明确LICENSE的项目不猜成MIT，私有Apple接口权利也不能从MIT/BSD继承。

## 审计范围与后续迁移

本次完成的是所请求的ANE研究文档与有界对照，不承诺全部开源项目/全部SoC/所有训练方式已验证。全模型benchmark、设备counter取得、训练收敛、私有接口发行与低内存验收均明确保持未验证；它们是采用方案时的后续门槛，不被文档或链接检查“自动通过”。

## 本地学习入口与复查

- [search_contracts.py](../labs/search_contracts.py)：原创CPU合同/显式实验客户端，不是上游完整实现。
- [ane/research_sources.py](../labs/ane/research_sources.py)：原创CPU合同/显式实验客户端，不是上游完整实现。

CPU入口：`python3 -B techniques/labs/run.py search-qualification`。只运行参考数学/合同，不自动触发Core ML、私有API或网络；实际设备命令与证据边界见[实验报告](experiments-20261016.md)。

SoC/OS/SDK、源码/编译入口、权重/LoRA、输入域/shape/layout/精度或实际消费者改变时重开；不继承旧数值、性能或物理设备资格。

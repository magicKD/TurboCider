# 来源审计、路线图覆盖与学习边界

[专题导航](README.md) · [机器可读来源](sources.json) · [实验](experiments.md) · [PR 记录](optimization-prs.md)

核查日期：2026-10-16。已阅读用户指定的 `../notes/TurboCider_Apple_Silicon_Wiki_Learning_Roadmap.md` 全文（1350行），并以工作树现有来源、实验和新外部一手资料交叉核对。该路径是背景说明，不是本专题离线运行依赖。

## 证据深度

- S：实际读了清单指定源码/patch 行范围，不表示全仓库审计。
- D：官方文档、项目 README 或作者报告可取得的正文，不表示源码/性能已验证。
- L：本库此前已保存的源码导读/实测，本轮重读原记录，不算新跑。
- M：本轮实际运行的小型客户端；不继承为上游引擎或完整请求资格。

这些是独立维度，不是一个自动从“链接存在”升级到“硬件已证实”的等级。获取失败、未读下载和未执行项目不标已完成源码审核。

## 核心来源与阅读路径

完整 40 位 commit、文件 SHA-256、行范围/关键词在 [sources.json](sources.json)，该清单含16个独立仓库的选定文件、6个 PR snapshot 和8个已核对的一手网页/PDF。不会因 fork 同内容而重复计算独立证据。

| 来源 | 深度 | 重点阅读位置 → 本地结论 |
| --- | --- | --- |
| MLX `36969c0` | S/M | quantized host dispatch、SDPA padding、NAX guard → [GPU GEMM](gpu-gemm-quantization.md)、[attention](gpu-attention-runtime.md)；M 是 installed wheel，不是同一主线构建 |
| GGML `64e9bce` | S | Metal small-M/MM 路由 → [引擎地图](inference-engines.md) |
| vpipe `c2989a7` | S | runtime slots、stage/weight layout、chunk、probe、emitter → [单独报告](vpipe-runtime-weights.md) |
| ccv `66dc16a` | S | K-split、Core ML evaluate、fast fence → [单独报告](ccv-rowwise-iosurface.md) |
| ANE Transformers `590eb35` | S/D | FFN/LayerNorm 与 Apple 文章 → [图表达与量化](ane-graphs-and-quantization.md) |
| ANEForge `896c494` | S/L | graph key、attention tile、e5rt lowering；已有13组设备案例 → [图搜索](ane-private-and-search.md) |
| maderix/ANE `d91c984` | S/D | dynamic MIL、IO spatial packing、作者文章 → [动态权重](ane-private-and-search.md) |
| Field Guide `83e73eb` | D | M3 Max 的 dispatch/batching/失败经验 → 私有图报告；不继承型号参数 |
| ggml-ane `fecc9c9` | S/D | backend cache/capability/compute 与 bridge → [合同风险](hybrid-projects-and-stages.md) |
| MNN/Candle 固定读本 | S | resize/encode、Rust GEMM stride/offset → [引擎地图](inference-engines.md) |
| vLLM-Metal `468b0c8` | D | 固定 README+团队架构文章 → serving/paged/varlen/NAX；未审其 shader |
| Uzu `ee5ec57` | D | API/UMA、session、model format、benchmark 入口 → 不接受未核实性能文案 |
| whisper.cpp `0c7be69` | S | Core ML encoder 短文件的 All policy 与 memcpy → [stage 分工](hybrid-projects-and-stages.md) |
| ANEMLL `f4ad26d` / Core AI `90cc644` | S选段/D | state/配置与公开版本要求 → 同页部署对照；没有运行模型 |

独立 MFA README 与普通 GPU kernel 既有来源复用，不伪装成16仓库清单内的完整源码审计。PyTorch MPS、MLC 部署文档属于 D；tinygrad、mflux、Splash、stable-diffusion.cpp、SolAttn 与本地三个 hybrid 项目主要复用 L。

## 官方资料与第一方博客

Apple [M5/MLX](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)、[M5/A19会话](https://developer.apple.com/videos/play/tech-talks/111432/)、[WWDC26 tensors](https://developer.apple.com/videos/play/wwdc2026/330/)用于硬件/接口区分；[ANE Transformers](https://machinelearning.apple.com/research/neural-engine-transformers)、[Core ML 压缩性能](https://apple.github.io/coremltools/docs-guides/source/opt-quantization-perf.html)用于图和精度条件。

[maderix 第一方逆向文章](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine-615)与 [vLLM 团队文章](https://vllm.ai/blog/2026-09-22-vllm-metal-v0-28-0)代表作者/实现者的解释，不将其微架构推断、宣传倍率或不同分母当本机事实。只能读到部分正文时不称全文已读。[MPP PDF](https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf) 的2026-03-16版已核对§2.2/2.3/3.1，并视觉检查印刷页6/8/10的直接访存、同步和cooperative布局；没有把指南建议当本机M5测量。

可变网页记录 capture date，不伪造 Git commit；PR status 单列观察日。GitHub API在本轮查询返回403，保留失败；补充仓库用只读 `git ls-remote` 取得 HEAD，再按 commit 下载指定 raw 文件。没有运行 remote install、下载模型或修改参考 checkout。

## 用户目标逐项覆盖

| 要求 | 当前证据/交付 | 边界 |
| --- | --- | --- |
| 阅读路线图并扩充 GPU/ANE | 全文与现有章节核对，新增本专题并从总 README 链入 | 不迁移旧目录、不改变生产路由 |
| 调研参考项目的 GPU+ANE 协同 | vpipe、ccv 深读；FLUX.2/Splash/本地三个项目与 whisper 对照 | 本地来源不虚构公开仓库；物理并发待设备证据 |
| 不同 GPU 推理引擎怎样加速 | 多层引擎地图，GEMV/GEMM/量化、attention、融合/提交两篇机制 | 未跑全部引擎，不给全局速度榜 |
| 不同方案分别 Markdown、README 总结指导 | README 的机制/项目/PR/实验/来源导航及选型表 | 原有基础合同保留，避免大规模复制 |
| GPU feature，含 M5 TensorOps | 专门硬件页、版本门槛、cooperative/quantized tensor 与 NAX 区分 | M4 本机不能测 M5 快路径 |
| ANE INT8、尺寸敏感性 | 官方条件、私有反例、图表达/tiling；两种尺寸共130次实际输出 | plan 不等于物理 placement，误差不隐去 |
| 开源项目、博客、优化 PR、链接追溯 | 固定文件 SHA、8个一手网页/PDF、6个 PR patch/status | 有限精选调研，不声称穷尽全网或全仓库 |
| 可选简单实验，先检查其他 GPU 工作 | 4个准入成功进程、896个 GPU计时输出+130个压缩调用 | 粗粒度占用不能排除所有短时外部干扰 |
| 验证可靠性/合理性 | 独立 FP64 reference、同 packed source、反序/ABBA、正负结果、离线重算和单元测试 | 不把简单测试升格成整模型/产品资格 |

## 如何进一步学习/采用

先按 [README](README.md) 选消费者，再读对应代码范围；将未测的迁移实验明确标 `not-run`。真实采用时重测目标 SoC/SDK、checkpoint/LoRA、cold/warm、实际内存/功耗、完整请求质量与失败回收。本专题研究交付并不以完成所有未来产品 qualification 为条件，也不隐瞒这些尚未验证的维度。

文档为原创综合，未分发上游源文件/权重/二进制；将来复制 shader、MIL emitter 或 bridge 需逐文件核对 license/NOTICE。私有 Apple 接口的兼容与分发风险不会因外围项目 MIT/BSD 许可证而消失。

## 本次交付验证

[验证摘要](validation.json)记录：24个选定源码文件的实际SHA/行范围、6份patch hash已与本地固定快照核对；本专题8项测试通过；techniques-only副本383项测试无失败（2项可选依赖跳过）、50个CPU演示以及链接/catalog/严格知识检查通过。副本不含产品或兄弟仓库，因此不依赖这些目录来解释/检查核心报告。

此验证不检查所有外链永久可用，也不证明全部技术叙述、未运行引擎或硬件并发。四个新设备进程和原始输出另由[实验记录](experiments.md)限定范围。其他同时变化的ANE/Metal/算法工作树内容保留，不据此认领其作者或更改其来源身份。

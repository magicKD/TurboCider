# dev-graph 合并至 dev-verify：代码整合与轻量验收

日期：2026-10-10。本机 Apple M4 Pro / 48 GiB / macOS 26.6 (25G72)。
不下载、复制或导出模型，不替换已安装 App，不推送远端。

## 分支与冲突处理

合并前 dev-verify 为 `1380e3c`，dev-graph 为 `2a86b89`，共同祖先为
`8d12b44`。双方分别有 40 / 1 个独有提交，需要三方合并。
24 个共同修改文件中有 16 个内容冲突，自动合并的其余文件也检查了接口和调用关系。

- **执行流程**：保留 dev-verify 的混合 GGUF、原始 BF16 streaming、LoRA source lease、
  共享 ranks、融合 B、prefix snapshot、显式 FFN phase 及已有 GPU encoder prefill。
  加入 dev-graph 的独立 Native DiT、Native encoder、变长 FFN island 和准备资源复用。
- **缓存**：统一使用 dev-verify 的 conditioning cache，源 generation、encoder recipe、
  reference 内容和尺寸共同控制 text / edit / visual / latent / prefix 的失效。
  reference feature cache 有容量上限，不保留整个 encoder 权重。
- **ANE 接口**：保留显式 channel/bucket override，再追加 request rows 和 bounded encoder bucket，
  修正调用参数位置，防止行数被隐式转换成通道 override。冲突的 bucket 模式明确拒绝。
- **LoRA**：保留已验证 held-fd 读取、seek、原始 A/B materialization、strict target 绑定和异常清理；
  原生后端的 metadata/snapshot 复用同一解析实现。
- **结果桥**：保留旧指标和精度标签，添加原生执行指标；缓存命中和历史累计调用不冒充本次 ANE 执行。
- **配置组合**：新增拒绝 BF16 streaming 静默覆盖 Native encoder/DiT/calibration，
  以及 Native DiT 静默忽略旧 LoRA 实验开关的回归检查。

两个纯 C++ 实现从 `native_encoder.mm` / `native_prefill.mm` 改为 `.cpp`；
内嵌、可独立编译的 Steel support 头文件整理到 kernels 目录，保留
[第三方许可与来源](../../THIRD_PARTY_NOTICES/mlx-steel-attention.md)。没有放宽仓库边界测试。

## 验证范围与构建

普通公开构建和启用 Private ANE 的构建分开验证。旧基线库的 536 个源码输入
与合并前提交逐个匹配；候选库的 605 个输入在测试前后校验。各构建 ID、二进制 SHA、
完整计时与输出误差见配套机器记录；不能将旧分支报告的百分比当作本次结果。

检查覆盖公开请求、原生请求/门禁、Metal 小张量、变长/补齐行数契约、真实缓存/LoRA source leases、
GGUF packed bank、typed encoder、BF16 streaming 的跨槽复用与取消、原生执行元数据和仓库边界。
普通库额外检查 private class strings、链接和 instrumentation 导出。App 编译与 UI 验收分别记录。

最终普通库、CLI、Swift App 与模型库工具编译通过；Private-enabled 库和 CLI 也编译通过。
仓库检查 17 项、公开契约 129 项（3 项按条件跳过）、原生路由/组件检查 41 项、
实际 GPU 回归 15 项（1 项跳过）通过。普通库另跑公开契约及 14 项路由检查，通过二进制发布审计。
Qwen 专项共 69 项，62 项通过、7 项按条件跳过；首次因测试环境未包含 Pillow 导致一个模块
导入失败，复用本地已有依赖后，仅补跑该模块的两项测试并通过，没有下载安装依赖。
七个独立 CPU 契约程序、原生结果桥及 generated 完整两层语言/视觉图检查也通过。
各套件有重叠，不将这些数字相加为独立测试总数。

本轮 App 只做编译，没有重新安装或进行 UI 验收。普通库不包含 Private ANE；性能对照使用
另建的 Private-enabled 库，默认 GPU 对照没有设置 ANE 环境变量。

构建中的接口命名/类型错误、首次误选旧库的测试、工程布局检查失败及构建脚本变更时的中断
保留原始日志。最后的检查使用最终源码和匹配库，不覆盖原始失败记录。
增量构建只复用同编译器/flags、内容未变且头文件未变的对象；受影响的翻译单元和生成身份强制重编译。
验收后清理本轮 479 个对象文件、模块编译缓存与旧 probe，目录占用减少约 **239.53 MiB**；
保留匹配的运行库、CLI、App、最终 probe、身份清单和实验记录，不删除原模型或原有有效 ANE 程序。

## 性能方法

两组有界实验，均使用本地已有 Qwen Image 2.1 BF16；第一组还使用已有 Viggle v0.2.1 r128 六步 LoRA。

1. 同参考图、prompt、seed、512²、1024-reference、六步编辑，依次比较合并前默认 GPU、
   合并后默认 GPU、合并后显式 Native GPU encoder。每臂一个独立进程、一次请求。
2. 同构建独立完整 encoder，Native GPU 与 Native GPU/ANE 各在同一 encoder owner 上编码两次，
   每次重新载入源权重，无 conditioning/KV cache。分别报告首次准备和资源复用。

各臂串行，间隔休息；不同时运行构建或其他模型实验。固定顺序、小样本、系统/驱动缓存未重置，
因此只是本机筛选，不是稳定平均收益或全场景认证。组件、encoder、denoise、完整请求和冷成本分别记录。

## 本轮结果

六步 LoRA 单图编辑每臂均实际完成六步、绑定 227 个 LoRA projection 并输出 512² 图片。
`text_encode` 包含此请求的完整视觉/语言 encoder 阶段及其准备成本。

| 同输入完整请求 | Encoder | Denoise | 完整请求 |
| --- | ---: | ---: | ---: |
| 合并前默认 GPU | 5.993 s | 27.413 s | 38.410 s |
| 合并后默认 GPU | 4.609 s | 27.016 s | 36.582 s |
| 合并后显式 Native GPU encoder | 6.498 s | 27.277 s | 38.653 s |

合并后默认路径的 encoder 省时 **23.09%**，完整请求省时 **4.76%**。
这是本轮同输入观测，不能将约 0.40 s 的 denoise 差异归因于 encoder 优化，
也不能将单次结果扩展为所有模型/输入的保证。

显式 Native GPU encoder 首次请求比旧默认慢 **0.63%**，比合并后默认慢约 **5.66%**。
该次 Native preparation 为 **2.457 s**；即使执行图已经可以完成全部 27 层视觉和 36 层语言，
准备成本仍会抵消计算收益，因此本次没有把它改成默认。

三张结果逐张查看，茶壶主体与暖色光照接近。旧默认→新默认的 RGB 相对 L2 为 **0.577%**，
MAE 为 **0.465/255**；新默认→Native GPU 为 **1.014%**，MAE 为 **1.288/255**。
输出并非逐位相同；本样本不是编辑指令遵循或全面质量认证。

完整独立 encoder 对照使用同一候选库，包含所有视觉和语言层，不含 DiT/VAE：

| 路线 | 首次 owner 编码（已有系统缓存） | 同 owner 再编码 |
| --- | ---: | ---: |
| Native GPU | 4.769 s | 3.971 s |
| Native GPU/ANE | 3.652 s | 3.357 s |

资源复用状态下 GPU/ANE 省时 **15.45%**（约 **1.18×**），语言阶段由 2.394 s 降到 1.799 s。
两次 hybrid 各实际执行 36 个 ANE FFN / 36 个 chunk，累计补齐 3096 行；物理完成标记通过。
conditioning 为完整 114688 个 float，均有限，两个状态相对误差均为 **4.510%**，
通过事先声明的 **8%** encoder 筛选线。

首次 owner 对照观测省时 23.42%，但固定测试顺序与共享系统缓存使它不能代表全新机器冷启动。
本轮没有再跑 GPU/ANE 六步整图，不能把独立 encoder 的 15.45%写成整次编辑收益，
更不能用 encoder 误差筛选代替最终图片质量验收。

## 性能判断与后续接线

合并后普通路径的 denoise 仍占完整请求约 **74%**；本样本的主要等待已经在 DiT，
继续只优化 encoder 对整次请求的影响有限。原生路线的首次准备、源权重就绪、
GPU/ANE handoff 也仍有成本，不能只比较 ANE kernel 内部时间。

当前可保留合并后的默认 GPU 优化。Native GPU/ANE 更适合先在持久 Session 中复用
准备资源、按模型/source generation 和实际行数选择有效 cache，再补一组同构建、
同输入的完整编辑对照。之后才适合增加 App 高级配置与收益提示。
DiT 的 W1、LoRA 和首步 prefill 需要分别验收；旧分支已有结果只保留参考资格。

## 使用与剩余限制

- 默认执行仍为 MLX GPU；Native encoder、Native DiT、Private W1、首步 ANE 是显式实验。
  合并代码不等于所有新路线自动启用。
- 独立 Native encoder 通过 `TURBOCIDER_QWEN21_ENCODER_BACKEND=metal` 选择；
  `TURBOCIDER_QWEN21_NATIVE_ENCODER_ANE_CACHE` 是另一个显式入口，需 Private-enabled 库。
  当前普通 App worker 的环境白名单没有提供该 encoder 选择，不声称 App 已自动接入这两个入口。
- Native edit-prefill ANE 仍为 base-only，不能据本轮 LoRA GPU encoder 测试授予其 LoRA 资格。
  W1 单槽尚未变成 W2 双权重流水。W8/A8 数据路径不证明硬件全程 INT8 计算。
- 本轮未重跑 20-step Native W1 整模型、所有模型/尺寸、长程压力测试或完整质量 campaign。
  历史加速记录保留其原构建与输入边界；发布仍需单独的产品/质量验收。

原始记录：`outputs/dev-graph-merge-20261010/`；有效路径、性能与检查摘要见
[机器记录](../design/validation/dev-graph-merge-light-20261010.json)。

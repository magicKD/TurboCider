# M1 Pro streaming 实施与实验记录

状态：进行中，尚无本轮真实模型性能结论。本文保留用户要求的 Flux.2 Klein **4B** 和 Z-Image Turbo 范围，不以已有 Flux 9B / M5 结果替代。

## 实机基线

- Apple M1 Pro，8 核 CPU（6P+2E），16 GiB 统一内存。
- macOS 26.4.1，build 25E253。
- 起始代码：`feat/stream@85443b7`，工作树干净；未发现运行中的模型实验。
- 已 fetch `origin/dev@61c0849`。整合 INT8 streaming、variant selection、历史任务批量删除、图片选择；四处冲突涉及 App、Z-Image runtime 和权重流接口/实现。
- 项目首次配置 Python 3.11.15 及仓库锁定的 MLX/Core ML 依赖。不能沿用旧机器的编译产物和认证。

## 模型获取

下载目录：`/Users/chencanhui/models/TurboCider`。下载完成、内容校验和完整推理是不同状态，当前仅已启动下载。

| 模型 | 来源 | 固定 revision | 文件范围 |
|---|---|---|---|
| Flux.2 Klein 4B | `black-forest-labs/FLUX.2-klein-4B` | `e7b7dc27f91deacad38e78976d1f2b499d76a294` | Diffusers transformer、text encoder、tokenizer、VAE、scheduler/config |
| Z-Image Turbo | `Comfy-Org/z_image_turbo` | `08d04455279082882deaabc8d0d09fc914c071e1` | Comfy BF16 transformer、Qwen3 4B text encoder、VAE |

Z-Image tokenizer 单独来自 `Tongyi-MAI/Z-Image-Turbo`，下载时将其具体 revision 写入本地 provenance。每个下载先写 `.partial`，curl 成功后 rename；后续仍须检查大小、内容 digest 和模型结构。

## dev 整合要点

- legacy INT8 支持可变预取槽；exact BF16 仍只接受编译计划规定的 K1/K2，并按真实槽数分配。
- exact constructor 显式拒绝 ConvRot，避免 metadata 扩展后把未转换 INT8 当作 BF16 exact 权重加载。
- BF16 保留批量 read 路径和 worker cancellation；INT8 转换 read 同样传递取消信号。
- legacy 预取策略进入缓存配置键，exact 请求不调用 legacy 自动预取规划。
- App 保留 public streaming 控件，同时合入 dev 的设备/精度驻留限制和批量历史操作。
- 扩展真实 Metal tiny-fixture 回归，覆盖 K1/K2 的预算、每槽数据、分配数及 worker cancellation。运行结果待后续记录。

## 验收清单（尚未完成）

1. 合并后的 native / Swift App 构建、host / Metal / App 回归。
2. docs/design/streaming 56 R1–R6 的来源身份、options、全组件 lease、失败清理、App 事务和运行容器缺口；按真实代码逐项修复，不靠空 catalog 宣称完成。
3. Flux 4B 和 Z-Image 完整模型下载、校验、GPU baseline 与 streaming 真实出图。
4. 本机 Core ML artifact 导出与验证，GPU/ANE 分区探索；报告 cold setup、denoise、完整请求 wall、质量、process-tree memory 和 swap。
5. 相同 partition/reference 的迁移正确性，与纯 GPU 的性能收益分别测量；M5 专属优化不能直接标为 M1 已支持。
6. 非空 catalog 的 App→worker→verified result→有效新产物、取消和重试；完整发布证据和生产 record。
7. 所有实现完成后的默认路径回归与最终性能验证。当前不能宣称功能完备、发布可用或 ANE 有加速收益。

## 本轮运行句柄（临时，仅供继续工作时重新核查）

- 依赖安装：exec session `97499`，日志 `/tmp/tc-setup.log`。
- host 测试：exec session `2385`，日志 `/tmp/tc-host-tests.log`。
- 模型下载：exec session `70822`，脚本 `/tmp/tc-download-models.py`，日志 `/tmp/tc-model-download.log`。
- 完整构建：exec session `44723`，等待安装进程退出后执行，日志 `/tmp/tc-build.log`。

句柄/日志不是完成证明；下一轮先 poll 存活进程，不能仅凭日志静止启动重复下载或构建。

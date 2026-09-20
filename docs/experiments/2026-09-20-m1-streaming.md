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

## 2026-09-20 首轮检查结果

- 合并提交 `7b886ab`；`git merge-base --is-ancestor origin/dev HEAD` 成功，包含本轮抓取的 `61c0849`。
- `make test-streaming-host PYTHON=python3`：exit 0。包含 layout/config 3,535 组合、descriptor、executor 14 组合、C bridge failure、actual receipt、source lease、public preset runtime/result、LTX metadata/model/snapshot、H3/Z-Image/Flux descriptor。这是合成 host 回归，不是模型出图或性能实验。
- repository `test_layout.py`、`test_independence.py`、`test_cpp_boundaries.py`：共 13 项通过。
- `git diff --cached --check`：通过。
- Metal fixture、完整 native/Swift 构建及 App 运行尚未完成。依赖安装、模型下载、等待依赖的构建句柄均已重新 poll 确认存活；没有重启重复作业。

## 第二轮：本机 native / Metal 验证与 App 结果绑定

依赖安装 exit 0，两个受管理 Python 环境的 `pip check` 通过。native dylib 与 CLI 编译/link 成功；完整构建仍在生成 Swift App / integration binaries（session `44723`），不能将 native 成功写成 App 验收完成。

新增 App 修复：非 LTX public 请求现在用 `resolveStreaming` 返回的 `exact_selector` 生成；native 仍重验最新 catalog 和 source。任务置为 succeeded 之前，Swift 严格解码并对照目标档、record/catalog/resolution、source/workload/runtime/device、授权与实际 layout、组件策略、container、memory scope 和 receipt 摘要；同时核对 model/operation/output/shape/seed/steps，拒绝 warmup 和伪造 boolean。此处消费 native verified result，不把 Swift 摘要检查称作独立 GPU receipt verifier。

`StreamingResolutionTests.swift` 已加入 build-app/test-app，独立编译运行 PASS：精确绑定、原始请求不变、各字段删除/篡改、数字/字符串冒充 bool、v2/v3 receipt。尚未覆盖 LTX worker-local resolution、新产物 staging/原子发布与全部 JobStore 错误恢复；这些仍是 R5 后续任务。

本轮验证：

| 检查 | 结果 | 范围 |
|---|---|---|
| `turbocider self-test` | PASS | M1 Pro Metal、BF16 Euler、layer norm、recipe/dependency rejection；不是完整模型 |
| `test_z_image_weight_stream.py` + `TURBOCIDER_TEST_GPU=1` | 4 PASS，6 SKIP | BF16 K1/K2 实际 GPU 值、复用/取消、损坏 metadata/截断拒绝；M5 专属 INT8/suffix 正确跳过 |
| `test_streaming_metal.py` | PASS | 真实 Metal K1/K2/K3、3 passes、输出相等；synthetic |
| `test_z_image_public_streaming.py` | PASS | source lease/layout、route rejection、失败清理、来源替换；host adapter |
| `test_contract.py` | 80 PASS，3 SKIP | 83 项 native API 合同；跳过原因保留在本机日志 |
| Swift streaming resolution regression | PASS | `/tmp/tc-resolution-tests.log`，独立 SDK fixture |

修复两个测试预期问题：原 exact 值检查对 13 个 tensor 求和，却只使用 1 个 tensor 的预期值；现断言 tensor 数为 13 并计算全部元素。旧 memory-constrained fixture 固定 48 GiB，会在本机 16 GiB 被正确拒绝；现使用不超过实机的 limit，分别验 plan-only / insufficient-budget rejection，并新增超过本机物理内存的拒绝检查。没有放松 runtime 的物理内存限制。

原始日志：`/tmp/tc-z-metal-tests.log`、`/tmp/tc-streaming-metal-tests.log`、`/tmp/tc-z-public-tests.log`、`/tmp/tc-contract-tests.log`。模型下载仍由 session `70822` 执行；没有真实 Flux/Z-Image 出图或 GPU/ANE 性能数据，M1 profile 仍明确关闭 M5 专属优化。

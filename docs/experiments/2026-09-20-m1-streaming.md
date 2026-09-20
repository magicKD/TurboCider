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

## 第三轮：Z-Image 全组件租约、App 启动与 Flux 4B 缺口

实现 56 R3 的来源/缓存接线：

- 公共 Z-Image SourceLease 从 transformer/text/VAE 三个文件扩展为四个，新增实际使用的 `tokenizer/tokenizer.json`。模板与配置逻辑目前在 runtime 中固定；没有将未使用的 tokenizer_config 声称为计算输入。
- 新的 `Tokenizer(fd, bytes)` 同步读取已授权 fd，不重开路径；probe 和 generate 都从本次租约构造 tokenizer，实际 conditioning 与 token 计数使用同一请求来源。
- public text encoder 与 VAE 经 `Weights::load_lease` 读取。默认路径仍使用原有 loader。
- 每次进入 public 或离开 public 到 legacy 时，先 synchronize 再清未绑定 source identity 的 conditioning/text/VAE 缓存。失败时保留 public-cache 标记，下一请求同步清理；没有在可能存在 pending 设备工作的 catch 中新增数组释放。
- adapter revision 更新为 `z-image-public-adapter-v3-all-component-lease`；组件策略更新为 `zimage-components-v2-all-sources-request-cache`。旧 calibration 不能直接充当新版本 evidence。

验证：

- 独立 `build/m1-component-lease` native 全量构建 exit 0，日志 `/tmp/tc-component-build.log`，避免覆盖在编译 App 的 `build/native`。
- `TURBOCIDER_TEST_NATIVE_DIR=.../build/m1-component-lease test_z_image_public_streaming.py` PASS。真实 fd 读取 tokenizer；替换路径后旧 fd 仍读原内容、旧 lease 拒绝、新 probe source identity 改变；invalid fd 和 truncated source 拒绝。transformer path replacement 继续拒绝。日志 `/tmp/tc-z-components-tests.log`。
- 相同新 dylib 的 `test_contract.py`：80 PASS、3 缺本地 LTX/Wan fixture 的 SKIP；日志 `/tmp/tc-component-contract-tests.log`。
- `DYLD_LIBRARY_PATH=.../build/m1-component-lease build/native/turbocider-studio-tests` PASS；日志 `/tmp/tc-component-studio-tests.log`。
- 原 App 构建的 Studio behavior / history batch-delete tests PASS。`TurboCiderNativeApp` 实际启动并存活约 2 分钟，启动日志无错误，随后主动关闭；这是启动 smoke，不是非空 catalog 的完整 GUI 生成验收。
- App 构建 session `44723` 仍在生成其余 integration binaries；完成后运行 `make test-app` 的等待任务为 session `81906`，日志 `/tmp/tc-app-suite.log`。不可提前宣称完整 suite PASS。

未完成：模型权重仍下载中（本轮约 4 GiB），未执行 public A→B→A 完整出图/质量回归；R1 内容身份分离、R4 drain/quarantine、R5 worker 与产物事务、R6 构建/container identity 和发布 evidence 均未因本改动完成。

### 下一项必须覆盖 Flux **4B**

代码复核确认：当前 `flux_streaming_descriptor.mm`、Flux public probe / exact run 以及 App publicStreamingModel 都只允许 9B。必须实现 4B，不能将文档里的 9B evidence 当作用户要求的 4B 完成。

固定官方 revision `e7b7dc27f91deacad38e78976d1f2b499d76a294` 的 `transformer/config.json` 已单独获取（`/tmp/tc-flux4-transformer-config.json`）：24 heads × 128 = 3072 hidden；5 dual + 20 single blocks；joint_attention_dim=7680；MLP ratio=3。4B 是单个 `diffusion_pytorch_model.safetensors`，不能套用现有 9B 的两 shard index 要求或 hidden×3 的文本输入宽度假设。

下一轮优先扩展 metadata/source/descriptor/runtime 的 4B 实际几何与单文件支持、独立 fixture 和 App admission，然后在下载完成后测 M1 的 4B streaming。继续保留 9B 兼容回归。

## 第四轮：Flux 4B exact streaming 接入

已补齐前一节发现的 4B 缺口（开发能力，尚未发布认证）：

- Metadata 接受 4B 官方单个 `diffusion_pytorch_model.safetensors`；逐 tensor 校验 shape/dtype/range/完整性，不要求伪造 9B 的 index。
- 4B 固定 3072 hidden、24 heads、7680 context width、5 dual + 20 single。公共 probe/source/runtime/layout 身份区分 4B 与 9B；9B 保持原两 shard 布局。
- Exact adapter 使用实际 metadata 的 hidden/head 数，不再写死 4096/32；保留每 block 的真实 eval 完成边界、两个 pool、每池 K2，以及 D0/1、Q1/2。
- Exact 4B 不自动开启常驻 compiled graph；默认非 exact 4B 的 compiled 路径保持原行为。诊断 direct executor 同样支持 25 groups，可用于同布局后续对照。
- Candidate C API 和 App public model selector 接入 4B。生产 catalog 仍空，正常 public generate/resolve 继续明确拒绝未认证档位，没有绕过 gate。
- `build_app.sh` 现在遵守 native build 同一个 `TURBOCIDER_BUILD_OUTPUT_DIR`，修复独立 native 构建后 App 仍误链接 `build/native` 的目录不一致。

本机验证：

| 检查 | 结果 | 证据范围 |
|---|---|---|
| 4B/9B metadata descriptor suite | PASS | 单文件与两 shard；字节数 oracle、3072/7680、25 groups、D/Q、lease/path parity；missing/dtype/shape/overlap 必须以受控异常拒绝，不接受 crash 作为 PASS |
| Flux public adapter 4B + 9B | PASS | source closure、layout identity、错误路由/target、来源替换 |
| Flux candidate/public gate 4B + 9B | PASS | 空 catalog 拒绝 public；private candidate 到达对应 runtime；prepare selector 拒绝 |
| **M1 真 GPU 的 4B block smoke** | PASS | 合成零权重、实际 3072/24-head 形状、5+20 blocks × 2 passes=50 次、残差独立 oracle、2 pools/4 slots/50 fills；不是官方模型质量/性能证明 |
| 新 native API contracts | 80 PASS、3 缺 fixture 的 SKIP | 83 项，`/tmp/tc-flux4-contract-tests.log` |
| 新 CLI self-test | PASS | `/tmp/tc-flux4-selftest.json` |

构建目录 `build/m1-flux4`；native 构建 exit 0（`/tmp/tc-flux4-build.log`）。测试日志：`/tmp/tc-flux-descriptor-tests.log`、`/tmp/tc-flux4-public-tests.log`、`/tmp/tc-flux4-gate-tests.log`、`/tmp/tc-flux9-gate-tests.log`、`/tmp/tc-flux4-metal-tests.log`。真实 GPU fixture 入口为 `make test-flux4-streaming-metal`，可配 `TURBOCIDER_TEST_NATIVE_DIR` 选择构建。

前一轮完整 App build 与 `make test-app` 都已 exit 0，日志 `/tmp/tc-app-suite.log`，覆盖 streaming identity、ANE library、variant、Studio、history、model library、store、run insights、tensor cache。**该完整 App suite 对应新增 4B selector 之前的构建**；新的独立 App 构建 session `30718` 正在运行（`/tmp/tc-flux4-app-build.log`），其中已加入 4B Studio intent 回归，完成后须重跑。

下载 session `70822` 仍存活且持续增长，已开始 Flux transformer 权重下载；完整 Flux/Z-Image 文件尚未齐备。下一步：确认新 App/4B intent、完成官方模型校验后以 private exact 流程实跑，对比 eager resident / compiled resident / streamed，分别记录质量、全请求内存和耗时；再开展 ANE artifact 与 partition 实验。合成 block smoke 不作为模型出图、收益或 public qualification。

## 第五轮：内容验证基础与完整 App 回归

### 内容证明与请求文件绑定分开

新增 `SourceLease::capture_verified()`：只从该 lease 已打开的 fd 计算 SHA-256，使用 1 MiB 堆缓冲、`pread` 和取消检查；Apple 使用 CommonCrypto，其他平台保留 portable SHA 实现。全部文件完成后重新校验 held fd、named path 和 canonical target，再发布结果与缓存，避免读到变化中的文件仍发布证明。

`artifact_digest()` 使用独立 domain `tc-streaming-artifact-content-v1`，只编码排序后的 logical id、字节数、实际验证的 SHA-256；现有 `digest()` 保留请求 stat binding 语义。`capture()` 与 `open_and_verify()` 不因调用方提供了 `content_digest` 就授予内容证明；读取其 artifact digest 会明确失败。调用方提供给 verified capture 的摘要是待核对的期望值，不是可直接信任的证明。

增加有锁、最多 256 项的进程内原生摘要缓存，key 包含 device/inode/size/mtime/ctime。`capture_preverified()` 只消费仍匹配的缓存证明，缺失、被逐出或文件变化时返回 `artifact_verification_required`，不会隐式读取大权重。暴露本次验证读取字节数与缓存命中数用于检查成本。

测试覆盖：独立 Python canonical golden、不同安装副本内容身份一致而 binding 不同、logical role 区分、排序稳定、缓存零字节读取、同尺寸修改并恢复 mtime 后缓存失效、假摘要拒绝、metadata/replay 不授予证明、取消、短读、fd offset 保持、跨缓冲边界 SHA 对照，以及 **256 KiB worker stack** 上的实际文件摘要。

验证均 exit 0：source lease fixture（`/tmp/tc-source-content-tests.log`）、最终 fixture 的 ASan+UBSan（`/tmp/tc-source-content-sanitizer.log`）、完整 `make test-streaming-host PYTHON=python3`（`/tmp/tc-content-host-tests.log`，含 3535 layouts、executor/receipt/public resolver、LTX/H3/Z/Flux 4B/9B descriptor）及 repository 13 项测试（`/tmp/tc-content-repository-tests.log`）。本轮新增接口尚未用于已构建的 App；App suite 对应上一轮的完整 4B native/App 构建。

**边界仍未关闭**：这只是 R1/A1 的底层接口。当前缓存不能跨进程或重启保存证明；导入/下载 UI、Z/Flux public probe、portable descriptor/layout、catalog/Python/schema/receipt 迁移尚未接入。正常 public adapter 仍使用原 metadata capture，生产 catalog 仍为空；不能声称跨安装 public resolve 已通过，也没有重用旧认证证据。

### App 与下载验证

`build/m1-flux4` 的完整 App 构建 session `30718` 已 exit 0。随后从此目录运行九组 App suite（streaming resolution、ANE library、Studio variant、Studio behavior、history、model library、library store、run insights、tensor cache），全部 PASS，exit 0。新 Studio behavior 包含 4B public streaming intent；日志 `/tmp/tc-flux4-app-suite.log`。这次与前节旧构建的 App 测试已明确区分；仍不是 GUI 完整出图或 LTX worker 严格结果验收的证明。

对固定 revision 调用官方 Hugging Face API 的 `blobs=true`，取得 LFS SHA-256/普通 Git blob SHA-1 和字节数；对已完整落盘的 **18 个 Flux 文件** 全量验证，hash 前后核对 stat，全部匹配。日志 `/tmp/tc-download-verification.log`，逐文件结果 `/tmp/tc-download-verification.json`，原始官方清单 `/tmp/tc-flux-verified-manifest.json` 与 `/tmp/tc-z-verified-manifest.json`。

| 完整权重文件 | 字节数 | 官方 SHA-256 |
|---|---:|---|
| Flux text encoder shard 1 | 4,967,215,360 | `8c0506e7f4936fa7e26183a4fd8da4e2bdbc5990ba64ae441f965d51228f36ea` |
| Flux text encoder shard 2 | 3,077,766,632 | `82f2bd839378541b0557bfabaf37c7d3d637071fdcb73302dedd7cf61162ce07` |
| Flux VAE | 168,120,878 | `ca70d2202afe6415bdbcb8793ba8cd99fd159cfe6192381504d6c4d3036e0f04` |

下载 session `70822` 持续运行，目录约 9.3 GiB；Flux transformer 与 Z BF16 transformer 正在下载。其余未完成文件未宣称通过校验。后续可重跑 `/tmp/tc-verify-downloads.py` 校验新完成文件，再执行真实模型推理。

## 第六轮：Flux 组件来源修复与新 App 启动

复核发现 Flux public probe 虽捕获 tokenizer 文件，却仍使用 engine 构造时的旧 tokenizer；run 也存在同样问题。现改为 probe 与 generate 各自从本次 lease 的 held fd 构建 tokenizer，解析前后重新校验来源，成功/失败都恢复请求绑定。新测试在同一 engine 内替换 tokenizer 的 special token，使实际 token 数改变：新 probe 必须匹配新文件，旧 fd 仍读到原内容，旧 snapshot/record/请求必须拒绝。

另外将原来仅 incoming public 才清理的组件缓存扩为 incoming/outgoing public 请求边界：run、prepare 和 load 在安全同步后清除 conditioning、encoder hybrid 和 VAE；失败时保留标记，下一次安全边界再清理。现有 text/VAE held-fd 加载保留。组件 revision 升为 `*-components-v2-all-sources-request-cache`，adapter 升为 `*-public-adapter-v3-all-component-lease`。修正 4B receipt runtime 的格式标签为 `diffusers-bf16-single-file`。

新 native 库与 CLI 已构建到 `build/m1-flux-components`，Flux 4B/9B public adapter 测试均 PASS（`/tmp/tc-flux-components-public-tests.log`），API contracts 80 PASS/3 缺 fixture SKIP（`/tmp/tc-flux-components-contract-tests.log`），本机 GPU self-test PASS（`/tmp/tc-flux-components-selftest.json`）。当前 build session `22293` 已进入后续完整 Swift 构建，尚未宣称整个 build exit 0。真实权重的 A→B→A 出图、失败恢复 owner 与缓存峰值仍待验，host fixture 不替代它们。

上一轮完整 `build/m1-flux4/TurboCiderNativeApp` 也已在桌面启动，90 秒保持存活，stdout/stderr 无报错，然后终止本次启动的进程。`/tmp/tc-flux4-app-launch.log`、`/tmp/tc-flux4-app-stderr.log` 为证据；这是 launch smoke，不是 UI 全流程验收，也不包含本节新 Flux native 修改。

## 第七轮：真实 Qwen3 权重的单 block ANE 工具链与桥接

Flux text encoder 已完整校验，因此无需等待 transformer 下载即可先验证 encoder artifact 工具链。首次在 App 的 Core ML Python 环境运行 `export_qwen3.py` 失败：无条件 `import mlx.core`，而该环境按 `coreml.lock.txt` 安装并不含 MLX。BF16/F16 路径只需要 NumPy；现将 MLX 导入移至实际 affine 量化解码分支。使用同一干净 Core ML 环境重跑，真实 BF16 的导出成功。量化源仍需要带 MLX 的构建环境，本次没有宣称覆盖它。

本次实验固定为 **Qwen3 layer 0、hidden=2560、MLP width=9728、ANE prefix=4864、64-token bucket、INT8 per-channel artifact、FP16 I/O、output_scale=1**。只导出一个 block，不将不完整 manifest 当作完整 encoder 路由。

导出命令（Python 为 `~/Library/Application Support/TurboCiderNative/toolchains/coreml/bin/python3`）：

```sh
python tools/coreml/export_qwen3.py \
  --model /Users/chencanhui/models/TurboCider/FLUX.2-klein-4B/text_encoder \
  --output /Users/chencanhui/models/TurboCider/experiments/m1-qwen3-block0-b64-w4864-int8 \
  --bucket 64 --ane-mlp-width 4864 --layer-count 1 --variant int8_pc
```

随后使用 native `coreml` 的 `compile` action 生成 managed-cache manifest（1 partition、0 cache hits），并通过 `tools/native/benchmark_coreml_ffn_bridge.py` 调用真实 C ABI：64 rows、seed=42、input_scale=0.25、1 次 warmup、5 次预测，额外以源 `.mlpackage` 的直接 Core ML predict 作桥接 oracle。export、compile、bridge 均 exit 0。

| 诊断项 | 结果 | 能证明的范围 |
|---|---:|---|
| native 与直接 Core ML 输出 | **bitwise equal**，max error=0，全部 finite | C ABI/FP16 backing/布局桥接正确；两侧是同一 INT8 artifact，不能证明相对原 BF16 的质量 |
| 5 次 native predict 中位数 | 2.780 ms | 单 block 诊断值；当时有 Swift 构建和下载，不作正式性能对照 |
| native create | 6.045 s | 包含来源校验与 model setup |
| manifest validation | 5.407 s | 冷启动成本显著，后续完整请求比较必须计入 |
| Core ML model load | 0.497 s | 与 manifest validation 分开报告 |
| compute plan preferred device | 两个 conv、split、silu、mul 共 5 个计算 op 偏好 Neural Engine | 静态 compute plan；不能替代运行时驻留/dispatch 观测 |
| observed ANE residency | **unknown** | 未取得运行时硬件证据，不宣称已证明所有计算驻留 ANE |

逐项原始结果、compute plan、导出来源、native library/exporter SHA-256 已封存到 [2026-09-20-m1-qwen3-block0-bridge.json](2026-09-20-m1-qwen3-block0-bridge.json)。本节没有 GPU baseline、原 BF16 数值对照、完整 encoder 或模型端到端收益，因此不能据此选择最佳 GPU/ANE 划分。下一步需要在空闲测量窗口完成数值 oracle、分宽度对照和完整请求验证。

当前权重目录约 11 GiB，下载 session `70822` 与新 App build session `22293` 继续运行；后者日志 `/tmp/tc-flux-components-build.log`。原目标仍未完成：两模型真实 streaming 出图/性能、ANE 最佳划分、R1 后续身份迁移、R2/R4/R5/R6 及产品认证仍须继续。

## 第八轮：R2 options 候选筛选与安装绑定验证

修复未安装查询拿不完整 workload 做完整相等比较的问题：新增独立 `PresetCandidateResolution` / `find_streaming_preset_candidate()`，仅按基础模型、操作、尺寸、步数、设备、容器等字段发现候选；继续检查 catalog、release、内存适配与校准。`resolve_streaming_preset()` 保留原完整 workload/source/runtime 比较，候选不构成执行权限。

同时发现实际请求的另一个失配：图片请求默认 fps=24，而 Z/Flux probe 的图片身份为 fps=0。公共基础 workload 构造现在将 image fps 归零；video 保留实际帧率。非空 catalog fixture 覆盖候选发现、缺组件/token 身份时 exact 仍拒绝、不同 token 数仍拒绝 exact、错误尺寸/容器/设备拒绝候选，以及图片/视频 fps 规则。

`tc_streaming_options_json` 对未绑定安装的匹配项返回 `candidate + artifact_verification_required`，不再返回误导性的 `available`。Swift 在提供模型目录且存在候选时才打开 metadata session，逐档调用原严格 resolver，核对返回 selector、catalog 与 container；只有成功项显示 available，所显示 record/校准/release 信息取自严格解析结果。空 catalog 不打开模型、不分配 GPU 权重。生成仍重新解析、冻结 exact selector 并验证实际结果。

App query key 纳入模型路径、prompt、动态文本、输入和加速策略等 draft 信息，排除返回的 streaming 状态以避免刷新循环。请求版本号与 key 防止旧结果覆盖新 draft，查询开始时清除旧可用快照。尚无已验证 options 时不按物理 RAM 推荐；首次使用保持 Off，即使有可用 streaming card 也不据此自动开启。显式用户选择保持。UI 区分“待验证”和“已校验本地模型”，不将技术错误码当成正常操作说明。

验证与范围：

- native host resolver suite PASS，`/tmp/tc-options-resolver-tests.log`；非空 catalog/不同 token 测试在纯 C++ 层执行。
- Swift resolution/options fixture PASS，覆盖严格结果替换候选信息、解析失败、catalog 变化、取消与既有结果校验；`/tmp/tc-options-resolution-new-native.log`。此处的非空 options 使用可控 resolver fixture，不冒充真实生产 catalog 验收。
- Studio behavior PASS，新增 prompt/path/token policy query identity、发布状态不触发循环、无记录不打开不存在的模型目录；`/tmp/tc-options-studio-new-native.log`。
- 新 native-only build 到 `build/m1-options`，session `55680` exit 0；83 API contracts 中 80 PASS/3 缺 fixture SKIP，GPU self-test PASS。日志 `/tmp/tc-options-native-build.log`、`/tmp/tc-options-contract-tests.log`、`/tmp/tc-options-selftest.json`。
- 最新 App 源码独立编译成功，binary 为 `build/m1-options/TurboCiderOptionsApp`；`/tmp/tc-options-app-build.log`。针对 Studio/options 的二进制在新库下重跑；本轮没有宣称重跑所有 App integration binaries。

前节后台完整 Swift build `22293` 实际以 exit 1 结束，原因是本轮更新源文件时编译器检测到 `TurboCiderNative.swift` 被修改。它不计为通过；上述新 native-only 构建与固定源码的 App/Studio 编译验证取代其本轮证据。临时把 Studio binary 放在 `/tmp` 的首次执行也因找不到同目录 `turbocider` helper 失败；移至 native 构建目录后通过，未把该测试布置错误当作产品通过。

生产 catalog 仍为空，portable identity/schema、真实非空发布包与 container 身份仍待完成。R5 的完整 worker 成功事务/原子产物发布和 R4 失败隔离仍未关闭。

### 下载复用尝试被内容校验拒绝

通过 pinned Comfy revision 的 HTTP Range 读取 Z Qwen3 原始 safetensors header，确认它与已校验 Flux text encoder 的 398 个 tensor 名称/shape/dtype 一致。尝试按该原始 header 拼接 Flux tensor 数据，得到完整 8,044,982,048 字节临时文件，但 SHA-256 为 `e37269b7ca1301ad72a92627ce95432ab5aad5f89143a06055886aad3419d12f`，**不匹配** Z 官方 `6c671498573ac2f7a5501502ccce8d2b08ea6ca2f661c458e708f36b36edfc5a`。

因此未发布到 Z 模型路径，已删除本次临时重组文件，继续原官方下载。首段 216,288 字节 payload 相同也不能推导整文件相同；不将这次尝试记为下载节省或成功复用。失败记录 `/tmp/tc-z-qwen-reconstruction.json`。下载 session `70822` 仍在运行，完整两模型尚未齐备。


## 第九轮：LTX worker 结果验证与原子发布

原 worker 将 exit 0 和输出文件存在视为成功，可能接收空 JSON、旧文件或不可解码视频。现在每次创建同目录私有 staging，子进程只写 staging；父进程严格检查结果类型、model/operation/seed/steps/尺寸/输出路径。公共 streaming 还要求完整 verified summary、合法摘要、授权与实际 layout 一致，以及请求固定的 selector 字段匹配。

发布前检查普通非空文件、视频尺寸/时长/音频轨道，逐帧解码并核对帧数；流式计算 SHA-256，在读取前后核对文件身份。构造带 SHA/字节数的 worker_artifact receipt 后以同文件系统 rename 发布。错误或取消保留原产物并清理 staging。stdout 上限 8 MiB、events 上限 16 MiB，避免结果与诊断无限增长。

测试期间定位并修复一个真实退出等待问题：子进程已结束，协程仍阻塞于同步 waitUntilExit()；改用 terminationHandler 和异步 continuation 通知退出。取消保留 SIGTERM 后 5 秒 SIGKILL 的行为，异常清理也等待本次子进程退出后再删除临时目录。

新增 `tests/integration/LTXWorkerTests.swift`，用真正编码的 9 帧 H.264 视频和受控 worker 覆盖：成功发布、legacy 路径、空/错误 JSON、错误 seed/path/target/layout/preset、verified 类型错误、缺 summary、空/缺失/损坏文件、symlink、尺寸/帧数/音频不符、日志超限、发布失败、原文件保留及取消清理。它已接入 build_app.sh 和 test-app。

验证结果：固定源码的 worker 编译和测试 exit 0（`/tmp/tc-worker-build.log`、`/tmp/tc-worker-tests.log`）；StreamingResolution 与 Studio 回归 PASS（`/tmp/tc-worker-resolution-tests.log`、`/tmp/tc-worker-studio-tests.log`）；完整 App 编译 exit 0，产物 `build/m1-options/TurboCiderNativeApp`（`/tmp/tc-worker-regressions-build.log`）。新 App 另启动 45 秒保持存活，stderr 为 0 字节，随后由测试终止本次进程（`/tmp/tc-worker-app-smoke.json`）；这只覆盖启动。worker 测试使用受控子进程，未宣称真实 LTX 推理成功。

这仍不是完整 R5 关闭：尚缺独立 worker resolution envelope、job/request hash 协议与持久化有界诊断；当前 summary 容器身份仍待 R6 修正。非 LTX 产物事务、进程树超时及 R4 GPU 失败隔离也仍待处理。

## 第十轮：真实 Qwen3 前缀与 BF16 GPU 数值对照

新增 `tools/native/compare_qwen3_partition.py`，读取原 checkpoint 的 gate/up/down，在显式 MLX GPU 上计算 BF16 MLP 前缀，并与 native Core ML C ABI 输出比较。两侧接收同一 BF16 舍入后的输入，Core ML 以 FP16 传输；按 native Qwen 调用方规则恢复 output_scale。使用 layer 0、64 rows、prefix 4864、seed 42、input_scale 0.25，与第七轮配置一致。额外导出并编译同形状 FP16 artifact 作控制组。

| artifact | 相对 L2 | cosine | 最大绝对误差 | 全部有限 |
|---|---:|---:|---:|---|
| INT8 per-channel | 0.029056662 | 0.999578075 | 0.044921875 | 是 |
| FP16 | 0.026692391 | 0.999644048 | 0.035156250 | 是 |

原始结果见 [INT8 oracle](2026-09-20-m1-qwen3-prefix-oracle.json) 和 [FP16 oracle](2026-09-20-m1-qwen3-fp16-prefix-oracle.json)。两个 native session 均验证 checkpoint SHA-256；运行时 ANE 驻留仍 unknown。FP16 本身相对 BF16 也有差异，不能把 INT8 的全部偏差归因于量化，亦不能把两项 L2 直接相减当作量化误差。

复现命令如下，MANIFEST 为各 JSON 中记录的已编译 manifest，REPORT 为新的结果路径：

```sh
.venv/bin/python tools/native/compare_qwen3_partition.py \
  --model /Users/chencanhui/models/TurboCider/FLUX.2-klein-4B/text_encoder \
  --manifest "$MANIFEST" --library build/m1-options/libturbocider.dylib \
  --output "$REPORT" --block 0 --rows 64 --seed 42 --input-scale 0.25
```

这是随机 hidden states 上单个真实权重前缀的误差测量，没有设置事后通过阈值，也不代表完整 encoder/image 质量通过。报告中的内部 quality_validation_passed=true 伴随 quality_validation_enabled=false，不是本次数值比较的通过证明。测量期间仍有构建与下载，不以这些调用耗时评判收益或最佳 GPU/ANE 划分。官方 transformer 下载仍在进行，两模型端到端 streaming、完整 encoder 对照及空闲窗口性能实验仍待完成。


## 第十一轮：Flux 请求恢复进程级 MLX 缓存上限

检查真实运行入口时发现 Flux prepare/run 直接设置进程级 allocator cache limit，请求结束后未恢复，会影响嵌入者或下一个模型。现在使用作用域对象保存并恢复原值，异常路径也恢复；不额外改变清缓存时机。

新增 C API 回归 `tests/native/test_flux_cache_scope.py`：先把嵌入者 cache limit 设为 19 MiB，再用真实 candidate exact 请求进入 Flux，故意在 encoder 权重缺失处失败。旧库返回后变为 512 MiB（`/tmp/tc-cache-before.log`），修复后的库恢复 19 MiB（`/tmp/tc-cache-after.log`）。本机 prepare 被较早的保守 BF16 内存准入拒绝；它验证原设置不变，不能冒充已覆盖 prepare setter 后的异常。成功路径须结合接下来真实出图验证。

本轮只重编译变化的 pipeline 对象，并与上一轮固定 native 对象重链接为 `build/m1-cache/libturbocider.dylib`；`/tmp/tc-build-cache.sh`、`/tmp/tc-cache-build.log` 记录命令和 exit 0，未把它称为全量重构建。新增测试已接入 Makefile。

R4 检查再次确认 Flux/Z exact owner 仍有完整生命周期缺口：StageExecutor 在 drain 失败后析构会 terminate；adapter 引用外部 plan/source/event/cancellation/pass tensors，不能只泄漏 executor 或取消 terminate。需要完整 owner 保留、断开调用方回调、构造失败保护和所有 engine 入口的 poison 检查。此项本轮尚未修复，不宣称失败隔离已经安全。


新库的 Flux 4B candidate gate 与 4B/9B public adapter host 回归均 PASS（`/tmp/tc-cache-gate.log`、`/tmp/tc-cache-public.log`）；public catalog 仍为空并保持拒绝。host 编译有目标 macOS 26.0 与 native 库最低 26.2 的链接警告，本机 26.4.1 执行通过，不外推到更早系统。

已启动 `/tmp/tc-run-flux-smoke.py`（session `79830`），等待官方 transformer 原子下载完成后，先按 pinned revision 对全部已下载 Flux 文件做 SHA-256/Git blob 校验，再调用真实 candidate constructor：256×256、4 步、seed 42、GPU eager、denoiser prefix 0/group 1/K2/D1/Q2。实验目录为 `/Users/chencanhui/models/TurboCider/experiments/m1-flux4-streaming-smoke`，stdout 为 `/tmp/tc-flux-real-smoke.log`。记录本段时它仍在等待权重，不构成推理成功证据；后续需检查实际 result、runtime layout、完整 PNG 解码和图像内容。该脚本不安装测试 catalog，也不把 candidate 路径称为已发布 public 功能。


## 第十二轮：完整 Flux 4B encoder 对照准备

`benchmark_qwen3_encoder_sweep.py` 原来将 flux_klein 固定为 hidden 4096 / MLP 12288，会拒绝 Klein 4B 的 hidden 2560 / MLP 9728。现在允许这两组配对几何，仍拒绝交叉混用；K/N 必须一致、prefix 范围正确、Flux 必须包含 0–26 全部 blocks，Z 必须保留 2560/9728 和 0–34 blocks。新增 `test_qwen3_sweep_geometry.py` 覆盖上述接受/拒绝分支，本机 unittest 通过（`/tmp/tc-qwen-sweep-geometry.log`）。不以单 block manifest 冒充完整 encoder。

已用官方 Flux text encoder 启动完整 27-block、64-row、4864-prefix、INT8 per-channel 导出（`/tmp/tc-qwen-full-export.log`，session `59965`），路径 `/Users/chencanhui/models/TurboCider/experiments/m1-qwen3-flux4-b64-w4864-int8`。native qwen3-quant-probe 已针对新库编译到 `build/m1-cache`（`/tmp/tc-qwen-probe-build.log`）。记录时导出尚未结束，完整 encoder 的输出质量和收益均未判定。

Flux smoke 进程 PID 36144 已在开始推理前接入 process-tree sampler，50 ms 间隔/250 ms gap 上限，correlation `m1-flux4-official-smoke-20260920`，证据写入该实验目录的 `process-memory.jsonl`（session `78685`）。采样从等待下载阶段开始，仍需独立 verifier 验证完整性；不能把下载与导出并行期间的结果用作正式冷/暖性能资格。


### 官方 Flux 4B：真实 streaming 出图成功

Flux transformer 下载完成，完整大小 7,751,109,744 字节，SHA-256 与 pinned 官方 `9f29f9edcfdae452a653ffb51a534ca4decd389952c225724ff3b94042612a6e` 一致；其余已下载 Flux 文件一并校验。以同一狐狸/雪地/松林提示、seed 42、256×256、4 步跑了两次真实 BF16 candidate exact streaming。

首次 native 已完成导出，事件最后时间 9.442 秒；Python harness 随后因缺 Pillow 退出，未保存 native result。保留原图和 events，不将 harness exit 1 当作完整测试通过。第二次改用系统 ImageIO（sips）验证，并在后处理前先保存 result，在新目录重跑：native status 0、图像解码成功，request wall 8.116 秒，text 1.570 秒、denoise 6.182 秒、VAE 0.229 秒。两张 PNG 的 SHA-256 均为 `505668fa0966029c4f0d4f943b482c1620b2433c947d006824bfc76930919b02`。图像目视有狐狸、雪地和松林，无明显损坏；这不是多提示质量评测。

真实 result 确认 generic_stage_executor_v1、25 groups × 4 passes = 100 fills、两个 retained pools/四个 slot bundles、K2/D1/Q2、drained=true；逻辑读取 29,834,145,792 字节，MLX peak 6,269,857,144 字节。实际布局 digest 为 `87c3bf901d70dc9261544e67bce9618bfa91c8c305df2b369962646659157e5f`。来源由实验脚本按官方 hash 核验，但 candidate result 的 source_lease_verified=false，没有 public receipt；不能称为 public catalog 功能已认证。

原始结果及源校验见 [Flux smoke JSON](2026-09-20-m1-flux4-streaming-smoke.json)，产物见 [狐狸图像](2026-09-20-m1-flux4-streaming-smoke.png)。正式性能比较仍需固定并发/缓存/进程生命周期条件；上述两次均有下载或导出/编译背景活动。

首次 process-tree 证据 verifier 为 complete，tree peak footprint 6,356,833,344 字节；第二次 native 子进程 exit 0，但 sampler 因短命 sips 子进程 unknown_child 返回 inconclusive，不能用于整棵进程树内存资格。两份 verifier 结果均封存在 smoke JSON，未丢弃失败记录。首次的系统 compression 增量也包含其他进程活动，不归因于本模型。

### 完整 Flux encoder：50% 分区的初步结果

27 个 INT8 分区导出和 native compile 均 exit 0。64-token 固定 bucket、GPU/ANE MLP 宽度各 4864、原 checkpoint、确定性 token IDs、输出 tap 8/17/26，分别用独立进程执行 GPU 与 hybrid，一次首轮加 5 次暖态。

- 完整 conditioning 相对 L2 **0.008728185**、cosine **0.999992121**、相对最大误差 **0.007407407**，全部 finite，符合预先设置的 0.025 / 0.999 / 0.05 门槛；这不是图像质量证明。
- GPU 暖态中位数 **0.138848 秒**，hybrid **0.114904 秒**，比值 **1.208×**；hybrid CV **0.3224** 超出既定 0.25，因此本轮不能保留为稳定的推荐分区。
- GPU load **2.412 秒**，hybrid load **16.418 秒**；首个 encoder pass 分别 **1.635 秒**和 **2.402 秒**。冷请求明显无收益，不能用暖态小收益掩盖冷启动成本。
- Core ML 模型数 27、累计预测调用 162、没有 runtime failure；compute 配置不等于实测 ANE 驻留证明。

[原始 encoder report](2026-09-20-m1-qwen3-full-encoder-sweep.json) 保留原值。其中 hybrid_executed=false 暴露工具的第二个问题：它将 flexible backing 资格混入执行观测，而本次单固定 bucket 不使用 flexible backing。现将“全部 blocks/calls 是否实际执行”和 backing 资格拆开：单 bucket 不要求 flexible 标志，多 bucket 仍要求；并强化为核对首轮/后续/总调用数和零失败，拒绝仅执行部分 blocks 的假阳性。新增回归通过，不篡改这份历史原始结果。正在以 10 个暖态样本复测波动，结果出来前不提高任何阈值。


10 样本复测已结束：[第二份完整 encoder report](2026-09-20-m1-qwen3-full-encoder-sweep-2.json)。完整调用数为 297 = 27×11，hybrid_executed=true，fixed backing 资格检查通过；conditioning 数值与上轮一致。GPU 暖态中位数 0.139203 秒，hybrid 0.117130 秒，比值 1.188×；hybrid 第一暖态 0.335283 秒导致 CV 0.4710，仍超过 0.25，retain=false。未删除尖峰或提高阈值。Core ML load 从 14.607 秒降至 6.871 秒，说明系统编译/模型缓存影响 setup；不能把第二轮当作独立冷启动加速证明。

为继续探索划分，已启动 `/tmp/tc-qwen-width-sweep.py`，依次导出/编译/测量 2432（25%）与 7296（75%）前缀，均 27 blocks、64 tokens、INT8、10 暖态样本；日志 `/tmp/tc-qwen-width-sweep.log`。这些仍是完整 encoder 的探索性对照，未宣称 Flux denoiser hybrid 或完整图像请求收益。Z 官方模型仍在下载，目标整体保持未完成。


## 第十三轮：64-token 完整 encoder 的分宽度探索

25% 和 75% 分区均已完整导出/编译，并完成 27 blocks、一次首轮＋10 次暖态的 GPU 对照。没有删除暖态尖峰，沿用相同质量/速度/CV 阈值。

| ANE prefix / 9728 | GPU 暖态中位数 | hybrid 暖态中位数 | GPU / hybrid | hybrid CV | conditioning 相对 L2 | retain |
|---|---:|---:|---:|---:|---:|---|
| 2432（25%） | 0.139081 s | 0.136343 s | 1.020× | 0.3754 | 0.0078693 | false：速度及稳定性 |
| 4864（50%，10 样本复测） | 0.139203 s | 0.117130 s | 1.188× | 0.4710 | 0.0087282 | false：稳定性 |
| 7296（75%） | 0.145837 s | 0.303986 s | 0.480× | 0.0812 | 0.0038033 | false：速度 |

原始结果：[25%](2026-09-20-m1-qwen3-w2432-encoder-sweep.json)、[75%](2026-09-20-m1-qwen3-w7296-encoder-sweep.json)。各轮全部 297 次 Core ML 调用均成功，conditioning finite/cosine/相对最大误差也都通过既定阈值。观察范围内 50% 的暖态中位数最好，但尚无合格推荐；75% 更慢也说明不能按 ANE 占比推导收益。25%/75% 的 hybrid load 分别 13.593/18.897 秒，冷启动仍不利。

这些是 64-token encoder 探索，GPU/ANE 芯片实际驻留仍未观测，不代表完整图像收益，更不代表所有 token 长度的最佳划分。75% 后段与 native audit 构建有 CPU 并发，背景下载持续；精确性能结论须在后续固定条件复测。

已冻结真实 Flux 4B 同布局 P1 对照 [policy](2026-09-20-m1-flux4-p1-policy.json)：256²/4 步、P0/G1/K2/D1/Q2、每请求新 engine、直接调度 vs 通用执行器、10 个 ABBA/BAAB blocks 共 20 matched pairs、无结果后剔除、原阈值不变。带 audit counters 的全量 native-only build 正在 `build/m1-audit` 进行（`/tmp/tc-m1-audit-build.log`，session `82430`）；构建与 encoder 实验结束后才启动计时。本段不构成 P1 通过证明。


### 当前 hybrid 完整请求仍被 runtime 拒绝

用 native tokenizer 构造了恰好 64 token 的自然提示，给同一 Flux candidate exact 请求增加已编译的 50% encoder manifest 与 allow_approximation=true。`build/m1-cache/turbocider plan /tmp/tc-flux-e2e-hybrid-request.json` 明确拒绝：`streaming_route_unsupported: manual streaming requires GPU-only execution`，来源 `native/runtime/plan.cpp`。因此没有运行这份完整 hybrid 请求；encoder sweep 不能被描述为完整 streaming hybrid 已支持。后续必须接入受约束的 hybrid route/身份/资源所有权，不能只移除门禁。

全量 audit native-only 构建 session `82430` 已 exit 0，日志 `/tmp/tc-m1-audit-build.log`，确认导出 audit reset/snapshot C API。已启动真实同布局 P1 campaign session `29177`，目录 `/Users/chencanhui/models/TurboCider/experiments/m1-flux4-p1`，日志 `/tmp/tc-flux4-p1.log`。记录本段时尚未完成；默认 environment 记录仍为 partial，背景 Z 下载持续，即使输出与计数器通过也不能忽略环境资格。27-block 分宽度导出/测量已全部结束，不与 P1 GPU 请求并发。


## 第十四轮：真实 Flux P1 同布局结果

audit build 的 10 blocks / 20 matched pairs 已完整结束：40/40 请求成功，无 fault，20 对 PNG SHA-256 全部相同；逐请求实际语义与冻结的 4B/256²/4-step/K2/D1/Q2 布局一致。计数器确认稳态 framework allocations=0、thread creates=0。来源、样本数、协议和 audit 证据通过。

| 量测 | 直接调度 baseline | 通用执行器 candidate | candidate / baseline | bootstrap 95% 比值上界 |
|---|---:|---:|---:|---:|
| 完整请求中位数 | 8.456937 s | 8.372259 s | 0.989987 | 1.004772 |
| 完整请求 P95 | 8.629044 s | 8.643324 s | 1.001655 | 1.004794 |
| denoise 中位数 | 6.200536 s | 6.130822 s | 0.988757 | 1.004509 |

数值区间落在冻结的 1.02/1.05/1.02 上限内，但**总体结果仍为 INCONCLUSIVE**：environment.json 是 partial，背景模型下载持续，未取得完整电源/热状态/SSD 环境记录。没有修改 environment 状态来取得 PASS，也不据此发布认证卡。它提供了真实 4B 上逐图一致、完整布局执行和框架开销的证据，不能替代完整发布门。

封存：[summary](2026-09-20-m1-flux4-p1-summary.json)、[audit](2026-09-20-m1-flux4-p1-audit.json)、[quality](2026-09-20-m1-flux4-p1-quality.json)、[semantics](2026-09-20-m1-flux4-p1-semantics.json)。完整原始 bundle 位于 `/Users/chencanhui/models/TurboCider/experiments/m1-flux4-p1`，两侧 native SHA-256 为 `1d6761714a05f18a0c31c0185a5db32851f8c013b7af4d5225ab9f6f4533e90f`。session `29177` exit 1 对应验证器 INCONCLUSIVE，并非请求失败。


## 第十五轮：App 图片产物验证与原子发布

非 LTX 图片请求此前直接写最终输出路径，公共结果校验失败时也可能已覆盖目标。新增 ImageOutputTransaction：请求只写同目录 0700 私有 staging，公共 selector 对暂存路径重新解析并冻结；父层先验证原 native result 与冻结 request，再验证图片、生成含 SHA-256/字节数的 receipt，最后同步 rename 发布。历史中的用户请求和返回 receipt 使用最终路径。

校验要求 model/operation/seed/steps/尺寸/path/warmup 的严格类型与请求匹配，输出是普通非空 PNG、单图、正确尺寸且完整解码；读取前后与发布前对 held-fd/path 的 inode/size/mtime/ctime 做一致性检查。后台验证任务参与 App 取消，最后发布前再检查取消状态。失败清理暂存目录，推理/校验/取消/rename 失败不会替换原文件。

首轮测试发现 ImageIO 能容忍某些截断 PNG，仅检查 decode/status 不足以证明完整。因此增加有界分块的 PNG signature/IHDR/IDAT/IEND 边界和逐块 CRC-32 校验，要求 IEND 完整且位于文件末尾，再进行完整栅格解码。新增测试现已通过：错误 JSON 字段、空/缺失/非 PNG/截断/错误 CRC/符号链接、尺寸不符、正常 SHA receipt 与发布、重复发布拒绝、验证后替换 inode、取消、发布目标不可替换及暂存清理。日志 `/tmp/tc-image-transaction-tests.log`；首次失败日志由测试运行记录，未把容错解码当作通过。

该修改不宣称完整 R5 已关闭：视频路径仍使用独立 LTX 事务，其他非 LTX 视频未统一；跨 jobs.json 与产物的崩溃一致提交、持久化 worker diagnostics、独立 job/request 协议仍待完成。已提交产物后的历史写盘失败，也不属于本次原文件回滚承诺范围。固定源码的完整 App、Studio、StreamingResolution 编译和测试均 exit 0（session `92350`）。图片事务、StreamingResolution 和 Studio 日志分别为 `/tmp/tc-image-transaction-tests.log`、`/tmp/tc-image-resolution-tests.log`、`/tmp/tc-image-studio-tests.log`，App 为 `build/m1-audit/TurboCiderNativeApp`。

新版 App 启动 30 秒保持存活，随后仅终止本次测试进程；`/tmp/tc-image-app-smoke.json` 记录结果，stderr 为 0 字节。这是启动检查，未冒充全部 UI 流程或 public 模型生成验收。

## 第十六轮：Flux exact owner 的失败隔离

新增 [真实权重生命周期记录](2026-09-20-m1-flux4-quarantine.json) 和可复跑的 `tools/native/check_streaming_owner_lifecycle.py`。这是 Flux Klein 4B BF16 的私有 candidate/test-hook 路径，尚不构成 public catalog 发布资格。

- adapter 持有当前 pass 的 Tensor/调制向量和 Event 副本，避免异常离开 denoise 栈后只保留 executor、却丢失其引用对象。失败 drain 会断开 API 栈回调，保留 pass/current weights；I/O 仍先 shutdown/join，未 detach。
- pool 初始化移到完整 exact owner 建立后的 start，覆盖 begin 部分失败的保留路径。安全 drain 后释放 owner；无法确认完成时 session 标记 quarantine，API 将其提升为进程级状态。`tc_engine_free` 使用无分配的链表保留整个 engine；后续 GPU generate/load/prepare/unload 拒绝运行，同进程新建 engine 也不能恢复 GPU 使用。
- API 同时保留 primary cancellation 和 drain incomplete 信息，返回 runtime failure。App 清除可复用句柄并显示“GPU 状态未恢复 · 请重启应用”，不再显示可重试。

真实权重检查三项通过：正常生成；首个 transformer block 取消后同 engine 再次生成；测试钩子令 drain 返回 false 后 generate/prepare/load/unload 全部拒绝、free 不 terminate、新 engine 的 load 也拒绝。正常和取消后重试的 PNG SHA-256 均为 `505668fa0966029c4f0d4f943b482c1620b2433c947d006824bfc76930919b02`，与原始真实 Flux smoke 逐字节相同。unsafe 用例记录 2 个完整 engine 保留至 disposable 测试进程退出，拒绝入口没有继续回调。

`make test-streaming-host` 通过；包含新增 quarantine 状态断言的 StudioBehaviorTests、StreamingResolutionTests、ImageOutputTransactionTests 通过，App 编译通过。native 完整 test-hook/audit 构建通过，之后针对 API 补充入口检查重新编译和链接通过。增量链接脚本曾因 H3 对象清单不完整/包含 probe main 失败，修正为 build.sh 的精确 runtime 对象清单后通过；未把失败链接当成测试成功。

限制：本轮只验证 Flux exact denoiser。Z-Image owner、encoder/VAE 阶段 GPU 失败、更多取消边界及真实后端 hang 仍未关闭；同步 backend 和 join 仍可能无界阻塞，不承诺故障后 60 秒内回收。测试钩子只模拟 drain 返回失败，没有制造 GPU hang。完整 R4 和 streaming 总目标仍未完成。最新 `origin/dev` 再次 fetch 后确认已是当前 HEAD 的祖先，无新增待合并提交。

隔离改动后又完成一轮冻结 P1（[policy](2026-09-20-m1-flux4-quarantine-p1-policy.json)、[summary](2026-09-20-m1-flux4-quarantine-p1-summary.json)、[audit](2026-09-20-m1-flux4-quarantine-p1-audit.json)、[quality](2026-09-20-m1-flux4-quarantine-p1-quality.json)、[semantics](2026-09-20-m1-flux4-quarantine-p1-semantics.json)）：40/40 成功、20 matched pairs PNG 相同、steady framework allocations/thread creates 均为 0。原 direct 基线库不变，新 generic 库 SHA 为 `e06dc49a0b8c6770857c70e2c2f96e2d50ca5ef1b0fb5dcefd8f51d6582c1c03`。wall median 8.52517 → 8.42341 s，ratio 0.988063、95% CI upper 1.005300；wall P95 ratio 0.983265、upper 1.008404；denoise median ratio 0.993680、upper 1.009260。数值位于冻结的 1.02/1.05/1.02 阈值内，但环境仍 partial（下载及启动阶段 App Swift 构建），正式结论 **INCONCLUSIVE**，不提升成 PASS 或生产资格。原始 bundle 为 `/Users/chencanhui/models/TurboCider/experiments/m1-flux4-quarantine-p1`。

## 第十七轮：Z-Image 官方权重真实 streaming 首次完成（9 月 21 日）

Z-Image 三份 Comfy-Org 权重全部下载完成并逐文件 SHA-256 校验，固定 revision `08d04455279082882deaabc8d0d09fc914c071e1`：transformer 12,309,866,400 bytes、Qwen3 8,044,982,048 bytes、VAE 335,304,388 bytes。另从 Tongyi-MAI/Z-Image-Turbo 固定 revision `f332072aa78be7aecdf3ee76d5c247082da564a6` 获取 tokenizer；tokenizer.json 按 LFS SHA-256 校验，tokenizer_config.json 按 Git blob SHA-1 校验。校验值、请求和完整 native 结果见 [记录](2026-09-21-m1-z-image-streaming-smoke.json)。没有发布或复用之前 SHA 不匹配的重建文件。

[输出 PNG](2026-09-21-m1-z-image-streaming-smoke.png) 为可辨认的雪地红狐、松树场景；SHA-256 `8a3e89a095a124aba019f09e47a7f34248d34680e4fd127494be8babb5848be3`。本地候选库为 `build/m1-quarantine/libturbocider.dylib`，该次 Z-Image 尚未包含其自身的后续 R4 owner 修复。

配置为 256×256、9 steps、seed 42、26 个实际文本 tokens，GPU BF16 exact P0/G1/K2/D0/Q1，30 groups × 9 passes = 270 fills，1 个 pool、2 个 slot bundles、1 个 refill worker。actual layout drained=true；私有 candidate 的 source_lease_verified=false，不作为 public preset 资格。逻辑 request_bytes_loaded 为 99,146,778,752 bytes，含 fixed weights 和每步 block 重读；不能当成物理 SSD 读量。

native request wall 32.8608 s，text encode 1.82564 s、denoise 29.90850 s、VAE decode 0.59175 s。MLX peak 为 8,978,303,336 bytes，结束 active 为 336,278,176 bytes；该指标不含 OS、file cache 或 Core ML。请求中的 8 GiB 是旧 denoiser budget 域，非已验证 request footprint 上限：不能声称该次运行通过 8 GiB 公共档位。后台 native 构建可能影响时延，本次是成功运行 smoke，未完成性能资格或数值 oracle。

App 新二进制再次直接启动并保持 30 秒，无提前退出、stdout/stderr 均为 0；随后仅终止测试自己启动的进程。这个观察证明可启动，不等于图形界面全部工作流都已验收。Z-Image owner 故障隔离和 GPU/Core ML encoder 对照仍在继续。

## 第十八轮：Z-Image exact owner 与取消异常传播

将 Flux 的完整 owner 保留方式接入 Z-Image exact denoiser：adapter 拥有 pass 的 unified/freqs/temb 和 Event 副本；完整 owner 建立后才 begin pool；unsafe drain 保留 owner/固定权重/lease 和 engine，断开回调并触发同一进程 quarantine。

真实首 block 取消测试发现：后台 refill 在 cleanup 中记录的 `generation cancelled` 覆盖了 primary `Cancelled`，导致 C API 返回 1 而非取消状态 2。Flux 同样存在该竞态。因此两 adapter 现在优先保留 typed cancellation，将 executor 的显式 `streaming_cancelled` 转为同一类型；非取消异常保留 primary 信息，再附加 fill detail，不让 cleanup 重命名原始错误。

[六项真实权重检查](2026-09-21-m1-streaming-owner-lifecycle.json)全部通过：每模型分别正常成功、首 block 取消后同 engine 成功重试、注入 false drain 后拒绝 generate/load/prepare/unload 且 free 不 terminate、新 engine 不能恢复 GPU 使用。Z-Image 首轮错误分类测试的失败被保留在 `/tmp/tc-z-quarantine-cancel-retry.log`；修复后的独立结果目录为 `/Users/chencanhui/models/TurboCider/experiments/m1-final-owner-tests`。两模型成功和重试图像都与各自先前 PNG 的 SHA-256 完全一致。

这推进了两模型 exact denoiser 的 R4 验收，但不关闭全部 R4：没有把假 drain failure 当成真实硬件 hang，也未覆盖 encoder/VAE 的 GPU 故障、全部 I/O/分配/取消边界、worker 有界恢复。legacy Z-Image streaming 的异常清理也不是本轮完整 owner 保留证明的范围。

## 第十九轮：无故障注入钩子的构建复跑

完整 native 构建输出 `build/m1-release`，未启用 test hooks/audit instrumentation；检查确认无 `tc_engine_test_streaming_drain_failure` 或 test catalog setter 导出。随后修复一个结果字段 bug：Z-Image 的普通 BF16 GPU generate 分支未设置 backend/precision，导致返回 null；现在明确返回 `mlx_cpp_metal` / `bf16`，不改动数值计算。更新该对象后重新链接成功。

[两模型复跑记录](2026-09-21-m1-streaming-release-smokes.json)：Flux 8.22962 s、Z-Image 33.82261 s，二者 PNG 均与先前 SHA 一致。库 SHA-256 为 `0d24ae5f29307b461a03d00b855f0f51f0f0d469c1641f8f3fdc60e88f0ca3df`。Z-Image backend/precision 断言通过，metadata-only public adapter 合约测试通过（四组件 lease、身份/layout snapshot、路由拒绝、目标失败清理与源替换检测）。这些是本机 smoke 时间，不是新的受控 P1 统计。

`check_streaming_owner_lifecycle.py` 的 success/cancel-retry 模式也可用于无测试钩子构建；quarantine 模式明确要求 test-hook 构建。上述运行仍通过研究用 candidate constructor，不把无测试钩子误称为 public preset 已发布。正式 streaming catalog、R1/R6 身份链及完整 R4/R5 验收仍未完成。

最新 App 使用已验证且源码未变的 Swift 二进制，配合新构建的相邻无测试钩子 native 库装配至 `build/m1-release/TurboCiderNativeApp`。再次启动观察 30 秒，无提前退出、stdout/stderr 为 0，随后只终止该测试进程；不将启动观察当成完整 GUI 工作流验收。

## 第二十轮：两模型 Qwen3 分宽度热态与跨进程复测

先保留 [Z-Image 首轮](2026-09-21-m1-z-image-qwen3-initial-sweep.json)：35 层、64 tokens、50% INT8 prefix 的 warm median 比 GPU 快 1.609×，但 GPU 前两次 post-first 调用为 2.99209/0.719663 s，GPU warm CV=1.1973，故 retain=false。没有删除这些样本或把原失败结果改成通过。

为区分加载/稳定化/常驻热态，新实验预先固定“1 次 first + 5 次 settling + 10 次 measured”，每个后端都执行同样次数，保留 first、全部 post-first、settling、measured 数组及加载时间。工具新增 `--settling-runs`（默认 0，保留原协议）与 `--first-backend`；运行前写出 plan.json。数值/速度/稳定性阈值仍为 relative L2 ≤0.025、cosine ≥0.999、relative max abs ≤0.05、warm speedup ≥1.1、双方 warm CV ≤0.25。单元测试验证计数不丢失、全部后端实际执行 settling、非法计数/非有限时间拒绝、加载顺序轮转及 Core ML 调用总数。

首轮 settled sweep 的相关记录：Z-Image [25%](2026-09-21-m1-z-image-qwen3-w2432-settled-report.json)、[50%](2026-09-21-m1-z-image-qwen3-settled-sweep.json)、[75%](2026-09-21-m1-z-image-qwen3-w7296-settled-report.json)；Flux [25%](2026-09-21-m1-flux4-qwen3-w2432-settled-report.json)、[50%](2026-09-21-m1-flux4-qwen3-w4864-settled-report.json)、[75%](2026-09-21-m1-flux4-qwen3-w7296-settled-report.json)。这轮期间曾有 native 构建，且部分分区发生明显跨进程排序变化，不能直接选一次最快的结果。

随后在我们启动的下载、导出、构建、完整图像推理全部结束后，预先冻结两轮独立进程复测，第二轮 hybrid 先运行，第三轮 GPU 先运行，并反转模型/宽度顺序：[完整计划](2026-09-21-m1-qwen3-repeatability-plan.json)、[全部 12 个复测结果](2026-09-21-m1-qwen3-repeatability-results.json)。下表为 warm GPU median / hybrid median；大于 1 才更快。

| 模型 | MLP prefix | 首轮 settled | 独立复测 2 | 独立复测 3 |
| --- | ---: | ---: | ---: | ---: |
| Flux 4B（27 层） | 25% / 2432 | 1.001× | 1.024× | 1.027× |
| Flux 4B | 50% / 4864 | 0.584× | 1.217× | 1.218× |
| Flux 4B | 75% / 7296 | 1.204× | 1.205× | 1.205× |
| Z-Image（35 层） | 25% / 2432 | 1.288× | 1.304× | 1.309× |
| Z-Image | 50% / 4864 | 1.608× | 1.606× | 1.611× |
| Z-Image | 75% / 7296 | 0.897× | 2.028× | 2.031× |

在后两轮、固定 64-token 常驻条件下，Flux 50%/75% 很接近（约 0.114/0.115 s，GPU 约 0.139 s）；Z-Image 75% 最快（约 0.176 s，GPU 约 0.357 s）。所有分区最终 conditioning 质量检查通过；relative L2：Flux 25/50/75% 为 0.007869/0.008728/0.003803，Z-Image 为 0.007553/0.008996/0.010440。每次 sweep 的 hybrid 调用总数严格为 Flux 432、Z-Image 560，没有 runtime failure，固定 bucket backing 检查通过。

这些仍是研究结果，不是默认路由资格：

- 首轮的显著回退仍有效，原因未定位；不能将排序变化直接归因于 ANE 硬件驻留、温度或编译干扰。Core ML 使用 CPU+NE 配置，调用计数不证明每个算子实际在 ANE 执行。
- 后两轮 hybrid load 仍约 7.9–9.1 s，first 和 settling 另计；不是零开销加速。真实 App 每次文本编码会释放 GPU text weights，不能直接套用常驻 probe 的热态时延。
- probe 使用确定性 64-token 输入；图像 smoke 的 Flux/Z 提示词分别是 30/26 tokens。生产 padding/profitability 策略没有被修改，probe 的显式 eligibility override 不授予实际 App 路由资格。
- 目前 private manual streaming 与 encoder ANE 的组合仍被计划器拒绝；尚未建立该组合的 typed identity、stage ownership/receipt 和端到端图像质量、内存、延迟证据。因此没有宣称“图片快 1.2×/2×”，也未写入自动加速策略或 public catalog。

下一步优化重点仍包括 Z-Image GPU refill 等待（首次完整图像报告约 17.26 s），以及把可重复的编码器候选接入正确的多阶段生命周期，再测完整请求；不能只删除 GPU-only guard。


## 第二十一轮：Z-Image exact refill 的 D/Q 并行探索（2026-09-21）

原先 Z-Image exact descriptor 只允许 D0/Q1，限制了双槽布局的预取重叠。本轮允许 K2/G1 下 D0/D1、Q1/Q2 的四种组合，仍拒绝 D≥K 或 Q>K；K1 保持 D0/Q1。两个 worker 各自写入独立槽，pread 保持并行，只在完成后对共享字节/次数/耗时计数加锁。F_NOCACHE、4 MiB 读取分块、数值 kernel 和池容量没有改变。

新增 [benchmark_z_image_exact_prefetch.py](../../tools/native/benchmark_z_image_exact_prefetch.py)，运行前冻结计划，先一次 D0/Q1 warmup，再按镜像顺序测量四种组合各两次。使用同一进程、每次新建 engine；wall 从 engine 创建计至安全销毁，包含 PNG 导出，不含事后文件哈希和审计序列化。模型、提示词、seed 42、256²、9 步、P0/G1/K2 与前述 Z-Image smoke 相同，均走 private candidate constructor。

证据：[预先记录的计划](2026-09-21-m1-z-image-exact-prefetch-plan.json)、[全部样本与汇总](2026-09-21-m1-z-image-exact-prefetch-summary.json)、[逐请求审计和内存](2026-09-21-m1-z-image-exact-prefetch-audit.json)、[D1/Q2 生命周期测试](2026-09-21-m1-z-image-exact-prefetch-lifecycle.json)。完整原始输出位于本机 models/TurboCider/experiments/m1-z-image-exact-prefetch。

| 布局 | 两次请求 wall（秒） | wall 中位数（秒） |
| --- | --- | ---: |
| D0/Q1 | 32.999、32.022 | 32.510 |
| D1/Q2 | 28.436、26.889 | 27.663 |
| D1/Q1 | 32.050、33.062 | 32.556 |
| D0/Q2 | 32.499、32.518 | 32.508 |

D1/Q2 的请求中位数比 D0/Q1 减少约 14.9%（约 1.175×），refill 等待中位数约从 17.188 s 降至 12.503 s。单独增加 D 或 Q 没有观察到相近收益。所有 8 个测量请求及 warmup 都成功，PNG SHA-256 均为 `8a3e89a095a124aba019f09e47a7f34248d34680e4fd127494be8babb5848be3`。每个测量请求均为 270 次 slot fills、一个双槽池、drained=true；steady framework allocations/thread creates 均为零，初始化 worker 数与 Q 一致。并行 worker 的读取耗时相加可以超过 wall，不能将它当作串行关键路径；逻辑加载字节也不是物理 SSD 读取量。

MLX peak 为约 8.88–8.98 GB，最大 8,975,747,452 bytes；该计数不含 Core ML、OS 和文件缓存。请求中的旧 8 GiB denoiser budget 不能解释为完整请求内存上限，本轮不构成 8 GiB 发布资格。每组只有两个样本，且未完成正式环境资格核验，因此以上只是探索结果，不是 P1/P2 PASS，也未写入 public catalog 或自动默认策略。

构建为带测试 hook/audit 的 `build/m1-z-prefetch/libturbocider.dylib`，SHA-256 为 `ff97422078dbc791812bbdcc8d97a69246eee903454d420f93e5e83a4a7e7538`。descriptor 的四组合及越界拒绝测试通过；GPU weight-stream suite 共 10 项，4 项通过、6 项因 ConvRot/suffix 设备资格跳过。其中双线程各 1000 次独立槽 refill 验证了精确计数、字节数及 GPU 读取值。D1/Q2 首块取消返回 cancelled，原 engine 重试成功且图片一致；注入 unsafe drain 后保留 owner，并拒绝旧/新 engine 后续 GPU 使用。最新 App 的既有启动结果见前文，本轮没有重新构建或宣称验证 GUI 的 D1/Q2 操作流程。


## 第二十二轮：Z-Image 可移植布局身份接口（2026-09-21）

为推进 R1，将文件内容身份与本次打开的文件绑定拆开：新增 `StreamingMetadata::describe_verified()`，要求调用方传入已经由 native 内容验证得到的 lease。checkpoint identity 使用全 lease 的 logical id/size/SHA-256 摘要，transformer artifact 使用其真实内容 SHA-256；路径、inode、mtime、ctime 和 generation 留在请求 lease 中。接口继续从 held fd 解析 header，并在描述时检查文件和命名路径，没有在 options/describe 中添加大文件哈希。

执行 `.venv/bin/python tests/native/test_z_image_streaming_descriptor.py` 通过。新增案例验证：原样复制到另一目录后 portable layout digest 相同而 binding/generation 不同；legacy snapshot layout 与 portable layout 不同；workload 步数或 D/Q 改变会改变布局；已验证 generation 的再次 capture 不读 payload；仅填写 caller content digest 仍被拒绝；同大小 payload 改写导致旧 lease 拒绝、新布局变化；删除文件导致拒绝；只有 tokenizer 改变时，全模型内容身份与布局也变化。测试使用临时 sparse safetensors fixture，由 native 实际读取全部逻辑内容验证，退出后清理。

这是后续 RecordV2 迁移所需接口与反例测试，尚未切换 public probe/compile、Python catalog builder 或 receipt schema，也没有 persistent import proof。现有 private/public legacy 路由保持其版本语义；不能据此宣称跨安装 public resolve 已通过。该轮未改变数值执行路径，没有复用旧图片性能数据作为新发布证据。

本轮 `git fetch origin dev` 成功，随后 `git merge-base --is-ancestor origin/dev HEAD` 返回 0，当前分支已包含最新 dev，无新增合并内容。


## 第二十三轮：Catalog 内容身份 v2 格式（2026-09-21）

新增显式 `source.identity_version=2`，只编码 model_variant、weight_format、artifact_manifest_digest，禁止把 source_snapshot_digest 放入 portable record。省略版本或显式 1 保留 legacy 字段与既有摘要；未知版本拒绝。native canonical record/digest/source identity 与 Python record/record-id/digest 使用独立 v2 domain；Python builder revision 更新为 v2，native test-catalog JSON 导入和导出同步处理版本。

原 v1 固定摘要 `13b5797176d713924b9c16857acbf0f7a35313d2426a22fdca38c488b3ea3df1` 保持不变，v2 native/Python 固定样例均为 `c322e18cabc2ed4756a8cbbc468c24989ba175dda64745902014a1971aca0bb0`。这验证的是序列化合同，不证明某个输入 JSON 的内容来源可信。v2 probe 在尚未迁移的 public resolver 中被明确拒绝；现有 Z-Image probe 仍为 v1，不能用 v2 record 授予执行权限。persistent import、verified lease 接入、request binding/authority/receipt 分离及正式证据重采集仍待完成。


验证结果：`test_streaming_preset_resolver.py` host suite 通过；`test_streaming_catalog_builder.py` 16 项通过；完整 native hook 构建成功，`test_streaming_test_catalog.py` 3 项通过，其中新库实际接受 Python 编码的 v2 record，但当前 Z-Image adapter resolve 返回 `artifact_verification_required`；带 snapshot 的 v2 JSON 被拒绝。既有 v1 catalog 生成、CLI 导出、安装、resolve、失败替换保留原快照仍通过。hook 库路径 `build/m1-catalog-v2/libturbocider.dylib`，SHA-256 `9c6b5f7bbf3bb0f18c2ada534db3f12b1df2643dc4d62c492993464cb6650684`；release hook 缺席检查使用前述保留的 m1-release 库，不宣称本轮重建了 release App。没有进行 GPU 性能复测或提升 public catalog。


## 第二十四轮：V2 resolver 内容证明与请求权限分离（2026-09-21）

核心 resolver 不再一律拒绝 v2 probe：只有 native verified lease 的全内容摘要与 record identity 一致、且 portable identity 不含 snapshot 时才接受。metadata-only lease 或 caller 提供的 digest 不会获得内容信任。probe/snapshot 仍必须共享同一 lease 实例；select/authorize 检查命名路径与 held fd，coordinator 的执行前检查继续生效。authority 除 portable source identity 外，独立保存本次 lease generation 和 snapshot digest，不能因内容相同而跨请求复用。

新增真实临时文件 host 案例：相同内容复制到另一路径后能选择同一 v2 record，并分别授权；原 authority 对复制请求 matches=false；原 probe 搭配复制 snapshot 授权被拒绝；metadata-only lease 被拒绝；同大小文件改写使旧 lease 失效，重新验证后的新内容也不能匹配旧 record。ValueProbe 另测 verified v2 成功，以及未验证 lease、混合 snapshot 的 v2 拒绝。

这完成 core resolver 的身份/权限接缝，不代表模型 public adapter 已迁移。Z-Image/Flux probe 仍捕获 legacy snapshot，导入验证入口和 persistent proof、portable layout 的实际模型执行接入、生产 catalog 和真实端到端验收仍待完成。本轮不变更 GPU/ANE 路由，不据 host 测试宣称 GPU 性能或 App 验收通过。

验证：`.venv/bin/python tests/native/test_streaming_preset_resolver.py` 与 `.venv/bin/python tests/native/test_streaming_source_lease.py` 均编译并通过（host-only，无 GPU）；覆盖既有 v1 authority、coordinator、receipt/drain 回归及上述新增 v2 案例。


## 第二十五轮：Z-Image 编译与执行重建保留内容身份（2026-09-21）

检查发现 `describe_verified()` 虽已存在，实际 `StreamingPlanView(lease, ...)` 仍固定调用 legacy describe，会在传入 native verified lease 时丢失 portable layout identity。现已让该构造器根据 native 内容证明选择 verified descriptor；metadata-only lease 与路径构造保留旧布局。public compile 与 exact execution 的 lease 构造均复用此接口，因此后续接入已验证 probe 时，两处重建遵守同一规则。`z_image_public_source_identity()` 对 verified lease 同步产生 v2 全内容身份，不再写 snapshot。

`.venv/bin/python tests/native/test_z_image_streaming_descriptor.py` 编译并通过：新增实际 plan 构造断言，其摘要与 verified descriptor 编译结果一致；复制安装和 capture_preverified 后的 plan 摘要相同，lease 实例/generation 仍分离。现有 metadata、布局边界和 stale source 拒绝测试继续通过。该轮尚未新增 public import/verify 入口，当前模型 probe 仍 capture metadata-only lease；没有宣称公开 v2 模型已可运行，也没有新 GPU 性能或 App 验收结果。

另使用当前托管 MLX headers 对 `native/models/z_image/z_image.cpp` 执行 clang++ C++20 `-fsyntax-only` 检查，返回 0；此项不等同于重新链接完整 native 库。


## 第二十六轮：显式 native 模型内容验证入口（2026-09-21）

新增 additive C API `tc_engine_verify_streaming_sources_json`。Z-Image 实现通过同一 held fd 验证 transformer、text encoder、VAE、tokenizer 的完整 SHA-256，返回全内容身份、文件摘要、读取字节和缓存命中数；支持 `tc_engine_cancel`，成功/取消/失败分别返回 0/2/1。该操作持有 engine mutex、不使用 GPU lock、不加载 GPU weights，其他尚未实现的模型返回 unsupported。

验证请求使该 Z-Image engine 进入内容身份模式；之后 public probe 只调用 capture_preverified，不在 options/resolve 中隐式哈希大文件。文件变化或证明缺失会返回 artifact_verification_required；不会静默退回 legacy。验证缓存目前仅在 native 进程内有效，API 返回的 JSON 不能自行恢复信任，也不是 persistent import proof。公开 catalog 仍为空；验证内容并不授予生成资格。


完整 native hook 构建 `build/m1-source-verify` 成功，库 SHA-256 为 `b2537bf2d532b9035cd4ec0b0c0ef812fb4170434eced06070cd016d309b8e10`。`test_streaming_test_catalog.py` 3 项通过，包含新 API 的首次验证/缓存命中、native/Python v2 digest 一致、验证后 public resolve 成功、文件改写后证明失效，以及重新验证后旧 record 仍拒绝新内容。release hook 缺席项使用保留的 m1-release 库，不代表本轮重建 release App。

[真实模型验证及生成记录](2026-09-21-m1-z-native-source-verification.json)：首次发出取消后，API 在本次调用 1.040 s 返回状态 2；这是单次观测，不是所有文件系统下的取消时限保证。随后重试用 10.652 s 验证 20,701,575,490 bytes，四份摘要均匹配此前固定 revision 下载的 SHA-256；再次调用耗时 0.001880 s，verification_bytes_read=0、cache_hits=4。全内容摘要为 `6668c65bf99c32a7c16a73396b611abf39f2eee4f9325a0180f7e5d7a3bfe759`。

同一 public engine 随后通过[合成测试 catalog](2026-09-21-m1-z-verified-public-catalog.json)运行[真实请求](2026-09-21-m1-z-verified-public-request.json)：256²、9 步、seed 42、P0/G1/K2/D0/Q1、目标 12 GiB。请求 wall 为 33.304 s，native denoise 28.684 s，PNG SHA-256 为 `8a3e89a095a124aba019f09e47a7f34248d34680e4fd127494be8babb5848be3`，与既有 private smoke 字节一致。authorized/actual layout digest 相同，actual_plan_verified=true、source_lease_verified=true、drained=true，270 fills 与 270 reader fences 均完成。MLX peak 为 8,975,747,432 bytes，不是完整 process-tree footprint。

测试 catalog 的 calibration 数字和 TEMPLATE performance 是 hook 生成的合成输入，**不是实测资格或发布证据**；生产 catalog 仍为空。该结果证明 native 验证→v2 record→公开 adapter→实际 receipt/图片这条链跑通，不能据此发布 12 GiB 预设。报告 execution_container 仍沿用 embedded_app，而本次实际由 Python host 调用；这也是 R6 待修的问题。App 尚未接入显式验证 UI，持久化 import proof、Flux 验证迁移、真实环境 calibration 和 GPU/ANE 多阶段集成仍未完成。


## 第二十七轮：CLI worker 的受控容器身份（2026-09-21）

修正此前 Python/CLI host 被记录为 embedded_app 的接入问题：新增 `tc_engine_create_model_worker`（以及内部 candidate worker 入口），在 engine 发布给调用方之前固定 `cli_worker`；原 App 构造入口保持 embedded_app。普通 request JSON 不能更改 engine 的容器。native CLI、test-catalog CLI 和 campaign 改用 worker 入口；campaign 的 build identity 同时记录容器。

campaign 默认 cli_worker，缺少新 symbol 的旧库会明确报错，不自动退回 App 身份。复现旧报告可在可信 campaign 配置中显式指定 execution_container=embedded_app；这不把旧 Python 测量变成 App 安装验收，也不升级旧证据的资格。此前冻结报告和 digest 不回写。全局无 engine 的 options API 仍保持 App discovery 语义，其他直接调用旧构造器的 host 也未被自动重新分类。

`test_streaming_campaign_verifier.py` 共 34 项通过，包括 public/candidate worker 入口、显式 legacy 入口、无效容器拒绝及原 campaign 失败/超时证据保留回归。本轮 `git fetch origin dev` 成功，`git merge-base --is-ancestor origin/dev HEAD` 返回 0，无新增 dev 合并内容。


完整 native hook 构建 `build/m1-worker-identity` 成功，dylib SHA-256 为 `6e1635d7de13599ebfa115ece5f62e534551be22c66c12a6cd75eb61a817176e`。`test_streaming_test_catalog.py` 3 项通过：CLI 导出的 workload/calibration 均为 cli_worker；worker engine 可解析同容器记录；仅将 workload/calibration 容器改成 embedded_app、重新计算合法 digest，其余身份不变，worker resolve 返回 unvalidated_workload。原 App/v1、verified v2 和变更失效回归继续通过。release hook 缺席检查仍使用保留的 m1-release 库。

[真实 worker 验证与生成结果](2026-09-21-m1-z-worker-source-verification.json)、[合成测试 catalog](2026-09-21-m1-z-verified-worker-catalog.json)、[请求](2026-09-21-m1-z-verified-worker-request.json)：取消返回 2（本次 wall 1.030 s），20,701,575,490 bytes 的复核 10.572 s，缓存复验 0.000514 s、零 payload bytes、4 hits。随后真实 Z-Image 256²/9 步/P0/G1/K2/D0/Q1 请求成功，报告 execution_container=cli_worker、actual_plan_verified=true、drained=true，PNG SHA-256 仍为 `8a3e89a095a124aba019f09e47a7f34248d34680e4fd127494be8babb5848be3`。请求 wall 33.619 s，MLX peak 8,975,747,432 bytes；未测完整进程树 footprint，也未进行时延统计比较。

此轮仍使用合成 TEMPLATE catalog 的 12 GiB 目标与 calibration 字段，不能据它声称 12 GiB 资格或发布预设。R6 的 CLI 容器入口已修正，但自动 runtime build fingerprint、最终包清单和完整 App/worker 安装验收仍待完成。


## 第二十八轮：标准 campaign 的显式内容验证（2026-09-21）

`build_test_streaming_catalog.py --verify-sources` 现在先显式调用 native 验证，再生成 v2 worker record；默认不隐式 hash。campaign native variant 新增布尔配置 `verify_streaming_sources: true`，在 engine 创建后、catalog 安装前完成同一操作，失败会释放 engine 并保留错误。缺少验证 API 的旧库明确失败，不用 Python 文件哈希冒充 native proof。新共享 helper 位于 `tools/native/streaming_source_verification.py`。

campaign build identity 记录该配置；raw sample 的 `source_verification` 保存 native 摘要、读取字节、缓存命中数、验证 wall 和 engine_setup scope。persistent engine 的首次验证发生在 worker ready 之前，其成本单独记录，不计入每次生成 wall；per_request engine 的验证在创建计时范围内，计入原有完整 lifecycle wall。重复样本上的 persistent setup report 不表示重复执行了哈希。启动截止时间仍由冻结的 worker_start_timeout_seconds 决定，不自适应放宽。

验证：campaign suite 34 项通过，新增验证先于 catalog 安装、验证失败只释放一次 engine、创建失败不会附带上一 engine 证明的检查；test-catalog API suite 3 项通过，新增 CLI 显式验证生成 v2/cli_worker record 的实际调用。使用已构建的 m1-worker-identity 库，没有重建 native 或 App。

[标准 campaign setup 的真实验证记录](2026-09-21-m1-z-campaign-source-verification.json)：load_native_worker 通过配置显式验证真实 Z-Image 四文件，共 20,701,575,490 bytes、10.628 s，然后安装上一轮 v2 测试 record 并成功 public resolve。本次未执行 GPU generate 或完整 P1/P2 campaign，因此不声称新增性能资格。内容证明仍仅在当前 native 进程有效：独立的 catalog CLI 与 campaign worker 都需各自验证，catalog JSON 不携带可恢复的信任。persistent import proof 和 App 集成仍待完成。


## 第二十九轮：Flux verified descriptor 与实际 plan（2026-09-21）

Flux metadata 新增 `describe_verified()`，要求 native verified lease。checkpoint identity 使用全 lease 内容摘要，各 transformer artifact 使用对应 logical id 的真实 SHA-256；config/index、文本编码器等已包含在 lease 中的辅助文件也参与全身份。lease 版 StreamingPlanView 按 native proof 选择 portable descriptor；路径/metadata-only 路径保留旧摘要。模型侧 source identity 对 verified lease 同步选择 v2，保持 source 与 plan 一致。

`.venv/bin/python tests/native/test_flux_streaming_descriptor.py` 编译并通过 9B legacy sharded 回归及 4B 新案例：未验证 metadata 拒绝 portable describe；实际 verified plan 与旧 snapshot plan 不同；迁移路径后 verified layout 相同；capture_preverified 零 payload 读取仍产生同一 plan；仅向 config.json 追加 JSON 空白（几何不变）会使旧 lease 失效、重新验证后的全内容/layout identity 改变。迁移 fixture 独立复制 config、对约 7 GB sparse payload 创建硬链接以避免实写零数据；这是路径迁移测试，不冒充独立完整副本测试。底层 SourceLease 的独立复制/改写覆盖见前述测试。

使用托管 MLX headers 对 Flux pipeline.cpp 执行 C++20 clang++ `-fsyntax-only`，返回 0。未重新链接完整 native 库或运行真实 Flux GPU。本轮尚未 override Flux 的 verify_streaming_sources，公开 probe 仍捕获 metadata-only lease；显式验证 API 接入、真实 v2 公开生成和 persistent proof 仍待完成。


## 第三十轮：Flux 显式内容验证接入（2026-09-21）

Flux 覆盖 ModelSession::verify_streaming_sources，复用已有 C API 和 campaign 显式验证选项。验证与 public probe 共用 artifact 清单（transformer/config/index、transformer shards、text encoder/config/weights、VAE/config/weights、tokenizer）；验证后 engine 保持内容身份模式，probe 使用 capture_preverified，不在查询阶段自动重读权重。文件变化或缺少 native proof 会失败，不能退回 legacy snapshot。该实现适用于既有 4B/9B Diffusers 布局，真实设备验收本轮只针对已下载的 4B。


完整 native hook 构建 `build/m1-flux-source-verify` 成功，库 SHA-256 为 `48c8be973b06e6e909bdc1925644498eb7aa760f3d81e51b90b24e0952ae68cf`。`TURBOCIDER_TEST_NATIVE_DIR=build/m1-flux-source-verify .venv/bin/python tests/native/test_flux_public_streaming.py` 的 9B/4B host adapter 回归通过；4B 新增 native 验证→v2 probe→portable plan 编译、零 payload probe、配置变化失效及旧 record 拒绝检查。链接器提示测试目标 macOS 26.0、库目标 26.2，本机 26.4.1 上运行通过；没有据此验证较旧 OS。

[真实验证/生成结果](2026-09-21-m1-flux4-worker-source-verification.json)：取消返回状态 2（本次 wall 0.111 s），随后 8 个文件共 15,975,638,166 bytes 的校验耗时 7.850 s。大文件 SHA-256 匹配固定官方 revision `e7b7dc27f91deacad38e78976d1f2b499d76a294` 的 LFS manifest，小文件先比对官方 Git blob SHA-1 再与 native SHA-256 对照；全部通过。全内容摘要为 `fdba1a4019cf925357b30afbc27390450b287dcf8d2d10f0d39c07ea08030fca`。再次验证用 0.001075 s，payload bytes=0，cache_hits=8。

同一 worker 通过[合成测试 catalog](2026-09-21-m1-flux4-verified-worker-catalog.json)运行[真实请求](2026-09-21-m1-flux4-verified-worker-request.json)：256²、4 步、seed 42、P0/G1/K2/D1/Q2、retain_all、目标 12 GiB。wall 8.184 s，native denoise 6.391 s，MLX peak 6,269,857,144 bytes。PNG SHA-256 `505668fa0966029c4f0d4f943b482c1620b2433c947d006824bfc76930919b02` 与原 private smoke 完全一致。报告为 cli_worker、actual_plan_verified=true、source_lease_verified=true、drained=true；100 fills、100 reader fences 均完成，authorized/actual layout digest 相同。

这与 Z-Image 一起证明两个目标模型的 native 内容验证→v2 worker record→公开 adapter→实际图片/receipt 链可运行。catalog calibration/TEMPLATE performance 仍为测试 hook 合成输入，不是 12 GiB 发布资格；本次未采完整进程树 footprint 或统计性能置信区间。persistent import proof、App 验证接入、生产 catalog、自动 runtime fingerprint、GPU/ANE 多阶段集成及其完整请求验收仍未完成。


## 第三十一轮：App 显式文件校验入口（2026-09-21）

Swift NativeEngine 增加异步 verifyStreamingSources，继续在自身串行队列调用 C API，按 0/2/1 映射成功/取消/失败。模型库已登记安装对 Z-Image、Flux 4B/9B 提供“校验文件”；通过当前会话打开指定安装，显示校验中/已校验文件数/取消/失败状态，完成后提示重启需重新校验。校验报告不交给推理统计视图，避免显示无意义的推理指标。会话栏新增取消任务按钮，模型/API 忙、加速解析中或需要重启时禁用校验；失败处理保留进程隔离的重启状态。

[构建、功能测试和启动记录](2026-09-21-m1-app-source-verification.json)：App 以最新 Swift 源码链接 m1-flux-source-verify native hook 库构建成功。既有 image transaction、streaming resolution、Studio behavior 回归通过。新增 `StreamingSourceVerificationTests.swift` 在真实 Flux 4B 上通过 Swift SDK 校验 8 个文件并确认缓存复验 payload=0。新增 `ModelSourceVerificationTests.swift` 使用隔离临时历史目录和真实模型，验证外部服务忙时拒绝、取消后恢复空闲、重试成功、报告状态清理和卸载。测试最初误写 JobStore 类型名导致编译失败，修正为 NativeJobStore 后通过；不隐去该失败。模型会话测试使用 -Onone，验证功能而非时延。

新 App 直接启动并观察 15 秒仍存活，stdout/stderr 为零字节，随后只终止本次启动的进程（退出 -15）。这是启动 smoke 与组件测试，不是自动点击按钮的完整 GUI 验收。校验由用户显式触发，不在 options 中增加隐藏哈希；当前仍没有 persistent import proof，也没有生产 catalog。新会话的 options/内容证明复用、正式预设选择、GPU/ANE 全链路及 App/worker 完整安装验收仍待完成。


## 第三十二轮：新会话复用 native 内容证明（2026-09-21）

修复 App 模型校验与 options 查询使用不同 engine 的接入缺口。Flux/Z-Image probe 现在调用 SourceLease::capture_for_query：新会话在全部来源均有有效 native 进程缓存证明时采用 v2 内容身份，查询不 hash payload；证明不完整时保留未校验的 legacy 身份。会话一旦采用或显式请求内容身份，文件改变、证明失效后只能拒绝，不能退回 legacy。仅捕获专用 ArtifactVerificationRequired 异常作为初次未验证路径，文件访问错误和 caller digest 不匹配等错误继续传播。缓存仍绑定打开文件的 generation，不接受 JSON 自授证明。

完整 native hook 构建 build/m1-query-proof 成功，dylib SHA-256 为 `726e4e84775b6c6cd784f9b0d616f1802e5012fab40e40ce092243bfa5731952`。SourceLease host 测试覆盖未验证查询、完整/部分缓存、零 payload 查询、错误 digest、同大小改写后拒绝且不降级；Flux public adapter 的 4B/9B 回归通过，4B 增加新会话 v2 probe 和变更后拒绝；test-catalog API suite 3 项通过，新增 Z-Image fresh public engine 在没有再次显式验证的情况下解析原 v2 catalog。release hook 缺席检查继续使用保留的 m1-release 库。Flux host 链接仍提示 macOS 26.0 测试目标与 26.2 库目标不同，本机 26.4.1 运行通过。

[真实 Flux 4B 跨会话记录](2026-09-21-m1-fresh-engine-source-proof.json)：首次显式校验 15,975,638,166 bytes 用 7.993 s；销毁首个 engine，再新建 worker，安装第三十轮合成测试 catalog，在第二个 engine 尚未调用验证 API 时 resolve 成功，wall 0.239556 s。随后缓存复验读取 0 bytes、8 hits。该行为支撑 Swift options 中新开 engine 的解析流程，但本轮未重新构建 App 或进行 GUI 点击验收。

本轮未执行 GPU generate 或新增性能比较；使用的 catalog 仍是 TEMPLATE 测试输入。证明只在当前 native 进程中有效，重启后需要显式校验；persistent import proof、生产 catalog、自动 runtime fingerprint、GPU/ANE 多阶段与完整 App/worker 验收仍待完成。


## 第三十三轮：构建输入自动绑定 runtime 身份（2026-09-21）

新增 `generate_runtime_build_identity.py`，标准 native build 在编译前生成内部常量和可审查的输入清单，构建结束时重新计算并拒绝变化的输入。Flux/Z-Image/H3/LTX 的 public runtime identity 改用该常量，取代固定日期标签。该路径没有手写 fallback，也不从请求或 catalog 接受 build id。

[本轮构建清单](2026-09-21-m1-runtime-build-manifest.json)含 330 项源码/构建文件和 386 项 MLX dependency 文件。native 代码、C binding、native CLI/service glue、构建脚本、MLX headers/dylib/metallib、clang/clang++/ld/ar、compiler version、SDKSettings、架构、部署目标及编译/测试/审计配置共同决定 key；安装根路径归一化，文档和 build 输出不参与。完整 hook 构建 build/m1-runtime-identity 成功并通过结束时的清单复核，runtime id 为 `tc-runtime-build-v1-fe428f00113044c377ed3f4c91839dcaac4688c4aecd7bb89c5bbf1d8e418b3a`；dylib SHA-256 为 `f43620285adce3bdd692ec5776cad0b5b54478fb6ea9e403e7a1ccc46a086c25`。

生成器 6 项测试通过：相同内容迁移目录保持 key；源码/构建脚本/MLX shader/headers/library、编译工具/SDK 身份和策略变化使 key 改变；编译期间输入变化以及未纳管的 compiler search 环境拒绝。native contract suite 83 项运行、3 项原有跳过，其余通过。四模型 public adapter host suite（Flux 同时覆盖 4B/9B）通过；H3/LTX 测试增加 TURBOCIDER_TEST_NATIVE_DIR，确保运行本轮库。test-catalog API suite 3 项通过，检查生成记录的 build id 与清单一致，并验证仅换 runtime id、重算合法 record digest 后仍返回 streaming_resolution_stale。host 链接的 macOS 26.0/26.2 目标提示保留，本机 26.4.1 通过，不扩展为较旧 OS 验收。

这是 native compatibility key，不是签名或最终 App/package manifest。Apple SDK 只记录 SDKSettings 身份，未 hash 整个 SDK；Swift frontend、外部 helper 可执行文件、动态依赖运行时完整性和最终安装验收仍待补齐。当前 bundled catalog 仍为空且内嵌于 native 源码，未来分离 catalog 数据时须避免 build key 自引用；这项自动指纹不使生产 catalog 自动具备发布资格。测试 hook/审计开关不同产生不同 key，不能重标旧实验或 hook 记录作为 release 证据。

[本轮功能验证及构建产物哈希](2026-09-21-m1-runtime-identity-validation.json)：显式验证官方模型、确认生成的 v2 record 使用本轮 runtime id 后，在同一 worker 中执行完整图片请求。两个模型均为 256²、seed 42、P0/G1/K2、目标 12 GiB；Flux 使用 D1/Q2、4 steps，Z-Image 使用 D0/Q1、9 steps。

| 模型 | 请求 wall | native denoise | MLX peak bytes | 图片 SHA-256 |
|---|---:|---:|---:|---|
| Flux 4B | 8.062 s | 6.281 s | 6,269,857,144 | `505668fa0966029c4f0d4f943b482c1620b2433c947d006824bfc76930919b02` |
| Z-Image | 33.493 s | 29.150 s | 8,978,303,336 | `8a3e89a095a124aba019f09e47a7f34248d34680e4fd127494be8babb5848be3` |

两张图片与原 golden 字节一致，actual_plan_verified=true、authorized/actual layout 相等，receipt 分别完成 100/270 fills 和 reader fences，并完成 drain。[Flux 原始报告](2026-09-21-m1-flux4-runtime-identity-smoke.json)、[catalog](2026-09-21-m1-flux4-runtime-identity-catalog.json)、[request](2026-09-21-m1-flux4-runtime-identity-request.json)及 [Z-Image 原始报告](2026-09-21-m1-z-runtime-identity-smoke.json)、[catalog](2026-09-21-m1-z-runtime-identity-catalog.json)、[request](2026-09-21-m1-z-runtime-identity-request.json)均保留。calibration/TEMPLATE performance 字段仍是测试 hook 合成输入；本轮未采完整进程树 footprint、未比较性能置信区间，也未重建/验收新 App，不能作为 12 GiB 发布或全链路 GPU/ANE 性能结论。


## 第三十四轮：catalog 数据与 runtime 指纹解耦（2026-09-21）

上一轮的指纹会覆盖 native 源码，因此继续把正式 record 直接写入 preset_catalog.cpp 会产生自引用：record 内包含 build id，填入后源码变化又改变 build id。本轮将生产输入移到独立 `native/runtime/streaming/bundled_catalog.json`，标准构建先生成 runtime id，再生成只读 catalog header 和独立 catalog manifest。数据 JSON 不参与源码指纹，生成器代码参与；结束时分别复核两份清单，不能在编译途中换 catalog 数据或生成 header。

非空 inventory 条目必须给出原始 P2 bundle、record input、review、P1/P0/P3 bundles 及预期 record digest。生成器实际重跑现有 build_record 和 evidence verifier，而非读取某份 JSON 的 verified 标签；只接受 public-stable/public-experimental，保持所有既有资格门槛，再检查当前 runtime、catalog revision、digest 和重复 id。生成的 C++ 只有结构化字段赋值，字符串按 UTF-8 字节转义、整数按 native 字段宽度检查；native 初始化再次校验 canonical record。默认 inventory 仍为空，不发布新预设，也不提供远程或请求可写入口。

[本轮验证记录](2026-09-21-m1-bundled-catalog-validation.json)及[完整 runtime 输入清单](2026-09-21-m1-bundled-catalog-runtime-manifest.json)：完整 hook 构建 build/m1-bundled-catalog 和结束时两份清单复核通过。runtime id 为 `tc-runtime-build-v1-f9110a48b068719c8bccb67c826b50c12c5cc4aae44cbfc583bc78b8d4ed4f1b`，dylib SHA-256 为 `3f7835ca1b7d5a60d104e47309cf59432e73b8380016bb0b475545fcd36aba73`。

新增 catalog suite 6 项通过，覆盖独立数据 revision、原始证据缺失拒绝、调用 builder 的完整输入、错误 runtime/digest/channel 和重复 id、字段注入/整数溢出拒绝、CLI 编译后 header/inventory 变化拒绝，以及生成 v1/v2 数据实际编译并通过 native canonical validation。首轮测试的 /var 与 /private/var 路径比较，以及遗漏 SDK/native-core include 的编译参数导致失败；修正测试后通过。生成器证明的只是数据接入，不把 mock builder 的 fixture 称作发布资格。

runtime identity suite 6 项、原 evidence builder suite 16 项、preset resolver host 回归均通过；native contract 83 项运行、3 项原有跳过，其余通过。新库的 test-catalog API 3 项及 Flux 4B/9B、Z-Image public adapter host 回归通过。release hook 缺席检查仍使用保留的 m1-release 库；本轮没有新 release/App 构建、GPU generate 或性能测量。host 链接的 macOS 26.0/26.2 目标提示保留，本机 26.4.1 运行通过。

此轮关闭的是 catalog 数据导致 build key 自引用的问题；非空生产记录的完整资格实验、最终 package manifest/安装验收、持久化 import proof、GPU/ANE 多阶段接入仍未完成，不能据此标记整体 production ready。


## 第三十五轮：编码器局部收益与完整图片请求（2026-09-21）

此前 Qwen3 probe 的 Z-Image 75% FFN 分流在暖态约有 2.03x 编码器收益，但不能外推完整图片。本轮在现有 legacy `residency=streamed` 路径运行完整 Z-Image 请求，没有放宽 explicit/public streaming 的 GPU-only guard。[实际路由检查](2026-09-21-m1-encoder-streaming-route-checks.json)确认两个目标模型的 explicit streaming + encoder ANE 均返回 streaming_route_unsupported；Flux 的 legacy streamed residency 也返回 FLUX block streaming is not supported。因此本轮完整分流图片只覆盖 Z-Image，不代表新框架 HY-M0/HY-M1 已接通。

[冻结计划](2026-09-21-m1-z-encoder-full-image-plan.json)、[完整结果](2026-09-21-m1-z-encoder-full-image-results.json)、[质量分析](2026-09-21-m1-z-encoder-full-image-analysis.json)：256²、9 steps、seed 42、native tokenizer 恰好 64 个有效 token，使用已导出的 b64、INT8、ANE FFN [0,7296)/9728（75%）encoder。每次新建 worker/engine，固定 GPU→分流→分流→GPU，零 warmup；两个路线使用相同 tensor dump。8 GiB 是 legacy denoiser 规划预算，不是整请求/process-tree 上限；OS/Core ML service cache 没有重置。Core ML 配置 cpuAndNeuralEngine，不证明实际 ANE residency。

| 顺序 | 路线 | 请求 wall | text encode | denoise |
|---|---|---:|---:|---:|
| 0 | GPU | 37.908 s | 3.279 s | 31.764 s |
| 1 | 75% encoder 分流 | 59.672 s | 18.954 s | 39.003 s |
| 2 | 75% encoder 分流 | 48.834 s | 9.119 s | 37.989 s |
| 3 | GPU | 33.237 s | 2.494 s | 29.036 s |

完整 wall 中位数比值为 1.525（分流更慢约 52.5%），仅是两对探索性样本，不是性能置信区间。首次分流的 native load_seconds=15.912 s，其中 manifest 验证 5.101 s、model load 10.788 s；实际完成 35 次 Core ML branch prediction、零 runtime failure，未发生 GPU fallback。两个路线的 legacy denoiser 均为 pinned=8、streamed=22、refill slots=2，prompt cache 均未命中。

两次 GPU 图片彼此字节一致，两次分流图片也彼此一致；初始 latent 一致、所有诊断 tensor 有限。RGB correlation=0.995445、cosine=0.999229、MAE=2.49195，但 final latent relative-L2=0.099705，高于预先冻结的 0.05，故整体探索质量门槛不通过。没有放宽阈值，也不能用局部 encoder relative-L2 或 RGB 接近掩盖该失败。单一 prompt/seed 本身也不构成一般质量资格。

新增可复现 driver `benchmark_z_image_encoder_streaming.py` 与 CPU 分析器 `analyze_z_image_encoder_streaming.py`，保留每个新 worker 的请求、事件、结果、图像和 tensor；worker 超时会对本次创建的进程组 TERM/KILL/reap，保留诊断并停止，不重跑到通过。分析器检查图片哈希、请求形状、branch 完整性和相同 denoiser 布局。4 项质量 oracle 测试覆盖非有限数、形状/初始噪声不匹配、完美 cosine 不能掩盖 latent L2 失败、BF16 解码/截断。第一次分析因环境缺少 Pillow 失败，安装 Pillow 12.3.0 后完成；Pillow 仅用于事后 PNG 解码，不参与 native 计时。


## 第三十六轮：编码器会话释放与三种比例的完整请求复测（2026-09-21）

发现 legacy Z-Image streaming 完成 text encode 后仍持有 encoder HybridSession，直到下一次编码或 engine 销毁。现在先 materialize conditioning 并同步 GPU，再在 denoiser 加载前释放 encoder 会话、清理 MLX cache。CoreMLBranch 构造和 predict 增加局部 autorelease pool，及时释放临时 feature provider 等对象；强引用 model 和 MLX output backing 的所有权仍由 session 保持。缓存 conditioning 单独保存 HybridMetrics，避免会话释放后把缓存结果误标为纯 GPU；只有成功重编码才替换缓存来源。新增 `session_released_after_encoding` 表示 native 会话已释放，不声称 Core ML 服务缓存或物理内存立即归还。

新库 `build/m1-encoder-release/libturbocider.dylib` SHA-256 为 `5f27260e93d7a9b9ca75963cbc9601ab3f9859d127ea0d3822fb2c356fe2484b`，runtime key 为 `tc-runtime-build-v1-76236c9d5a6bba55cf66fd32b521ccd8e95327d631e5f33127232bcd4060dd44`。native 构建及 runtime/catalog 编译后核对通过。真实模型生命周期测试在同一 engine 连续运行首次分流、同 prompt 缓存命中、切回 GPU 三个请求，通过释放事件顺序、缓存 provenance、无重复预测及图片一致性检查；见[原始结果](2026-09-21-m1-z-encoder-lifecycle.json)。64²/1 step 仅用于功能回归，不参与性能结论。native contract 83 项运行、3 项原有跳过；test-catalog API 3 项、Flux 4B/9B 与 Z public adapter host、质量 oracle 4 项均通过。host 链接仍提示 macOS 26.0/26.2 目标差异，本机 26.4.1 运行通过。release hook 缺席检查使用原有 m1-release 库；本轮没有新 release/App 构建。

在首次结果后、任何复测前冻结[三比例 sweep 计划](2026-09-21-m1-z-encoder-release-sweep-plan.json)，依次测试 75%、50%、25%，各自固定 GPU→分流→分流→GPU，共 12 个新 worker。沿用上一轮 prompt、256²、9 steps、seed 42、64 tokens、INT8、相同 denoiser 布局及诊断输出，不追加样本、不修改质量阈值。[汇总](2026-09-21-m1-z-encoder-release-sweep-results.json)如下；表中 wall 为两次完整 native 请求的中位数，ratio 是两组中位数之比，不是统计置信区间。

| Encoder ANE FFN 比例 | GPU wall | 分流 wall | 分流/GPU | RGB MAE | Final latent relative-L2 | 冻结质量门槛 |
|---|---:|---:|---:|---:|---:|---|
| 75%（7296） | 33.368 s | 38.844 s | 1.164 | 2.49195 | 0.099705 | 未通过 |
| 50%（4864） | 34.467 s | 42.777 s | 1.241 | 1.98086 | 0.083553 | 未通过 |
| 25%（2432） | 34.284 s | 42.499 s | 1.240 | 2.21099 | 0.086658 | 未通过 |

每个比例均完成 35 次 Core ML prediction、零失败，且 release 事件早于 denoiser 加载；初始 latent 完全相同、所有 tensor 有限、各路线重复图片字节相同。三个比例的 RGB 指标均通过，但 final latent relative-L2 均超过冻结的 0.05，因此质量总体不通过。原始计划、结果、逐 tensor 分析分别保存在 `2026-09-21-m1-z-encoder-released-w{7296,4864,2432}-{plan,results,analysis}.json`；本地完整图片和 tensor 路径由这些计划索引。

75% 修复前后对应 GPU/分流图片均字节相同。修复后 denoise 中位数为 GPU 29.019 s、分流 28.255 s，但分流 text 阶段仍需 8.871 s（GPU 2.588 s），完整请求仍慢约 16.4%。前后实验是顺序执行、OS/Core ML 缓存未重置且每组只有两对样本，不能把前后全部延迟变化归因于释放修复。三个比例的 worker 等待时间比值同样大于 1；该口径从 Popen 返回到进程退出，不包含 Popen 调用本身。

结论限定为这台 M1 Pro、此 prompt/seed/shape 的冷 worker 完整请求：编码器局部暖态加速未转化为完整请求收益，这三种分流均不能据此晋升；纯 GPU 仍是本组实验支持的选择。这里没有 process-tree 全程内存采样，MLX peak 不包括全部 Core ML/OS 内存，CPU+NE 配置也不证明实际 ANE residency。该修复只验证正常完成路径，尚未关闭 encoder/VAE 的故障、取消和阻塞恢复问题。新框架 hybrid guard 保持原状，HY-M0 denoiser 分流、正式质量/性能资格及 App 安装验收仍待完成。


## 第三十七轮：H1 后缀权重转换的共享实现（2026-09-21）

为推进 denoiser GPU/ANE 新框架，新增 `native/models/z_image/suffix_materialization.hpp/.cpp` 并接入现有 `ZImageWeightStream::pack_suffix()`。metadata-only 几何函数计算 w1/w3 连续后缀范围及 w2 逐行后缀尺寸；执行函数只使用持有的 fd，最大 4 MiB scratch，检查 off_t/计数溢出、源文件范围和不同 regular file。保留 legacy 32 个 branch（2 noise + 30 main）及 context refiners 全 GPU、ConvRot scale/对齐规则，未放宽设备与 public route guard。

host 独立字节 oracle 覆盖 BF16/I8、a=0/1/M-1、非零源/目标偏移、多批次，检查所有输出字节；确定性 syscall 注入覆盖 EINTR、短读/短写、读中途 EOF、写入 ENOSPC、零进展、读后取消和最后写入后取消。失败前实际传输字节仍被累计；调用方只在成功后绑定派生记录，异常由 loader 构造器关闭临时 fd。源 lease/revalidation、private fd 生命周期和 Ready 发布责任仍在调用方，不把一个 packing 函数解释为已完成来源安全合同。

复核时发现初版共享函数使用普通 runtime_error 表达取消，会丢失原 `tc::Cancelled` 分类；修正为原异常类型并新增类型断言。最终 host 测试及 AddressSanitizer/UndefinedBehaviorSanitizer 通过。注入通过单独测试对象的 pread/pwrite 符号重命名实现，生产代码没有新增测试后门。

`git fetch origin dev` 成功，远端为 `61c08495815d645bb54d75ae9dbea466f0648b2d`；`git merge-base --is-ancestor origin/dev HEAD` 返回 0，当前分支已包含最新 dev，无新合并内容。本轮不运行新的性能 campaign，不改变上一轮编码器分流结论；HY-MAT-01/02 只完成转换子项，hybrid descriptor/source identity、Core ML bundle lease、完整 owner、receipt 和 public qualification 仍未完成。

最终 native hook 构建及 runtime/catalog 编译后核对通过，库 SHA-256 `7c05d57714cb5d43b445c01829fdcce5034984d6055a4187005aac20f2930e9e`，runtime key `tc-runtime-build-v1-94f33a9c5126ac9c9ce34780e384f7f6ba5fb09084afaf37943005d7b9d404c4`。native contract 83 项运行、3 项原有跳过，其余通过；runtime identity 6 项通过；新库 Z weight stream 10 项运行、6 项因 M1 不支持 legacy suffix/ConvRot 跳过，其余 4 项通过（包含实际 GPU 重复执行与取消）；Z public adapter host 回归通过。host 链接 macOS 26.0/26.2 提示保留，本机 26.4.1 运行通过。见[构建与回归记录](2026-09-21-m1-z-suffix-materialization-validation.json)。没有新 release/App 构建，也没有以跳过的设备测试支持 hybrid 验收结论。


## 第三十八轮：GPU 后缀 metadata 与布局计量（2026-09-21）

新增内部 `StreamingMetadata::describe_gpu_suffix()` 返回 `GpuSuffixPlan`：原 checkpoint 上 w1/w3 连续后缀读取范围、32 个 w2 打包记录、派生文件偏移/大小、完整 fixed fields 与 30 层 fields。noise refiner 的固定权重参与裁剪，context refiner 保持完整，常驻 prefix 与 streamed slot 使用同一字段定义。沿用 common layout compiler 计算对齐、P/G/K 布局、每 pass 读取量；没有修改 exact descriptor 或放宽 public/设备 guard。

转换配方以独立 canonical domain 绑定 parent 内容身份（有 native proof 时）或 snapshot（未验证时）、源文件大小、first_gpu_channel、转换版本和 96 个 FFN tensor 的名称/范围/投影。派生 artifact 明确使用 `recipe:` 的 metadata identity，不冒充派生字节 SHA-256；新 adapter/backend revision 标记 metadata-only。该对象不创建临时文件、不打包、不分配 GPU，也不能直接交给 exact adapter 执行。一次性 setup read/write 与后续 refill read 分开记录。

稀疏 BF16 fixture 的独立公式覆盖 a=1/2560/5120/10239、P=0/7/29，核对 fixed、prefix、slot、每 pass I/O、w1/w3 原文件偏移、w2 派生偏移、所有未裁剪字段保持一致、32 branch 映射、错误 shape/缺 tensor/通道边界拒绝及 exact 布局不变。首轮 P29/K2 测试被 common compiler 正确拒绝 slot_count_exceeds_groups，修正测试为 P29/K1，未改变 runtime 规则。ASan/UBSan 下 sparse fixture 和真实模型 header 均通过；原 exact descriptor suite（含 verified portable layouts）通过。新 suffix 配方的跨安装内容身份尚未单独做双副本验证，不以原 exact 测试替代该项。

真实 Z-Image header、512²/9 steps/caption rows 64、P7/G1/K2/D0/Q1 的 metadata 容量如下（十进制 bytes）；此处没有 Core ML bundle，也没有执行 denoiser：

| ANE FFN 通道占比 | 固定 GPU 字段 | 常驻 GPU prefix | GPU slot 池 | setup 原文件读 | setup 派生写 |
|---|---:|---:|---:|---:|---:|
| 25%（a=2560） | 1,337,232,640 | 2,119,867,904 | 605,676,544 | 2,516,582,400 | 1,887,436,800 |
| 50%（a=5120） | 1,219,267,840 | 1,706,991,104 | 487,711,744 | 2,516,582,400 | 1,258,291,200 |

这些数字仅是规划的 GPU 权重容量与逻辑 I/O，不能当作完整请求内存或性能收益；ANE models、activation、encoder/VAE、OS/Core ML 服务开销没有包含。a=1/10239 仅是几何边界测试，不是已存在的 ANE artifact。见[源码与测试记录](2026-09-21-m1-z-suffix-descriptor-validation.json)。native contract 83 项运行、3 项原有跳过，其余通过。本轮编译并运行了 standalone native metadata 测试，没有重建完整 dylib/App；下一执行接入还需 verified derived source、typed Core ML bundle、完整 owner/join、receipt 与资格验证。


## 第三十九轮：后缀派生文件的验证与请求所有权（2026-09-21）

H1 从 metadata 继续推进到实际文件：`materialize_gpu_suffix()` 要求父 SourceLease 已通过 native 内容验证，再内部生成配方；从 held source fd 打包 32 个 w2 后缀，创建的 private 文件在写 payload 前 unlink。打包完成核对实际读写量和尺寸，关闭写句柄，对只读句柄计算派生 SHA-256，复核父/派生 generation 后才返回 `GpuSuffixSource`。返回对象保留父 lease，只能复制 artifact 0/1 的只读 CLOEXEC fd；未知 artifact、父文件修改/替换均拒绝。

配方 descriptor 不因文件生成而改写：稳定 recipe 仍用于编译布局，实际派生 SHA-256 作为独立运行证据，不能把两者混成同一个内容哈希。`verification_read_bytes` 单列二次哈希读取，setup pack read/write 已由实际 I/O 对照计划核验；没有把校验成本遗漏为零。

host 测试使用两个独立生成、内容相同但 inode/path 不同的稀疏 BF16 checkpoint（3840×10240 的真实 FFN 几何）。每个 w2 的指定行/末通道写入不同哨兵，a=10239 使派生文件为 245,760 bytes，逐元素检查所有 32×3840 个结果，并独立重算 SHA-256。a=10239 仅用于边界/生命周期测试，不代表支持该 ANE artifact。父文件仍完整经过 native 哈希，不把 sparse header 或传入 digest 当作内容证明。

验证覆盖：无内容证明拒绝；打包前、完成一个 branch 后、哈希前、哈希后四处取消保留 `Cancelled` 类型且 fd 数恢复；事件异常关闭临时 fd；返回 fd 为只读/CLOEXEC/unlinked，pwrite 失败；metadata/外部 lease 退出后 owner 仍能验证；独立副本获得相同 recipe、派生 SHA 与 checkpoint identity；打包中父文件内容修改、完成后父路径被移走均拒绝。普通运行与 ASan/UBSan 通过，原 exact descriptor 回归通过。

这关闭的是 H1 派生源的基础所有权与内容验证子项；仍需把它接入实际 weight reader/hybrid adapter、Core ML bundle/partition、GPU join、receipt 和产品准入。没有以 fd 生命周期测试替代 GPU pending 时的完整 owner 隔离，也没有新 full-image 性能结果。


补充真实模型文件验证：对已下载 Z-Image checkpoint 的 12,309,866,400 bytes 做 native 内容验证，得到 SHA-256 `2407613050b809ffdff18a4ac99af83ea6b95443ecebdf80e064a79c825574a6`，与模型下载记录一致。以 a=5120（50%）打包 32 个分支，实际原文件读 2,516,582,400 bytes、派生写 1,258,291,200 bytes、派生哈希再读 1,258,291,200 bytes。派生 SHA-256 为 `59b03803a9dac69d0fe89a79ef318056260b77d48cdc588744aef6bee89fcb47`；逐分支抽查第 0/1234/3839 行，共 96 行，与原文件列后缀逐字节相同。owner 释放后 fd 数恢复，private 派生文件未持久保存。见[原始结果](2026-09-21-m1-z-suffix-source-real.json)。完整字节 oracle 来自稀疏 fixture，真实模型采用抽样，不把抽样表述为完整张量数值校验。本次没有 GPU/Core ML 预测或速度比较。

完整 native hook 构建及 runtime/catalog 编译后核对通过：库 SHA-256 `26e3accd085636deee1e1c2accf4164f4737caa984dad1333d6428f360877fc3`，runtime key `tc-runtime-build-v1-bb953df620c174a36ee5407afee90e82b0109e164c42d1dcc58bd9d8d0e4b783`。直接链接新 dylib 的同一 source 生命周期/内容测试通过；Z public adapter 回归通过；suffix metadata、原 exact descriptor suite 通过；native contract 83 项运行、3 项原有跳过，其余通过。host 链接 macOS 26.0/26.2 提示保留，本机 26.4.1 运行通过。见[验证记录](2026-09-21-m1-z-suffix-source-validation.json)。本轮没有 release/App 构建。


## 第四十轮：已验证后缀来源接入 GPU 权重读取器（2026-09-21）

新增 `ZImageWeightStream(shared_ptr<const GpuSuffixSource>, …)` 内部构造路径。它持有完成验证的 source owner，从原始/派生只读 fd 读取 metadata 对应字段，覆盖 fixed noise、完整 context、resident prefix 与 streamed blocks；不重新打开 checkpoint 数据路径，不重复 packing，不自动改变 P/K。构造完成及每个后缀 fill 前后检查来源 generation。既有 legacy 与 public GPU-only 准入保持不变，新构造器本身不授予 hybrid route 或 Core ML 权限。

同时修正新路径的容量口径：真实 checkpoint 的 `final_layer.linear.bias` 为 128 bytes，但 descriptor 按 256-byte 字段对齐预留。新 reader 调用 common layout compiler 获取 aligned fixed/slot capacity，预算与布局一致，实际 I/O 仍按未填充的 tensor bytes 计数。最初构建在发现该问题后明确停止，修正后重新完整构建；不是因观察超时而重启。派生文件的 unlink guard 也提前登记，避免创建后校验抛出异常时遗漏路径清理。

`test_z_image_suffix_reader.py` 显式选择 Metal device。使用真实 FFN 几何 3840×10240、a=10239 的稀疏 fixture，对每个 branch 的 w1/w2/w3 后缀写入独立哨兵，再在 GPU 上验证全部后缀值和未裁剪字段；context 保持完整形状且内容为零。P2/K1 与 P2/K2 各运行 2 个 pass，每 pass 28 个 suffix blocks；K2 使用两个独立 host worker 填充两个 slots，所有 GPU 比较完成后才复用 backing。两轮内仅有 K×13 个 backing 指针，refill/I/O 计数符合 oracle，未在 refill 中分配新 slots。

fixture 加入 128-byte 固定字段：预留 256 bytes、实际只读 128 bytes；恰好达到 aligned 预算可构造，少 1 byte 则拒绝。另覆盖 fixed 加载后取消的构造清理、worker 取消与重试、caller 释放 source 后 reader 继续持有、父路径移走后 fill 拒绝，以及 reader fd 回收。打包发生在 source owner，reader 的 request_pack_read/write 为 0，避免重复计量。

最终 native hook 构建、runtime/catalog 编译后核对、GPU buffer suite、来源 ASan/UBSan 回归及 Z public adapter 均通过。原 weight stream 10 项运行、6 项因 M1 未支持 legacy suffix/ConvRot 跳过，其余通过；native contract 83 项运行、3 项原有跳过，其余通过。库 SHA-256 `d86d32cacf14e5f2914c193ddc5527d1c1f1422f25acc4b509c45ee5a76b31f8`，runtime key `tc-runtime-build-v1-4d4b6fe89ffe1c1525aa6fb41467e938735ce5f70aa3124659845a26831d846e`。host 的 macOS 26.0/26.2 链接提示保留，本机 26.4.1 执行通过。见[验证记录](2026-09-21-m1-z-suffix-reader-validation.json)。

这证明新 reader 能在实际 GPU buffer 上消费已验证来源，尚未证明 denoiser FFN 分流、Core ML/GPU join 或完整输出。a=10239 是边界 fixture，不代表存在该 ANE artifact；没有新真实模型出图、性能 campaign、release/App 构建。下一步仍需 typed Core ML bundle/partition、hybrid adapter/完整 owner、实际 receipt 与质量准入。


## 第四十一轮：32 个 denoiser Core ML 分支与数值筛查（2026-09-21）

为 H0 准备真实 Z-Image 的 32 个前缀 FFN artifact：2 noise + 30 main，hidden 3840、总 MLP 10240、ANE 通道 [0,5120)、bucket 1088、activation/output scale 8/32，初始候选为 INT8 per-channel。该 bucket 对应规划的 512² 图像加 64 caption rows；本轮没有完整图片执行。32 个导出与 native manifest compile 完成，32 次 compile cache miss，96 个源 package 文件的 SHA 校验通过，compiled manifest 保留源 checkpoint SHA、分区、shape 和全部 branch 映射。模型/package/大数组保存在 `/Users/chencanhui/models/TurboCider/experiments/`，仓库只收录小型计划与结果。

保留两处流程偏差：[artifact plan](2026-09-21-m1-z-denoiser-artifact-plan.json) 的首次写入因系统 Python 缺 coremltools metadata 失败，而 shell 继续启动已明确参数的 exporter；该 JSON 实际在启动后补写，不能称为预注册。参数未改变、导出未重启。首次把 manifest 传给单 artifact 的 compile-coreml CLI，被扩展名检查拒绝，尚未编译；随后使用 `coreml` 的 `action: compile` manifest 接口，完成全部编译。[INT8 compile 原始结果](2026-09-21-m1-z-denoiser-int8-compile.json) 保留实际缓存与路径。

`compare_z_image_partitions.py` 在预测前写独立计划，以 seed 42、scale 0.25 的合成 hidden states 做固定 32 分支筛查。输入先 BF16 舍入再无损经 FP16 transport；GPU oracle 从原始 BF16 checkpoint 取前 5120 通道。noise 使用 1024 rows、main 使用 1088 rows；native Core ML 输出按实际顺序先转 BF16 再乘 32。每分支 1 次预测、0 warmups，冻结阈值 relative-L2 ≤ 0.025、cosine ≥ 0.999、relative-max-abs ≤ 0.05，不因失败重试或放宽阈值。

32 次 native prediction 均完成，runtime failures 为 0、checkpoint SHA 验证通过；数值筛查 **31 通过、1 失败，整个 INT8 候选不通过**。失败为 block 29（layers.27），relative-L2 0.0256199662、cosine 0.9996722038、relative-max-abs 0.0093261719。见[冻结计划](2026-09-21-m1-z-denoiser-prefix-smoke-plan.json)与[全部结果](2026-09-21-m1-z-denoiser-prefix-smoke-results.json)。native metrics 中内置 quality validator 未启用，不能用其默认 passed 字段覆盖外部失败。

对唯一失败分支进行离线算术诊断：原 BF16 oracle 重放逐字节相同；BF16 与 FP32 的 relative-L2 为 0.00380200，而原 INT8 candidate 与 FP32 为 0.02532920，说明仅 BF16 舍入不足以解释误差。模拟 FP16 GPU 图与 BF16 的 relative-L2 为 0.00415740；它不是 Core ML 验证。见[离线计划](2026-09-21-m1-z-denoiser-precision-plan.json)与[离线结果](2026-09-21-m1-z-denoiser-precision-results.json)。

随后在导出前冻结[单分支 FP16 对照计划](2026-09-21-m1-z-denoiser-fp16-diagnostic-plan.json)，仅导出 block 29，保留同一输入、oracle、分区、scales 和阈值。单 artifact native compile 后，用 Core ML CompiledMLModel 的 CPU_AND_NE 执行 1 次、0 warmups；source package 树哈希绑定 compile receipt，compiled tree 在预测前后保持同一哈希。此 partial manifest 不能作为连续 32 分支 native session 加载，因此没有伪造 block index。exporter 的固定文件名/artifact key 仍含 int8_pc，实际 export_identity.variant 为 fp16，不能按文件名推断精度。

该 FP16 对照 relative-L2 **0.0046775426**、cosine **0.9999890993**、relative-max-abs **0.00625**，通过原阈值。见[compile 记录](2026-09-21-m1-z-denoiser-fp16-compile.json)与[对照结果](2026-09-21-m1-z-denoiser-fp16-check-results.json)。这支持继续研究该分支的精度选择，不证明全部误差均由量化导致，也不把已有 INT8 候选改判通过。尚未建立或验证混合精度 bank。

全部结果仅覆盖合成输入上的前缀 FFN；没有 GPU suffix join、真实 latent 轨迹、完整出图或速度对照。CPU+NE 配置不证明 ANE 驻留。源/compiled 哈希和研究脚本亦不等于新的 VerifiedCoreMLBundleLease：immutable generation、typed partition、执行 owner/join、288 次真实 step/branch receipt 与产品质量/性能准入仍待实现。本轮没有 native 代码变更或新 native/App 构建，使用第四十轮的 hook dylib；未开放 public hybrid guard。

三个新脚本均已实际执行，py_compile 通过；误差分析回归 4 项通过。归档后再次核对三个 driver 的 SHA 与运行计划/结果一致，FP16 对照计划 SHA 一致，32 个结果均有限，唯一 INT8 失败为 block 29，FP16 对照独立通过。git diff --check 通过。


## 第四十二轮：Core ML 私有 generation 导入与所有权（2026-09-21）

新增内部 `z_image::CoreMLGeneration`，作为后续 VerifiedCoreMLBundleLease 的文件来源基础。导入要求 self-contained regular-file tree，拒绝 symlink（包括祖先路径）、非普通文件和空文件树。先枚举目录/文件并通过 native SourceLease 对 held fd 做内容验证，再从这些 fd 复制到 mkdtemp 的独立私有目录；使用有限 1 MiB scratch，处理短读写/EINTR/取消。复制完成后再次验证源 generation 和完整 entry set，关闭所有写 fd，独立验证目标内容 SHA，确认跨目录的 logical-id/size/content 身份相同，最后将文件/目录设为 owner 只读/可遍历并发布 owner。

每个导入使用新的目录，源更新不会原地覆盖现有请求的 generation。owner 保留目标 SourceLease 与目录 stat/entry snapshot，提供围绕 Core ML 路径加载和 drain 的 revalidate；完整 owner 释放后清理目录。若根路径已被替换，析构拒绝删除替换目录，此类外部篡改可能留下被移走的旧目录，不宣称自动恢复。权限约束和 generation 检查面向受管理更新，不构成对同权限恶意写入者的绝对隔离。

host fixture 验证多 chunk 内容复制与字节数、跨安装内容身份、独立目录、源更新隔离、多 owner 生命周期和清理、入口取消保留 Cancelled、symlink/非普通文件/空树拒绝、同尺寸目标修改、新增文件与根目录替换拒绝，以及析构不删除替换目录。普通运行与 ASan/UBSan 通过。首轮测试因 macOS `/var` 为 symlink 祖先被拒绝；fixture 改用解析后的 `/private/var` 路径，生产拒绝规则不变。当前取消测试覆盖入口，不声称覆盖全部中途取消和 I/O 故障注入。

另对上一轮 block 29 的真实 FP16 `model.mlmodelc` 执行导入、目标内容验证、revalidate 和 owner 释放。复制 117,968,870 bytes，native portable file-list 内容 digest 为 `f98a425fd17d2c11f4d369598d0dde7c510b82e44390d50d250554e9346f7538`；其编码与第四十一轮 legacy tree hash 不同，不能混用。临时 generation 已删除。本轮没有从该新目录执行 Core ML prediction。

新源文件已加入 native build source list；本轮只编译并执行 standalone host 测试，没有重建完整 dylib/App。见[验证记录](2026-09-21-m1-z-coreml-generation-validation.json)。该类没有解析 manifest、验证父 checkpoint/分区/precision、签发执行 authority 或提供安装 ID 注册表；它不能替代完整 VerifiedCoreMLBundleLease。H0/H1 仍需语义绑定与 session 接入，H2/H3 完整 join/receipt、真实出图质量和性能验证继续待完成。


## 第四十三轮：generation 导入故障清理与实际 Core ML 加载（2026-09-21）

为上一轮 CoreMLGeneration 补充确定性 syscall 故障测试。只对单独编译的测试对象重命名 pread/write/fchmod，生产代码、SourceLease 内容哈希和 runtime 均无新增测试开关。12 类覆盖：EINTR/短读写、部分读取后 EOF/EIO、部分写入后 ENOSPC、零进展写、读后/写后/seal 时取消、seal 失败、复制字节损坏、复制期间源路径替换和原地修改。每例分别检查是否返回对象与异常类型，失败后 fd 数恢复、私有目录为空，再进行干净重试并逐字节比较独立 oracle。复制损坏必须由目标 native content digest 校验拒绝。普通运行和 ASan/UBSan 均通过；这些测试没有发现需要修改生产实现的新错误。

另新增 opt-in Objective-C++ 集成测试，将第四十一轮真实 block 29 FP16 compiled model 导入只读 generation，直接以该私有目录路径调用 MLModel 加载。验证 x/y 均为 FP16 [1,3840,1,1088]，CPU_AND_NE 配置下执行 1 次、0 warmups 的全零输入预测；该分支无 bias，独立 oracle 为所有输出精确零。实际输出逐元素通过，加载前后、预测后、Core ML autorelease pool 退出后的 generation revalidate 全部通过，最后 owner 释放后目录被删除。

复制字节和 portable 内容 digest 与上一轮相同：117,968,870 bytes、`f98a425fd17d2c11f4d369598d0dde7c510b82e44390d50d250554e9346f7538`。见[原始验证与源码身份记录](2026-09-21-m1-z-coreml-generation-execution-validation.json)。该测试证明 sealed generation 能被实际 Core ML 路径加载并预测，弥补上一轮仅文件操作的证据范围；它没有执行 GPU suffix join，也不替代原 32 分支非零输入数值筛查或完整出图。CPU+NE 不证明实际 ANE 驻留，未做性能比较。

本轮只新增测试与文档，没有修改生产 native 实现或重建完整 dylib/App。VerifiedCoreMLBundleLease 的 manifest/父 checkpoint/partition/precision 语义绑定、安装 ID 注册和 HybridSession 接入仍未完成；Core ML 预测本身的取消/阻塞 owner/drain 也没有由导入 syscall 测试覆盖。


## 第四十四轮：Core ML bundle 的父内容与分区绑定（2026-09-21）

新增内部 `VerifiedCoreMLBundleLease::bind()`，接收已导入的 CoreMLGeneration 和 native 已验证的父 SourceLease。它持有两者，读取 generation 内相对 manifest，拒绝重复 JSON 字段、布尔值冒充数字、非整数/非有限数、路径越界/绝对路径、缺失/重复/空 compiled model。验证 schema 2、Z exporter recipe 1、无 ConvRot/LoRA、3840×10240 FFN、[0,a) 且 a 为 128 的倍数并严格位于 MLP 内、固定 bucket 1088、scale 8/32，以及 source/export_identity 的全部对应字段一致。source/export_identity 的 checkpoint SHA/bytes 必须匹配 native 已验证的父文件；导出机绝对 checkpoint 路径不参与路径等价判定。

分支映射要求 0/1 为 noise、2..31 为 30 main，32 个不同 compiled 路径由 bundle 提供。当前接受单一 fp16 或 int8_pc precision revision；legacy artifact key 固定为 int8_pc，即使 variant 为 fp16 也以 export_identity 为准。没有混合精度 bank 支持。partition identity 采用 canonical domain `tc-z-image-hybrid-partition-v1`，绑定 generation 内容、父 SHA、几何/范围、bucket、precision、scale 的明确 float32 IEEE 位模式以及 branch/path 映射，不使用浮点字符串或安装绝对路径。

host fixture 的 2 个有效安装副本得到相同身份，20 个无效案例被拒绝；另检查未验证父 lease 拒绝、manifest 越界路径拒绝、generation caller 引用释放后 bundle 继续持有，以及父路径替换后 revalidate 拒绝。普通测试和 ASan/UBSan 通过。fixture compiled files 是 stub，仅验证元数据/来源合同，不能当作模型 ABI 或算术证明。最初测试编译有一处混合类型 auto 声明，修正后重新运行通过。

首次真实验证直接导入整个 compiled-cache，被 SourceLease 的空文件规则拒绝：该目录含 33 个空锁文件，进程已退出；未更改验证规则。后续测试工具明确组装只含 manifest 与其引用的 32 个 model.mlmodelc 的研究 bundle，保持 manifest 字节/相对路径，排除缓存管理文件，再运行独立验证。此组装步骤是测试 fixture，不是产品安装接口；SourceLease/当前 generation 仍要求每个 regular file 非空。

真实完整父 checkpoint 验证 SHA 为 `2407613050b809ffdff18a4ac99af83ea6b95443ecebdf80e064a79c825574a6`。32 个 model.mlmodelc 共 128 个模型 regular files，加 manifest，私有导入 1,888,495,798 bytes；bundle content digest `6f46bcb483d2c23fde3deae1a67e794332d03623ae816a1322a78c091dce3546`，partition identity `ef19a8dbedc7a4ffd59f49d7aa8555e39a999964a3d39360a70e0a51d94eef34`。外部 generation/parent 引用释放后 bundle revalidate 通过，最后 bundle 释放后目录删除。见[验证记录](2026-09-21-m1-z-coreml-bundle-validation.json)。

该对象验证元数据所声明的父关联，不证明任意 compiled code 确由该 checkpoint 导出；模型加载时的实际 feature ABI、算术质量和独立 release qualification 仍须验证，父 checkpoint 的实际 tensor 几何也由模型 metadata/reader 校验。第四十一轮 INT8 候选仍因 block 29 不通过数值筛查。本轮没有 Core ML prediction、GPU suffix join 或速度比较。新文件已加入 build source list，只编译了 standalone native 测试，尚未重建完整 dylib/App。安装 ID 注册、HybridSession typed 接入、执行 owner/join/receipt 及 public authority 仍待完成。


## 第四十五轮：verified bundle 接入真实 HybridSession（2026-09-21）

HybridSession 新增内部 typed 构造器，直接消费 VerifiedCoreMLBundleLease 提供的 32 个 generation 内路径与分区，不再重开/重解释 legacy manifest，也不要求导出机 checkpoint 路径等价。加载前后复核 bundle，固定 FP16 bucket ABI 在每个 CoreMLBranch 加载时核对，沿用一个 MLX 共享输出 backing，无 warmup。新增显式 revalidate_source，供执行 owner 在需要的边界核验；未将其当作 public execution authority。

会话 Impl 持有 bundle；无论正常析构或构造中抛出异常，先保留 bundle 局部 owner，在 autorelease pool 内清理 branches，pool 退出后才释放 bundle，避免路径文件早于 Core ML 模型对象释放。预测入口对 typed 路径按不可变 partition 检查分支编号、完整 padded FP16 [1,R,H]、连续 backing 与实际元素数，再交给 Core ML；错误输入在预测前拒绝，不标记 runtime prediction failure。既有 Z 调用先补齐再裁回有效行数的约定保留。该资源顺序不替代 GPU consumer 的 drain，调用方仍必须等共享输出读取完成才能复用或释放。

构建期间两次明确停止本任务拥有的 build process group：首次补上 typed 输入检查，第二次将检查依据收紧到不可变 partition，避免依赖公开的统计字段；不是因等待超时重启。最终完整 native hook 构建与 runtime/catalog 编译后核对通过。库 SHA-256 `cb901a231ccdd8ac5ee45bdae9bb84b58160a4c51c8950f7b8a4f311079fbbf0`，runtime key `tc-runtime-build-v1-75560d32bb86bce2c3290a110ddaa6ffa046f6f8c254c3626b6f2dfb89553e6a`。

按[固定计划](2026-09-21-m1-z-verified-session-plan.json)对实际 32 个 INT8 compiled models 执行新路径测试：先在首个 load callback 设置取消，验证部分构造退出保留 Cancelled 类型；再在第二个 load callback 注入事件异常，验证传播和干净重试。完整构造后释放外部 generation/parent/bundle 引用，会话仍能保留目录并验证来源。32 次全零输入预测、0 warmups、0 runtime failures，每次通过 GPU 逐元素验证全零输出，并在消费完成后复用同一输出 backing 地址；未补齐行数、float32、非连续/广播 backing、越界分支全部拒绝且会话仍可运行。最后 revalidate 和 session 释放后的 generation 目录删除通过。

旧 manifest/C API bridge 亦用新库完成 block 29 的一次全零预测，checkpoint SHA 验证通过、零 runtime failures、close 正常。native contract 83 项运行、3 项原有跳过，其余通过；Z public adapter 的 shared lease/identity、route rejection、失败清理和 source replacement 回归通过。host 链接 macOS 26.0/26.2 提示保留，本机 26.4.1 运行通过。见[验证记录](2026-09-21-m1-z-verified-session-validation.json)。

本轮证明 typed source 能进入真实会话并预测，零输入只验证 bias-free FFN 的基本 ABI/执行与所有权，不改变第四十一轮 INT8 非零输入质量失败结论，也不证明 ANE 驻留或速度收益。未接通新 StageExecutor 的 GPU suffix join、noise/main 完整 owner 与 9 steps/288 branch receipts，没有新完整出图或 release/App 构建；public hybrid 准入保持关闭。


## 第四十六轮：共享 GPU 后缀算术与真实分支 join（2026-09-21）

将既有 Z GPU suffix 编译图提取到 `z_image/hybrid_math.hpp/.cpp`，保持 full weights 与 compact suffix 两种输入的切片规则，并补齐参数数量/输入形状检查。新增共享 join：先将 Core ML FP16 输出转成 GPU dtype，再乘同 dtype scale，最后与 GPU suffix 相加；检查 shape/dtype/标量 scale ABI。普通和 compiled-post 两条 Z 路径均消费该实现。旧诊断分支仍单独计算 ane_scaled 以保留前缀独立统计；复核时修复了提取后该局部变量的引用，随后才开始完整构建。

join 返回 lazy tensor，不声称自身完成同步或拥有所有来源；调用方必须保留权重/ANE backing，并在复用前完成所有 consumer。full/compact 图使用的公式与旧实现相同，没有改变 BF16 转换顺序、分区或 scale policy。

Metal 测试采用真实 FFN 几何 3840×10240、a=5120、两个输入 rows，在三个分散的有效后缀通道填入可独立计算的稀疏权重。full 与 compact suffix 输出逐元素相同，独立浮点 sigmoid/乘加 oracle 的 max-abs < 0.03。额外检查 eager/compiled join 相同；FP16 32752 先转 BF16 再乘 32、加 BF16 的 1 得到有限 1048576，避免把低精度乘法挪到转换前。消费完成后复用 ANE 输入，已完成 join 输出保持独立。错误输入数量/形状/通道边界及 join 输入/scale dtype 被拒绝。最初单独编译 helper 进行测试，完整构建后又直接链接新 dylib 验证同一测试通过。

按[冻结计划](2026-09-21-m1-z-verified-join-plan.json)进一步连接实际 verified parent、GpuSuffixSource、ZImageWeightStream、typed HybridSession 与共享 GPU 图/join。该手动 driver 使用 P1/K1：2 个 fixed noise branches、1 个常驻 main layer、29 个 streamed main layers；noise 有效 rows 1024，main 1088，Core ML 输入统一补齐到 bucket 1088。每个 branch 先提交 GPU suffix，再调用 Core ML prefix，裁回有效行并合并；GPU 全元素比较完成后才允许复用 slot 和 Core ML output backing。

真实联合测试完成 32 个 join、32 次 Core ML runtime prediction、零 runtime failures；29 次 fill 实际读取 7,071,820,288 bytes（包含 reader 加载的全部 block fields，不只是 FFN），reader 未再次 packing。派生 SHA 与第三十九轮一致：`59b03803a9dac69d0fe89a79ef318056260b77d48cdc588744aef6bee89fcb47`；父 checkpoint SHA `2407613050b809ffdff18a4ac99af83ea6b95443ecebdf80e064a79c825574a6`。所有 joined BF16 输出符合零输入的精确零 oracle，来源前后校验与最终 Core ML generation 目录清理通过。这里是单 pass 的手动分支 driver；metadata 使用 512²/9 steps 的布局参数不代表运行了 9 个 denoise steps。

完整 native hook 构建及 runtime/catalog 编译后核对通过，库 SHA-256 `88d2a7925b55cc3ab47693e2bb1505d67cacd54484c7143417649eba47603065`，runtime key `tc-runtime-build-v1-6b4148d20c99784f5bdcddf5e21462ffab732566dd50c40608a706c68e9df474`。native contract 83 项运行、3 项原有跳过，其余通过；Z public adapter 回归通过。host macOS 26.0/26.2 链接提示保留，本机 26.4.1 执行通过。见[验证记录](2026-09-21-m1-z-verified-join-validation.json)。

这证明已验证 reader 和 Core ML 会话能共同完成真实权重的 FFN join，尚未接入新 StageExecutor 的完整 request owner、noise/main receipts、attention/latent 轨迹和输出发布。零输入不能评价非零数值质量，原 INT8 失败仍保留；未验证预测阻塞/取消 drain，也没有完整图片、速度比较或新 release/App 构建。GPU async submission 与 CPU+NE 配置不证明实际硬件重叠或 ANE 驻留。


## 第四十七轮：GPU/ANE 身份绑定到 common layout（2026-09-21）

检查 StageExecutor 接入时发现：既有 suffix descriptor 虽然精确描述 GPU 字段、读范围和 packing recipe，但未包含 Core ML bundle/partition；直接复用会让同一 GPU 切分对应的不同前缀模型共用布局身份。本轮新增 metadata-only `describe_hybrid_streaming()`，只接受已验证的 GPU metadata 和 VerifiedCoreMLBundleLease，检查二者父 checkpoint SHA 一致，冻结 HY-M0 512²/caption 64/9 steps 与 1088 bucket 范围，调用 common layout compiler 保留用户选择的 P/G/K/D/Q。

新的 descriptor workload 绑定 bundle 内容、partition identity、precision、bucket、scale 的 IEEE float32 位模式、GPU suffix kernel、join 顺序和共享 backing 复用政策；backend/stage adapter revision 与 pure GPU、suffix-only metadata 独立，明确标记 layout-only。该对象不授予执行或 release authority。GPU fixed/prefix/slot fields 和源 artifact 列表保持原样，派生文件依旧是 recipe identity，不把尚未生成的字节称为 content SHA。

测试 setup 对官方完整 checkpoint 做 native 内容验证，再导入 32 个 stub compiled tree 的 fixture；规划函数本身只做 metadata/来源 generation 检查，不打包、不加载 Core ML、不分配 GPU。两个安装路径不同但字节相同的 bundle 得到相同布局 digest；只改 precision 或一个模型 payload 字节都改变布局 digest。不同但 native 已验证的父文件、未验证 GPU metadata，以及不符固定范围的 width/caption rows/steps 均拒绝。逐字段检查 fixed 和 30 层 fields、packing recipe、GPU capacity 与逻辑 I/O 相对原 suffix plan 不变；P1/K1/D0/Q1 与 P7/K2/D1/Q2 均保留策略和 9 passes，exact descriptor digest 保持不变。所有 fixture generation 最终清理通过。

普通执行与 ASan/UBSan 通过；sanitizer 覆盖本轮新 factory 和 test 对象，链接的上一轮 support dylib 未重新插桩，不声称覆盖全库。见[验证记录](2026-09-21-m1-z-hybrid-layout-validation.json)。host macOS 26.0/26.2 链接提示保留，本机执行通过。新源文件已加入 build source list，本轮仅单独编译/运行 metadata 测试，没有完整 dylib/App 重建。

这补齐 StageExecutor 接入前的布局身份约束，并未实现 hybrid adapter 或完整 request owner。stub 模型不提供 Core ML 算术证据，本轮没有模型预测、GPU join、完整出图或性能比较；原 INT8 非零质量失败保持。GPU 字段容量也不包含 Core ML models/activation 等完整请求内存，不能据此晋升 release record。


## 第四十八轮：完整 hybrid transformer 接入 StageExecutor（2026-09-21）

新增内部单线程 owner `ZImageHybridStream`，将 verified parent/bundle、派生 suffix source、reader、typed HybridSession、GPU graph、输入与尚未完成的 tensor、adapter 和 StageExecutor 放入同一生命周期。复用原 transformer 的 embedding、noise/context refiners、attention、main blocks 和 final projection；noise 两个分支直接执行，main 的 resident prefix 与 streamed groups 交给 common StageExecutor。保留原 denoise 的 BF16 模型输入/F32 velocity 边界。hybrid layout revision 从 layout-only 更新到实际 stage-v1，并绑定 block kernel 与 owner/drain 政策；上一轮身份记录不回写。

每个 block 的 GPU suffix/ANE output/join consumer 完成后才记录 branch completion 或 reader fence，允许复用 slot/output backing。异常后 owner 失败状态保持，拒绝再次 transform；drain 先处理 StageExecutor 的 I/O/reader，再同步覆盖 noise、embedding、final projection 的 GPU 工作。只有确认完成才能释放 pending tensors、输入和来源。无法确认 drain 时保留整个 owner，避免只留 buffer 却释放 Core ML generation；此路径仍待整合到 ModelEngine 的 poison 状态。

按[执行前计划](2026-09-21-m1-z-hybrid-stage-plan.json)，实际运行官方权重、既有 32 个 INT8 compiled models，512²、P1/G1/K2/D1/Q2、9 步现有 sigma/Euler 调度。初始 latent 为非零 sin 序列，caption 为 64×2560 全零合成输入，不涉及 prompt encoder。完成 288 次 Core ML 调用、261 个 streamed group、63,646,382,592 bytes group 读取；每步 32 个分支、29 个 group。每步 velocity 与最终 latent 为有限 F32，最终 latent 与初始不同。native actual stage receipt verifier 通过；另保存完整 [branch 轨迹](2026-09-21-m1-z-hybrid-stage-branches.csv)和 [group receipt 字段](2026-09-21-m1-z-hybrid-stage-groups.csv)，逐项复核 9-pass 矩阵、fill/reader completion 与实际字节。补充 branch 轨迹尚不是完整 H3 component receipt schema。

独立进程取消测试在 4 个分支完成后抛出 typed Cancelled；独立事件异常在 3 个分支完成后触发。后者注入无法确认 drain，验证返回 false 且私有 generation 仍存在；解除注入后安全 drain，最终 owner 释放清理目录。两种故障均拒绝复用失败 owner；正常路径同样先确认最终 generation 清理才写 passed 结果。错误 step、输入维度、过早读取 receipt 和跨线程执行均被拒绝。注入检查的是保留政策，不是实际硬件 hang 恢复。

完整 native 构建、runtime/catalog 核对通过。库 SHA `2820120234420a2cc88d944eadf041a78c9cae9f890c52262f45e63bdbb5a72a`，runtime key `tc-runtime-build-v1-1f0d08ef12a3036b47dde60ab88976a3a2975ccde2152d0e5fa2902cb01dc7b5`。contract 83 项（3 项原有跳过）、public adapter、reader 10 项（6 项设备限制跳过）及更新后的 metadata layout 测试通过。首次构建因补齐 BF16/F32 边界主动中止，修正后重新构建；没有因超时重启实际实验。链接 macOS 26.0/26.2 提示保留，本机 26.4.1 执行通过。见[结果与 tensor 哈希](2026-09-21-m1-z-hybrid-stage-validation.json)。

这是内部完整 transformer 的非零轨迹与 owner 验证，尚未连接 ModelEngine/public hybrid 路由，没有 Qwen prompt/VAE/完整图片，也未进行纯 GPU 数值或速度对照。原 INT8 block 29 质量失败保持，有限输出不能替代质量门槛。Core ML CPU+NE 配置不证明 ANE 驻留或硬件重叠；没有整体请求内存准入、完整 component receipt、产品安装注册或新 App E2E 资格，HY-M0 尚不能整体完成。


## 第四十九轮：真实 prompt 输入与 App 当前构建（2026-09-21）

已 fetch origin/dev，并执行 merge；远端 `61c08495815d645bb54d75ae9dbea466f0648b2d` 已是当前 `3e12d28` 的祖先，返回 Already up to date，无冲突或待合入提交。

新增研究输入准备工具 `tools/native/prepare_z_image_hybrid_inputs.cpp`，从 native verified text/tokenizer SourceLease 读取，复用 Tokenizer::z_image_prompt 与 qwen3_conditioning 的 Z-Image 配置。冻结[真实 prompt/seed 42/512²/9 steps 计划](2026-09-21-m1-z-real-prompt-stage-plan.json)，得到 47 个有效 token。Qwen dynamic 输入保留实际 47 行，transformer 在 caption embedding 中补齐至 64；准备工具第一次错误要求 Qwen 输入本身有 64 行，在加载权重前拒绝，修正检查后按同一 prompt/seed 重跑，保留[修正记录](2026-09-21-m1-z-real-prompt-stage-preparation-correction.json)。没有调整样本或质量门槛。

stage 测试新增 `--inputs`，加载并记录外部 initial.npy/caption.npy/inputs.json 的 SHA，限定完整执行模式；原合成输入与取消/unsafe 模式保留。新运行使用真实 Qwen caption、现有随机 key(42) 初始噪声和原 Euler 调度，完成 9 passes、288 次 Core ML 调用、261 个 groups 与最终 owner 清理。全部输出有限；落盘输入与准备结果逐元素一致、SHA 与[执行计划](2026-09-21-m1-z-real-prompt-stage-execution-plan.json)一致。模型/bundle/partition 与第四十八轮相同，仍为原先有一个分支质量失败的 INT8 bank。见[结果和 tensor 哈希](2026-09-21-m1-z-real-prompt-stage-validation.json)。

最新 native 库的 Swift App/集成测试已启动完整构建；本轮完成并运行通过 library-store、studio-behavior、installation-inspection、tensor-cache 四组。其余构建进程仍在运行，日志 `/tmp/tc-m1-current-app-build.log`，不能把这四组通过视为完整 App/E2E 通过。本轮研究 stage 与 Swift 编译同时运行，不采纳 wall time 为性能样本。

真实 prompt denoiser 路径现在有执行证据，但还没有 VAE 解码、完整图片、同 partition 迁移对照或纯 GPU 质量/性能比较；未开放 public hybrid，也未完成 ModelEngine poison/component receipt 接入。下一步保留这些输入供完整输出与对照复用，原 INT8 局部失败不被有限值测试覆盖。


## 第五十轮：hybrid 完整图像、GPU 数值对照与 App 实测（2026-09-21）

将原 ZImage::decode 使用的 BF16 输入边界及 Comfy VAE 函数通过内部 `z_image::decode_vae` 共享，原生成路径也调用它，计算体未改动。研究工具从 native verified VAE lease 读取权重，检查上一轮真实 prompt 的最终 latent，完成原生 VAE 解码、有限值检查和 PNG 导出；这次通过独立进程拼接各研究阶段，不是 ModelEngine 的完整 hybrid 请求或 public receipt。执行前[解码计划](2026-09-21-m1-z-hybrid-image-decode-plan.json)固定 latent SHA、VAE SHA 与 512² 几何；[新库身份](2026-09-21-m1-z-hybrid-image-runtime.json)和[解码结果](2026-09-21-m1-z-hybrid-image-decode-result.json)保留。完整 native 构建及内置 runtime/catalog 核对通过，contract 83 项运行、3 项原有跳过，其余通过。

另外使用相同 prompt、逐元素相同的初始噪声、seed 42、9 steps 和 P1/G1/K2/D1/Q2 跑完整 pure GPU candidate request。四个官方源均再次 native 验证，内容 artifact digest 与既有记录一致；该请求输出所有 step latent、最终 latent 和 decoded tensor。[GPU 计划](2026-09-21-m1-z-hybrid-image-gpu-plan.json)在执行前落盘，[完整结果](2026-09-21-m1-z-hybrid-image-gpu-result.json)保留。新工具可复跑参考生成和误差统计。

单样本对照的最终 latent relative-L2 为 **0.1076367**、cosine **0.9942141**；原始 decoded relative-L2 **0.0898619**；PNG RGB 的 0–1 RMSE **0.0220601**、cosine **0.9990394**。两张图片均可辨认出雪地狐狸，构图相近，但不能据此称数值等价或质量合格。[误差与完整来源验证](2026-09-21-m1-z-hybrid-image-comparison.json)保留；原 INT8 block 29 局部质量失败保持，没有用这一次探索性图像设定新准入门槛。没有同 partition legacy 迁移对照或多 prompt/seeds confirmation，不能把全部差异直接归因于 INT8。研究步骤拆开执行且有后台构建，不做速度收益声明。

![Hybrid candidate](2026-09-21-m1-z-hybrid-image-candidate.png)

![Pure GPU reference](2026-09-21-m1-z-hybrid-image-gpu.png)

Swift App 和全部 integration test executables 完整构建退出 0，链接第四十八轮 `m1-hybrid-stage` 库。本轮连同第四十九轮实际通过 12 组：模型库持久化、Studio 状态、安装检查、tensor cache、历史管理、模型选择、helper transport、本地 API、运行指标、ANE 模型库、图片产物事务和 streaming resolution。独立状态目录启动 App 并观察到窗口，随后只关闭本次测试进程；窗口截图命令失败，不能声称已完成视觉 UI 验收。StudioControls 曾因缺少必需的 LoRA 配置参数错误调用，记录为调用错误，不算功能失败或通过。

通过当前 JobStore 的 Z-Image catalog 默认请求（1024²/9/resident GPU）完成出图、媒体验证、最终发布和重新打开历史；独立复核 PNG 1024² 与结果 SHA 一致。记录 MLX peak **21,607,357,244 bytes**，request wall **418.34 s**，同时有后台编译，因此只作功能观察，且明显不是本机低内存配置。Flux4 的 catalog 默认 resident 请求在推理前被 `insufficient physical memory for the conservative BF16 plan` 拒绝；没有伪称 Flux App 出图成功。当前 public catalog 仍为空，App 还不能通过已批准的 streaming 档位弥补该缺口；native private streaming 已有的出图证据不等于 App 路由可用。详见[App 实测记录](2026-09-21-m1-current-app-observations.json)。

后续仍需完成 hybrid 请求 owner/engine poison/component receipt、同 partition 迁移与更完整质量评估；App 的 Flux streaming 准入与真实端到端生命周期也是明确未完成项。本轮没有新增性能或 public hybrid 资格。


## 第五十一轮：App 图片发布的目标覆盖竞态（2026-09-21）

继续核对 Flux 的 App streaming 准入时，确认 production catalog 仍为空，builder 的 public channel 仍要求 P0/P1/P2/P3；56 中 `public-calibrated` 仅是拟议策略，不能直接把现有 research 记录当作 App 授权。发布链检查同时发现一个实际产物问题：图片经过 prepare 校验后，普通 rename 会覆盖最终路径新出现的文件，与唯一输出名合同不符。

将发布改为 Darwin `renamex_np(..., RENAME_EXCL)`，原子地要求目标不存在；不使用存在性预检查代替原子操作，也不在失败时回退覆盖。成功路径仍是同目录 staging 的 rename；目标冲突时失败、保留原目标，staging 由本次 owner 清理。旧目标是普通文件、有效符号链接或悬空符号链接均不能被替换。

回归测试在 prepare 后创建冲突目标，验证原目标/符号链接及其指向文件不变、失败不消费 staging。使用修改前 production 源编译同一测试，确实以 `Publication replaced a concurrent destination: regular` 失败；新实现运行通过，同时保留 request correlation、PNG CRC/解码、SHA receipt、取消、双发布拒绝和清理等测试。见[红灯与修复验证](2026-09-21-m1-image-publication-collision.json)。本次针对 App 源重建已退出 0，新 App SHA 记录在验证文件中；不将定向测试称为整个 App 生命周期验收。

此修复没有解决发布后 jobs.json 持久化失败或 crash 的 finalizing 恢复窗口，也没有补足 Flux public catalog 的资格。两者仍是后续工作，目标覆盖保护不替代完整产物事务。


## 第五十二轮：App 图片发布的 finalizing 恢复（2026-09-21）

图片发布前，JobStore 现在先保存 `finalizing` 与完整已校验结果，然后才执行上一轮的排他 rename，最后保存 succeeded。ImageOutputTransaction 在发布意图保存后保留未提交 staging，避免异常结束时丢掉恢复所需文件；正常发布后清理空 staging。若图片已发布而最后保存失败，内存状态回到 finalizing 并设置 storageError，保留已落盘意图，阻止继续生成；不再将已发布图片改写成 failed/cancelled。UI 增加正在保存结果状态。

图片 receipt 增加验证时的 device/inode。重启读取 finalizing 时，先校验收据字段、输出路径及同目录 UUID staging 范围，再复用原 PNG framing/CRC/解码/尺寸/request/hash 校验，并核对 device/inode。staging 仍在时继续排他发布；已改名时核对最终文件再完成成功记录。同字节但不同 inode 的替换文件不能冒充已完成 rename。冲突、损坏、符号链接、缺失/非法收据均拒绝成功，保留诊断文件；恢复只尝试删除空 staging 目录，不递归删除从历史重建的路径。恢复后的状态再次保存，重复启动保持一致。

定向 transaction 测试与完整 JobStore/history 测试通过，覆盖发布前/后持久化状态、重复恢复、缺失 staging、内容变更、相同字节不同 inode、符号链接、非法 staging 路径、损坏图片和缺失收据，原有 PNG/取消/冲突/历史删除回归保持通过。最初 history 测试编译因局部 FileManager 变量名缺失失败，修正后测试与 App 重建退出 0；见[验证与源码/App 哈希](2026-09-21-m1-image-finalizing-recovery.json)。

本轮通过构造实际落盘的中断边界状态并重新打开 JobStore 验证恢复，没有执行断电、OS crash 或真实模型运行时的磁盘故障注入，不声明 fsync/断电持久性。单次请求的 worker envelope、重启时旧进程存活判断、完整 source/quality/release 资格仍未关闭；此修复也不授予 Flux public streaming 档位。


## 第五十三轮：Flux4 512² streaming 的当前观测（2026-09-21）

核对第五十轮 App 默认请求拒绝的原因：Flux4 resident 的保守 estimate 是 `12 GiB + width×height×8192`，准入要求再留 4 GiB。因此在 16 GiB 机器上所有非零尺寸都会被拒绝；减小分辨率并不能让这条默认路径通过。exact streaming 路径使用独立布局/内存合同，不受此 resident-only estimate 限制。本轮未凭单次实验降低默认保护，也未将 catalog-empty 当作 streaming 不可执行。

新增复用 native campaign helper 的单请求工具 `run_image_streaming_smoke.py`，在加载模型前保存请求、库/工具 SHA 和计划，显式做 native source verification，保存 events/result 并释放 engine。外部 process-tree sampler 覆盖整个子进程。以现有真实官方 Flux4、512²、4 steps、seed 42、P0/G1/K2/D1/Q2、retain_all 跑 private exact GPU 请求：[计划](2026-09-21-m1-flux4-512-current-plan.json)、[结果](2026-09-21-m1-flux4-512-current-result.json)、[源验证](2026-09-21-m1-flux4-512-current-source-verification.json)。

完成 25 blocks×4 passes、100 次 fill、29,834,145,792 bytes 读取，两个 pool/4 个实际 slot bundles，最终 drain=true。PNG 512² 检查通过。native request wall **12.868 s**、denoise **10.498 s**；MLX peak **6,269,857,144 bytes**。1067 个进程树样本最大间隔 **29.77 ms**，低于冻结的 100 ms 上限；tree phys-footprint peak **6,354,752,576 bytes**，采样窗口 swap-out **0**。[内存摘要](2026-09-21-m1-flux4-512-current-memory-summary.json)及[原始证据哈希/限制](2026-09-21-m1-flux4-512-current-validation.json)保留。swap/compression 是采样窗口观测，不按因果归属于本请求；这不是重复样本性能或 P95 校准。

![Flux4 512-square private streaming](2026-09-21-m1-flux4-512-current-image.png)

需要区分 setup 的 native 内容验证与 execution 的 lease：本次 private 执行明确报告 `source_lease_verified=false`、`flux2-private-components-v1`，不能反推它消费了 public shared verified lease，也不能移用为 public full-request 资格。现有 public-calibrated channel/schema 尚未实现，正式记录仍需绑定实际 public component/container、完成要求的校准/质量/生命周期/发布 gate。本轮说明当前 private streaming 可在本机完成这一尺寸，不代表 Flux App 的 public streaming 已可用。


## 第五十四轮：发布策略的独立 record v3 身份（2026-09-21）

推进 56 的 public-calibrated 方案时发现，record v2 已经用于 portable source identity，不能复用同一个 canonical domain 表示新增 release 字段。本轮增加可选 `release.policy_revision`；缺省字段保持原 v1/v2 的字节与 digest，显式策略使用 record/digest/identity v3，并要求 portable source v2。当前仅接受 `tc-public-strict-v1`；未知策略及 `tc-public-calibrated-v1` 仍拒绝，public channel 与必需 gate 不变。

策略不仅进入完整 record digest，也进入排除 review 本身的 record identity 和 campaign catalog_binding，防止同一份 review/实验身份被换用到不同策略。native 与 Python 的 strict v3 golden digest 同为 `3b09dce86f9116b1b1fffcf7373e5584ad645294724cd99aeb8e9abfb57535a0`；v1/v2 既有 golden 保持。测试还验证显式 strict 缺 P3 仍拒绝、旧 campaign binding 不能用于 v3、新策略不能绑定旧路径 snapshot source。Objective-C++ test/calibration catalog parser 与 serializer 已保留该可选字段；生产目录仍为空。

host preset/runtime 测试、builder 17 项、bundled catalog 6 项、policy generator 4 项及 ObjC++ 语法检查通过。第一次新 builder 测试误用仅供 native canonical 对照、并不符合 builder estimator 约束的 fixture，出现 estimator unsupported；改为独立校验 canonical golden 与有效 builder fixture 后通过，没有放宽校准检查。完整 native 构建及 runtime/catalog 编译后核对通过；新库 C API 测试 3 项中 2 项通过、1 项因未提供 release-without-hooks 库跳过，实际验证 v3 strict record 装载、未验证来源仍拒绝 resolve、未知策略拒绝。首次 C API 调用环境变量错误导致 hook 测试跳过，纠正后才计为通过。contract 83 项、3 项原有跳过，其余通过。详见[验证记录](2026-09-21-m1-release-policy-v3-validation.json)。

这一步是显式策略编码的基础，不是 calibrated 发布实现完成。P3 必需性尚未改变；未来新增 calibrated channel 前仍需冻结并实现 builder/verifier 的 gate 集合、保留 P3 verdict、全局正确性/生命周期阻断和 App 展示，并取得实际所需证据。不得把当前 strict v3 身份当作已授予 public-calibrated 资格。

## 第五十五轮：完整 FP16 denoiser bank 与真实图片对照（2026-09-21）

冻结新的 32 分区 FP16 candidate，保持原 checkpoint、bucket 1088、ANE width 5120、activation/output scale 8/32 和局部门槛不变。第一次导出因输出目录已有计划文件而被 exporter 拒绝；保留失败日志，将 artifacts 放到独立子目录后完整导出，未改变模型参数。见[计划](2026-09-21-m1-z-denoiser-fp16-all-plan.json)和[输出目录修正](2026-09-21-m1-z-denoiser-fp16-all-output-correction.json)。通过 native manifest 编译接口完成全部 32 分区，cache hits=0；[编译计划](2026-09-21-m1-z-denoiser-fp16-all-compile-plan.json)、[结果](2026-09-21-m1-z-denoiser-fp16-all-compile-result.json)保留。

复用原 BF16 GPU oracle 和固定合成输入，实际输入及全部 32 个 reference.npy 与原 INT8 实验逐字节相同。FP16 **32/32** 通过原门槛 relative-L2≤0.025、cosine≥0.999、relative-max-abs≤0.05；最大 relative-L2 **0.00470822**，最低 cosine **0.99998896**。原失败 branch 29 本次 relative-L2 **0.00467754**。这是独立 precision candidate，不修改原 INT8 的失败结果。见[逐分区结果](2026-09-21-m1-z-denoiser-fp16-all-smoke-results.json)及[验证摘要](2026-09-21-m1-z-denoiser-fp16-all-validation.json)。CPU_AND_NE 是请求配置，不代表已测得硬件 ANE 驻留。

随后使用第五十轮相同的 47-token 真实 prompt、seed 42 初始噪声、512²、9 steps 与 P1/G1/K2/D1/Q2，通过 typed hybrid owner 完成全部 288 branches、261 groups 和资源清理。branch 顺序、group 读取字节、reader completion 再次检查，随后共享 native VAE 解码输出图片。[执行计划](2026-09-21-m1-z-fp16-hybrid-image-stage-plan.json)、[执行结果](2026-09-21-m1-z-fp16-hybrid-image-stage-result.json)、[解码计划](2026-09-21-m1-z-fp16-hybrid-image-decode-plan.json)和[完整验证记录](2026-09-21-m1-z-fp16-hybrid-image-validation.json)保留。

比较工具增加独立 frozen reference fixture 参数，并校验 stage plan 中输入哈希、实际 initial/caption 张量。旧 INT8 fixture 的 latent/decoded/RGB 数值和图片哈希回归不变。FP16 相对原纯 GPU 参考的最终 latent relative-L2 为 **0.09715543**（INT8 0.10763672），decoded relative-L2 **0.07757605**（0.08986189），RGB 0–1 RMSE **0.01906227**（0.02206008）。完整数据见[图像对照](2026-09-21-m1-z-fp16-hybrid-image-comparison.json)。图片可辨认出雪地狐狸，但局部误差显著下降没有带来同等幅度的完整轨迹改善；仍需同 partition legacy 对照，不能把剩余误差全归因于量化，或据此称迁移等价/质量合格。

![FP16 hybrid candidate](2026-09-21-m1-z-fp16-hybrid-image-candidate.png)

本轮为分阶段研究运行，不是 ModelEngine 的完整 hybrid 请求，不提供性能、内存或 public 发布资格。未改变 production catalog、strict P3 门或原失败记录。再次 fetch origin/dev 成功，远端仍为 `61c0849` 且已是当前分支祖先，没有新的待合并改动。

## 第五十六轮：同分区 legacy 参考入口与冻结对照（2026-09-21，运行中）

新增 `run_z_image_hybrid_legacy_reference.py`，复用 native worker、源验证和原始事件/result 保存流程，使用相同真实 prompt、初始噪声、512²/9 steps 与完整 FP16 bank。第一次选择 legacy streamed 时，原生准入在推理前明确拒绝 M1：GPU+ANE streaming 仅允许已测 M5 Pro 24 GiB profile。[请求计划](2026-09-21-m1-z-fp16-legacy-streamed-rejection-plan.json)和[拒绝结果](2026-09-21-m1-z-fp16-legacy-streamed-rejection-result.json)保留，没有修改或绕过设备准入。

改为现有允许的 resident hybrid 请求，保持相同 ANE 分区，使用完整 GPU 权重作为诊断参考。[首个 resident 计划](2026-09-21-m1-z-fp16-legacy-resident-1-plan.json)在推理前保存；[对照计划](2026-09-21-m1-z-fp16-legacy-comparison-plan.json)要求两次独立进程先检查所有已导出的 step/final/decoded 张量逐字节重复，再检查 typed stage 最终 latent 的逐字节一致性。失败不改门槛。新增比较工具校验 manifest、runtime、初始噪声、prompt、执行配置与实际 shape/steps，保存误差及原始计划/结果哈希。

本段写入时首个 resident 进程仍在运行，已进入第四步；尚无完整结果或 repeatability/migration verdict。新工具目前仅完成语法检查，不能称完整对照通过。原始运行目录为 `models/TurboCider/experiments/m1-z-fp16-legacy-migration`，必须继续观察同一运行，不因观察超时重启。

Resident legacy 与 typed streaming 的 P/G/K/D/Q、权重布局和 retention 不同，因此即使数值相同也不是 57 §7.1 的 same-plan P1；legacy 还会重算 Qwen conditioning，没有直接导出 caption。它用于缩小迁移差异范围，后续仍需同 source/spec/kernel 的顺序 private reference harness。没有性能或低内存声明。静态检查另发现结果展示层仍有固定的 `bf16_gpu+int8_mlp_fp16_io` 标签；本次精度以 FP16 export identity 为准，该展示问题尚未修复。
